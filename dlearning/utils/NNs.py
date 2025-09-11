# Copyright (c) 2025 Zhaolong Shen, Beihang University
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize, 
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix,
    euler_to_quaternion,
    quat_axis,
    quaternion_to_euler
)

class LyapunovFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))
        
        # 保留softplus配置项但不再必须使用
        self.softplus = nn.Softplus() if cfg.softplus else None  
        
        # 添加可学习的缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        base = torch.sum(state**2, dim=-1, keepdim=True) 
        
        nn_term = self.net(state)
        
        if self.softplus is not None:
            nn_term = self.softplus(nn_term)
        
        # return self.alpha * base + self.beta * nn_term
        return nn_term


class StructuredLyapunovFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化可学习参数（指数映射确保初始正定性）
        self.k = nn.Parameter(torch.tensor(1.0))  # 位置误差系数
        self.m = nn.Parameter(torch.tensor(1.0))  # 速度误差系数
        self.c = nn.Parameter(torch.tensor(0.1))  # 交叉项系数

    def forward(self, state: torch.Tensor):
        # 分解状态量：假设前3维为位置误差，后3维为速度误差
        ex = state[..., :3]  # 位置误差 [..., 3]
        ev = state[..., 3:6] # 速度误差 [..., 3]

        # 计算各分量（保持维度用于广播）
        term_k = self.k * torch.sum(ex**2, dim=-1, keepdim=True)  # k||ex||^2
        term_m = self.m * torch.sum(ev**2, dim=-1, keepdim=True)  # m||ev||^2
        term_c = self.c * torch.sum(ex * ev, dim=-1, keepdim=True)  # c ex·ev
        
        # 组合Lyapunov函数并保证非负
        V = term_k + term_m + term_c
        return F.relu(V) + 1e-6 


class BacksteppingLyapunovFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))
        
        # 保留softplus配置项但不再必须使用
        self.softplus = nn.Softplus() if cfg.softplus else None  
        
        # 添加可学习的缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.k_1 = nn.Parameter(torch.tensor(1.0))  # 虚拟控制量系数
        self.net = nn.Sequential(*layers)

        self.k = nn.Parameter(torch.tensor(1.0))  # 位置误差系数
        self.m = nn.Parameter(torch.tensor(1.0))  # 速度误差系数
        self.c = nn.Parameter(torch.tensor(0.1))  # 交叉项系数

    def forward(self, state: torch.Tensor):
        ex = state[..., :3]  # 位置误差 [..., 3]
        v = state[..., 3:6] # 速度 [..., 3]
        virtual_plan = -1*self.k_1*ex  # 虚拟规划量
        ev = v - virtual_plan  # 速度误差
        state = torch.cat([ex, ev], dim=-1) 

        term_k = self.k * torch.sum(ex**2, dim=-1, keepdim=True)  # k||ex||^2
        term_m = self.m * torch.sum(ev**2, dim=-1, keepdim=True)  # m||ev||^2
        term_c = self.c * torch.sum(ex * ev, dim=-1, keepdim=True)  # c ex·ev
        base = term_k + term_m + term_c
        # base = torch.sum(state**2, dim=-1, keepdim=True) 
        nn_term = self.net(state)
        if self.softplus is not None:
            nn_term = self.softplus(nn_term)
        # return self.alpha * base + self.beta * nn_term**2
        return self.alpha * base + self.beta * nn_term


class NeuralBacksteppingLyapunovFunction(nn.Module):
    '''
    - 混合Lyapunov函数函数
        - 形式化的Lyapunov函数作为warm start
        - 神经网络Lyapunov作为补充
        - 形式化的Lyapunov函数和神经网络Lyapunov参数可调节
    - 基于反步法的虚拟规划量设计
        - 引入虚拟规划量神经网络
        - 通过数据同时学习虚拟规划量
    '''
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))
        self.softplus = nn.Softplus() if cfg.softplus else None  
        
        # 添加可学习的缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.9))  
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.k_1 = nn.Parameter(torch.tensor(1.0))  # 虚拟控制量系数
        self.net = nn.Sequential(*layers)

        self.k = nn.Parameter(torch.tensor(1.0))  # 位置误差系数
        self.m = nn.Parameter(torch.tensor(1.0))  # 速度误差系数
        self.c = nn.Parameter(torch.tensor(0.1))  # 交叉项系数
        
        # 添加 virtual_plan 神经网络
        self.virtual_plan_net = nn.Sequential(
            nn.LazyLinear(cfg.virtual_plan_units),
            nn.LeakyReLU(),
            nn.Linear(cfg.virtual_plan_units, 3)  # 输出3维速度指令
        )
        self.virtual_plan_net_params = list(self.virtual_plan_net.parameters())

    def forward(self, state: torch.Tensor):
        ex = state[..., :3]  # 位置误差 [..., 3]
        v = state[..., 3:6] # 速度 [..., 3]
        # virtual_plan = -1*self.k_1*ex  # 虚拟规划量
        virtual_plan = self.virtual_plan_net(ex)
        ev = v - virtual_plan  # 速度误差
        state1 = torch.cat([ex, ev], dim=-1) 

        term_k = self.k * torch.sum(ex**2, dim=-1, keepdim=True)  # k||ex||^2
        term_m = self.m * torch.sum(ev**2, dim=-1, keepdim=True)  # m||ev||^2
        term_c = self.c * torch.sum(ex * ev, dim=-1, keepdim=True)  # c ex·ev
        base = term_k + term_m + term_c 
        nn_term = self.net(state1)


        return self.alpha * base + self.beta * nn_term
        '''
        base这个部分真的很有用，因为它可以提供一个基准值，用于热启动，甚至可以直接影响训练结果，建议加上
        '''
        # return base

    # def V_with_JV(self, state: torch.Tensor):
    #     '''
    #     计算Lyapunov函数和其梯度
    #     '''
    #     state = state.detach().requires_grad_(True)
    #     V = self.forward(state)
    #     JV = torch.autograd.grad(V, state, grad_outputs=torch.ones_like(V), create_graph=True)
    #     return V, JV


class DFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        cat = torch.cat([state, action], dim=-1)  # [..., n_states + n_actions]
        return self.net(cat)


class DFunctionwithPriorKnowledge(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(6))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        '''
        D(x,u) = JV(x)^T*f(x,u)
        在这里只计算f(x,u)
        TODO： 完善这个带有先验知识的函数
        '''
        cat = torch.cat([state, action], dim=-1)  # [..., n_states + n_actions]
        virtualdynamics = self.net(cat)
        return virtualdynamics
    
# class DFunctionwithPriorKnowledge(nn.Module):
#     def __init__(self, cfg, Lyapunovfunction):
#         super().__init__()
#         self.Lyapunovfunction = Lyapunovfunction

#         # 冻结Lyapunovfunction的参数
#         for p in self.Lyapunovfunction.parameters():
#             p.requires_grad_(False)

#         layers = []
#         num_units = cfg.hidden_units
#         for n in num_units:
#             layers.append(nn.LazyLinear(n))
#             layers.append(nn.LeakyReLU())
#             if cfg.layer_norm:
#                 layers.append(nn.LayerNorm(n))
#         layers.append(nn.LazyLinear(6))

#         self.net = nn.Sequential(*layers)

#     def forward(self, state: torch.Tensor, action: torch.Tensor):
#         '''
#         D(x,u) = JV(x)^T*f(x,u)
#         '''
#         cat = torch.cat([state, action], dim=-1)  # [..., n_states + n_actions]
#         virtualdynamics = self.net(cat)
#         print('virtualdynamics.shape:', virtualdynamics.shape)
#         # torch.Size([4, 1])
#         V, JV = self.Lyapunovfunction.V_with_JV(state)
#         print('JV[0]:', JV[0])
#         print('JV[0].shape:', JV[0].shape)
#         D = JV[0] * virtualdynamics
#         return D



class NNController(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        layers = []
        num_units = cfg.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.net(state)


class GRUController(nn.Module):
    '''
    门控循环单元控制器
        输入： 
            (batch_size, seq_len, state_dim)
        输出​​：
            output 所有时间步的隐藏状态(batch_size, seq_len, hidden_size)
            h_n 最终隐藏状态 (num_layers * num_directions, batch_size, hidden_size)
        全链接层：
            最后时间步的隐藏状态 hidden_size
            动作 action_dim
        
    '''
    def __init__(self, cfg, state_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(
                        input_size=state_dim, 
                        hidden_size=cfg.gru.hidden_size,
                        num_layers=cfg.gru.num_layers,
                        batch_first=True      # 输入/输出为 (batch, seq, feature)
                        )
        self.fc = nn.Linear(cfg.gru.hidden_size, action_dim)

    def forward(self, state: torch.Tensor):
        output, hidden = self.gru(state)
        return self.fc(output[:, -1, :])

