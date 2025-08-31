# Copyright (c) 2025 Zhaolong Shen, Beihang University
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import matplotlib.pyplot as plt

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
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.beta = nn.Parameter(torch.tensor(1.0))
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
        state = torch.cat([ex, ev], dim=-1) 

        term_k = self.k * torch.sum(ex**2, dim=-1, keepdim=True)  # k||ex||^2
        term_m = self.m * torch.sum(ev**2, dim=-1, keepdim=True)  # m||ev||^2
        term_c = self.c * torch.sum(ex * ev, dim=-1, keepdim=True)  # c ex·ev
        base = term_k + term_m + term_c 
        nn_term = self.net(state)

        if self.softplus is not None:
            nn_term = self.softplus(nn_term)
        # return self.alpha * base + self.beta * nn_term
        return nn_term


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


class HierarchicalDLearning(TensorDictModuleBase):
    def __init__(
        self, 
        cfg,
        uav_params,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        controller,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        action_dim = action_spec.shape[-1]
        pos_action_dim = 3
        atti_action_dim = 3
        self.equilibrium_thrust = 7.0231
        self.mass = torch.tensor(uav_params["mass"])
        self.controller = controller
        # 初始化神经网络
        # 构造完整的虚拟输入 TensorDict
        dummy_input = TensorDict({
            "agents": {
                "pos_control_input": torch.zeros(4, 6, device=self.device),
                "pos_control_output": torch.zeros(4, 3, device=self.device),
                "atti_control_input": torch.zeros(4, 6, device=self.device),
                "atti_control_output": torch.zeros(4, 3, device=self.device),
            }
        }, batch_size=4).to(self.device)
        # position part 
        self.pos_lyapunov = TensorDictModule(
            # LyapunovFunction(self.cfg.algo.lyapunov.pos),
            # BacksteppingLyapunovFunction(self.cfg.algo.lyapunov.pos),
            NeuralBacksteppingLyapunovFunction(self.cfg.algo.lyapunov.pos),
            # StructuredLyapunovFunction(self.cfg),
            [("agents", "pos_control_input")],
            [("agents", "pos_lyapunov_value")]
        ).to(self.device)
        self.pos_lyapunov(dummy_input)
        self.pos_dfunction = TensorDictModule(
            DFunction(self.cfg.algo.dfunction.pos),
            [("agents", "pos_control_input"), ("agents", "pos_control_output")],
            [("agents", "pos_dfunction_value")]
        ).to(self.device)
        self.pos_dfunction(dummy_input)
        self.pos_controller = TensorDictModule(
            NNController(self.cfg.algo.controller.pos, pos_action_dim),
            [("agents", "pos_control_input")],
            [("agents", "pos_control_output")]
        ).to(self.device)
        self.pos_controller(dummy_input)

        # attitude part 
        self.atti_lyapunov = TensorDictModule(
            # LyapunovFunction(self.cfg.algo.lyapunov.atti),
            NeuralBacksteppingLyapunovFunction(self.cfg.algo.lyapunov.atti),
            [("agents", "atti_control_input")],
            [("agents", "atti_lyapunov_value")]
        ).to(self.device)
        self.atti_lyapunov(dummy_input)
        self.atti_dfunction = TensorDictModule(
            DFunction(self.cfg.algo.dfunction.atti),
            [("agents", "atti_control_input"), ("agents", "atti_control_output")],
            [("agents", "atti_dfunction_value")]
        ).to(self.device)
        self.atti_dfunction(dummy_input)
        self.atti_controller = TensorDictModule(
            NNController(self.cfg.algo.controller.atti, atti_action_dim),
            [("agents", "atti_control_input")],
            [("agents", "atti_control_output")]
        ).to(self.device)
        self.atti_controller(dummy_input)

        if self.cfg.checkpoint_path is not None:
            pos_lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_lyapunov_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            pos_dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_dfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            pos_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            atti_lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_lyapunov_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            atti_dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_dfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            atti_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")

            if os.path.exists(pos_lyapunov_ckpt_path):
                pos_lyapunov_state_dict = torch.load(pos_lyapunov_ckpt_path)
                self.pos_lyapunov.load_state_dict(pos_lyapunov_state_dict, strict=False)
                logging.info(f"Loaded pos_lyapunov checkpoint from {pos_lyapunov_ckpt_path}")
            else:
                logging.warning(f"pos_lyapunov checkpoint not found at {pos_lyapunov_ckpt_path}")

            if os.path.exists(pos_dfunction_ckpt_path):
                pos_dfunction_state_dict = torch.load(pos_dfunction_ckpt_path)
                self.pos_dfunction.load_state_dict(pos_dfunction_state_dict, strict=False)
                logging.info(f"Loaded pos_DFunction checkpoint from {pos_dfunction_ckpt_path}")
            else:
                logging.warning(f"pos_DFunction checkpoint not found at {pos_dfunction_ckpt_path}")

            if os.path.exists(pos_controller_ckpt_path):
                pos_controller_state_dict = torch.load(pos_controller_ckpt_path)
                self.pos_controller.load_state_dict(pos_controller_state_dict, strict=False)
                logging.info(f"Loaded pos_Controller checkpoint from {pos_controller_ckpt_path}")
                print('--------------------controller loaded------------------------')
            else:
                logging.warning(f"pos_Controller checkpoint not found at {pos_controller_ckpt_path}")
            
            if os.path.exists(atti_lyapunov_ckpt_path):
                atti_lyapunov_state_dict = torch.load(atti_lyapunov_ckpt_path)
                self.atti_lyapunov.load_state_dict(atti_lyapunov_state_dict, strict=False)
                logging.info(f"Loaded atti_lyapunov checkpoint from {atti_lyapunov_ckpt_path}")
            else:
                logging.warning(f"atti_lyapunov checkpoint not found at {atti_lyapunov_ckpt_path}")

            if os.path.exists(atti_dfunction_ckpt_path):
                atti_dfunction_state_dict = torch.load(atti_dfunction_ckpt_path)
                self.atti_dfunction.load_state_dict(atti_dfunction_state_dict, strict=False)
                logging.info(f"Loaded atti_DFunction checkpoint from {atti_dfunction_ckpt_path}")
            else:
                logging.warning(f"atti_DFunction checkpoint not found at {atti_dfunction_ckpt_path}")

            if os.path.exists(atti_controller_ckpt_path):
                atti_controller_state_dict = torch.load(atti_controller_ckpt_path)
                self.atti_controller.load_state_dict(atti_controller_state_dict, strict=False)
                logging.info(f"Loaded atti_Controller checkpoint from {atti_controller_ckpt_path}")
                print('--------------------controller loaded------------------------')
            else:
                logging.warning(f"atti_Controller checkpoint not found at {atti_controller_ckpt_path}")
        else:
            def mini_init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            def kaiming_init_(module):
                if isinstance(module, nn.Linear):
                    # Kaiming 正态分布初始化
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    # 偏置初始化为 0
                    nn.init.constant_(module.bias, 0.)

            # 应用初始化函数
            self.pos_lyapunov.apply(kaiming_init_)
            self.pos_dfunction.apply(kaiming_init_)
            self.pos_controller.apply(kaiming_init_)
            self.atti_lyapunov.apply(kaiming_init_)
            self.atti_dfunction.apply(kaiming_init_)
            self.atti_controller.apply(kaiming_init_)

        # self.pos_lya_opt = torch.optim.Adam(self.pos_lyapunov.parameters(), lr=cfg.algo.lyapunov.pos.learning_rate)
        self.pos_lya_opt = torch.optim.AdamW(
            self.pos_lyapunov.parameters(), 
            lr=cfg.algo.lyapunov.pos.learning_rate,
            weight_decay=0.01
        )
        self.pos_dfun_opt = torch.optim.Adam(self.pos_dfunction.parameters(), lr=cfg.algo.dfunction.pos.learning_rate)
        self.pos_ctrl_opt = torch.optim.Adam(self.pos_controller.parameters(), lr=cfg.algo.controller.pos.learning_rate)
        # self.atti_lya_opt = torch.optim.Adam(self.atti_lyapunov.parameters(), lr=cfg.algo.lyapunov.atti.learning_rate)
        self.atti_lya_opt = torch.optim.AdamW(
            self.atti_lyapunov.parameters(), 
            lr=cfg.algo.lyapunov.atti.learning_rate,
            weight_decay=0.01
        )
        self.atti_dfun_opt = torch.optim.Adam(self.atti_dfunction.parameters(), lr=cfg.algo.dfunction.atti.learning_rate)
        self.atti_ctrl_opt = torch.optim.Adam(self.atti_controller.parameters(), lr=cfg.algo.controller.atti.learning_rate)


    def __call__(self, tensordict: TensorDict):
        # get state
        root_state = tensordict.get(("agents", "observation"))[...,:13]
        # target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        target_yaw = torch.zeros_like(root_state[..., 0])
        batch_shape = root_state.shape[:-1]
        # reshape data to (batch_size, value)
        root_state = root_state.reshape(-1, 13)
        target_yaw =target_yaw.reshape(-1, 1)
        pos, rot, vel, ang_vel_w = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        # pos control
        pos_control_input = torch.cat([pos, vel], dim=-1)
        tensordict.set(("agents","pos_control_input"), pos_control_input.reshape(*batch_shape, -1))
        self.pos_controller(tensordict)
        acc_des = tensordict[("agents", "pos_control_output")].reshape(-1, 3)
        b3_des = -normalize(acc_des)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ], dim=-1).to(torch.float32)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1)
        R = quaternion_to_rotation_matrix(rot)
        thrust = -self.mass * (acc_des * R[:, :, 2]).sum(-1, True)
        # atti control
        ang_vel_b = quat_rotate_inverse(rot, ang_vel_w)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        e_R = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ], dim=-1)
        atti_control_input = torch.cat([e_R, ang_vel_b], dim=-1)
        tensordict.set(("agents","atti_control_input"), atti_control_input.reshape(*batch_shape, -1))
        self.atti_controller(tensordict)
        torque = tensordict[("agents", "atti_control_output")].reshape(-1, 3)
        thrust = thrust.reshape(*batch_shape, -1)
        torque = torque.reshape(*batch_shape, -1)
        cmd = self.controller.controlleroutput2cmd(root_state, thrust, torque)
        # CTBR = torch.cat([thrust, torque], dim=-1)
        # CTBR = CTBR.reshape(*batch_shape, -1)
        tensordict.set(("agents","action"), cmd)
        return tensordict
    
    # TODO: hierarchical d learning training
    def train_pos_lyapunov(self, tensordict: TensorDict, run):
        # equilibrium_observation include pos_err vel_err
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","pos_control_input"),torch.zeros_like(tensordict[("agents","pos_control_input")]))
        
        next_tensordict = tensordict["next"]
        # print(next_tensordict)
        dt = self.cfg.sim.dt
        for i in range(self.cfg.algo.learning.pos.lyapunov_GD_steps):
            V0 = self.pos_lyapunov(equilibrium_observation)[("agents", "pos_lyapunov_value")]
            V = self.pos_lyapunov(tensordict)[("agents", "pos_lyapunov_value")]
            V_ = self.pos_lyapunov(next_tensordict)[("agents", "pos_lyapunov_value")]
            Vdot = (V_ - V) / dt

            # # 计算 V 值为负数的比例
            # negative_V_count = torch.sum(V < 0).float()
            # total_V_count = torch.numel(V)
            # negative_V_ratio = negative_V_count / total_V_count

            # # 计算 Vdot 为正数的比例
            # positive_Vdot_count = torch.sum(Vdot > 0).float()
            # total_Vdot_count = torch.numel(Vdot)
            # positive_Vdot_ratio = positive_Vdot_count / total_Vdot_count

            SemiNegativeDefinite = torch.sum(F.relu(Vdot))
            PositiveDefinite = torch.sum(F.relu(-V))
            EquilibriumValue = torch.sum(V0**2)

            loss = SemiNegativeDefinite*10 + PositiveDefinite*10 + EquilibriumValue + self.param_sum_square(self.pos_lyapunov.module) * 0.0
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_lya_opt.step()
                self.pos_lya_opt.zero_grad()

            step_info = {
                "pos_lyapunov_loss": loss.item(),
                "pos_lyapunov_semi_negative_definite": SemiNegativeDefinite.item(),
                "pos_lyapunov_positive_definite": PositiveDefinite.item(),
                # "pos_lyapunov_negative_V_ratio":negative_V_ratio,
                # "pos_lyapunov_positive_Vdot_ratio":positive_Vdot_ratio,
                "pos_lyapunov_equilibrium_value": EquilibriumValue.item(),
            }
            if run is not None:
                run.log(step_info)


    def train_atti_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","atti_control_input"),torch.zeros_like(tensordict[("agents","atti_control_input")]))
        
        next_tensordict = tensordict["next"]
        # print(next_tensordict)
        dt = self.cfg.sim.dt
        for i in range(self.cfg.algo.learning.atti.lyapunov_GD_steps):
            V0 = self.atti_lyapunov(equilibrium_observation)[("agents", "atti_lyapunov_value")]
            V = self.atti_lyapunov(tensordict)[("agents", "atti_lyapunov_value")]
            V_ = self.atti_lyapunov(next_tensordict)[("agents", "atti_lyapunov_value")]
            Vdot = (V_ - V) / dt

            # # 计算 V 值为负数的比例
            # negative_V_count = torch.sum(V < 0).float()
            # total_V_count = torch.numel(V)
            # negative_V_ratio = negative_V_count / total_V_count

            # # 计算 Vdot 为正数的比例
            # positive_Vdot_count = torch.sum(Vdot > 0).float()
            # total_Vdot_count = torch.numel(Vdot)
            # positive_Vdot_ratio = positive_Vdot_count / total_Vdot_count

            SemiNegativeDefinite = torch.sum(F.relu(Vdot))
            PositiveDefinite = torch.sum(F.relu(-V))
            EquilibriumValue = torch.sum(V0**2)

            loss = SemiNegativeDefinite*100 + PositiveDefinite*10 + EquilibriumValue + self.param_sum_square(self.atti_lyapunov.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.atti_lya_opt.step()
                self.atti_lya_opt.zero_grad()

            step_info = {
                "atti_lyapunov_loss": loss.item(),
                "atti_lyapunov_semi_negative_definite": SemiNegativeDefinite.item(),
                "atti_lyapunov_positive_definite": PositiveDefinite.item(),
                # "atti_lyapunov_negative_V_ratio":negative_V_ratio,
                # "atti_lyapunov_positive_Vdot_ratio":positive_Vdot_ratio,
                "atti_lyapunov_equilibrium_value": EquilibriumValue.item(),
            }
            if run is not None:
                run.log(step_info)


    def train_pos_dfunction(self, tensordict: TensorDict, run):
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","pos_control_input"),torch.zeros_like(tensordict[("agents","pos_control_input")]))
        # equilibrium_action include acc_des
        equilibrium_action = torch.zeros_like(tensordict[("agents","pos_control_output")])
        equilibrium_action[..., -1] = self.cfg.sim.gravity[2]
        equilibrium_action = TensorDict({"agents": {"pos_control_output": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)
        # print(equilibrium_observation)
        
        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.pos_lyapunov(tensordict)[("agents", "pos_lyapunov_value")]
        V_ = self.pos_lyapunov(next_tensordict)[("agents", "pos_lyapunov_value")]
        Vdot = (V_ - V) / dt

        # loss_values = []
        loss_fn = nn.MSELoss()
        for i in range(self.cfg.algo.learning.pos.dfunction_GD_steps):
            D0 = self.pos_dfunction(equilibrium_observation)[('agents','pos_dfunction_value')]
            D = self.pos_dfunction(tensordict)[('agents','pos_dfunction_value')]

            # positive_D_count = torch.sum(D > 0).float()
            # total_D_count = torch.numel(D)
            # positive_D_ratio = positive_D_count / total_D_count

            fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)
            loss = fitting_loss  + self.param_sum_square(self.pos_dfunction.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_dfun_opt.step()
                self.pos_dfun_opt.zero_grad()
            # loss_values.append(loss.item())
            step_info = {
                "pos_dfunction_loss": loss.item(),
                "pos_dfunction_fitting_loss":fitting_loss.item(),
                # "dfunction_positive_D_ratio":positive_D_ratio,
            }
            if run is not None:
                run.log(step_info)


    def train_atti_dfunction(self, tensordict: TensorDict, run):
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","atti_control_input"),torch.zeros_like(tensordict[("agents","atti_control_input")]))
        equilibrium_action = torch.zeros_like(tensordict[("agents","atti_control_output")])
        equilibrium_action = TensorDict({"agents": {"atti_control_output": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)
        # print(equilibrium_observation)
        
        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.atti_lyapunov(tensordict)[("agents", "atti_lyapunov_value")]
        V_ = self.atti_lyapunov(next_tensordict)[("agents", "atti_lyapunov_value")]
        Vdot = (V_ - V) / dt

        # loss_values = []
        loss_fn = nn.MSELoss()
        for i in range(self.cfg.algo.learning.atti.dfunction_GD_steps):
            D0 = self.atti_dfunction(equilibrium_observation)[('agents','atti_dfunction_value')]
            D = self.atti_dfunction(tensordict)[('agents','atti_dfunction_value')]

            # positive_D_count = torch.sum(D > 0).float()
            # total_D_count = torch.numel(D)
            # positive_D_ratio = positive_D_count / total_D_count

            fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)
            loss = fitting_loss  + self.param_sum_square(self.atti_dfunction.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.atti_dfun_opt.step()
                self.atti_dfun_opt.zero_grad()
            # loss_values.append(loss.item())
            step_info = {
                "atti_dfunction_loss": loss.item(),
                "atti_dfunction_fitting_loss":fitting_loss.item(),
                # "dfunction_positive_D_ratio":positive_D_ratio,
            }
            if run is not None:
                run.log(step_info)


    def pos_policy_improvement(self, tensordict: TensorDict, run):

        stable_action = tensordict[('agents','pos_control_output')]
        for i in range(self.cfg.algo.learning.pos.controller_GD_steps): 
            tensordict = self.pos_controller(tensordict)  
            nn_action = tensordict[('agents','pos_control_output')]
            if stable_action.shape != nn_action.shape:
                raise ValueError("控制器输出不一致")
            
            D = self.pos_dfunction(tensordict)[('agents','pos_dfunction_value')]
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)

            controller_correction = torch.sum((stable_action - nn_action)**2)

            loss = self.dfunction_upper_bound_mean_variance_loss(D)*0.0 + controller_correction * 0.1 + self.param_sum_square(self.pos_controller.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_ctrl_opt.step()
                self.pos_ctrl_opt.zero_grad()

            # positive_D_count = torch.sum(D > 0).float()
            # total_D_count = torch.numel(D)
            # positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "pos_policy_loss": loss.item(),
                # "pos_policy_positive_D_ratio":positive_D_ratio,
                "pos_policy_positive_penalty":positive_penalty,
                "pos_policy_upper_bound":upper_bound,
                "pos_policy_mean":mean,
                "pos_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)

    
    def atti_policy_improvement(self, tensordict: TensorDict, run):

        stable_action = tensordict[('agents','atti_control_output')]
        for i in range(self.cfg.algo.learning.atti.controller_GD_steps): 
            tensordict = self.atti_controller(tensordict)  
            nn_action = tensordict[('agents','atti_control_output')]
            if stable_action.shape != nn_action.shape:
                raise ValueError("控制器输出不一致")
            
            D = self.atti_dfunction(tensordict)[('agents','atti_dfunction_value')]
            # print('D函数值的形状',D.shape)
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)

            controller_correction = torch.sum((stable_action - nn_action)**2)

            loss = self.dfunction_upper_bound_mean_variance_loss(D)*0.0 + controller_correction * 10 + self.param_sum_square(self.pos_controller.module) * 0.001
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.atti_ctrl_opt.step()
                self.atti_ctrl_opt.zero_grad()

            # positive_D_count = torch.sum(D > 0).float()
            # total_D_count = torch.numel(D)
            # positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "atti_policy_loss": loss.item(),
                # "pos_policy_positive_D_ratio":positive_D_ratio,
                "atti_policy_positive_penalty":positive_penalty,
                "atti_policy_upper_bound":upper_bound,
                "atti_policy_mean":mean,
                "atti_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)


    def eval_pos_lyapunov(self, tensordict: TensorDict,run):
        V = self.pos_lyapunov(tensordict)[("agents", "pos_lyapunov_value")]
        # Vdot.shape: torch.Size([8, 256, 1, 1])
        V_splits = torch.split(V.squeeze(-1).squeeze(-1), 1, dim=0)
        plt.figure(figsize=(10, 6))

        # for i in range(V.shape[0]%10):
        for i in range(V.shape[0]):
            V_i = V_splits[i].squeeze(0).cpu().detach().numpy()
            plt.plot(V_i, label=f"V[{i}]")
        plt.legend()
        plt.title("pos lyapunov values")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        # 确保保存目录存在
        save_path = os.path.join(run.public.run_dir, "pos lyapunov values.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存图片
        plt.savefig(save_path)
        # 显示图片
        plt.show()

    
    def eval_atti_lyapunov(self, tensordict: TensorDict,run):
        V = self.atti_lyapunov(tensordict)[("agents", "atti_lyapunov_value")]
        # Vdot.shape: torch.Size([8, 256, 1, 1])
        V_splits = torch.split(V.squeeze(-1).squeeze(-1), 1, dim=0)
        plt.figure(figsize=(10, 6))

        for i in range(V.shape[0]):
        # for i in range(V.shape[0]%10):
            V_i = V_splits[i].squeeze(0).cpu().detach().numpy()
            plt.plot(V_i, label=f"V[{i}]")
        plt.legend()
        plt.title("atti lyapunov values")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        save_path = os.path.join(run.public.run_dir, "atti lyapunov values.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存图片
        plt.savefig(save_path)
        plt.show()


    def eval_dfunction(self, tensordict: TensorDict, run):
        equilibrium_observation = self.observation_spec.zero()
        equilibrium_action = self.action_spec.zero()
        equilibrium_action[..., -1] = self.equilibrium_thrust
        if not isinstance(equilibrium_action, TensorDict):
            equilibrium_action = TensorDict({"agents": {"action": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)
        D0 = self.dfunction(equilibrium_observation)[('agents','dfunction_value')]

        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
        V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
        Vdot = (V_ - V) / dt

        D = self.dfunction(tensordict)[('agents','dfunction_value')]
        loss_fn = nn.MSELoss()
        fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)

        positive_D_count = torch.sum(D > 0).float()
        total_D_count = torch.numel(D)
        positive_D_ratio = positive_D_count / total_D_count
        # print('equilibrium_action',equilibrium_action)
        # print('equilibrium_observation',equilibrium_observation)
        # print('tensordict',tensordict)
        # print('equilibrium_observation',equilibrium_observation[("agents", "transformed_drone_state")])
        # print('equilibrium_observation',equilibrium_observation[("agents", "action")])
        # print('tensordict',tensordict[("agents", "transformed_drone_state")])
        # print('tensordict',tensordict[("agents", "action")])
        eval_info = {
            "dfunction_eval_fitting_loss":fitting_loss.item(),
            "dfunction_eval_positive_D_ratio":positive_D_ratio,
        }
        run.log(eval_info)


    def param_sum_square(self, net: nn.Module):
        param_squares = [p ** 2 for p in net.parameters()]
        return  sum(torch.sum(p) for p in param_squares)


    def dfunction_upper_bound_mean_variance_loss(self, dvalue):
        """
        """
        # 确保控制量使dvalue小于
        positive_penalty = torch.sum(torch.relu(dvalue))
        # 确保dvalue上届(平衡点附近)是0
        upper_bound = torch.max(dvalue)**2
        # dvalue下界
        lower_bound = torch.min(dvalue)
        # 确保dvalue均值小于0
        mean = torch.mean(dvalue)
        # 减小dvalue方差
        variance = torch.var(dvalue)
        # return upper_bound*100 + lower_bound*0 + mean*30 + variance*0 + positive_penalty*10
        return upper_bound*0.01 + lower_bound*0 + mean*0.01 + variance*0 + positive_penalty*1