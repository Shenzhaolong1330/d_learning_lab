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

from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler



class LyapunovFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.algo.lyapunov.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.algo.lyapunov.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))
        if cfg.algo.lyapunov.softplus:
            layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.net(state)


class DFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        num_units = cfg.algo.dfunction.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.algo.lyapunov.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        cat = torch.cat([state, action], dim=-1)  # [..., n_states + n_actions]
        return self.net(cat)


class ControllerFunction(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        layers = []
        num_units = cfg.algo.controller.hidden_units
        for n in num_units:
            layers.append(nn.LazyLinear(n))
            layers.append(nn.LeakyReLU())
            if cfg.algo.lyapunov.layer_norm:
                layers.append(nn.LayerNorm(n))
        layers.append(nn.LazyLinear(action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.net(state)


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) # 把所有的前两个dimension合并成为batch
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


class DLearning(TensorDictModuleBase):
    def __init__(
        self, 
        cfg,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        action_dim = action_spec.shape[-1]
        self.equilibrium_thrust = 7.0231
        equilibrium_observation = observation_spec.zero()
        equilibrium_action = action_spec.zero()
        equilibrium_action[..., -1] = self.equilibrium_thrust # 悬停推力
        if not isinstance(equilibrium_action, TensorDict):
            equilibrium_action = TensorDict({"agents": {"action": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)

        # init lyapunov function
        self.lyapunovfunction = TensorDictModule(
            LyapunovFunction(self.cfg),
            [("agents", "transformed_drone_state")],
            # [("agents", "observation")],
            [("agents", "lyapunov_value")]
        ).to(self.device)
        self.lyapunovfunction(equilibrium_observation)

        # init dfunction
        self.dfunction = TensorDictModule(
            DFunction(self.cfg),
            [("agents", "transformed_drone_state"), ("agents", "action")],
            # [("agents", "observation"), ("agents", "action")],
            [("agents", "dfunction_value")]
        ).to(self.device)
        self.dfunction(equilibrium_observation)

        # init controller
        self.controller = TensorDictModule(
            ControllerFunction(self.cfg, action_dim),
            [("agents", "transformed_drone_state")],
            # [("agents", "observation")],
            [("agents", "action")]
        ).to(self.device)
        self.controller(equilibrium_observation)
        
        if self.cfg.checkpoint_path is not None:
            # 构建各个子模块的检查点路径
            # lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, "lyapunovfunction_checkpoint_final.pt")
            # dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, "dfunction_checkpoint_final.pt")
            # controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, "controller_checkpoint_final.pt")
            lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"lyapunovfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"dfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            # 加载 LyapunovFunction 模块的参数
            if os.path.exists(lyapunov_ckpt_path):
                lyapunov_state_dict = torch.load(lyapunov_ckpt_path)
                self.lyapunovfunction.load_state_dict(lyapunov_state_dict, strict=False)
                logging.info(f"Loaded LyapunovFunction checkpoint from {lyapunov_ckpt_path}")
            else:
                logging.warning(f"LyapunovFunction checkpoint not found at {lyapunov_ckpt_path}")
            # 加载 DFunction 模块的参数
            if os.path.exists(dfunction_ckpt_path):
                dfunction_state_dict = torch.load(dfunction_ckpt_path)
                self.dfunction.load_state_dict(dfunction_state_dict, strict=False)
                logging.info(f"Loaded DFunction checkpoint from {dfunction_ckpt_path}")
            else:
                logging.warning(f"DFunction checkpoint not found at {dfunction_ckpt_path}")
            # 加载 ControllerFunction 模块的参数
            if os.path.exists(controller_ckpt_path):
                controller_state_dict = torch.load(controller_ckpt_path)
                self.controller.load_state_dict(controller_state_dict, strict=False)
                logging.info(f"Loaded ControllerFunction checkpoint from {controller_ckpt_path}")
                print('--------------------controller loaded------------------------')
            else:
                logging.warning(f"ControllerFunction checkpoint not found at {controller_ckpt_path}")
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
            
            self.lyapunovfunction.apply(kaiming_init_)
            self.dfunction.apply(kaiming_init_)
            self.controller.apply(kaiming_init_)

        self.lya_opt = torch.optim.Adam(self.lyapunovfunction.parameters(), lr=cfg.algo.lyapunov.learning_rate)
        self.dfun_opt = torch.optim.Adam(self.dfunction.parameters(), lr=cfg.algo.dfunction.learning_rate)
        self.ctrl_opt = torch.optim.Adam(self.controller.parameters(), lr=cfg.algo.controller.learning_rate)

    def __call__(self, tensordict: TensorDict):
        # self.lyapunovfunction(tensordict)
        self.controller(tensordict)
        return tensordict

    def train_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_observation = self.observation_spec.zero()
        # TODO: 稳定点的状态和控制量,设置错了,不能为00000
        next_tensordict = tensordict["next"]
        dt = self.cfg.sim.dt
        for i in range(self.cfg.algo.learning.lyapunov_GD_steps):
            V0 = self.lyapunovfunction(equilibrium_observation)[("agents", "lyapunov_value")]
            V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
            V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
            Vdot = (V_ - V) / dt

            # 计算 V 值为负数的比例
            negative_V_count = torch.sum(V < 0).float()
            total_V_count = torch.numel(V)
            negative_V_ratio = negative_V_count / total_V_count

            # 计算 Vdot 为正数的比例
            positive_Vdot_count = torch.sum(Vdot > 0).float()
            total_Vdot_count = torch.numel(Vdot)
            positive_Vdot_ratio = positive_Vdot_count / total_Vdot_count

            SemiNegativeDefinite = torch.sum(F.relu(Vdot))
            PositiveDefinite = torch.sum(F.relu(-V))
            EquilibriumValue = torch.sum(V0**2)

            loss = SemiNegativeDefinite + PositiveDefinite + EquilibriumValue + self.param_sum_square(self.lyapunovfunction.module) * 0.1
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.lya_opt.step()
                self.lya_opt.zero_grad()

            step_info = {
                "lyapunov_loss": loss.item(),
                "lyapunov_semi_negative_definite": SemiNegativeDefinite.item(),
                "lyapunov_positive_definite": PositiveDefinite.item(),
                "lyapunov_negative_V_ratio":negative_V_ratio,
                "lyapunov_positive_Vdot_ratio":positive_Vdot_ratio,
                "lyapunov_equilibrium_value": EquilibriumValue.item(),
            }
            if run is not None:
                run.log(step_info)

    def eval_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_observation = self.observation_spec.zero()
        next_tensordict = tensordict["next"]

        V0 = self.lyapunovfunction(equilibrium_observation)[("agents", "lyapunov_value")]
        V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
        V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
        Vdot = (V_ - V) / self.cfg.sim.dt
        # Vdot.shape: torch.Size([8, 256, 1, 1])
        V_splits = torch.split(V.squeeze(-1).squeeze(-1), 1, dim=0)
        plt.figure(figsize=(10, 6))

        for i in range(V.shape[0]):
            V_i = V_splits[i].squeeze(0).cpu().detach().numpy()
            plt.plot(V_i, label=f"V[{i}]")
        plt.legend()
        plt.title("V values for 8 groups")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
        # print("equilibrium_observation",equilibrium_observation[("agents", "transformed_drone_state")])
        # print("equilibrium_observation",equilibrium_observation)
        # print("V0",V0)

        # 计算 V 值为负数的比例
        negative_V_count = torch.sum(V < 0).float()
        total_V_count = torch.numel(V)
        negative_V_ratio = negative_V_count / total_V_count
        # 计算 Vdot 为正数的比例
        positive_Vdot_count = torch.sum(Vdot > 0).float()
        total_Vdot_count = torch.numel(Vdot)
        positive_Vdot_ratio = positive_Vdot_count / total_Vdot_count

        SemiNegativeDefinite = torch.sum(F.relu(Vdot))
        PositiveDefinite = torch.sum(F.relu(-V))
        loss = SemiNegativeDefinite + PositiveDefinite + torch.sum(V0**2)
        loss_values = loss.item()
        semi_negative_definite_values = SemiNegativeDefinite.item()
        positive_definite_values = PositiveDefinite.item()

        eval_info = {
            "lyapunov_eval_loss": loss_values,
            "lyapunov_eval_semi_negative_definite": semi_negative_definite_values,
            "lyapunov_eval_positive_definite": positive_definite_values,
            "lyapunov_eval_negative_V_ratio":negative_V_ratio,
            "lyapunov_eval_positive_Vdot_ratio":positive_Vdot_ratio,
        }
        run.log(eval_info)


    def train_dfunction(self, tensordict: TensorDict, run):
        equilibrium_observation = self.observation_spec.zero()
        equilibrium_action = self.action_spec.zero()
        equilibrium_action[..., -1] = self.equilibrium_thrust
        if not isinstance(equilibrium_action, TensorDict):
            equilibrium_action = TensorDict({"agents": {"action": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)

        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
        V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
        Vdot = (V_ - V) / dt

        # loss_values = []
        loss_fn = nn.MSELoss()
        for i in range(self.cfg.algo.learning.dfunction_GD_steps):
            D0 = self.dfunction(equilibrium_observation)[('agents','dfunction_value')]
            D = self.dfunction(tensordict)[('agents','dfunction_value')]

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)
            loss = fitting_loss  + self.param_sum_square(self.dfunction.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.dfun_opt.step()
                self.dfun_opt.zero_grad()
            # loss_values.append(loss.item())
            step_info = {
                "dfunction_loss": loss.item(),
                "dfunction_fitting_loss":fitting_loss.item(),
                "dfunction_positive_D_ratio":positive_D_ratio,
            }
            if run is not None:
                run.log(step_info)

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

    def policy_improvement(self, tensordict: TensorDict, run):

        stable_action = tensordict[('agents','action')]
        for i in range(self.cfg.algo.learning.controller_GD_steps): 
            tensordict = self.controller(tensordict)  
            nn_action = tensordict[('agents','action')]
            if stable_action.shape != nn_action.shape:
                raise ValueError("控制器输出不一致")
            
            D = self.dfunction(tensordict)[('agents','dfunction_value')]
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)

            controller_correction = torch.sum((stable_action - nn_action)**2)

            loss = self.dfunction_upper_bound_mean_variance_loss(D) + controller_correction * 1 + self.param_sum_square(self.controller.module) * 0
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.ctrl_opt.step()
                self.ctrl_opt.zero_grad()

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "policy_improvement_loss": loss.item(),
                "policy_improvement_positive_D_ratio":positive_D_ratio,
                "policy_improvement_positive_penalty":positive_penalty,
                "policy_improvement_upper_bound":upper_bound,
                "policy_improvement_mean":mean,
                "policy_improvement_controller_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)

    def investigate_state_scale(self, tensordict: TensorDict):
        state = tensordict[("agents", "transformed_drone_state")]
        # 对前三个维度进行聚合，计算最后一个维度（12 个状态）的统计信息
        min_values = state.flatten(end_dim=-2).min(dim=0)[0]
        max_values = state.flatten(end_dim=-2).max(dim=0)[0]
        mean_values = state.flatten(end_dim=-2).mean(dim=0)
        std_values = state.flatten(end_dim=-2).std(dim=0)

        print("Minimum values per state dimension:")
        print(min_values)
        print("Maximum values per state dimension:")
        print(max_values)
        print("Mean values per state dimension:")
        print(mean_values)
        print("Standard deviation per state dimension:")
        print(std_values)

        return min_values, max_values, mean_values, std_values

    def param_sum_square(self, net: nn.Module):
        param_squares = [p ** 2 for p in net.parameters()]
        return  sum(torch.sum(p) for p in param_squares)
        
    def dfunction_upper_bound_mean_variance_loss(self, dvalue):
        """
        """
        positive_penalty = torch.sum(torch.relu(dvalue))
        upper_bound = torch.max(dvalue)
        lower_bound = torch.min(dvalue)
        mean = torch.mean(dvalue)
        variance = torch.var(dvalue)
        # return upper_bound*100 + lower_bound*0 + mean*30 + variance*0 + positive_penalty*10
        return upper_bound*0 + lower_bound*0 + mean*0 + variance*0 + positive_penalty*0