# Copyright (c) 2025 Zhaolong Shen, Beihang University

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
        fake_observation = observation_spec.zero()
        fake_action = action_spec.zero()
        if not isinstance(fake_action, TensorDict):
            fake_action = TensorDict({"agents": {"action": fake_action}}, batch_size=fake_action.shape[:-1])
        fake_observation = fake_observation.update(fake_action)

        # init lyapunov function
        self.lyapunovfunction = TensorDictModule(
            LyapunovFunction(self.cfg),
            [("agents", "transformed_drone_state")],
            [("agents", "lyapunov_value")]
        ).to(self.device)
        self.lyapunovfunction(fake_observation)

        # init dfunction
        self.dfunction = TensorDictModule(
            DFunction(self.cfg),
            [("agents", "transformed_drone_state"), ("agents", "action")],
            [("agents", "dfunction_value")]
        ).to(self.device)
        self.dfunction(fake_observation)

        # init controller
        self.controller = TensorDictModule(
            ControllerFunction(self.cfg, action_dim),
            [("agents", "transformed_drone_state")],
            [("agents", "action")]
        ).to(self.device)
        self.controller(fake_observation)
        
        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
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
        return upper_bound*10 + lower_bound*0 + mean*50 + variance*0 + positive_penalty*100

    def __call__(self, tensordict: TensorDict):
        # self.lyapunovfunction(tensordict)
        return tensordict

    def train_lyapunov(self, tensordict: TensorDict, run):
        print('---------------start training lyapunov----------------')
        equilibrium_input = self.observation_spec.zero()
        # print('equilibrium_input: ', equilibrium_input)
        # equilibrium_input.shape torch.Size([8])
        next_tensordict = tensordict["next"]
        loss_values = []
        semi_negative_definite_values = []
        positive_definite_values = []
        dt = self.cfg.sim.dt
        for i in range(self.cfg.algo.learning.lyapunov_GD_steps):
            V0 = self.lyapunovfunction(equilibrium_input)[("agents", "lyapunov_value")]
            V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
            V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
            Vdot = (V_ - V) / dt
            # Vdot.shape: torch.Size([8, 256, 1, 1])

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
            loss = SemiNegativeDefinite + PositiveDefinite + self.param_sum_square(self.lyapunovfunction.module)
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.lya_opt.step()
                self.lya_opt.zero_grad()
            loss_values.append(loss.item())
            semi_negative_definite_values.append(SemiNegativeDefinite.item())
            positive_definite_values.append(PositiveDefinite.item())

            # 记录当前步骤的信息到 SwanLab
            step_info = {
                "lyapunov_loss": loss.item(),
                "semi_negative_definite": SemiNegativeDefinite.item(),
                "positive_definite": PositiveDefinite.item(),
                "negative_V_ratio":negative_V_ratio,
                "positive_Vdot_ratio":positive_Vdot_ratio,
            }
            run.log(step_info)

        return {
            "lyapunov_loss": loss_values,
            "semi_negative_definite": semi_negative_definite_values,
            "positive_definite": positive_definite_values,
            "negative_V_ratio":negative_V_ratio,
            "positive_Vdot_ratio":positive_Vdot_ratio,
        }

    def eval_lyapunov(self, tensordict: TensorDict, run):
        print('---------------start evaluating lyapunov----------------')
        equilibrium_input = self.observation_spec.zero()
        next_tensordict = tensordict["next"]

        V0 = self.lyapunovfunction(equilibrium_input)[("agents", "lyapunov_value")]
        V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
        V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
        Vdot = (V_ - V) / self.cfg.sim.dt
        # Vdot.shape: torch.Size([8, 256, 1, 1])

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
        loss = SemiNegativeDefinite + PositiveDefinite
        loss_values = loss.item()
        semi_negative_definite_values = SemiNegativeDefinite.item()
        positive_definite_values = PositiveDefinite.item()

        # 记录当前步骤的信息到 SwanLab
        eval_info = {
            "eval_lyapunov_loss": loss_values,
            "eval_semi_negative_definite": semi_negative_definite_values,
            "eval_positive_definite": positive_definite_values,
            "eval_negative_V_ratio":negative_V_ratio,
            "eval_positive_Vdot_ratio":positive_Vdot_ratio,
        }
        run.log(eval_info)

        return {
            "eval_lyapunov_loss": loss_values,
            "eval_semi_negative_definite": semi_negative_definite_values,
            "eval_positive_definite": positive_definite_values,
            "eval_negative_V_ratio":negative_V_ratio,
            "eval_positive_Vdot_ratio":positive_Vdot_ratio,
        }

    def train_dfunction(self, tensordict: TensorDict, run):
        print('---------------start training dfunction----------------')
        fake_observation = self.observation_spec.zero()
        fake_action = self.action_spec.zero()
        if not isinstance(fake_action, TensorDict):
            fake_action = TensorDict({"agents": {"action": fake_action}}, batch_size=fake_action.shape[:-1])
        fake_observation = fake_observation.update(fake_action)

        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.lyapunovfunction(tensordict)[("agents", "lyapunov_value")]
        V_ = self.lyapunovfunction(next_tensordict)[("agents", "lyapunov_value")]
        Vdot = (V_ - V) / dt
        loss_values = []
        loss_fn = nn.MSELoss()
        for i in range(self.cfg.algo.learning.dfunction_GD_steps):
            D0 = self.dfunction(fake_observation)[('agents','dfunction_value')]
            D = self.dfunction(tensordict)[('agents','dfunction_value')]

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2) + self.param_sum_square(self.dfunction.module)
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.dfun_opt.step()
                self.dfun_opt.zero_grad()
            loss_values.append(loss.item())
            step_info = {
                "dfunction_loss": loss.item(),
                "positive_D_ratio":positive_D_ratio,
            }
            run.log(step_info)

        return {
            "dfunction_loss": loss_values,
        }

    def eval_dfunction(self, tensordict: TensorDict, run):
        print('---------------start evaluating dfunction----------------')
        D = self.dfunction(tensordict)[('agents','dfunction_value')]
        positive_D_count = torch.sum(D > 0).float()
        total_D_count = torch.numel(D)
        positive_D_ratio = positive_D_count / total_D_count
        eval_info = {
            "eval_positive_D_ratio":positive_D_ratio,
        }
        run.log(eval_info)

        return {
            "eval_positive_D_ratio":positive_D_ratio,
        }

    def policy_improvement(self, tensordict: TensorDict, run):
        print('---------------start improving policy----------------')
        loss_values = []
        for i in range(self.cfg.algo.learning.controller_GD_steps): 
            # 置换掉tensordict中的action
            tensordict = self.controller(tensordict)  
            D = self.dfunction(tensordict)[('agents','dfunction_value')]
            # print(D.shape)
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)
            loss = self.dfunction_upper_bound_mean_variance_loss(D) + self.param_sum_square(self.controller.module)
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.ctrl_opt.step()
                self.ctrl_opt.zero_grad()
            loss_values.append(loss.item())

            
            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "policy_improvement_loss": loss.item(),
                "policy_improvement_positive_D_ratio":positive_D_ratio,
                "policy_improvement_positive_penalty":positive_penalty,
                "policy_improvement_upper_bound":upper_bound,
                "policy_improvement_mean":mean,
            }
            run.log(step_info)

        return {
            "policy_improvement_idx": loss_values,
        }

