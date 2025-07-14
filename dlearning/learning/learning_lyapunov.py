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


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) # 把所有的前两个dimension合并成为batch
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


def transform_drone_state(tensordict: TensorDict):
    pos = tensordict.get("agent", "observation")[..., :3]
    quat = tensordict.get("agent", "observation")[..., 3:7]
    euler = quaternion_to_euler(quat)
    lin_vel = tensordict.get("agent", "observation")[...,7:10]
    ang_vel = tensordict.get("agent", "observation")[...,10:13]
    
    transformed_drone_state = [pos, lin_vel, euler, ang_vel]
    transformed_drone_state = torch.cat(transformed_drone_state, dim=-1)
    
    print('transformed_drone_state.shape',transformed_drone_state.shape)

    tensordict.set("agent", "transformed_drone_state", transformed_drone_state)
    return tensordict


class LearningLyapunov(TensorDictModuleBase):
    def __init__(
        self, 
        cfg,
        observation_spec: CompositeSpec, 
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec

        fake_input = observation_spec.zero()
        
        # print(fake_input.keys(True, True))

        # 修改成员检查方式
        # if ('agents', 'transformed_drone_state') not in fake_input.keys(True, True):
        #     print(f"fake_input keys: {list(fake_input.keys(True, True))}")
        #     raise KeyError("('agents', 'transformed_drone_state') not found in fake_input")
        
        self.lyapunovfunction = TensorDictModule(
            LyapunovFunction(self.cfg),
            [("agents", "transformed_drone_state")],
            ["lyapunov_value"]
        ).to(self.device)
        self.lyapunovfunction(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            
        self.lyapunovfunction.apply(init_)
        self.optimizer = torch.optim.Adam(self.lyapunovfunction.parameters(), lr=cfg.algo.lyapunov.learning_rate)

    def __call__(self, tensordict: TensorDict):
        self.lyapunovfunction(tensordict)
        return tensordict

    def train_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_input = self.observation_spec.zero()
        # print('equilibrium_input: ', equilibrium_input)
        # equilibrium_input.shape torch.Size([8])
        next_tensordict = tensordict["next"]
        loss_values = []
        semi_negative_definite_values = []
        positive_definite_values = []

        for i in range(self.cfg.algo.learning.GD_steps):
            V0 = self.lyapunovfunction(equilibrium_input)['lyapunov_value']
            V = self.lyapunovfunction(tensordict)['lyapunov_value']
            V_ = self.lyapunovfunction(next_tensordict)['lyapunov_value']
            Vdot = (V_ - V) / self.cfg.sim.dt
            # Vdot.shape: torch.Size([8, 256, 1, 1])
            SemiNegativeDefinite = torch.sum(F.relu(Vdot))
            PositiveDefinite = torch.sum(F.relu(-V))
            loss = SemiNegativeDefinite + PositiveDefinite
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_values.append(loss.item())
            semi_negative_definite_values.append(SemiNegativeDefinite.item())
            positive_definite_values.append(PositiveDefinite.item())

            # 记录当前步骤的信息到 SwanLab
            step_info = {
                "lyapunov_loss": loss.item(),
                "semi_negative_definite": SemiNegativeDefinite.item(),
                "positive_definite": PositiveDefinite.item()
            }
            run.log(step_info)

        return {
            "lyapunov_loss": loss_values,
            "semi_negative_definite": semi_negative_definite_values,
            "positive_definite": positive_definite_values
        }
        # print('loss_values:', loss_values[-1])
        # print('SemiNegativeDefinite:', semi_negative_definite_values[-1])
        # print('PositiveDefinite:', positive_definite_values[-1])

    def eval_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_input = self.observation_spec.zero()
        next_tensordict = tensordict["next"]

        V0 = self.lyapunovfunction(equilibrium_input)['lyapunov_value']
        V = self.lyapunovfunction(tensordict)['lyapunov_value']
        V_ = self.lyapunovfunction(next_tensordict)['lyapunov_value']
        Vdot = (V_ - V) / self.cfg.sim.dt
        # Vdot.shape: torch.Size([8, 256, 1, 1])
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
        }
        run.log(eval_info)

        return {
            "eval_lyapunov_loss": loss_values,
            "eval_semi_negative_definite": semi_negative_definite_values,
            "eval_positive_definite": positive_definite_values,
        }

