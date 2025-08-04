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


class HierarchicalDLearning(TensorDictModuleBase):
    def __init__(
        self, 
        cfg,
        uav_params,
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
        pos_action_dim = 3
        atti_action_dim = 3
        self.equilibrium_thrust = 7.0231
        self.mass = torch.tensor(uav_params["mass"])
        # equilibrium_observation = observation_spec.zero()
        # equilibrium_action = action_spec.zero()
        # equilibrium_action["action"][..., 0] = self.equilibrium_thrust # 悬停推力
        # if not isinstance(equilibrium_action, TensorDict):
        #     equilibrium_action = TensorDict({"agents": {"action": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        # equilibrium_observation = equilibrium_observation.update(equilibrium_action)

        # position part 
        self.pos_lyapunov = TensorDictModule(
            LyapunovFunction(self.cfg),
            [("agents", "pos_control_input")],
            [("agents", "pos_lyapunov_value")]
        ).to(self.device)
        self.pos_dfunction = TensorDictModule(
            DFunction(self.cfg),
            [("agents", "pos_control_input"), ("agents", "pos_control_output")],
            [("agents", "pos_dfunction_value")]
        ).to(self.device)
        self.pos_controller = TensorDictModule(
            ControllerFunction(self.cfg, pos_action_dim),
            [("agents", "pos_control_input")],
            [("agents", "pos_control_output")]
        ).to(self.device)

        # attitude part 
        self.atti_lyapunov = TensorDictModule(
            LyapunovFunction(self.cfg),
            [("agents", "atti_control_input")],
            [("agents", "atti_lyapunov_value")]
        ).to(self.device)
        self.atti_dfunction = TensorDictModule(
            DFunction(self.cfg),
            [("agents", "atti_control_input"), ("agents", "atti_control_output")],
            [("agents", "atti_dfunction_value")]
        ).to(self.device)
        self.atti_controller = TensorDictModule(
            ControllerFunction(self.cfg, atti_action_dim),
            [("agents", "atti_control_input")],
            [("agents", "atti_control_output")]
        ).to(self.device)
        
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
            
            self.pos_lyapunov.apply(kaiming_init_)
            self.pos_dfunction.apply(kaiming_init_)
            self.pos_controller.apply(kaiming_init_)
            self.atti_lyapunov.apply(kaiming_init_)
            self.atti_dfunction.apply(kaiming_init_)
            self.atti_controller.apply(kaiming_init_)

        self.pos_lya_opt = torch.optim.Adam(self.pos_lyapunov.parameters(), lr=cfg.algo.lyapunov.learning_rate)
        self.pos_dfun_opt = torch.optim.Adam(self.pos_dfunction.parameters(), lr=cfg.algo.dfunction.learning_rate)
        self.pos_ctrl_opt = torch.optim.Adam(self.pos_controller.parameters(), lr=cfg.algo.controller.learning_rate)
        self.atti_lya_opt = torch.optim.Adam(self.atti_lyapunov.parameters(), lr=cfg.algo.lyapunov.learning_rate)
        self.atti_dfun_opt = torch.optim.Adam(self.atti_dfunction.parameters(), lr=cfg.algo.dfunction.learning_rate)
        self.atti_ctrl_opt = torch.optim.Adam(self.atti_controller.parameters(), lr=cfg.algo.controller.learning_rate)

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
        acc_des = tensordict[("agents", "pos_control_output")].reshape(-1, 1)
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
        body_rate_des = tensordict[("agents", "atti_control_output")].reshape(-1, 3)
        CTBR = torch.cat([thrust, body_rate_des], dim=-1)
        CTBR = CTBR.reshape(*batch_shape, -1)
        tensordict.set(("agents","CTBR"), CTBR)
        return tensordict

    def train_lyapunov(self, tensordict: TensorDict, run):
        equilibrium_observation = self.observation_spec.zero()
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