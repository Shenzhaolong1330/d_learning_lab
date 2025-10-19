# Copyright (c) 2025 Zhaolong Shen, Beihang University

import os
import logging
import sys

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


from dlearning.utils import (
    LyapunovFunction,
    StructuredLyapunovFunction,
    BacksteppingLyapunovFunction,
    NeuralBacksteppingLyapunovFunction,
    DFunction,
    DFunctionwithPriorKnowledge,
    NNController,
    GRUController,
    Dynamics
)

import matplotlib.pyplot as plt

class HierarchicalDLearning_pk():
    '''
    Hierarchical D-Learning with Prior Knowledge
    1. pos control -> desired acc
    2. desired acc -> desired attitude
    3. atti control -> desired body rate
    4. desired body rate + desired thrust -> cmd
    '''
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
        # self.equilibrium_thrust = 7.0231 # hummingbird
        self.mass = torch.tensor(uav_params["mass"])
        self.controller = controller
        # 初始化神经网络
        # 构造完整的虚拟输入 TensorDict
        dummy_input = TensorDict({
            "agents": {
                "pos_control_input": torch.zeros(4, 6, device=self.device),
                "pos_control_output": torch.zeros(4, 3, device=self.device),
                # "atti_control_input": torch.zeros(4, 6, device=self.device),
                "atti_control_input": torch.zeros(4, 3, device=self.device),
                "atti_control_output": torch.zeros(4, 3, device=self.device),
            }
        }, batch_size = 4).to(self.device)
        
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
        self.pos_dynamics = Dynamics(self.cfg.algo.dynamics.pos).to(self.device)
        self.pos_dynamics(dummy_input[("agents", "pos_control_input")], dummy_input[("agents", "pos_control_output")])
        
        # attitude part 
        self.atti_lyapunov = TensorDictModule(
            LyapunovFunction(self.cfg.algo.lyapunov.atti),
            # BacksteppingLyapunovFunction(self.cfg.algo.lyapunov.atti),
            # NeuralBacksteppingLyapunovFunction(self.cfg.algo.lyapunov.atti),
            [("agents", "atti_control_input")],
            [("agents", "atti_lyapunov_value")]
        ).to(self.device)
        self.atti_lyapunov(dummy_input)
        self.atti_dfunction = TensorDictModule(
            DFunction(self.cfg.algo.dfunction.atti),
            # DFunctionwithPriorKnowledge(self.cfg.algo.dfunction.atti),
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
        self.atti_dynamics = Dynamics(self.cfg.algo.dynamics.atti).to(self.device)
        self.atti_dynamics(dummy_input[("agents", "atti_control_input")], dummy_input[("agents", "atti_control_output")])

        if self.cfg.checkpoint_path is not None:
            pos_lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_lyapunov_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            # pos_dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_dfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            pos_dynamics_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_dynamics_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            pos_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")

            atti_lyapunov_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_lyapunov_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            # atti_dfunction_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_dfunction_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            atti_dynamics_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_dynamics_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            atti_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            # TODO: load dynamics model
            if os.path.exists(pos_lyapunov_ckpt_path):
                pos_lyapunov_state_dict = torch.load(pos_lyapunov_ckpt_path)
                self.pos_lyapunov.load_state_dict(pos_lyapunov_state_dict, strict=False)
                logging.info(f"Loaded pos_lyapunov checkpoint from {pos_lyapunov_ckpt_path}")
            else:
                logging.warning(f"pos_lyapunov checkpoint not found at {pos_lyapunov_ckpt_path}")

            # if os.path.exists(pos_dfunction_ckpt_path):
            #     pos_dfunction_state_dict = torch.load(pos_dfunction_ckpt_path)
            #     self.pos_dfunction.load_state_dict(pos_dfunction_state_dict, strict=False)
            #     logging.info(f"Loaded pos_DFunction checkpoint from {pos_dfunction_ckpt_path}")
            # else:
            #     logging.warning(f"pos_DFunction checkpoint not found at {pos_dfunction_ckpt_path}")
            if os.path.exists(pos_dynamics_ckpt_path):
                pos_dynamics_state_dict = torch.load(pos_dynamics_ckpt_path)
                self.pos_dynamics.load_state_dict(pos_dynamics_state_dict, strict=False)
                logging.info(f"Loaded pos_Dynamics checkpoint from {pos_dynamics_ckpt_path}")
            else:
                logging.warning(f"pos_Dynamics checkpoint not found at {pos_dynamics_ckpt_path}")
            
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

            # if os.path.exists(atti_dfunction_ckpt_path):
            #     atti_dfunction_state_dict = torch.load(atti_dfunction_ckpt_path)
            #     self.atti_dfunction.load_state_dict(atti_dfunction_state_dict, strict=False)
            #     logging.info(f"Loaded atti_DFunction checkpoint from {atti_dfunction_ckpt_path}")
            # else:
            #     logging.warning(f"atti_DFunction checkpoint not found at {atti_dfunction_ckpt_path}")
            if os.path.exists(atti_dynamics_ckpt_path):
                atti_dynamics_state_dict = torch.load(atti_dynamics_ckpt_path)
                self.atti_dynamics.load_state_dict(atti_dynamics_state_dict, strict=False)
                logging.info(f"Loaded atti_Dynamics checkpoint from {atti_dynamics_ckpt_path}")
            else:
                logging.warning(f"atti_Dynamics checkpoint not found at {atti_dynamics_ckpt_path}")
            
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

        self.pos_lya_opt = torch.optim.AdamW(
            self.pos_lyapunov.parameters(), 
            lr=cfg.algo.lyapunov.pos.learning_rate,
            weight_decay=0.01
            )   
        self.pos_dfun_opt = torch.optim.Adam(
            self.pos_dfunction.parameters(), 
            lr=cfg.algo.dfunction.pos.learning_rate
            )
        self.pos_dfun_opt_pk = torch.optim.Adam(
            self.pos_dynamics.parameters(), 
            lr=cfg.algo.dynamics.pos.learning_rate
            )
        self.pos_ctrl_opt = torch.optim.Adam(
            self.pos_controller.parameters(), 
            lr=cfg.algo.controller.pos.learning_rate
            )
        self.atti_lya_opt = torch.optim.AdamW(
            self.atti_lyapunov.parameters(), 
            lr=cfg.algo.lyapunov.atti.learning_rate,
            weight_decay=0.01
            )
        self.atti_dfun_opt = torch.optim.Adam(
            self.atti_dfunction.parameters(), 
            lr=cfg.algo.dfunction.atti.learning_rate
            )
        self.atti_dfun_opt_pk = torch.optim.Adam(
            self.atti_dynamics.parameters(), 
            lr=cfg.algo.dynamics.atti.learning_rate
            )
        self.atti_ctrl_opt = torch.optim.Adam(
            self.atti_controller.parameters(), 
            lr=cfg.algo.controller.atti.learning_rate
            )
        
    

    def __call__(
        self, 
        tensordict: TensorDict
        ):
        print('call hierarchical d-learning controller')
        '''
        pos control -> desried acc -> desired attitude -> 
        atti control -> desired body rate -> 
        lower level CTBR controller convert to cmd
        '''
        # get state
        root_state = tensordict.get(("agents", "observation"))[...,:13]
        # target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        # target_yaw = torch.zeros_like(root_state[..., 0])
        batch_shape = root_state.shape[:-1]
        # reshape data to (batch_size, value)
        root_state = root_state.reshape(-1, 13)
        # target_yaw =target_yaw.reshape(-1, 1)
        pos, quat, vel, ang_vel_w = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        # if control_target is None:
        control_target = torch.zeros_like(root_state[..., :7])
        target_pos, target_vel, target_yaw = torch.split(control_target, [3, 3, 1], dim=-1)
        pos_error = target_pos - pos
        vel_error = target_vel - vel

        # pos control
        pos_control_input = torch.cat([pos_error, vel_error], dim=-1)
            # call d-learning trained pos controller
        tensordict.set(("agents","pos_control_input"), pos_control_input.reshape(*batch_shape, -1))
        self.pos_controller(tensordict)
        acc_des = tensordict[("agents", "pos_control_output")].reshape(-1, 3)
        b3_des = normalize(acc_des)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ], dim=-1).to(torch.float32)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        rot_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1)
        rot = quaternion_to_rotation_matrix(quat)
 
        # atti control
        ang_error_matrix = (
            torch.bmm(rot_des.transpose(-2, -1), rot) 
            - torch.bmm(rot.transpose(-2, -1), rot_des)
        )
        e_rot = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ], dim=-1)
        atti_control_input = e_rot
            # call d-learning trained atti controller
        tensordict.set(("agents","atti_control_input"), atti_control_input.reshape(*batch_shape, -1))
        self.atti_controller(tensordict)
        body_rate = tensordict[("agents", "atti_control_output")].reshape(-1, 3)
        print('e_rot:', e_rot[0], ' body_rate: ', body_rate[0])
        
        thrust = acc_des.reshape(*batch_shape, -1)
        scalar_thrust = torch.matmul(thrust, rot)[..., 2]
        CTBR = torch.cat([scalar_thrust, body_rate], dim=-1).reshape(*batch_shape, -1)


        if CTBR.ndim == 2:
            CTBR = CTBR.unsqueeze(1)
        cmd = self.controller.CTBR2cmd(
            root_state = root_state,
            CTBR = CTBR
        )

        tensordict.set(("agents","action"), cmd)
        return tensordict

    # check
    def train_pos_lyapunov(self, tensordict: TensorDict, run):
        '''
        函数功能正常
        '''
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

            loss = SemiNegativeDefinite*10 + PositiveDefinite*10 + EquilibriumValue + self.param_sum_square(self.pos_lyapunov.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_lya_opt.step()
                self.pos_lya_opt.zero_grad()

            step_info = {
                "pos_lyapunov_loss": loss.item(),
                "pos_lyapunov_semi_negative_definite": SemiNegativeDefinite.item(),
                "pos_lyapunov_positive_definite": PositiveDefinite.item(),
                "pos_lyapunov_negative_V_ratio":negative_V_ratio,
                "pos_lyapunov_positive_Vdot_ratio":positive_Vdot_ratio,
                "pos_lyapunov_equilibrium_value": EquilibriumValue.item(),
            }
            if run is not None:
                run.log(step_info)

    # check
    def train_atti_lyapunov(self, tensordict: TensorDict, run):
        '''
        函数功能正常
        '''
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

            loss = SemiNegativeDefinite * 100 + PositiveDefinite * 10 + EquilibriumValue*1 + self.param_sum_square(self.atti_lyapunov.module) * 0.01
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.atti_lya_opt.step()
                self.atti_lya_opt.zero_grad()

            step_info = {
                "atti_lyapunov_loss": loss.item(),
                "atti_lyapunov_semi_negative_definite": SemiNegativeDefinite.item(),
                "atti_lyapunov_positive_definite": PositiveDefinite.item(),
                "atti_lyapunov_negative_V_ratio":negative_V_ratio,
                "atti_lyapunov_positive_Vdot_ratio":positive_Vdot_ratio,
                "atti_lyapunov_equilibrium_value": EquilibriumValue.item(),
            }
            if run is not None:
                run.log(step_info)


    def pos_dfunction_core(self,control_input, control_output):
        '''
        pos dfunction with prior knowledge
        '''
        # state: pos error vel error
        control_input = control_input.detach().requires_grad_(True)
        # action: acceleration
        control_output = control_output.detach().requires_grad_(True)
        pos_lyapunov_value = self.pos_lyapunov.module(control_input) # torch.Size([512, 1, 1])
        # atti_lyapunov_value = torch.sum(control_input**2, dim=-1, keepdim=True) 
        grad_outputs = torch.ones_like(pos_lyapunov_value)
        gradients = torch.autograd.grad(
            outputs=pos_lyapunov_value,
            inputs=control_input,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            )[0] # torch.Size([512, 1, 3])
        '''
        dynamcis = nn_dynamics + nominal_dynamics
        nominal_dynamics = control_input + self.cfg.sim.dt*control_output
        '''
        # nominal_dynamics
        pos_error = control_input[..., :3]
        vel_error = control_input[..., 3:]
        # 离散神经网络动力学
        # nominal_dynamics_pos = pos_error + self.cfg.sim.dt*vel_error + 0.5*(self.cfg.sim.dt**2)*control_output
        # nominal_dynamics_vel = vel_error + self.cfg.sim.dt*control_output
        # nominal_dynamics = torch.cat([nominal_dynamics_pos, nominal_dynamics_vel], dim=-1)
        # state_next = self.pos_dynamics(control_input, control_output) +  nominal_dynamics
        # dyn = (state_next - control_input) / self.cfg.sim.dt
        # 连续神经网络动力学
        nominal_dynamics = torch.cat([vel_error, control_output], dim=-1)
        dyn = self.pos_dynamics(control_input, control_output) + nominal_dynamics

        dfunction_value = torch.sum(gradients * dyn, dim=-1, keepdim=True) # torch.Size([512, 1, 1])
        return dfunction_value


    def pos_dfunction_pk(self, tensordict: TensorDict):
       
        pos_control_input = tensordict[("agents","pos_control_input")].detach().requires_grad_(True)
        pos_control_output = tensordict[("agents","pos_control_output")].detach().requires_grad_(True)
        dfunction_value = self.pos_dfunction_core(pos_control_input, pos_control_output)
        tensordict.set(("agents","pos_dfunction_value"), dfunction_value)
        return tensordict


    def train_pos_dfunction(self, tensordict: TensorDict, run):
        '''
        引入先验知识的D函数训练
        '''
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","pos_control_input"),torch.zeros_like(tensordict[("agents","pos_control_input")]))
        equilibrium_action = torch.zeros_like(tensordict[("agents","pos_control_output")])
        equilibrium_action[..., -1] = 0.31794 # crazyflie的equilibrium_action
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
            # D0 = self.pos_dfunction(equilibrium_observation)[('agents','pos_dfunction_value')]
            # D = self.pos_dfunction(tensordict)[('agents','pos_dfunction_value')]
            D0 = self.pos_dfunction_pk(equilibrium_observation)[('agents','pos_dfunction_value')]
            D = self.pos_dfunction_pk(tensordict)[('agents','pos_dfunction_value')]

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)
            # loss = fitting_loss  + self.param_sum_square(self.pos_dfunction.module) * 0.001
            loss = fitting_loss  + self.param_sum_square(self.pos_dynamics) * 0.001  + torch.sum(F.relu(D)) * 0.1
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                # self.pos_dfun_opt.step()
                # self.pos_dfun_opt.zero_grad()
                self.pos_dfun_opt_pk.step()
                self.pos_dfun_opt_pk.zero_grad()
            # loss_values.append(loss.item())
            step_info = {
                "pos_dfunction_loss": loss.item(),
                "pos_dfunction_fitting_loss":fitting_loss.item(),
                "pos_dfunction_positive_D_ratio":positive_D_ratio, 
                "pos_dfunction_positive_penalty":torch.sum(F.relu(D)).item(), 
            }
            if run is not None:
                run.log(step_info)

    # check
    def atti_dfunction_core(self,control_input, control_output):
        '''
        atti dfunction with prior knowledge
        '''
        # state: angular error
        control_input = control_input.detach().requires_grad_(True)
        # action: angular velocity
        control_output = control_output.detach().requires_grad_(True)
        atti_lyapunov_value = self.atti_lyapunov.module(control_input) # torch.Size([512, 1, 1])
        # atti_lyapunov_value = torch.sum(control_input**2, dim=-1, keepdim=True) 
        grad_outputs = torch.ones_like(atti_lyapunov_value)
        gradients = torch.autograd.grad(
            outputs=atti_lyapunov_value,
            inputs=control_input,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            )[0] # torch.Size([512, 1, 3])
        '''
        dynamcis = nn_dynamics + nominal_dynamics
        nominal_dynamics = control_input + self.cfg.sim.dt*control_output
        '''
        # 神经网络动力学也要讨论离散和连续两个情况的表现
        # 离散神经网络动力学
        # state_next = self.atti_dynamics(control_input, control_output) +  control_input + self.cfg.sim.dt*control_output # torch.Size([512, 1, 3])
        # dyn = (state_next - control_input) / self.cfg.sim.dt
        # 连续神经网络动力学
        dyn = self.atti_dynamics(control_input, control_output) + control_output
        dfunction_value = torch.sum(gradients * dyn, dim=-1, keepdim=True) # torch.Size([512, 1, 1])
        return dfunction_value

    # check
    def atti_dfunction_pk(self, tensordict: TensorDict):
       
        atti_control_input = tensordict[("agents","atti_control_input")].detach().requires_grad_(True)
        atti_control_output = tensordict[("agents","atti_control_output")].detach().requires_grad_(True)
        dfunction_value = self.atti_dfunction_core(atti_control_input, atti_control_output)
        tensordict.set(("agents","atti_dfunction_value"), dfunction_value)
        return tensordict
        
    # check
    def train_atti_dfunction(self, tensordict: TensorDict, run):
        '''
        一体化的D函数效果不好??
        替换成引入先验信息的D函数
        替换之后训练效果还行
        '''
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

        loss_fn = nn.MSELoss()
        for i in range(self.cfg.algo.learning.atti.dfunction_GD_steps):
            # D0 = self.atti_dfunction(equilibrium_observation)[('agents','atti_dfunction_value')]
            # D = self.atti_dfunction(tensordict)[('agents','atti_dfunction_value')]
            D0 = self.atti_dfunction_pk(equilibrium_observation)[('agents','atti_dfunction_value')]
            D = self.atti_dfunction_pk(tensordict)[('agents','atti_dfunction_value')]

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            state_norm = tensordict[("agents","atti_control_input")].norm(dim=-1, keepdim=True)

            # fitting_loss = torch.sum(loss_fn(Vdot,D)/(state_norm+0.00001)) + torch.sum(D0**2)
            fitting_loss = torch.sum(loss_fn(Vdot,D)) + torch.sum(D0**2)
            # loss = fitting_loss  + self.param_sum_square(self.atti_dfunction.module) * 0.01
            loss = fitting_loss  + self.param_sum_square(self.atti_dynamics) * 0.001  + torch.sum(F.relu(D)) * 0.1
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                # self.atti_dfun_opt.step()
                # self.atti_dfun_opt.zero_grad()
                self.atti_dfun_opt_pk.step()
                self.atti_dfun_opt_pk.zero_grad()

            step_info = {
                "atti_dfunction_loss": loss.item(),
                "atti_dfunction_fitting_loss":fitting_loss.item(),
                "atti_dfunction_positive_D_ratio":positive_D_ratio,
                "atti_dfunction_positive_penalty":torch.sum(F.relu(D)).item(),
            }
            if run is not None:
                run.log(step_info)


    def pos_policy_improvement(self, tensordict: TensorDict, run):
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","pos_control_input"),torch.zeros_like(tensordict[("agents","pos_control_input")]))
        equilibrium_action = torch.zeros_like(tensordict[("agents","pos_control_output")])
        equilibrium_action[..., -1] = 0.31794 # crazyflie的equilibrium_action
        # equilibrium_action = TensorDict({"agents": {"pos_control_output": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        # equilibrium_observation = equilibrium_observation.update(equilibrium_action)
        stable_action = tensordict[('agents','pos_control_output')]
        for i in range(self.cfg.algo.learning.pos.controller_GD_steps): 
            nn_action = self.pos_controller(tensordict)[('agents','pos_control_output')]
            nn_action_0 = self.pos_controller(equilibrium_observation)[('agents','pos_control_output')]
            
            if stable_action.shape != nn_action.shape:
                raise ValueError("控制器输出不一致")
            
            # D = self.pos_dfunction(tensordict)[('agents','pos_dfunction_value')]
            D = self.pos_dfunction_pk(tensordict)[('agents','pos_dfunction_value')]
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)

            controller_correction = torch.sum((stable_action - nn_action)**2) + torch.sum((equilibrium_action - nn_action_0)**2)

            loss = self.dvalue_upper_bound_mean_variance_loss(D) * 0.0 + controller_correction * 1.0 + self.param_sum_square(self.pos_controller.module) * 0.001
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_ctrl_opt.step()
                self.pos_ctrl_opt.zero_grad()

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "pos_policy_loss": loss.item(),
                "pos_policy_positive_D_ratio":positive_D_ratio,
                "pos_policy_positive_penalty":positive_penalty,
                "pos_policy_upper_bound":upper_bound,
                "pos_policy_mean":mean,
                "pos_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)

    # check
    def atti_policy_improvement(self, tensordict: TensorDict, run):
        '''
        为什么策略提升没有作用？？问题表现为：
            优化采样点上的D函数的正惩罚和上界，效果不明显
            是D函数的问题吗，记得检查一下
                D函数好像没学好，因为positive_D_ratio = positive_D_count / total_D_count比较大
                画出来V和D
        '''
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","atti_control_input"),torch.zeros_like(tensordict[("agents","atti_control_input")]))
        stable_action = tensordict[('agents','atti_control_output')]
        # original_tensordict = tensordict.clone()
        for i in range(self.cfg.algo.learning.atti.controller_GD_steps): 

            atti_control_input = tensordict[('agents','atti_control_input')]
            # atti_control_output = self.atti_controller.module(atti_control_input)
            nn_action = self.atti_controller(tensordict)[('agents','atti_control_output')]
            nn_action_0 = self.atti_controller(equilibrium_observation)[('agents','atti_control_output')]

            if stable_action.shape != nn_action.shape:
                raise ValueError("控制器输出不一致")
            
            # D = self.atti_dfunction.module(atti_control_input, atti_control_output)
            D = self.atti_dfunction_pk(tensordict)[('agents','atti_dfunction_value')]
            positive_penalty = torch.sum(torch.relu(D))
            upper_bound = torch.max(D)
            mean = torch.mean(D)

            controller_correction = torch.sum((stable_action - nn_action)**2) + torch.sum((nn_action_0)**2)

            with torch.enable_grad():
                loss = self.dvalue_upper_bound_mean_variance_loss(D)*0.0 + controller_correction * 1.0 + self.param_sum_square(self.atti_controller.module) * 0.001
            loss.backward()
            with torch.no_grad(): 
                self.atti_ctrl_opt.step()
                self.atti_ctrl_opt.zero_grad()

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "atti_policy_loss": loss.item(),
                "atti_policy_positive_D_ratio":positive_D_ratio,
                "atti_policy_positive_penalty":positive_penalty,
                "atti_policy_upper_bound":upper_bound,
                "atti_policy_mean":mean,
                "atti_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)

    def dvalue_upper_bound_mean_variance_loss(self, dvalue):
        """
        """
        # 确保控制量使dvalue小于
        positive_penalty = torch.sum(torch.relu(dvalue))
        # 确保dvalue上界(平衡点附近)是0
        upper_bound = torch.max(dvalue)**2
        # dvalue下界
        lower_bound = torch.min(dvalue)
        # 确保dvalue均值小于0
        mean = torch.mean(dvalue)
        # 减小dvalue方差
        variance = torch.var(dvalue)
        # return upper_bound*100 + lower_bound*0 + mean*30 + variance*0 + positive_penalty*10
        return upper_bound*0 + lower_bound*0 + mean*0 + variance*0 + positive_penalty*1.0
    
    

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


    def eval_atti_dfunction(self, tensordict: TensorDict, run):
        equilibrium_observation = TensorDict({}, batch_size=tensordict.shape[:-1])
        equilibrium_observation.set(("agents","atti_control_input"),torch.zeros_like(tensordict[("agents","atti_control_input")]))
        equilibrium_action = torch.zeros_like(tensordict[("agents","atti_control_output")])
        equilibrium_action = TensorDict({"agents": {"atti_control_output": equilibrium_action}}, batch_size=equilibrium_action.shape[:-1])
        equilibrium_observation = equilibrium_observation.update(equilibrium_action)

        D0 = self.atti_dfunction(equilibrium_observation)[('agents','atti_dfunction_value')]

        dt = self.cfg.sim.dt
        next_tensordict = tensordict["next"]
        V = self.atti_lyapunov(tensordict)[("agents", "atti_lyapunov_value")]
        V_ = self.atti_lyapunov(next_tensordict)[("agents", "atti_lyapunov_value")]
        Vdot = (V_ - V) / dt

        D = self.atti_dfunction(tensordict)[('agents','atti_dfunction_value')]
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
            "atti_dfunction_eval_fitting_loss":fitting_loss.item(),
            "atti_dfunction_eval_positive_D_ratio":positive_D_ratio,
        }
        run.log(eval_info)
        print("atti_dfunction_eval_fitting_loss",fitting_loss.item(),
            "atti_dfunction_eval_positive_D_ratio",positive_D_ratio)


    def param_sum_square(self, net: nn.Module):
        param_squares = [p ** 2 for p in net.parameters()]
        return  sum(torch.sum(p) for p in param_squares)


    def plot_lyapunov_contour(
        self,
        tensordict: TensorDict,
        xlim = 0.3,
        ylim = 0.3,
        select = 'atti', # pos
        save_fig = 1,
        save_path = None,
        index = 0
    ):
        print('-----------------------Plotting {}------------------------'.format(select))
        if select == 'pos':
            state_num = 6
        elif select == 'atti':
            state_num = 3

        import numpy as np
        x = np.linspace(-xlim, xlim, 100)
        y = np.linspace(-ylim, ylim, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        torch.manual_seed(114514)
        others = torch.rand(1,state_num-2).to(tensordict.device)*0.3
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(tensordict.device)
                if state_num > 2:
                    xy = torch.cat((xy, others), dim=1).to(tensordict.device)
                if select == 'pos':
                    Z[i, j] = self.pos_lyapunov.module(xy).item()
                elif select == 'atti':  
                    Z[i, j] = self.atti_lyapunov.module(xy).item()

        fig = plt.figure(clear=True)
        ax = fig.add_subplot()
        contour = ax.contourf(X, Y, Z, levels=25, cmpa='RdBu', alpha = 1)
        cbar = plt.colorbar(contour)
        cbar.set_label('Lyapunov Value')
        if select == 'pos':
            ax.set_xlabel(r"ex / $m$")
            ax.set_ylabel(r"ey / $m$")
        elif select == 'atti':
            ax.set_xlabel(r"Pitch angle / $rad$")
            ax.set_ylabel(r"Roll angle / $rad$")

        # ax.set_aspect(1)
        # ax.set_xlabel(r"$x_{err}$")
        # ax.set_ylabel(r"$y_{err}$")

        ax.set_title(select +' Lyapunov Contour Plot' + ' in Iteration {}'.format(index))
        ax.grid(True)
        if save_fig:
            fig.savefig(save_path+'/'+select +' Lypaunov Contour' + ' in Iteration {}'.format(index)+'.png', dpi=600)
        # fig.show()
        # plt.show()
        plt.clf()
        plt.close()


    def plot_dfunction_contour(
        self,
        tensordict: TensorDict,
        xlim = 0.3,
        ylim = 0.3,
        select = 'atti', # pos
        save_fig = 1,
        save_path = None,
        index = 0
    ):
        # TODO 检查D函数的绘制/检查学到的D函数是否正确
        print('-----------------------Plotting {}------------------------'.format(select))
        if select == 'pos':
            state_num = 6
        elif select == 'atti':
            state_num = 3

        import numpy as np
        x = np.linspace(-xlim, xlim, 100)
        y = np.linspace(-ylim, ylim, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        torch.manual_seed(114514)
        others = torch.rand(1,state_num-2).to(tensordict.device)*0.2
        # others = torch.zeros(1,state_num-2).to(tensordict.device)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(tensordict.device)
                if state_num > 2:
                    xy = torch.cat((xy, others), dim=1).to(tensordict.device)
                if select == 'pos':
                    # Z[i, j] = self.pos_lyapunov.module(xy).item()
                    pos_control_output = self.pos_controller.module(xy)
                    # Z[i, j] = self.pos_dfunction.module(xy,pos_control_output).item()
                    Z[i, j] = self.pos_dfunction_core(xy,pos_control_output).item()
                elif select == 'atti':  
                    # Z[i, j] = self.atti_lyapunov.module(xy).item()
                    atti_control_output = self.atti_controller.module(xy)
                    # atti_control_output = torch.tensor([5.0, 5.0, 3.0]).to(tensordict.device)*0.8*xy
                    # Z[i, j] = self.atti_dfunction.module(xy,atti_control_output).item()
                    Z[i, j] = self.atti_dfunction_core(xy,atti_control_output).item()

        fig = plt.figure(clear=True)
        ax = fig.add_subplot()
        contour = ax.contourf(X, Y, Z, levels=25, cmap='RdBu', alpha = 1)
        cbar = plt.colorbar(contour)
        cbar.set_label('D-function Value')

        contour_zero = ax.contour(X, Y, Z, levels=[0], colors='orange', linewidths=2, linestyles='-')
        ax.clabel(contour_zero, inline=True, fontsize=10, fmt='Z=0')
        if select == 'pos':
            ax.set_xlabel(r"ex / $m$")
            ax.set_ylabel(r"ey / $m$")
        elif select == 'atti': 
            ax.set_xlabel(r"Pitch angle / $rad$")
            ax.set_ylabel(r"Roll angle / $rad$")

        # ax.set_aspect(1)
        # ax.set_xlabel(r"$x_{err}$")
        # ax.set_ylabel(r"$y_{err}$")

        ax.set_title(select +' D-function Contour Plot' + ' in Iteration {}'.format(index))
        ax.grid(True)
        if save_fig:
            fig.savefig(save_path+'/'+select +' Dfunction Contour' + ' in Iteration {}'.format(index)+'.png', dpi=600)
        # fig.show()
        # plt.show()
        plt.clf()
        plt.close()