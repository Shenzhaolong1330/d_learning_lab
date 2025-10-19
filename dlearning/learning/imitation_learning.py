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

class Imitation_learning():
    '''
    Behavior Cloning 
    learning hierarchical controller
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

        dummy_input = TensorDict({
            "agents": {
                "pos_control_input": torch.zeros(4, 6, device=self.device),
                "pos_control_output": torch.zeros(4, 3, device=self.device),
                # "atti_control_input": torch.zeros(4, 6, device=self.device),
                "atti_control_input": torch.zeros(4, 3, device=self.device),
                "atti_control_output": torch.zeros(4, 3, device=self.device),
            }
        }, batch_size = 4).to(self.device)
        
        self.pos_controller = TensorDictModule(
            NNController(self.cfg.algo.controller.pos, pos_action_dim),
            [("agents", "pos_control_input")],
            [("agents", "pos_control_output")]
        ).to(self.device)
        self.pos_controller(dummy_input)
    
        self.atti_controller = TensorDictModule(
            NNController(self.cfg.algo.controller.atti, atti_action_dim),
            [("agents", "atti_control_input")],
            [("agents", "atti_control_output")]
        ).to(self.device)
        self.atti_controller(dummy_input)

        if self.cfg.checkpoint_path is not None:
            pos_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"pos_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")

            atti_controller_ckpt_path = os.path.join(self.cfg.checkpoint_path, f"atti_controller_checkpoint_episode_{self.cfg.checkpoint_episode}.pt")
            
            if os.path.exists(pos_controller_ckpt_path):
                pos_controller_state_dict = torch.load(pos_controller_ckpt_path)
                self.pos_controller.load_state_dict(pos_controller_state_dict, strict=False)
                logging.info(f"Loaded pos_Controller checkpoint from {pos_controller_ckpt_path}")
                print('--------------------controller loaded------------------------')
            else:
                logging.warning(f"pos_Controller checkpoint not found at {pos_controller_ckpt_path}")
                       
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
            self.pos_controller.apply(kaiming_init_)
            self.atti_controller.apply(kaiming_init_)

        self.pos_ctrl_opt = torch.optim.Adam(
            self.pos_controller.parameters(), 
            lr=cfg.algo.controller.pos.learning_rate
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

    def pos_policy_learning(self, tensordict: TensorDict, run):
        
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
            
            controller_correction = torch.sum((stable_action - nn_action)**2) + torch.sum((equilibrium_action - nn_action_0)**2)

            loss = controller_correction * 1.0 + self.param_sum_square(self.pos_controller.module) * 0.001
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                self.pos_ctrl_opt.step()
                self.pos_ctrl_opt.zero_grad()

            positive_D_count = torch.sum(D > 0).float()
            total_D_count = torch.numel(D)
            positive_D_ratio = positive_D_count / total_D_count

            step_info = {
                "pos_policy_loss": loss.item(),
                "pos_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)

    # check
    def atti_policy_learning(self, tensordict: TensorDict, run):

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
            

            controller_correction = torch.sum((stable_action - nn_action)**2) + torch.sum((nn_action_0)**2)

            with torch.enable_grad():
                loss = controller_correction * 1.0 + self.param_sum_square(self.atti_controller.module) * 0.001
            loss.backward()
            with torch.no_grad(): 
                self.atti_ctrl_opt.step()
                self.atti_ctrl_opt.zero_grad()

            step_info = {
                "atti_policy_loss": loss.item(),
                "atti_policy_correction":controller_correction,
            }
            if run is not None:
                run.log(step_info)
