# Copyright (c) 2025 Zhaolong Shen, BUAA
# SE3 controllers

import torch
import torch.nn as nn
from tensordict import TensorDict

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize, 
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
import yaml
import os.path as osp


def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    # 计算 mixer 矩阵，将力和力矩转换为电机油门
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])
    max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer


class RateController(nn.Module):

    def __init__(self, g, uav_params) -> None:
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"se3_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))


    def forward(
        self, 
        root_state: torch.Tensor, 
        CTBR: torch.Tensor,
    ):
        CTBR = CTBR.reshape(-1, 4)
        target_thrust, target_rate = CTBR.split([1, 3], -1)
        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel)
        rate_error = body_rate - target_rate
        # print(f"Rate error: {rate_error[0,:]}")
        acc_des = (
            - rate_error * self.ang_rate_gain
            + angvel.cross(angvel)
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        angacc_thrust = angacc_thrust.to(self.mixer.dtype)

        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd


class AttitudeController(nn.Module):
    r"""
    
    """
    def __init__(self, g, uav_params):
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_attitude = nn.Parameter(
            torch.tensor([3., 3., 0.035]) @ I[:3, :3].inverse()
        )
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self, 
        root_state: torch.Tensor, 
        target_thrust: torch.Tensor,
        target_yaw_rate: torch.Tensor=None,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*batch_shape, 1, device=device)
        if target_roll is None:
            target_roll = torch.zeros(*batch_shape, 1, device=device)
        
        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_yaw_rate=target_yaw_rate.reshape(-1, 1),
            target_roll=target_roll.reshape(-1, 1),
            target_pitch=target_pitch.reshape(-1, 1),
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self, 
        root_state: torch.Tensor,
        target_thrust: torch.Tensor, 
        target_yaw_rate: torch.Tensor, 
        target_roll: torch.Tensor,
        target_pitch: torch.Tensor
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        device = pos.device

        R = quaternion_to_rotation_matrix(rot)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw = axis_angle_to_matrix(yaw, torch.tensor([0., 0., 1.], device=device))
        roll = axis_angle_to_matrix(target_roll, torch.tensor([1., 0., 0.], device=device))
        pitch = axis_angle_to_matrix(target_pitch, torch.tensor([0., 1., 0.], device=device))
        R_des = torch.bmm(torch.bmm(yaw,  roll), pitch)
        angle_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )

        angle_error = torch.stack([
            angle_error_matrix[:, 2, 1], 
            angle_error_matrix[:, 0, 2], 
            torch.zeros(yaw.shape[0], device=device)
        ], dim=-1)

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        angular_rate_error = ang_vel - torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), angular_rate_des.unsqueeze(2)).squeeze(2)

        angular_acc = (
            - angle_error * self.gain_attitude 
            - angular_rate_error * self.gain_angular_rate 
            + torch.cross(ang_vel, ang_vel)
        )
        print(angular_acc.shape)
        print(target_thrust.shape)
        angular_acc_thrust = torch.cat([angular_acc, target_thrust], dim=-1)
        cmd = (self.mixer @ angular_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd


class Se3PositionController(nn.Module):
    """
    Computes rotor commands/force and torque for the given control target using the controller
    described in https://arxiv.org/abs/1003.2005.

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * control_target: tensor of shape (7,) contining target position, linear velocity,
        and yaw angle.
    
    Outputs:
        * cmd: tensor of shape (num_rotors,) containing the computed rotor commands.
        * force: tensor of shape (1,) containing the computed force.
        * torque: tensor of shape (3,) containing the computed torque.
        * controller_state: empty dict.
    """
    def __init__(
        self, 
        g: float, 
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"se3_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)
        
        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g]).abs())

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.attitude_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)

    def forward(
        self, 
        root_state: torch.Tensor, 
        target_pos: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        body_rate: bool=False
    ):
        batch_shape = root_state.shape[:-1] # 到最后一个shape之前的shape
        device = root_state.device
        if target_pos is None:
            target_pos = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_pos = target_pos.expand(batch_shape+(3,)).to(device)
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            pass
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)

        root_state = root_state.reshape(-1, 13)
        target_pos = target_pos.reshape(-1, 3)
        target_vel = target_vel.reshape(-1, 3)
        target_acc = target_acc.reshape(-1, 3)
        target_yaw =target_yaw.reshape(-1, 1)

        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        acc = self.position_control(
            pos, vel, 
            target_pos, target_vel, target_acc
        )

        b3_des = -normalize(acc)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ], dim=-1)
        
        b3_des = b3_des.to(torch.float32)
        b1_des = b1_des.to(torch.float32)

        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1)
        R = quaternion_to_rotation_matrix(rot)
        thrust = -self.mass * (acc * R[:, :, 2]).sum(-1, True)
        if not body_rate:
            ang_vel = quat_rotate_inverse(rot, ang_vel)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        M = self.attitude_control(ang_error_matrix, ang_vel)
        ang_acc_thrust = torch.cat([M, thrust], dim=-1)
        ang_acc_thrust = ang_acc_thrust.to(self.mixer.dtype)
        # ang_acc_thrust = self.direct_compute(
        #     root_state.reshape(-1, 13),
        #     target_pos.reshape(-1, 3),
        #     target_vel.reshape(-1, 3),
        #     target_acc.reshape(-1, 3),
        #     target_yaw.reshape(-1, 1),
        #     body_rate
        # )
        cmd = (self.mixer @ ang_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1 # 归一化的四个电机油门

        return cmd.reshape(*batch_shape, -1)

    def direct_compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            # convert angular velocity from world frame to body frame
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        pos_error = pos - target_pos
        vel_error = vel - target_vel

        acc = (
            pos_error * self.pos_gain 
            + vel_error * self.vel_gain 
            - self.g
            - target_acc
        )

        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b3_des = -normalize(acc)

        b1_des = b1_des.to(torch.float64)
        b3_des = b3_des.to(torch.float64)

        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1).to(R.dtype)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitude_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel)
        )
        thrust = (-self.mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1).to(R.dtype) # 推力和力矩
        return ang_acc_thrust
        # print('lee position controller ang_acc_thrust: ',ang_acc_thrust.shape)
        # cmd = (self.mixer @ ang_acc_thrust.T).T
        # cmd = (cmd / self.max_thrusts) * 2 - 1 # 归一划的四个电机油门
        # return cmd

    def position_control(self, pos, vel, target_pos, target_vel, target_acc):
        # 位置误差计算
        pos_error = pos - target_pos
        vel_error = vel - target_vel
        # 加速度命令生成
        acc = (
            pos_error * self.pos_gain 
            + vel_error * self.vel_gain 
            - self.g
            - target_acc
        )
        return acc
    
    def attitude_control(self, ang_error_matrix, ang_vel):
        """
        姿态控制器 - 计算力矩
        输入:
            rot: 当前旋转 (四元数) (batch, 4)
            ang_vel: 当前角速度 (机体坐标系) (batch, 3)
            R_des: 期望姿态矩阵 (batch, 3, 3)
        
        返回:
            M: 力矩 (batch, 3)
        """
        e_R = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ], dim=-1)
        
        e_Ω = ang_vel  # 假设期望角速度为0

        M = (
            -e_R * self.attitude_gain
            - e_Ω * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel)
        )
        return M
    

class Se3PositionControllerCTBR(nn.Module):
    """
    Computes thrust and body rates for the given control target

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * target_pos: tensor of shape (3,) containing target position.
        * target_vel: tensor of shape (3,) containing target linear velocity.
        * target_acc: tensor of shape (3,) containing target linear acceleration.
        * target_yaw: tensor of shape (1,) containing target yaw angle.
        * body_rate: bool, whether the input angular velocity is in body frame.
    
    Outputs:
        * CTBR: tensor of shape (4,) containing thrust and body rates (p, q, r).
    """
    def __init__(
        self, 
        g: float, 
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"se3_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)
        
        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g]).abs())

        inertia = uav_params["inertia"]
        
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.attitude_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)
        rotor_config = uav_params["rotor_configuration"]
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

    def forward(
        self, 
        root_state: torch.Tensor, 
        target_pos: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        use_body_rate: bool=False
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device
        if target_pos is None:
            target_pos = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_pos = target_pos.expand(batch_shape+(3,)).to(device)
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            pass
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)

        root_state = root_state.reshape(-1, 13)
        target_pos = target_pos.reshape(-1, 3)
        target_vel = target_vel.reshape(-1, 3)
        target_acc = target_acc.reshape(-1, 3)
        target_yaw = target_yaw.reshape(-1, 1)

        pos_control_input, pos_control_output, R_des, thrust = self.position_control(root_state, target_pos, target_vel, target_acc, target_yaw)
        # R_des = pos_control_output["R_des"]
        atti_control_input, atti_control_output = self.attitude_control(root_state, R_des, use_body_rate)
        
        pos_control_input = pos_control_input.reshape(*batch_shape, -1)
        pos_control_output = pos_control_output.reshape(*batch_shape, -1)
        atti_control_input = atti_control_input.reshape(*batch_shape, -1)
        atti_control_output = atti_control_output.reshape(*batch_shape, -1)
        thrust = thrust.reshape(*batch_shape, -1)

        # for td in [pos_control_input, pos_control_output, atti_control_input, atti_control_output]:
        #     for key in td:
        #         td[key] = td[key].reshape(*batch_shape, -1)

        # pos_control_input = TensorDict(pos_control_input, batch_size=batch_shape)
        # pos_control_output = TensorDict(pos_control_output, batch_size=batch_shape)
        # atti_control_input = TensorDict(atti_control_input, batch_size=batch_shape)
        # atti_control_output = TensorDict(atti_control_output, batch_size=batch_shape)

        # thrust = pos_control_output["thrust"]
        # body_rate_des = atti_control_output["body_rate_des"]
        body_rate_des = atti_control_output
        CTBR = torch.cat([thrust, body_rate_des], dim=-1)
        CTBR = CTBR.reshape(*batch_shape, -1)
        result = TensorDict(
            {
                "action": CTBR,
                "pos_control_input": pos_control_input,
                "pos_control_output": pos_control_output,
                "att_control_input": atti_control_input,
                "att_control_output": atti_control_output,
            },
            batch_size=batch_shape
        )
        return result

    def position_control(
        self, 
        root_state: torch.Tensor, 
        target_pos: torch.Tensor, 
        target_vel: torch.Tensor, 
        target_acc: torch.Tensor, 
        target_yaw: torch.Tensor
    ):
        pos, rot, vel, ang_vel_w = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        pos_error = pos - target_pos
        vel_error = vel - target_vel
        acc = (
            pos_error * self.pos_gain 
            + vel_error * self.vel_gain 
            - self.g
        )
        b3_des = -normalize(acc).to(torch.float32)
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
        thrust = -self.mass * (acc * R[:, :, 2]).sum(-1, True)
        # pos_control_input = {
        #     "pos_error": pos_error,
        #     "vel_error": vel_error,
        #     "attitude": rot,
        #     "target_yaw": target_yaw
        # }
        pos_control_input = torch.cat([pos_error, vel_error], dim=-1)
        # pos_control_output = {
        #     "acc_des": acc,
        #     "R_des": R_des,
        #     "thrust": thrust,
        # }
        pos_control_output = acc
        return pos_control_input, pos_control_output, R_des, thrust
    
    def attitude_control(
        self, 
        root_state: torch.Tensor,
        R_des: torch.Tensor, 
        use_body_rate: bool=False
    ):
        pos, rot, vel, ang_vel_w = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        R = quaternion_to_rotation_matrix(rot)
        if not use_body_rate:
            body_rate = quat_rotate_inverse(rot, ang_vel_w)
        else:
            body_rate = ang_vel_w
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        e_R = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ], dim=-1)
        body_rate_des = (
            - self.attitude_gain * e_R 
            - self.ang_rate_gain * body_rate 
        )
        # atti_control_input = {
        #     "R": R,
        #     "R_des": R_des,
        #     "e_R": e_R,
        #     "body_rate": body_rate,
        # }
        atti_control_input = torch.cat([e_R, body_rate], dim=-1)
        # atti_control_output = {
        #     "body_rate_des": body_rate_des,
        # }
        atti_control_output = body_rate_des
        return atti_control_input, atti_control_output
    
    def rate_control(
        self, 
        root_state: torch.Tensor, 
        CTBR: torch.Tensor,
        use_body_rate: bool=False
    ):
        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        CTBR = CTBR.reshape(-1, 4)
        target_rate = CTBR[..., 1:]
        target_thrust = CTBR[..., 0]

        pos, rot, linvel, ang_vel_w = root_state.split([3, 4, 3, 3], dim=1)
        if not use_body_rate:
            body_rate = quat_rotate_inverse(rot, ang_vel_w)
        else:
            body_rate = ang_vel_w
        rate_error = body_rate - target_rate
        print(f"Rate error: {rate_error[0,:]}")
        acc_des = (
            - rate_error * self.ang_rate_gain
            + ang_vel_w.cross(ang_vel_w)
        )
        target_thrust = target_thrust.unsqueeze(1)
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        angacc_thrust = angacc_thrust.to(self.mixer.dtype)

        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd