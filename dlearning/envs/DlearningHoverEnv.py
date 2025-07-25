# Copyright (c) 2025 Zhaolong Shen, Beihang University
# this is a d-learning env for hover task

import functorch
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils
# 只有simulation开始之后才能创建prim，所以在train中需要把这部分放在simulation_app = init_simulation_app(cfg)之后

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec


def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)


class DlearningHoverEnv(IsaacEnv):
    """
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. This task is designed
    to serve as a sanity check.

    ## Observation
    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position), 
      containing its rotation (in quaternion 4), velocities (linear and angular 6), 
      heading 1 and up vectors 1, and the current throttle 4.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional vector encoding the current
      progress of the episode.
        
    ## Episode End
    The episode ends when the drone mishebaves, i.e., it crashes into the ground or flies too far away:

    ```{math}
        d_\text{pos} > 4 \text{ or } x^w_z < 0.2
    ```
    
    or when the episode reaches the maximum length.


    ## Config

    | Parameter               | Type  | Default   | Description |
    |-------------------------|-------|-----------|-------------|
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
    | `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the drone and its target. |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `has_payload`           | bool  | False     | Indicates whether the drone has a payload attached. If set to True, it means that a payload is attached; otherwise, if set to False, no payload is attached. |


    """
    def __init__(self, cfg, headless):
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.time_encoding = cfg.task.time_encoding

        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()
        
        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        match cfg.task.init_state:
            case 'fixed_zero':
                print("init as fixed_zero")
                self.init_pos_dist = D.Uniform(
                    torch.tensor([0., 0., 8.0], device=self.device),
                    torch.tensor([0., 0., 8.0], device=self.device)
                )
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
                    torch.tensor([0., 0., 0.], device=self.device) * torch.pi
                )
            case 'random1':
                print("init as random1")
                self.init_pos_dist = D.Uniform(
                    torch.tensor([-1.5, -1.5, 8.5], device=self.device),
                    torch.tensor([1.5, 1.5, 7.5], device=self.device)
                )
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
                    torch.tensor([0.2, 0.2, 0.], device=self.device) * torch.pi
                )
            case 'random2':
                print("init as random2")
                self.init_pos_dist = D.Uniform(
                    torch.tensor([-1.8, -1.8, 6.2], device=self.device),
                    torch.tensor([1.8, 1.8, 9.8], device=self.device)
                )
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([-.5, -.5, 0.], device=self.device) * torch.pi,
                    torch.tensor([0.5, 0.5, 0.], device=self.device) * torch.pi
                )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )
        self.target_pos = torch.tensor([[0.0, 0.0, 8.0]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 8.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(), 
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.5, 0.5, 8.)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "transformed_drone_state": UnboundedContinuousTensorSpec((1, 12), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way 
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device), 
                env_indices=env_ids, joint_indices=joint_indices)
            
            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        self.stats[env_ids] = 0.
    

    def compute_parameters(
        self,
        rotor_config,
        inertia_matrix,
    ):
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

    # def _pre_sim_step(self, tensordict: TensorDictBase):
    #     actions是油门
    #     actions = tensordict[("agents", "action")]
    #     self.effort = self.drone.apply_action(actions)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # actions是力和力矩
        uav_params = self.drone.params
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        mixer = torch.nn.Parameter(self.compute_parameters(rotor_config, I)).to(self.device)
        # print(mixer.shape)
        # print(actions.shape)
        cmd = (mixer @ actions.squeeze(1).T).T.unsqueeze(1)
        # print(cmd.shape)

        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        max_thrusts = torch.nn.Parameter(max_rot_vel.square() * force_constants).to(self.device)

        cmd = (cmd / max_thrusts) * 2 - 1

        self.effort = self.drone.apply_action(cmd)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        # torch.Size([64, 1, 23])
        self.info["drone_state"][:] = self.root_state[..., :13] # Position 3, Quaternion 4, Velocity linear+angular 6
        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        # # torch.Size([64, 1, 3])
        self.rheading = self.target_heading - self.root_state[..., 13:16]
        # # torch.Size([64, 1, 3])
        obs = [self.rpos, self.root_state[..., 3:], self.rheading,]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        # 自定义的标准化的观测格式
        quaternion = self.root_state[..., 3:7]
        euler = quaternion_to_euler(quaternion)
        linear_velocity = self.root_state[...,7:10]
        angular_velocity = self.root_state[...,10:13]
        transformed_drone_state = torch.cat([self.rpos, linear_velocity, euler, angular_velocity], dim=-1)
    
        obs = torch.cat(obs, dim=-1)

        return TensorDict({
            "agents": {
                "observation": obs,
                "transformed_drone_state": transformed_drone_state,
                "intrinsics": self.drone.intrinsics
            },
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)
        
        distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)

        reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)
        # uprightness
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # spin reward
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose 
            + reward_pose * (reward_up + reward_spin) 
            + reward_effort 
            + reward_action_smoothness
        )
        
        terminated = (self.drone.pos[..., 2] < 0.2) | (distance > 4)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.root_state[..., 18], (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated
            },
            self.batch_size,
        )
