import logging
import os
import sys
import time

import hydra
import torch
import numpy as np
import pandas as pd
import swanlab as wandb
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    VelController,
    AttitudeController,
    RateController,
    History
)
from omni_drones.controllers import DSLPIDController,LeePositionController,AttitudeController
# from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

FILE_PATH = os.path.dirname(__file__)


@hydra.main(config_path=FILE_PATH, config_name="eval", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    run = init_wandb(cfg)

    print(OmegaConf.to_yaml(cfg))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from dlearning.envs import DlearningHoverEnv
    from dlearning.learning import DLearning, HierarchicalDLearning, HierarchicalDLearning_pk
    from dlearning.utils import ControllerWrapper, HierarchicalControllerWrapper, DSLPIDControllerWrapper, tensordict_next_hierarchical_control
    from dlearning.controllers import Se3PositionController,Se3PositionControllerCTBR, DSLPIDController, CTBRController

    print('-----------Create ',cfg.task.name,'------------')
    env = DlearningHoverEnv(cfg = cfg, headless = cfg.headless)
    
    # # DONE: 使用transformer修改env的输入动作空间
    # action_transform: str = cfg.task.get("action_transform", None)
    # if action_transform is not None:
    #     if action_transform == "CTBR":
    #         from dlearning.controllers import RateController as _RateController
    #         transforms = [InitTracker()]
    #         controller = _RateController(g = np.linalg.norm(cfg.sim.gravity), uav_params = env.drone.params).to(env.device)
    #         transform = RateController(controller)
    #         transforms.append(transform)
    
    # env = TransformedEnv(env, Compose(*transforms)).train()
    
    env.set_seed(cfg.seed)

    # policy1 = Se3PositionControllerCTBR(
    #     g = np.linalg.norm(cfg.sim.gravity), 
    #     uav_params = env.drone.params
    #     ).to(env.device)
    # wrapped_policy = HierarchicalControllerWrapper(policy1)
    
    # policy1 = DSLPIDController(
    #     dt = cfg.sim.dt, 
    #     g = np.linalg.norm(cfg.sim.gravity), 
    #     uav_params = env.drone.params
    #     ).to(env.device)
    
    policy1 = CTBRController(
        dt = cfg.sim.dt, 
        g = np.linalg.norm(cfg.sim.gravity), 
        uav_params = env.drone.params
        ).to(env.device)
    wrapped_policy = DSLPIDControllerWrapper(policy1)

    # policy = HierarchicalDLearning(
    policy = HierarchicalDLearning_pk(
        cfg = cfg, 
        uav_params = env.drone.params, 
        observation_spec = env.observation_spec, 
        action_spec = env.action_spec, 
        controller = policy1,
        device = env.device
        )
    # frames_per_batch = env.num_envs * env.max_episode_length

    env.eval()
    trajs = env.rollout(
        max_steps=env.max_episode_length,
        policy=policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    policy.eval_pos_lyapunov(trajs,run)
    policy.eval_atti_lyapunov(trajs, run)
    policy.plot_lyapunov_contour(trajs, select = 'atti', save_fig = True, save_path = run.public.run_dir, index = 1)
    policy.plot_dfunction_contour(trajs, select = 'atti', save_fig = True, save_path = run.public.run_dir, index = 1)
    policy.plot_lyapunov_contour(trajs, 
                                 xlim = 1.0,
                                 ylim = 1.0,
                                 select = 'pos', 
                                 save_fig = True, 
                                 save_path = run.public.run_dir, 
                                 index = 1)
    policy.plot_dfunction_contour(trajs, 
                                  xlim = 1.0,
                                  ylim = 1.0,
                                  select = 'pos', 
                                  save_fig = True, 
                                  save_path = run.public.run_dir, 
                                  index = 1)     
    # env.reset()
    # trajs = env.rollout(
    #     max_steps=env.max_episode_length,
    #     policy=wrapped_policy,
    #     auto_reset=True,
    #     break_when_any_done=False,
    #     return_contiguous=False,
    # )
    # policy.eval_pos_lyapunov(trajs, run)
    # policy.eval_atti_lyapunov(trajs, run)
    # trajs = trajs.detach().clone()
    # trajs = tensordict_next_hierarchical_control(trajs)

    # policy.eval_atti_lyapunov(trajs, run)
    # policy.eval_atti_dfunction(trajs, run)
    
    import matplotlib.pyplot as plt
    # print(trajs["agents"]["atti_control_output"].shape)
    data = trajs["agents"]["atti_control_output"][6, :, 0, :].cpu().detach().numpy()
    data1 = trajs["agents"]["pos_control_input"][:, :, 0, 0].cpu().detach().numpy()
    # [64 16 1 3]
    timesteps = np.arange(data1.shape[1])
    colors = plt.cm.tab20(np.linspace(0, 1, 64))
    plt.figure(figsize=(12, 6))
    # plt.plot(timesteps, data1[:, 0], 
    #         color=colors[1],
    #         linewidth=1.5,
    #         label=f'atti error')
    # plt.plot(timesteps, data1[:, 3], 
    #         color=colors[2],
    #         linewidth=1.5,
    #         label=f'body rate')
    # plt.plot(timesteps, data[:, 0], 
    #         color=colors[3],
    #         linewidth=1.5,
    #         label=f'control torque')
    
    for i in range(64):
        plt.plot(timesteps, data1[i, :], 
                color=colors[i],
                linewidth=1.5,
                label=f'State {i+1}')
        
    # for i in range(3):
    #     plt.plot(timesteps, data[:, i], 
    #             color=colors[i+6],
    #             linewidth=1.5,
    #             label=f'Control {i+1}')

    # data = trajs["agents"]["atti_control_output"][:, :, 0, 0].cpu().detach().numpy()
    # colors = plt.cm.tab20(np.linspace(0, 1, data.shape[0]))
    # print(data.shape)
    # for i in range(data.shape[0]):
    #     plt.plot(timesteps, data[i, :], 
    #             color=colors[i],
    #             linewidth=1.5,
    #             label=f'agent {i}')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('State Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("atti_states_combined.png", bbox_inches='tight')
    plt.show()

    env.enable_render(not cfg.headless)
    env.reset()

    wandb.finish()
    simulation_app.close()

if __name__ == "__main__":
    main()
