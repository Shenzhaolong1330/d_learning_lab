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
# from omni_drones.controllers import DSLPIDController,LeePositionController,AttitudeController
# from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

FILE_PATH = os.path.dirname(__file__)


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
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
    from dlearning.utils import ControllerWrapper, HierarchicalControllerWrapper, DSLPIDControllerWrapper, make_batch, tensordict_next_hierarchical_control
    from dlearning.controllers import Se3PositionControllerCTBR, DSLPIDController, CTBRController
    
    print('-----------Create ',cfg.task.name,'------------')
    env = DlearningHoverEnv(cfg = cfg, headless = cfg.headless)
    
    # set action space as CTBR
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

    # policy = Se3PositionControllerCTBR(
    #     g = np.linalg.norm(cfg.sim.gravity), 
    #     uav_params = env.drone.params
    #     ).to(env.device)
    # wrapped_policy = HierarchicalControllerWrapper(policy)

    # policy = DSLPIDController(
    #     dt = cfg.sim.dt, 
    #     g = np.linalg.norm(cfg.sim.gravity), 
    #     uav_params = env.drone.params
    #     ).to(env.device)

    policy = CTBRController(
        dt = cfg.sim.dt, 
        g = np.linalg.norm(cfg.sim.gravity), 
        uav_params = env.drone.params
        ).to(env.device)
    wrapped_policy = DSLPIDControllerWrapper(policy)

    # d_learning = HierarchicalDLearning(
    d_learning = HierarchicalDLearning_pk(
        cfg = cfg, 
        uav_params = env.drone.params, 
        observation_spec = env.observation_spec, 
        action_spec = env.action_spec, 
        controller = policy,
        device = env.device
        )

    frames_per_batch = env.num_envs * env.max_episode_length
    # print('frames_per_batch',frames_per_batch)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    collector = SyncDataCollector(
        env,
        # policy = d_learning,
        policy = wrapped_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    pbar = tqdm(collector)
    # env.train()
    print('------------start training-------------')
    for i, data in enumerate(pbar):
        # env.reset()
        data = data.detach().clone()
        data = tensordict_next_hierarchical_control(data)
        # episode_stats.add(data.to_tensordict())
        '''
        所有数据同时训练，GD的步数就多一些
        使用minibatch训练，GD的步数就少一些
        '''
        # d_learning.train_pos_lyapunov(data, run)
        # d_learning.train_pos_dfunction(data, run)
        # d_learning.pos_policy_improvement(data, run)
        # d_learning.train_atti_lyapunov(data, run)
        # d_learning.train_atti_dfunction(data, run)
        # d_learning.atti_policy_improvement(data, run)
        batch = make_batch(data, cfg.num_minibatches)
        j=0
        for minibatch in tqdm(batch, desc="Processing minibatches"):
            
            if cfg.task.train_mode == 'pos':
                d_learning.train_pos_lyapunov(minibatch, run)
                d_learning.train_pos_dfunction(minibatch, run)
            elif cfg.task.train_mode == 'atti':
                d_learning.train_atti_lyapunov(minibatch, run)
                d_learning.train_atti_dfunction(minibatch, run)
                
            d_learning.pos_policy_improvement(minibatch, run)
            d_learning.atti_policy_improvement(minibatch, run)
            j+=1
        # pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})
        # env.reset()
            if save_interval > 0 and (j-1) % save_interval == 0:
                pos_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"pos_lyapunov_checkpoint_minibatch_{j}.pt")
                # pos_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"pos_dfunction_checkpoint_episode_{i}.pt")
                pos_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"pos_dynamics_checkpoint_minibatch_{j}.pt")
                pos_controller_ckpt_path = os.path.join(run.public.run_dir, f"pos_controller_checkpoint_minibatch_{j}.pt")
                atti_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"atti_lyapunov_checkpoint_minibatch_{j}.pt")
                # atti_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"atti_dfunction_checkpoint_episode_{i}.pt")
                atti_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"atti_dynamics_checkpoint_minibatch_{j}.pt")
                atti_controller_ckpt_path = os.path.join(run.public.run_dir, f"atti_controller_checkpoint_minibatch_{j}.pt")
                torch.save(d_learning.pos_lyapunov.state_dict(), pos_lyapunov_ckpt_path)
                # torch.save(d_learning.pos_dfunction.state_dict(), pos_dfunction_ckpt_path)
                torch.save(d_learning.pos_dynamics.state_dict(), pos_dynamics_ckpt_path)
                torch.save(d_learning.pos_controller.state_dict(), pos_controller_ckpt_path)
                torch.save(d_learning.atti_lyapunov.state_dict(), atti_lyapunov_ckpt_path)
                # torch.save(d_learning.atti_dfunction.state_dict(), atti_dfunction_ckpt_path)
                torch.save(d_learning.atti_dynamics.state_dict(), atti_dynamics_ckpt_path)
                torch.save(d_learning.atti_controller.state_dict(), atti_controller_ckpt_path)
            print(f"model saved in minibatch{j}")

        if max_iters > 0 and i >= max_iters - 1:
            break 

        if save_interval > 0 and (i-1) % save_interval == 0:
            pos_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"pos_lyapunov_checkpoint_episode_{i}.pt")
            # pos_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"pos_dfunction_checkpoint_episode_{i}.pt")
            pos_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"pos_dynamics_checkpoint_episode_{i}.pt")
            pos_controller_ckpt_path = os.path.join(run.public.run_dir, f"pos_controller_checkpoint_episode_{i}.pt")
            atti_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"atti_lyapunov_checkpoint_episode_{i}.pt")
            # atti_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"atti_dfunction_checkpoint_episode_{i}.pt")
            atti_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"atti_dynamics_checkpoint_episode_{i}.pt")
            atti_controller_ckpt_path = os.path.join(run.public.run_dir, f"atti_controller_checkpoint_episode_{i}.pt")
            torch.save(d_learning.pos_lyapunov.state_dict(), pos_lyapunov_ckpt_path)
            # torch.save(d_learning.pos_dfunction.state_dict(), pos_dfunction_ckpt_path)
            torch.save(d_learning.pos_dynamics.state_dict(), pos_dynamics_ckpt_path)
            torch.save(d_learning.pos_controller.state_dict(), pos_controller_ckpt_path)
            torch.save(d_learning.atti_lyapunov.state_dict(), atti_lyapunov_ckpt_path)
            # torch.save(d_learning.atti_dfunction.state_dict(), atti_dfunction_ckpt_path)
            torch.save(d_learning.atti_dynamics.state_dict(), atti_dynamics_ckpt_path)
            torch.save(d_learning.atti_controller.state_dict(), atti_controller_ckpt_path)
            print(f"model saved in episode{i}")

    pos_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"pos_lyapunov_checkpoint_episode_final.pt")
    # pos_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"pos_dfunction_checkpoint_episode_final.pt")
    pos_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"pos_dynamics_checkpoint_episode_final.pt")
    pos_controller_ckpt_path = os.path.join(run.public.run_dir, f"pos_controller_checkpoint_episode_final.pt")
    atti_lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"atti_lyapunov_checkpoint_episode_final.pt")
    # atti_dfunction_ckpt_path = os.path.join(run.public.run_dir, f"atti_dfunction_checkpoint_episode_final.pt")
    atti_dynamics_ckpt_path = os.path.join(run.public.run_dir, f"atti_dynamics_checkpoint_episode_final.pt")
    atti_controller_ckpt_path = os.path.join(run.public.run_dir, f"atti_controller_checkpoint_episode_final.pt")
    torch.save(d_learning.pos_lyapunov.state_dict(), pos_lyapunov_ckpt_path)
    # torch.save(d_learning.pos_dfunction.state_dict(), pos_dfunction_ckpt_path)
    torch.save(d_learning.pos_dynamics.state_dict(), pos_dynamics_ckpt_path)
    torch.save(d_learning.pos_controller.state_dict(), pos_controller_ckpt_path)
    torch.save(d_learning.atti_lyapunov.state_dict(), atti_lyapunov_ckpt_path)
    # torch.save(d_learning.atti_dfunction.state_dict(), atti_dfunction_ckpt_path)
    torch.save(d_learning.atti_dynamics.state_dict(), atti_dynamics_ckpt_path)
    torch.save(d_learning.atti_controller.state_dict(), atti_controller_ckpt_path)
    print(f"final model saved to ",run.public.run_dir)

    env.eval()
    env.reset()
    trajs = env.rollout(
        max_steps=256,
        policy=wrapped_policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    trajs = trajs.detach().clone()
    trajs = tensordict_next_hierarchical_control(trajs)
    
    d_learning.eval_pos_lyapunov(trajs, run) 
    d_learning.eval_atti_lyapunov(trajs, run)
    d_learning.eval_atti_dfunction(trajs, run)  
    env.enable_render(not cfg.headless)
    env.reset()

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
