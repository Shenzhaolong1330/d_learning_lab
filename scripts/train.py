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


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) # 把所有的前两个dimension合并成为batch
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


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
    from dlearning.envs.DlearningHoverEnv import DlearningHoverEnv
    from dlearning.learning.d_learning import DLearning
    from dlearning.utils.controller_wrapper import ControllerWrapper
    
    print('-----------Create ',cfg.task.name,'------------')
    env = DlearningHoverEnv(cfg = cfg, headless = cfg.headless)
    env.set_seed(cfg.seed)

    policy = LeePositionController(g = np.linalg.norm(cfg.sim.gravity), uav_params = env.drone.params).to(env.device)
    wrapped_policy = ControllerWrapper(policy)

    d_learning = DLearning(cfg, env.observation_spec, env.action_spec, env.device)

    '''
    TODO：定义评估指标
        - 收敛速度
        - 鲁棒性
        - 跟踪误差
    '''
    
    frames_per_batch = env.num_envs * env.max_episode_length
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
        # policy=d_learning,
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
        # 这里有一个batch的数据 frames_per_batch = env.num_envs * env.max_episode_length
        # 循环到收集够 total_frames 为止, 即 total_frames // frames_per_batch * frames_per_batch iters
        
        # episode 指的是从环境初始化到环境结束
        # iteration 指的是训练过程中收集一个批次数据并更新策略的完整周期
        # batch 指的是一次更新策略所使用的数据样本数量

        data = data.detach().clone()
        episode_stats.add(data.to_tensordict())
        
        # if len(episode_stats) >= env.num_envs:
            # 判断 episode_stats 中收集的 episode 是不是大于 env.num_envs
            # stats = {
            #     "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
            #     for k, v in episode_stats.pop().items(True, True)
            # }
            # info.update(stats)
        batch = make_batch(data, cfg.num_minibatches)
        for minibatch in batch:
            d_learning.train_lyapunov(minibatch, run)
            d_learning.train_dfunction(minibatch, run)
            d_learning.policy_improvement(minibatch, run)

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break 

        if save_interval > 0 and (i-1) % save_interval == 0:
            try:
                lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"lyapunovfunction_checkpoint_episode_{i}.pt")
                dfunction_ckpt_path = os.path.join(run.public.run_dir, f"dfunction_checkpoint_episode_{i}.pt")
                controller_ckpt_path = os.path.join(run.public.run_dir, f"controller_checkpoint_episode_{i}.pt")
                # lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"lyapunovfunction_checkpoint_final.pt")
                # dfunction_ckpt_path = os.path.join(run.public.run_dir, f"dfunction_checkpoint_final.pt")
                # controller_ckpt_path = os.path.join(run.public.run_dir, f"controller_checkpoint_final.pt")
                torch.save(d_learning.lyapunovfunction.state_dict(), lyapunov_ckpt_path)
                logging.info(f"Saved checkpoint to {str(lyapunov_ckpt_path)}")
                torch.save(d_learning.dfunction.state_dict(), dfunction_ckpt_path)
                logging.info(f"Saved checkpoint to {str(dfunction_ckpt_path)}")
                torch.save(d_learning.controller.state_dict(), controller_ckpt_path)
                logging.info(f"Saved checkpoint to {str(controller_ckpt_path)}")

            except AttributeError:
                logging.warning(f"LyapunovFunction {d_learning.lyapunovfunction} does not implement `.state_dict()`")

    lyapunov_ckpt_path = os.path.join(run.public.run_dir, f"lyapunovfunction_checkpoint_final.pt")
    dfunction_ckpt_path = os.path.join(run.public.run_dir, f"dfunction_checkpoint_final.pt")
    controller_ckpt_path = os.path.join(run.public.run_dir, f"controller_checkpoint_final.pt")
    torch.save(d_learning.lyapunovfunction.state_dict(), lyapunov_ckpt_path)
    logging.info(f"Saved checkpoint to {str(lyapunov_ckpt_path)}")
    torch.save(d_learning.dfunction.state_dict(), dfunction_ckpt_path)
    logging.info(f"Saved checkpoint to {str(dfunction_ckpt_path)}")
    torch.save(d_learning.controller.state_dict(), controller_ckpt_path)
    logging.info(f"Saved checkpoint to {str(controller_ckpt_path)}")

    env.eval()
    trajs = env.rollout(
        max_steps=env.max_episode_length,
        policy=wrapped_policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    env.enable_render(not cfg.headless)
    env.reset()
    
    d_learning.eval_lyapunov(trajs, run)

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
