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

    print(OmegaConf.to_yaml(cfg))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from dlearning.envs.DlearningHoverEnv import DlearningHoverEnv
    from dlearning.learning.d_learning import DLearning
    
    print('-----------Create ',cfg.task.name,'------------')
    env = DlearningHoverEnv(cfg = cfg, headless = cfg.headless)
    env.set_seed(cfg.seed)

    policy = DLearning(cfg, env.observation_spec, env.action_spec, env.device)
    
    frames_per_batch = env.num_envs * env.max_episode_length
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    env.eval()
    trajs = env.rollout(
        max_steps=env.max_episode_length,
        policy=policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    dfunctionvalue = policy.dfunction(trajs)[("agents", "dfunction_value")]
    print('sum(dfunctionvalue): ',sum(dfunctionvalue))

    positive_D_count = torch.sum(dfunctionvalue > 0).float()
    total_D_count = torch.numel(dfunctionvalue)
    positive_D_ratio = positive_D_count / total_D_count
    print('positive_D_ratio: ',positive_D_ratio)

    env.enable_render(not cfg.headless)
    env.reset()

    simulation_app.close()


if __name__ == "__main__":
    main()
