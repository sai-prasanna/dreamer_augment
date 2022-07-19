import os

from dreamer2.utils import create_env
os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ray
from ray import tune
from ray.tune.registry import register_env
import wandb
from ray.rllib.examples.env.dm_control_suite import cheetah_run, hopper_hop
from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)

from dreamer2.dreamer import DREAMERTrainer
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

"""
8 Environments from Deepmind Control Suite
"""


def finger_spin(
    from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True
):
    return DMCEnv(
        "finger",
        "spin",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first,
    )


def run_experiment():
    """Example for using a WandbLoggerCallback with the function API"""
    register_env("cheetah", lambda x: create_env(cheetah_run, **x))
    register_env("finger_spin", lambda x: create_env(finger_spin, **x))


    analysis = tune.run(
        DREAMERTrainer,
        name="dmc-dreamer",
        stop={"timesteps_total": 100000},
        local_dir=os.path.join("/data/ramans/dmc", "dreamer2"),
        checkpoint_at_end=True,
        num_samples=1,
        config={
            "seed": tune.grid_search([42, 3048, 1337]),
            "prefill_timesteps": 1000,
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            #"train_iters_per_rollout": 1,
            "env": tune.grid_search(["cheetah", "finger_spin"]),
            "env_config": {"frame_skip": 2},
            "pretrain": 100,
            "replay": {"prioritize_ends": False},
            "dreamer_model": {
                "clip_rewards": "identity",
                "model_opt": {"lr": 3e-4},
                "actor_ent": 1e-4,
                "rssm": {"ensemble": 1, "deter": 200, "hidden": 200, "min_std": 0.1},}
        },
        callbacks=[WandbLoggerCallback(api_key="fd1a595a3c1caa35b1f907727fb99c479fcf59ae", project="augmented_dreams", entity='neuromancers')]

    )

if __name__ == "__main__":
    run_experiment()