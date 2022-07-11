import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ray
from ray import tune
from ray.tune.registry import register_env
import wandb
from dreamer.env import cheetah_run, finger_spin
from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)

from dreamer.dreamer import DREAMERTrainer
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

"""
8 Environments from Deepmind Control Suite
"""




def run_experiment():
    """Example for using a WandbLoggerCallback with the function API"""
    register_env("cheetah", lambda env_args: cheetah_run(**env_args))
    register_env("finger_spin", lambda env_args: finger_spin(**env_args))

    analysis = tune.run(
        DREAMERTrainer,
        name="dmc-dreamer",
        stop={"timesteps_total": 100000},
        local_dir=os.path.join("/data/ramans", "dmc"),
        checkpoint_at_end=True,
        num_samples=1,
        config={
            "seed": tune.grid_search([42, 1337, 13]),
            "batch_size": 64,
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            "min_train_iters": 100,
            "max_train_iters": 100,
            #"train_iters_per_rollout": 1,
            "env": tune.grid_search(["finger_spin", "cheetah"]),
            "dreamer_model": {
                "contrastive_loss": "triplet",
                "augment": {"type": "Augmentation", "params": {"consistent": False, "strong": tune.grid_search([True, False])}, "augmented_target": False}},
        },
        callbacks=[WandbLoggerCallback(api_key="fd1a595a3c1caa35b1f907727fb99c479fcf59ae", project="augmented_dreams", entity='neuromancers')]
    )

if __name__ == "__main__":
    run_experiment()