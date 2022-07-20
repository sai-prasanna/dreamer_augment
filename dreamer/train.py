import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
import ray
from ray import tune
from ray.tune.suggest.bohb import *
from ray.tune.schedulers import *

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


def run_experiment(env):
    """Example for using a WandbLoggerCallback with the function API"""

    analysis = tune.run(
        DREAMERTrainer,
        name="dmc-dreamer",
        stop={"timesteps_total": 100000},
        local_dir=os.path.join("../dreamer_augment/data", "dmc"),
        checkpoint_at_end=True,
        config={
            "seed": tune.grid_search([42, 1337]),
            "batch_size": 4,
            "batch_length": 7,
            "td_model_lr": tune.loguniform(6e-5, 1e-3),
            "actor_lr": tune.loguniform(6e-6, 1e-4),
            "critic_lr": tune.loguniform(6e-6, 1e-4),
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            #"env": tune.grid_search(["finger_spin", "cheetah"]),
            "env": env,
            "dreamer_model": {
                "contrastive_loss": "cpc_augment",
                "augment": {"type": "Augmentation", 
                            "params": {"consistent": False, "strong": tune.choice([True, False])}, 
                            "augmented_target": False}
                }
        },
        metric="episode_reward_mean",
        mode="max",
        num_samples = 10,
        callbacks=[WandbLoggerCallback(api_key="fd1a595a3c1caa35b1f907727fb99c479fcf59ae", project="augmented_dreams", entity='neuromancers')]
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    



if __name__ == "__main__":
    register_env("cheetah", lambda env_args: cheetah_run(**env_args))
    register_env("finger_spin", lambda env_args: finger_spin(**env_args))
    run_experiment("cheetah")
    run_experiment("finger_spin")
