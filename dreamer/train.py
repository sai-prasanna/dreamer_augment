import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
    base_config = {
            "td_model_lr": 9e-4,
            "actor_lr": 8e-5,
            "critic_lr": 8e-5,
            "framework": "torch",
            "num_gpus": 1,
            "min_train_iters": 200,
            "max_train_iters": 200,
            "env": env,
            "dreamer_model": {
                "contrastive_loss": "cpc",
                "augment": {"type": "Augmentation", "params": {"consistent": False}},
        }}

    algo = TuneBOHB(
        metric="episode_reward_mean",
#        points_to_evaluate=[base_config],
        mode="max")

    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        #reduction_factor = 2,
        #grace_period = 20, 
        #metric="episode_reward_mean",
        #mode="max",
        max_t=120)

    analysis = tune.run(
        DREAMERTrainer,
        name="dmc-dreamer",
        stop={"timesteps_total": 100000},
        local_dir=os.path.join("../dreamer_augment/data/ramans", "dmc"),
        checkpoint_at_end=True,
        config={
            #"seed": tune.choice([42, 1337]),
            "batch_size": 50,
            "batch_length": 50,
            "td_model_lr": tune.loguniform(1e-4, 6e-3),
            "actor_lr": tune.loguniform(1e-5, 6e-4),
            "critic_lr": tune.loguniform(1e-5, 6e-4),
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            "min_train_iters": 200,
            "max_train_iters": 200,
            #"env": tune.grid_search(["finger_spin", "cheetah"]),
            "env": env,
            "dreamer_model": {
                "contrastive_loss": "cpc_augment",
                "augment": {"type": "Augmentation", 
                            "params": {"consistent": False, "strong": tune.choice([True, False])}, 
                            "augmented_target": False}
                }
        },
        scheduler=bohb,
        search_alg=algo,
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
