import os
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

from dreamer.dreamer import DREAMERTrainer
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
    register_env("cheetah", lambda x: cheetah_run())
    register_env("hopper", lambda x: hopper_hop())
    register_env("finger_spin", lambda x: finger_spin())

    def timesteps_based_on_env(spec):
        if spec.config.env == "finger_spin" or spec.config.env=="cheetah":
            return 500000
        elif spec.config.env == "hopper":
            return 1000000
        raise ValueError("Wrong env")

    analysis = tune.run(
        DREAMERTrainer,
        name="dmc-dreamer",
        stop={"timesteps_total": 500000},
        local_dir=os.path.join(os.getcwd(), "dmc"),
        checkpoint_at_end=True,
        #restore="/Users/kfarid/PycharmProjects/hyperdreamer/dmc/dmc-dreamer/DREAMER_cheetah_run-v20_beca2_00000_0_2021-08-13_18-37-27/checkpoint_26/checkpoint-26",
        num_samples=1,
        config={
            "seed": tune.grid_search([42, 1337]),
            "batch_size": 64,
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            #"min_train_iters": 10,#tune.grid_search([100, 10]),
            #"max_train_iters": 500,#tune.sample_from(lambda spec: 100 if spec.config.min_train_iters == 100 else 500),
            #"train_iters_per_rollout": 1,
            "env": tune.grid_search(["finger_spin", "cheetah"]),
            "dreamer_model": tune.grid_search([
                {"augment": {"type": "RandShiftsAug", "params": {"consistent": True}, "augmented_target": False}},
                {"augment": {"type": "RandShiftsAug", "params": {"consistent": False}, "augmented_target": False}},
                {"augment": None},
            ])
        },
        callbacks=[WandbLoggerCallback(api_key="fd1a595a3c1caa35b1f907727fb99c479fcf59ae", project="augmented_dreams", entity='neuromancers')]
    )

if __name__ == "__main__":
    run_experiment()