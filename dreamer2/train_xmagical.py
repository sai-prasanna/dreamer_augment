import os
import numpy as np
os.environ['MUJOCO_GL'] = 'egl'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ray import tune
from ray.tune.registry import register_env
import xmagical
from ray.tune.integration.wandb import (
    WandbLoggerCallback,
)
import gym
from gym.spaces.box import Box
from dreamer2.dreamer import DREAMERTrainer
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

"""
8 Environments from Deepmind Control Suite
"""


class XMagical(gym.Wrapper):
    def __init__(
        self,
        task,
        embodiment,
        observation_space,
        view_mode,
        variant,
        use_dense_reward,
        max_episode_steps,
    ):

        import xmagical

        xmagical.register_envs()
        env = gym.make(
            f"{task}-{embodiment}-{observation_space}-{view_mode}-{variant}-v0",
            res_hw=(64, 64),
            use_dense_reward=use_dense_reward,
            max_episode_steps=max_episode_steps,
        )
        super().__init__(env)
        if observation_space == "Pixels":
            self._obs_key = "image"
        else:
            self._obs_key = "state"
        self.observation_space = Box(low=0, high=255, shape=(3, 64, 64))
        self.embodiment = embodiment

    def reset(self):
        obs = super().reset()

        return obs.transpose((2, 0, 1)).astype(np.float32)

    def step(self, action):
        # if self.embodiment != "Gripper":
        #     # Ignore final action for envs other than gripper
        #     action = action[:2]
        obs, reward, done, info = super().step(action)
        obs = obs.transpose((2, 0, 1)).astype(np.float32)
        
        done = done or self.env.score_on_end_of_traj() == 1.0

        return obs, reward, done, info


def run_experiment():
    """Example for using a WandbLoggerCallback with the function API"""
    register_env("xmagical-Longstick", lambda x: XMagical("SweepToTop", "Longstick", "Pixels", "Allo", "Demo", True, 500))
    register_env("xmagical-Shortstick", lambda x: XMagical("SweepToTop", "Shortstick", "Pixels", "Allo", "Demo", True, 500))
    register_env("xmagical-Mediumstick", lambda x: XMagical("SweepToTop", "Mediumstick", "Pixels", "Allo", "Demo", True, 500))

    analysis = tune.run(
        DREAMERTrainer,
        name="xmagical-dreamer",
        stop={"timesteps_total": 500000},
        local_dir=os.path.join(os.getcwd(), "xmagical"),
        checkpoint_at_end=True,
        num_samples=1,
        config={
            "seed": 42,
            "batch_size": 50,
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 0,
            "env": "xmagical-Mediumstick",
            "dreamer_model": {"augment": None},
            "env_config": {"frame_skip": 1},
            "horizon": 500,
            "actor_lr": 1e-4,
            # Critic LR
            "critic_lr": 1e-4,
            "explore_noise": 0.0
        },
        callbacks=[WandbLoggerCallback(api_key="fd1a595a3c1caa35b1f907727fb99c479fcf59ae", project="augmented_dreams", entity='neuromancers')]
    )

if __name__ == "__main__":
    run_experiment()