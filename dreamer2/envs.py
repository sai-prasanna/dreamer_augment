import os
import gym
import numpy as np


from gym import Env, Wrapper


class GymWrapper(Wrapper):
    def __init__(self, env: Env, obs_key: str='image', act_key: str='action') -> None:
        super().__init__(env)
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
    
    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = done
        obs['is_terminal'] = info.get('is_terminal', done)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        return obs


class DMC(Wrapper):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            env = suite.load(domain, task)
        super().__init__(env)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
                quadruped_fetch=2, locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {'action': action}

    def step(self, action):
        assert np.isfinite(action['action']).all(), action['action']
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action['action'])
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            'reward': reward,
            'is_first': False,
            'is_last': time_step.last(),
            'is_terminal': time_step.discount == 0,
            'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update({
            k: v for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys})
        return obs, reward, time_step.last(), {}

    def reset(self):
        time_step = self._env.reset()
        obs = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update({
            k: v for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys})
        return obs

