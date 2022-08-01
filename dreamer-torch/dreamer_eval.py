import argparse
import collections
import functools
import os
import pathlib
import random
import sys
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers


import wandb
import torch
from torch import nn
from torch import distributions as torchd
from torchvision.transforms import Resize
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger, dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = 0
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = dataset
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()

  def __call__(self, obs, reset, state=None, reward=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _policy(self, obs, state, training):
    if self._config.curl:
      images = torch.tensor(obs['image'], device=self._config.device)
      orig_size = images.size()
      resized_images = tools.center_crop_image(images.permute(0, 3, 1, 2), self._config.augment_crop_size).permute(0, 2, 3, 1)
      obs['image'] = resized_images
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs))
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)
    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data):
    metrics = {}
    post, context, mets = self._wm._train(data)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior._train(start, reward)[-1])
    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, distractor_env):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size, None, distractor_env)
        env = wrappers.NormalizeActions(env)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, config.size,
            grayscale=config.grayscale,
            life_done=False and ('train' in mode),
            sticky_actions=True,
            all_actions=True)
        env = wrappers.OneHotAction(env)
    elif suite == 'dmlab':
        env = wrappers.DeepMindLabyrinth(
            task,
            mode if 'train' in mode else 'test',
            config.action_repeat)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    env = wrappers.RewardObs(env)
    return env


def main(config):
    root_dir = pathlib.Path(config.logdir)
    logdir = root_dir / "evaluation_distracting_easy"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)

    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    if config.wandb_api_key:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        wandb.init(name=config.wandb_name, entity=config.wandb_entity, project=config.wandb_project, dir=str(logdir),
                   config=vars(config))
        wandb.tensorboard.patch(root_logdir=str(logdir))
    
    logger = tools.Logger(logdir, 0)
    make = lambda mode: make_env(config, mode, '')
    eval_envs = [make('eval') for _ in range(config.envs)]
    acts = eval_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]
    agent = Dreamer(config, logger, None).to(config.device)

    agent.requires_grad_(requires_grad=False)
    if (logdir / '..' / 'latest_model.pt').exists():
        agent.load_state_dict(torch.load(logdir / '..' / 'latest_model.pt'))
        agent._should_pretrain._once = False
    eval_policy = functools.partial(agent, training=False)

    for distractor_env in ['easy', '']:#['easy', 'medium', 'hard', '']:
      eval_env = make_env(config, 'eval', distractor_env)
      eps_return_mean = 0
      episodes_to_log = []
      n_episodes = 10
      eids_to_log = random.choices(list(range(n_episodes)), k=2)
      for i in range(n_episodes):
        eval_return, obses = evaluate_one_episode(eval_policy, eval_env)
        eps_return_mean += eval_return / n_episodes
        if i in eids_to_log:
          episodes_to_log.append(obses['image'])
        if distractor_env != '':
          eval_env = make_env(config, 'eval', distractor_env)
      for i, episode in enumerate(episodes_to_log):
        logger.video(f'eval_{distractor_env}_{i}', episode[None])
      logger.scalar(f'eval_mean_return_{distractor_env}', float(eps_return_mean))
      logger.write()
    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass


def evaluate_one_episode(agent, env):
  # Initialize or unpack simulation state.
  done = False
  obs = env.reset()
  agent_state = None
  eps_return = 0
  obses = [obs]
  reward = 0
  while not done:
    # Step agents.
    obs = {k: np.stack([v]) for k, v  in obs.items()}
    action, agent_state = agent(obs, np.array([done]), agent_state, np.array([reward]))
    if isinstance(action, dict):
      action = {k: np.array(action[k][0].detach().cpu()) for k in action}
    # Step envs.
    obs, reward, done, _ = env.step(action)
    eps_return += reward
    obses.append(obs)
  obses = {k: np.stack([frame[k] for frame in obses]) for k in obses[0].keys()}
  return eps_return, obses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
