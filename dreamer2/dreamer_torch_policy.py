import logging

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import (
    LocalOptimizer,
    TensorType,
    AgentID
)
from dreamer2.dreamer_model import DreamerModel
import dreamer2

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td
    from ray.rllib.utils.torch_utils import convert_to_torch_tensor

if TYPE_CHECKING:
    from ray.rllib.policy.torch_policy import TorchPolicy
    
logger = logging.getLogger(__name__)


# This is the computation graph for workers (inner adaptation steps)
def compute_dreamer_loss(
    obs,
    action,
    reward,
    model: DreamerModel,
    log=False,
):
    """Constructs loss for the Dreamer objective

    Args:
        obs (TensorType): Observations (o_t)
        action (TensorType): Actions (a_(t-1))
        reward (TensorType): Rewards (r_(t-1))
        model (TorchModelV2): DreamerModel, encompassing all other models
        log (bool): If log, generate gifs
    """
    discount = obs["discount"]  if "discount" in obs else None
    if discount:
        del obs["discount"]
    states, wm_losses = model.world_model.compute_states_and_losses(obs, action, reward, discount, log)
    ac_losses = model.actor_critic.compute_losses(model.world_model, states)
    return_dict = {
        **wm_losses,
        **ac_losses
    }
    return return_dict



def dreamer_loss(policy, model, dist_class, train_batch):
    log_gif = False
    if "log_gif" in train_batch:
        log_gif = True
    policy.stats_dict = compute_dreamer_loss(
        {k: convert_to_torch_tensor(v, policy.device) for k, v in train_batch["obs"].items()},
        train_batch["actions"],
        train_batch["rewards"],
        policy.model,
        log_gif,
    )

    loss_dict = policy.stats_dict

    return (loss_dict["model_loss"], loss_dict["actor_loss"], loss_dict["critic_loss"])


def build_dreamer_model(policy, obs_space, action_space, config):

    model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        config["dreamer_model"],
        name="DreamerModel",
        framework="torch",
    )

    policy.model_variables = model.variables()

    return model


def action_sampler_fn(policy, model, input_dict, state_action, explore, timestep):
    """Action sampler function has two phases. During the prefill phase,
    actions are sampled uniformly [-1, 1]. During training phase, actions
    are evaluated through DreamerPolicy and an additive gaussian is added
    to incentivize exploration.
    """
    obs = input_dict["obs"]

    # Custom Exploration
    if timestep <= policy.config["prefill_timesteps"]:
        logp = None
        # Random action in space [-1.0, 1.0]
        action = 2.0 * torch.rand(1, model.action_space.shape[0]) - 1.0
        state_action = model.world_model.get_initial_state_action()
    else:
        # Weird RLlib Handling, this happens when env rests
        if not state_action or len(state_action[0]['deter'].size()) == 3:
            # Very hacky, but works on all envs
            state_action = model.world_model.get_initial_state_action()
        action, logp, state_action = model.policy(obs, state_action, explore)
        if policy.config["explore_noise"] > 0.0:
            action = td.Normal(action, policy.config["explore_noise"]).sample()
        action = torch.clamp(action, min=-1.0, max=1.0)

    policy.global_timestep += policy.config["action_repeat"]
    return action, logp, [state_action]


def dreamer_stats(policy, train_batch):
    return policy.stats_dict


def dreamer_optimizer_fn(policy, config):
    model = policy.model
    encoder_weights = list(model.world_model.encoder.parameters())
    decoder_weights = list(model.world_model.decoder.parameters())
    reward_weights = list(model.world_model.reward_predictor.parameters())
    dynamics_weights = list(model.world_model.dynamics.parameters())
    actor_weights = list(model.actor_critic.actor.parameters())
    critic_weights = list(model.actor_critic.critic.parameters())
    clip_wm = config["dreamer_model"]["model_opt"]["clip"]
    del config["dreamer_model"]["model_opt"]["clip"]
    clip_actor = config["dreamer_model"]["actor_opt"]["clip"]
    del config["dreamer_model"]["actor_opt"]["clip"]
    clip_critic = config["dreamer_model"]["critic_opt"]["clip"]
    del config["dreamer_model"]["critic_opt"]["clip"]
    model_opt = torch.optim.AdamW(
        [{"params": encoder_weights + decoder_weights + reward_weights + dynamics_weights, "clip": clip_wm}],
        **config["dreamer_model"]["model_opt"],
    )
    actor_opt = torch.optim.AdamW([{"params": actor_weights, "clip": clip_actor}], **config["dreamer_model"]["actor_opt"])
    critic_opt = torch.optim.AdamW([{"params": critic_weights, "clip": clip_critic}], **config["dreamer_model"]["critic_opt"])

    return (model_opt, actor_opt, critic_opt)


def preprocess_episode(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Batch format should be in the form of (s_t, a_(t-1), r_(t-1))
    When t=0, the resetted obs is paired with action and reward of 0.
    """
    obs = sample_batch[SampleBatch.OBS]
    new_obs = sample_batch[SampleBatch.NEXT_OBS]
    action = sample_batch[SampleBatch.ACTIONS]
    reward = sample_batch[SampleBatch.REWARDS]
    eps_ids = sample_batch[SampleBatch.EPS_ID]
    act_shape = action.shape
    act_reset = np.array([0.0] * act_shape[-1])[None]
    rew_reset = np.array(0.0)[None]

    batch_obs = {}
    for k in obs.keys():
        obs_end = np.array(new_obs[k][act_shape[0] - 1])[None]
        batch_obs[k] = np.concatenate([obs[k], obs_end], axis=0)
    batch_action = np.concatenate([act_reset, action], axis=0)
    batch_rew = np.concatenate([rew_reset, reward], axis=0)
    batch_eps_ids = np.concatenate([eps_ids, eps_ids[-1:]], axis=0)

    new_batch = {
        SampleBatch.OBS: batch_obs,
        SampleBatch.REWARDS: batch_rew,
        SampleBatch.ACTIONS: batch_action,
        SampleBatch.EPS_ID: batch_eps_ids,
    }
    return SampleBatch(new_batch)

def apply_grad_clipping(
    policy: "TorchPolicy", optimizer: LocalOptimizer, loss: TensorType
) -> Dict[str, TensorType]:
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Args:
        policy: The TorchPolicy, which calculated `loss`.
        optimizer: A local torch optimizer object.
        loss: The torch loss tensor.

    Returns:
        An info dict containing the "grad_norm" key and the resulting clipped
        gradients.
    """
    info = {}
    for param_group in optimizer.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
        clip = param_group['clip']
        if not clip:
            continue
        params = list(filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            grad_gnorm = nn.utils.clip_grad_norm_(
                params, clip
            )
            if isinstance(grad_gnorm, torch.Tensor):
                grad_gnorm = grad_gnorm.cpu().numpy()
            info["grad_gnorm"] = grad_gnorm
    return info

DreamerTorchPolicy = build_policy_class(
    name="DreamerTorchPolicy",
    framework="torch",
    get_default_config=lambda: dreamer2.dreamer.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    postprocess_fn=preprocess_episode,
    loss_fn=dreamer_loss,
    stats_fn=dreamer_stats,
    make_model=build_dreamer_model,
    optimizer_fn=dreamer_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping,
)
