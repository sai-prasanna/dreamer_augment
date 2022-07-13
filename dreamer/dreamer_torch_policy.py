import logging
from typing import Dict, Optional

import numpy as np
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.utils.typing import AgentID

import dreamer
from dreamer.dreamer_model import DreamerModel
from dreamer.utils import BarlowTwins, FeatureTripletBuilder, distance, compute_cpc_loss
from dreamer.utils import FreezeParameters

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td

logger = logging.getLogger(__name__)


# This is the computation graph for workers (inner adaptation steps)
def compute_dreamer_loss(
        obs,
        action,
        reward,
        model: DreamerModel,
        imagine_horizon: int,
        discount=0.99,
        lambda_=0.95,
        kl_coeff=1.0,
        kl_balance=0.8,
        cpc_batch_amount=10,
        cpc_time_amount=30,
        log=False,
):
    """Constructs loss for the Dreamer objective

    Args:
        obs (TensorType): Observations (o_t)
        action (TensorType): Actions (a_(t-1))
        reward (TensorType): Rewards (r_(t-1))
        model (TorchModelV2): DreamerModel, encompassing all other models
        imagine_horizon (int): Imagine horizon for actor and critic loss
        discount (float): Discount
        lambda_ (float): Lambda, like in GAE
        kl_coeff (float): KL Coefficient for Divergence loss in model loss
        kl_balance (float): ratio to balance between the rhs and lhs in kl div.
        cpc_batch_amount (int): window of batches to look for negative samples for cpc loss
        cpc_time_amount (int): window of times to look for negative samples for cpc loss
        log (bool): If log, generate gifs
    """
    encoder_weights = list(model.encoder.parameters())
    decoder_weights = list(model.decoder.parameters())
    reward_weights = list(model.reward.parameters())
    dynamics_weights = list(model.dynamics.parameters())
    critic_weights = list(model.value.parameters())
    model_weights = list(
        encoder_weights + decoder_weights + reward_weights + dynamics_weights
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpc_amount = (cpc_batch_amount, cpc_time_amount)

    # PlaNET Model Loss
    if model.augment:
        obs_aug = model.augment(obs).contiguous()
        if model.augmented_target:
            obs_target = obs_aug
        else:
            obs_target = obs
    else:
        obs_aug = obs
        obs_target = obs

    latent = model.encoder(obs_aug)
    post, prior = model.dynamics.observe(latent, action)
    features = model.dynamics.get_feature(post)
    # Compute tripet loss
    if model.contrastive_loss == "triplet":
        assert model.augment
        obs_aug_2 = model.augment(obs).contiguous()
        latent_2 = model.encoder(obs_aug_2)
        post_2, _ = model.dynamics.observe(latent_2, action)
        features_2 = model.dynamics.get_feature(post_2)
        contrastive_loss = compute_triplet_loss(features, features_2)
    # Compute barlow twins loss
    elif model.contrastive_loss == "barlow_twins":
        assert model.augment
        obs_aug_2 = model.augment(obs).contiguous()
        latent_2 = model.encoder(obs_aug_2)
        post_2, _ = model.dynamics.observe(latent_2, action)
        features_2 = model.dynamics.get_feature(post_2)
        contrastive_loss = compute_barlow_twins_loss(features, features_2)

    elif model.contrastive_loss == 'cpc':

        state_preds = model.state_model(model.encoder(obs))
        contrastive_loss = compute_cpc_loss(state_preds, features, cpc_amount)
        contrastive_loss = contrastive_loss.mean()

    elif model.contrastive_loss == "cpc_augment":
        assert model.augment
        state_preds = model.state_model(latent)

        latent_noaug = model.encoder(obs)
        state_preds_noaug = model.state_model(latent_noaug)
        post_noaug, _ = model.dynamics.observe(latent_noaug, action)
        features_noaug = model.dynamics.get_feature(post_noaug)

        contrastive_loss = compute_cpc_loss(state_preds_noaug, features, cpc_amount)
        contrastive_loss += compute_cpc_loss(state_preds, features, cpc_amount)
        contrastive_loss += compute_cpc_loss(state_preds_noaug, features_noaug, cpc_amount)
        contrastive_loss += compute_cpc_loss(state_preds, features_noaug, cpc_amount)
        contrastive_loss = contrastive_loss.mean()

    # Don't use decoder to train the state representations when using contrastive
    image_pred = model.decoder(features.detach() if model.contrastive_loss else features)
    reward_pred = model.reward(features)
    image_loss = -torch.mean(image_pred.log_prob(obs_target))
    reward_loss = -torch.mean(reward_pred.log_prob(reward))
    # prior_dist = model.dynamics.get_dist(prior[0], prior[1])
    # post_dist = model.dynamics.get_dist(post[0], post[1])
    prior_dist = model.dynamics.get_dist(prior[0], prior[1])
    prior_dist_detached = model.dynamics.get_dist(prior[0].detach(), prior[1].detach())
    post_dist = model.dynamics.get_dist(post[0], post[1])
    post_dist_detached = model.dynamics.get_dist(post[0].detach(), post[1].detach())
    kl_lhs = torch.mean(
        torch.distributions.kl_divergence(post_dist_detached, prior_dist))  # .sum(dim=2))
    kl_rhs = torch.mean(
        torch.distributions.kl_divergence(post_dist, prior_dist_detached))  # .sum(dim=2))
    div = kl_balance * kl_lhs + (1 - kl_balance) * kl_rhs
    div = torch.mean(
        torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=2)
    )
    if model.contrastive_loss:
        model_loss = kl_coeff * div + reward_loss + image_loss + contrastive_loss
    else:
        model_loss = kl_coeff * div + reward_loss + image_loss

    # Actor Loss
    # [imagine_horizon, batch_length*batch_size, feature_size]
    with torch.no_grad():
        actor_states = [v.detach() for v in post]
    with FreezeParameters(model_weights):
        imag_feat = model.imagine_ahead(actor_states, imagine_horizon)
    with FreezeParameters(model_weights + critic_weights):
        reward = model.reward(imag_feat).mean
        value = model.value(imag_feat).mean
    pcont = discount * torch.ones_like(reward)
    returns = lambda_return(reward[:-1], value[:-1], pcont[:-1], value[-1], lambda_)
    discount_shape = pcont[:1].size()
    discount = torch.cumprod(
        torch.cat([torch.ones(*discount_shape).to(device), pcont[:-2]], dim=0), dim=0
    )
    actor_loss = -torch.mean(discount * returns)

    # Critic Loss
    with torch.no_grad():
        val_feat = imag_feat.detach()[:-1]
        target = returns.detach()
        val_discount = discount.detach()
    val_pred = model.value(val_feat)
    critic_loss = -torch.mean(val_discount * val_pred.log_prob(target))

    # Logging purposes
    prior_ent = torch.mean(prior_dist.entropy())
    post_ent = torch.mean(post_dist.entropy())

    log_gif = None
    if log:
        log_gif = log_summary(obs, action, latent, image_pred, model)

    return_dict = {
        "model_loss": model_loss,
        "reward_loss": reward_loss,
        "image_loss": image_loss,
        "divergence": div,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "prior_ent": prior_ent,
        "post_ent": post_ent,
    }
    if model.contrastive_loss:
        return_dict["contrastive_loss"] = contrastive_loss

    if log_gif is not None:
        return_dict["log_gif"] = log_gif
    return return_dict


def compute_triplet_loss(feature1, feature2, loss_margin=2, negative_frame_margin=10):
    tripletBuilder = FeatureTripletBuilder(feature1, feature2, negative_frame_margin=negative_frame_margin)

    anchor_frames, positive_frames, negative_frames = tripletBuilder.build_set()

    d_positive = distance(anchor_frames, positive_frames)
    d_negative = distance(anchor_frames, negative_frames)
    loss_triplet = torch.clamp(loss_margin + d_positive - d_negative, min=0.0).mean()

    return loss_triplet


def compute_barlow_twins_loss(feature1, feature2, lambd=0.0051):
    feature1 = feature1.view(-1, feature1.shape[2])
    feature2 = feature2.view(-1, feature2.shape[2])
    bt = BarlowTwins(feature1.shape[0], feature1.shape[1], lambd).cuda()
    loss = bt.forward(feature1, feature2)
    return loss


# Similar to GAE-Lambda, calculate value targets
def lambda_return(reward, value, pcont, bootstrap, lambda_):
    def agg_fn(x, y):
        return y[0] + y[1] * lambda_ * x

    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + pcont * next_values * (1 - lambda_)

    last = bootstrap
    returns = []
    for i in reversed(range(len(inputs))):
        last = agg_fn(last, [inputs[i], pcont[i]])
        returns.append(last)

    returns = list(reversed(returns))
    returns = torch.stack(returns, dim=0)
    return returns


# Creates gif
def log_summary(obs, action, embed, image_pred, model):
    truth = obs[:6] + 0.5
    recon = image_pred.mean[:6]
    init, _ = model.dynamics.observe(embed[:6, :5], action[:6, :5])
    init = [itm[:, -1] for itm in init]
    prior = model.dynamics.imagine(action[:6, 5:], init)
    openl = model.decoder(model.dynamics.get_feature(prior)).mean

    mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (mod - truth + 1.0) / 2.0
    return torch.cat([truth, mod, error], 3)


def dreamer_loss(policy, model, dist_class, train_batch):
    log_gif = False
    if "log_gif" in train_batch:
        log_gif = True

    policy.stats_dict = compute_dreamer_loss(
        train_batch["obs"],
        train_batch["actions"],
        train_batch["rewards"],
        policy.model,
        policy.config["imagine_horizon"],
        policy.config["discount"],
        policy.config["lambda"],
        policy.config["kl_coeff"],
        policy.config["kl_balance"],
        policy.config["cpc_batch_amount"],
        policy.config["cpc_time_amount"],
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


def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
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
        state = model.get_initial_state()
    else:
        # Weird RLlib Handling, this happens when env rests
        if len(state[0].size()) == 3:
            # Very hacky, but works on all envs
            state = model.get_initial_state()
        action, logp, state = model.policy(obs, state, explore)
        if policy.config["explore_noise"] > 0.0:
            action = td.Normal(action, policy.config["explore_noise"]).sample()
        action = torch.clamp(action, min=-1.0, max=1.0)

    policy.global_timestep += policy.config["action_repeat"]

    return action, logp, state


def dreamer_stats(policy, train_batch):
    return policy.stats_dict


def dreamer_optimizer_fn(policy, config):
    model = policy.model
    encoder_weights = list(model.encoder.parameters())
    decoder_weights = list(model.decoder.parameters())
    reward_weights = list(model.reward.parameters())
    dynamics_weights = list(model.dynamics.parameters())
    actor_weights = list(model.actor.parameters())
    critic_weights = list(model.value.parameters())
    model_opt = torch.optim.Adam(
        encoder_weights + decoder_weights + reward_weights + dynamics_weights,
        lr=config["td_model_lr"],
    )
    actor_opt = torch.optim.Adam(actor_weights, lr=config["actor_lr"])
    critic_opt = torch.optim.Adam(critic_weights, lr=config["critic_lr"])

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
    obs_end = np.array(new_obs[act_shape[0] - 1])[None]

    batch_obs = np.concatenate([obs, obs_end], axis=0)
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


DreamerTorchPolicy = build_policy_class(
    name="DreamerTorchPolicy",
    framework="torch",
    get_default_config=lambda: dreamer.dreamer.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    postprocess_fn=preprocess_episode,
    loss_fn=dreamer_loss,
    stats_fn=dreamer_stats,
    make_model=build_dreamer_model,
    optimizer_fn=dreamer_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping,
)
