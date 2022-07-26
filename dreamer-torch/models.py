import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import augmentations
import networks
import tools
from tools import FeatureTripletBuilder, distance

to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):

    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.augment = None
        self.triplet = None
        if self._config.contrastive:
            self._config.grad_heads = [h for h in self._config.grad_heads if h != 'image']
            print("disable image gradinent to the RSSM")
        if config.augment:
            self.augment = augmentations.Augmentation(config.augment_random_crop, config.augment_strong, config.augment_consistent,
                                                      config.augment_pad, config.size[0])
        self.encoder = networks.ConvEncoder(config.grayscale,
                                            config.cnn_depth, config.act, config.encoder_kernels)
        if self._config.contrastive == "triplet":
            self.triplet = True
            print("triplet activated")
        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = 2 ** (len(config.encoder_kernels) - 1) * config.cnn_depth
            embed_size *= 2 * 2
        else:
            raise NotImplemented(f"{config.size} is not applicable now")
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers,
            config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
            config.act, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
            config.num_actions, embed_size, config.device)
        self.heads = nn.ModuleDict()
        channels = (1 if config.grayscale else 3)
        shape = (channels,) + config.size
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads['image'] = networks.ConvDecoder(
            feat_size,  # pytorch version
            config.cnn_depth, config.act, shape, config.decoder_kernels,
            config.decoder_thin)
        self.heads['reward'] = networks.DenseHead(
            feat_size,  # pytorch version
            [], config.reward_layers, config.units, config.act)
        if config.pred_discount:
            self.heads['discount'] = networks.DenseHead(
                feat_size,  # pytorch version
                [], config.discount_layers, config.units, config.act, dist='binary')

        if config.contrastive == 'cpc' or config.contrastive == 'cpc_augment':
            self.heads["cpc"] = networks.DenseHead(
                32 * config.cnn_depth,  # pytorch version
                feat_size, config.cpc_layers, config.units, config.act)
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt,
            use_amp=self._use_amp)
        self._scales = dict(
            reward=config.reward_scale, discount=config.discount_scale, cpc_scale=config.cpc_scale)

    def _train(self, data):

        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):

                obs = data['image']
                if self.augment:
                    obs = self.augment(obs)

                embed = self.encoder({'image': obs})
                post, prior = self.dynamics.observe(embed, data['action'])
                kl_balance = tools.schedule(self._config.kl_balance, self._step)
                kl_free = tools.schedule(self._config.kl_free, self._step)
                kl_scale = tools.schedule(self._config.kl_scale, self._step)
                kl_loss, kl_value = self.dynamics.kl_loss(
                    post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = (name in self._config.grad_heads)
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()

                    if name == 'cpc' and (self._config.contrastive == 'cpc' or self._config.contrastive == 'cpc_augment'):
                        cpc_amount = (self._config.cpc_batch_amount, self._config.cpc_time_amount)
                        pred_aug = head(embed)
                        feat_aug = feat
                        likes[name] = self.compute_cpc_obj(pred_aug, feat_aug, cpc_amount)

                        if self._config.contrastive == 'cpc_augment':

                            assert self.augment is not None, 'Augmenting should be on'
                            #in that case we need to addd the original part without augmentation
                            #other than that we can use the previous stuff already
                            embed_noaug = self.encoder({'image': data['image']})
                            post_noaug, _ = self.dynamics.observe(embed_noaug, data['action'])
                            feat_noaug = self.dynamics.get_feat(post_noaug)
                            pred_noaug = head(embed_noaug)
                            likes[name] += self.compute_cpc_obj(pred_aug, feat_aug, cpc_amount)
                            likes[name] += self.compute_cpc_obj(pred_aug, feat_noaug, cpc_amount)
                            likes[name] += self.compute_cpc_obj(pred_noaug, feat_aug, cpc_amount)
                            likes[name] /= 4
                        losses[name] = -torch.mean(likes[name]) * self._scales.get(name, 1.0)

                    else:
                        pred = head(feat)
                        like = pred.log_prob(data[name])
                        likes[name] = like
                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

                if self.triplet:
                    feat = self.dynamics.get_feat(post)
                    embed2 = self.encoder({'image': self.augment(data['image'])})
                    post2, prior2 = self.dynamics.observe(embed2, data['action'])
                    feat2 = self.dynamics.get_feat(post2)
                    feat = feat.view(-1, feat.shape[2])
                    feat2 = feat2.view(-1, feat2.shape[2])
                    triplet_loss = self.compute_triplet_loss(feat, feat2)
                if self.triplet:
                    model_loss = sum(losses.values()) + kl_loss + triplet_loss
                else:
                    model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())
        if self.triplet:
            metrics.update({'triplet_loss': to_np(triplet_loss)})
        metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed, feat=self.dynamics.get_feat(post),
                kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
        if self._config.clip_rewards == 'tanh':
            obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
        elif self._config.clip_rewards == 'identity':
            obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
        else:
            raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
            obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data['image'][:6] + 0.5
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        recon = self.heads['image'](
            self.dynamics.get_feat(states)).mode()[:6]
        reward_post = self.heads['reward'](
            self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
        reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2

        return torch.cat([truth, model, error], 2)

    def compute_triplet_loss(self, feature1, feature2, loss_margin=2, negative_frame_margin=20):
        tripletBuilder = FeatureTripletBuilder(feature1, feature2, negative_frame_margin=negative_frame_margin)

        anchor_frames, positive_frames, negative_frames = tripletBuilder.build_set()

        d_positive = distance(anchor_frames, positive_frames)
        d_negative = distance(anchor_frames, negative_frames)
        loss_triplet = torch.clamp(loss_margin + d_positive - d_negative, min=0.0).mean()

        # print("TRIPLET LOSS: " + str(loss_triplet.item()))

        return loss_triplet

    def compute_cpc_obj(self, pred, features, cpc_amount=(10, 30), cpc_contrast='window'):

        batch_size, batch_len = features.size(0), features.size(1)
        if cpc_contrast == 'batch':
            ta = []
            for i in range(features.size(0)):
                ta.append(pred.log_prob(torch.roll(features, i, 0)))

            positive = pred.log_prob(features)
            negative = torch.logsumexp(torch.stack(ta), 0)
            return positive - negative

        if cpc_contrast == 'time':
            ta = []
            for i in range(features.size(0)):
                ta.append(pred.log_prob(torch.roll(features, i, 1)))

            positive = pred.log_prob(features)
            negative = torch.logsumexp(torch.stack(ta), 1)
            return positive - negative

        elif cpc_contrast == 'window':
            batch_amount, time_amount = cpc_amount[0], cpc_amount[1]
            assert batch_amount <= batch_size
            assert time_amount <= batch_len
            total_amount = batch_amount * time_amount
            ta = []

            def compute_negatives(index):
                batch_shift = index // time_amount
                time_shift = index % time_amount
                batch_shift -= batch_amount // 2
                time_shift -= time_amount // 2
                rolled = torch.roll(torch.roll(features, batch_shift, 0), time_shift, 1)
                return pred.log_prob(rolled)

            for i in range(total_amount):
                ta.append(compute_negatives(i))
            positive = pred.log_prob(features)
            negative = torch.logsumexp(torch.stack(ta), 0)
            return positive - negative


class ImagBehavior(nn.Module):

    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,  # pytorch version
            config.num_actions, config.actor_layers, config.units, config.act,
            config.actor_dist, config.actor_init_std, config.actor_min_std,
            config.actor_dist, config.actor_temp, config.actor_outscale)
        self.value = networks.DenseHead(
            feat_size,  # pytorch version
            [], config.value_layers, config.units, config.act,
            config.value_head)
        if config.slow_value_target or config.slow_actor_target:
            self._slow_value = networks.DenseHead(
                feat_size,  # pytorch version
                [], config.value_layers, config.units, config.act)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
            **kw)
        self._value_opt = tools.Optimizer(
            'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
            **kw)

    def _train(
            self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats)
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(
                    imag_state).entropy()
                target, weights = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
                    self._config.slow_actor_target)
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
                    weights)
                metrics.update(mets)
                if self._config.slow_value_target != self._config.slow_actor_target:
                    target, weights = self._compute_target(
                        imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
                        self._config.slow_value_target)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics['reward_mean'] = to_np(torch.mean(reward))
        metrics['reward_std'] = to_np(torch.std(reward))
        metrics['actor_ent'] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        feat = 0 * dynamics.get_feat(start)
        action = policy(feat).mode()
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, feat, action))
        states = {k: torch.cat([
            start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
            self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            slow):
        if 'discount' in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._world_model.heads['discount'](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        if slow:
            value = self._slow_value(imag_feat).mode()
        else:
            value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[:-1], value[:-1], discount[:-1],
            bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights

    def _compute_actor_loss(
            self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
            weights):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        target = torch.stack(target, dim=1)
        if self._config.imag_gradient == 'dynamics':
            actor_target = target
        elif self._config.imag_gradient == 'reinforce':
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
                    target - self.value(imag_feat[:-1]).mode()).detach()
        elif self._config.imag_gradient == 'both':
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
                    target - self.value(imag_feat[:-1]).mode()).detach()
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics['imag_gradient_mix'] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_target += self._config.actor_entropy() * actor_ent[:-1][:, :, None]
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            actor_target += self._config.actor_state_entropy() * state_ent[:-1]
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target or self._config.slow_actor_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
