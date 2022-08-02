from email import utils
import functools
from multiprocessing.sharedctypes import Value
import re
from sre_parse import State
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

from dreamer2.utils import SampleDist

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td
    from dreamer2.utils import (
        Linear,
        Conv2d,
        ConvTranspose2d,
        _GRUCell,
        TanhBijector,
        SafeTruncatedNormal,
        StreamNorm,
        FreezeParameters
    )
    import torch.nn.functional as F
    from torch import Tensor

ActFunc = Any

# Similar to GAE-Lambda, calculate value targets
def lambda_return(reward, value, weights, bootstrap, lambda_):
    def agg_fn(x, y):
        return y[0] + y[1] * lambda_ * x

    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + weights * next_values * (1 - lambda_)

    last = bootstrap
    returns = []
    for i in reversed(range(len(inputs))):
        last = agg_fn(last, [inputs[i], weights[i]])
        returns.append(last)

    returns = list(reversed(returns))
    returns = torch.stack(returns, dim=0)
    return returns

class NormLayer(nn.Module):
  def __init__(self, name):
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = nn.LayerNorm()
    else:
      raise NotImplementedError(name)
  def forward(self, features):
    if not self._layer:
        return features
    return self._layer(features)

class ConvEncoder(nn.Module):
    def __init__(self, channels: int, depth: int, act: ActFunc = None, kernels: Sequence[int]=(4, 4, 4, 4)):
        super(ConvEncoder, self).__init__()
        self._act = act or nn.ELU
        self._depth = depth
        self._kernels = kernels

        layers = []
        for i, kernel in enumerate(self._kernels):
            if i == 0:
                inp_dim = channels
            else:
                inp_dim = 2 ** (i - 1) * self._depth
            depth = 2 ** i * self._depth
            layers.append(Conv2d(inp_dim, depth, kernel, stride=2))
            layers.append(self._act())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        x = obs.view((-1,) + tuple(obs.shape[-3:]))
        x = self.layers(x)

        x = x.view([x.shape[0], np.prod(x.shape[1:])])
        shape = list(obs.shape[:-3]) + [x.shape[-1]]
        return x.view(shape)

    def compute_conv_output_shapes(self, image_size):
        with torch.no_grad():
            x = torch.rand((1, *image_size)).to(self.layers[0].weight.device)

            shapes = [x.shape[1:]]
            for l in self.layers:
                x = l(x)
                if not type(l) != self._act:
                    shapes.append(x.shape[1:])
        return shapes

class MLP(nn.Module):
    def __init__(self, layers: Sequence[int], act: ActFunc = None):
        super(MLP, self).__init__()
        act = act or nn.ELU
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(act())
        self.net = nn.Sequential(*modules)
    
    def forward(self, X):
        return self.net(X)

class MLPDistribution(nn.Module):
    def __init__(self, inp_size: int, out_size: int, layers: int, units: int, act: ActFunc = None, **dist_layer):
        super(MLPDistribution, self).__init__()
        act = act or nn.ELU
        mlp_layers = [inp_size] + [units] * layers
        modules = []
        for i in range(len(mlp_layers) - 1):
            modules.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            modules.append(act())
        self.net = nn.Sequential(*modules)
        self.dist_layer = DistLayer(inp_size=mlp_layers[-1], out_size=out_size, **dist_layer)
    def forward(self, features):
        
        return self.dist_layer(self.net(features))


class Encoder(nn.Module):
    def __init__(self, 
                 obs_shapes: Dict[str, List[int]],
                 act: ActFunc=None,
                 cnn_keys_pattern: str=r'.*', 
                 mlp_keys_pattern: str=r'.*',
                 cnn_depth: int=48,
                 cnn_kernels: Sequence[int]=(4, 4, 4, 4), 
                 mlp_layers: Sequence[int]=(400, 400, 400, 400)):
        super(Encoder, self).__init__()
        self._shapes = obs_shapes
        self.cnn_keys = [
            k for k, v in obs_shapes.items() if re.match(cnn_keys_pattern, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in obs_shapes.items() if re.match(mlp_keys_pattern, k) and len(v) == 1]
        self.embed_size = 0
        self.cnn_kernels = cnn_kernels
        self.mlp_layers = mlp_layers
        self.cnn_keys_pattern = cnn_keys_pattern
        self.mlp_keys_pattern = mlp_keys_pattern
        self.act = act or nn.ELU
        if self.cnn_keys:
            channels = sum([self._shapes[k][-3] for k in self.cnn_keys])
            self.cnn_encoder = ConvEncoder(channels, cnn_depth, self.act, cnn_kernels)
            cnn_shapes = [self._shapes[k] for k in self.cnn_keys]
            channels = sum([s[-3] for s in cnn_shapes])
            # Assuming height and width of all our multi-view image observations
            # is the same. Can later alter to have a different CNN and fuse
            height, width = self._shapes[self.cnn_keys[0]][1:3]
            self.cnn_output_shapes = self.cnn_encoder.compute_conv_output_shapes((channels, height, width))
            self.embed_size += np.prod(self.cnn_output_shapes[-1]) * len(self.cnn_keys)
        else:
            self.cnn_output_shapes = []
        if self.mlp_keys:
            state_dim = sum([self._shapes[k][-1] for k in self.mlp_keys])
            all_mlp_layers = [state_dim] + mlp_layers
            self.mlp_encoder = MLP(all_mlp_layers, self.act)
            self.embed_size += all_mlp_layers[-1] * len(self.mlp_keys)

    def forward(self, obs: Dict[str, Tensor]):
        outputs = []
        if self.cnn_keys:
            cnn_input = torch.cat([v for k, v in obs.items() if k in self.cnn_keys], dim=-1)
            outputs.append(self.cnn_encoder(cnn_input))
        if self.mlp_keys:
            mlp_input = torch.cat([v for k, v in obs.items() if k in self.mlp_keys], dim=-1) 
            outputs.append(self.mlp_encoder(mlp_input))
        out = torch.cat(outputs, dim=-1)
        return out

class TransposedConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        enc_conv_output_sizes: List[List[int]],
        enc_conv_kernels: Sequence[int],
        act: ActFunc=None,
    ):
        super(TransposedConvDecoder, self).__init__()
        self._act = act or nn.ELU
        self._linear_layer = nn.Linear(
            latent_dim, functools.reduce(lambda a, b: a * b, enc_conv_output_sizes[-1])
        )
        self._tconv_output_shapes = list(reversed(enc_conv_output_sizes))
        self._tconv_kernels = list(reversed(enc_conv_kernels))
        layers = []
        for i in range(len(self._tconv_output_shapes) - 1):
            layers.append(
                ConvTranspose2d(
                    self._tconv_output_shapes[i][0],
                    self._tconv_output_shapes[i + 1][0],
                    self._tconv_kernels[i],
                    2,
                )
            )
            if i != len(self._tconv_output_shapes) - 2:
                layers.append(self._act())
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        x = self._linear_layer(features)
        x = x.view([-1, *self._tconv_output_shapes[0]])

        output_idx = 1
        for l in self.layers:
            if isinstance(l, ConvTranspose2d):
                x = l(x, output_size=self._tconv_output_shapes[output_idx][1:])
                output_idx += 1
            else:
                x = l(x)
        mean = x.view(features.shape[:-1] + self._tconv_output_shapes[-1])
        return mean


class Decoder(nn.Module):
    def __init__(self, 
                 shapes: Dict[str, List[List[int]]],
                 latent_dim: int,
                 cnn_enc_shapes: List[List[int]],
                 enc_conv_kernels: Sequence[int],
                 mlp_hidden_layers: Sequence[int]=(400, 400, 400, 400, 400),
                 act: ActFunc = None,
                 cnn_keys: str=r'.*', 
                 mlp_keys: str=r'.*'):
        super(Decoder, self).__init__()
        self._shapes = shapes
        self._act = act
        self.cnn_keys = [
                k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        self.cnn_enc_shapes = cnn_enc_shapes
        
        if self.cnn_keys:
            self.cnn_decoder = TransposedConvDecoder(latent_dim, cnn_enc_shapes, enc_conv_kernels, act)
        if self.mlp_keys:
            state_dim = sum([self._shapes[k][-1] for k in self.mlp_keys])
            mlp_hidden_layers = [latent_dim] + list(mlp_hidden_layers) + [state_dim]
            self.mlp_decoder = MLP(mlp_hidden_layers, act)
    
    def forward(self, features):
        dists = {}
        if self.cnn_keys:
            cnn_outputs = [self._shapes[k][-1] for k in self.cnn_keys]
            cnn_out = self.cnn_decoder(features)
            means = torch.split(cnn_out, cnn_outputs, -1)
            dists.update({
                key: td.Independent(td.Normal(mean, 1), 3)
                for key, mean in zip(self.cnn_keys, means)})
        if self.mlp_keys:
            mlp_outputs = [self._shapes[k][0] for k in self.mlp_keys]
            mlp_out = self.mlp_decoder(features)
            means = torch.split(mlp_out, mlp_outputs, -1)
            dists.update({
                key: td.Independent(td.Normal(mean, 1), 1) # MSE
                for key, mean in zip(self.mlp_keys, means)})
        return dists



class DistLayer(nn.Module):
    def __init__(
        self,
        inp_size: int,
        out_size: int,
        act: ActFunc=None,
        dist: str="mse",
        init_std: float=0.0,
        min_std: float=0.1,
        action_disc: int=5,
        temp: float=0.1,
        outscale: int=0,
    ):
        super(DistLayer, self).__init__()
        self._dist = dist
        self._act = act or nn.ELU
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp
        self._outscale = outscale
        self._out_size = out_size

        if self._dist in ["tanh_normal", "normal", "trunc_normal", "tanh_normal_5"]:
            self._dist_layer = nn.Linear(inp_size, 2 * out_size)
        elif self._dist in ["mse", "onehot", "onehot_gumble"]:
            self._dist_layer = nn.Linear(inp_size, out_size)
        else:
            raise NotImplementedError(self._dist)

    def forward(self, features):
        x = features
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = SampleDist(td.Independent(dist, 1))
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = td.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = SampleDist(td.Independent(dist, 1))
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = td.Independent(dist, 1)
        elif self._dist == "mse":
            mean = self._dist_layer(x)
            dist = td.Normal(mean, 1.0)
            dist = td.Independent(dist, 1)
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._out_size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = td.Independent(dist, 1)
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = td.OneHotCategoricalStraightThrough(x)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = td.gumbel.Gumbel(x, 1 / temp)
        else:
            raise NotImplementedError(self._dist)
        return dist


# Represents TD model in PlaNET
class EnsembleRSSM(nn.Module):
    """RSSM is the core recurrent part of the PlaNET module. It consists of
    two networks, one (obs) to calculate posterior beliefs and states and
    the second (img) to calculate prior beliefs and states. The prior network
    takes in the previous state and action, while the posterior network takes
    in the previous state, action, and a latent embedding of the most recent
    observation.
    """

    def __init__(
        self,
        action_size: int,
        embed_size: int,
        ensemble: int = 5,
        discrete: int = 0,
        stoch: int = 30,
        deter: int = 200,
        hidden: int = 200,
        act: ActFunc = None,
        std_act: str = 'sigmoid2',
        min_std: float=0.1
    ):
        """Initializes RSSM

        Args:
            action_size (int): Action space size
            embed_size (int): Size of ConvEncoder embedding
            stoch (int): Size of the distributional hidden state
            deter (int): Size of the deterministic hidden state
            hidden (int): General size of hidden layers
            act (Any): Activation function
        """
        super().__init__()
        self._stoch = stoch
        self._discrete = discrete
        self._deter = deter
        if self._discrete:
            imag_input_dim = self._discrete * self._stoch + action_size
            self.state_dim = self._discrete * self._stoch + self._deter
        else:
            imag_input_dim = self._stoch + action_size
            self.state_dim = self._stoch + self._deter
        self.hidden_size = hidden
        self.act = act or nn.ELU
        self.obs_layer = Linear(embed_size + deter, hidden)
        if self._discrete > 0:
            self.obs_suff_stats = Linear(hidden, stoch * discrete)
        else:
            self.obs_suff_stats = Linear(hidden, 2 * stoch)

        self.cell = _GRUCell(inp_size=self.hidden_size, out_size=self._deter, norm=True)
        self.img_inp = Linear(imag_input_dim, hidden)

        img_suff_stats_ensemble = []
        for i in range(ensemble):
            layer_1 = Linear(deter, hidden)
            if self._discrete > 0:
                layer_2 = Linear(hidden, stoch * discrete)
            else:
                layer_2 = Linear(hidden, 2 * stoch)
            img_suff_stats_ensemble.append(nn.Sequential(
                layer_1,
                self.act(),
                layer_2
            ))
        self.img_suff_stats_ensemble = nn.ModuleList(img_suff_stats_ensemble)

        self.std_act = std_act
        self.min_std = min_std

    def get_initial_state(self, batch_size: int) -> Dict[str, TensorType]:
        """Returns the inital state for the RSSM, which consists of mean,
        std for the stochastic state, the sampled stochastic hidden state
        (from mean, std), and the deterministic hidden state, which is
        pushed through the GRUCell.

        Args:
            batch_size (int): Batch size for initial state

        Returns:
            List of tensors
        """
        device = next(self.parameters()).device
        if self._discrete:
            state = {
                'logit': torch.zeros(batch_size, self._stoch, self._discrete).to(device),
                'stoch': torch.zeros(batch_size, self._stoch, self._discrete).to(device),
                'deter': torch.zeros(batch_size, self._deter).to(device)
            }
        else:
            state = {
                'mean': torch.zeros(batch_size, self._stoch).to(device),
                'std': torch.zeros(batch_size, self._stoch).to(device),
                'stoch': torch.zeros(batch_size, self._stoch).to(device),
                'deter': torch.zeros(batch_size, self._deter).to(device)
            }
        return state

    def observe(
        self, embed: TensorType, action: TensorType, state: List[TensorType] = None, is_first: TensorType
    ) -> Tuple[Dict[str, TensorType], Dict[str, TensorType]]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the RNN from the
        starting state through each index of embed and action, saving all
        intermediate states between.

        Args:
            embed (TensorType): ConvEncoder embedding
            action (TensorType): Actions
            state (List[TensorType]): Initial state before rollout

        Returns:
            Posterior states and prior states (both List[TensorType])
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.get_initial_state(action.size()[0])

        if embed.dim() <= 2:
            embed = torch.unsqueeze(embed, 1)

        if action.dim() <= 2:
            action = torch.unsqueeze(action, 1)

        embed, action, is_first  = swap(embed), swap(action), swap(is_first) # T x B x enc_dim, T x B x action_dim

        posts = {k: [] for k in state.keys()}
        priors = {k: [] for k in state.keys()}
        last_post, last_prior = (state, state)
        for index in range(len(action)):
            # Tuple of post and prior
            last_post, last_prior = self.obs_step(last_post, action[index], embed[index], is_first)
            for k, v in last_post.items():
                posts[k].append(v)
            for k, v in last_prior.items():
                priors[k].append(v)

        post = {k: swap(torch.stack(v, dim=0)) for k, v in posts.items()}
        prior = {k: swap(torch.stack(v, dim=0)) for k, v in priors.items()}

        return post, prior

    def imagine(
        self, action: TensorType, state: List[TensorType] = None
    ) -> List[TensorType]:
        """Imagines the trajectory starting from state through a list of actions.
        Similar to observe(), requires rolling out the RNN for each timestep.

        Args:
            action (TensorType): Actions
            state (List[TensorType]): Starting state before rollout

        Returns:
            Prior states
        """
        if state is None:
            state = self.get_initial_state(action.size()[0])
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))

        action = swap(action)

        indices = range(len(action))
        priors = {k: [] for k in state.keys()}
        last = state
        for index in indices:
            last = self.img_step(last, action[index])
            for k, v in last.items():
                priors[k].append(v)

        prior = {k: swap(torch.stack(v, dim=0)) for k, v in priors.items()}
        return prior

    def obs_step(
        self, prev_state: Dict[str, TensorType], prev_action: TensorType, embed: TensorType, is_first: TensorType, sample=True
    ) -> Tuple[List[TensorType], List[TensorType]]:
        """Runs through the posterior model and returns the posterior state

        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action
            embed (TensorType): Embedding from ConvEncoder

        Returns:
            Post and Prior state
        """
        for k, v in prev_state.items():
            prev_state[k] = torch.einsum('b,b...->b...', 1.0 - is_first.type(v.dtype), v)
        prev_action = torch.einsum('b,b...->b...', 1.0 - is_first.type(prev_action.dtype), prev_action)
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.obs_layer(x)
        x = self.act()(x)
        stats = self._get_suff_stats(x, self.obs_suff_stats)
        dist = self._get_dist(stats)
        stoch = dist.rsample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def _get_suff_stats(self, x, suff_stats_module):
        x = suff_stats_module(x)
        if self._discrete > 0:
            stats = {'logit': x.view(x.shape[:-1] + (self._stoch, self._discrete))}
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = {
                'softplus': lambda: F.Softplus()(std),
                'sigmoid': lambda: torch.sigmoid(std),
                'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
            }[self.std_act]()
            std = std + self.min_std
            stats = {'mean': mean, 'std': std}
        return stats


    def img_step(
        self, prev_state: Dict[str, TensorType], prev_action: TensorType, sample=True
    ) -> List[TensorType]:
        """Runs through the prior model and returns the prior state

        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action

        Returns:
            Prior state
        """
        prev_stoch = prev_state['stoch']
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.view(shape)
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.img_inp(x)
        x = self.act()(x)
        deter = self.cell(x, prev_state['deter'])
        x = deter
        stats = self._get_suff_stats_ensemble(x)
        index =  torch.randint(len(self.img_suff_stats_ensemble)) if len(self.img_suff_stats_ensemble) > 1 else 0
        # Pick one randomly
        stats = {k: v[index] for k, v in stats.items()}
        dist = self._get_dist(stats)
        stoch = dist.rsample() if sample else dist.mode()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def get_feature(self, state: Dict[str, TensorType]) -> TensorType:
        # Constructs feature for input to reward, decoder, actor, critic
        stoch = state['stoch']
        if self._discrete:
            shape = stoch.shape[:-2] + (self._stoch * self._discrete,)
            stoch = stoch.view(shape)
        return torch.concat([stoch, state['deter']], -1)

    def _get_suff_stats_ensemble(self, inp):
        # TODO: Optimize with torch vmap
        stats = [self._get_suff_stats(inp, m) for m in self.img_suff_stats_ensemble]
        # We are relying on the fact that stats[0].keys() is ordered the same always
        # True in python 3.8 and above I guess
        stats = {k: torch.stack([x[k] for x in stats]) for k in stats[0].keys()}
        return stats

    def _get_dist(self, state: Dict[str, TensorType], ensemble=False) -> TensorType:
        if ensemble:
            state = self._get_suff_stats_ensemble(state['deter'])
        if self._discrete:
            logit = state['logit']
            dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        else:
            mean, std = state['mean'], state['std']
            dist = td.Independent(td.Normal(mean, std), 1)
        return dist


class DreamerWorldModel(nn.Module):
    def __init__(self, obs_space, action_space, model_config):
        super().__init__()
        self.action_size = action_space.shape[0]
        obs_shapes = {k: tuple(v.shape) for k, v in obs_space.items() if k != 'reward' or k != 'discount'}
        self.encoder = Encoder(obs_shapes, **model_config['encoder'])
        self.dynamics = EnsembleRSSM(
            self.action_size,
            self.encoder.embed_size,
            **model_config["rssm"]
        )
        self.decoder = Decoder(obs_shapes, self.dynamics.state_dim, self.encoder.cnn_output_shapes, self.encoder.cnn_kernels, self.encoder.mlp_layers)

        self.reward_predictor = MLPDistribution(self.dynamics.state_dim, 1, **model_config["reward_head"])
        self.discount_predictor = None
        if model_config['predict_discount']:
            self.discount_predictor = MLPDistribution(self.dynamics.state_dim, 1, **model_config["discount_head"])
        self.kl = model_config['kl']
        self.loss_scale = model_config['loss_scale']
        self.clip_rewards = model_config['clip_rewards']


    def imagine_ahead(self,  policy: Callable[[TensorType], 'td.Distribution'], start_state: Dict[str, TensorType], is_terminal: TensorType, horizon: int) -> TensorType:
        """Given a batch of states, rolls out more state of length horizon."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_state.items()}
        start['feat'] = self.dynamics.get_feature(start)
        start['action'] = torch.zeros_like(policy(start['feat']).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq['feat'][-1].detach()).rsample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.itemss()}, action)
            feature = self.dynamics.get_feature(state)
            for key, value in {**state, 'action': action, 'feat': feature}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        
        if self.discount_predictor:
            disc = self.discount_predictor(seq['feat']).mean()
            if is_terminal is not None:
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = torch.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.discount * torch.ones(seq['feat'].shape[:-1])
        seq['discount'] = disc
        seq['weight'] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq

    def get_initial_state_action(self) -> Tuple[Dict[str, TensorType], TensorType]:
        state = self.dynamics.get_initial_state(1)
        action = torch.zeros(1, self.action_size).to(state['deter'].device)
        return state, action
    
    def encode_latent(self, previous_state: Dict[str, TensorType], action: TensorType, obs: Dict[str, TensorType], explore: bool) -> Tuple[Dict[str, TensorType], TensorType]:
        embed = self.encoder(obs)
        post, _ = self.dynamics.obs_step(previous_state, action, embed, obs['is_first'], sample=explore)
        feature = self.dynamics.get_feature(post)
        return post, feature
    
    def compute_states_and_losses(self, obs: Dict[str, TensorType], actions: TensorType, log_gif: bool = False):
        latent = self.encoder(obs)
        post, prior = self.dynamics.observe(latent, actions)
        features = self.dynamics.get_feature(post)
        losses = {}
        obs_pred = self.decoder(features)
        for k, v in obs_pred.items():
            losses[k] = -v.log_prob(obs[k]).mean()
        reward_pred = self.reward_predictor(features)
        rewards = {
            "identity": lambda x: x,
            "tanh": lambda x: torch.tanh(x),
            "sign": lambda x: torch.sign(x)
        }[self.clip_rewards](rewards)
        losses['reward'] =  -reward_pred.log_prob(obs['reward'].unsqueeze(-1)).mean()

        if self.discount_predictor:
            discount_pred = self.discount_predictor(features)
            losses['discount'] = -discount_pred.log_prob(obs['discount'].unsqueeze(-1)).mean()
        
        
        div, prior_dist_detached, post_dist_detached = self.kl_loss(post, prior, self.kl['balance'], self.kl['free'])
        losses['kl'] = div
        model_loss =  sum(self.loss_scale.get(k, 1.0) * v for k, v in losses.items())
        losses['model_loss'] = model_loss

        return_dict = losses
        return_dict['prior_ent'] = prior_dist_detached.entropy()
        return_dict['post_ent'] = post_dist_detached.entropy()
        if log_gif:
            return_dict["log_gif"] = self.video_predict(obs, actions, latent, obs_pred, log_key='image')
        return post, return_dict
    
    def kl_loss(self, post, prior, balance, free):
        prior_dist = self.dynamics._get_dist(prior)
        prior_dist_detached = self.dynamics._get_dist({k: v.detach() for k, v in prior.items()})
        post_dist = self.dynamics._get_dist(post)
        post_dist_detached = self.dynamics._get_dist({k: v.detach() for k, v in post.items()})
        kl_lhs = torch.mean(
            torch.distributions.kl_divergence(post_dist_detached, prior_dist))
        kl_rhs = torch.mean(
            torch.distributions.kl_divergence(post_dist, prior_dist_detached))
        div = balance * torch.maximum(kl_lhs, Tensor([free])[0])  + (1- balance) *torch.maximum(kl_rhs, Tensor([free])[0])
        return div, prior_dist_detached, post_dist_detached

    def log_summary(self, obs, action, embed, image_pred, log_key):
        truth = obs[:6] + 0.5
        recon = image_pred[log_key].mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], action[:6, :5], None, obs['is_first'][:6, :5])
        init = [itm[:, -1] for itm in init]
        prior = self.dynamics.imagine(action[:6, 5:], init)
        openl = self.decoder(self.dynamics.get_feature(prior))[log_key].mean

        mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (mod - truth + 1.0) / 2.0
        return torch.cat([truth, mod, error], 3)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, model_config):
        super().__init__()
        self.action_size = action_space.shape[0]
        discrete = hasattr(action_space, 'n')
        if model_config['actor']['dist'] == 'auto':
            model_config['actor']['dist'] = 'onehot' if discrete else 'trunc_normal'
            print(f"Setting actor output to {model_config['actor']['dist']}")

        if model_config['actor_grad'] == 'auto':
            model_config['actor_grad'] = 'reinforce' if discrete else 'dynamics'
            print(f"Setting gradient to {model_config['actor_grad']}")
        self.actor_grad = model_config['actor_grad']
        self.actor = MLPDistribution(state_dim, self.action_size, **model_config['actor'])
        self.critic = MLPDistribution(
            state_dim, 1, **model_config['critic']
        )
        self.slow_target = model_config['slow_target']
        self.slow_target_update = model_config['slow_target_update']
        self.slow_target_fraction = model_config['slow_target_fraction']
        if self.slow_target:
            self.target_critic =  MLPDistribution(
                state_dim, 1, **model_config['critic']
            )
            for p in self.target_critic.parameters():
                p.requires_grad = False
            self.target_updates = nn.Parameter(torch.zeros(()), requires_grad=False)
        else:
            self.target_critic = self.critic
        self.imagine_horizon = model_config["imagine_horizon"]
        self.discount_factor = model_config["discount"]
        self.discount_lambda = model_config["discount_lambda"]
        self.actor_ent = model_config["actor_ent"]
        self.reward_norm = StreamNorm(**model_config["reward_norm"])
    
    def compute_losses(self, world_model: DreamerWorldModel, start_state: Dict[str, TensorType], is_terminal: TensorType):
        metrics = {}
        hor = self.imagine_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with torch.no_grad():
            start_state = {k: v.detach() for k, v in start_state.items()}
        with FreezeParameters(list(world_model.parameters())):
            seq = world_model.imagine_ahead(self.actor, start_state, is_terminal, self.imagine_horizon)
            reward = world_model.reward_predictor(seq["feat"]).mean
            reward, mets1 = self.reward_norm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
        critic_loss, mets4 = self.critic_loss(seq, target)
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        }
    
    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(seq['feat'][:-2].detach())
        if self.actor_grad == 'dynamics':
            actor_target = target[1:]
        elif self.actor_grad == 'reinforce':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            action = (seq['action'][1:-1]).detach()
            objective = policy.log_prob(action) * advantage
        elif self.actor_grad == 'both':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            objective = policy.log_prob(seq['action'][1:-1]) * advantage
            mix = self.actor_grad_mix
            objective = mix * target[1:] + (1 - mix) * objective
        else:
            raise NotImplementedError(self.actor_grad)
        ent = policy.entropy()
        ent_scale = self.actor_ent
        objective += ent_scale * ent
        weight = seq['weight'].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['actor_ent'] = ent.mean()
        metrics['actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq['feat'][:-1])
        target = target.detach()
        weight = seq['weight'].detach()
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics


    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = seq['reward']
        disc = seq['discount']
        value = self.target_critic(seq['feat']).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.discount_lambda)
        metrics = {}
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics
        
    def update_slow_target(self):
        if self.slow_target:
            if self.target_updates % self.slow_target_update == 0:
                mix = 1.0 if self.target_updates == 0 else self.slow_target_fraction
                for s, d in zip(self.critic.parameters(), self.target_critic.parameters()):
                    d.data = (mix * s.data + (1 - mix) * d.data)
            self.target_updates += 1

# Represents all models in Dreamer, unifies them all into a single interface
class DreamerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)
        self.world_model = DreamerWorldModel(obs_space, action_space, model_config)
        self.actor_critic = ActorCritic(self.world_model.dynamics.state_dim, action_space, model_config)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def policy(
        self, obs: TensorType, state_action: Tuple[Dict[str, TensorType], TensorType], explore: bool
    ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, recurrent model,
        and policy to obtain action.
        """
        # if state_action is None:
        #     self.state_action = self.world_model.get_initial_state_action(batch_size=obs.shape[0])
        # else:
        #     self.state_action = state_action
        next_state, next_state_feat = self.world_model.encode_latent(self.state_action[0], self.state_action[1], obs, explore)

        action_dist = self.actor_critic.actor(next_state_feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)

        self.state_action = next_state, action
        return action, logp, [self.state_action]

    def value_function(self) -> TensorType:
        return None
