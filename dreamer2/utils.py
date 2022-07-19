import numpy as np
from typing import Any

from gym.spaces import Dict as DictSpace
from gym import Env, ObservationWrapper
from ray.rllib.utils.framework import try_import_torch

ActFunc = Any

torch, nn = try_import_torch()

# Custom initialization for different types of layers
if torch:
    import torch.nn.functional as F
    import torch.distributions as td

    class Linear(nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    class Conv2d(nn.Conv2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    class ConvTranspose2d(nn.ConvTranspose2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    class _GRUCell(nn.Module):
        def __init__(self, inp_size: int, out_size: int, norm: bool=False, act: ActFunc=None, update_bias: int=-1):
            super(_GRUCell, self).__init__()
            self._inp_size = inp_size
            self._out_size = out_size
            self._act = act or torch.tanh
            self._norm = norm
            self._update_bias = update_bias
            self._layer = nn.Linear(inp_size + out_size, 3 * out_size, bias=norm)
            if norm:
                self._norm = nn.LayerNorm(3 * out_size)

        @property
        def state_size(self):
            return self._out_size

        def forward(self, inputs, state):
            parts = self._layer(torch.cat([inputs, state], -1))
            if self._norm:
                parts = self._norm(parts)
            reset, cand, update = torch.split(parts, [self._out_size] * 3, -1)
            reset = torch.sigmoid(reset)
            cand = self._act(reset * cand)
            update = torch.sigmoid(update + self._update_bias)
            output = update * cand + (1 - update) * state
            return output

    # Custom Tanh Bijector due to big gradients through Dreamer Actor
    class TanhBijector(td.Transform):
        def __init__(self):
            super().__init__()

            self.bijective = True
            self.domain = torch.distributions.constraints.real
            self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

        def atanh(self, x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        def sign(self):
            return 1.0

        def _call(self, x):
            return torch.tanh(x)

        def _inverse(self, y):
            y = torch.where(
                (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
            )
            y = self.atanh(y)
            return y

        def log_abs_det_jacobian(self, x, y):
            return 2.0 * (np.log(2) - x - nn.functional.softplus(-2.0 * x))

    class SafeTruncatedNormal(td.Normal):
        def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
            super().__init__(loc, scale)
            self._low = low
            self._high = high
            self._clip = clip
            self._mult = mult

        def sample(self, sample_shape):
            event = super().sample(sample_shape)
            if self._clip:
                clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
                event = event - event.detach() + clipped.detach()
            if self._mult:
                event *= self._mult
            return event
    
    
    class OneHotDist(td.OneHotCategoricalStraightThrough):
        def __init__(self, logits=None, probs=None):
            super().__init__(logits=logits, probs=probs)

        def mode(self):
            _mode = F.one_hot(
                torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
            )
            return _mode.detach() + super().logits - super().logits.detach()

        def sample(self, sample_shape=(), seed=None):
            if seed is not None:
                raise ValueError("need to check")
            sample = super().sample(sample_shape)
            probs = super().probs
            while len(probs.shape) < len(sample.shape):
                probs = probs[None]
            sample += probs - probs.detach()
            return sample


    class SampleDist:
        def __init__(self, dist, samples=100):
            self._dist = dist
            self._samples = (samples, )

        @property
        def name(self):
            return "SampleDist"

        def __getattr__(self, name):
            return getattr(self._dist, name)

        def mean(self):
            samples = self._dist.sample(self._samples)
            return torch.mean(samples, 0)

        def mode(self):
            sample = self._dist.sample(self._samples)
            logprob = self._dist.log_prob(sample)
            return sample[torch.argmax(logprob)][0]

        def entropy(self):
            sample = self._dist.sample(self._samples)
            logprob = self.log_prob(sample)
            return -torch.mean(logprob, 0)

    class StreamNorm(nn.Module):

        def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
            # Momentum of 0 normalizes only based on the current batch.
            # Momentum of 1 disables normalization.
            super().__init__()
            self._shape = tuple(shape)
            self._momentum = momentum
            self._scale = scale
            self._eps = eps
            self.mag = nn.Parameter(torch.ones(shape, dtype=torch.float64), requires_grad=False)

        def forward(self, inputs):
            metrics = {}
            self.update(inputs)
            metrics['mean'] = inputs.mean()
            metrics['std'] = inputs.std()
            outputs = self.transform(inputs)
            metrics['normed_mean'] = outputs.mean()
            metrics['normed_std'] = outputs.std()
            return outputs, metrics

        def reset(self):
            self.mag.data = torch.ones_like(self.mag)

        def update(self, inputs):
            batch = inputs.reshape((-1,) + self._shape)
            mag = torch.abs(batch).mean(0).type(torch.float64)
            self.mag.data = self._momentum * self.mag + (1 - self._momentum) * mag

        def transform(self, inputs):
            values = inputs.reshape((-1,) + self._shape)
            values /= self.mag.data.type(inputs.dtype)[None] + self._eps
            values *= self._scale
            return values.reshape(inputs.shape)

# Modified from https://github.com/juliusfrost/dreamer-pytorch
class FreezeParameters:
    def __init__(self, parameters):
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]


def create_env(env_creator, wrap_obs_key: str, **kwargs):
    env = env_creator(**kwargs)
    if isinstance(env.observation_space, DictSpace):
        return env
    else:
        wrapped = DictWrapper(wrap_obs_key, env)
        return wrapped

class DictWrapper(ObservationWrapper):
    def __init__(self, wrap_obs_key: str, env: Env) -> None:
        super().__init__(env)
        self.wrap_obs_key = wrap_obs_key
        self.observation_space = DictSpace({self.wrap_obs_key: env.observation_space})
    def observation(self, obs):
        return {
            self.wrap_obs_key: obs
        }