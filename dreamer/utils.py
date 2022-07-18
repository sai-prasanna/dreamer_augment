import numpy as np

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
from torch import distributions as td

# Custom initialization for different types of layers
if torch:

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


    class GRUCell(nn.GRUCell):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight_ih)
            nn.init.orthogonal_(self.weight_hh)
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)


    # Custom Tanh Bijector due to big gradients through Dreamer Actor
    class TanhBijector(torch.distributions.Transform):
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


class FeatureTripletBuilder(object):
    def __init__(self, feature_1, feature_2, negative_frame_margin=10):
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.negative_frame_margin = negative_frame_margin
        self.anchor = feature_1.view(-1, feature_1.shape[2])
        self.positive = feature_2.view(-1, feature_2.shape[2])
        self.frame_lengths = self.anchor.shape[0]
        self.negatives = torch.Tensor(self.anchor.shape[0], self.anchor.shape[1]).cuda()

    def sample_negative(self, anchor_index):
        negative_index = self.sample_negative_frame_index(anchor_index)
        negative_frame = self.positive[negative_index, :]
        return negative_frame

    def build_set(self):
        for i in range(0, self.anchor.shape[0]):
            self.negatives[i, :] = self.sample_negative(i)

        return (self.anchor, self.positive,
                self.negatives)

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))


def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, batch_size, feature_dim, lambd=0.0051):
        super().__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.lambd = lambd
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.feature_dim, affine=False)

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def compute_cpc_obj(pred, features, cpc_amount=(10, 30), cpc_contrast='window'):

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