import time
import torch.nn.functional as F
from torch import Tensor
import kornia.augmentation as K
import torch
import torch.nn as nn

class Augmentation(nn.Module):
    def __init__(
            self,
            rand_crop: bool,
            strong: bool,
            consistent: bool,
            pad: int = 4,
            image_size: int = 64
    ) -> None:
        super().__init__()
        self.random_crop = None
        if rand_crop:
            self.random_crop = K.VideoSequential(
                K.RandomCrop((image_size, image_size), padding=pad, padding_mode='replicate'))
        self.strong_transform = None
        if strong:
            self.strong_transform = [
                #K.RandomRotation(5, p=1 / 8),
                K.RandomSharpness((0, 1.0), p=1 / 7),
                K.RandomPosterize((3, 8), p=1 / 7),
                K.RandomSolarize(0.1, p=1 / 7),
                K.RandomEqualize(p=1 / 7),
                K.ColorJiggle(brightness=(0.4, 1.4), p=1 / 7),
                K.ColorJiggle(saturation=(0.4, 1.4), p=1 / 7),
                K.ColorJiggle(contrast=(0.4, 1.4), p=1 / 7),
            ]
            # Strong augmentations are always consistent across timesteps
            self.strong_transfom_fn = K.VideoSequential(*self.strong_transform, same_on_frame=True)
        self.consistent_crop = consistent

    def forward(self, X: Tensor) -> Tensor:
        X = X.permute(0, 1, 4, 2, 3)
        X = X + 0.5 # The image should be in range 0 to 1.0 for strong transforms
        orig_shape = X.shape
        if self.random_crop:
            if self.consistent_crop:
                X = self.random_crop(X.reshape(orig_shape[0], 1, orig_shape[1] * orig_shape[2], *orig_shape[3:])).view(
                    orig_shape).contiguous()
            else:
                X = self.random_crop(X).contiguous()
        if self.strong_transform:
            X = self.strong_transfom_fn(X)
        X = X - 0.5
        X = X.permute(0, 1, 3, 4, 2)
        return X


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4, consistent=False):
        super().__init__()
        self.pad = pad
        self.consistent = consistent

    def forward(self, x):
        orig_size = x.size()
        if len(orig_size) > 4:
            x = x.permute(1, 0, 2, 3, 4)
            orig_size = x.size()
            t, n, c, h, w = x.size()
        else:
            n, c, h, w = x.size()

        assert h == w

        padding = tuple([self.pad] * 4 + [0, 0])  # len(x.size()))
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid[None, None, ...].repeat(t, n, 1, 1, 1) if len(x.size()) > 4 else base_grid.unsqueeze(
            0).repeat(n, 1, 1, 1)

        if self.consistent:
            shift = torch.randint(0,
                                  2 * self.pad + 1,
                                  size=(n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)
            shift = shift.repeat(t, 1, 1, 1, 1) if len(orig_size) > 4 else shift
        else:
            shift = torch.randint(0,
                                  2 * self.pad + 1,
                                  size=(t, n, 1, 1, 2) if len(orig_size) > 4 else (n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)

        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        if len(orig_size) > 4:
            x = x.view((-1, *x.size()[2:]))
            grid = grid.view((-1, *grid.size()[2:]))

        samples = F.grid_sample(x,
                                grid,
                                padding_mode='zeros',
                                align_corners=False)
        # samples = samples.view(orig_size)
        if len(orig_size) > 4:
            samples = samples.view(orig_size)
            samples = samples.permute(1, 0, 2, 3, 4)
        else:
            return samples

        return samples
