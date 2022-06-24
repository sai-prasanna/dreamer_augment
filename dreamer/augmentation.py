from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
if torch:
    import torch.nn.functional as F

class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4, consistent = False):
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

        padding = tuple([self.pad] * 4 + [0, 0]) #len(x.size()))
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid =  base_grid[None, None, ...].repeat(t, n, 1, 1, 1)  if len(x.size()) > 4 else base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

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
                                  size= (t, n, 1, 1, 2) if len(orig_size)>4 else (n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)



        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        if len(orig_size)>4:
            x = x.view((-1, *x.size()[2:]))
            grid = grid.view((-1, *grid.size()[2:]))

        samples = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        #samples = samples.view(orig_size)
        if len(orig_size)>4:
            samples = samples.view(orig_size)
            samples = samples.permute(1, 0, 2, 3, 4)
        else:
            return samples

        return samples