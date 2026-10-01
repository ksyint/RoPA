import math

import torch
from torch import nn


def temporal_frequencies(dim, target_range=64.0, local_scale=1.0):
    """HTA, Eq. (4). Frequencies ordered from long to short wavelength."""
    if dim < 4 or dim % 2 or not 0 < local_scale <= target_range:
        raise ValueError('HTA needs an even dimension >= 4 and 0 < T0 <= T*.')
    return math.pi / target_range * torch.exp(
        torch.linspace(0, math.log(target_range / local_scale), dim // 2)
    )


def sample_spacing(batch_size, span, theta_min, low=0.5, high=2.0, device=None):
    """Sample the truncated log-uniform TSJ distribution, independently per clip."""
    if span < 0 or theta_min <= 0 or not 0 < low <= high:
        raise ValueError('Invalid spacing interval, span, or frequency.')
    upper = min(high, math.pi / (theta_min * span)) if span else high
    if upper < low:
        raise ValueError('Clip span has no feasible temporal spacing in the requested interval.')
    return torch.exp(torch.empty(batch_size, device=device).uniform_(math.log(low), math.log(upper)))


def rotate_pairs(x, angles):
    """x (..., D), angles broadcastable to (..., D/2)."""
    paired = x.reshape(*x.shape[:-1], -1, 2)
    real, imag = paired.unbind(-1)
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(-2)


class Rotary3D(nn.Module):
    def __init__(self, dims=(16, 24, 24), target_range=64.0, local_scale=1.0):
        super().__init__()
        if any(d % 2 or d <= 0 for d in dims):
            raise ValueError('Each coordinate block must have a positive even dimension.')
        self.dims = tuple(dims)
        self.register_buffer('time_freq', temporal_frequencies(dims[0], target_range, local_scale))
        for axis, dim in zip(('height', 'width'), dims[1:]):
            self.register_buffer(axis + '_freq', 10000 ** (-torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, coordinates, spacing=None):
        # x: B,H,N,D. Coordinates: N,3 in tubelet-time, patch-row, patch-column units.
        if x.shape[-1] != sum(self.dims) or coordinates.shape != (x.shape[-2], 3):
            raise ValueError('Rotary block dimensions or coordinate shape do not match tokens.')
        if spacing is None:
            spacing = x.new_ones(x.shape[0])
        blocks = x.split(self.dims, -1)
        result = []
        for index, (block, freq) in enumerate(zip(blocks, (self.time_freq, self.height_freq, self.width_freq))):
            position = coordinates[:, index].to(x)[None, None, :, None]
            if index == 0:
                position = position * spacing[:, None, None, None]
            result.append(rotate_pairs(block, position * freq.to(x)))
        return torch.cat(result, -1)


def band_diagnostics(frequencies, offsets):
    offsets = torch.as_tensor(offsets, dtype=frequencies.dtype, device=frequencies.device)
    return {'half_cycle': math.pi / frequencies.min().item(),
            'displacement': (1 - torch.cos(offsets[..., None] * frequencies)).mean(-1)}
