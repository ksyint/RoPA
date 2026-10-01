"""Rotary coordinates, temporal bands and pretrained attention adaptation."""
from __future__ import annotations
import math
import torch
from torch import nn
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2RopeAttention


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


def _rotate(x, positions, frequencies):
    angles = positions.float().unsqueeze(-1) * frequencies.float()
    cosine = angles.cos().repeat_interleave(2, -1).to(x.dtype)
    sine = angles.sin().repeat_interleave(2, -1).to(x.dtype)
    pairs = x.unflatten(-1, (-1, 2))
    turned = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return x * cosine + turned * sine


class HorizonAttention(VJEPA2RopeAttention):
    """Keep all pretrained projections and replace only coordinate rotations."""
    def configure_band(self, target_range, local_scale):
        head_dim = self.attention_head_size
        dims = (16, 24, 24) if head_dim == 64 else (8, 12, 12)
        if sum(dims) != head_dim:
            raise ValueError(f'Unsupported V-JEPA 2 head dimension: {head_dim}')
        self.d_dim, self.h_dim, self.w_dim = dims
        self.register_buffer('time_freq', torch.logspace(
            torch.log10(torch.tensor(torch.pi / target_range)),
            torch.log10(torch.tensor(torch.pi / local_scale)), self.d_dim // 2))
        for name, dim in [('height_freq', self.h_dim), ('width_freq', self.w_dim)]:
            self.register_buffer(name, 10000.0 ** (-torch.arange(dim // 2).float() / (dim // 2)))
        self.spacing = None

    def apply_rotary_embeddings(self, qk, pos_ids):
        time, height, width = [value.to(qk.device) for value in pos_ids]
        if time.ndim == 1:
            time, height, width = [value[None, None, :] for value in (time, height, width)]
        if self.spacing is not None:
            time = time * self.spacing[:, None, None]
        chunks = qk.split((self.d_dim, self.h_dim, self.w_dim), dim=-1)
        return torch.cat([_rotate(chunks[0], time, self.time_freq),
                          _rotate(chunks[1], height, self.height_freq),
                          _rotate(chunks[2], width, self.width_freq)], -1)


def install_horizon_attention(model, target_range, local_scale):
    for parent in list(model.modules()):
        for name, module in list(parent.named_children()):
            if isinstance(module, VJEPA2RopeAttention) and not isinstance(module, HorizonAttention):
                replacement = HorizonAttention(module.config, module.hidden_size, module.num_attention_heads)
                replacement.load_state_dict(module.state_dict(), strict=True)
                replacement.configure_band(target_range, local_scale)
                setattr(parent, name, replacement)
