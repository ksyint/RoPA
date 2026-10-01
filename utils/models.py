import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.rope import Rotary3D


class Attention(nn.Module):
    def __init__(self, dim, heads, rotary_dims, target_range, local_scale):
        super().__init__()
        if dim != heads * sum(rotary_dims):
            raise ValueError('Model dimension must equal heads * sum(rotary_dims).')
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = Rotary3D(rotary_dims, target_range, local_scale)
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(dim // heads))))

    def forward(self, x, coordinates, spacing):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q = self.rope(F.normalize(q, dim=-1), coordinates, spacing)
        k = self.rope(F.normalize(k, dim=-1), coordinates, spacing)
        logits = (q @ k.transpose(-1, -2)) * self.log_scale.exp()
        # Within-frame attention is unrestricted; future tubelets are unavailable.
        future = coordinates[None, :, 0] > coordinates[:, None, 0]
        logits = logits.masked_fill(future, -torch.inf)
        return self.proj((logits.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads, rotary_dims, target_range, local_scale):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, rotary_dims, target_range, local_scale)
        hidden = int(dim * 8 / 3)
        self.fc1, self.fc2 = nn.Linear(dim, hidden * 2), nn.Linear(hidden, dim)
        self.scale1 = nn.Parameter(torch.full((dim,), 1e-5))
        self.scale2 = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x, coordinates, spacing):
        x = x + self.scale1 * self.attn(self.norm1(x), coordinates, spacing)
        gate, value = self.fc1(self.norm2(x)).chunk(2, -1)
        return x + self.scale2 * self.fc2(F.silu(gate) * value)


class VideoEncoder(nn.Module):
    def __init__(self, dim=64, depth=2, heads=1, rotary_dims=(16, 24, 24),
                 target_range=64.0, local_scale=1.0, patch_size=8, tubelet=2):
        super().__init__()
        self.patch_size, self.tubelet = patch_size, tubelet
        self.embed = nn.Conv3d(3, dim, kernel_size=(tubelet, patch_size, patch_size),
                               stride=(tubelet, patch_size, patch_size))
        self.blocks = nn.ModuleList([Block(dim, heads, rotary_dims, target_range, local_scale) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, video, spacing=None):
        # video B,C,T,H,W; reject truncation to make temporal units explicit.
        if video.shape[2] % self.tubelet or any(s % self.patch_size for s in video.shape[-2:]):
            raise ValueError('Video dimensions must be divisible by tubelet and patch size.')
        x = self.embed(video)
        b, d, t, h, w = x.shape
        coords = torch.stack(torch.meshgrid(torch.arange(t, device=x.device), torch.arange(h, device=x.device),
                                            torch.arange(w, device=x.device), indexing='ij'), -1).reshape(-1, 3)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x, coords, spacing)
        return F.normalize(self.norm(x), dim=-1).reshape(b, t, h * w, d)


class TemporalPredictor(nn.Module):
    """A small residual, offset-conditioned predictor; replace for large-scale runs."""
    def __init__(self, dim):
        super().__init__()
        self.offset = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, z, delta):
        offset = torch.as_tensor(delta, device=z.device, dtype=z.dtype).reshape(1)
        return F.normalize(z + self.net(z + self.offset(offset)), dim=-1)


class RoPA(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = VideoEncoder(**kwargs)
        self.predictor = TemporalPredictor(kwargs.get('dim', 64))

    def forward(self, video, spacing=None):
        return self.encoder(video, spacing)
