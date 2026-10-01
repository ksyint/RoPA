import torch
from torch import nn
import torch.nn.functional as F

from .attention import Attention


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
