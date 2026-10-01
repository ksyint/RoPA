import math

import torch
from torch import nn
import torch.nn.functional as F

from .rotary import Rotary3D


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
