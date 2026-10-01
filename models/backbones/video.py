import torch
from torch import nn
import torch.nn.functional as F

from ..layers.block import Block


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
