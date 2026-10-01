import torch
from torch import nn
import torch.nn.functional as F


class TemporalPredictor(nn.Module):
    """A small residual, offset-conditioned predictor; replace for large-scale runs."""
    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        self.offset = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        hidden = int(dim * hidden_ratio)
        if hidden < 1:
            raise ValueError('Predictor hidden width must be positive.')
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, z, delta):
        offset = torch.as_tensor(delta, device=z.device, dtype=z.dtype).reshape(1)
        return F.normalize(z + self.net(z + self.offset(offset)), dim=-1)
