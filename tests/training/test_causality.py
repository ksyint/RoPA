import math

import pytest
import torch

from criterion.functional import consistency_loss, paga_loss
from models.layers.rotary import Rotary3D, rotate_pairs, sample_spacing, temporal_frequencies
from models import VideoEncoder
from propagation import propagate_labels


def test_causal_encoder_does_not_see_future():
    torch.manual_seed(0)
    model = VideoEncoder(depth=1).eval()
    video = torch.randn(1, 3, 8, 16, 16)
    changed = video.clone()
    changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:]) * 50
    assert torch.allclose(model(video)[:, :2], model(changed)[:, :2], atol=1e-6)
