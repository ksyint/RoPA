import math

import pytest
import torch

from criterion.functional import consistency_loss, paga_loss
from models.layers.rotary import Rotary3D, rotate_pairs, sample_spacing, temporal_frequencies
from models import VideoEncoder
from propagation import propagate_labels


def test_hta_endpoints_and_geometric_ratio():
    f = temporal_frequencies(16, 64, 1)
    assert f[0].item() == pytest.approx(math.pi / 64)
    assert f[-1].item() == pytest.approx(math.pi)
    assert torch.allclose(f[1:] / f[:-1], (f[1] / f[0]).expand(7))

def test_relative_rotation_dot_product():
    torch.manual_seed(0)
    q, k = torch.randn(4, 16, dtype=torch.float64), torch.randn(4, 16, dtype=torch.float64)
    freq = temporal_frequencies(16).double()
    left = (rotate_pairs(q, freq * 3) * rotate_pairs(k, freq * 11)).sum(-1)
    right = (q * rotate_pairs(k, freq * 8)).sum(-1)
    assert torch.allclose(left, right, atol=1e-12, rtol=1e-12)

def test_spacing_respects_half_cycle():
    scale = sample_spacing(1000, 100, math.pi / 64)
    assert scale.min() >= 0.5
    assert scale.max() <= 0.64
    with pytest.raises(ValueError):
        sample_spacing(1, 200, math.pi / 64)
