import math

import pytest
import torch

from criterion.functional import consistency_loss, paga_loss
from models.layers.rotary import Rotary3D, rotate_pairs, sample_spacing, temporal_frequencies
from models import VideoEncoder
from propagation import propagate_labels


def test_paga_offset_sum_and_detached_teacher():
    a = torch.randn(2, 3, 4, 8, requires_grad=True)
    b = torch.randn_like(a)
    teacher = torch.randn_like(a, requires_grad=True)
    single = paga_loss(a[:, :1], b[:, :1], teacher[:, :1], b[:, :1])
    repeated = paga_loss(a[:, :1].expand(-1, 3, -1, -1), b[:, :1].expand(-1, 3, -1, -1),
                         teacher[:, :1].expand(-1, 3, -1, -1), b[:, :1].expand(-1, 3, -1, -1))
    assert torch.allclose(repeated, 3 * single)
    repeated.backward()
    assert a.grad is not None and teacher.grad is None

def test_exact_translation_has_zero_composition_loss():
    def predictor(z, delta):
        return z + delta
    assert consistency_loss(predictor, torch.randn(2, 4, 8), 2, 3).item() < 1e-10

def test_paga_averages_source_anchors():
    a, b, teacher = [torch.randn(2, 1, 4, 8) for _ in range(3)]
    loss = paga_loss(a, b, teacher, b)
    repeated = paga_loss(a.repeat(3, 1, 1, 1), b.repeat(3, 1, 1, 1),
                         teacher.repeat(3, 1, 1, 1), b.repeat(3, 1, 1, 1))
    assert torch.allclose(loss, repeated)
