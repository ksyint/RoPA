import math
import pytest
import torch
from ropa import consistency_loss, paga_loss
from models.backbone import rotate_pairs, sample_spacing, temporal_frequencies
from video import propagate_labels


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


def test_identity_correspondence_keeps_labels():
    features = torch.eye(4).expand(3, 4, 4)
    labels = torch.eye(4)
    output = propagate_labels(features, labels, 2, 2, topk=1)
    assert torch.equal(output.argmax(-1), torch.arange(4).expand(3, 4))
