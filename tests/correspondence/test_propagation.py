import math

import pytest
import torch

from criterion.functional import consistency_loss, paga_loss
from models.layers.rotary import Rotary3D, rotate_pairs, sample_spacing, temporal_frequencies
from models import VideoEncoder
from propagation import propagate_labels


def test_identity_correspondence_keeps_labels():
    features = torch.eye(4).expand(3, 4, 4)
    labels = torch.eye(4)
    output = propagate_labels(features, labels, 2, 2, topk=1)
    assert torch.equal(output.argmax(-1), torch.arange(4).expand(3, 4))
