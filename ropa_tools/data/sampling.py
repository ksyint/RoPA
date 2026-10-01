"""Deterministic resumable clip batches and timestamp-aware frame sampling."""
import math

import numpy as np
import torch


def uniform_indices(length, frames):
    if length < 1 or frames < 1:
        raise ValueError('Frame sampling requires positive source and target lengths.')
    return np.linspace(0, length - 1, frames).round().astype(np.int64)


def stride_indices(length, frames, stride, start=None, generator=None):
    if min(length, frames, stride) < 1:
        raise ValueError('Frame count, source length and sampling stride must be positive.')
    span = 1 + (frames - 1) * stride
    if length < span:
        raise ValueError('The video interval is shorter than the requested strided clip.')
    maximum = length - span
    if start is None:
        generator = generator or np.random.default_rng()
        start = int(generator.integers(maximum + 1))
    if not 0 <= start <= maximum:
        raise ValueError('The selected clip start lies outside the valid sampling window.')
    return start + np.arange(frames) * stride


def nearest_timestamps(stamps, requested):
    stamps = np.asarray(stamps, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)
    if stamps.ndim != 1 or not len(stamps) or not np.isfinite(stamps).all():
        raise ValueError('Decoded timestamps must be a nonempty finite vector.')
    if np.any(np.diff(stamps) <= 0):
        raise ValueError('Decoded timestamps must be strictly increasing.')
    if not np.isfinite(requested).all() or requested.min() < stamps[0] or requested.max() > stamps[-1]:
        raise ValueError('Requested timestamps lie outside the decoded interval.')
    right = np.searchsorted(stamps, requested).clip(max=len(stamps) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(stamps[right] - requested) < abs(stamps[left] - requested), right, left)


class ClipStream:
    def __init__(self, dataset, batch_size, seed, rank=0, world_size=1):
        if len(dataset) < 1 or batch_size < 1:
            raise ValueError('A clip stream needs a nonempty dataset and a positive batch size.')
        self.dataset = dataset
        self.rank, self.world_size = int(rank), int(world_size)
        if not 0 <= self.rank < self.world_size or len(dataset) < self.world_size:
            raise ValueError('Each training rank requires at least one dataset example.')
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)

    def __len__(self):
        return math.ceil(len(self.dataset) / (self.batch_size * self.world_size))

    def __next__(self):
        if self.offset >= len(self.order):
            self.epoch += 1
            self.offset = 0
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
        from ropa_tools.training.distributed import distributed_stream_indices
        selected, next_offset = distributed_stream_indices(
            self.order, self.offset, self.batch_size, self.rank, self.world_size,
        )
        values = [self.dataset[int(index)] for index in selected]
        if len({tuple(value.shape) for value in values}) != 1:
            raise ValueError('Training clips must share C,T,H,W dimensions within a batch.')
        self.offset = next_offset
        return torch.stack(values)

    def __iter__(self):
        return self

    def state_dict(self):
        return dict(
            seed=self.seed, batch_size=self.batch_size, dataset_length=len(self.dataset),
            rank=self.rank, world_size=self.world_size,
            epoch=self.epoch, offset=self.offset, order=self.order,
            generator=self.generator.get_state(),
        )

    def load_state_dict(self, state):
        for key in ('seed', 'batch_size', 'rank', 'world_size'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Training stream differs in {key}.')
        if state['dataset_length'] != len(self.dataset):
            raise ValueError('Training stream dataset length changed.')
        order = state['order'].cpu().long()
        if not torch.equal(order.sort().values, torch.arange(len(self.dataset))):
            raise ValueError('Saved clip order is not a permutation of this dataset.')
        if not 0 <= state['offset'] <= len(order) or state['epoch'] < 0:
            raise ValueError('Saved stream position is invalid.')
        self.epoch, self.offset = int(state['epoch']), int(state['offset'])
        self.order = order
        self.generator.set_state(state['generator'].cpu())
