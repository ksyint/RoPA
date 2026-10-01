from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Directory of .npy clips, each float C,T,H,W in [0,1] (or uint8)."""
    def __init__(self, root):
        self.paths = sorted(Path(root).glob('*.npy'))
        if not self.paths:
            raise ValueError('No .npy clips found.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arr = np.load(self.paths[index], allow_pickle=False)
        if arr.ndim != 4 or arr.shape[0] != 3 or not np.isfinite(arr).all():
            raise ValueError(f'Invalid C,T,H,W clip: {self.paths[index]}')
        return torch.from_numpy(arr.astype(np.float32)) / (255 if arr.dtype == np.uint8 else 1)


class SyntheticVideo(Dataset):
    def __init__(self, size=16, frames=8, image_size=16, seed=42):
        self.size, self.frames, self.image_size, self.seed = size, frames, image_size, seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        base = torch.rand(3, self.image_size, self.image_size, generator=generator)
        return torch.stack([base.roll(t // 2, -1) for t in range(self.frames)], 1)
