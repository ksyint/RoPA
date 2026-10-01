from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Directory of .npy clips, each already processor-normalized float C,T,H,W."""
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
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError('Cached clips must contain processor-normalized floating point pixels.')
        return torch.from_numpy(arr.astype(np.float32))
