"""Timestamped video loading and frozen dense-label propagation."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import av
from transformers import AutoVideoProcessor
import torch.nn.functional as F


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


class VideoManifest(Dataset):
    def __init__(self, manifest, model_options, frames=16):
        self.root = Path(manifest).resolve().parent
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows or frames < 4 or frames % 2:
            raise ValueError('Supply nonempty video JSONL and an even frame count of at least four.')
        self.frames = frames
        self.paths = [Path(row['video']) for row in self.rows]
        self.processor = AutoVideoProcessor.from_pretrained(
            model_options['pretrained'], cache_dir=model_options.get('cache_dir'),
            revision=model_options.get('revision', 'main'),
            local_files_only=model_options.get('local_files_only', False))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['video'])
        path = path if path.is_absolute() else self.root / path
        start, end = float(row.get('start', 0)), float(row.get('end', float('inf')))
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base)
            if start > 0:
                container.seek(int((start + origin) * av.time_base), backward=True)
            for frame in container.decode(video=0):
                stamp = float(frame.time) - origin if frame.time is not None else None
                if stamp is None:
                    raise ValueError(f'Video frame has no presentation timestamp: {path}')
                if stamp >= end:
                    break
                if stamp >= start:
                    frames.append(frame.to_ndarray(format='rgb24'))
        if not frames:
            raise ValueError(f'No decoded frames inside [{start}, {end}) for {path}')
        indices = np.linspace(0, len(frames) - 1, self.frames).round().astype(int)
        video = np.stack([frames[i] for i in indices])
        values = self.processor(videos=[video], return_tensors='pt')['pixel_values_videos'][0]
        return values.transpose(0, 1).contiguous()


def propagate_labels(features, first_labels, height, width, history=7, topk=10,
                     temperature=0.07, radius=12):
    """Frozen affinities, first frame + preceding frames. Output T,N,C probabilities."""
    if features.ndim != 3 or features.shape[1] != height * width:
        raise ValueError('Features must have T,H*W,D shape.')
    if topk <= 0 or temperature <= 0 or history < 1:
        raise ValueError('topk, temperature, and history must be positive.')
    features = F.normalize(features, dim=-1)
    labels = [first_labels.float()]
    grid = torch.stack(torch.meshgrid(torch.arange(height, device=features.device),
                                     torch.arange(width, device=features.device), indexing='ij'), -1).reshape(-1, 2)
    allowed = (grid[:, None] - grid[None, :]).abs().amax(-1) <= radius
    for frame in range(1, len(features)):
        context = sorted(set([0] + list(range(max(0, frame - history), frame))))
        affinity = features[frame] @ features[context].flatten(0, 1).T / temperature
        affinity = affinity.masked_fill(~allowed.repeat(1, len(context)), -torch.inf)
        values, indices = affinity.topk(min(topk, affinity.shape[-1]), -1)
        weights = values.softmax(-1)
        source = torch.cat([labels[t] for t in context])
        labels.append((source[indices] * weights[..., None]).sum(1))
    return torch.stack(labels)
