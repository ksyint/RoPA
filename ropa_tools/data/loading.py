"""Timestamped video loading and frozen dense-label propagation."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import av
from transformers import AutoVideoProcessor
import torch.nn.functional as F
from ropa_tools.data.sampling import uniform_indices, stride_indices


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
        return video_item_with_metadata(self, index)[0]

    def with_metadata(self, index):
        return video_item_with_metadata(self, index)


def propagate_labels(features, first_labels, height, width, history=7, topk=10,
                     temperature=0.07, radius=12, query_chunk=256, context_stride=1):
    values, _ = propagate_with_context(features, first_labels, height, width, history,
                                       topk, temperature, radius, query_chunk, context_stride)
    return values


def decode_interval(path, start=0., end=None):
    if start < 0 or (end is not None and end <= start):
        raise ValueError('Video decode intervals must have nonnegative start and positive duration.')
    frames, stamps = [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f'No video stream found: {path}')
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        if start:
            container.seek(int((start + origin) * av.time_base), backward=True)
        previous = None
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError('Every decoded video frame must expose a presentation timestamp.')
            timestamp = float(frame.time) - origin
            if timestamp < start:
                continue
            if end is not None and timestamp >= end:
                break
            if previous is not None and timestamp <= previous:
                raise ValueError('Decoded presentation timestamps are not strictly increasing.')
            frames.append(frame.to_ndarray(format='rgb24'))
            stamps.append(timestamp)
            previous = timestamp
        metadata = dict(
            stream_width=stream.width,
            stream_height=stream.height,
            time_base_numerator=stream.time_base.numerator,
            time_base_denominator=stream.time_base.denominator,
            stream_origin_seconds=origin,
            average_rate=float(stream.average_rate) if stream.average_rate is not None else None,
        )
    if not frames:
        raise ValueError(f'No decoded frames in the selected interval: {path}')
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError('Video resolution changed inside the selected interval.')
    metadata.update(decoded_frames=len(frames), first_timestamp=stamps[0], last_timestamp=stamps[-1])
    return frames, np.asarray(stamps, dtype=np.float64), metadata


def select_decoded_frames(row, stamps, count):
    from ropa_tools.data.sampling import nearest_timestamps

    if row.get('timestamps') is not None:
        requested = np.asarray(row['timestamps'], dtype=np.float64)
        if requested.shape != (count,):
            raise ValueError('Explicit timestamps must contain one value per requested input frame.')
        chosen = nearest_timestamps(stamps, requested)
        mode = 'timestamps'
    elif row.get('stride') is not None:
        chosen = stride_indices(len(stamps), count, int(row['stride']), start=int(row.get('frame_start', 0)))
        mode = 'frame_stride'
    elif row.get('sampling', 'uniform') == 'time':
        requested = np.linspace(stamps[0], stamps[-1], count)
        chosen = nearest_timestamps(stamps, requested)
        mode = 'presentation_time_uniform'
    elif row.get('sampling', 'uniform') == 'uniform':
        chosen = uniform_indices(len(stamps), count)
        mode = 'decoded_frame_uniform'
    else:
        raise ValueError('Video sampling must be uniform, time, stride, or explicit timestamps.')
    return chosen, dict(
        sampling=mode,
        selected_decoded_indices=chosen.tolist(),
        selected_timestamps=stamps[chosen].tolist(),
        unique_decoded_frames=len(np.unique(chosen)),
        repeated_input_frames=count - len(np.unique(chosen)),
    )


def video_item_with_metadata(dataset, index):
    row = dataset.rows[index]
    source = Path(row['video'])
    source = source if source.is_absolute() else dataset.root / source
    frames, stamps, decoded = decode_interval(source, float(row.get('start', 0)), row.get('end'))
    chosen, sampling = select_decoded_frames(row, stamps, dataset.frames)
    video = np.stack([frames[index] for index in chosen])
    values = dataset.processor(videos=[video], return_tensors='pt')['pixel_values_videos'][0]
    pixels = values.transpose(0, 1).contiguous()
    if pixels.ndim != 4 or pixels.shape[0] != 3 or pixels.shape[1] != dataset.frames:
        raise ValueError('The video processor returned incompatible C,T,H,W dimensions.')
    if not torch.isfinite(pixels).all():
        raise ValueError('The video processor returned nonfinite normalized pixels.')
    return pixels, dict(video=str(source.resolve()), **decoded, **sampling)


def tubelet_timestamps(metadata, tubelet):
    stamps = np.asarray(metadata['selected_timestamps'], dtype=np.float64)
    if tubelet < 1 or len(stamps) % tubelet:
        raise ValueError('Sampled frame count must be divisible by the model tubelet size.')
    grouped = stamps.reshape(-1, tubelet)
    return dict(
        tubelet_timestamps=grouped.mean(-1).tolist(),
        tubelet_start_timestamps=grouped[:, 0].tolist(),
        tubelet_end_timestamps=grouped[:, -1].tolist(),
        coordinate_unit='seconds from video stream origin',
    )


def context_frame_ids(frame, history, stride=1, retain_first=True):
    if frame < 1 or history < 1 or stride < 1:
        raise ValueError('Context selection requires a positive target frame, history and stride.')
    values = list(range(frame - 1, max(-1, frame - 1 - history * stride), -stride))
    if retain_first:
        values.append(0)
    return sorted(set(values))


@torch.no_grad()
def propagate_step(query, context, labels, height, width, topk=10,
                   temperature=.07, radius=12, query_chunk=256):
    if query.ndim != 2 or context.ndim != 3 or labels.ndim != 3:
        raise ValueError('Propagation requires N,D query, K,N,D contexts and K,N,C labels.')
    tokens = height * width
    if query.shape[0] != tokens or context.shape[1:] != query.shape:
        raise ValueError('Propagation feature grids or channel dimensions differ.')
    if labels.shape[:2] != context.shape[:2] or labels.shape[-1] < 1:
        raise ValueError('Context labels must align with every context feature patch.')
    if min(topk, temperature, query_chunk) <= 0 or radius < 0:
        raise ValueError('Propagation top-k, temperature and chunk must be positive with nonnegative radius.')
    query = F.normalize(query.float(), dim=-1)
    context = F.normalize(context.float(), dim=-1)
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(height, device=query.device), torch.arange(width, device=query.device), indexing='ij',
    ), -1).reshape(tokens, 2)
    keys = context.flatten(0, 1)
    probabilities = labels.float().flatten(0, 1)
    outputs, entropies, confidence = [], [], []
    for start in range(0, tokens, query_chunk):
        stop = min(start + query_chunk, tokens)
        scores = query[start:stop] @ keys.T / temperature
        local = (coordinates[start:stop, None] - coordinates[None]).abs().amax(-1) <= radius
        scores.masked_fill_(~local.repeat(1, len(context)), -torch.inf)
        strongest, indices = scores.topk(min(topk, scores.shape[-1]), -1)
        weights = strongest.softmax(-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError('Propagation neighborhood contains no valid source patches.')
        prediction = (probabilities[indices] * weights[..., None]).sum(1)
        outputs.append(prediction)
        entropies.append(-(weights * weights.clamp_min(1e-12).log()).sum(-1))
        confidence.append(weights.max(-1).values)
    return torch.cat(outputs), dict(neighbor_entropy=torch.cat(entropies), maximum_neighbor_weight=torch.cat(confidence))


@torch.no_grad()
def propagate_with_context(features, first_labels, height, width, history=7, topk=10,
                           temperature=.07, radius=12, query_chunk=256, context_stride=1,
                           retain_first=True, seed_labels=None):
    if features.ndim != 3 or features.shape[1] != height * width:
        raise ValueError('Video feature arrays must have T,H*W,D shape.')
    if first_labels.ndim != 2 or first_labels.shape[0] != features.shape[1]:
        raise ValueError('Initial label probabilities must align with the feature patch grid.')
    if features.device.type != 'cuda' or first_labels.device != features.device:
        raise ValueError('Feature propagation requires one CUDA device for features and labels.')
    if not torch.isfinite(features).all() or not torch.isfinite(first_labels).all() or (first_labels < 0).any():
        raise ValueError('Features must be finite and initial probabilities nonnegative.')
    annotations = dict(seed_labels or {})
    if any(type(index) is not int or not 0 <= index < len(features) for index in annotations):
        raise ValueError('Every supplied seed annotation must identify a feature frame.')
    labels = [first_labels.float()]
    records = []
    for frame in range(1, len(features)):
        if frame in annotations:
            current = annotations[frame].to(features.device).float()
            if current.shape != first_labels.shape or not torch.isfinite(current).all() or (current < 0).any():
                raise ValueError('All seed annotations must share valid N,C probability dimensions.')
            labels.append(current)
            records.append(dict(frame=frame, seeded=True, context=[]))
            continue
        selected = context_frame_ids(frame, history, context_stride, retain_first)
        contexts = features[selected]
        previous = torch.stack([labels[index] for index in selected])
        prediction, diagnostics = propagate_step(features[frame], contexts, previous, height, width,
                                                topk, temperature, radius, query_chunk)
        labels.append(prediction)
        records.append(dict(
            frame=frame, seeded=False, context=selected,
            mean_neighbor_entropy=float(diagnostics['neighbor_entropy'].mean()),
            mean_maximum_neighbor_weight=float(diagnostics['maximum_neighbor_weight'].mean()),
            mean_label_confidence=float(prediction.max(-1).values.mean()),
        ))
    return torch.stack(labels), records
