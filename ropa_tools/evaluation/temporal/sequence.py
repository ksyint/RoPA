"""Sequence manifests, aligned masks and CUDA region/boundary measurements."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.backbone import cuda_device
from ropa_tools.data.video.loading import propagate_labels


def load_sequences(path):
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = set()
    for row in records:
        if not row.get('name') or row['name'] in names:
            raise ValueError('Every sequence needs a unique name.')
        names.add(row['name'])
        for key in ('features', 'labels', 'annotation_dir'):
            if row.get(key):
                value = Path(row[key])
                row[key] = str(value if value.is_absolute() else path.parent / value)
        if not row.get('labels') and not row.get('annotation_dir'):
            raise ValueError('Supply aligned label arrays or an annotation directory.')
    if not records:
        raise ValueError('The sequence manifest is empty.')
    return records


def project_mask(path, grid, crop_size, resize_shorter):
    with Image.open(path) as image:
        width, height = image.size
        scale = resize_shorter / min(width, height)
        resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.NEAREST)
        left = (resized.width - crop_size) // 2
        top = (resized.height - crop_size) // 2
        if left < 0 or top < 0:
            raise ValueError('Mask resize must cover the model crop.')
        cropped = resized.crop((left, top, left + crop_size, top + crop_size))
        return np.asarray(cropped.resize((grid[1], grid[0]), Image.Resampling.NEAREST)).copy()


def load_sequence(row, device):
    features = np.load(row['features'], allow_pickle=False)
    if features.ndim != 3 or not np.isfinite(features).all():
        raise ValueError('Features must be finite T,N,D arrays.')
    if row.get('labels'):
        labels = np.load(row['labels'], allow_pickle=False)
    else:
        if not row.get('center_crop', True):
            raise ValueError('PNG projection uses centered crops. Supply aligned label arrays for other geometry.')
        masks = sorted(Path(row['annotation_dir']).glob('*.png'))
        indices = row.get('frame_indices')
        if indices is None or len(indices) != len(features):
            raise ValueError('Specify one annotation frame index per extracted tubelet.')
        if any(i < 0 or i >= len(masks) for i in indices):
            raise ValueError('Annotation frame index is outside the sequence.')
        grid = (int(row['grid_height']), int(row['grid_width']))
        crop = int(row.get('crop_size', 384))
        resize = int(row.get('resize_shorter', crop))
        labels = np.stack([project_mask(masks[i], grid, crop, resize) for i in indices])
    if labels.ndim != 3 or len(labels) != len(features):
        raise ValueError('Label arrays need T,H,W with the same tubelet count.')
    if labels.shape[1] * labels.shape[2] != features.shape[1]:
        raise ValueError('Labels and feature patch grids differ.')
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError('Labels must be nonnegative integer instance IDs.')
    return torch.from_numpy(features).float().to(device), torch.from_numpy(labels.astype(np.int64)).to(device)


def boundary(mask):
    result = torch.zeros_like(mask, dtype=torch.bool)
    result[..., :-1, :] |= mask[..., :-1, :] != mask[..., 1:, :]
    result[..., :, :-1] |= mask[..., :, :-1] != mask[..., :, 1:]
    result[..., :-1, :-1] |= mask[..., :-1, :-1] != mask[..., 1:, 1:]
    return result


def dilate_boundary(mask, radius):
    offsets = torch.arange(-radius, radius + 1, device=mask.device)
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    disk = (xx.square() + yy.square() <= radius * radius).float()[None, None]
    return F.conv2d(mask.float()[None, None], disk, padding=radius)[0, 0] > 0


def frame_metrics(predicted, target, category, tolerance=.008):
    first, second = predicted == category, target == category
    union = (first | second).sum()
    jaccard = (first & second).sum().float() / union if union else predicted.new_tensor(1.0, dtype=torch.float32)
    first_boundary, second_boundary = boundary(first), boundary(second)
    radius = max(1, math.ceil(tolerance * math.hypot(*first.shape)))
    first_count, second_count = first_boundary.sum(), second_boundary.sum()
    if first_count == 0 and second_count == 0:
        f_score = jaccard.new_tensor(1.)
    elif first_count == 0 or second_count == 0:
        f_score = jaccard.new_tensor(0.)
    else:
        precision = (first_boundary & dilate_boundary(second_boundary, radius)).sum() / first_count
        recall = (second_boundary & dilate_boundary(first_boundary, radius)).sum() / second_count
        f_score = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {'region_iou': float(jaccard), 'boundary_f': float(f_score),
            'jf': float((jaccard + f_score) / 2)}


def summarize_frames(rows):
    if not rows:
        return {'frames': 0, 'region_iou': None, 'boundary_f': None, 'jf': None}
    return {'frames': len(rows), **{key: sum(row[key] for row in rows) / len(rows)
                                   for key in ('region_iou', 'boundary_f', 'jf')}}


def evaluate_sequence(row, device, history=7, topk=10, temperature=.07, radius=12, tolerance=.008):
    features, labels = load_sequence(row, device)
    categories = sorted(int(value) for value in labels[0].unique().tolist() if value != 0)
    if not categories:
        raise ValueError(f'{row["name"]}: first annotation contains no foreground instance.')
    classes = int(labels.max()) + 1
    first = F.one_hot(labels[0].flatten(), classes).float()
    with torch.no_grad():
        probabilities = propagate_labels(features, first, labels.shape[1], labels.shape[2],
                                         history=history, topk=topk, temperature=temperature, radius=radius)
        predicted = probabilities.argmax(-1).reshape_as(labels)
    measurements = []
    objects = {}
    for category in categories:
        values = []
        for frame in range(1, len(labels)):
            metrics = frame_metrics(predicted[frame], labels[frame], category, tolerance)
            values.append(metrics)
            measurements.append(dict(sequence=row['name'], frame=frame, category=category, **metrics))
        objects[str(category)] = summarize_frames(values)
    return predicted, measurements, objects


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--history', type=int, default=7)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--radius', type=int, default=12)
    parser.add_argument('--boundary-tolerance', type=float, default=.008)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sequences = [], {}
    for row in load_sequences(args.sequences):
        prediction, values, objects = evaluate_sequence(row, device, args.history, args.topk,
            args.temperature, args.radius, args.boundary_tolerance)
        safe_name = Path(row['name']).name
        if safe_name != row['name']:
            raise ValueError('Sequence names must be plain filenames.')
        np.save(output / f'{safe_name}.npy', prediction.cpu().numpy(), allow_pickle=False)
        rows.extend(values)
        sequences[row['name']] = {'objects': objects, **summarize_frames(values)}
    (output / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'summary': summarize_frames(rows), 'sequences': sequences,
              'grid': 'processor-aligned feature patches', 'exclude_first_frame': True}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
