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
from ropa_tools.data.loading import propagate_labels


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


def frame_metrics(predicted, target, category, tolerance=.008, void_label=255):
    valid = target != void_label
    if not valid.any():
        raise ValueError('An evaluated frame has no annotated pixels.')
    first, second = (predicted == category) & valid, (target == category) & valid
    union = (first | second).sum()
    jaccard = (first & second).sum().float() / union if union else predicted.new_tensor(1.0, dtype=torch.float32)
    first_boundary, second_boundary = boundary(first), boundary(second)
    if not valid.all():
        near_void = F.max_pool2d((~valid).float()[None, None], 3, 1, 1)[0, 0].bool()
        first_boundary &= ~near_void
        second_boundary &= ~near_void
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


def evaluate_sequence(row, device, history=7, topk=10, temperature=.07, radius=12, tolerance=.008,
                      query_chunk=256, context_stride=1):
    features, labels = load_sequence(row, device)
    void = int(row.get('void_label', 255))
    categories = sorted(int(value) for value in labels[0].unique().tolist() if value not in (0, void))
    if not categories:
        raise ValueError(f'{row["name"]}: first annotation contains no foreground instance.')
    first, category_ids = encode_instance_labels(labels[0], void)
    with torch.no_grad():
        probabilities = propagate_labels(features, first, labels.shape[1], labels.shape[2],
                                         history=history, topk=topk, temperature=temperature, radius=radius, query_chunk=query_chunk, context_stride=context_stride)
        predicted = category_ids[probabilities.argmax(-1)].reshape_as(labels)
    measurements = []
    objects = {}
    for category in categories:
        values = []
        for frame in range(1, len(labels)):
            if (labels[frame] == void).all():
                continue
            metrics = frame_metrics(predicted[frame], labels[frame], category, tolerance, void)
            values.append(metrics)
            measurements.append(dict(sequence=row['name'], frame=frame, category=category,
                                     area_fraction=float((labels[frame] == category).float().mean()), **metrics))
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
    parser.add_argument('--aggregation', choices=('objects', 'frames'), default='objects')
    parser.add_argument('--exclude-last-frame', action='store_true')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--query-chunk', type=int, default=256)
    parser.add_argument('--context-stride', type=int, default=1)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sequences = [], {}
    for row in load_sequences(args.sequences):
        prediction, values, objects = evaluate_sequence(row, device, args.history, args.topk,
            args.temperature, args.radius, args.boundary_tolerance, args.query_chunk, args.context_stride)
        safe_name = Path(row['name']).name
        if safe_name != row['name']:
            raise ValueError('Sequence names must be plain filenames.')
        np.save(output / f'{safe_name}.npy', prediction.cpu().numpy(), allow_pickle=False)
        if args.png:
            palette = annotation_palette(row['annotation_dir']) if row.get('annotation_dir') else None
            export_sequence_masks(prediction, output / 'masks' / safe_name, palette, row.get('frame_indices'))
        rows.extend(values)
        sequences[row['name']] = {'objects': objects, 'region_sizes': region_size_summary(values), **summarize_frames(values)}
    (output / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = object_summary(rows, args.exclude_last_frame) if args.aggregation == 'objects' else summarize_frames(rows)
    report = {'summary': summary, 'sequences': sequences,
              'grid': 'processor-aligned feature patches', 'exclude_first_frame': True}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))


def curve_statistics(values):
    if not values:
        return dict(mean=None, recall=None, decay=None, frames=0)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Region and contour scores must be finite values in [0,1].')
    count = len(values)
    quarter = max(1, math.ceil(count / 4))
    return dict(
        mean=sum(values) / count,
        recall=sum(value > .5 for value in values) / count,
        decay=sum(values[:quarter]) / quarter - sum(values[-quarter:]) / quarter,
        frames=count,
    )


def object_summary(measurements, exclude_last=False):
    groups = {}
    final_frames = {}
    for row in measurements:
        name = row['sequence']
        final_frames[name] = max(final_frames.get(name, 0), row['frame'])
    for row in measurements:
        if exclude_last and row['frame'] == final_frames[row['sequence']]:
            continue
        groups.setdefault((row['sequence'], row['category']), []).append(row)
    result = []
    for (sequence, category), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row['frame'])
        region = curve_statistics([row['region_iou'] for row in rows])
        boundary = curve_statistics([row['boundary_f'] for row in rows])
        result.append(dict(
            sequence=sequence, category=category,
            region=region, boundary=boundary,
            jf=(region['mean'] + boundary['mean']) / 2,
            first_evaluated_frame=rows[0]['frame'], last_evaluated_frame=rows[-1]['frame'],
        ))
    if not result:
        raise ValueError('No object-frame measurements remain after endpoint selection.')
    average = lambda name, statistic: sum(row[name][statistic] for row in result) / len(result)
    return dict(
        objects=len(result), sequences=len({row['sequence'] for row in result}),
        region_iou=average('region', 'mean'), boundary_f=average('boundary', 'mean'),
        jf=sum(row['jf'] for row in result) / len(result),
        region_recall=average('region', 'recall'), boundary_recall=average('boundary', 'recall'),
        region_decay=average('region', 'decay'), boundary_decay=average('boundary', 'decay'),
        aggregation='equal weight per annotated foreground object',
        exclude_last_frame=exclude_last, per_object=result,
    )


def annotation_palette(directory):
    masks = sorted(Path(directory).glob('*.png'))
    if not masks:
        raise ValueError('The annotation directory has no PNG masks.')
    with Image.open(masks[0]) as first:
        palette = first.getpalette()
    return palette


def export_sequence_masks(predictions, output, palette=None, frame_indices=None):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if predictions.ndim != 3:
        raise ValueError('Mask export requires T,H,W predictions.')
    values = predictions.detach().cpu().numpy()
    if values.min() < 0 or values.max() > 255:
        raise ValueError('Palette masks support integer IDs between zero and 255.')
    indices = list(range(len(values))) if frame_indices is None else frame_indices
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError('PNG export needs one distinct annotation frame index per prediction.')
    records = []
    for index, mask in zip(indices, values):
        filename = directory / f'{int(index):05d}.png'
        image = Image.fromarray(mask.astype(np.uint8), mode='P')
        if palette is not None:
            image.putpalette(palette)
        image.save(filename)
        records.append(str(filename))
    return records


def region_size_summary(measurements):
    grouped = {'small': [], 'medium': [], 'large': []}
    for row in measurements:
        fraction = row['area_fraction']
        group = 'small' if fraction < .01 else 'medium' if fraction < .1 else 'large'
        grouped[group].append(row)
    return {name: dict(summarize_frames(rows), object_frame_fraction_range=interval)
            for (name, rows), interval in zip(grouped.items(), ((0., .01), (.01, .1), (.1, 1.)))}


def encode_instance_labels(labels, void_label=255):
    if labels.ndim != 2 or labels.dtype != torch.long:
        raise ValueError('Instance labels must be an integer H,W map.')
    valid = labels != void_label
    if not valid.any():
        raise ValueError('An instance seed must contain at least one labelled pixel.')
    categories = labels[valid].unique(sorted=True)
    if (categories < 0).any():
        raise ValueError('Instance IDs must be nonnegative.')
    if not (categories == 0).any():
        categories = torch.cat((categories.new_zeros(1), categories))
    flat = labels.flatten()
    positions = torch.searchsorted(categories, flat.clamp_min(0)).clamp_max(len(categories) - 1)
    encoded = F.one_hot(positions, len(categories)).float()
    encoded[~valid.flatten()] = 0.
    return encoded, categories
