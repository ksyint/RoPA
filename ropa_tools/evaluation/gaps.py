"""Single-context temporal-gap propagation and point correspondence measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import cuda_device
from ropa_tools.evaluation.sequence import frame_metrics, load_sequence, load_sequences, summarize_frames


def match_frame(source, target, source_labels, shape, topk=10, temperature=.07, radius=12):
    if min(topk, temperature) <= 0 or radius < 0:
        raise ValueError('Use positive top-k/temperature and nonnegative search radius.')
    source = F.normalize(source, dim=-1)
    target = F.normalize(target, dim=-1)
    height, width = shape
    grid = torch.stack(torch.meshgrid(torch.arange(height, device=source.device),
                       torch.arange(width, device=source.device), indexing='ij'), -1).reshape(-1, 2)
    allowed = (grid[:, None] - grid[None, :]).abs().amax(-1) <= radius
    affinities = (target @ source.T / temperature).masked_fill(~allowed, -torch.inf)
    weights, positions = affinities.topk(min(topk, source.shape[0]), dim=-1)
    classes = int(source_labels.max()) + 1
    encoded = F.one_hot(source_labels.flatten(), classes).float()
    return (weights.softmax(-1)[..., None] * encoded[positions]).sum(1).argmax(-1).reshape(height, width)


def evaluate_gaps(row, gaps, device, options):
    features, labels = load_sequence(row, device)
    records = []
    with torch.no_grad():
        for gap in gaps:
            for target in range(gap, len(features)):
                source = target - gap
                prediction = match_frame(features[source], features[target], labels[source],
                    labels.shape[1:], options['topk'], options['temperature'], options['radius'])
                categories = [int(value) for value in labels[source].unique().tolist() if value]
                for category in categories:
                    metrics = frame_metrics(prediction, labels[target], category)
                    records.append(dict(sequence=row['name'], gap=gap, source=source,
                                        target=target, category=category, **metrics))
    return records


def gap_summary(records, ratio=.9):
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row['gap'])].append(row)
    summaries = {str(gap): summarize_frames(rows) for gap, rows in sorted(grouped.items())}
    if not summaries:
        raise ValueError('No evaluated frame pair fits the requested gaps.')
    baseline_gap = min(grouped)
    reference = summaries[str(baseline_gap)]['jf']
    crossing = next((gap for gap in sorted(grouped) if summaries[str(gap)]['jf'] < ratio * reference), None)
    return {'gaps': summaries, 'baseline_gap': baseline_gap, 'threshold_ratio': ratio,
            'first_threshold_crossing': crossing}


def sample_query_features(features, queries, height, width):
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError('Queries need Q,3 in [frame,y,x] pixel coordinates.')
    frames = queries[:, 0].long()
    if (frames < 0).any() or (frames >= len(features)).any():
        raise ValueError('Query frame is outside the feature sequence.')
    x = queries[:, 2] / max(1, width - 1) * 2 - 1
    y = queries[:, 1] / max(1, height - 1) * 2 - 1
    if (x.abs() > 1).any() or (y.abs() > 1).any():
        raise ValueError('Query coordinates lie outside the feature grid.')
    planes = features[frames].reshape(len(frames), height, width, -1).permute(0, 3, 1, 2)
    points = torch.stack((x, y), -1)[:, None, None]
    return F.grid_sample(planes, points, align_corners=True)[:, :, 0, 0]


def track_points(features, queries, height, width):
    anchors = F.normalize(sample_query_features(features, queries, height, width), dim=-1)
    tokens = F.normalize(features, dim=-1)
    affinities = torch.einsum('qd,tnd->qtn', anchors, tokens)
    indices = affinities.argmax(-1)
    return torch.stack((indices % width, indices // width), -1).float()


def point_metrics(prediction, target, visible, query_frames, thresholds=(1, 2, 4, 8, 16)):
    if prediction.shape != target.shape or visible.shape != target.shape[:-1]:
        raise ValueError('Predicted and target points must share Q,T,2 with Q,T visibility.')
    valid = visible.bool().clone()
    valid[torch.arange(len(valid), device=valid.device), query_frames.long()] = False
    if not valid.any():
        raise ValueError('No visible non-query point observations are available.')
    distances = (prediction - target).square().sum(-1).sqrt()
    return {'observations': int(valid.sum()), 'mean_pixel_error': float(distances[valid].mean()),
            'pck': {str(value): float((distances[valid] <= value).float().mean()) for value in thresholds}}


def run_points(args):
    with np.load(args.points, allow_pickle=False) as source:
        required = {'features', 'queries', 'targets', 'visible', 'grid_height', 'grid_width'}
        if not required <= set(source.files):
            raise ValueError(f'Point archive needs {sorted(required)}')
        arrays = {key: source[key] for key in required}
    device = cuda_device(args.device)
    height, width = int(arrays['grid_height']), int(arrays['grid_width'])
    if height < 1 or width < 1:
        raise ValueError('Point grids need positive dimensions.')
    features = torch.from_numpy(arrays['features']).float().to(device)
    queries = torch.from_numpy(arrays['queries']).float().to(device)
    targets = torch.from_numpy(arrays['targets']).float().to(device)
    visible = torch.from_numpy(arrays['visible']).bool().to(device)
    if features.ndim != 3 or features.shape[1] != height * width:
        raise ValueError('Point features must have T,H*W,D dimensions.')
    if not torch.isfinite(features).all() or not torch.isfinite(queries).all():
        raise ValueError('Features and point queries must be finite.')
    if targets.ndim != 3 or targets.shape[-1] != 2 or visible.shape != targets.shape[:-1]:
        raise ValueError('Point targets need Q,T,2 coordinates and Q,T visibility.')
    if not torch.isfinite(targets[visible]).all():
        raise ValueError('Visible target points must have finite coordinates.')
    if not args.thresholds or min(args.thresholds) <= 0:
        raise ValueError('Point-distance thresholds must be positive.')
    with torch.no_grad():
        predicted = track_points(features, queries, height, width)
        metrics = point_metrics(predicted, targets, visible, queries[:, 0], args.thresholds)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / 'tracks.npy', predicted.cpu().numpy(), allow_pickle=False)
    (destination / 'points.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--sequences')
    inputs.add_argument('--points')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gaps', nargs='+', type=int, default=[2, 32, 64, 128, 256, 512])
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--radius', type=int, default=12)
    parser.add_argument('--crossing-ratio', type=float, default=.9)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1, 2, 4, 8, 16])
    args = parser.parse_args(argv)
    if args.points:
        run_points(args)
        return
    if any(gap < 1 for gap in args.gaps) or not 0 < args.crossing_ratio <= 1:
        raise ValueError('Gaps must be positive and crossing ratio in (0,1].')
    device = cuda_device(args.device)
    rows = []
    for row in load_sequences(args.sequences):
        rows.extend(evaluate_gaps(row, sorted(set(args.gaps)), device, vars(args)))
    report = gap_summary(rows, args.crossing_ratio)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'gaps.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'pairs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps(report, indent=2))
