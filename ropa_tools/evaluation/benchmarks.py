"""Evaluate TAP-Vid and JHMDB tracking archives using frozen feature correspondences."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.backbone import cuda_device
from ropa_tools.evaluation.geometry import (
    FrameGeometry, feature_timestamps_from_record, interpolate_body_geometry, tracking_resolution,
)
from ropa_tools.evaluation.tracking import correspondence, cycle_errors
from ropa_tools.evaluation import tapvid, jhmdb


def read_records(path):
    path = Path(path).resolve()
    rows = []
    identifiers = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        name = row.get('name')
        if not name or name in identifiers or Path(name).name != name:
            raise ValueError(f'{path}:{number}: unique plain sequence name required.')
        identifiers.add(name)
        for key in ('features', 'points'):
            source = Path(row[key])
            row[key] = str(source if source.is_absolute() else path.parent / source)
        FrameGeometry.from_record(row)
        rows.append(row)
    if not rows:
        raise ValueError('The tracking benchmark manifest is empty.')
    return rows


def load_record(row, device, tracking_stride=None):
    geometry = FrameGeometry.from_record(row)
    features = np.load(row['features'], allow_pickle=False)
    with np.load(row['points'], allow_pickle=False) as source:
        required = {'queries', 'targets', 'visible'}
        if not required <= set(source.files):
            raise ValueError('Point archives require queries, targets and visible arrays.')
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    geometry, features = tracking_resolution(geometry, features, tracking_stride)
    queries = torch.as_tensor(arrays['queries'], dtype=torch.float32, device=device)
    targets = torch.as_tensor(arrays['targets'], dtype=torch.float32, device=device)
    visible = torch.as_tensor(arrays['visible'], dtype=torch.bool, device=device)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError('Queries must use Q,3 frame,y,x image coordinates.')
    if (('source_timestamps' not in arrays and targets.shape != (len(queries), len(features), 2))
            or visible.shape != targets.shape[:-1]):
        raise ValueError('Point targets need Q,T,2 image coordinates and matching Q,T visibility.')
    if 'source_timestamps' in arrays:
        from ropa_tools.evaluation.geometry import align_temporal_annotations
        aligned_record = dict(row, shape=list(features.shape))
        queries, targets, visible = align_temporal_annotations(aligned_record, queries, targets, visible, arrays['source_timestamps'])
        arrays = interpolate_body_geometry(arrays, arrays['source_timestamps'],
                                           feature_timestamps_from_record(aligned_record, device), device)
    grid, in_crop = geometry.to_grid(queries[:, [2, 1]])
    if not in_crop.all():
        raise ValueError('A query lies outside the processor crop. Prepare crop-visible query tracks.')
    grid_queries = torch.stack((queries[:, 0], grid[:, 1], grid[:, 0]), -1)
    grid_queries[:, 1].clamp_(0, geometry.grid_height - 1)
    grid_queries[:, 2].clamp_(0, geometry.grid_width - 1)
    return geometry, features, grid_queries, targets, visible, arrays


@torch.no_grad()
def run_sequence(row, args, device):
    geometry, features, queries, targets, visible, arrays = load_record(row, device, args.tracking_stride)
    tracks = correspondence(
        features, queries, geometry.grid_height, geometry.grid_width,
        args.query_batch, args.temperature, args.topk, args.visibility_threshold,
    )
    cycle_mask = None
    if args.cycle_threshold is not None:
        errors = cycle_errors(features, tracks['points'], queries[:, 0],
                              geometry.grid_height, geometry.grid_width, args.query_batch)
        cycle_mask = errors <= args.cycle_threshold
        tracks['visible'] &= cycle_mask
    prediction = geometry.to_image(tracks['points'])
    if args.task == 'tapvid':
        metrics = tapvid.evaluate_tracks(
            prediction, targets, tracks['visible'], visible, queries[:, 0],
            geometry.image_height, geometry.image_width, args.query_mode,
        )
        metrics['temporal_gaps'] = tapvid.temporal_breakdown(
            prediction, targets, tracks['visible'], visible, queries[:, 0],
            geometry.image_height, geometry.image_width, args.temporal_bins, args.query_mode)
        if args.visibility_sweep:
            metrics['visibility_sweep'] = tapvid.visibility_sweep(
                prediction, targets, tracks['confidence'], visible, queries[:, 0],
                geometry.image_height, geometry.image_width, args.visibility_sweep, args.query_mode, cycle_mask)
    else:
        if 'normalizers' in arrays:
            normalizers = torch.as_tensor(arrays['normalizers'], device=device, dtype=torch.float32)
        elif 'boxes' in arrays:
            normalizers = jhmdb.box_normalizers(torch.as_tensor(arrays['boxes'], device=device), args.normalization)
        else:
            raise ValueError('JHMDB point archives require normalizers or body boxes.')
        names = arrays['joint_names'].tolist() if 'joint_names' in arrays else None
        metrics = jhmdb.evaluate_joints(prediction, targets, visible, queries[:, 0], normalizers,
                                       args.pck, args.query_mode, names)
    return metrics, dict(points=prediction.cpu().numpy(), visible=tracks['visible'].cpu().numpy(),
                         confidence=tracks['confidence'].cpu().numpy(), entropy=tracks['entropy'].cpu().numpy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('tapvid', 'jhmdb'))
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--query-mode', choices=('first', 'strided'), default='strided')
    parser.add_argument('--query-batch', type=int, default=64)
    parser.add_argument('--tracking-stride', type=int, help='Resample feature maps to this crop-pixel stride')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--visibility-threshold', type=float)
    parser.add_argument('--visibility-sweep', type=float, nargs='+')
    parser.add_argument('--temporal-bins', type=int, nargs='+', default=[1, 4, 16, 64, 256])
    parser.add_argument('--cycle-threshold', type=float)
    parser.add_argument('--normalization', choices=('max-side', 'diagonal'), default='max-side')
    parser.add_argument('--pck', type=float, nargs='+', default=[.1, .2])
    args = parser.parse_args(argv)
    if args.cycle_threshold is not None and args.cycle_threshold < 0:
        parser.error('Cycle threshold is a nonnegative distance in patch-grid coordinates.')
    device = cuda_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for row in read_records(args.sequences):
        metrics, tracks = run_sequence(row, args, device)
        results[row['name']] = metrics
        np.savez_compressed(output / (row['name'] + '.npz'), **tracks)
    aggregate = tapvid.aggregate_sequences if args.task == 'tapvid' else jhmdb.aggregate_sequences
    report = dict(task=args.task, sequences=results, summary=aggregate(list(results.values())),
                  options=vars(args))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
