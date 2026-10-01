"""Convert explicit point arrays into processor-aware tracking benchmark manifests."""
import argparse
import json
from pathlib import Path

import numpy as np

from ropa_tools.data.bank import read_index, validate_array


def validate_points(arrays, frames):
    required = {'queries', 'targets', 'visible'}
    if not required <= arrays.keys():
        raise ValueError('The point archive needs queries, targets and visible.')
    queries, targets, visible = (arrays[key] for key in ('queries', 'targets', 'visible'))
    if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries):
        raise ValueError('Queries need Q,3 frame,y,x coordinates.')
    if targets.shape != (len(queries), frames, 2) or visible.shape != targets.shape[:2]:
        raise ValueError('Targets need Q,T,2 image x,y coordinates with Q,T visibility.')
    if visible.dtype != np.bool_:
        if not np.isin(visible, [0, 1]).all():
            raise ValueError('Visibility must contain only Boolean or zero/one entries.')
        visible = visible.astype(bool)
        arrays['visible'] = visible
    if not np.isfinite(queries).all() or not np.isfinite(targets[visible]).all():
        raise ValueError('Queries and visible targets must be finite.')
    times = queries[:, 0]
    if np.any(times != times.round()) or times.min() < 0 or times.max() >= frames:
        raise ValueError('Query frames must be integer feature-time indices.')
    if not visible[np.arange(len(queries)), times.astype(int)].all():
        raise ValueError('Every query point must be visible in its query frame.')
    if 'boxes' in arrays:
        boxes = arrays['boxes']
        if boxes.shape not in ((frames, 4), (len(queries), frames, 4)):
            raise ValueError('Boxes need T,4 or Q,T,4 coordinates.')
        if not np.isfinite(boxes).all() or (boxes[..., 2:] <= boxes[..., :2]).any():
            raise ValueError('Body boxes must be finite with positive dimensions.')
    if 'normalizers' in arrays:
        sizes = arrays['normalizers']
        if sizes.shape not in ((frames,), visible.shape) or not np.isfinite(sizes).all() or (sizes <= 0).any():
            raise ValueError('Body normalizers need positive T or Q,T values.')
    return arrays


def convert(index, annotations, destination):
    bank = {row['clip_id']: row for row in read_index(index)}
    annotations = Path(annotations).resolve()
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for number, line in enumerate(annotations.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = record['clip_id']
        if identifier not in bank or identifier in seen:
            raise ValueError(f'{annotations}:{number}: unknown or duplicate clip ID.')
        seen.add(identifier)
        row = bank[identifier]
        features = validate_array(row)
        source = Path(record['points'])
        source = source if source.is_absolute() else annotations.parent / source
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        if record.get('format') == 'tapvid':
            arrays = convert_tap_arrays(arrays, record['image_height'], record['image_width'],
                                        record.get('target_layout', 'qtx'),
                                        record.get('query_coordinates', 'pixels'),
                                        record.get('target_coordinates', 'pixels'))
        if record.get('frame_indices') is not None:
            arrays, alignment = align_feature_frames(
                arrays, record['frame_indices'], record.get('query_policy', 'exact'),
                record.get('maximum_query_distance', 0),
            )
        else:
            alignment = None
        if record.get('body_masks'):
            mask_path = Path(record['body_masks'])
            mask_path = mask_path if mask_path.is_absolute() else annotations.parent / mask_path
            masks = np.load(mask_path, allow_pickle=False)
            if record.get('frame_indices') is not None:
                masks = masks[np.asarray(record['frame_indices'])]
            arrays['boxes'] = mask_to_body_boxes(masks, record.get('foreground_ids'))
        if record.get('coordinates') == 'normalized' and record.get('format') != 'tapvid':
            arrays['targets'][..., 0] *= record['image_width']
            arrays['targets'][..., 1] *= record['image_height']
            arrays['queries'][:, 1] *= record['image_height']
            arrays['queries'][:, 2] *= record['image_width']
        elif record.get('format') != 'tapvid' and record.get('coordinates', 'pixels') != 'pixels':
            raise ValueError('Coordinates must be pixels or normalized fractions.')
        validate_points(arrays, len(features))
        name = record.get('name', identifier)
        if Path(name).name != name or name in {item['name'] for item in rows}:
            raise ValueError('Sequence names must be distinct plain filenames.')
        target = output / 'points' / (name + '.npz')
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        rows.append(dict(
            name=name, clip_id=identifier, features=row['features'], points=str(target),
            grid_height=row['grid_height'], grid_width=row['grid_width'],
            image_height=int(record['image_height']), image_width=int(record['image_width']),
            crop_size=row.get('crop_size', 384), resize_shorter=row.get('resize_shorter', 384),
            model_id=row.get('model_id'), alignment=alignment,
        ))
    if not rows:
        raise ValueError('No point annotation records were converted.')
    (output / 'sequences.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return dict(sequences=len(rows), manifest=str(output / 'sequences.jsonl'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.index, args.annotations, args.output)))


def align_feature_frames(arrays, frame_indices, query_policy='exact', maximum_query_distance=0):
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Feature frame indices must be a nonempty integer vector.')
    if indices.min() < 0 or np.any(np.diff(indices) <= 0):
        raise ValueError('Feature frame indices must be nonnegative and strictly increasing.')
    targets = arrays['targets']
    if targets.ndim != 3 or targets.shape[-1] != 2 or indices.max() >= targets.shape[1]:
        raise ValueError('A feature frame index lies outside the original point tracks.')
    queries = np.asarray(arrays['queries'], dtype=np.float64).copy()
    if query_policy not in ('exact', 'nearest') or maximum_query_distance < 0:
        raise ValueError('Use exact or nearest query alignment with a nonnegative distance limit.')
    query_frames = queries[:, 0]
    if np.any(query_frames != np.round(query_frames)):
        raise ValueError('Original query times must be integer frame indices.')
    distance = np.abs(query_frames[:, None] - indices[None])
    chosen = distance.argmin(-1)
    errors = distance[np.arange(len(queries)), chosen]
    allowed = 0 if query_policy == 'exact' else maximum_query_distance
    if np.any(errors > allowed):
        missing = np.where(errors > allowed)[0].tolist()
        raise ValueError(f'Query frames are not represented within the alignment tolerance: {missing}')
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    result['targets'] = arrays['targets'][:, indices]
    result['visible'] = arrays['visible'][:, indices]
    if query_policy == 'nearest':
        chosen_xy = result['targets'][np.arange(len(queries)), chosen]
        if not result['visible'][np.arange(len(queries)), chosen].all():
            raise ValueError('A nearest query frame contains an occluded target point.')
        queries[:, 1] = chosen_xy[:, 1]
        queries[:, 2] = chosen_xy[:, 0]
    queries[:, 0] = chosen
    result['queries'] = queries.astype(np.float32)
    for field in ('normalizers', 'boxes'):
        if field not in arrays:
            continue
        source = arrays[field]
        per_query = source.ndim == (3 if field == 'boxes' else 2)
        if per_query and source.shape[:2] == targets.shape[:2]:
            result[field] = source[:, indices]
        elif not per_query and source.shape[0] == targets.shape[1]:
            result[field] = source[indices]
        else:
            raise ValueError(f'{field} does not match the original point-track frame axis.')
    return result, dict(
        original_frames=targets.shape[1], feature_frames=len(indices),
        frame_indices=indices.tolist(), query_policy=query_policy,
        maximum_query_frame_distance=float(errors.max()),
        query_frame_distances=errors.tolist(),
    )


def mask_to_body_boxes(masks, foreground_ids=None):
    if masks.ndim != 3:
        raise ValueError('Body masks must have T,H,W dimensions.')
    boxes = []
    for frame, mask in enumerate(masks):
        foreground = mask != 0 if foreground_ids is None else np.isin(mask, foreground_ids)
        y, x = np.nonzero(foreground)
        if not len(x):
            raise ValueError(f'No body foreground is available in frame {frame}.')
        boxes.append([int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])
    return np.asarray(boxes, dtype=np.float32)


def convert_tap_arrays(source, image_height, image_width, target_layout='qtx',
                       query_coordinates='pixels', target_coordinates='pixels'):
    required = {'query_points', 'target_points', 'occluded'}
    if not required <= source.keys():
        raise ValueError('TAP input needs query_points, target_points and occluded arrays.')
    queries = np.asarray(source['query_points'], dtype=np.float32).copy()
    targets = np.asarray(source['target_points'], dtype=np.float32).copy()
    occluded = np.asarray(source['occluded'])
    if queries.ndim == 3 and queries.shape[0] == 1:
        queries = queries[0]
    if targets.ndim == 4 and targets.shape[0] == 1:
        targets = targets[0]
        occluded = occluded[0]
    if queries.ndim != 2 or queries.shape[-1] != 3 or targets.ndim != 3:
        raise ValueError('Convert one TAP sequence at a time, with optional singleton batch axes.')
    if target_layout == 'tqx':
        targets = targets.transpose(1, 0, 2)
        occluded = occluded.T
    elif target_layout != 'qtx':
        raise ValueError('TAP target layout must be qtx or tqx.')
    if min(image_height, image_width) < 1:
        raise ValueError('TAP source image dimensions must be positive.')
    conventions = {'pixels', 'normalized'}
    if query_coordinates not in conventions or target_coordinates not in conventions:
        raise ValueError('TAP query and target coordinates must be pixels or normalized fractions.')
    if target_coordinates == 'normalized':
        targets[..., 0] *= image_width
        targets[..., 1] *= image_height
    if query_coordinates == 'normalized':
        queries[:, 1] *= image_height
        queries[:, 2] *= image_width
    arrays = dict(queries=queries, targets=targets, visible=~occluded.astype(bool))
    return validate_points(arrays, targets.shape[1])
