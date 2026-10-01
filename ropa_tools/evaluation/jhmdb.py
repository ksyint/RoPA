"""Visible-joint PCK with an explicit per-frame body-size normalization."""
import torch

from ropa_tools.evaluation.tracking import query_evaluation_mask


def box_normalizers(boxes, convention='max-side'):
    if boxes.ndim not in (2, 3) or boxes.shape[-1] != 4:
        raise ValueError('Body boxes use T,4 or Q,T,4 x1,y1,x2,y2 coordinates.')
    if not torch.isfinite(boxes).all():
        raise ValueError('Body boxes must be finite.')
    size = boxes[..., 2:] - boxes[..., :2]
    if (size <= 0).any():
        raise ValueError('Body boxes must have positive width and height.')
    if convention == 'max-side':
        return size.max(-1).values
    if convention == 'diagonal':
        return size.square().sum(-1).sqrt()
    raise ValueError('Body-size normalization must be max-side or diagonal.')


def evaluate_joints(predicted, target, visible, query_frames, normalizers,
                    fractions=(.1, .2), mode='strided', joint_names=None):
    if predicted.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError('Joint coordinates must have matching Q,T,2 dimensions.')
    if visible.shape != target.shape[:2]:
        raise ValueError('Joint visibility must have Q,T dimensions.')
    if normalizers.ndim == 1:
        normalizers = normalizers[None].expand(len(target), -1)
    if normalizers.shape != visible.shape or not torch.isfinite(normalizers).all() or (normalizers <= 0).any():
        raise ValueError('Body sizes need positive T or Q,T measurements.')
    if not fractions or min(fractions) <= 0 or max(fractions) > 1:
        raise ValueError('PCK thresholds must be fractions in (0,1].')
    valid = visible.bool() & query_evaluation_mask(query_frames, target.shape[1], mode)
    if not valid.any():
        raise ValueError('No visible non-query joint observations remain.')
    if not torch.isfinite(predicted).all() or not torch.isfinite(target[visible]).all():
        raise ValueError('Joint predictions and visible targets must be finite.')
    errors = (predicted - target).norm(dim=-1) / normalizers
    names = joint_names or [str(index) for index in range(len(target))]
    if len(names) != len(target) or len(set(names)) != len(names):
        raise ValueError('Provide one distinct name per tracked joint.')
    rows = []
    for index, name in enumerate(names):
        selected = valid[index]
        count = int(selected.sum())
        rows.append(dict(
            joint=name,
            observations=count,
            normalized_error=float(errors[index][selected].mean()) if count else None,
            pck={str(threshold): float((errors[index][selected] <= threshold).float().mean()) if count else None
                 for threshold in fractions},
        ))
    return dict(
        observations=int(valid.sum()),
        joints=len(target),
        mean_normalized_error=float(errors[valid].mean()),
        pck={str(threshold): float((errors[valid] <= threshold).float().mean()) for threshold in fractions},
        per_joint=rows,
        query_mode=mode,
    )


def aggregate_sequences(records):
    if not records:
        raise ValueError('No JHMDB sequence results are available.')
    thresholds = records[0]['pck'].keys()
    if any(row['pck'].keys() != thresholds for row in records):
        raise ValueError('JHMDB runs use different PCK thresholds.')
    count = sum(row['observations'] for row in records)
    return dict(
        sequences=len(records),
        observations=count,
        mean_normalized_error=sum(row['mean_normalized_error'] * row['observations'] for row in records) / count,
        pck={key: sum(row['pck'][key] * row['observations'] for row in records) / count for key in thresholds},
        aggregation='visible observation weighted',
    )
