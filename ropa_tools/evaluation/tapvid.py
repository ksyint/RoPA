"""Point and occlusion scores at the 256-pixel TAP-Vid evaluation scale."""
import torch

from ropa_tools.evaluation.tracking import query_evaluation_mask


def evaluate_tracks(predicted, target, predicted_visible, visible, query_frames,
                    image_height, image_width, mode='strided', thresholds=(1, 2, 4, 8, 16), observation_mask=None):
    if predicted.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError('TAP-Vid point arrays need matching Q,T,2 dimensions.')
    if visible.shape != target.shape[:2] or predicted_visible.shape != visible.shape:
        raise ValueError('Visibility arrays must have Q,T dimensions.')
    if query_frames.shape != target.shape[:1]:
        raise ValueError('Supply one query frame for every point track.')
    if image_height < 1 or image_width < 1 or not thresholds or min(thresholds) <= 0:
        raise ValueError('Image sizes and point-distance thresholds must be positive.')
    if not torch.isfinite(predicted).all() or not torch.isfinite(target[visible]).all():
        raise ValueError('Predictions and visible reference points must be finite.')
    selected = query_evaluation_mask(query_frames, target.shape[1], mode)
    if observation_mask is not None:
        if observation_mask.shape != selected.shape:
            raise ValueError('The observation mask must have Q,T dimensions.')
        selected &= observation_mask.bool()
    if not selected.any():
        raise ValueError('No frames remain after applying the query evaluation mask.')
    visibility = visible.bool()
    predicted_visibility = predicted_visible.bool()
    scale = predicted.new_tensor([256 / image_width, 256 / image_height])
    distance = ((predicted - target) * scale).square().sum(-1).sqrt()
    reference = visibility & selected
    target_count = reference.sum(-1)
    occlusion_count = selected.sum(-1)
    occlusion_correct = ((visibility == predicted_visibility) & selected).sum(-1)
    points, jaccards, details = [], [], {}
    for threshold in thresholds:
        within = distance < threshold
        correct = within & reference
        true_positive = correct & predicted_visibility
        false_positive = (~visibility | ~within) & predicted_visibility & selected
        union = target_count + false_positive.sum(-1)
        per_track_pck = correct.sum(-1).float() / target_count.clamp_min(1)
        per_track_jaccard = true_positive.sum(-1).float() / union.clamp_min(1)
        visible_total = int(target_count.sum())
        union_total = int(union.sum())
        pck = float(correct.sum()) / visible_total if visible_total else None
        jaccard = float(true_positive.sum()) / union_total if union_total else None
        if pck is not None:
            points.append(pck)
        if jaccard is not None:
            jaccards.append(jaccard)
        details[str(threshold)] = dict(
            points_within=pck,
            jaccard=jaccard,
            per_track_points_within=per_track_pck.tolist(),
            per_track_jaccard=per_track_jaccard.tolist(),
            visible_observations=target_count.tolist(),
            union_observations=union.tolist(),
            true_positives=true_positive.sum(-1).tolist(),
            false_positives=false_positive.sum(-1).tolist(),
        )
    eligible = occlusion_count > 0
    return dict(
        tracks=len(target),
        evaluated_tracks=int(eligible.sum()),
        evaluated_observations=int(selected.sum()),
        visible_observations=int(reference.sum()),
        query_mode=mode,
        coordinate_scale=256,
        average_pts_within_thresh=sum(points) / len(points) if points else None,
        average_jaccard=sum(jaccards) / len(jaccards) if jaccards else None,
        occlusion_accuracy=float(occlusion_correct.sum()) / int(occlusion_count.sum()),
        thresholds=details,
    )


def aggregate_sequences(records):
    if not records:
        raise ValueError('No TAP-Vid sequence results are available.')
    keys = ('average_pts_within_thresh', 'average_jaccard', 'occlusion_accuracy')
    summary = dict(sequences=len(records), tracks=sum(row['tracks'] for row in records))
    for key in keys:
        values = [row[key] for row in records if row[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    summary['aggregation'] = 'equal weight per sequence, pooled query-frame observations within each sequence'
    return summary


@torch.no_grad()
def temporal_breakdown(predicted, target, predicted_visible, visible, query_frames,
                       image_height, image_width, boundaries, mode='strided'):
    if not boundaries or min(boundaries) < 1 or list(boundaries) != sorted(set(boundaries)):
        raise ValueError('Temporal distance boundaries must be distinct increasing positive integers.')
    frame = torch.arange(target.shape[1], device=target.device)
    distance = (frame[None] - query_frames.long()[:, None]).abs()
    evaluated = query_evaluation_mask(query_frames, target.shape[1], mode)
    lower = 1
    result = []
    intervals = list(boundaries)
    if intervals[-1] < target.shape[1] - 1:
        intervals.append(target.shape[1] - 1)
    for upper in intervals:
        selected = (distance >= lower) & (distance <= upper) & evaluated
        if selected.any():
            scores = evaluate_tracks(predicted, target, predicted_visible, visible, query_frames,
                                     image_height, image_width, mode, observation_mask=selected)
            scores.pop('thresholds')
            result.append(dict(minimum_gap=lower, maximum_gap=upper, **scores))
        lower = upper + 1
    return result


@torch.no_grad()
def visibility_sweep(predicted, target, confidence, visible, query_frames,
                     image_height, image_width, thresholds, mode='strided', cycle_mask=None):
    if not thresholds or any(not -1 <= value <= 1 for value in thresholds):
        raise ValueError('Visibility thresholds must lie in [-1,1].')
    selected = query_evaluation_mask(query_frames, target.shape[1], mode)
    output = []
    for threshold in sorted(set(thresholds)):
        prediction = confidence >= threshold
        if cycle_mask is not None:
            prediction &= cycle_mask
        metrics = evaluate_tracks(predicted, target, prediction, visible, query_frames,
                                  image_height, image_width, mode)
        metrics.pop('thresholds')
        true_positive = int((prediction & visible & selected).sum())
        predicted_positive = int((prediction & selected).sum())
        reference_positive = int((visible & selected).sum())
        output.append(dict(threshold=threshold, visible_precision=true_positive / predicted_positive if predicted_positive else None,
                           visible_recall=true_positive / reference_positive if reference_positive else None, **metrics))
    return output
