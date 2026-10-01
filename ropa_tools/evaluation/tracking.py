"""Batched frozen-feature tracking with explicit visibility and query conventions."""
import torch
import torch.nn.functional as F

from ropa_tools.evaluation.gaps import sample_query_features


def validate_tracking(features, queries, height, width):
    if features.device.type != 'cuda' or queries.device != features.device:
        raise ValueError('Tracking features and queries must share one CUDA device.')
    if features.ndim != 3 or features.shape[1] != height * width:
        raise ValueError('Tracking features need T,H*W,D dimensions.')
    if queries.ndim != 2 or queries.shape[-1] != 3 or not len(queries):
        raise ValueError('Tracking queries need nonempty Q,3 frame,y,x entries.')
    if not torch.isfinite(features).all() or not torch.isfinite(queries).all():
        raise ValueError('Tracking features and queries must be finite.')
    if not torch.equal(queries[:, 0], queries[:, 0].round()):
        raise ValueError('Query frame indices must be integral.')
    if queries[:, 0].min() < 0 or queries[:, 0].max() >= len(features):
        raise ValueError('Query frame indices are outside the sequence.')


@torch.no_grad()
def correspondence(features, queries, height, width, query_batch=64,
                   temperature=.07, topk=1, confidence_threshold=None):
    validate_tracking(features, queries, height, width)
    if query_batch < 1 or topk < 1 or topk > height * width or temperature <= 0:
        raise ValueError('Tracking chunk, top-k and temperature values are invalid.')
    if confidence_threshold is not None and not -1 <= confidence_threshold <= 1:
        raise ValueError('Cosine visibility threshold must lie in [-1,1].')
    normalized = F.normalize(features.float(), dim=-1)
    positions, similarities, entropies = [], [], []
    for start in range(0, len(queries), query_batch):
        selected = queries[start:start + query_batch]
        anchors = F.normalize(sample_query_features(features, selected, height, width).float(), dim=-1)
        scores = torch.einsum('qd,tnd->qtn', anchors, normalized)
        best, indices = scores.topk(topk, -1)
        weights = (best / temperature).softmax(-1)
        x = (indices % width).float()
        y = torch.div(indices, width, rounding_mode='floor').float()
        predicted = torch.stack(((weights * x).sum(-1), (weights * y).sum(-1)), -1)
        frame = selected[:, 0].long()
        index = torch.arange(len(selected), device=features.device)
        predicted[index, frame] = selected[:, [2, 1]]
        maximum = best[..., 0]
        probability = (scores / temperature).softmax(-1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
        positions.append(predicted)
        similarities.append(maximum)
        entropies.append(entropy)
    points = torch.cat(positions)
    confidence = torch.cat(similarities)
    visible = torch.ones_like(confidence, dtype=torch.bool)
    if confidence_threshold is not None:
        visible = confidence >= confidence_threshold
    return dict(points=points, confidence=confidence, visible=visible, entropy=torch.cat(entropies))


@torch.no_grad()
def cycle_errors(features, tracks, query_frames, height, width, query_batch=64):
    if tracks.ndim != 3 or tracks.shape[1:] != (len(features), 2):
        raise ValueError('Cycle tracks must have Q,T,2 dimensions.')
    if query_frames.shape != tracks.shape[:1]:
        raise ValueError('One source frame is required for each track.')
    tokens = F.normalize(features.float(), dim=-1)
    result = tracks.new_empty(tracks.shape[:2])
    for frame in range(len(features)):
        for start in range(0, len(tracks), query_batch):
            selected = tracks[start:start + query_batch, frame]
            queries = torch.cat((selected.new_full((len(selected), 1), frame), selected[:, [1, 0]]), -1)
            anchors = F.normalize(sample_query_features(features, queries, height, width).float(), dim=-1)
            origins = query_frames[start:start + len(selected)].long()
            reference = tokens[origins]
            matches = torch.einsum('qd,qnd->qn', anchors, reference).argmax(-1)
            returned = torch.stack((matches % width, matches // width), -1).float()
            target = tracks[start:start + len(selected)][torch.arange(len(selected), device=tracks.device), origins]
            result[start:start + len(selected), frame] = (returned - target).norm(dim=-1)
    return result


def query_evaluation_mask(query_frames, frame_count, mode):
    if mode not in ('first', 'strided'):
        raise ValueError('Query evaluation mode must be first or strided.')
    frame = torch.arange(frame_count, device=query_frames.device)[None]
    queries = query_frames.long()[:, None]
    return frame > queries if mode == 'first' else frame != queries
