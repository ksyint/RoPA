import torch
import torch.nn.functional as F


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
