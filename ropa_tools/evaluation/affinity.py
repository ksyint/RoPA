"""Gap-resolved feature affinity, rank and round-trip correspondence diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import cuda_device
from ropa_tools.data.bank import read_index, validate_array


@torch.no_grad()
def pair_diagnostics(source, target, height, width, temperature=.07, chunk=256):
    if source.shape != target.shape or source.ndim != 2 or len(source) != height * width:
        raise ValueError('Paired feature maps must share H*W,D dimensions.')
    if temperature <= 0 or chunk < 1:
        raise ValueError('Affinity temperature and query chunk must be positive.')
    source = F.normalize(source.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError('Paired feature maps must contain finite values.')
    forward, entropy, margin, similarity = [], [], [], []
    for start in range(0, len(source), chunk):
        logits = source[start:start + chunk] @ target.T
        best, positions = logits.topk(min(2, len(target)), -1)
        probability = (logits / temperature).softmax(-1)
        forward.append(positions[:, 0])
        entropy.append(-(probability * probability.clamp_min(1e-12).log()).sum(-1))
        margin.append(best[:, 0] - best[:, 1] if best.shape[1] == 2 else best[:, 0])
        similarity.append(best[:, 0])
    forward = torch.cat(forward)
    reverse = []
    for start in range(0, len(target), chunk):
        reverse.append((target[start:start + chunk] @ source.T).argmax(-1))
    reverse = torch.cat(reverse)
    index = torch.arange(len(source), device=source.device)
    returned = reverse[forward]
    displacement = torch.stack((forward % width - index % width, forward // width - index // width), -1).float().norm(dim=-1)
    cycle = torch.stack((returned % width - index % width, returned // width - index // width), -1).float().norm(dim=-1)
    assigned = torch.bincount(forward, minlength=len(target))
    return dict(
        patches=len(source),
        mean_best_cosine=float(torch.cat(similarity).mean()),
        mean_top2_margin=float(torch.cat(margin).mean()),
        mean_entropy=float(torch.cat(entropy).mean()),
        mean_displacement=float(displacement.mean()),
        mutual_match_fraction=float((returned == index).float().mean()),
        mean_cycle_error=float(cycle.mean()),
        target_coverage=float((assigned > 0).float().mean()),
        maximum_target_fanout=int(assigned.max()),
    )


@torch.no_grad()
def effective_rank(features, maximum_tokens=4096):
    values = features.flatten(0, 1).float()
    if maximum_tokens < 2:
        raise ValueError('Rank diagnostics need at least two sampled tokens.')
    if len(values) > maximum_tokens:
        chosen = torch.linspace(0, len(values) - 1, maximum_tokens, device=values.device).round().long()
        values = values[chosen]
    values = values - values.mean(0)
    singular = torch.linalg.svdvals(values)
    energy = singular.square()
    if not energy.sum():
        return dict(tokens=len(values), dimension=values.shape[-1], effective_rank=0., stable_rank=0.)
    probability = energy / energy.sum()
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return dict(
        tokens=len(values), dimension=values.shape[-1],
        effective_rank=float(entropy.exp()),
        stable_rank=float(energy.sum() / energy.max()),
        leading_energy_fraction=float(probability[0]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--gaps', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--pair-stride', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--rank-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not args.gaps or min(args.gaps) < 1 or args.pair_stride < 1:
        parser.error('Temporal gaps and pair stride must be positive.')
    device = cuda_device(args.device)
    records = []
    for row in read_index(args.index):
        values = torch.from_numpy(np.array(validate_array(row), copy=True)).to(device)
        pairs = []
        for gap in sorted(set(args.gaps)):
            for start in range(0, len(values) - gap, args.pair_stride):
                report = pair_diagnostics(values[start], values[start + gap], row['grid_height'],
                                          row['grid_width'], args.temperature, args.chunk_size)
                pairs.append(dict(source=start, target=start + gap, gap=gap, **report))
        records.append(dict(clip_id=row['clip_id'], rank=effective_rank(values, args.rank_tokens), pairs=pairs))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(options=vars(args), clips=records), indent=2) + '\n')
