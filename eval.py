import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from utils.propagation import propagate_labels
from utils.rope import band_diagnostics, temporal_frequencies


def main(args):
    if not args.data:
        frequencies = temporal_frequencies(16, args.target_range, 1)
        diagnostics = band_diagnostics(frequencies, [1, 2, 32, 64, 128, 256, 512])
        diagnostics['displacement'] = diagnostics['displacement'].tolist()
        print(json.dumps(diagnostics, indent=2))
        return
    with np.load(args.data, allow_pickle=False) as data:
        features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['labels']).long()  # T,H,W
    if labels.min() < 0:
        raise ValueError('Labels must be nonnegative class IDs; remap ignore labels before evaluation.')
    t, h, w = labels.shape
    classes = int(labels.max()) + 1
    first = F.one_hot(labels[0].flatten(), classes).float()
    result = propagate_labels(features, first, h, w, topk=args.topk).argmax(-1).reshape(t, h, w)
    ious = []
    for frame in range(1, t):
        for category in range(1, classes):
            a, b = result[frame] == category, labels[frame] == category
            union = (a | b).sum().item()
            if union:
                ious.append((a & b).sum().item() / union)
    print(json.dumps({'foreground_mean_iou': float(np.mean(ious)) if ious else None,
                      'pixel_accuracy': (result[1:] == labels[1:]).float().mean().item(),
                      'note': 'Patch-grid evaluation; not the official DAVIS J&F evaluator.'}, indent=2))
    if args.output:
        np.save(args.output, result.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='.npz with features T,H*W,D and labels T,H,W')
    parser.add_argument('--target_range', type=float, default=64)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--output')
    main(parser.parse_args())
