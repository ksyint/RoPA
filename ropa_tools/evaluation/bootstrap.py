"""Sequence-level paired confidence intervals without treating video frames as independent."""
import argparse
import json
import math
from pathlib import Path

import torch

from models.backbone import cuda_device


def metric_value(row, path):
    current = row
    for field in path.split('.'):
        if not isinstance(current, dict) or field not in current:
            raise ValueError(f'Metric {path!r} is absent from a sequence result.')
        current = current[field]
    if type(current) not in (int, float) or not math.isfinite(current):
        raise ValueError(f'Metric {path!r} must be a finite number.')
    return float(current)


def read_results(path, metric):
    data = json.loads(Path(path).read_text())
    rows = data.get('sequences')
    if not isinstance(rows, dict) or not rows:
        raise ValueError('The result file needs a nonempty sequence-keyed mapping.')
    return {name: metric_value(row, metric) for name, row in rows.items()}


@torch.no_grad()
def bootstrap(values, device, repetitions=10000, confidence=.95, seed=42, chunk=256):
    if repetitions < 2 or chunk < 1 or not 0 < confidence < 1:
        raise ValueError('Bootstrap count, chunk size and confidence level are invalid.')
    values = torch.as_tensor(values, device=device, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError('At least two finite independent sequence values are required.')
    generator = torch.Generator(device=device).manual_seed(seed)
    means = []
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        indices = torch.randint(len(values), (size, len(values)), device=device, generator=generator)
        means.append(values[indices].mean(-1))
    sampled = torch.cat(means)
    tail = (1 - confidence) / 2
    bounds = torch.quantile(sampled, sampled.new_tensor([tail, 1 - tail]))
    return dict(
        units=len(values), mean=float(values.mean()),
        sample_std=float(values.std(unbiased=True)),
        confidence=confidence, lower=float(bounds[0]), upper=float(bounds[1]),
        repetitions=repetitions, seed=seed, resampling_unit='sequence',
    )


def paired(first, second, device, **options):
    if first.keys() != second.keys():
        raise ValueError('Paired comparison requires identical sequence identifiers.')
    names = sorted(first)
    differences = [second[name] - first[name] for name in names]
    result = bootstrap(differences, device, **options)
    result['direction'] = 'second minus first'
    result['improved_sequences'] = sum(value > 0 for value in differences)
    result['unchanged_sequences'] = sum(value == 0 for value in differences)
    result['degraded_sequences'] = sum(value < 0 for value in differences)
    result['sequence_differences'] = dict(zip(names, differences))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True)
    parser.add_argument('--second')
    parser.add_argument('--metric', default='jf')
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    first = read_results(args.first, args.metric)
    options = dict(repetitions=args.repetitions, confidence=args.confidence, seed=args.seed)
    report = dict(metric=args.metric, first=bootstrap(list(first.values()), device, **options))
    if args.second:
        second = read_results(args.second, args.metric)
        report['second'] = bootstrap(list(second.values()), device, **options)
        report['paired'] = paired(first, second, device, **options)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
