"""Collect compatible frozen-transfer results across correspondence and classification tasks."""
import argparse
import csv
import json
import math
from pathlib import Path


TASK_METRICS = {
    'davis': ('region_iou', 'boundary_f', 'jf'),
    'tapvid': ('average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy'),
    'jhmdb': ('pck', 'mean_normalized_error'),
    'vspw': ('mean_iou', 'pixel_accuracy'),
    'kinetics400': ('top1', 'top5'),
    'ssv2': ('top1', 'top5'),
    'diving48': ('top1', 'top5'),
}


def flatten_metrics(values, prefix=''):
    result = {}
    for key, value in values.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metrics(value, name))
        elif type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite transfer metric: {name}')
            result[name] = value
    return result


def read_transfer(expression):
    if '=' not in expression:
        raise ValueError('A transfer result uses task=metrics.json notation.')
    task, filename = expression.split('=', 1)
    if task not in TASK_METRICS:
        raise ValueError(f'Unknown transfer task {task}.')
    path = Path(filename).resolve()
    data = json.loads(path.read_text())
    values = data.get('summary', data)
    if not isinstance(values, dict):
        raise ValueError('Transfer result must contain one completed evaluation summary.')
    selected = {}
    for key in TASK_METRICS[task]:
        if key not in values:
            raise ValueError(f'Transfer result for {task} has no {key} metric.')
        selected[key] = values[key]
    return dict(task=task, path=str(path), metrics=flatten_metrics(selected),
                samples=values.get('samples'), sequences=values.get('sequences'))


def aggregate_runs(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row['task'], []).append(row)
    result = {}
    for task, rows in sorted(grouped.items()):
        keys = rows[0]['metrics'].keys()
        if any(row['metrics'].keys() != keys for row in rows):
            raise ValueError(f'Transfer runs for {task} do not contain identical metric keys.')
        metrics = {}
        for key in keys:
            values = [row['metrics'][key] for row in rows]
            mean = sum(values) / len(values)
            metrics[key] = dict(
                mean=mean,
                std=math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
                minimum=min(values), maximum=max(values), runs=len(values),
            )
        result[task] = metrics
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--csv')
    args = parser.parse_args(argv)
    records = [read_transfer(value) for value in args.result]
    if len({(row['task'], row['path']) for row in records}) != len(records):
        parser.error('A result file cannot be counted twice for the same task.')
    summary = aggregate_runs(records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(runs=records, summary=summary), indent=2) + '\n')
    if args.csv:
        destination = Path(args.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['task', 'metric', 'mean', 'std', 'minimum', 'maximum', 'runs'])
            writer.writeheader()
            for task, metrics in summary.items():
                for metric, values in metrics.items():
                    writer.writerow(dict(task=task, metric=metric, **values))
    print(json.dumps(summary, indent=2))
