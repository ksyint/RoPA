"""Create video-disjoint train/validation manifests from an acquired video folder."""
import argparse
import json
from pathlib import Path
import random
import subprocess


def main(args):
    root = Path(args.videos).resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.mp4', '.mkv', '.mov', '.avi', '.webm'})
    if len(paths) < 2 or not 0 < args.validation_fraction < 1 or args.clip_seconds <= 0:
        raise ValueError('Provide at least two videos, 0 < validation_fraction < 1 and positive clip duration.')
    random.Random(args.seed).shuffle(paths)
    count = min(len(paths) - 1, max(1, round(len(paths) * args.validation_fraction)))
    validation = set(paths[:count])
    splits = {'train': [], 'validation': []}
    for path in paths:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'json', str(path)], check=True, capture_output=True, text=True)
        duration = float(json.loads(result.stdout)['format']['duration'])
        split = 'validation' if path in validation else 'train'
        start = 0.0
        while start + args.clip_seconds <= duration + 1e-6:
            splits[split].append({'video': str(path), 'video_id': path.relative_to(root).as_posix(),
                                  'start': start, 'end': min(duration, start + args.clip_seconds)})
            start += args.clip_seconds
    if any(not rows for rows in splits.values()):
        raise ValueError('Both splits need full-length clips; lower --clip-seconds or add longer videos.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({name: len(rows) for name, rows in splits.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True)
    parser.add_argument('--output', default='data')
    parser.add_argument('--clip-seconds', type=float, default=2.0)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
