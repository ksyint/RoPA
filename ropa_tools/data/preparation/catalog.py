"""Inspect clip manifests and keep recordings together across data partitions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess


VIDEO_SUFFIXES = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number}: invalid JSON') from error
        if not isinstance(record, dict) or not record.get('video'):
            raise ValueError(f'{path}:{number}: video path is required')
        video = Path(record['video']).expanduser()
        video = video if video.is_absolute() else path.parent / video
        record = dict(record, video=str(video.resolve()))
        start = float(record.get('start', 0))
        end = record.get('end')
        if start < 0 or (end is not None and float(end) <= start):
            raise ValueError(f'{path}:{number}: invalid clip interval')
        record['start'] = start
        if end is not None:
            record['end'] = float(end)
        record.setdefault('video_id', str(video.resolve()))
        records.append(record)
    if not records:
        raise ValueError(f'Empty clip manifest: {path}')
    return records


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    temporary.replace(path)


def clip_identity(record):
    payload = [record['video'], record['start'], record.get('end')]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def recording_groups(records, group_key='video_id'):
    groups = defaultdict(list)
    for record in records:
        key = str(record.get(group_key) or record['video'])
        groups[key].append(record)
    return groups


def probe_video(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    result = json.loads(result.stdout)
    if not result.get('streams'):
        raise ValueError(f'No video stream in {path}')
    stream = result['streams'][0]
    numerator, denominator = stream['r_frame_rate'].split('/')
    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'duration': float(result['format']['duration']),
        'fps': float(numerator) / max(float(denominator), 1),
    }


def inspect_manifest(records, inspect_media=False):
    missing, duplicates, invalid_bounds = [], [], []
    known, media = set(), {}
    sources = Counter()
    total_seconds = 0.0
    for row in records:
        path = Path(row['video'])
        key = clip_identity(row)
        if key in known:
            duplicates.append(key)
        known.add(key)
        sources[str(row.get('source', 'unspecified'))] += 1
        if not path.is_file():
            missing.append(str(path))
            continue
        if inspect_media and str(path) not in media:
            media[str(path)] = probe_video(path)
        end = row.get('end')
        if end is None and str(path) in media:
            end = media[str(path)]['duration']
        if end is not None:
            total_seconds += end - row['start']
            if str(path) in media and end > media[str(path)]['duration'] + .05:
                invalid_bounds.append(key)
    return {
        'clips': len(records), 'recordings': len(recording_groups(records)),
        'sources': dict(sources), 'duration_seconds': total_seconds,
        'missing': sorted(set(missing)), 'duplicate_clips': duplicates,
        'invalid_bounds': invalid_bounds, 'media': media,
    }


def exclude_overlap(records, holdouts, group_key='video_id'):
    held_groups = {str(row.get(group_key) or row['video']) for rows in holdouts for row in rows}
    held_paths = {row['video'] for rows in holdouts for row in rows}
    kept, removed = [], []
    for row in records:
        group = str(row.get(group_key) or row['video'])
        (removed if group in held_groups or row['video'] in held_paths else kept).append(row)
    return kept, removed


def perceptual_hash(frame):
    import numpy as np
    from PIL import Image
    pixels = np.asarray(Image.fromarray(frame).convert('L').resize((32, 32)), dtype=np.float64)
    positions = np.arange(32) + .5
    frequencies = np.arange(8)[:, None]
    transform = np.cos(np.pi / 32 * frequencies * positions)
    coefficients = (transform @ pixels @ transform.T).flatten()
    median = np.median(coefficients[1:])
    value = 0
    for bit in coefficients > median:
        value = (value << 1) | int(bit)
    return value


def clip_hashes(row, count=8):
    import av
    import numpy as np
    start = float(row.get('start', 0))
    end = float(row['end']) if row.get('end') is not None else probe_video(row['video'])['duration']
    if end <= start:
        raise ValueError('Perceptual screening requires a nonempty clip interval.')
    times = np.linspace(start, end, count, endpoint=False)
    hashes = []
    with av.open(row['video']) as video:
        stream = video.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for stamp in times:
            video.seek(int((stamp + origin) * av.time_base), backward=True)
            for frame in video.decode(video=0):
                if frame.time is not None and float(frame.time) - origin >= stamp:
                    hashes.append(perceptual_hash(frame.to_ndarray(format='rgb24')))
                    break
            else:
                raise ValueError(f'No timestamped frame could be sampled from {row["video"]}.')
    return hashes


class HashIndex:
    def __init__(self):
        self.root = None

    def add(self, value, identity):
        if self.root is None:
            self.root = [value, identity, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[2]:
                node[2][distance] = [value, identity, {}]
                return
            node = node[2][distance]

    def match(self, value, threshold):
        pending = [self.root] if self.root is not None else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                return node[1], distance
            pending.extend(child for edge, child in node[2].items() if distance - threshold <= edge <= distance + threshold)
        return None


def perceptual_exclusion(records, holdouts, threshold=8):
    if not 0 <= threshold <= 64:
        raise ValueError('Perceptual hash distance must be between zero and 64.')
    index = HashIndex()
    for rows in holdouts:
        for row in rows:
            for value in clip_hashes(row):
                index.add(value, clip_identity(row))
    kept, removed, matches = [], [], []
    for row in records:
        match = next((found for value in clip_hashes(row) if (found := index.match(value, threshold)) is not None), None)
        if match is None:
            kept.append(row)
        else:
            removed.append(row)
            matches.append({'clip_id': clip_identity(row), 'heldout_clip_id': match[0], 'hamming_distance': match[1]})
    return kept, removed, matches


def partition(records, fractions, seed, group_key):
    if set(fractions) != {'train', 'validation', 'test'}:
        raise ValueError('Use train, validation and test fractions.')
    if any(value <= 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError('Split fractions must be positive and sum to one.')
    groups = list(recording_groups(records, group_key).items())
    if len(groups) < 3:
        raise ValueError('Three independent recording groups are required.')
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    result = {name: [] for name in fractions}
    for index, (_, rows) in enumerate(groups):
        empty = [name for name, values in result.items() if not values]
        if len(groups) - index == len(empty):
            selected = empty[0]
        else:
            selected = min(result, key=lambda name: len(result[name]) / fractions[name])
        result[selected].extend(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heldout', action='append', default=[])
    parser.add_argument('--inspect-media', action='store_true')
    parser.add_argument('--perceptual', action='store_true')
    parser.add_argument('--hash-distance', type=int, default=8)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--group-key', default='video_id')
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--test-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    rows = read_manifest(args.manifest)
    excluded = []
    matches = []
    if args.heldout:
        holdouts = [read_manifest(p) for p in args.heldout]
        rows, excluded = exclude_overlap(rows, holdouts, args.group_key)
        if args.perceptual:
            rows, visual_duplicates, matches = perceptual_exclusion(rows, holdouts, args.hash_distance)
            excluded.extend(visual_duplicates)
    elif args.perceptual:
        parser.error('--perceptual requires at least one --heldout manifest.')
    if not rows:
        raise ValueError('No clips remain after held-out exclusion.')
    report = inspect_manifest(rows, args.inspect_media)
    report['excluded_clips'] = len(excluded)
    report['perceptual_matches'] = matches
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    write_manifest(destination / 'accepted.jsonl', rows)
    write_manifest(destination / 'excluded.jsonl', excluded)
    if args.split:
        fractions = {'validation': args.validation_fraction, 'test': args.test_fraction}
        fractions['train'] = 1 - sum(fractions.values())
        splits = partition(rows, fractions, args.seed, args.group_key)
        for name, values in splits.items():
            write_manifest(destination / f'{name}.jsonl', values)
        report['splits'] = {name: len(values) for name, values in splits.items()}
    (destination / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['missing'] or report['duplicate_clips'] or report['invalid_bounds']:
        raise SystemExit('Manifest audit found records requiring correction.')
