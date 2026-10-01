"""Create indexed feature archives with explicit clip and patch-grid metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.backbone import create_model, cuda_device, load_checkpoint_model
from ropa_tools.data.loading import VideoManifest
from ropa_tools.data.preparation.catalog import clip_identity, read_manifest, write_manifest
from ropa_tools.temporal import read_recipe


def file_digest(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def read_index(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('The feature-bank index is empty.')
    seen = set()
    for row in rows:
        required = {'clip_id', 'features', 'shape', 'grid_height', 'grid_width'}
        if not required <= row.keys():
            raise ValueError(f'Feature row needs {sorted(required)}')
        if row['clip_id'] in seen:
            raise ValueError(f'Duplicate feature clip: {row["clip_id"]}')
        seen.add(row['clip_id'])
        file = Path(row['features'])
        row['features'] = str(file if file.is_absolute() else path.parent / file)
    return rows


def validate_array(row, check_hash=False):
    path = Path(row['features'])
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.ndim != 3 or list(array.shape) != row['shape']:
        raise ValueError(f'Feature dimensions differ from index: {path}')
    if array.shape[1] != row['grid_height'] * row['grid_width']:
        raise ValueError(f'Patch grid does not match token count: {path}')
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f'Feature archive needs floating point values: {path}')
    for start in range(0, len(array), 16):
        if not np.isfinite(array[start:start + 16]).all():
            raise ValueError(f'Nonfinite feature values: {path}')
    if check_hash and row.get('sha256') != file_digest(path):
        raise ValueError(f'Feature checksum differs: {path}')
    return array


def save_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def resolve_model(args):
    device = cuda_device(args.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        config = state['config']
    else:
        state = None
        config = read_recipe(args.config)
    for option in ('pretrained', 'cache_dir'):
        value = getattr(args, option)
        if value is not None:
            config['model'][option] = value
    if args.offline:
        config['model']['local_files_only'] = True
    model = load_checkpoint_model(state) if state is not None else create_model(config)
    return model.to(device).eval(), config, device


def dictionary_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extraction_contract(model, config, dataset, checkpoint):
    if checkpoint:
        weights = {'checkpoint_sha256': file_digest(checkpoint)}
    else:
        source = Path(config['model']['pretrained'])
        if source.is_dir():
            files = sorted(path for path in source.rglob('*')
                           if path.is_file() and path.suffix in {'.bin', '.safetensors', '.json'})
            if not any(path.suffix in {'.bin', '.safetensors'} for path in files):
                raise ValueError(f'No pretrained weight files found in {source}')
            weights = {str(path.relative_to(source)): file_digest(path) for path in files}
        else:
            commit = getattr(model.backbone.config, '_commit_hash', None)
            if not commit:
                raise ValueError('Feature extraction needs a resolved model revision or local checkpoint.')
            weights = {'pretrained': config['model']['pretrained'], 'commit': commit}
    model_id = dictionary_digest({'weights': weights, 'model': config['model']})
    contract = {'format': 2, 'model_id': model_id, 'frames': dataset.frames,
                'sampling': 'presentation-time-clip-uniform-rounded-rgb24-v1',
                'processor': dataset.processor.to_dict(), 'compute_dtype': 'bfloat16',
                'storage_dtype': 'float32', 'crop_size': model.backbone.config.crop_size,
                'patch_size': model.backbone.config.patch_size,
                'tubelet_size': model.backbone.config.tubelet_size}
    return json.loads(json.dumps(contract))


def source_signature(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def plan_extraction(rows, existing, contract):
    sources, prepared = {}, []
    contract_sha256 = dictionary_digest(contract)
    for row in rows:
        video = row['video']
        if video not in sources:
            before = source_signature(video)
            fingerprint = file_digest(video)
            if before != source_signature(video):
                raise ValueError(f'Source video changed while fingerprinting: {video}')
            sources[video] = (fingerprint, before)
        fingerprint, signature = sources[video]
        identity = clip_identity(row)
        cache_key = dictionary_digest({'clip_id': identity, 'source_sha256': fingerprint,
                                       'contract_sha256': contract_sha256})
        previous = existing.get(identity)
        if previous is not None:
            if (previous.get('cache_key') != cache_key or previous.get('contract') != contract
                    or previous.get('source_sha256') != fingerprint):
                raise ValueError(f'Source or extraction contract changed for {video}. Use a new output directory.')
            validate_array(previous, check_hash=True)
        prepared.append((row, identity, cache_key, fingerprint, signature, previous))
    return prepared


def extract(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model, config, device = resolve_model(args)
    rows = read_manifest(args.manifest)
    dataset = VideoManifest(args.manifest, config['model'], config.get('data', {}).get('frames', 16))
    processor_directory = root / 'processor'
    geometry = dataset.processor.to_dict()
    index_path = root / 'index.jsonl'
    existing = {row['clip_id']: row for row in read_index(index_path)} if args.resume and index_path.exists() else {}
    contract = extraction_contract(model, config, dataset, args.checkpoint)
    prepared = plan_extraction(rows, existing, contract)
    dataset.processor.save_pretrained(processor_directory)
    indexed = []
    try:
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for number, (row, identity, cache_key, fingerprint, signature, previous) in enumerate(prepared):
                if signature != source_signature(row['video']):
                    raise ValueError(f'Source video changed during extraction: {row["video"]}')
                if previous is not None:
                    indexed.append(previous)
                    continue
                pixels = dataset[number][None].to(device)
                if signature != source_signature(row['video']):
                    raise ValueError(f'Source video changed during decoding: {row["video"]}')
                values = model(pixels)[0].float().cpu().numpy()
                filename = root / 'features' / f'{cache_key}.npy'
                save_array(filename, values)
                crop = model.backbone.config.crop_size
                patch = model.backbone.config.patch_size
                record = dict(row, clip_id=identity, features=str(filename), shape=list(values.shape),
                              grid_height=crop // patch, grid_width=crop // patch,
                              crop_size=crop, tubelet_size=model.backbone.config.tubelet_size,
                              model_id=contract['model_id'], sha256=file_digest(filename),
                              cache_key=cache_key, source_sha256=fingerprint,
                              source_bytes=signature[0], contract=contract,
                              contract_sha256=dictionary_digest(contract))
                size = geometry.get('size', {})
                record['resize_shorter'] = int(size.get('shortest_edge', crop)) if isinstance(size, dict) else crop
                record['center_crop'] = bool(geometry.get('do_center_crop', True))
                record['processor'] = str(processor_directory)
                indexed.append(record)
                write_manifest(index_path, indexed)
    finally:
        if indexed:
            write_manifest(index_path, indexed)
    (root / 'model.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'clips': len(indexed), 'index': str(index_path)}))


def summarize(rows, check_hash=False):
    frames = patches = dimensions = 0
    model_ids = set()
    for row in rows:
        array = validate_array(row, check_hash)
        frames += array.shape[0]
        patches += array.shape[0] * array.shape[1]
        dimensions = max(dimensions, array.shape[2])
        model_ids.add(row.get('model_id'))
    return {'clips': len(rows), 'tubelets': frames, 'patch_tokens': patches,
            'maximum_feature_dimension': dimensions, 'model_ids': sorted(str(v) for v in model_ids)}


def pooled_features(rows, device):
    vectors = []
    dimensions = set()
    for row in rows:
        array = validate_array(row)
        dimensions.add(array.shape[-1])
        mean = np.asarray(array, dtype=np.float32).mean(axis=(0, 1))
        vectors.append(torch.from_numpy(mean))
    if len(dimensions) != 1:
        raise ValueError('Compared feature banks must use one embedding dimension.')
    values = torch.stack(vectors).to(device)
    return torch.nn.functional.normalize(values, dim=-1)


def screen_overlap(training, heldout, threshold, device, block_size=256):
    if not 0 < threshold <= 1 or block_size < 1:
        raise ValueError('Use a cosine threshold in (0,1] and positive comparison blocks.')
    first_ids = {row.get('model_id') for row in training}
    second_ids = {row.get('model_id') for row in heldout}
    if first_ids != second_ids or len(first_ids) != 1 or None in first_ids:
        raise ValueError('Overlap screening requires one identical model identity in both banks.')
    first = pooled_features(training, device)
    second = pooled_features(heldout, device)
    if first.shape[1] != second.shape[1]:
        raise ValueError('Training and held-out embedding dimensions differ.')
    matches, accepted = [], []
    with torch.no_grad():
        for start in range(0, len(first), block_size):
            query = first[start:start + block_size]
            best = torch.full((len(query),), -torch.inf, device=device)
            positions = torch.zeros(len(query), dtype=torch.long, device=device)
            for other_start in range(0, len(second), block_size):
                similarity = query @ second[other_start:other_start + block_size].T
                values, indices = similarity.max(-1)
                better = values > best
                positions = torch.where(better, indices + other_start, positions)
                best = torch.maximum(best, values)
            for offset, (value, index) in enumerate(zip(best.tolist(), positions.tolist())):
                row = training[start + offset]
                if value >= threshold:
                    matches.append({'clip_id': row['clip_id'], 'heldout_clip_id': heldout[index]['clip_id'],
                                    'cosine_similarity': value})
                else:
                    accepted.append(row)
    return accepted, matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    build = commands.add_parser('extract')
    build.add_argument('--manifest', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--checkpoint')
    build.add_argument('--config', default='vjepa2.yaml')
    build.add_argument('--pretrained')
    build.add_argument('--cache-dir')
    build.add_argument('--offline', action='store_true')
    build.add_argument('--device', default='cuda')
    build.add_argument('--resume', action='store_true')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--index', required=True)
    inspect.add_argument('--checksum', action='store_true')
    merge = commands.add_parser('merge')
    merge.add_argument('--index', action='append', required=True)
    merge.add_argument('--output', required=True)
    screen = commands.add_parser('screen')
    screen.add_argument('--index', required=True)
    screen.add_argument('--heldout', required=True)
    screen.add_argument('--output', required=True)
    screen.add_argument('--threshold', type=float, default=.95)
    screen.add_argument('--block-size', type=int, default=256)
    screen.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.operation == 'extract':
        extract(args)
    elif args.operation == 'inspect':
        print(json.dumps(summarize(read_index(args.index), args.checksum), indent=2))
    elif args.operation == 'screen':
        accepted, matches = screen_overlap(read_index(args.index), read_index(args.heldout),
                                           args.threshold, cuda_device(args.device), args.block_size)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_manifest(output / 'accepted.jsonl', accepted)
        (output / 'matches.json').write_text(json.dumps(matches, indent=2) + '\n')
        print(json.dumps({'accepted': len(accepted), 'removed': len(matches)}))
    else:
        rows = [row for path in args.index for row in read_index(path)]
        if len({r['clip_id'] for r in rows}) != len(rows):
            raise ValueError('Merged banks contain duplicate clip identifiers.')
        summarize(rows, check_hash=True)
        write_manifest(args.output, rows)
