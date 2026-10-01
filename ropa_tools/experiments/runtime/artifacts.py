"""Inspect learned RoPA state and assemble portable inference artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


PROCESSOR_FILES = ('preprocessor_config.json', 'video_preprocessor_config.json', 'processor_config.json')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not {'model', 'config'} <= state.keys():
        raise ValueError('Expected a RoPA task checkpoint with model and config entries.')
    if not isinstance(state['model'], dict) or not state['model']:
        raise ValueError('The checkpoint has no model tensors.')
    if any(not torch.is_tensor(value) for value in state['model'].values()):
        raise ValueError('The model state must contain only tensors.')
    return state


def tensor_inventory(state):
    result = {}
    for name, value in state['model'].items():
        group = 'predictor' if '.predictor.' in name else 'encoder'
        entry = result.setdefault(group, {'tensors': 0, 'elements': 0, 'bytes': 0, 'dtypes': {}})
        dtype = str(value.dtype).removeprefix('torch.')
        entry['tensors'] += 1
        entry['elements'] += value.numel()
        entry['bytes'] += value.numel() * value.element_size()
        entry['dtypes'][dtype] = entry['dtypes'].get(dtype, 0) + value.numel()
    return result


def inspect_rotations(state):
    values = []
    for name, tensor in state['model'].items():
        if not name.endswith('time_freq'):
            continue
        if tensor.ndim != 1 or not len(tensor):
            raise ValueError(f'Invalid rotary frequency tensor: {name}')
        if not torch.isfinite(tensor).all() or (tensor <= 0).any():
            raise ValueError(f'Nonpositive or nonfinite rotary frequencies: {name}')
        values.append({'name': name, 'pairs': tensor.numel(),
                       'half_cycle': float(torch.pi / tensor.min()),
                       'local_scale': float(torch.pi / tensor.max())})
    if not values:
        raise ValueError('The checkpoint contains no temporal rotary frequency buffers.')
    return values


def inspect_checkpoint(path):
    state = load_state(path)
    model = state['config']['model']
    rotations = inspect_rotations(state)
    return {
        'file': str(Path(path).resolve()), 'sha256': checksum(path),
        'step': state.get('step'), 'pretrained': model.get('pretrained'),
        'target_range': model.get('target_range'), 'local_scale': model.get('local_scale'),
        'has_embedded_architecture': isinstance(model.get('hf_config'), dict),
        'tensor_groups': tensor_inventory(state), 'rotary_buffers': rotations,
    }


def compare_checkpoints(first, second):
    first, second = load_state(first), load_state(second)
    left, right = first['model'], second['model']
    common = sorted(left.keys() & right.keys())
    shape_changes = []
    differences = {}
    for name in common:
        a, b = left[name], right[name]
        if a.shape != b.shape:
            shape_changes.append({'name': name, 'first': list(a.shape), 'second': list(b.shape)})
            continue
        if a.is_floating_point() and b.is_floating_point():
            total, maximum = 0., 0.
            first_flat, second_flat = a.flatten(), b.flatten()
            for start in range(0, a.numel(), 1 << 18):
                delta = first_flat[start:start + (1 << 18)].float() - second_flat[start:start + (1 << 18)].float()
                total += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.)
            differences[name] = {'rms': (total / max(1, a.numel())) ** .5, 'maximum': maximum}
    return {'added': sorted(right.keys() - left.keys()), 'removed': sorted(left.keys() - right.keys()),
            'shape_changes': shape_changes, 'parameter_differences': differences}


def bundle(checkpoint, destination, processor):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Bundle destination must be empty.')
    state = load_state(checkpoint)
    if not state['config']['model'].get('hf_config'):
        raise ValueError('The task checkpoint must embed its Hugging Face architecture configuration.')
    processor = Path(processor).resolve()
    present = [name for name in PROCESSOR_FILES if (processor / name).is_file()]
    if not present:
        raise ValueError('The processor directory needs its original preprocessing JSON.')
    processor_output = destination / 'processor'
    processor_output.mkdir()
    for name in present:
        shutil.copy2(processor / name, processor_output / name)
    (processor_output / 'config.json').write_text(json.dumps(state['config']['model']['hf_config'], indent=2) + '\n')
    present.append('config.json')
    checkpoint_output = destination / 'last.pt'
    shutil.copy2(checkpoint, checkpoint_output)
    metadata = {'checkpoint': 'last.pt', 'processor': 'processor',
                'sha256': checksum(checkpoint_output),
                'processor_files': {name: checksum(processor_output / name) for name in present}}
    (destination / 'bundle.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def verify_bundle(path):
    path = Path(path).resolve()
    metadata = json.loads((path / 'bundle.json').read_text())
    checkpoint = (path / metadata['checkpoint']).resolve()
    processor = (path / metadata['processor']).resolve()
    if path not in checkpoint.parents or path not in processor.parents:
        raise ValueError('Bundle paths must remain inside the exported directory.')
    if checksum(checkpoint) != metadata['sha256']:
        raise ValueError('Checkpoint digest differs from the bundle manifest.')
    for filename, digest in metadata['processor_files'].items():
        artifact = (processor / filename).resolve()
        if processor not in artifact.parents:
            raise ValueError('Processor files must remain inside the processor directory.')
        if checksum(artifact) != digest:
            raise ValueError(f'Processor digest differs: {filename}')
    return inspect_checkpoint(checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--checkpoint', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--first', required=True)
    compare.add_argument('--second', required=True)
    package = commands.add_parser('bundle')
    package.add_argument('--checkpoint', required=True)
    package.add_argument('--processor', required=True)
    package.add_argument('--destination', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--bundle', required=True)
    parser.add_argument('--report')
    args = parser.parse_args(argv)
    if args.operation == 'inspect':
        result = inspect_checkpoint(args.checkpoint)
    elif args.operation == 'compare':
        result = compare_checkpoints(args.first, args.second)
    elif args.operation == 'bundle':
        result = bundle(args.checkpoint, args.destination, args.processor)
    else:
        result = verify_bundle(args.bundle)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
