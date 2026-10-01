"""Temporal correspondence experiments, from video manifests to label propagation."""
import argparse
import copy
import json
import math
import random
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.rotary import band_diagnostics, temporal_frequencies
from models.backbone import create_model, cuda_device, load_checkpoint_model
from models.backbone import load_initialization, resolve_model_config
from models.rotary import sample_spacing
from ropa_tools.data.loading import VideoDataset, VideoManifest, propagate_labels
from ropa_tools.temporal import catalog_layout, read_recipe, write_recipe


def cross_gram(predicted, target):
    return F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).transpose(-1, -2)


def paga_loss(student_prediction, student_target, teacher_prediction, teacher_target, indices=None):
    """Eq. (5): sum across offsets, average across clips and sampled patch pairs.

    Inputs B,offset,N,D. All four tensors use the same patch subset. Teacher
    structure is detached. Target-frame student features retain gradients.
    """
    features = [student_prediction, student_target, teacher_prediction, teacher_target]
    if any(t.shape != features[0].shape for t in features) or features[0].ndim != 4:
        raise ValueError('PAGA inputs must share B,offset,N,D shape.')
    if indices is not None:
        features = [t.index_select(-2, indices) for t in features]
    student = cross_gram(*features[:2])
    with torch.no_grad():
        teacher = cross_gram(*features[2:])
    return (student - teacher).square().mean((-1, -2)).sum(-1).mean()


def consistency_loss(predictor, z, delta1, delta2, target_range=64.0, identity_weight=1.0, normalize=False):
    if delta1 < 0 or delta2 < 0 or delta1 + delta2 > target_range:
        raise ValueError('Composition offsets must be nonnegative and sum to at most Tband.')
    direct = predictor(z, delta1 + delta2)
    composed = predictor(predictor(z, delta1), delta2)
    identity_prediction = predictor(z, 0)
    if normalize:
        direct, composed, identity_prediction, z = [F.normalize(value, dim=-1)
            for value in (direct, composed, identity_prediction, z)]
    # Eq. (6) is squared L2 in feature dimension, followed by an expectation.
    composition = (direct - composed).square().sum(-1).mean()
    identity = (identity_prediction - z).square().sum(-1).mean()
    return composition + identity_weight * identity


def gram_weight(step, total_steps, weight=1.0, warmup_steps=5000):
    start = int(0.9 * total_steps)
    if step < start:
        return 0.0
    return weight * min(1.0, (step - start + 1) / max(1, warmup_steps))


class RoPAObjective(torch.nn.Module):
    """Offset prediction with an explicit anchor axis and one selected temporal offset."""
    def __init__(self, lambda_gram=1.0, lambda_rope=0.1, gram_warmup=5000,
                 sample_patches=64, target_range=64.0, prediction_offset=1):
        super().__init__()
        self.lambda_gram = lambda_gram
        self.lambda_rope = lambda_rope
        self.gram_warmup = gram_warmup
        self.sample_patches = sample_patches
        self.target_range = target_range
        self.prediction_offset = int(prediction_offset)

    def forward(self, model, anchor, video, spacing, step, total_steps):
        features = model(video, spacing)
        delta = self.prediction_offset
        if features.shape[1] <= delta:
            raise ValueError("The clip must contain more tubelets than prediction_offset.")
        predicted = model.predictor(features[:, :-delta], delta)
        with torch.no_grad():
            target = anchor(video, spacing)
            target_prediction = anchor.predictor(target[:, :-delta], delta)
        prediction = F.mse_loss(
            F.normalize(predicted, dim=-1),
            F.normalize(target[:, delta:], dim=-1),
        )
        selected = torch.randperm(features.shape[-2], device=video.device)[:self.sample_patches]
        # The offset axis is length one. Source times belong to the anchor batch.
        gram_inputs = [
            value.flatten(0, 1).unsqueeze(1)
            for value in (
                predicted, features[:, delta:], target_prediction, target[:, delta:]
            )
        ]
        gram = paga_loss(*gram_inputs, indices=selected)
        consistency = consistency_loss(
            model.predictor, features[:, 0], 1, 1,
            self.target_range, normalize=True,
        )
        weight = gram_weight(step, total_steps, self.lambda_gram, self.gram_warmup)
        total = prediction + weight * gram + self.lambda_rope * consistency
        return {
            'loss': total,
            'prediction': prediction,
            'paga': gram,
            'rcl': consistency,
            'gram_weight': weight,
        }


def endless_batches(loader: Iterable):
    while True:
        yield from loader


def train_steps(model, anchor, objective, loader, optimizer, *, steps: int,
                tubelet: int, target_range: float, device: torch.device, spacing_config=None) -> list:
    model.train()
    anchor.eval()
    history = []
    batches = endless_batches(loader)
    spacing_options = spacing_config or {}
    for step in range(steps):
        video = next(batches).to(device)
        span = video.shape[2] // tubelet - 1
        if span < 1:
            raise ValueError('Each clip must contain at least two tubelets.')
        if spacing_options.get('enabled', True):
            spacing = sample_spacing(len(video), span, torch.pi / target_range,
                                     low=spacing_options.get('low', 0.5),
                                     high=spacing_options.get('high', 2.0), device=device)
        else:
            spacing = video.new_ones(len(video))
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            values = objective(model, anchor, video, spacing, step, steps)
        optimizer.zero_grad(set_to_none=True)
        values['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {'step': step, **{key: value.detach().item() if torch.is_tensor(value) else value
                              for key, value in values.items()}}
        history.append(row)
        if step % 10 == 0 or step == steps - 1:
            print(json.dumps(row))
    return history


def validate_config(config):
    required = {'model', 'runtime', 'optim', 'objective'}
    if not required <= config.keys():
        raise ValueError(f'Missing config sections: {sorted(required - config.keys())}')
    model, runtime = config['model'], config['runtime']
    if not 0 < model['local_scale'] <= model['target_range']:
        raise ValueError('Temporal scales must satisfy 0 < T0 <= T*.')
    if config['objective'].get('prediction_offset', 1) < 1:
        raise ValueError('Prediction offset must be positive.')
    if model.get('name') != 'vjepa2_giant' or not model.get('pretrained'):
        raise ValueError('A released V-JEPA 2 model ID or local snapshot is required.')
    frames = config.get('data', {}).get('frames', 16)
    offset = config['objective'].get('prediction_offset', 1)
    if frames % 2 or frames // 2 <= offset:
        raise ValueError('The even frame count must provide more tubelets than the prediction offset.')
    if runtime['steps'] < 1 or runtime['batch_size'] < 1:
        raise ValueError('Training requires positive steps and batch size.')
    if config['optim']['lr'] <= 0 or config['optim']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings.')
    if any(config['objective'][key] < 0 for key in ('lambda_gram', 'lambda_rope')):
        raise ValueError('Regularizer weights must be nonnegative.')
    spacing = config.get('spacing', {})
    low, high = spacing.get('low', 0.5), spacing.get('high', 2.0)
    if not all(math.isfinite(value) for value in (low, high)) or not 0 < low <= high:
        raise ValueError('Spacing jitter needs finite bounds 0 < low <= high.')
    return config


ROOT = Path(__file__).resolve().parent


def profile_yaml_path(band, jitter, prediction_offset, gram, rcl):
    return ROOT / 'experiments' / 'temporal' / f'band_{band}' / f'jitter_{jitter}' / f'predictor_delta{prediction_offset}' / f'gram_{gram}' / f'rcl_{rcl}.yaml'


def profile_path(band, jitter, prediction_offset, gram, rcl):
    key = (band, jitter, prediction_offset, gram, rcl)
    relative = profile_yaml_path(*key).relative_to(ROOT)
    return ROOT / PROFILE_LAYOUT[relative].with_suffix('.py' if key in PYTHON_PROFILES else '.yaml')


def command_sweep(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', type=int, choices=[64, 160, 640, 1280], default=64)
    parser.add_argument('--jitter', choices=['fixed', 'narrow', 'full'], default='full')
    parser.add_argument('--prediction-offset', type=int, choices=[2, 4], default=2)
    parser.add_argument('--gram', choices=['0p5', '1p0'], default='1p0')
    parser.add_argument('--rcl', choices=['0p00', '0p01', '0p05', '0p10', '0p20'], default='0p10')
    parser.add_argument('--data')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default='outputs/band_sweep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = profile_path(args.band, args.jitter, args.prediction_offset, args.gram, args.rcl)
    if not config.is_file():
        parser.error(f'Profile not found: {config}')
    if not args.dry_run and not args.data:
        parser.error('--data is required for a temporal-band experiment.')
    command = [sys.executable, str(ROOT / 'ropa.py'), 'train', '--config', str(config), '--device', args.device,
               '--output', args.output]
    for option in ('data', 'checkpoint'):
        if getattr(args, option):
            command.extend(['--' + option, getattr(args, option)])
    if args.dry_run:
        command.append('--dry-run')
    subprocess.run(command, cwd=ROOT, check=True)


BANDS = (64, 160, 640, 1280)
SPACING = {'fixed': dict(enabled=False, low=1.0, high=1.0),
           'narrow': dict(enabled=True, low=0.75, high=1.5),
           'full': dict(enabled=True, low=0.5, high=2.0)}
GRAM = {'0p5': 0.5, '1p0': 1.0}
RCL = {'0p00': 0.0, '0p01': 0.01, '0p05': 0.05, '0p10': 0.1, '0p20': 0.2}
PYTHON_PROFILES = frozenset(sorted(product(BANDS, SPACING, (2, 4), GRAM, RCL),
                                 key=lambda values: profile_yaml_path(*values).as_posix())[:109])
PROFILE_LAYOUT = catalog_layout(profile_yaml_path(*values).relative_to(ROOT)
    for values in product(BANDS, SPACING, (2, 4), GRAM, RCL))


def command_profiles(argv=None):
    parser = argparse.ArgumentParser(description='Regenerate the temporal experiment catalog.')
    parser.parse_args(argv)
    baseline = read_recipe(ROOT / 'vjepa2.yaml')
    count = 0
    for band, jitter, predictor, gram, rcl in product(BANDS, SPACING, (2, 4), GRAM, RCL):
        config = copy.deepcopy(baseline)
        config['model'].update(target_range=float(band))
        config['objective']['prediction_offset'] = predictor
        config['spacing'] = dict(SPACING[jitter])
        config['objective'].update(lambda_gram=GRAM[gram], lambda_rope=RCL[rcl])
        validate_config(config)
        output = profile_path(band, jitter, predictor, gram, rcl)
        write_recipe(output, config)
        count += 1
    print(f'Wrote {count} executable temporal profiles under experiments/.')


def run_prepare(args):
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


def command_prepare(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True)
    parser.add_argument('--output', default='data')
    parser.add_argument('--clip-seconds', type=float, default=2.0)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    run_prepare(parser.parse_args(argv))


def run_train(args):
    config = read_recipe(args.config)
    validate_config(config)
    for key in ('pretrained', 'cache_dir'):
        if getattr(args, key) is not None:
            config['model'][key] = getattr(args, key)
    if args.offline:
        config['model']['local_files_only'] = True
    if args.dry_run:
        print(json.dumps({'config': config, 'resolved_model': resolve_model_config(config),
                          'data': args.data, 'checkpoint': args.checkpoint, 'device': args.device}, indent=2))
        return
    runtime, optim, objective = config['runtime'], config['optim'], config['objective']
    torch.manual_seed(args.seed if args.seed is not None else runtime['seed'])
    torch.set_num_threads(args.threads)
    device = cuda_device(args.device)
    if not args.data:
        raise ValueError('--data requires a video JSONL manifest or a preprocessed NPY directory.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if checkpoint and 'hf_config' in checkpoint['config']['model']:
        config['model']['hf_config'] = checkpoint['config']['model']['hf_config']
    model = create_model(config).to(device)
    if checkpoint:
        load_initialization(model, checkpoint)
    if hasattr(model, 'backbone'):
        config['model']['hf_config'] = model.backbone.config.to_dict()
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    if Path(args.data).is_file():
        dataset = VideoManifest(args.data, config['model'], config.get('data', {}).get('frames', 16))
    else:
        dataset = VideoDataset(args.data)
    loader = DataLoader(dataset, batch_size=runtime['batch_size'], shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim['lr'],
        betas=tuple(optim['betas']),
        weight_decay=optim['weight_decay'],
    )
    model_options = resolve_model_config(config)
    criterion = RoPAObjective(**objective, target_range=model_options['target_range'])
    steps = args.steps or runtime['steps']
    history = train_steps(
        model, anchor, criterion, loader, optimizer,
        steps=steps,
        tubelet=model_options['tubelet'],
        target_range=model_options['target_range'],
        device=device,
        spacing_config=config.get('spacing'),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config, 'step': steps}, output / 'last.pt')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


def command_train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the resolved experiment without loading data or CUDA.')
    parser.add_argument('--config', default='vjepa2.yaml')
    parser.add_argument('--data', help='Video JSONL manifest, or directory of already normalized C,T,H,W .npy clips.')
    parser.add_argument('--checkpoint')
    parser.add_argument('--pretrained', help='Official HF V-JEPA 2 ID or local snapshot directory.')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', default='outputs/ropa')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    run_train(parser.parse_args(argv))


def run_extract(args):
    device = cuda_device(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        config = checkpoint['config']
    else:
        config = read_recipe(args.config)
    for key in ('pretrained', 'cache_dir'):
        if getattr(args, key) is not None:
            config['model'][key] = getattr(args, key)
    if args.offline:
        config['model']['local_files_only'] = True
    model = (load_checkpoint_model(checkpoint) if args.checkpoint else create_model(config)).to(device).eval()
    if Path(args.data).is_file():
        dataset = VideoManifest(args.data, config['model'], config.get('data', {}).get('frames', 16))
    else:
        dataset = VideoDataset(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for index, (path, video) in enumerate(zip(dataset.paths, dataset)):
            features = model(video[None].to(device))[0].float().cpu().numpy()
            destination = output / f'{index:06d}_{path.stem}.npy'
            np.save(destination, features)
            print(f'{destination.name}: {features.shape}')


def command_extract(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Trained RoPA checkpoint; otherwise use the released foundation initialization.')
    parser.add_argument('--config', default='vjepa2.yaml')
    parser.add_argument('--pretrained')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    run_extract(parser.parse_args(argv))


def run_evaluate(args):
    args.device = str(cuda_device(args.device))
    if not args.data:
        frequencies = temporal_frequencies(16, args.target_range, 1).to(args.device)
        diagnostics = band_diagnostics(frequencies, [1, 2, 32, 64, 128, 256, 512])
        diagnostics['displacement'] = diagnostics['displacement'].tolist()
        print(json.dumps(diagnostics, indent=2))
        return
    with np.load(args.data, allow_pickle=False) as data:
        features = torch.from_numpy(data['features']).float().to(args.device)
        labels = torch.from_numpy(data['labels']).long().to(args.device)  # T,H,W
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
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, result.cpu().numpy())


def command_evaluate(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data', help='.npz with features T,H*W,D and labels T,H,W')
    parser.add_argument('--target_range', type=float, default=64)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--output')
    run_evaluate(parser.parse_args(argv))


COMMANDS = {
    'manifest': 'ropa_tools.data.catalog',
    'features': 'ropa_tools.data.bank',
    'sequences': 'ropa_tools.evaluation.sequence',
    'temporal': 'ropa_tools.evaluation.gaps',
    'checkpoint': 'ropa_tools.artifacts',
    'study': 'ropa_tools.temporal',

    'sweep': command_sweep,
    'profiles': command_profiles,
    'prepare': command_prepare,
    'train': command_train,
    'extract': command_extract,
    'evaluate': command_evaluate,
}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    if isinstance(handler, str):
        from importlib import import_module
        handler = import_module(handler).main
    handler(args.arguments)


if __name__ == '__main__':
    main()
