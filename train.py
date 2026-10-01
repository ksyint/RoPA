import argparse
import copy
import json
from pathlib import Path

import torch
from models.runtime import cuda_device
import yaml
from torch.utils.data import DataLoader

from datasets import SyntheticVideo, VideoDataset
from engine import train_steps
from models import create_model, load_initialization, resolve_model_config
from criterion import RoPAObjective


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    from tools.profile_config import validate_config
    validate_config(config)
    if args.dry_run:
        print(json.dumps({'config': config, 'resolved_model': resolve_model_config(config),
                          'data': args.data, 'checkpoint': args.checkpoint, 'device': args.device}, indent=2))
        return
    runtime, optim, objective = config['runtime'], config['optim'], config['objective']
    torch.manual_seed(args.seed if args.seed is not None else runtime['seed'])
    torch.set_num_threads(args.threads)
    device = cuda_device(args.device)
    model = create_model(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        load_initialization(model, checkpoint)
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    dataset = VideoDataset(args.data) if args.data else SyntheticVideo(**config['synthetic'])
    loader = DataLoader(dataset, batch_size=runtime['batch_size'], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim['lr'], betas=tuple(optim['betas']),
                                 weight_decay=optim['weight_decay'])
    model_options = resolve_model_config(config)
    criterion = RoPAObjective(**objective, target_range=model_options['target_range'])
    steps = args.steps or runtime['steps']
    history = train_steps(model, anchor, criterion, loader, optimizer, steps=steps,
                          tubelet=model_options['tubelet'], target_range=model_options['target_range'], device=device,
                          spacing_config=config.get('spacing'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config, 'step': steps}, output / 'last.pt')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the resolved experiment without loading data or CUDA.')
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--data', help='Directory containing C,T,H,W .npy video clips; omitted = synthetic smoke.')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default='outputs/smoke')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    main(parser.parse_args())
