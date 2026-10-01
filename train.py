import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from utils.data import SyntheticVideo, VideoDataset
from utils.losses import consistency_loss, gram_weight, paga_loss
from utils.models import RoPA
from utils.rope import sample_spacing


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed if args.seed is not None else config['seed'])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = RoPA(**config['model']).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)['model'])
    # A frozen earlier checkpoint anchors prediction. Real runs should initialize
    # from a trained model; the random frozen teacher is only a smoke fixture.
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    dataset = VideoDataset(args.data) if args.data else SyntheticVideo()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps or config['steps']
    iterator = iter(loader)
    log = []
    for step in range(steps):
        try:
            video = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            video = next(iterator)
        video = video.to(device)
        span = video.shape[2] // config['model']['tubelet'] - 1
        if span < 1:
            raise ValueError('Each clip must contain at least two tubelets.')
        spacing = sample_spacing(len(video), span, torch.pi / config['model']['target_range'], device=device)
        z = model(video, spacing)
        predicted = model.predictor(z[:, :-1], 1)
        with torch.no_grad():
            target = teacher(video, spacing)
            teacher_predicted = teacher.predictor(target[:, :-1], 1)
        prediction_loss = F.mse_loss(predicted, target[:, 1:])
        patches = torch.randperm(z.shape[-2], device=device)[:config['sample_patches']]
        # The time axis enumerates source anchors, all at offset delta=1.
        # Average anchors in the batch axis; PAGA separately sums offsets.
        gram_inputs = [value.flatten(0, 1).unsqueeze(1)
                       for value in (predicted, z[:, 1:], teacher_predicted, target[:, 1:])]
        structural_loss = paga_loss(*gram_inputs, indices=patches)
        rope_loss = consistency_loss(model.predictor, z[:, 0], 1, 1, config['model']['target_range'])
        weight = gram_weight(step, steps, config['lambda_gram'], config['gram_warmup'])
        loss = prediction_loss + weight * structural_loss + config['lambda_rope'] * rope_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {'step': step, 'loss': loss.item(), 'prediction': prediction_loss.item(),
               'paga': structural_loss.item(), 'rcl': rope_loss.item(), 'gram_weight': weight}
        log.append(row)
        if step % 10 == 0 or step == steps - 1:
            print(json.dumps(row))
    torch.save({'model': model.state_dict(), 'config': config, 'step': steps}, output / 'last.pt')
    (output / 'metrics.json').write_text(json.dumps(log, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--data', help='Directory containing C,T,H,W .npy video clips; omitted = synthetic smoke.')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default='outputs/smoke')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    main(parser.parse_args())
