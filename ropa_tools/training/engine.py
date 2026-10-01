"""Temporal optimization with gradient accumulation, cosine rates and exact clip continuation."""
from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch

from models.rotary import sample_spacing
from ropa_tools.data.sampling import ClipStream
from ropa_tools.training import snapshot
from ropa_tools.training.distributed import distributed_objective


def spacing_for(video, model_options, config):
    tubelet = model_options['tubelet']
    if video.shape[2] % tubelet:
        raise ValueError('Frame count must be divisible by the temporal tubelet size.')
    span = video.shape[2] // tubelet - 1
    if span < 1:
        raise ValueError('Temporal training needs at least two tubelets.')
    if not config.get('enabled', True):
        return video.new_ones(len(video))
    return sample_spacing(
        len(video), span, torch.pi / model_options['target_range'],
        low=config.get('low', .5), high=config.get('high', 2.), device=video.device,
    )


def write_history(output, history):
    path = Path(output) / 'metrics.json'
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(history, indent=2) + '\n')
    temporary.replace(path)


def optimize(model, anchor, objective, dataset, optimizer, config, model_options,
             steps, device, output, data_path, resume=None, seed=None, group=None):
    runtime = config['runtime']
    accumulation = int(runtime.get('accumulation_steps', 1))
    checkpoint_interval = int(runtime.get('checkpoint_interval', 1000))
    log_interval = int(runtime.get('log_interval', 10))
    clip_norm = float(runtime.get('clip_grad_norm', 1.))
    if min(accumulation, checkpoint_interval, log_interval, steps) < 1 or clip_norm <= 0:
        raise ValueError('Training intervals, accumulation and clipping must be positive.')
    rank, world_size = (group.rank, group.world_size) if group is not None else (0, 1)
    stream = ClipStream(dataset, runtime['batch_size'], runtime['seed'] if seed is None else seed,
                        rank=rank, world_size=world_size)
    forward = distributed_objective(model, anchor, objective, group) if group is not None else None
    schedule = CosineSchedule(
        optimizer, steps,
        warmup_steps=config['optim'].get('warmup_steps', 0),
        minimum_ratio=config['optim'].get('minimum_lr_ratio', 0.),
    )
    identity = snapshot.dataset_identity(data_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    if resume:
        history = snapshot.restore(resume, model, anchor, optimizer, schedule,
                                   config, stream, identity, device, group)
    model.train()
    anchor.eval().requires_grad_(False)
    for step in range(schedule.step_number, steps):
        optimizer.zero_grad(set_to_none=True)
        window = [next(stream) for _ in range(accumulation)]
        denominator = sum(len(video) for video in window)
        values_sum = {}
        for microbatch, raw in enumerate(window):
            video = raw.to(device, non_blocking=True)
            spacing = spacing_for(video, model_options, config.get('spacing', {}))
            synchronize = microbatch + 1 == len(window)
            context = forward.no_sync() if world_size > 1 and not synchronize else nullcontext()
            with context:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    values = forward(video, spacing, step, steps) if forward is not None else objective(model, anchor, video, spacing, step, steps)
                if not torch.isfinite(values['loss']):
                    raise FloatingPointError(f'Nonfinite temporal loss at step {step}.')
                fraction = len(video) / denominator
                (values['loss'] * fraction).backward()
            for key, value in values.items():
                number = float(value.detach()) if torch.is_tensor(value) else float(value)
                values_sum[key] = values_sum.get(key, 0.) + number * fraction
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        learning_rates = [group['lr'] for group in optimizer.param_groups]
        optimizer.step()
        schedule.step()
        if group is not None:
            values_sum, global_samples = group.weighted_metrics(values_sum, denominator)
        else:
            global_samples = denominator
        record = dict(step=step, epoch=stream.epoch, samples=global_samples,
                      gradient_norm=float(norm), learning_rates=learning_rates, **values_sum)
        history.append(record)
        if (group is None or group.primary) and (step % log_interval == 0 or step + 1 == steps):
            print(json.dumps(record), flush=True)
        if (step + 1) % checkpoint_interval == 0 or step + 1 == steps:
            snapshot.save(output / 'last.pt', model, anchor, optimizer, schedule,
                          config, history, stream, identity, device, group)
            if group is None or group.primary:
                write_history(output, history)
    return history



class CosineSchedule:
    def __init__(self, optimizer, total_steps, warmup_steps=0, minimum_ratio=0.):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.minimum_ratio = float(minimum_ratio)
        self.base_rates = [float(group['lr']) for group in optimizer.param_groups]
        self.step_number = 0
        if self.total_steps < 1 or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError('The warmup must leave at least one ordinary training step.')
        if not 0 <= self.minimum_ratio <= 1:
            raise ValueError('Minimum learning-rate ratio must lie in [0,1].')
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.base_rates):
            raise ValueError('Base learning rates must be finite and positive.')
        self.apply()

    def multiplier(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        fraction = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps - 1)
        fraction = min(1., max(0., fraction))
        return self.minimum_ratio + (1 - self.minimum_ratio) * .5 * (1 + math.cos(math.pi * fraction))

    def apply(self):
        scale = self.multiplier(self.step_number)
        for group, initial in zip(self.optimizer.param_groups, self.base_rates):
            group['lr'] = initial * scale

    def step(self):
        self.step_number += 1
        self.apply()

    def state_dict(self):
        return dict(total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    minimum_ratio=self.minimum_ratio, base_rates=self.base_rates, step_number=self.step_number)

    def load_state_dict(self, state):
        for name in ('total_steps', 'warmup_steps', 'minimum_ratio', 'base_rates'):
            if getattr(self, name) != state[name]:
                raise ValueError(f'Resume schedule differs in {name}.')
        step = int(state['step_number'])
        if not 0 <= step <= self.total_steps:
            raise ValueError('Saved step is outside the requested training schedule.')
        self.step_number = step
        self.apply()


def decay_groups(model, weight_decay):
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError('Weight decay must be finite and nonnegative.')
    decay, no_decay = [], []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError('Trainable parameters must occur once in optimizer groups.')
        seen.add(id(parameter))
        if parameter.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not seen:
        raise ValueError('The model has no trainable parameters.')
    groups = []
    if decay:
        groups.append(dict(params=decay, weight_decay=weight_decay, name='matrix'))
    if no_decay:
        groups.append(dict(params=no_decay, weight_decay=0., name='bias_and_scale'))
    return groups
