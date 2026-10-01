"""Step-based pretraining engine; data/model construction lives in train.py."""
import json
from typing import Dict, Iterable

import torch

from models.layers.rotary import sample_spacing


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
