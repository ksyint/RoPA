"""Typed factory for the released V-JEPA 2 initialization and RoPA checkpoints."""
from typing import Any, Dict, Optional


def resolve_model_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    values = dict(config['model'])
    name = values.pop('name', 'vjepa2_giant')
    if name != 'vjepa2_giant':
        raise ValueError(f'Expected vjepa2_giant, received {name}.')
    values.update(overrides or {})
    return values


def create_model(config: Dict[str, Any]):
    from .backbones.vjepa import VJEPA2RoPA
    return VJEPA2RoPA(**resolve_model_config(config))


def load_initialization(model, checkpoint: Dict[str, Any]) -> None:
    """Load learned parameters, retaining frequencies of the requested experiment."""
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model'].items()}
    for key, value in model.state_dict().items():
        if key.endswith(('time_freq', 'height_freq', 'width_freq')):
            state[key] = value.detach().clone()
    model.load_state_dict(state, strict=True)


def load_checkpoint_model(checkpoint: Dict[str, Any]):
    model = create_model(checkpoint['config'])
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    return model
