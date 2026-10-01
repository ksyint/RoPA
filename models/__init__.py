"""RoPA model construction and checkpoint configuration resolution."""
from typing import Any, Dict, Optional

from torch import nn

from .heads.predictor import TemporalPredictor
from .backbones.video import VideoEncoder
from .layers.rotary import Rotary3D


class RoPA(nn.Module):
    def __init__(self, predictor_ratio=2.0, **kwargs):
        super().__init__()
        self.encoder = VideoEncoder(**kwargs)
        self.predictor = TemporalPredictor(kwargs.get('dim', 64), hidden_ratio=predictor_ratio)

    def forward(self, video, spacing=None):
        return self.encoder(video, spacing)


MODEL_DEFAULTS = {
    'ropa_small': dict(dim=64, depth=2, heads=1, rotary_dims=(16, 24, 24), patch_size=8),
    'ropa_base': dict(dim=768, depth=12, heads=12, rotary_dims=(16, 24, 24), patch_size=16),
}


def resolve_model_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve named architecture defaults, explicit YAML options, then overrides."""
    values = dict(config['model'])
    name = values.pop('name', 'ropa_small')
    if name not in MODEL_DEFAULTS:
        raise ValueError(f'Unknown RoPA architecture: {name}')
    resolved = dict(MODEL_DEFAULTS[name])
    resolved.update(values)
    resolved.update(overrides or {})
    return resolved


def create_model(config: Dict[str, Any]) -> RoPA:
    return RoPA(**resolve_model_config(config))


def load_initialization(model: RoPA, checkpoint: Dict[str, Any]) -> None:
    """Load learned weights while retaining the configured rotary frequencies."""
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model'].items()}
    for name, module in model.named_modules():
        if isinstance(module, Rotary3D):
            for buffer_name in ('time_freq', 'height_freq', 'width_freq'):
                key = f'{name}.{buffer_name}' if name else buffer_name
                state[key] = getattr(module, buffer_name).detach().clone()
    model.load_state_dict(state, strict=True)


def load_checkpoint_model(checkpoint: Dict[str, Any]) -> RoPA:
    model = create_model(checkpoint['config'])
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    return model
