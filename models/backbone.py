"""V-JEPA 2 adaptation and rotary geometry on the pretrained attention projections."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from models.rotary import HorizonAttention, install_horizon_attention


def cuda_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('RoPA model execution requires a CUDA device (cuda or cuda:N).')
    return device


class VJEPA2RoPA(nn.Module):
    def __init__(self, pretrained='facebook/vjepa2-vitg-fpc64-384', cache_dir=None,
                 local_files_only=False, revision='main', target_range=64.0,
                 local_scale=1.0, causal=True, gradient_checkpointing=True,
                 hf_config=None, initialize=True, tubelet=2):
        super().__init__()
        options = dict(cache_dir=cache_dir, local_files_only=local_files_only, revision=revision)
        if hf_config is not None:
            config = AutoConfig.for_model(**hf_config)
            config._attn_implementation = 'sdpa'
            self.backbone = AutoModel.from_config(config)
        elif initialize:
            self.backbone = AutoModel.from_pretrained(pretrained, attn_implementation='sdpa', **options)
        else:
            config = AutoConfig.from_pretrained(pretrained, **options)
            config._attn_implementation = 'sdpa'
            self.backbone = AutoModel.from_config(config)
        if self.backbone.config.model_type != 'vjepa2':
            raise ValueError('RoPA requires a V-JEPA 2 encoder plus predictor checkpoint.')
        if self.backbone.config.tubelet_size != tubelet:
            raise ValueError('Configured tubelet size must match the released checkpoint.')
        install_horizon_attention(self.backbone, target_range, local_scale)
        self.causal = causal
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # Predictor calls use different anchor batches within one loss. Keep their
        # rotations bound to each forward. Only encoder blocks are recomputed.
        for module in self.backbone.predictor.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
        self._spacing = None

    def _set_spacing(self, root, spacing):
        for module in root.modules():
            if isinstance(module, HorizonAttention):
                module.spacing = spacing

    def forward(self, video, spacing=None):
        # Input C,T,H,W is preprocessed by the official AutoVideoProcessor.
        batch, _, frames, height, width = video.shape
        cfg = self.backbone.config
        if height != cfg.crop_size or width != cfg.crop_size or frames % cfg.tubelet_size:
            raise ValueError('Input crop/tubelet dimensions must match the saved V-JEPA configuration.')
        n_spatial = (height // cfg.patch_size) * (width // cfg.patch_size)
        times = frames // cfg.tubelet_size
        self._spacing = spacing
        self._set_spacing(self.backbone.encoder, spacing)
        mask = None
        if self.causal:
            time_ids = torch.arange(times, device=video.device).repeat_interleave(n_spatial)
            mask = (time_ids[:, None] >= time_ids[None, :])[None].expand(batch, -1, -1)
        values = self.backbone(
            pixel_values_videos=video.transpose(1, 2),
            context_head_mask=mask,
            skip_predictor=True,
        ).last_hidden_state
        return values.unflatten(1, (times, n_spatial))

    def predictor(self, features, delta):
        """Use released mask tokens, transformer blocks, and 1408-D output projection."""
        if int(delta) != delta or delta < 0:
            raise ValueError('V-JEPA temporal offsets are nonnegative integer tubelet steps.')
        shape = features.shape
        tokens, dim = shape[-2:]
        flat = features.reshape(-1, tokens, dim)
        spacing = self._spacing
        if spacing is not None and flat.shape[0] != len(spacing):
            spacing = spacing.repeat_interleave(flat.shape[0] // len(spacing))
        self._set_spacing(self.backbone.predictor, spacing)
        context = torch.arange(tokens, device=flat.device)[None].expand(len(flat), -1)
        target = context + int(delta) * tokens
        values = self.backbone.predictor(
            flat,
            context_mask=[context],
            target_mask=[target],
        ).last_hidden_state
        return values.reshape(shape)


def resolve_model_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    values = dict(config['model'])
    name = values.pop('name', 'vjepa2_giant')
    if name != 'vjepa2_giant':
        raise ValueError(f'Expected vjepa2_giant, received {name}.')
    values.update(overrides or {})
    return values


def create_model(config: Dict[str, Any]):
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
