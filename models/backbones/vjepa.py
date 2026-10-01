"""V-JEPA 2 ViT-g encoder and pretrained transformer predictor for RoPA."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2RopeAttention


def _rotate(x, positions, frequencies):
    angles = positions.float().unsqueeze(-1) * frequencies.float()
    cosine = angles.cos().repeat_interleave(2, -1).to(x.dtype)
    sine = angles.sin().repeat_interleave(2, -1).to(x.dtype)
    pairs = x.unflatten(-1, (-1, 2))
    turned = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return x * cosine + turned * sine


class HorizonAttention(VJEPA2RopeAttention):
    """Keep all pretrained projections and replace only coordinate rotations."""
    def configure_band(self, target_range, local_scale):
        head_dim = self.attention_head_size
        dims = (16, 24, 24) if head_dim == 64 else (8, 12, 12)
        if sum(dims) != head_dim:
            raise ValueError(f'Unsupported V-JEPA 2 head dimension: {head_dim}')
        self.d_dim, self.h_dim, self.w_dim = dims
        self.register_buffer('time_freq', torch.logspace(
            torch.log10(torch.tensor(torch.pi / target_range)),
            torch.log10(torch.tensor(torch.pi / local_scale)), self.d_dim // 2))
        for name, dim in [('height_freq', self.h_dim), ('width_freq', self.w_dim)]:
            self.register_buffer(name, 10000.0 ** (-torch.arange(dim // 2).float() / (dim // 2)))
        self.spacing = None

    def apply_rotary_embeddings(self, qk, pos_ids):
        time, height, width = [value.to(qk.device) for value in pos_ids]
        if time.ndim == 1:
            time, height, width = [value[None, None, :] for value in (time, height, width)]
        if self.spacing is not None:
            time = time * self.spacing[:, None, None]
        chunks = qk.split((self.d_dim, self.h_dim, self.w_dim), dim=-1)
        return torch.cat([_rotate(chunks[0], time, self.time_freq),
                          _rotate(chunks[1], height, self.height_freq),
                          _rotate(chunks[2], width, self.width_freq)], -1)


def install_horizon_attention(model, target_range, local_scale):
    for parent in list(model.modules()):
        for name, module in list(parent.named_children()):
            if isinstance(module, VJEPA2RopeAttention) and not isinstance(module, HorizonAttention):
                replacement = HorizonAttention(module.config, module.hidden_size, module.num_attention_heads)
                replacement.load_state_dict(module.state_dict(), strict=True)
                replacement.configure_band(target_range, local_scale)
                setattr(parent, name, replacement)


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
        values = self.backbone(pixel_values_videos=video.transpose(1, 2),
                               context_head_mask=mask, skip_predictor=True).last_hidden_state
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
        values = self.backbone.predictor(flat, context_mask=[context], target_mask=[target]).last_hidden_state
        return values.reshape(shape)
