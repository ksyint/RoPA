"""One-query attentive classification over frozen spatiotemporal tokens."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentiveProbe(nn.Module):
    def __init__(self, dimension, classes, heads=8, dropout=0.):
        super().__init__()
        if dimension < 1 or classes < 2 or heads < 1 or dimension % heads:
            raise ValueError('Probe dimension must be divisible by its positive head count.')
        if not 0 <= dropout < 1:
            raise ValueError('Probe dropout must lie in [0,1).')
        self.dimension = int(dimension)
        self.classes = int(classes)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.query = nn.Parameter(torch.empty(1, 1, dimension))
        self.query_norm = nn.LayerNorm(dimension)
        self.token_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, classes)
        nn.init.trunc_normal_(self.query, std=.02)

    def configuration(self):
        return dict(dimension=self.dimension, classes=self.classes, heads=self.heads, dropout=self.dropout)

    def forward(self, features, mask=None, return_attention=False):
        if features.ndim == 4:
            features = features.flatten(1, 2)
        if features.ndim != 3 or features.shape[-1] != self.dimension:
            raise ValueError('Probe features need B,N,D or B,T,N,D dimensions.')
        if features.device.type != 'cuda':
            raise ValueError('Attentive probes require CUDA feature batches.')
        if mask is not None:
            if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
                raise ValueError('Padding mask must be Boolean B,N.')
            if mask.all(-1).any():
                raise ValueError('Every clip must contain at least one unmasked token.')
        tokens = self.token_norm(features.detach())
        query = self.query_norm(self.query.expand(len(features), -1, -1))
        pooled, weights = self.attention(
            query, tokens, tokens,
            key_padding_mask=mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        logits = self.classifier(self.output_norm(pooled[:, 0]))
        return (logits, weights) if return_attention else logits

    @torch.no_grad()
    def attention_summary(self, features, mask=None):
        mode = self.training
        self.eval()
        try:
            logits, weights = self(features, mask, return_attention=True)
            probabilities = weights[:, :, 0].float()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            maximum = probabilities.max(-1).values
            return dict(
                prediction=logits.argmax(-1),
                entropy=entropy,
                maximum_weight=maximum,
                effective_tokens=entropy.exp(),
                attention=probabilities,
            )
        finally:
            self.train(mode)


class ClassificationMeter:
    def __init__(self, classes, device):
        self.classes = classes
        self.confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
        self.count = 0
        self.loss = 0.
        self.top5 = 0

    @torch.no_grad()
    def update(self, logits, labels):
        if logits.shape != (len(labels), self.classes):
            raise ValueError('Classification logits and target dimensions differ.')
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError('Classification target is outside the class range.')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Classification logits are nonfinite.')
        prediction = logits.argmax(-1)
        self.confusion.add_(torch.bincount(
            labels * self.classes + prediction, minlength=self.classes ** 2,
        ).reshape(self.classes, self.classes))
        self.loss += float(F.cross_entropy(logits, labels, reduction='sum'))
        self.count += len(labels)
        best = logits.topk(min(5, self.classes), -1).indices
        self.top5 += int((best == labels[:, None]).any(-1).sum())

    def report(self):
        if not self.count:
            raise ValueError('The classification partition is empty.')
        counts = self.confusion.sum(-1)
        correct = self.confusion.diag()
        present = counts > 0
        recall = correct / counts.clamp_min(1)
        return dict(
            samples=self.count,
            loss=self.loss / self.count,
            top1=float(correct.sum()) / self.count,
            top5=self.top5 / self.count,
            mean_class_accuracy=float(recall[present].mean()),
            class_counts=counts.tolist(),
            class_recall=recall.tolist(),
            confusion=self.confusion.tolist(),
        )


def cosine_rate(step, total, peak, warmup=0, minimum=0.):
    if total < 1 or not 0 <= warmup < total:
        raise ValueError('Warmup must be shorter than the probe schedule.')
    if peak <= 0 or not 0 <= minimum <= peak:
        raise ValueError('Learning-rate bounds are invalid.')
    if step < warmup:
        return peak * (step + 1) / warmup
    fraction = min(1., max(0., (step - warmup) / max(1, total - warmup - 1)))
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * fraction))
