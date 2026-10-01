# RoPA

**Improving Video Correspondence with Temporal Rotary Embeddings**
Carl S. Kim, Junyoung Koh, Kyeonghun Kim, Kumud Dhabhai, Seunghyeok Hong.

An independent PyTorch implementation of the supplied research manuscript. This repository implements the temporal rotary band, spacing jitter, predictor-aligned Gram loss, temporal composition regularization, and frozen label propagation. It contains no pretrained weights or reproduced benchmark claims.

## Method

- `utils/rope.py`: horizon-derived geometric temporal allocation (Eq. 4), separate temporal/spatial rotary blocks, and per-clip truncated log-uniform spacing jitter. Time is measured in **tubelet steps**.
- `utils/models.py`: a causal video transformer with tubelet embedding, QK normalization, SwiGLU, LayerScale, and an offset-conditioned residual predictor.
- `utils/losses.py`: predicted/target cross-Gram alignment (Eq. 5), composition and identity losses (Eq. 6), and the final-10% Gram schedule.
- `utils/propagation.py`: first-frame plus seven-frame history, local cosine affinity, top-k propagation, temperature 0.07.

The included predictor is a small per-patch residual MLP. Base prediction uses MSE against an earlier, frozen checkpoint. Student and teacher cross-Grams are formed **after prediction** against their own target-frame features. Spatial rotary frequencies use independent base-10000 ladders. The predictor and initialization checkpoint can be replaced for larger experiments.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/smoke.yaml --output outputs/smoke
python -m pytest -q
python eval.py --target_range 64
```

The smoke run performs real optimization on generated moving textures. Its frozen teacher starts from random weights and it verifies execution only. For real clips, store each video as `.npy` with shape `C,T,H,W` (three channels, float32 in `[0,1]` or uint8), using dimensions divisible by tubelet/patch size and equal clip sizes within each batch:

```bash
python train.py --config configs/vit_b.yaml --data data/clips \
  --checkpoint outputs/pretrained/last.pt --device cuda --output outputs/ropa
python inference.py --checkpoint outputs/ropa/last.pt --data data/clips --output outputs/features
```

The initialization checkpoint must use the same model configuration and the repository checkpoint format (`model`, `config`). Public foundation-model checkpoints need architecture/key conversion; they are not automatically compatible. `vit_b.yaml` describes the paper's dimensional layout, but batch size, loss weights, total steps and predictor architecture are practical configuration choices. This single-process runner does not reproduce the paper's 4096 global batch, data curation, pretrained initialization, 40-epoch warmup/cosine schedule, or complete downstream benchmark suite. Supply licensed HowTo100M/Ego4D data, trained initialization and distributed infrastructure for a comparable experiment.

## Frozen correspondence

Store `features` (`T,H*W,D`) and `labels` (`T,H,W`, nonnegative integer IDs, background 0) in an NPZ archive. Features and annotations must share the tubelet timeline and patch grid.

```bash
python eval.py --data data/sequence.npz --output outputs/prediction.npy
```

The evaluator reports foreground patch-grid mean IoU and pixel accuracy, not official DAVIS J&F. The first frame provides the only ground-truth labels used during propagation. The included tests verify geometric endpoints, relative-rotation identity, TSJ bounds, PAGA offset scaling and teacher detachment, temporal causality, and correspondence.

## Citation

The supplied manuscript has no verified public identifier here. Cite its published record when available; no venue or identifier is inferred.
