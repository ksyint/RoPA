# RoPA

**Improving Video Correspondence with Temporal Rotary Embeddings**
Carl S. Kim, Junyoung Koh, Kyeonghun Kim, Kumud Dhabhai, Seunghyeok Hong.

PyTorch implementation of temporal rotary allocation, spacing jitter, predictor-aligned Gram anchoring, composition regularization, and frozen video correspondence.

## Model and objective

The encoder uses causal tubelet attention, independent temporal/height/width rotary blocks, QK normalization, SwiGLU, and LayerScale. Time coordinates are measured in tubelet steps. The geometric temporal band spans `pi / target_range` to `pi / local_scale`.

An offset-conditioned residual predictor estimates the next latent frame. Training combines latent MSE against a frozen initialization checkpoint, predicted/target cross-Gram alignment, and temporal composition/identity losses. PAGA averages source anchors and sampled patch pairs while summing prediction offsets. Its weight activates during the final 10% of optimization and follows the configured warmup.

| Component | Location |
| --- | --- |
| Rotary allocation and temporal spacing | `models/layers/rotary.py` |
| Causal video encoder | `models/backbones/video.py` |
| Offset predictor and configurable expansion | `models/heads/predictor.py` |
| Prediction, PAGA and composition objective | `criterion/pretraining.py` |
| Step-based optimization | `engine.py` |
| Frozen affinity propagation | `propagation.py` |

## Prepare data and train

Use a CUDA-enabled PyTorch installation. Model-running commands accept `cuda` or `cuda:N`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Store each clip as a `.npy` array with shape `C,T,H,W`: three channels, float32 values in `[0,1]` or uint8 pixels. Spatial dimensions must be divisible by the patch size, temporal length by the tubelet size, and clips in a batch must have equal shapes.

Prepare an initialization checkpoint containing `model` and `config` entries with matching encoder and predictor dimensions. Initialization loads the learned parameters and constructs rotary frequencies from the selected training configuration. The saved configuration records the named architecture and its explicit overrides.

```bash
python train.py --config configs/vit_b.yaml --data data/clips \
  --checkpoint outputs/pretrained/last.pt --device cuda --output outputs/ropa
python inference.py --checkpoint outputs/ropa/last.pt --data data/clips \
  --output outputs/features --device cuda
```

`configs/vit_b.yaml` selects a 768-dimensional, 12-layer, 12-head encoder with rotary blocks `(16,24,24)`, patch size 16 and tubelet size 2. Configure optimizer settings under `optim`, sampling/steps under `runtime`, and regularization under `objective`.

## Temporal experiment catalog

`experiments/temporal/` contains **240 directly executable training configurations**. Each path identifies a point in the following grid:

| Axis | Values |
| --- | --- |
| Temporal horizon | 64, 160, 640, 1280 tubelet steps |
| Temporal spacing | fixed 1; log-uniform `[0.75,1.5]`; log-uniform `[0.5,2.0]` |
| Predictor hidden expansion | 2× or 4× encoder dimension |
| PAGA coefficient | 0.5, 1.0 |
| Composition coefficient | 0, 0.01, 0.05, 0.10, 0.20 |

These axes control the actual encoder band, coordinate sampling, predictor layers, and optimized losses. The fixed-spacing setting uses one unscaled time coordinate per tubelet. Jittered settings obey the selected band's half-cycle constraint.

Select a profile by experiment coordinates:

```bash
python tools/band_sweep.py --band 160 --jitter full --predictor-ratio 2 \
  --gram 1p0 --rcl 0p10 --data data/clips --checkpoint outputs/pretrained/last.pt \
  --output outputs/band160 --device cuda
```

The same file can be passed to `train.py --config`. A different predictor expansion requires an initialization checkpoint with the matching predictor width. Use `--dry-run` to validate settings and inspect the resolved configuration without loading data or starting optimization:

```bash
python tools/band_sweep.py --band 160 --jitter narrow --predictor-ratio 4 \
  --gram 0p5 --rcl 0p05 --dry-run
python tools/build_profiles.py
```

The builder deterministically regenerates the catalog from the ViT-B configuration. Model factories resolve named defaults, explicit YAML options, and checkpoint state keys; training writes `last.pt`, `config.json`, and `metrics.json` to the selected output directory.

## Correspondence evaluation

Prepare an NPZ archive with `features` of shape `T,H*W,D` and nonnegative class-ID `labels` of shape `T,H,W` (background 0). Both arrays must use the same tubelet timeline and patch grid.

```bash
python eval.py --data data/sequence.npz --output outputs/prediction.npy --device cuda
python eval.py --target_range 160 --device cuda
```

Propagation uses first-frame labels plus seven preceding predictions, radius-12 spatial locality, top-10 affinities, and temperature 0.07. The evaluator reports foreground patch-grid mean IoU and pixel accuracy; the band diagnostic reports rotation displacement and the low-frequency half-cycle.
