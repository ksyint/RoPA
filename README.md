# RoPA

**Improving Video Correspondence with Temporal Rotary Embeddings**

RoPA training and dense video feature extraction with the released **Facebook V-JEPA 2 ViT-g encoder and transformer predictor**. The pretrained encoder supplies 1408-dimensional patch features. The released 12-layer predictor performs temporal latent prediction. HTA replaces temporal rotation frequencies inside the actual transformer attention, TSJ scales clip coordinates, and optimization combines latent prediction, PAGA cross-Gram anchoring, and temporal composition/identity regularization.

## Install

Use Python 3.10+ and CUDA-enabled PyTorch. Model execution accepts `cuda` or `cuda:N`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runtime is pinned to Transformers 4.57.1. The model factory uses `AutoModel` and the checkpoint's `AutoVideoProcessor`. SDPA and CUDA BF16 autocast are enabled. Encoder activation checkpointing reduces training memory. Allocate GPU memory for the trainable ViT-g, frozen anchor, optimizer states, and clip activations.

## Foundation weights

Training automatically downloads [facebook/vjepa2-vitg-fpc64-384](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) on first use. This is Facebook's V-JEPA 2 initialization: a 40-layer, 22-head, 1408-dimensional encoder, patch size 16, tubelet size 2, and a 384-dimensional, 12-layer pretrained predictor. The released crop is 384 pixels. Both encoder and predictor weights participate in the RoPA training graph. Native feature magnitudes are retained at the pretrained predictor input. Prediction, Gram, and composition objectives compare L2-normalized feature vectors.

By default Hugging Face stores snapshots under `~/.cache/huggingface/hub/models--facebook--vjepa2-vitg-fpc64-384/`. `--cache-dir /path/to/cache` selects another cache. For an explicit local snapshot:

```bash
hf download facebook/vjepa2-vitg-fpc64-384 --local-dir weights/vjepa2-vitg-fpc64-384
python train.py --data data/train.jsonl \
  --pretrained weights/vjepa2-vitg-fpc64-384 --offline --device cuda --output outputs/ropa
```

The download supplies foundation initialization. `train.py` creates the RoPA task checkpoint used by downstream inference. A saved task checkpoint includes the Hugging Face architecture configuration, all encoder/predictor weights, and rotary buffers. Restoring its model does not fetch foundation weights again. Keep the processor snapshot available through its saved path or override `--pretrained` with a local snapshot directory.

## Acquire and split videos

Acquire local video files from the official [Ego4D download workflow](https://ego4d-data.org/docs/start-here/) or the [HowTo100M dataset page](https://www.di.ens.fr/willow/research/howto100m/), then place the selected media under `data/videos/`. The training loader consumes video media rather than precomputed dataset features. Keep evaluation-corpus videos outside the pretraining folder. Existing MP4/MKV/MOV/AVI/WebM clips with decodable timestamps can be used directly.

```text
data/
  videos/
    recording_001.mp4
    recording_002.mp4
  train.jsonl
  validation.jsonl
```

Install FFmpeg so `ffprobe` is on PATH. Generate two-second crop manifests and a deterministic split by whole video:

```bash
python tools/prepare_videos.py --videos data/videos --output data \
  --clip-seconds 2 --validation-fraction 0.1 --seed 42
```

Every crop from one source video stays in one split. The helper writes absolute video paths, source IDs, and start/end seconds without duplicating video files. Use `train.jsonl` for optimization and `validation.jsonl` for held-out feature extraction. With related recordings from one session or subject, place those recordings into the same split before training.

Optional codec normalization preserves the full-resolution geometry used by the official processor:

```bash
ffmpeg -i recording.mov -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p data/videos/recording.mp4
```

## Train from videos

A JSONL manifest names real video clips, optionally cropped in seconds. Relative paths resolve from the manifest directory:

```json
{"video":"clips/clip_001.mp4","start":0.0,"end":2.0}
{"video":"clips/clip_002.mp4","start":3.0,"end":5.0}
```

The loader decodes presentation timestamps, samples 16 frames uniformly within each interval, and applies the released resize, center-crop, and ImageNet normalization. A directory of cached `.npy` clips is also accepted. Arrays must already contain processor-normalized floating point pixels in `C,T,H,W` order.

```bash
python train.py --config configs/vjepa2.yaml --data data/train.jsonl \
  --cache-dir weights/cache --device cuda --output outputs/ropa
python train.py --config configs/vjepa2.yaml --data data/train.jsonl \
  --checkpoint outputs/earlier/last.pt --device cuda --output outputs/continued
```

`--checkpoint` initializes a new adaptation run from earlier learned parameters while retaining the selected temporal band. The frozen copy at the start of that run supplies anchor targets. PAGA activates during the final 10% of the configured steps and follows `gram_warmup`. The runnable pretrained path adapts V-JEPA 2 ViT-g with RoPA rotations and causal tubelet attention. The paper also studies matched ViT-B pretraining and reports a separate PAGA retrofit on public V-JEPA 2 ViT-g. Temporal coordinates are measured in tubelet steps. Encoder head allocation is `(16,24,24)`. The 32-channel predictor heads use `(8,12,12)`.

Training writes `last.pt`, `config.json`, and `metrics.json`. `runtime` controls batch size/steps, `optim` controls AdamW, `objective` controls prediction/PAGA/RCL, and `spacing` controls TSJ.

## Temporal experiment catalog

The **240 executable configurations** under `experiments/temporal/` use the same pretrained ViT-g and transformer predictor:

| Axis | Values |
| --- | --- |
| Temporal horizon | 64, 160, 640, 1280 tubelet steps |
| Coordinate spacing | fixed 1, log-uniform `[0.75,1.5]`, log-uniform `[0.5,2.0]` |
| Predicted offset | 2 or 4 tubelet steps |
| PAGA coefficient | 0.5, 1.0 |
| Composition coefficient | 0, 0.01, 0.05, 0.10, 0.20 |

Each selected offset changes the transformer's target mask positions and the latent/PAGA target frame. The predictor's pretrained dimensions remain fixed.

```bash
python tools/band_sweep.py --band 160 --jitter full --prediction-offset 2 \
  --gram 1p0 --rcl 0p10 --data data/train.jsonl --device cuda --output outputs/band160
python tools/band_sweep.py --band 160 --jitter narrow --prediction-offset 4 \
  --gram 0p5 --rcl 0p05 --dry-run
python tools/build_profiles.py
```

Pass any catalog YAML directly to `train.py --config`. Dry-run validates and prints settings without loading models. The builder regenerates profiles from `configs/vjepa2.yaml`.

## Extract and propagate dense features

```bash
python inference.py --checkpoint outputs/ropa/last.pt --data data/validation.jsonl \
  --output outputs/features --device cuda
python inference.py --config configs/vjepa2.yaml --data data/validation.jsonl \
  --cache-dir weights/cache --output outputs/initial_features --device cuda
```

The second command extracts the foundation-initialized RoPA representation before adaptation. Each output is `T_tubelets,N_patches,1408`, with `N_patches=24*24` for the 384-pixel crop.

Prepare a sequence NPZ containing `features` with shape `T,H*W,D` and integer class-ID `labels` with shape `T,H,W`, aligned to the same tubelet timeline and patch grid:

```bash
python eval.py --data data/sequence.npz --output outputs/prediction.npy --device cuda
python eval.py --target_range 160 --device cuda
```

Frozen propagation uses first-frame labels plus seven previous predictions, radius-12 locality, top-10 affinities, and temperature 0.07. The evaluator reports foreground patch-grid mean IoU and pixel accuracy.
