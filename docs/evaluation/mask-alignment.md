# Mask alignment

The sequence evaluator accepts labels already aligned to the processor patch grid or a PNG annotation directory. For PNG input, supply one `frame_indices` entry for every feature tubelet. Those indices refer to the sorted original annotation filenames. Match them to the timestamps used during feature extraction.

```bash
python ropa.py sequences --sequences data/evaluation/sequences.jsonl --output outputs/segmentation --device cuda
```

Mask projection resizes the shorter image edge, applies a centered crop and downsamples with nearest-neighbor interpolation. Set `resize_shorter` and `crop_size` to the processor geometry. `examples/evaluation/frame-annotations.jsonl` shows the fields. For pre-aligned arrays, labels must be integer T,H,W and features must be T,H*W,D.
