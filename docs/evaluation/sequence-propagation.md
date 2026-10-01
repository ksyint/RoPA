# Sequence propagation

Sequence evaluation propagates first-frame instance labels through frozen features. Seven preceding frames and the first frame form the default context. Locality is radius 12 with top-10 affinities and temperature 0.07. The first annotated frame initializes propagation and is excluded from measurements.

```bash
python ropa.py sequences --sequences data/evaluation/sequences.jsonl --history 7 --topk 10 --temperature 0.07 --radius 12 --output outputs/sequence_scores --device cuda
```

The output contains predicted patch-grid masks, per-frame per-instance region IoU and boundary F, plus sequence summaries. Boundary tolerance is a fraction of the patch-grid diagonal. These measurements use the supplied processor-aligned grid and its instance IDs.
