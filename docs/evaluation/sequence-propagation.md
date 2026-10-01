# Sequence propagation

Sequence evaluation propagates first-frame instance labels through frozen features. Seven preceding frames and the first frame form the default context. Locality is radius 12 with top-10 affinities and temperature 0.07. The first annotated frame initializes propagation and is excluded from measurements.

```bash
python ropa.py sequences --sequences data/evaluation/sequences.jsonl --history 7 --topk 10 --temperature 0.07 --radius 12 --output outputs/sequence_scores --device cuda
```

The output contains predicted patch-grid masks, per-frame per-instance region IoU and boundary F, plus sequence summaries. Boundary tolerance is a fraction of the patch-grid diagonal. These measurements use the supplied processor-aligned grid and its instance IDs.

Propagation uses query chunks so the temporary affinity matrix scales with the chunk size:

```bash
python ropa.py sequences --sequences prepared/sequences.jsonl --query-chunk 256 --context-stride 1 --png --output results/davis
python ropa.py parsing --sequences prepared/semantic.jsonl --classes 124 --ignore 255 --output results/vspw.json
```

The default sequence summary weights foreground objects equally. `--aggregation frames` retains a pooled object-frame average. Region and contour recall use the fraction of frame scores above 0.5. Decay compares the first and last quarters of each object's measured frames. `--exclude-last-frame` removes the final frame from the object summary. `--png` exports indexed-color masks alongside the prediction arrays.

Semantic propagation reports the confusion matrix, per-class IoU, pixel accuracy and temporal consistency over 8- and 16-frame windows. Ignored semantic pixels do not contribute to the confusion counts.

```bash
python ropa.py probe train --index features/train/index.jsonl --labels prepared/action_train.jsonl --validation-index features/validation/index.jsonl --validation-labels prepared/action_validation.jsonl --classes 400 --epochs 20 --output outputs/probe
python ropa.py probe evaluate --checkpoint outputs/probe/best.pt --index features/test/index.jsonl --labels prepared/action_test.jsonl --multi-view probability --output results/action
python ropa.py probe attention --checkpoint outputs/probe/best.pt --index features/test/index.jsonl --labels prepared/action_test.jsonl --top-tokens 20 --output results/attention
```

The probe learns one attention query and a class head over detached frozen features. Feature banks must share the same model identity, and training and validation recording IDs must be disjoint. Optional `--max-tokens` samples training tokens with a seed and epoch dependent generator and selects evenly spaced validation tokens. Multi-view evaluation averages clip probabilities or logits within a source recording.

`ropa.py intervals` computes paired sequence confidence intervals from completed metrics. `ropa.py transfer` combines task summaries for DAVIS, TAP-Vid, JHMDB, VSPW, Kinetics-400, SSv2 and Diving-48 without rerunning any model.

Instance masks reserve label 255 for void pixels by default. Set `void_label` in a sequence record to change it. Void pixels contribute no seed-label mass, are excluded from region scores, and have their adjacent contour pixels removed from boundary matching. Foreground IDs are mapped to compact probability channels and mapped back for exported masks.

Probe preparation fingerprints every feature archive and verifies any checksum recorded during extraction. Training and resumed training therefore use the same feature contents. Attention exports preserve the original flattened token indices after token sampling and also report each selected token's frame, row and column.
