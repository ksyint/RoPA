# Held-out exclusion

Training recordings must be separate from correspondence evaluation corpora. Identity screening compares video paths and recording or session keys. Optional perceptual screening samples eight timestamped frames per clip and compares 64-bit low-frequency image hashes. A BK-tree retrieves held-out hashes within the selected Hamming distance.

```bash
python ropa.py manifest --manifest data/train.jsonl --heldout data/heldout.jsonl --perceptual --hash-distance 8 --output outputs/disjoint
python ropa.py features screen --index outputs/train_features/index.jsonl --heldout outputs/heldout_features/index.jsonl --threshold 0.95 --output outputs/feature_screen --device cuda
```

Feature screening uses the mean patch embedding and cosine similarity on CUDA. Both banks must come from the same checkpoint. Review the removed identifiers and use the accepted manifest or index for the next stage.
