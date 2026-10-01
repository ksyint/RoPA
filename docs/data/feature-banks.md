# Indexed feature banks

A bank keeps one T,N,D array per clip and an index recording the source interval, patch grid, model identity and file checksum. Every source video is hashed once per extraction. The cache key combines its SHA-256 hash with the clip interval, weights identity, frame count, uniform sampling rule and full processor settings. Foundation-model identities record the resolved revision or hashes of the local snapshot. The checkpoint architecture fixes patch size and tubelet size.

Extraction saves each completed row immediately. Resume verifies source fingerprints, extraction contracts and existing array hashes before writing. Replacing a video at the same path or changing frame sampling, weights or preprocessing requires a new output directory. The processor snapshot is saved beside the index.

```bash
python ropa.py features extract --manifest data/train.jsonl --checkpoint outputs/ropa/last.pt --output outputs/features --device cuda
python ropa.py features inspect --index outputs/features/index.jsonl --checksum
```

Use `features merge --index BANK1 --index BANK2 --output MERGED` after extracting disjoint shards. Duplicate clip IDs are rejected. Feature arrays remain in their existing locations. The merged index records absolute paths.
