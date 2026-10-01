# Indexed feature banks

A bank keeps one T,N,D array per clip and an index recording the source interval, patch grid, model identity and file checksum. Every source video is hashed once per extraction. The cache key combines its SHA-256 hash with the clip interval, weights identity, frame count, uniform sampling rule and full processor settings. Foundation-model identities record the resolved revision or hashes of the local snapshot. The checkpoint architecture fixes patch size and tubelet size.

Extraction saves each completed row immediately. Resume verifies source fingerprints, extraction contracts and existing array hashes before writing. Replacing a video at the same path or changing frame sampling, weights or preprocessing requires a new output directory. The processor snapshot is saved beside the index.

```bash
python ropa.py features extract --manifest data/train.jsonl --checkpoint outputs/ropa/last.pt --output outputs/features --device cuda
python ropa.py features inspect --index outputs/features/index.jsonl --checksum
```

Use `features merge --index BANK1 --index BANK2 --output MERGED` after extracting disjoint shards. Duplicate clip IDs are rejected. Feature arrays remain in their existing locations. The merged index records absolute paths.

Feature extraction can use a longer input clip through `--frames`. The saved index includes selected presentation timestamps and tubelet-center timestamps:

```bash
python ropa.py features extract --manifest prepared/evaluation.jsonl --checkpoint outputs/ropa/last.pt --frames 128 --output features/evaluation
```

A manifest row may use `stride` and `frame_start` to select decoded frames at a fixed stride. `sampling: "time"` selects frames closest to evenly spaced presentation times. `timestamps` supplies the exact requested times in seconds from the video stream origin. The feature-cache identity includes these sampling choices. A temporal gap is measured in saved tubelets, so gap 64 requires at least 65 encoded tubelets.

Action-probe labels can be joined to this index from dataset annotations:

```bash
python ropa.py probe labels --index features/train/index.jsonl --annotations annotations/kinetics_train.csv --dataset kinetics400 --output prepared/action_train.jsonl
python ropa.py probe labels --index features/ssv2/index.jsonl --annotations annotations/ssv2_train.json --dataset ssv2 --class-map annotations/ssv2_labels.json --output prepared/ssv2_train.jsonl
```

Kinetics CSV uses `youtube_id`, `time_start`, `time_end` and `label`. Reuse the same class map for validation. SSv2 uses its template-to-class-ID JSON map. Diving-48 uses a JSON list of `vid_name` and integer `label` entries. Source recording IDs or video filename stems must match exactly one annotation. The resulting JSONL maps `clip_id` to an integer class label.
