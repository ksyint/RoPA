# Clip manifests

Clip rows identify a video and a half-open timestamp interval. Relative paths resolve from the manifest location. `video_id` groups all crops from one recording. Add `session` when several recordings belong to one subject or capture session. The inspector verifies paths and intervals, counts duplicate clips and can call ffprobe for frame rate and duration. Grouped splitting assigns whole recordings or sessions to one partition.

```bash
python ropa.py manifest --manifest data/train.jsonl --output outputs/manifest --inspect-media --split --group-key session
```

Replace the paths in `examples/clips/` with acquired media. The accepted and excluded manifests use absolute paths so they can be moved independently of the original input file. Inspect `audit.json` before providing the accepted manifest to training.
