# Temporal studies

Study planning selects existing temporal recipes and expands only the requested seeds, bands and predictor offsets. Each job records recipe, training-manifest and optional initialization-checkpoint SHA-256 hashes. These hashes and the seed determine its identity and output directory. Planning writes commands without initializing a model. Execution verifies every input before launching each job and retains console logs and elapsed time.

```bash
python ropa.py study plan --manifest data/train.jsonl --bands 64 160 --offsets 2 --seeds 42 43 --output outputs/study
python ropa.py study run --plan outputs/study/study.json --output outputs/study_status --skip-complete
```

A completed run needs its task checkpoint, metrics history and an exact matching study record with a successful exit code. Changing the manifest, seed or initialization checkpoint creates a separate run and cannot reuse a previous completion. Existing task artifacts are preserved. Use `study summarize` to write JSON and CSV comparisons of the observed final training objectives.
