# Temporal studies

Study planning selects existing temporal recipes and expands only the requested seeds, bands and predictor offsets. Each job records recipe, training-manifest and optional initialization-checkpoint SHA-256 hashes. These hashes and the seed determine its identity and output directory. Planning writes commands without initializing a model. Execution verifies every input before launching each job and retains console logs and elapsed time.

```bash
python ropa.py study plan --manifest data/train.jsonl --bands 64 160 --offsets 2 --seeds 42 43 --output outputs/study
python ropa.py study run --plan outputs/study/study.json --output outputs/study_status --skip-complete
```

A completed run needs its task checkpoint, metrics history and an exact matching study record with a successful exit code. Changing the manifest, seed or initialization checkpoint creates a separate run and cannot reuse a previous completion. Existing task artifacts are preserved. Use `study summarize` to write JSON and CSV comparisons of the observed final training objectives.

Training now preserves the fixed teacher, AdamW state, cosine schedule, random generators and exact clip-stream position in `last.pt`:

```bash
python ropa.py train --config vjepa2.yaml --data prepared/train.jsonl --output outputs/ropa
python ropa.py train --config vjepa2.yaml --data prepared/train.jsonl --resume outputs/ropa/last.pt --output outputs/ropa
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 ropa.py train --config vjepa2.yaml --data prepared/train.jsonl --output outputs/ropa
```

Under `torchrun`, each process uses its `LOCAL_RANK` CUDA device. The checkpoint contains one stream and random-generator state per rank. Exact continuation keeps the process count, data, total schedule and accumulation settings unchanged. The last partial distributed batch repeats initial epoch examples as needed to give every rank the same local batch size.

`runtime.accumulation_steps`, `runtime.checkpoint_interval` and `runtime.log_interval` control optimizer windows and output frequency. `optim.warmup_steps` and `optim.minimum_lr_ratio` set the warmup/cosine schedule. `--checkpoint` initializes the student weights. `--teacher-checkpoint` supplies a separate earlier fixed teacher. A full `--resume` restores the teacher already stored in that run.

`objective.prediction_offsets: [1, 2, 4]` enables several temporal offsets. Prediction errors are averaged over offsets, and PAGA terms are summed over offsets. `objective.consistency_pairs: [[1, 1], [1, 2]]` selects the composition pairs used by this mode.

```bash
python ropa.py composition --checkpoint outputs/ropa/last.pt --index features/index.jsonl --pairs 1,1 1,2 2,2 --output results/composition.json
python ropa.py affinity --index features/index.jsonl --gaps 1 2 4 --output results/affinity.json
```

`composition` accepts a feature index extracted from its selected task checkpoint. Feature metadata carries the extraction weight identity, which is checked before predictor composition is measured. Use the same checkpoint for feature extraction and composition analysis.
