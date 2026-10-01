# Portable checkpoints

The RoPA task checkpoint contains encoder and predictor tensors, rotary buffers and its Hugging Face architecture configuration. A portable package adds the original video processor JSON files. The package manifest hashes the checkpoint and every included processor file.

```bash
python ropa.py checkpoint bundle --checkpoint outputs/ropa/last.pt --processor weights/vjepa2-vitg-fpc64-384 --destination exports/ropa
python ropa.py checkpoint verify --bundle exports/ropa
```

To extract features offline, pass the packaged checkpoint with `--pretrained exports/ropa/processor --offline`. The embedded architecture constructs the network while the local processor directory supplies the preprocessing configuration. Keep the package manifest alongside the files.
