# Source layout

`models/backbone.py` retains the released V-JEPA attention adaptation and checkpoint factory. `video.py` handles decoding and frozen propagation. `ropa.py` keeps native objectives and training. Extended commands load their modules only when selected.

```bash
python ropa.py features --help
python ropa.py temporal --help
python ropa.py study --help
```

The nested branches are `ropa_tools/data/manifests`, `data/features`, `evaluation/segmentation`, `evaluation/temporal`, `experiments/checkpoints` and `experiments/studies`. Each branch owns its input contract, processing and command options. Schemas under `schemas/` describe records exchanged between these stages.
