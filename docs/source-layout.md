# Source layout

`models/backbone.py` keeps the V-JEPA wrapper and checkpoint factory, with rotary geometry and attention adaptation in `models/rotary.py`. `ropa.py` keeps native objectives and training. Extended commands load their modules only when selected.

```bash
python ropa.py features --help
python ropa.py temporal --help
python ropa.py study --help
```

The `ropa_tools` directory holds checkpoint packaging and study planning beside its data and evaluation packages. `ropa_tools/data` groups manifest cataloging, decoded clip loading and indexed feature extraction. `ropa_tools/evaluation` pairs sequence propagation with temporal-gap and point evaluation. Each package keeps related processing in separate modules. Schemas under `schemas/` describe the records they exchange.
