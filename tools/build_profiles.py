"""Regenerate the 240 temporal-band/PAGA/RCL experiment configurations."""
import copy
from itertools import product
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.band_sweep import profile_path
from tools.profile_config import validate_config


BANDS = (64, 160, 640, 1280)
SPACING = {'fixed': dict(enabled=False, low=1.0, high=1.0),
           'narrow': dict(enabled=True, low=0.75, high=1.5),
           'full': dict(enabled=True, low=0.5, high=2.0)}
GRAM = {'0p5': 0.5, '1p0': 1.0}
RCL = {'0p00': 0.0, '0p01': 0.01, '0p05': 0.05, '0p10': 0.1, '0p20': 0.2}


def main():
    baseline = yaml.safe_load((ROOT / 'configs/vjepa2.yaml').read_text())
    count = 0
    for band, jitter, predictor, gram, rcl in product(BANDS, SPACING, (2, 4), GRAM, RCL):
        config = copy.deepcopy(baseline)
        config['model'].update(target_range=float(band))
        config['objective']['prediction_offset'] = predictor
        config['spacing'] = dict(SPACING[jitter])
        config['objective'].update(lambda_gram=GRAM[gram], lambda_rope=RCL[rcl])
        validate_config(config)
        output = profile_path(band, jitter, predictor, gram, rcl)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(config, sort_keys=False))
        count += 1
    print(f'Wrote {count} executable temporal profiles under experiments/temporal.')


if __name__ == '__main__':
    main()
