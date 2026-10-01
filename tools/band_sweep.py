"""Resolve a temporal-band experiment by coordinates and launch the native trainer."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def profile_path(band, jitter, predictor_ratio, gram, rcl):
    return ROOT / 'experiments' / 'temporal' / f'band_{band}' / f'jitter_{jitter}' / f'predictor_x{predictor_ratio}' / f'gram_{gram}' / f'rcl_{rcl}.yaml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', type=int, choices=[64, 160, 640, 1280], default=64)
    parser.add_argument('--jitter', choices=['fixed', 'narrow', 'full'], default='full')
    parser.add_argument('--predictor-ratio', type=int, choices=[2, 4], default=2)
    parser.add_argument('--gram', choices=['0p5', '1p0'], default='1p0')
    parser.add_argument('--rcl', choices=['0p00', '0p01', '0p05', '0p10', '0p20'], default='0p10')
    parser.add_argument('--data')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default='outputs/band_sweep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = profile_path(args.band, args.jitter, args.predictor_ratio, args.gram, args.rcl)
    if not config.is_file():
        parser.error(f'Profile not found: {config}')
    if not args.dry_run and not args.data:
        parser.error('--data is required for a temporal-band experiment.')
    command = [sys.executable, str(ROOT / 'train.py'), '--config', str(config), '--device', args.device,
               '--output', args.output]
    for option in ('data', 'checkpoint'):
        if getattr(args, option):
            command.extend(['--' + option, getattr(args, option)])
    if args.dry_run:
        command.append('--dry-run')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
