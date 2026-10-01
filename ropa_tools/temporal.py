"""Plan, execute and summarize temporal-band experiments with immutable recipe hashes."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from pprint import pformat
import shlex
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def catalog_layout(paths):
    layout = {Path(path): Path(path) for path in paths}
    directories = sorted({parent for path in layout for parent in path.parents if parent != Path('.')},
                         key=lambda path: (-len(path.parts), str(path)))
    for directory in directories:
        direct = {path for path in layout.values() if path.parent == directory}
        candidates = sorted(key for key, path in layout.items()
                            if directory in path.parents and path.parent != directory)
        for key in candidates:
            if len(direct) >= 2:
                break
            current = layout[key]
            if sum(path.parent == current.parent for path in layout.values()) <= 2:
                continue
            destination = directory / '--'.join(current.relative_to(directory).parts)
            if destination in layout.values():
                raise ValueError(f'Conflicting recipe destination: {destination}')
            layout[key] = destination
            direct.add(destination)
    return layout


def read_recipe(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix == '.py':
        statements = ast.parse(text, filename=str(path)).body
        if (len(statements) != 1 or not isinstance(statements[0], ast.Assign)
                or len(statements[0].targets) != 1
                or not isinstance(statements[0].targets[0], ast.Name)
                or statements[0].targets[0].id != 'cfg'):
            raise ValueError('Python recipes contain one cfg dictionary assignment.')
        value = ast.literal_eval(statements[0].value)
    elif path.suffix in {'.yaml', '.yml'}:
        value = yaml.safe_load(text)
    else:
        raise ValueError(f'Unsupported recipe extension: {path.suffix}')
    if not isinstance(value, dict):
        raise ValueError(f'Recipe must contain a dictionary: {path}')
    return value


def write_recipe(path, value):
    path = Path(path)
    text = ('cfg = ' + pformat(value, width=100, sort_dicts=False) + '\n'
            if path.suffix == '.py' else yaml.safe_dump(value, sort_keys=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.with_suffix('.yaml' if path.suffix == '.py' else '.py').unlink(missing_ok=True)


def recipe_digest(path):
    path = Path(path)
    if path.suffix != '.py':
        return digest(path)
    text = yaml.safe_dump(read_recipe(path), sort_keys=False)
    return hashlib.sha256(text.encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def run_identity(run):
    keys = ('config_sha256', 'manifest_sha256', 'checkpoint_sha256', 'seed')
    inputs = {key: run[key] for key in keys}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def verify_inputs(run):
    for key in ('config', 'manifest', 'checkpoint'):
        path, expected = run[key], run[f'{key}_sha256']
        if path is None and expected is None and key == 'checkpoint':
            continue
        actual = recipe_digest(path) if path and key == 'config' else digest(path) if path else None
        if not path or not expected or actual != expected:
            raise ValueError(f'Study input changed after planning: {key} {path}')
    if run['id'] != run_identity(run):
        raise ValueError('Study input identity differs from the saved plan.')


def discover(catalog, bands=None, offsets=None):
    selected = []
    if Path(catalog).resolve() == ROOT / 'experiments/temporal':
        catalog = ROOT / 'experiments'
    paths = sorted(path for path in Path(catalog).rglob('*') if path.suffix in {'.yaml', '.yml', '.py'})
    identities = [path.with_suffix('') for path in paths]
    if len(identities) != len(set(identities)):
        raise ValueError('Each temporal recipe must have one active configuration file.')
    for path in paths:
        config = read_recipe(path)
        if not isinstance(config, dict) or 'model' not in config:
            continue
        band = config['model'].get('target_range')
        offset = config.get('objective', {}).get('prediction_offset')
        if bands and band not in bands:
            continue
        if offsets and offset not in offsets:
            continue
        selected.append((path, config))
    if not selected:
        raise ValueError('No temporal recipes match the requested selection.')
    return selected


def create_plan(args):
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    runs = []
    root = Path(args.output).resolve()
    manifest_sha256 = digest(manifest)
    checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    checkpoint_sha256 = digest(checkpoint) if checkpoint else None
    for path, config in discover(args.catalog, args.bands, args.offsets):
        config_sha256 = recipe_digest(path)
        for seed in args.seeds:
            run = {'config': str(path.resolve()), 'config_sha256': config_sha256,
                   'manifest': str(manifest), 'manifest_sha256': manifest_sha256,
                   'checkpoint': checkpoint, 'checkpoint_sha256': checkpoint_sha256,
                   'seed': seed}
            identity = run_identity(run)
            directory = root / f'band{config["model"]["target_range"]:g}' / identity[:16] / f'seed{seed}'
            command = [sys.executable, str(ROOT / 'ropa.py'), 'train', '--config', str(path.resolve()),
                       '--data', str(manifest), '--seed', str(seed), '--device', args.device,
                       '--output', str(directory)]
            if checkpoint:
                command.extend(['--checkpoint', checkpoint])
            run.update(id=identity, band=config['model']['target_range'],
                       offset=config['objective']['prediction_offset'],
                       output=str(directory), command=command)
            runs.append(run)
    return {'format': 2, 'root': str(ROOT), 'runs': runs}


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('format') != 2 or not isinstance(plan.get('runs'), list) or not plan['runs']:
        raise ValueError('Expected a nonempty temporal study plan.')
    identities = [row['id'] for row in plan['runs']]
    if len(identities) != len(set(identities)):
        raise ValueError('Study run identifiers must be unique.')
    for run in plan['runs']:
        verify_inputs(run)
    return plan


def run_status(run):
    output = Path(run['output'])
    checkpoint = output / 'last.pt'
    history = output / 'metrics.json'
    if not checkpoint.is_file() or not history.is_file():
        return 'incomplete'
    rows = json.loads(history.read_text())
    if not rows or 'loss' not in rows[-1]:
        return 'incomplete'
    completed = output / 'study-run.json'
    if not completed.is_file():
        return 'unregistered'
    metadata = json.loads(completed.read_text())
    if any(metadata.get(key) != value for key, value in run.items()):
        return 'different_inputs'
    if metadata.get('returncode') != 0:
        return 'failed'
    return 'complete'


def execute(plan, skip_complete=False, keep_going=False):
    results = []
    for run in plan['runs']:
        verify_inputs(run)
        status = run_status(run)
        if skip_complete and status == 'complete':
            results.append({'id': run['id'], 'status': 'skipped'})
            continue
        output = Path(run['output'])
        if (output / 'last.pt').exists():
            raise FileExistsError(f'Choose an empty run output or skip completed runs: {output}')
        output.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with (output / 'console.log').open('w') as stream:
            result = subprocess.run(run['command'], cwd=plan['root'], stdout=stream, stderr=subprocess.STDOUT)
        row = dict(run, returncode=result.returncode, elapsed_seconds=time.monotonic() - start)
        (output / 'study-run.json').write_text(json.dumps(row, indent=2) + '\n')
        results.append(row)
        if result.returncode and not keep_going:
            raise RuntimeError(f'Run failed: {run["id"]}. Inspect {output / "console.log"}')
    return results


def summarize(plan):
    rows = []
    for run in plan['runs']:
        row = {key: run[key] for key in ('id', 'seed', 'band', 'offset', 'output')}
        row['status'] = run_status(run)
        path = Path(run['output']) / 'metrics.json'
        if path.exists():
            history = json.loads(path.read_text())
            if history:
                row.update({f'last_{key}': value for key, value in history[-1].items() if isinstance(value, (int, float))})
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    plan_parser = commands.add_parser('plan')
    plan_parser.add_argument('--manifest', required=True)
    plan_parser.add_argument('--catalog', default=str(ROOT / 'experiments'))
    plan_parser.add_argument('--bands', nargs='+', type=int)
    plan_parser.add_argument('--offsets', nargs='+', type=int)
    plan_parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    plan_parser.add_argument('--checkpoint')
    plan_parser.add_argument('--device', default='cuda')
    plan_parser.add_argument('--output', required=True)
    for name in ('run', 'summarize'):
        child = commands.add_parser(name)
        child.add_argument('--plan', required=True)
        child.add_argument('--output', required=True)
        if name == 'run':
            child.add_argument('--skip-complete', action='store_true')
            child.add_argument('--keep-going', action='store_true')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.operation == 'plan':
        plan = create_plan(args)
        (output / 'study.json').write_text(json.dumps(plan, indent=2) + '\n')
        (output / 'commands.txt').write_text('\n'.join(shlex.join(row['command']) for row in plan['runs']) + '\n')
        print(json.dumps({'runs': len(plan['runs']), 'plan': str(output / 'study.json')}))
        return
    plan = read_plan(args.plan)
    rows = execute(plan, args.skip_complete, args.keep_going) if args.operation == 'run' else summarize(plan)
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if args.operation == 'summarize':
        keys = sorted({key for row in rows for key in row})
        with (output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'runs': len(rows), 'output': str(output)}))
