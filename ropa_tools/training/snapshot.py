"""Full optimization snapshots that keep the fixed Gram teacher across continuation."""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def dataset_identity(path):
    root = Path(path).resolve()
    if root.is_file():
        return manifest_identity(root)
    files = sorted(root.glob('*.npy'))
    if not files:
        raise ValueError('The training directory has no preprocessed clips.')
    records = [(source.name, digest(source)) for source in files]
    return dict(kind='npy', files=records)


def training_contract(config):
    value = copy.deepcopy(config)
    for field in ('pretrained', 'cache_dir', 'local_files_only', 'hf_config'):
        value['model'].pop(field, None)
    return value


def random_state(device):
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy_name=state[0], numpy_keys=state[1].tolist(), numpy_position=state[2],
        numpy_gaussian=state[3], numpy_cached=state[4],
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
    )


def restore_random(state, device):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'], np.asarray(state['numpy_keys'], dtype=np.uint32),
                         state['numpy_position'], state['numpy_gaussian'], state['numpy_cached']))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def save(path, model, teacher, optimizer, schedule, config, history, stream, identity, device, group=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local_state = dict(stream=stream.state_dict(), random=random_state(device))
    rank_states = group.gather_state(local_state) if group is not None else [local_state]
    if group is not None and not group.primary:
        return
    payload = dict(
        rank_states=rank_states,
        world_size=len(rank_states),
        format='ropa-training-v2',
        model=model.state_dict(),
        teacher=teacher.state_dict(),
        optimizer=optimizer.state_dict(),
        schedule=schedule.state_dict(),
        config=config,
        step=schedule.step_number,
        history=history,
        stream=stream.state_dict(),
        data_identity=identity,
        random=random_state(device),
    )
    temporary = path.with_suffix(path.suffix + '.partial')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore(path, model, teacher, optimizer, schedule, config, stream, identity, device, group=None):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('format') != 'ropa-training-v2':
        raise ValueError('Use --checkpoint to initialize from weights, or --resume with a full training snapshot.')
    if training_contract(state['config']) != training_contract(config):
        raise ValueError('Resume requires matching temporal, objective and optimization configurations.')
    if state['data_identity'] != identity:
        raise ValueError('The training manifest or preprocessed clip contents changed.')
    model.load_state_dict(state['model'], strict=True)
    teacher.load_state_dict(state['teacher'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    world_size = group.world_size if group is not None else 1
    rank = group.rank if group is not None else 0
    if state.get('world_size', 1) != world_size:
        raise ValueError('Exact continuation requires the saved number of CUDA processes.')
    rank_states = state.get('rank_states', [dict(stream=state['stream'], random=state['random'])])
    stream.load_state_dict(rank_states[rank]['stream'])
    restore_random(rank_states[rank]['random'], device)
    return list(state['history'])


def manifest_identity(path):
    sources = {}
    clips = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        filename = str(row['video'])
        if filename not in sources:
            source = Path(filename)
            source = source if source.is_absolute() else path.parent / source
            before = source.stat()
            checksum = digest(source)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('A training video changed while its content identity was being recorded.')
            sources[filename] = dict(sha256=checksum, bytes=after.st_size)
        clips += 1
    if not clips:
        raise ValueError('The temporal training manifest contains no clips.')
    return dict(kind='manifest', sha256=digest(path), clips=clips, videos=sources)
