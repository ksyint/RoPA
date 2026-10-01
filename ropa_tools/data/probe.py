"""Class-labelled feature-bank records with variable-length attentive-probe batches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ropa_tools.data.bank import file_digest, read_index, validate_array


def read_targets(path):
    path = Path(path).resolve()
    targets = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row['clip_id'])
        label = row['label']
        if identifier in targets:
            raise ValueError(f'{path}:{number}: duplicate clip label.')
        if type(label) is not int or label < 0:
            raise ValueError('Probe labels must be nonnegative class integers.')
        targets[identifier] = label
    if not targets:
        raise ValueError('The clip-label manifest is empty.')
    return targets


def bank_fingerprint(rows, labels):
    records = [
        dict(clip_id=row['clip_id'], model_id=row.get('model_id'),
             sha256=row.get('sha256'), label=labels[row['clip_id']])
        for row in sorted(rows, key=lambda row: row['clip_id'])
    ]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class ProbeDataset(Dataset):
    def __init__(self, index, targets, classes, max_tokens=None, training=False, seed=0):
        self.rows = read_index(index)
        self.id_to_index = {row['clip_id']: i for i, row in enumerate(self.rows)}
        self.labels = read_targets(targets)
        ids = {row['clip_id'] for row in self.rows}
        if ids != self.labels.keys():
            raise ValueError('The feature index and class-label manifest must have identical clip IDs.')
        if max(self.labels.values()) >= classes:
            raise ValueError('A class label exceeds the configured class count.')
        if max_tokens is not None and max_tokens < 1:
            raise ValueError('The token sampling limit must be positive.')
        models = {row.get('model_id') for row in self.rows}
        if None in models or len(models) != 1:
            raise ValueError('A probe dataset requires one known frozen backbone identity.')
        dimensions = {row['shape'][-1] for row in self.rows}
        if len(dimensions) != 1:
            raise ValueError('Feature dimensions differ within the probe bank.')
        self.model_id = next(iter(models))
        self.dimension = next(iter(dimensions))
        self.max_tokens = max_tokens
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.classes = classes
        self.inventory = feature_inventory(self.rows)
        self.fingerprint = bank_fingerprint(self.rows, self.labels)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        array = validate_array(row)
        values = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
        signature = feature_signature(row['features'])
        if signature != self.inventory[row['clip_id']]['signature']:
            raise ValueError('A feature archive changed after the probe dataset was prepared.')
        chosen = self.token_indices(index, len(values))
        values = values[chosen]
        return torch.from_numpy(values.copy()), self.labels[row['clip_id']], row['clip_id']

    def token_indices(self, index, count):
        if self.max_tokens is None or count <= self.max_tokens:
            return np.arange(count, dtype=np.int64)
        if self.training:
            generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            return np.sort(generator.choice(count, self.max_tokens, replace=False))
        return np.linspace(0, count - 1, self.max_tokens).round().astype(np.int64)

    def contract(self):
        return dict(
            fingerprint=self.fingerprint,
            model_id=self.model_id,
            dimension=self.dimension,
            classes=self.classes,
            max_tokens=self.max_tokens,
            clips=len(self),
        )


def probe_collate(samples):
    if not samples:
        raise ValueError('Cannot collate an empty feature batch.')
    maximum = max(len(row[0]) for row in samples)
    dimension = samples[0][0].shape[-1]
    features = torch.zeros(len(samples), maximum, dimension)
    mask = torch.ones(len(samples), maximum, dtype=torch.bool)
    labels, identifiers = [], []
    for index, (tokens, label, identifier) in enumerate(samples):
        if tokens.ndim != 2 or tokens.shape[-1] != dimension or len(tokens) == 0:
            raise ValueError('Every feature clip needs at least one token and a shared dimension.')
        features[index, :len(tokens)] = tokens
        mask[index, :len(tokens)] = False
        labels.append(label)
        identifiers.append(identifier)
    return features, mask, torch.tensor(labels, dtype=torch.long), identifiers


def verify_partitions(training, validation):
    if training.model_id != validation.model_id or training.dimension != validation.dimension:
        raise ValueError('Train and validation features must share the frozen model identity.')
    first = {row['clip_id'] for row in training.rows}
    second = {row['clip_id'] for row in validation.rows}
    if first & second:
        raise ValueError('A clip occurs in both probe training and validation.')
    groups = {str(row.get('video_id', row.get('video'))) for row in training.rows}
    heldout = {str(row.get('video_id', row.get('video'))) for row in validation.rows}
    groups.discard('None')
    heldout.discard('None')
    if groups & heldout:
        raise ValueError('A source recording occurs in both probe partitions.')


def annotation_rows(path, dataset, label_map=None):
    import csv
    path = Path(path)
    if dataset == 'kinetics400':
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        required = {'youtube_id', 'time_start', 'time_end', 'label'}
        if not rows or not required <= rows[0].keys():
            raise ValueError('Kinetics CSV needs youtube_id,time_start,time_end,label columns.')
        names = sorted({row['label'] for row in rows})
        mapping = label_map or {name: index for index, name in enumerate(names)}
        result = []
        for row in rows:
            if row['label'] not in mapping:
                raise ValueError(f'Kinetics label is absent from the class map: {row["label"]}')
            start, stop = int(float(row['time_start'])), int(float(row['time_end']))
            if stop <= start:
                raise ValueError('Kinetics annotation has a nonpositive clip interval.')
            result.append(dict(
                keys=[row['youtube_id'], f'{row["youtube_id"]}_{start:06d}_{stop:06d}'],
                label=int(mapping[row['label']]), class_name=row['label'],
                start=start, end=stop,
            ))
        return result, mapping
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError('SSv2 and Diving-48 annotations must be nonempty JSON lists.')
    if dataset == 'ssv2':
        if label_map is None:
            raise ValueError('SSv2 label conversion requires the official template-to-ID JSON map.')
        result = []
        for row in values:
            template = str(row['template']).replace('[', '').replace(']', '')
            if template not in label_map:
                raise ValueError(f'SSv2 template is missing from the label map: {template}')
            result.append(dict(keys=[str(row['id'])], label=int(label_map[template]), class_name=template))
        return result, {str(key): int(value) for key, value in label_map.items()}
    if dataset == 'diving48':
        result = []
        for row in values:
            label = row['label']
            if type(label) is not int or not 0 <= label < 48:
                raise ValueError('Diving-48 labels must be integers between 0 and 47.')
            result.append(dict(keys=[str(row['vid_name'])], label=label, class_name=str(label)))
        return result, {str(index): index for index in range(48)}
    raise ValueError('Choose kinetics400, ssv2 or diving48 annotation conversion.')


def match_annotations(index, annotations, dataset, labels=None):
    rows = read_index(index)
    mapping = json.loads(Path(labels).read_text()) if labels else None
    references, class_names = annotation_rows(annotations, dataset, mapping)
    lookup = {}
    for reference in references:
        for key in reference['keys']:
            lookup.setdefault(key, []).append(reference)
    matched, missing = [], []
    for row in rows:
        keys = []
        if row.get('video_id'):
            keys.append(str(row['video_id']))
        if row.get('video'):
            keys.append(Path(row['video']).stem)
        candidates = []
        seen = set()
        for key in keys:
            for candidate in lookup.get(key, []):
                signature = (candidate['label'], candidate.get('start'), candidate.get('end'))
                if signature not in seen:
                    candidates.append(candidate)
                    seen.add(signature)
        if len(candidates) > 1 and 'start' in row and 'end' in row:
            candidates = [candidate for candidate in candidates
                          if abs(candidate.get('start', row['start']) - row['start']) < .01
                          and abs(candidate.get('end', row['end']) - row['end']) < .01]
        if len(candidates) != 1:
            missing.append(dict(clip_id=row['clip_id'], keys=keys, candidates=len(candidates)))
            continue
        reference = candidates[0]
        matched.append(dict(clip_id=row['clip_id'], label=reference['label'], class_name=reference['class_name']))
    if missing:
        raise ValueError(f'Every feature clip requires one unambiguous annotation: {missing[:20]}')
    return matched, class_names


def prepare_labels(args):
    records, classes = match_annotations(args.index, args.annotations, args.dataset, args.class_map)
    destination = Path(args.output).resolve()
    inputs = {Path(args.index).resolve(), Path(args.annotations).resolve()}
    if destination in inputs:
        raise ValueError('Label output must not replace an annotation or feature index.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    metadata = destination.with_suffix('.classes.json')
    metadata.write_text(json.dumps(dict(dataset=args.dataset, classes=classes, clips=len(records)), indent=2) + '\n')
    return dict(labels=str(destination), classes=str(metadata), clips=len(records))


def feature_signature(path):
    source = Path(path).stat()
    return source.st_size, source.st_mtime_ns, source.st_ino


def feature_inventory(rows):
    result = {}
    for row in rows:
        path = row['features']
        before = feature_signature(path)
        checksum = file_digest(path)
        if before != feature_signature(path):
            raise ValueError(f'A feature archive changed while it was being fingerprinted: {path}')
        if row.get('sha256') is not None and row['sha256'] != checksum:
            raise ValueError(f'Feature contents differ from the recorded extraction checksum: {path}')
        row['sha256'] = checksum
        result[row['clip_id']] = dict(signature=before, sha256=checksum, bytes=before[0])
    return result
