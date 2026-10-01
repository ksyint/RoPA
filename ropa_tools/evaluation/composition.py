"""Measure direct, composed and identity predictions for a saved temporal predictor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import cuda_device, load_checkpoint_model
from ropa_tools.data.bank import dictionary_digest, file_digest, read_index, validate_array


@torch.no_grad()
def composition_statistics(model, features, pairs, target_range):
    if features.device.type != 'cuda' or features.ndim != 3:
        raise ValueError('Composition features must have T,N,D dimensions on CUDA.')
    model.eval()
    model._spacing = None
    rows = []
    identity = model.predictor(features, 0)
    identity_error = (identity.float() - features.float()).square().sum(-1)
    normalized_identity = F.normalize(identity.float(), dim=-1)
    normalized_source = F.normalize(features.float(), dim=-1)
    for first, second in pairs:
        if min(first, second) < 0 or first + second > target_range:
            raise ValueError('Composition offsets must be nonnegative and stay within the temporal band.')
        direct = model.predictor(features, first + second)
        composed = model.predictor(model.predictor(features, first), second)
        error = (direct.float() - composed.float()).square().sum(-1)
        normalized_direct = F.normalize(direct.float(), dim=-1)
        normalized_composed = F.normalize(composed.float(), dim=-1)
        angular = (normalized_direct - normalized_composed).square().sum(-1)
        rows.append(dict(
            first=first, second=second, combined=first + second,
            squared_l2_mean=float(error.mean()),
            squared_l2_max=float(error.max()),
            normalized_squared_l2=float(angular.mean()),
            cosine_similarity=float((normalized_direct * normalized_composed).sum(-1).mean()),
            direct_feature_norm=float(direct.float().norm(dim=-1).mean()),
            composed_feature_norm=float(composed.float().norm(dim=-1).mean()),
        ))
    return dict(
        tubelets=len(features), patches=features.shape[1],
        identity_squared_l2=float(identity_error.mean()),
        identity_normalized_squared_l2=float((normalized_identity - normalized_source).square().sum(-1).mean()),
        compositions=rows,
    )


def parse_pairs(values):
    result = []
    for value in values:
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError('Composition offsets use first,second notation.')
        pair = tuple(map(int, parts))
        if pair in result:
            raise ValueError('Composition offset pairs must be unique.')
        result.append(pair)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--pairs', nargs='+', default=['1,1', '1,2', '2,2'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = load_checkpoint_model(state).to(device).eval()
    pairs = parse_pairs(args.pairs)
    records = []
    checkpoint_hash = file_digest(args.checkpoint)
    for row in read_index(args.index):
        verify_checkpoint_features(row, state, checkpoint_hash)
        array = validate_array(row)
        features = torch.from_numpy(np.array(array, copy=True)).float().to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            report = composition_statistics(model, features, pairs, state['config']['model']['target_range'])
        records.append(dict(clip_id=row['clip_id'], **report))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(checkpoint=args.checkpoint, clips=records), indent=2) + '\n')


def verify_checkpoint_features(row, state, checkpoint_hash):
    contract = row.get('contract', {})
    weights = contract.get('weights')
    if weights is not None:
        if weights != {'checkpoint_sha256': checkpoint_hash}:
            raise ValueError('Composition analysis requires features extracted from the selected checkpoint.')
    else:
        expected = dictionary_digest(dict(weights={'checkpoint_sha256': checkpoint_hash}, model=state['config']['model']))
        if row.get('model_id') != expected:
            raise ValueError('The feature model identity differs from this predictor checkpoint.')
    if not row.get('model_id') or row['model_id'] != contract.get('model_id', row['model_id']):
        raise ValueError('Feature row and extraction contract have different model identities.')
    width = state['config']['model'].get('hf_config', {}).get('hidden_size')
    if width is not None and row['shape'][-1] != width:
        raise ValueError('Predictor width differs from the archived feature channels.')
