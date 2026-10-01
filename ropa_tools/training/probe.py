"""Train and evaluate the one-query classifier on frozen CUDA feature banks."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.backbone import cuda_device
from models.probe import AttentiveProbe, ClassificationMeter, cosine_rate
from ropa_tools.data.probe import ProbeDataset, probe_collate, verify_partitions


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_loader(dataset, batch_size, training, generator=None):
    if batch_size < 1:
        raise ValueError('Probe batch size must be positive.')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training,
        collate_fn=probe_collate, pin_memory=True, generator=generator,
    )


@torch.no_grad()
def evaluate(probe, loader, device, destination=None):
    probe.eval()
    meter = ClassificationMeter(probe.classes, device)
    rows = []
    for features, padding, labels, identifiers in loader:
        features = features.to(device, non_blocking=True)
        padding = padding.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = probe(features, padding)
        meter.update(logits, labels)
        if destination:
            for identifier, target, scores in zip(identifiers, labels.tolist(), logits.tolist()):
                rows.append(dict(clip_id=identifier, target=target, logits=scores))
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return meter.report()


def train(args):
    device = cuda_device(args.device)
    torch.manual_seed(args.seed)
    training = ProbeDataset(args.index, args.labels, args.classes, args.max_tokens, True, args.seed)
    validation = ProbeDataset(args.validation_index, args.validation_labels, args.classes, args.max_tokens)
    verify_partitions(training, validation)
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_loader(training, args.batch_size, True, generator)
    val_loader = make_loader(validation, args.batch_size, False)
    probe = AttentiveProbe(training.dimension, args.classes, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total = args.epochs * len(loader)
    warmup = args.warmup_epochs * len(loader)
    cosine_rate(0, total, args.learning_rate, warmup)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, start, best = [], 0, -1.
    contract = dict(train=training.contract(), validation=validation.contract())
    settings = dict(epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs, batch_size=args.batch_size, seed=args.seed)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state.get('format') != 'ropa-probe-v1':
            raise ValueError('Probe continuation requires a complete attentive-probe snapshot.')
        if not 0 <= state['epoch'] < args.epochs:
            raise ValueError('The resumed probe must precede the requested final epoch.')
        if state['contract'] != contract or state['settings'] != settings:
            raise ValueError('Probe continuation requires identical data and optimization settings.')
        if state['probe_config'] != probe.configuration():
            raise ValueError('Probe continuation architecture differs.')
        probe.load_state_dict(state['probe'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['loader_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        history, start, best = state['history'], state['epoch'], state['best_top1']
    for epoch in range(start, args.epochs):
        training.set_epoch(epoch)
        probe.train()
        meter = ClassificationMeter(args.classes, device)
        for batch, (features, padding, labels, _) in enumerate(loader):
            step = epoch * len(loader) + batch
            rate = cosine_rate(step, total, args.learning_rate, warmup)
            for group in optimizer.param_groups:
                group['lr'] = rate
            features, padding, labels = features.to(device), padding.to(device), labels.to(device)
            logits = probe(features, padding)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Probe loss is nonfinite.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            meter.update(logits.detach(), labels)
        scores = evaluate(probe, val_loader, device)
        history.append(dict(epoch=epoch + 1, training=meter.report(), validation=scores, learning_rate=rate))
        improved = scores['top1'] > best
        best = max(best, scores['top1'])
        state = dict(
            format='ropa-probe-v1', probe=probe.state_dict(), probe_config=probe.configuration(),
            optimizer=optimizer.state_dict(), contract=contract, settings=settings,
            epoch=epoch + 1, best_top1=best, history=history,
            loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
        )
        if improved:
            atomic_save(output / 'best.pt', state)
        atomic_save(output / 'last.pt', state)
        (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(history[-1]), flush=True)


def test(args):
    device = cuda_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get('format') != 'ropa-probe-v1':
        raise ValueError('Use a saved attentive-probe checkpoint.')
    configuration = state['probe_config']
    dataset = ProbeDataset(args.index, args.labels, configuration['classes'], args.max_tokens)
    expected = state['contract']['train']
    if dataset.dimension != expected['dimension'] or dataset.model_id != expected['model_id']:
        raise ValueError('Evaluation features differ from the frozen backbone used to train the probe.')
    probe = AttentiveProbe(**configuration).to(device)
    probe.load_state_dict(state['probe'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.multi_view:
        report = evaluate_views(probe, make_loader(dataset, args.batch_size, False), device, args.multi_view)
        scores = report['summary']
        (output / 'recordings.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        scores = evaluate(probe, make_loader(dataset, args.batch_size, False), device, output / 'predictions.jsonl')
    (output / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    training = commands.add_parser('train')
    testing = commands.add_parser('evaluate')
    attention = commands.add_parser('attention')
    attention.add_argument('--checkpoint', required=True)
    attention.add_argument('--top-tokens', type=int, default=20)
    preparation = commands.add_parser('labels')
    preparation.add_argument('--index', required=True)
    preparation.add_argument('--annotations', required=True)
    preparation.add_argument('--dataset', choices=('kinetics400', 'ssv2', 'diving48'), required=True)
    preparation.add_argument('--class-map')
    preparation.add_argument('--output', required=True)
    for command in (training, testing, attention):
        command.add_argument('--index', required=True)
        command.add_argument('--labels', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--max-tokens', type=int)
        command.add_argument('--device', default='cuda')
    training.add_argument('--validation-index', required=True)
    training.add_argument('--validation-labels', required=True)
    training.add_argument('--classes', type=int, required=True)
    training.add_argument('--heads', type=int, default=8)
    training.add_argument('--dropout', type=float, default=0.)
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--warmup-epochs', type=int, default=2)
    training.add_argument('--learning-rate', type=float, default=.001)
    training.add_argument('--weight-decay', type=float, default=.05)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--resume')
    testing.add_argument('--checkpoint', required=True)
    testing.add_argument('--multi-view', choices=('probability', 'logit'))
    args = parser.parse_args(argv)
    if args.operation == 'train':
        train(args)
    elif args.operation == 'evaluate':
        test(args)
    elif args.operation == 'attention':
        if args.top_tokens < 1:
            parser.error('Attention token count must be positive.')
        attention_export(args)
    else:
        from ropa_tools.data.probe import prepare_labels
        print(json.dumps(prepare_labels(args)))


@torch.no_grad()
def attention_export(args):
    device = cuda_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get('format') != 'ropa-probe-v1':
        raise ValueError('Attention export requires an attentive-probe snapshot.')
    configuration = state['probe_config']
    dataset = ProbeDataset(args.index, args.labels, configuration['classes'], args.max_tokens)
    if dataset.model_id != state['contract']['train']['model_id']:
        raise ValueError('Attention features differ from the probe backbone identity.')
    probe = AttentiveProbe(**configuration).to(device).eval()
    probe.load_state_dict(state['probe'])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for features, padding, labels, identifiers in make_loader(dataset, args.batch_size, False):
        features, padding = features.to(device), padding.to(device)
        measured = probe.attention_summary(features, padding)
        for index, identifier in enumerate(identifiers):
            valid = ~padding[index]
            weights = measured['attention'][index, :, valid]
            order = weights.mean(0).argsort(descending=True, stable=True)
            chosen = order[:min(args.top_tokens, len(order))]
            source = dataset.rows[dataset.id_to_index[identifier]]
            time, patches, _ = source['shape']
            sampled_indices = dataset.token_indices(dataset.id_to_index[identifier], time * patches)
            original_indices = sampled_indices[chosen.cpu().numpy()]
            coordinates = [dict(frame=int(token // patches),
                                y=int((token % patches) // source['grid_width']),
                                x=int(token % source['grid_width'])) for token in original_indices]
            row = dict(
                clip_id=identifier, label=int(labels[index]),
                prediction=int(measured['prediction'][index]),
                tokens=int(valid.sum()),
                per_head_entropy=measured['entropy'][index].tolist(),
                per_head_effective_tokens=measured['effective_tokens'][index].tolist(),
                maximum_weight=measured['maximum_weight'][index].tolist(),
                top_token_indices=original_indices.tolist(),
                top_token_coordinates=coordinates,
                top_token_weights=weights[:, chosen].tolist(),
            )
            rows.append(row)
    (destination / 'attention.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (destination / 'contract.json').write_text(json.dumps(dataset.contract(), indent=2) + '\n')
    return rows


@torch.no_grad()
def evaluate_views(probe, loader, device, reduction='probability'):
    probe.eval()
    if reduction not in ('probability', 'logit'):
        raise ValueError('Multi-view reduction must average probabilities or logits.')
    groups = {}
    for features, padding, labels, identifiers in loader:
        logits = probe(features.to(device), padding.to(device)).float()
        values = logits.softmax(-1) if reduction == 'probability' else logits
        for index, identifier in enumerate(identifiers):
            group_name = str(loader.dataset.rows[loader.dataset.id_to_index[identifier]].get('video_id', identifier))
            label = int(labels[index])
            if group_name not in groups:
                groups[group_name] = dict(total=values[index].clone(), label=label, views=1, clips=[identifier])
            else:
                group = groups[group_name]
                if group['label'] != label:
                    raise ValueError('Multiple clips of one recording must share the same action label.')
                group['total'].add_(values[index])
                group['views'] += 1
                group['clips'].append(identifier)
    meter = ClassificationMeter(probe.classes, device)
    records = []
    for name, group in sorted(groups.items()):
        values = group['total'] / group['views']
        logits = values.clamp_min(1e-12).log() if reduction == 'probability' else values
        label = torch.tensor([group['label']], device=device)
        meter.update(logits[None], label)
        records.append(dict(video_id=name, label=group['label'], views=group['views'], clips=group['clips'],
                            prediction=int(values.argmax()), scores=values.tolist()))
    return dict(reduction=reduction, summary=meter.report(), recordings=records)
