"""Streaming semantic confusion matrices for frozen VSPW label propagation."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.backbone import cuda_device
from ropa_tools.data.loading import propagate_labels
from ropa_tools.evaluation.sequence import load_sequence, load_sequences


class ParsingMeter:
    def __init__(self, classes, device, ignore=255):
        if classes < 2 or 0 <= ignore < classes:
            raise ValueError('Use at least two semantic classes and an external ignore label.')
        self.classes = classes
        self.ignore = ignore
        self.confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
        self.frames = 0
        self.ignored = 0

    @torch.no_grad()
    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError('Semantic prediction and target maps must share dimensions.')
        valid = target != self.ignore
        if valid.any():
            actual = target[valid].long()
            predicted = prediction[valid].long()
            if actual.min() < 0 or actual.max() >= self.classes:
                raise ValueError('Reference semantic class is outside the label space.')
            if predicted.min() < 0 or predicted.max() >= self.classes:
                raise ValueError('Predicted semantic class is outside the label space.')
            counts = torch.bincount(actual * self.classes + predicted, minlength=self.classes ** 2)
            self.confusion.add_(counts.reshape(self.classes, self.classes))
        self.ignored += int((~valid).sum())
        self.frames += 1

    def report(self):
        target = self.confusion.sum(1)
        predicted = self.confusion.sum(0)
        intersection = self.confusion.diag()
        union = target + predicted - intersection
        active = union > 0
        present = target > 0
        if not present.any():
            raise ValueError('No labelled semantic pixels were evaluated.')
        iou = intersection.double() / union.clamp_min(1)
        accuracy = intersection.double() / target.clamp_min(1)
        frequency = target.double() / target.sum()
        return dict(
            frames=self.frames,
            pixels=int(target.sum()),
            ignored_pixels=self.ignored,
            mean_iou=float(iou[active].mean()),
            mean_accuracy=float(accuracy[present].mean()),
            pixel_accuracy=float(intersection.sum()) / int(target.sum()),
            frequency_weighted_iou=float((frequency * iou).sum()),
            mean_iou_classes='nonzero reference-or-prediction union',
            per_class=[dict(class_id=i, target=int(target[i]), predicted=int(predicted[i]),
                            intersection=int(intersection[i]), union=int(union[i]),
                            iou=float(iou[i]) if union[i] else None) for i in range(self.classes)],
            confusion=self.confusion.tolist(),
        )


@torch.no_grad()
def evaluate(row, device, classes, ignore, options):
    features, labels = load_sequence(row, device)
    first = labels[0].flatten()
    valid = first != ignore
    if not valid.any():
        raise ValueError('The first semantic annotation has no labelled pixels.')
    if first[valid].max() >= classes:
        raise ValueError('The first annotation exceeds the semantic class count.')
    encoded = F.one_hot(first.masked_fill(~valid, 0), classes).float()
    encoded[~valid] = 0.
    predictions = propagate_labels(features, encoded, labels.shape[1], labels.shape[2], **options)
    predicted = predictions.argmax(-1).reshape_as(labels)
    meter = ParsingMeter(classes, device, ignore)
    for index in range(1, len(labels)):
        meter.update(predicted[index], labels[index])
    meter.consistency = video_consistency(predicted, labels, ignore=ignore)
    return meter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--classes', type=int, default=124)
    parser.add_argument('--ignore', type=int, default=255)
    parser.add_argument('--history', type=int, default=7)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--radius', type=int, default=12)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    total = ParsingMeter(args.classes, device, args.ignore)
    options = dict(history=args.history, topk=args.topk, temperature=args.temperature, radius=args.radius)
    sequences = {}
    for row in load_sequences(args.sequences):
        meter = evaluate(row, device, args.classes, args.ignore, options)
        total.confusion.add_(meter.confusion)
        total.frames += meter.frames
        total.ignored += meter.ignored
        sequences[row['name']] = dict(meter.report(), temporal_consistency=meter.consistency)
    consistency = summarize_consistency([row['temporal_consistency'] for row in sequences.values()])
    report = dict(summary=total.report(), sequences=sequences, ignore=args.ignore, options=options,
                  temporal_consistency=consistency)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))


@torch.no_grad()
def video_consistency(prediction, target, windows=(8, 16), ignore=255):
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError('Video consistency requires matching T,H,W semantic maps.')
    if not windows or min(windows) < 2 or len(set(windows)) != len(windows):
        raise ValueError('Video consistency windows must be distinct lengths of at least two.')
    result = {}
    for window in windows:
        correct = valid = evaluated = 0
        per_window = []
        for start in range(0, len(target) - window + 1):
            reference = target[start:start + window]
            outputs = prediction[start:start + window]
            stable_reference = (reference == reference[0]).all(0) & (reference != ignore).all(0)
            stable_prediction = (outputs == outputs[0]).all(0)
            count = int(stable_reference.sum())
            matches = int((stable_reference & stable_prediction).sum())
            valid += count
            correct += matches
            evaluated += 1
            per_window.append(dict(start=start, pixels=count, consistent=matches,
                                   score=matches / count if count else None))
        result[str(window)] = dict(
            windows=evaluated, eligible_pixels=valid, consistent_pixels=correct,
            score=correct / valid if valid else None,
            observations=per_window,
        )
    return result


def summarize_consistency(records):
    result = {}
    windows = sorted({window for row in records for window in row}, key=int)
    for window in windows:
        available = [row[window] for row in records if window in row]
        count = sum(row['eligible_pixels'] for row in available)
        correct = sum(row['consistent_pixels'] for row in available)
        scores = [row['score'] for row in available if row['score'] is not None]
        result[window] = dict(
            sequences=len(scores),
            windows=sum(row['windows'] for row in available),
            eligible_pixels=count,
            pixel_weighted_score=correct / count if count else None,
            sequence_mean_score=sum(scores) / len(scores) if scores else None,
        )
    return result
