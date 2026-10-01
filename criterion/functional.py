import torch
import torch.nn.functional as F


def cross_gram(predicted, target):
    return F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).transpose(-1, -2)


def paga_loss(student_prediction, student_target, teacher_prediction, teacher_target, indices=None):
    """Eq. (5): sum across offsets, average across clips and sampled patch pairs.

    Inputs B,offset,N,D. All four tensors use the same patch subset. Teacher
    structure is detached; target-frame student features retain gradients.
    """
    features = [student_prediction, student_target, teacher_prediction, teacher_target]
    if any(t.shape != features[0].shape for t in features) or features[0].ndim != 4:
        raise ValueError('PAGA inputs must share B,offset,N,D shape.')
    if indices is not None:
        features = [t.index_select(-2, indices) for t in features]
    student = cross_gram(*features[:2])
    with torch.no_grad():
        teacher = cross_gram(*features[2:])
    return (student - teacher).square().mean((-1, -2)).sum(-1).mean()


def consistency_loss(predictor, z, delta1, delta2, target_range=64.0, identity_weight=1.0):
    if delta1 < 0 or delta2 < 0 or delta1 + delta2 > target_range:
        raise ValueError('Composition offsets must be nonnegative and sum to at most Tband.')
    direct = predictor(z, delta1 + delta2)
    composed = predictor(predictor(z, delta1), delta2)
    # Eq. (6) is squared L2 in feature dimension, followed by an expectation.
    composition = (direct - composed).square().sum(-1).mean()
    identity = (predictor(z, 0) - z).square().sum(-1).mean()
    return composition + identity_weight * identity


def gram_weight(step, total_steps, weight=1.0, warmup_steps=5000):
    start = int(0.9 * total_steps)
    if step < start:
        return 0.0
    return weight * min(1.0, (step - start + 1) / max(1, warmup_steps))
