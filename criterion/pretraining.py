import torch
import torch.nn.functional as F

from .functional import consistency_loss, gram_weight, paga_loss


class RoPAObjective(torch.nn.Module):
    """Offset prediction with an explicit anchor axis and one selected temporal offset."""
    def __init__(self, lambda_gram=1.0, lambda_rope=0.1, gram_warmup=5000,
                 sample_patches=64, target_range=64.0, prediction_offset=1):
        super().__init__()
        self.lambda_gram = lambda_gram
        self.lambda_rope = lambda_rope
        self.gram_warmup = gram_warmup
        self.sample_patches = sample_patches
        self.target_range = target_range
        self.prediction_offset = int(prediction_offset)

    def forward(self, model, anchor, video, spacing, step, total_steps):
        features = model(video, spacing)
        delta = self.prediction_offset
        if features.shape[1] <= delta:
            raise ValueError("The clip must contain more tubelets than prediction_offset.")
        predicted = model.predictor(features[:, :-delta], delta)
        with torch.no_grad():
            target = anchor(video, spacing)
            target_prediction = anchor.predictor(target[:, :-delta], delta)
        prediction = F.mse_loss(F.normalize(predicted, dim=-1), F.normalize(target[:, delta:], dim=-1))
        selected = torch.randperm(features.shape[-2], device=video.device)[:self.sample_patches]
        # The offset axis is length one. Source times belong to the anchor batch.
        gram_inputs = [value.flatten(0, 1).unsqueeze(1)
                       for value in (predicted, features[:, delta:], target_prediction, target[:, delta:])]
        gram = paga_loss(*gram_inputs, indices=selected)
        consistency = consistency_loss(model.predictor, features[:, 0], 1, 1, self.target_range, normalize=True)
        weight = gram_weight(step, total_steps, self.lambda_gram, self.gram_warmup)
        total = prediction + weight * gram + self.lambda_rope * consistency
        return {'loss': total, 'prediction': prediction, 'paga': gram, 'rcl': consistency,
                'gram_weight': weight}
