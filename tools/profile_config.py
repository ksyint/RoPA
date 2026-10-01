"""Schema checks shared by the temporal-grid builder and native trainer."""
import math


def validate_config(config):
    required = {'model', 'runtime', 'optim', 'objective', 'synthetic'}
    if not required <= config.keys():
        raise ValueError(f'Missing config sections: {sorted(required - config.keys())}')
    model, runtime = config['model'], config['runtime']
    if not 0 < model['local_scale'] <= model['target_range']:
        raise ValueError('Temporal scales must satisfy 0 < T0 <= T*.')
    if model.get('predictor_ratio', 2) <= 0:
        raise ValueError('Predictor expansion must be positive.')
    if runtime['steps'] < 1 or runtime['batch_size'] < 1:
        raise ValueError('Training requires positive steps and batch size.')
    if config['optim']['lr'] <= 0 or config['optim']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings.')
    if any(config['objective'][key] < 0 for key in ('lambda_gram', 'lambda_rope')):
        raise ValueError('Regularizer weights must be nonnegative.')
    spacing = config.get('spacing', {})
    low, high = spacing.get('low', 0.5), spacing.get('high', 2.0)
    if not all(math.isfinite(value) for value in (low, high)) or not 0 < low <= high:
        raise ValueError('Spacing jitter needs finite bounds 0 < low <= high.')
    return config
