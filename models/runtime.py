import torch


def cuda_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('RoPA model execution requires a CUDA device (cuda or cuda:N).')
    return device
