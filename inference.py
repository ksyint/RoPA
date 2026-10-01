import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from models.runtime import cuda_device
from datasets import VideoDataset
from models import create_model, load_checkpoint_model


def main(args):
    device = cuda_device(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        config = checkpoint['config']
    else:
        config = yaml.safe_load(Path(args.config).read_text())
    for key in ('pretrained', 'cache_dir'):
        if getattr(args, key) is not None:
            config['model'][key] = getattr(args, key)
    if args.offline:
        config['model']['local_files_only'] = True
    model = (load_checkpoint_model(checkpoint) if args.checkpoint else create_model(config)).to(device).eval()
    if Path(args.data).is_file():
        from video_data import VideoManifest
        dataset = VideoManifest(args.data, config['model'], config.get('data', {}).get('frames', 16))
    else:
        dataset = VideoDataset(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for index, (path, video) in enumerate(zip(dataset.paths, dataset)):
            features = model(video[None].to(device))[0].float().cpu().numpy()
            destination = output / f'{index:06d}_{path.stem}.npy'
            np.save(destination, features)
            print(f'{destination.name}: {features.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Trained RoPA checkpoint; otherwise use the released foundation initialization.')
    parser.add_argument('--config', default='configs/vjepa2.yaml')
    parser.add_argument('--pretrained')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
