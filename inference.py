import argparse
from pathlib import Path

import numpy as np
import torch
from models.runtime import cuda_device

from datasets import VideoDataset
from models import load_checkpoint_model


def main(args):
    args.device = str(cuda_device(args.device))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = load_checkpoint_model(checkpoint).to(args.device).eval()
    dataset = VideoDataset(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path, video in zip(dataset.paths, dataset):
            features = model(video[None].to(args.device))[0].cpu().numpy()
            np.save(output / path.name, features)
            print(f'{path.name}: {features.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
