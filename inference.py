import argparse
from pathlib import Path

import numpy as np
import torch

from utils.data import VideoDataset
from utils.models import RoPA


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = RoPA(**checkpoint['config']['model']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
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
    parser.add_argument('--device', default='cpu')
    main(parser.parse_args())
