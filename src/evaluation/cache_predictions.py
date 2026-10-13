"""Her checkpoint'in validation ve test setindeki ham skorlarini (logit) bir kez hesaplayip kaydeder.

Ensemble denemeleri (kombinasyon, kalibrasyon, agirlik) modelleri tekrar calistirmadan bu
kayitli sayilar uzerinde yapilir. Olasilik degil logit sakliyoruz: olasiliga her zaman
softmax ile cevrilebilir, ama temperature scaling logit'leri T'ye bolmeyi gerektiriyor.

Cikti: checkpoint'in yanina `predictions.npz` (val_logits, val_targets, val_image_ids,
test_logits, test_targets, test_image_ids, labels, architecture, image_size).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.evaluation.evaluate import load_checkpoint
from src.training.train import build_transforms, select_device


def predict_logits(model, loader, device):
    model.eval()
    targets = []
    logits = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(device)).float().cpu().numpy())
            targets.extend(batch_labels.tolist())
    return np.array(targets), np.concatenate(logits, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Checkpoint\'lerin val/test logit\'lerini predictions.npz olarak kaydet.')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'Using device: {device}')

    for checkpoint_path in args.checkpoints:
        model, labels, image_size = load_checkpoint(checkpoint_path, device)
        architecture = torch.load(checkpoint_path, map_location='cpu', weights_only=False).get('architecture', 'efficientnet_b0')
        outputs = {'labels': np.array(labels), 'architecture': architecture, 'image_size': image_size}
        for split in ('val', 'test'):
            dataset = SkinCancerDataset(args.manifest, split=split, transform=build_transforms(image_size))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            targets, logits = predict_logits(model, loader, device)
            outputs[f'{split}_logits'] = logits
            outputs[f'{split}_targets'] = targets
            # Modeller arasinda goruntu sirasinin ayni oldugunu sonradan dogrulayabilmek icin.
            outputs[f'{split}_image_ids'] = np.array([row['image_id'] for row in dataset.manifest])
            accuracy = (logits.argmax(axis=1) == targets).mean()
            print(f'{checkpoint_path} [{split}] {len(targets)} goruntu, accuracy={accuracy:.4f}')

        out_path = Path(checkpoint_path).with_name('predictions.npz')
        np.savez_compressed(out_path, **outputs)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
