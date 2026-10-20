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


# TTA bakislari: egitimde (v1 tarifi) gorulen cevirmeler. Iki eksen birden cevirmek = 180 derece
# dondurme. Modelin hic gormedigi donusumler (or. 90 derece) yardim yerine kafa karistirabilir.
TTA_FLIP_DIMS = [[], [3], [2], [2, 3]]  # orijinal, yatay, dikey, ikisi birden (tensor: N, C, H, W)


def predict_logits(model, loader, device, tta: bool = False):
    model.eval()
    targets = []
    logits = []
    views = TTA_FLIP_DIMS if tta else [[]]
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            # Bakislarin logit ortalamasi: kalibrasyon (logit / T) ayni sekilde calismaya devam eder.
            view_logits = [model(torch.flip(images, dims) if dims else images).float() for dims in views]
            logits.append(torch.stack(view_logits).mean(dim=0).cpu().numpy())
            targets.extend(batch_labels.tolist())
    return np.array(targets), np.concatenate(logits, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Checkpoint\'lerin val/test logit\'lerini predictions.npz olarak kaydet.')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: 4 cevirme bakisinin logit ortalamasi (predictions_tta.npz)')
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'Using device: {device}')

    for checkpoint_path in args.checkpoints:
        model, labels, image_size = load_checkpoint(checkpoint_path, device)
        architecture = torch.load(checkpoint_path, map_location='cpu', weights_only=False).get('architecture', 'efficientnet_b0')
        outputs = {'labels': np.array(labels), 'architecture': architecture, 'image_size': image_size, 'tta': args.tta}
        for split in ('val', 'test'):
            dataset = SkinCancerDataset(args.manifest, split=split, transform=build_transforms(image_size))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            targets, logits = predict_logits(model, loader, device, tta=args.tta)
            outputs[f'{split}_logits'] = logits
            outputs[f'{split}_targets'] = targets
            # Modeller arasinda goruntu sirasinin ayni oldugunu sonradan dogrulayabilmek icin.
            outputs[f'{split}_image_ids'] = np.array([row['image_id'] for row in dataset.manifest])
            accuracy = (logits.argmax(axis=1) == targets).mean()
            print(f'{checkpoint_path} [{split}] {len(targets)} goruntu, accuracy={accuracy:.4f}')

        # best_model_ema.pt gibi ayni klasordeki ikinci checkpoint'ler kendi dosyasina yazilir (predictions_ema.npz).
        suffix = Path(checkpoint_path).stem.removeprefix('best_model').lstrip('_')
        name = '_'.join(['predictions'] + [s for s in (suffix, 'tta' if args.tta else '') if s])
        out_path = Path(checkpoint_path).with_name(f'{name}.npz')
        np.savez_compressed(out_path, **outputs)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
