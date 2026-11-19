"""Baseline CV'nin OOF logit'lerini Mac'te (fp32) yeniden hesaplar.

Neden: final v2'nin T ve melanom eşiği OOF'tan seçilecek, test ise Mac'te ölçülecek. Kaggle'da (T4, farklı
torch/OpenCV) hesaplanan OOF ile Mac çıktısı arasında 0.07-0.19 logit farkı var (argmax aynı). v1'deki "tek sayı
kaynağı: Mac, fp32" kuralı için seçim de ölçüm de aynı ortamdan gelmeli.

Her fold'un `last_model.pt`'si yalnızca kendi doğrulama fold'unu tahmin eder (o görüntüleri eğitimde görmedi).
Mac'i yormamak için modeller sırayla yüklenir, bellekte tek model durur.

    python -m src.evaluation.recompute_oof
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.training.train import build_cv_datasets, build_model


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-dir', default='models/cv5_baseline')
    parser.add_argument('--kaggle-oof-dir', default='baselines/cv5_baseline')
    parser.add_argument('--cv-folds', default='splits/cv5_v1.csv')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--out-dir', default='baselines/final_v2/oof_mac')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(5):
        start = time.time()
        weights = Path(args.models_dir) / f'fold{k}' / 'last_model.pt'
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        labels = list(ckpt['labels'])
        _, val = build_cv_datasets(args.manifest, args.cv_folds, k, ckpt['image_size'], ckpt['augmentation'])

        model = build_model(len(labels), ckpt['architecture'], pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()

        loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        logits, targets = [], []
        with torch.no_grad():
            for images, batch_labels in loader:
                logits.append(model(images.to(device)).float().cpu().numpy())  # AMP yok: fp32
                targets.extend(batch_labels.tolist())
        logits = np.concatenate(logits)
        image_ids = np.array([row['image_id'] for row in val.manifest])

        # Kaggle OOF ile aynı görüntüler, aynı sıra, aynı etiketler olmalı; yalnızca logit'ler ortamdan dolayı biraz farklı.
        kaggle = np.load(Path(args.kaggle_oof_dir) / f'fold{k}' / f'oof_fold{k}.npz', allow_pickle=True)
        assert (kaggle['image_ids'] == image_ids).all(), f'fold {k}: görüntü sırası Kaggle OOF ile farklı'
        assert (kaggle['targets'] == np.array(targets)).all(), f'fold {k}: etiketler Kaggle OOF ile farklı'
        assert list(kaggle['labels']) == labels

        np.savez(out_dir / f'oof_fold{k}.npz', logits=logits, targets=np.array(targets), image_ids=image_ids,
                 labels=np.array(labels), fold=k, architecture=ckpt['architecture'], image_size=ckpt['image_size'],
                 weights_md5=md5(weights))

        f1_mac = f1_score(targets, logits.argmax(1), average='macro')
        f1_kaggle = f1_score(targets, kaggle['logits'].argmax(1), average='macro')
        changed = int((logits.argmax(1) != kaggle['logits'].argmax(1)).sum())
        print(f'fold {k}: {len(image_ids)} görüntü, F1 Mac {f1_mac:.4f} / Kaggle {f1_kaggle:.4f}, '
              f'argmax farklı {changed}, max|logit farkı| {np.abs(logits - kaggle["logits"]).max():.3f}, '
              f'{time.time() - start:.0f} s', flush=True)

        del model
        if device.type == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
