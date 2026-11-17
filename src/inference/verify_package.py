"""Paketlenmis model, dondurulmus tarifin sayilarini birebir veriyor mu?

Predictor ile validation goruntulerini bastan tahmin eder ve sonucu recipe.json'daki modellerin kayitli
logit'lerinden (predictions.npz, ayni T'ler) hesaplanan olasiliklarla karsilastirir. Paketleme, on isleme
ya da sinif sirasinda bir hata varsa burada gorunur. Yalnizca val kullanilir; test setine gerek yok.

Calistirma: python -m src.inference.verify_package
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.inference.predictor import Predictor


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description='Model paketini kayitli val logit\'leriyle karsilastir.')
    parser.add_argument('--model-dir', default='models/final_v1')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    predictor = Predictor(args.model_dir, device=args.device)
    recipe = json.loads(Path(predictor.config['recipe']).read_text())

    # Beklenen: recipe'teki her modelin kayitli val logit'leri, ayni T ile, esit ortalama.
    expected, ids = [], None
    for name, path in recipe['models'].items():
        saved = np.load(path, allow_pickle=True)
        assert list(saved['labels']) == predictor.labels, 'sinif sirasi paketle ayni degil'
        ids = saved['val_image_ids'] if ids is None else ids
        expected.append(softmax(saved['val_logits'] / recipe['temperatures'][name]))
    expected = np.mean(expected, axis=0)

    dataset = SkinCancerDataset(args.manifest, split='val', transform=predictor.transform)
    assert [row['image_id'] for row in dataset.manifest] == list(ids), 'val goruntu sirasi farkli'
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    got = np.concatenate([predictor.predict_tensor(images)[0] for images, _ in loader])

    diff = np.abs(got - expected)
    same_normal = (got.argmax(1) == expected.argmax(1)).mean()
    mel = predictor.mel_idx
    tarama = lambda p: np.where(p[:, mel] >= predictor.mel_threshold, mel, p.argmax(1))
    same_tarama = (tarama(got) == tarama(expected)).mean()
    print(f'{len(got)} val goruntusu')
    print(f'olasilik farki: en buyuk {diff.max():.2e}, ortalama {diff.mean():.2e}')
    print(f'ayni karar: normal %{100 * same_normal:.2f}, tarama %{100 * same_tarama:.2f}')
    if same_normal < 1 or same_tarama < 1:
        flips = np.where(got.argmax(1) != expected.argmax(1))[0]
        for i in flips[:10]:
            print(f'  {ids[i]}: paket {predictor.labels[got[i].argmax()]} ({got[i].max():.3f}), '
                  f'kayitli {predictor.labels[expected[i].argmax()]} ({expected[i].max():.3f})')


if __name__ == '__main__':
    main()
