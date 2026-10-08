"""Ensemble birden fazla checkpoint'i birlestirir (direct average - en basit ve en etkili yontem).

Her model, test setindeki her goruntu icin bir softmax olasilik dagilimi uretir;
bu dagilimlar modeller arasinda ortalanir, tahmin bu ortalama uzerinden (argmax) yapilir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.evaluation.evaluate import (
    apply_threshold_override,
    bootstrap_ci,
    load_checkpoint,
    predict_probabilities,
    print_report,
    summarize,
)
from src.training.train import build_transforms, select_device


def main():
    parser = argparse.ArgumentParser(description='Birden fazla checkpoint\'i ortalayarak (ensemble) test setinde degerlendir.')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Ortalanacak checkpoint dosyalari (en az 2)')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--output', default='models/ensemble_test_metrics.json')
    parser.add_argument(
        '--mel-threshold', type=float, default=None,
        help='Argmax yerine: "mel" olasılığı (ortalanmış) bu eşiği geçerse mel tahmin et',
    )
    parser.add_argument('--bootstrap-iterations', type=int, default=1000)
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise ValueError('En az 2 checkpoint gerekli, yoksa ensemble yapmanın anlamı yok.')

    device = select_device(args.device)
    print(f'Using device: {device}')

    test_dataset = SkinCancerDataset(args.manifest, split='test', transform=build_transforms(args.image_size))
    if len(test_dataset) == 0:
        raise ValueError('Manifest must contain a non-empty test split. Run src.data.split_manifest first.')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_probs = []
    targets = None
    labels = None
    for checkpoint_path in args.checkpoints:
        print(f'Checkpoint yükleniyor: {checkpoint_path}')
        model, ckpt_labels = load_checkpoint(checkpoint_path, device)
        if labels is None:
            labels = ckpt_labels
        elif labels != ckpt_labels:
            raise ValueError(
                f'{checkpoint_path} etiket sırası diğerlerinden farklı ({ckpt_labels} != {labels}) '
                '- aynı pipeline/manifest ile eğitilmiş checkpoint\'ler kullanın.'
            )
        ckpt_targets, probs = predict_probabilities(model, test_loader, device)
        if targets is None:
            targets = ckpt_targets
        all_probs.append(probs)

    # Direct average: butun modellerin softmax olasiliklarinin basit ortalamasi.
    avg_probs = np.mean(all_probs, axis=0)

    if args.mel_threshold is not None:
        if 'mel' not in labels:
            raise ValueError('--mel-threshold verildi ama checkpoint etiketleri arasında "mel" yok.')
        predictions = apply_threshold_override(avg_probs, labels.index('mel'), args.mel_threshold)
    else:
        predictions = avg_probs.argmax(axis=1)

    results = summarize(targets, predictions, labels)
    results['ensemble_checkpoints'] = args.checkpoints
    if args.mel_threshold is not None:
        results['mel_threshold'] = args.mel_threshold
    if args.bootstrap_iterations > 0:
        results['bootstrap_ci'] = bootstrap_ci(targets, predictions, labels, n_iterations=args.bootstrap_iterations)
    print_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\nWrote ensemble test metrics to {output_path}')


if __name__ == '__main__':
    main()
