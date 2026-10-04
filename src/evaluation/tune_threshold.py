"""Explore the precision/recall trade-off for the 'mel' class on the validation split and suggest a threshold.

Only the validation split is used here, never the test split — picking a threshold by looking at test
performance would quietly turn the test set into another validation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # ekran olmadan (headless) PNG kaydetmek icin
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.evaluation.evaluate import load_checkpoint, predict_probabilities
from src.training.train import build_transforms, select_device

COLOR_PRECISION = '#2a78d6'
COLOR_RECALL = '#eb6834'
COLOR_GRID = '#e1e0d9'
COLOR_AXIS = '#c3c2b7'
COLOR_MUTED = '#898781'
COLOR_TEXT = '#0b0b0b'


def choose_threshold(precision, recall, thresholds, target_recall: float):
    eligible = recall >= target_recall
    if eligible.any():
        best_idx = int(np.argmax(np.where(eligible, precision, -1)))
    else:
        print(f'Uyarı: hedef recall ({target_recall}) hiçbir eşikte sağlanamadı; en yüksek recall seçildi.')
        best_idx = int(np.argmax(recall))
    return best_idx


def plot_curve(thresholds, precision, recall, chosen_threshold, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#fcfcfb')
    ax.plot(thresholds, precision, color=COLOR_PRECISION, linewidth=2, label='precision')
    ax.plot(thresholds, recall, color=COLOR_RECALL, linewidth=2, label='recall')
    ax.axvline(chosen_threshold, color=COLOR_MUTED, linewidth=1, linestyle='--')
    ax.set_xlabel('mel eşiği (threshold)', color=COLOR_TEXT)
    ax.set_ylabel('değer', color=COLOR_TEXT)
    ax.set_title("'mel' için eşiğe göre precision/recall (val seti)", color=COLOR_TEXT)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())


def main():
    parser = argparse.ArgumentParser(description="'mel' karar eşiğini doğrulama (val) setinde ayarla.")
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--checkpoint', default='models/best_model.pt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument(
        '--target-recall', type=float, default=0.85,
        help='mel için hedeflenen minimum recall; bunu sağlayan en yüksek precision\'lı eşik seçilir',
    )
    parser.add_argument('--output', default='models/mel_threshold.json')
    parser.add_argument('--plot', default='models/mel_threshold_curve.png')
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'Using device: {device}')

    model, labels = load_checkpoint(args.checkpoint, device)
    if 'mel' not in labels:
        raise ValueError('Checkpoint etiketleri arasında "mel" yok.')
    mel_idx = labels.index('mel')

    val_dataset = SkinCancerDataset(args.manifest, split='val', transform=build_transforms(args.image_size))
    if len(val_dataset) == 0:
        raise ValueError('Manifest must contain a non-empty val split.')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    targets, probs = predict_probabilities(model, val_loader, device)
    mel_scores = probs[:, mel_idx]
    is_mel = (targets == mel_idx).astype(int)

    precision, recall, thresholds = precision_recall_curve(is_mel, mel_scores)
    precision, recall = precision[:-1], recall[:-1]  # sklearn son noktayı eşiksiz (recall=0 sınırı) ekliyor

    best_idx = choose_threshold(precision, recall, thresholds, args.target_recall)
    chosen_threshold = float(thresholds[best_idx])
    chosen_precision = float(precision[best_idx])
    chosen_recall = float(recall[best_idx])

    print(f'Seçilen eşik: {chosen_threshold:.4f}  (val precision={chosen_precision:.4f}, val recall={chosen_recall:.4f})')
    print(f'Test setinde karşılaştırmak için: python -m src.evaluation.evaluate --mel-threshold {chosen_threshold:.4f}')

    plot_path = Path(args.plot)
    plot_curve(thresholds, precision, recall, chosen_threshold, plot_path)
    print(f'Eşik eğrisi grafiği: {plot_path}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'threshold': chosen_threshold,
        'val_precision': chosen_precision,
        'val_recall': chosen_recall,
        'target_recall': args.target_recall,
    }, indent=2))
    print(f'Eşik bilgisi: {output_path}')


if __name__ == '__main__':
    main()
