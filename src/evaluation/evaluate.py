"""Evaluate a trained checkpoint on the held-out test split (use only once, after training is final)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.training.train import build_model, build_transforms, select_device


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = checkpoint['labels']
    model = build_model(num_classes=len(labels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, labels


def predict_probabilities(model, loader, device):
    """Returns (targets, probs); probs is an (N, num_classes) array of softmax probabilities."""
    model.eval()
    targets = []
    probs = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)
            probs.append(batch_probs.cpu().numpy())
            targets.extend(batch_labels.tolist())
    return np.array(targets), np.concatenate(probs, axis=0)


def apply_threshold_override(probs, class_idx, threshold):
    """Argmax predictions, but force `class_idx` whenever its probability clears `threshold`."""
    preds = probs.argmax(axis=1)
    if threshold is not None:
        preds = preds.copy()
        preds[probs[:, class_idx] >= threshold] = class_idx
    return preds


def summarize(targets, predictions, labels) -> dict:
    label_ids = list(range(len(labels)))
    overall = {
        'accuracy': accuracy_score(targets, predictions),
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall': recall_score(targets, predictions, average='macro', zero_division=0),
        'f1': f1_score(targets, predictions, average='macro', zero_division=0),
    }
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets, predictions, labels=label_ids, zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=label_ids)
    return {
        'overall': overall,
        'per_class': {
            label: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
            for label, p, r, f, s in zip(labels, per_class_precision, per_class_recall, per_class_f1, per_class_support)
        },
        'confusion_matrix': {'labels': labels, 'matrix': matrix.tolist()},
    }


def format_confusion_matrix(matrix, labels) -> str:
    header = '        ' + ' '.join(f'{l:>6s}' for l in labels)
    lines = [header]
    for true_label, row in zip(labels, matrix):
        row_str = ' '.join(f'{v:6d}' for v in row)
        lines.append(f'{true_label:>8s} {row_str}')
    return '\n'.join(lines)


def print_report(results: dict) -> None:
    overall = results['overall']
    print(
        f'\nTest accuracy: {overall["accuracy"]:.4f}  balanced_accuracy: {overall["balanced_accuracy"]:.4f}  '
        f'macro_f1: {overall["f1"]:.4f}\n'
    )
    print(f'{"class":>8s} {"precision":>10s} {"recall":>10s} {"f1":>10s} {"support":>8s}')
    for label, m in results['per_class'].items():
        print(f'{label:>8s} {m["precision"]:10.4f} {m["recall"]:10.4f} {m["f1"]:10.4f} {m["support"]:8d}')
    print('\nConfusion matrix (satır = gerçek etiket, sütun = model tahmini):')
    print(format_confusion_matrix(results['confusion_matrix']['matrix'], results['confusion_matrix']['labels']))


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on the held-out test split.')
    parser.add_argument('--manifest', default='data/processed/manifest.csv', help='Manifest file with image_path,label,split columns')
    parser.add_argument('--checkpoint', default='models/best_model.pt', help='Checkpoint to evaluate (best_model.pt = best val-F1 epoch)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--output', default='models/test_metrics.json')
    parser.add_argument(
        '--mel-threshold', type=float, default=None,
        help='Argmax yerine: "mel" olasılığı bu eşiği geçerse, başka sınıf daha olası olsa bile mel tahmin et',
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'Using device: {device}')

    model, labels = load_checkpoint(args.checkpoint, device)

    test_dataset = SkinCancerDataset(args.manifest, split='test', transform=build_transforms(args.image_size))
    if len(test_dataset) == 0:
        raise ValueError('Manifest must contain a non-empty test split. Run src.data.split_manifest first.')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    targets, probs = predict_probabilities(model, test_loader, device)

    if args.mel_threshold is not None and 'mel' not in labels:
        raise ValueError('--mel-threshold verildi ama checkpoint etiketleri arasında "mel" yok.')
    if args.mel_threshold is not None:
        predictions = apply_threshold_override(probs, labels.index('mel'), args.mel_threshold)
    else:
        predictions = probs.argmax(axis=1)

    results = summarize(targets, predictions, labels)
    if args.mel_threshold is not None:
        results['mel_threshold'] = args.mel_threshold
    print_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\nWrote test metrics to {output_path}')


if __name__ == '__main__':
    main()
