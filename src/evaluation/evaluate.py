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
)
from torch.utils.data import DataLoader

from src.data.loader import SkinCancerDataset
from src.training.train import build_model, build_transforms, select_device


def load_checkpoint(checkpoint_path: str, device: torch.device):
    # weights_only=False: kendi urettigimiz, guvendigimiz bir checkpoint - PyTorch 2.6+'nin
    # varsayilan guvenlik kisitlamasi (numpy skalerleri reddetmesi) burada gereksiz.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = checkpoint['labels']
    # Eski checkpoint'lerde (baseline-v1 gibi) 'architecture' alani yok - o zaman tek
    # mimarimiz EfficientNet-B0'di, varsayilan olarak ona dusuyoruz (geriye uyumluluk).
    architecture = checkpoint.get('architecture', 'efficientnet_b0')
    model = build_model(num_classes=len(labels), architecture=architecture)
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
    """Metrik hiyerarsisi (sunumla kararlastirdigimiz standart):
    1. Headline: macro_f1 - model secimi / kendi aramizdaki karsilastirma icin.
    2. Dis karsilastirma: balanced_accuracy (=macro recall, ISIC leaderboard ile kiyas icin)
       ve macro_precision (sadece ilgili makaleyle kiyas icin, "MCA" DEMIYORUZ - yaniltici).
    3. Klinik: mel_recall + mel_precision - kanseri kacirmama onceligi.
    """
    label_ids = list(range(len(labels)))
    overall = {
        'accuracy': accuracy_score(targets, predictions),
        'macro_f1': f1_score(targets, predictions, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),  # = macro recall
        'macro_precision': precision_score(targets, predictions, average='macro', zero_division=0),
    }
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets, predictions, labels=label_ids, zero_division=0,
    )
    per_class = {
        label: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
        for label, p, r, f, s in zip(labels, per_class_precision, per_class_recall, per_class_f1, per_class_support)
    }
    clinical = {}
    if 'mel' in per_class:
        clinical = {'mel_recall': per_class['mel']['recall'], 'mel_precision': per_class['mel']['precision']}

    matrix = confusion_matrix(targets, predictions, labels=label_ids)
    return {
        'overall': overall,
        'clinical': clinical,
        'per_class': per_class,
        'confusion_matrix': {'labels': labels, 'matrix': matrix.tolist()},
    }


def bootstrap_ci(targets, predictions, labels, n_iterations: int = 1000, seed: int = 42) -> dict:
    """Test setini yerine-koyarak (with replacement) n_iterations kez yeniden orneklep her seferinde
    headline metrikleri hesaplar; %95 guven araligini (2.5-97.5 persentil) dondurur. Model tekrar
    calistirilmiyor - sadece elimizdeki (targets, predictions) ciftleri uzerinde, hizli bir islem."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    mel_idx = labels.index('mel') if 'mel' in labels else None

    samples = {'macro_f1': [], 'balanced_accuracy': [], 'macro_precision': [], 'mel_recall': [], 'mel_precision': []}
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        t, p = targets[idx], predictions[idx]
        samples['macro_f1'].append(f1_score(t, p, average='macro', zero_division=0))
        samples['balanced_accuracy'].append(balanced_accuracy_score(t, p))
        samples['macro_precision'].append(precision_score(t, p, average='macro', zero_division=0))
        if mel_idx is not None:
            pc_precision, pc_recall, _, _ = precision_recall_fscore_support(
                t, p, labels=[mel_idx], zero_division=0,
            )
            samples['mel_precision'].append(float(pc_precision[0]))
            samples['mel_recall'].append(float(pc_recall[0]))

    ci = {}
    for key, values in samples.items():
        if not values:
            continue
        arr = np.array(values)
        ci[key] = {'ci_95_low': float(np.percentile(arr, 2.5)), 'ci_95_high': float(np.percentile(arr, 97.5))}
    return ci


def format_confusion_matrix(matrix, labels) -> str:
    header = '        ' + ' '.join(f'{l:>6s}' for l in labels)
    lines = [header]
    for true_label, row in zip(labels, matrix):
        row_str = ' '.join(f'{v:6d}' for v in row)
        lines.append(f'{true_label:>8s} {row_str}')
    return '\n'.join(lines)


def print_report(results: dict) -> None:
    overall = results['overall']
    ci = results.get('bootstrap_ci', {})

    def fmt(key: str, value: float) -> str:
        if key in ci:
            return f'{value:.4f} (%95 GA: {ci[key]["ci_95_low"]:.4f}-{ci[key]["ci_95_high"]:.4f})'
        return f'{value:.4f}'

    print(f'\n[1. Headline]      macro_f1: {fmt("macro_f1", overall["macro_f1"])}')
    print(
        f'[2. Dış karşılaştırma] balanced_accuracy (=macro recall, ISIC): {fmt("balanced_accuracy", overall["balanced_accuracy"])}\n'
        f'                       macro_precision (sadece ilgili makaleyle kıyas): {fmt("macro_precision", overall["macro_precision"])}'
    )
    if results.get('clinical'):
        clinical = results['clinical']
        print(
            f'[3. Klinik]         mel_recall: {fmt("mel_recall", clinical["mel_recall"])}\n'
            f'                       mel_precision: {fmt("mel_precision", clinical["mel_precision"])}'
        )
    print(f'\naccuracy (bilgi amaçlı, dengesiz veride yanıltıcı olabilir): {overall["accuracy"]:.4f}\n')
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
    parser.add_argument('--bootstrap-iterations', type=int, default=1000, help='Bootstrap %95 güven aralığı için yeniden örnekleme sayısı (0 = kapalı)')
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
    if args.bootstrap_iterations > 0:
        results['bootstrap_ci'] = bootstrap_ci(targets, predictions, labels, n_iterations=args.bootstrap_iterations)
    print_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\nWrote test metrics to {output_path}')


if __name__ == '__main__':
    main()
