"""Ensemble kombinasyonlarini VALIDATION setinde karsilastirir (test setine bakmadan).

cache_predictions.py'nin kaydettigi logit'leri okur, her kombinasyonda modellerin softmax
olasiliklarini esit agirlikla ortalar ve metrikleri hesaplar. Model secimi burada yapilir;
test seti sadece tarif kesinlestikten sonra bir kez kullanilir.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

MODELS = {
    'B0': 'baselines/baseline_v1/predictions.npz',
    'ConvNeXt': 'baselines/convnext_tiny_v1/predictions.npz',
    'V2-S': 'baselines/efficientnet_v2_s_v1/predictions.npz',
    'Swin': 'baselines/swin_v2_t_v1/predictions.npz',
}


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, targets, temperature=1.0):
    """Ortalama negatif log-olabilirlik: emin olup yanilan tahminleri en agir cezalandiran loss."""
    probs = softmax(logits / temperature)
    return -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-12))


def fit_temperature(logits, targets):
    # Tek bir sayiyi ariyoruz; aralik taramasi hem yeterince hizli hem de ne yaptigimizi acikca gosteriyor.
    # T > 1 olasiliklari yumusatir (asiri ozguven), T < 1 keskinlestirir. argmax'i (karari) degistirmez.
    candidates = np.arange(0.5, 5.0001, 0.01)
    losses = [nll(logits, targets, t) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def load_split(split: str):
    data = {name: np.load(path) for name, path in MODELS.items()}
    reference = next(iter(data.values()))
    for name, d in data.items():
        if not (d[f'{split}_image_ids'] == reference[f'{split}_image_ids']).all():
            raise ValueError(f'{name}: goruntu sirasi diger modellerden farkli')
    labels = list(reference['labels'])
    logits = {name: d[f'{split}_logits'] for name, d in data.items()}
    return logits, reference[f'{split}_targets'], labels


def score(targets, predictions, mel_idx):
    mel_true = targets == mel_idx
    mel_pred = predictions == mel_idx
    return {
        'macro_f1': f1_score(targets, predictions, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),
        'macro_precision': precision_score(targets, predictions, average='macro', zero_division=0),
        'mel_recall': recall_score(mel_true, mel_pred),
        'mel_precision': precision_score(mel_true, mel_pred, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description='Ensemble kombinasyonlarini validation setinde karsilastir.')
    parser.add_argument('--calibrate', action='store_true',
                        help='Ortalamadan once her modeli validation setinde temperature scaling ile kalibre et')
    args = parser.parse_args()

    logits, targets, labels = load_split('val')
    mel_idx = labels.index('mel')
    temperatures = {name: 1.0 for name in logits}
    if args.calibrate:
        print(f'{"Model":10} {"T":>5} {"NLL once":>9} {"NLL sonra":>10}')
        for name, l in logits.items():
            temperatures[name] = fit_temperature(l, targets)
            print(f'{name:10} {temperatures[name]:5.2f} {nll(l, targets):9.3f} {nll(l, targets, temperatures[name]):10.3f}')
        print()
    probs = {name: softmax(l / temperatures[name]) for name, l in logits.items()}

    rows = []
    names = list(MODELS)
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            # Duz ortalama: her model esit soz hakkina sahip.
            avg = np.mean([probs[name] for name in combo], axis=0)
            rows.append((' + '.join(combo), score(targets, avg.argmax(axis=1), mel_idx)))

    rows.sort(key=lambda row: row[1]['macro_f1'], reverse=True)
    title = 'Kombinasyon (validation, ' + ('kalibre' if args.calibrate else 'duz ortalama') + ')'
    print(f'{title:36} {"F1":>6} {"BAcc":>6} {"Prec":>6} {"melR":>6} {"melP":>6}')
    for name, s in rows:
        print(f'{name:36} {s["macro_f1"]:6.3f} {s["balanced_accuracy"]:6.3f} {s["macro_precision"]:6.3f} '
              f'{s["mel_recall"]:6.3f} {s["mel_precision"]:6.3f}')


if __name__ == '__main__':
    main()
