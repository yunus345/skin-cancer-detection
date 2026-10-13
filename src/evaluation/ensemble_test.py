"""Kesinlesmis ensemble tarifini TEST setinde bir kez olcer.

Tarif (2026-10-13, tamamen validation setinde secildi, bkz. ensemble_val.py):
- Dort model: B0, ConvNeXt, V2-S, Swin
- Her model validation setinde bulunan T ile kalibre (T test setinde yeniden AYARLANMAZ)
- Esit agirlikli olasilik ortalamasi
- Iki calisma noktasi: normal karar (argmax) ve "tarama modu" (mel olasiligi >= 0.30 ise mel)

Bu betik ayar yapmaz; test sonucuna bakip tarifi degistirmek test setini validation setine cevirir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.ensemble_val import fit_temperature, load_split, softmax
from src.evaluation.evaluate import apply_threshold_override, bootstrap_ci, print_report, summarize

OPERATING_POINTS = {'normal': None, 'tarama_mel_esik_0.30': 0.30}
OUT_DIR = Path('baselines/ensemble_v1')


def main():
    val_logits, val_targets, labels = load_split('val')
    test_logits, test_targets, test_labels = load_split('test')
    assert labels == test_labels

    temperatures = {name: fit_temperature(l, val_targets) for name, l in val_logits.items()}
    print('Validation setinde bulunan T degerleri:', {k: round(v, 2) for k, v in temperatures.items()})
    avg = np.mean([softmax(test_logits[name] / temperatures[name]) for name in test_logits], axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mel_idx = labels.index('mel')
    for point, threshold in OPERATING_POINTS.items():
        predictions = apply_threshold_override(avg, mel_idx, threshold)
        results = summarize(test_targets, predictions, labels)
        results['recipe'] = {
            'models': list(test_logits), 'temperatures': temperatures,
            'weights': 'esit', 'mel_threshold': threshold,
        }
        results['bootstrap_ci'] = bootstrap_ci(test_targets, predictions, labels)
        print(f'\n===== Calisma noktasi: {point} =====')
        print_report(results)
        out_path = OUT_DIR / f'test_metrics_{point}.json'
        out_path.write_text(json.dumps(results, indent=2))
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
