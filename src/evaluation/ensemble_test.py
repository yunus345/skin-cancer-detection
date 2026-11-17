"""Kesinlesmis ensemble tarifini TEST setinde bir kez olcer.

Varsayilan tarif ensemble_v1 (2026-10-13, tamamen validation setinde secildi, bkz. ensemble_val.py):
- Dort model: B0, ConvNeXt, V2-S, Swin
- Her model validation setinde bulunan T ile kalibre (T test setinde yeniden AYARLANMAZ)
- Esit agirlikli olasilik ortalamasi
- Iki calisma noktasi: normal karar (argmax) ve "tarama modu" (mel olasiligi >= 0.30 ise mel)

--recipe ile baska bir dondurulmus tarif olculur (or. baselines/final_v1/recipe.json, bkz.
freeze_recipe.py): modeller, T degerleri ve mel esigi dosyadan okunur, burada hesaplanmaz.

Bu betik ayar yapmaz; test sonucuna bakip tarifi degistirmek test setini validation setine cevirir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.ensemble_val import MODELS, fit_temperature, load_split, softmax
from src.evaluation.evaluate import apply_threshold_override, bootstrap_ci, print_report, summarize

OPERATING_POINTS = {'normal': None, 'tarama_mel_esik_0.30': 0.30}
OUT_DIR = Path('baselines/_archive/ensemble_v1')


def resolve_recipe(recipe_path: str | None):
    """(models, temperatures, operating_points, out_dir) dondurur.

    recipe_path yoksa ensemble_v1: T'ler validation setinde bulunur. Varsa her sey dosyadan gelir;
    dondurulmus tarifte T'yi yeniden hesaplamiyoruz ki olculen sey commit edilen tarifin aynisi olsun.
    """
    if recipe_path is None:
        val_logits, val_targets, _ = load_split('val')
        temperatures = {name: fit_temperature(l, val_targets) for name, l in val_logits.items()}
        return MODELS, temperatures, OPERATING_POINTS, OUT_DIR
    recipe = json.loads(Path(recipe_path).read_text())
    threshold = recipe['mel_threshold']
    points = {'normal': None, f'tarama_mel_esik_{threshold:.2f}': threshold}
    return recipe['models'], recipe['temperatures'], points, Path(recipe_path).parent


def main():
    parser = argparse.ArgumentParser(description='Dondurulmus ensemble tarifini test setinde bir kez olc.')
    parser.add_argument('--recipe', default=None, help='recipe.json (varsayilan: ensemble_v1)')
    parser.add_argument('--out-dir', default=None, help='Varsayilan: tarifin klasoru')
    args = parser.parse_args()

    models, temperatures, operating_points, out_dir = resolve_recipe(args.recipe)
    out_dir = Path(args.out_dir) if args.out_dir else out_dir
    print('Validation setinde bulunan T degerleri:', {k: round(v, 2) for k, v in temperatures.items()})
    test_logits, test_targets, labels = load_split('test', models=models)
    avg = np.mean([softmax(test_logits[name] / temperatures[name]) for name in test_logits], axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    mel_idx = labels.index('mel')
    for point, threshold in operating_points.items():
        predictions = apply_threshold_override(avg, mel_idx, threshold)
        results = summarize(test_targets, predictions, labels, probs=avg)
        results['recipe'] = {
            'models': list(test_logits), 'temperatures': temperatures,
            'weights': 'esit', 'mel_threshold': threshold,
        }
        results['bootstrap_ci'] = bootstrap_ci(test_targets, predictions, labels, probs=avg)
        print(f'\n===== Calisma noktasi: {point} =====')
        print_report(results)
        out_path = out_dir / f'test_metrics_{point}.json'
        out_path.write_text(json.dumps(results, indent=2))
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
