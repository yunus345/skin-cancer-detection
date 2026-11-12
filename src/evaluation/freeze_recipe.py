"""Final tarifini YALNIZCA validation setinde secer ve dondurur (baselines/final_v1/recipe.json).

Secim kurali bu dosya calistirilmadan once yazildi (PROJECT_STATUS.md, denetim raporu Adim 2):
- Adaylar: A = seed 42 tek model, B = seed 42 + seed 2 (her biri val'de T ile kalibre, esit agirlik).
- Val macro F1 farki gurultu cetvelinden (0.013) kucukse: val'de mel recall >= 0.80 saglayan noktada
  daha az yanlis melanom alarmi veren aday secilir. Fark buyukse F1'i yuksek olan secilir.
- Tarama esigi: val'de mel recall >= 0.80 saglayan en yuksek esik (0.01 adimli).

Test logit'leri predictions.npz icinde duruyor ama bu betik onlari OKUMAZ (load_split('val')).
recipe.json commit edildikten sonra tarif degismez; test olcumu ayri betikte, bir kez.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.ensemble_val import fit_temperature, load_split, score, softmax
from src.evaluation.evaluate import apply_threshold_override

SEED42 = 'baselines/convnext_tiny_in22k_384_v1/predictions.npz'
SEED2 = 'baselines/convnext_tiny_in22k_384_seed2_ema/predictions.npz'
CANDIDATES = {
    'A_tek_model': {'seed42': SEED42},
    'B_iki_seed': {'seed42': SEED42, 'seed2': SEED2},
}
NOISE_F1 = 0.013  # seed cetveli: sadece seed degisince val F1 bu kadar oynuyor
TARGET_MEL_RECALL = 0.80
OUT_PATH = Path('baselines/final_v1/recipe.json')


def evaluate_candidate(models: dict) -> dict:
    logits, targets, labels = load_split('val', models=models)
    mel_idx = labels.index('mel')
    temperatures = {name: fit_temperature(l, targets) for name, l in logits.items()}
    avg = np.mean([softmax(logits[name] / temperatures[name]) for name in logits], axis=0)
    normal = score(targets, avg.argmax(axis=1), mel_idx)

    # Esigi yuksekten asagi tariyoruz: hedef recall'i saglayan ILK (en yuksek) esik en az alarm vereni.
    # None = argmax; hedef zaten saglaniyorsa esik gerekmez.
    for threshold in [None] + [round(t, 2) for t in np.arange(0.50, 0.0, -0.01)]:
        pred = apply_threshold_override(avg, mel_idx, threshold)
        screening = score(targets, pred, mel_idx)
        if screening['mel_recall'] >= TARGET_MEL_RECALL:
            break
    false_alarms = int(((pred == mel_idx) & (targets != mel_idx)).sum())
    return {
        'models': models, 'temperatures': temperatures, 'val_normal': normal,
        'mel_threshold': threshold, 'val_screening': screening, 'val_false_mel_alarms': false_alarms,
    }


def main():
    results = {name: evaluate_candidate(models) for name, models in CANDIDATES.items()}
    print(f'{"(validation)":14} {"F1":>6} {"BAcc":>6} {"melR":>6} | {"esik":>5} {"F1":>6} {"melR":>6} {"alarm":>6}')
    for name, r in results.items():
        n, s = r['val_normal'], r['val_screening']
        print(f'{name:14} {n["macro_f1"]:6.3f} {n["balanced_accuracy"]:6.3f} {n["mel_recall"]:6.3f} | '
              f'{str(r["mel_threshold"]):>5} {s["macro_f1"]:6.3f} {s["mel_recall"]:6.3f} {r["val_false_mel_alarms"]:6d}')

    a, b = results['A_tek_model'], results['B_iki_seed']
    f1_diff = b['val_normal']['macro_f1'] - a['val_normal']['macro_f1']
    if abs(f1_diff) < NOISE_F1:
        chosen = min(results, key=lambda k: results[k]['val_false_mel_alarms'])
        reason = f'F1 farki {f1_diff:+.3f} gurultu icinde (< {NOISE_F1}); daha az yanlis melanom alarmi'
    else:
        chosen = max(results, key=lambda k: results[k]['val_normal']['macro_f1'])
        reason = f'F1 farki {f1_diff:+.3f} gurultuyu asiyor; F1 yuksek olan'
    print(f'\nSecilen: {chosen} ({reason})')

    recipe = {'name': 'final_v1', 'chosen': chosen, 'reason': reason, 'weights': 'esit',
              'rule': {'noise_f1': NOISE_F1, 'target_mel_recall': TARGET_MEL_RECALL},
              **{k: results[chosen][k] for k in ('models', 'temperatures', 'mel_threshold')},
              'candidates_val': results}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(recipe, indent=2))
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
