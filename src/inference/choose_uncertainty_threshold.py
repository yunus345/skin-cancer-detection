"""Arayuzdeki "emin degilim" esigini VALIDATION setinde secer (test setine bakilmaz).

Guven = kalibre ortalama olasiligin en buyugu. Esigin altindaki tahminlerde arayuz karar yerine
"doktor incelemesi onerilir" uyarisi gosterir. Kalibrasyon (T) burada onemli: T ile ayarlanmis
olasiliklarda "0.9" gercekten yaklasik %90 dogruluk anlamina geliyor.

Kural (calistirmadan once yazildi, 2026-11-17): tau 0.30-0.95, 0.05 adim. "Emin" tahminlerde val
dogrulugu >= 0.95 saglayan EN KUCUK tau secilir (model "eminim" dediginde 20 vakanin en az 19'unda
hakli olsun; en kucuk tau = en az sayida vaka doktora yonlendirilir).

Cikti: baselines/final_v1/uncertainty.json (package_model.py config.json'a ekler).
Calistirma: python -m src.inference.choose_uncertainty_threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TARGET_ACCURACY = 0.95
CANDIDATES = np.round(np.arange(0.30, 0.951, 0.05), 2)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    recipe_path = Path('baselines/final_v1/recipe.json')
    recipe = json.loads(recipe_path.read_text())
    probs, targets, labels = [], None, None
    for name, path in recipe['models'].items():
        saved = np.load(path, allow_pickle=True)
        targets, labels = saved['val_targets'], list(saved['labels'])
        probs.append(softmax(saved['val_logits'] / recipe['temperatures'][name]))
    probs = np.mean(probs, axis=0)
    confidence, predictions = probs.max(axis=1), probs.argmax(axis=1)
    correct = predictions == targets
    mel = labels.index('mel')

    rows, chosen = [], None
    print(f'{"tau":>5} {"emin %":>7} {"emin dogruluk":>14} {"yakalanan hata":>15} {"emin ama kacan mel":>19}')
    for tau in CANDIDATES:
        sure = confidence >= tau
        accuracy = float(correct[sure].mean()) if sure.any() else float('nan')
        caught = int((~correct & ~sure).sum())
        missed_mel_sure = int(((targets == mel) & (predictions != mel) & sure).sum())
        rows.append({'tau': float(tau), 'sure_fraction': float(sure.mean()), 'sure_accuracy': accuracy,
                     'errors_flagged': caught, 'errors_total': int((~correct).sum()),
                     'missed_mel_while_sure': missed_mel_sure})
        print(f'{tau:5.2f} {100 * sure.mean():7.1f} {accuracy:14.3f} {caught:>8d} / {(~correct).sum():<4d} {missed_mel_sure:>19d}')
        if chosen is None and accuracy >= TARGET_ACCURACY:
            chosen = rows[-1]

    if chosen is None:
        raise SystemExit('Hicbir esik hedefi saglamadi; kural geregi esik secilmedi.')
    print(f'\nSecilen tau = {chosen["tau"]:.2f}: vakalarin %{100 * (1 - chosen["sure_fraction"]):.0f}\'i '
          f'"doktor incelemesi" olarak isaretlenir, emin olunanlarda dogruluk {chosen["sure_accuracy"]:.3f}.')
    out = recipe_path.parent / 'uncertainty.json'
    out.write_text(json.dumps({
        'threshold': chosen['tau'], 'rule': f'val emin-dogruluk >= {TARGET_ACCURACY} saglayan en kucuk tau',
        'chosen': chosen, 'scan': rows, 'set': 'val (992)',
    }, indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
