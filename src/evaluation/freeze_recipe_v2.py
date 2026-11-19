"""final_v2 tarifini birleşik OOF'ta (Mac, fp32) seçer ve dondurur. Test setine bakmaz.

Kurallar bu betik çalıştırılmadan önce `baselines/final_v2/README.md`'de yazıldı ve commit edildi:
- T: her fold modeli için kendi OOF'unda `fit_temperature` (v1 ile aynı fonksiyon).
- Tarama eşiği t: birleşik OOF'ta (kalibre, tek model olasılıkları) melanom recall >= 0.80 sağlayan EN YÜKSEK eşik, 0.01 adım.
- "Emin değilim" eşiği tau: 0.30-0.95 (0.05 adım), emin olunanlarda doğruluk >= 0.95 sağlayan EN KÜÇÜK tau.

Neden birleşik OOF: her görüntü, onu eğitimde görmemiş tek bir model tarafından tahmin edildi; 9029 görüntü (v1'de val 992).
Bilinen sınırlama: testte 5 modelin ortalaması kullanılır, burada tek model olasılıkları (README).

    python -m src.evaluation.freeze_recipe_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.ensemble_val import fit_temperature, score, softmax
from src.evaluation.evaluate import apply_threshold_override
from src.inference.choose_uncertainty_threshold import CANDIDATES as TAU_CANDIDATES, TARGET_ACCURACY

OUT_DIR = Path('baselines/final_v2')
TARGET_MEL_RECALL = 0.80
THRESHOLDS = np.round(np.arange(0.99, 0.0, -0.01), 2)  # yüksekten düşüğe: ilk tutan = en yüksek


def main():
    probs, targets, image_ids, temperatures, models = [], [], [], {}, {}
    labels = None
    for k in range(5):
        oof = np.load(OUT_DIR / 'oof_mac' / f'oof_fold{k}.npz', allow_pickle=True)
        labels = labels or list(oof['labels'])
        assert list(oof['labels']) == labels
        name = f'fold{k}'
        temperatures[name] = fit_temperature(oof['logits'], oof['targets'])
        models[name] = {'weights': f'models/cv5_baseline/fold{k}/last_model.pt', 'md5': str(oof['weights_md5'])}
        probs.append(softmax(oof['logits'] / temperatures[name]))
        targets.append(oof['targets'])
        image_ids.append(oof['image_ids'])
    probs, targets, image_ids = np.concatenate(probs), np.concatenate(targets), np.concatenate(image_ids)
    assert len(set(image_ids)) == len(image_ids) == 9029, 'OOF eksik ya da tekrar var'
    mel = labels.index('mel')
    print('T:', {k: round(v, 2) for k, v in temperatures.items()})

    # Tarama eşiği
    mel_threshold = None
    for t in THRESHOLDS:
        s = score(targets, apply_threshold_override(probs, mel, float(t)), mel)
        if s['mel_recall'] >= TARGET_MEL_RECALL:
            mel_threshold = float(t)
            break
    assert mel_threshold is not None
    normal = score(targets, probs.argmax(1), mel)
    screening = score(targets, apply_threshold_override(probs, mel, mel_threshold), mel)
    false_alarm = lambda pred: int(((pred == mel) & (targets != mel)).sum())
    print(f'OOF normal: F1 {normal["macro_f1"]:.4f} BA {normal["balanced_accuracy"]:.4f} melR {normal["mel_recall"]:.4f}')
    print(f'Tarama eşiği t = {mel_threshold:.2f}: F1 {screening["macro_f1"]:.4f} melR {screening["mel_recall"]:.4f} '
          f'yanlış mel alarmı {false_alarm(apply_threshold_override(probs, mel, mel_threshold))} '
          f'(normal: {false_alarm(probs.argmax(1))})')

    # "Emin değilim" eşiği
    confidence, predictions = probs.max(1), probs.argmax(1)
    correct = predictions == targets
    tau_rows, tau = [], None
    for candidate in TAU_CANDIDATES:
        sure = confidence >= candidate
        row = {'tau': float(candidate), 'sure_fraction': float(sure.mean()),
               'sure_accuracy': float(correct[sure].mean()) if sure.any() else float('nan'),
               'errors_flagged': int((~correct & ~sure).sum()), 'errors_total': int((~correct).sum()),
               'missed_mel_while_sure': int(((targets == mel) & (predictions != mel) & sure).sum())}
        tau_rows.append(row)
        if tau is None and row['sure_accuracy'] >= TARGET_ACCURACY:
            tau = row
    assert tau is not None, 'Hiçbir tau hedefi sağlamadı'
    print(f'tau = {tau["tau"]:.2f}: vakaların %{100 * (1 - tau["sure_fraction"]):.0f}\'i doktora, emin doğruluk '
          f'{tau["sure_accuracy"]:.3f}, emin iken kaçan melanom {tau["missed_mel_while_sure"]}')

    recipe = {
        'name': 'final_v2', 'architecture': 'convnext_tiny_in22k_384', 'image_size': 384, 'labels': labels,
        'models': models, 'temperatures': temperatures, 'weights': 'esit', 'mel_threshold': mel_threshold,
        'uncertainty': {'threshold': tau['tau'], 'chosen': tau, 'scan': tau_rows,
                        'rule': f'OOF emin-dogruluk >= {TARGET_ACCURACY} saglayan en kucuk tau'},
        'selection_set': 'birlesik OOF, Mac fp32 (baselines/final_v2/oof_mac, 9029)',
        'rule': {'target_mel_recall': TARGET_MEL_RECALL, 'threshold_step': 0.01},
        'oof_normal': normal, 'oof_screening': screening,
    }
    (OUT_DIR / 'recipe.json').write_text(json.dumps(recipe, indent=2, ensure_ascii=False))
    print(f'Wrote {OUT_DIR / "recipe.json"}')


if __name__ == '__main__':
    main()
