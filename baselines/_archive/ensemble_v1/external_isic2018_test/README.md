# Dış doğrulama: ensemble_v1, ISIC 2018 Task 3 resmi test seti (2026-10-16)

**Bu setteki ilk ölçüm.** Donmuş `ensemble_v1` tarifi (4 model, HAM10000 validation'ında
bulunan T'lerle kalibre, eşit ağırlık); bu sette hiçbir şey ayarlanmadı.
Kod: `src/evaluation/external_validation.py`. 1512 görüntü (nv 909, bkl 217, mel 171, bcc 93,
df 44, akiec 43, vasc 35). Kurallar: `external_test/README.md`.

## Sonuç

| | Kendi test setimiz (HAM10000, 986) | **Dış test (ISIC 2018, 1512)** |
|---|---|---|
| macro_f1, normal | 0.783 (0.736–0.823) | **0.749** (0.711–0.780) |
| balanced_accuracy, normal | 0.809 | **0.765** (0.728–0.800) |
| mel_recall, normal | 0.692 | **0.749** |
| macro_f1, tarama 0.30 | 0.781 | **0.729** |
| balanced_accuracy, tarama 0.30 | 0.823 | **0.757** |
| mel_recall, tarama 0.30 | 0.888 | **0.842** |

Tek modeller (dış test, macro_f1 / balanced_accuracy): B0 0.651 / 0.697, ConvNeXt 0.724 / 0.731,
V2-S 0.719 / 0.734, Swin 0.728 / 0.760.

## Bulgular

- **Kendi test setimiz iyimsermiş:** balanced accuracy 0.809 → 0.765 (yaklaşık 4.5 puan).
  Küçük set ve aynı sete birçok kez bakılması bu iyimserliği açıklıyor; dış doğrulamanın
  amacı tam olarak buydu.
- **Ensemble bağımsız veride de işe yarıyor:** en iyi tek modelden +0.02 macro_f1
  (0.749'a karşı 0.728).
- **Yarışmayla aynı sette kıyas:** ek verisiz en iyi skor 0.845 (Zhuang ve ark.), bizim 0.765.
  Aradaki fark gerçek: ~8 puan.
- **Tarama modu burada daha pahalı:** melanom recall 0.75 → 0.84, ama yanlış melanom alarmı
  79 → 173 ve F1 -0.02. Kendi test setimizdeki "neredeyse bedava" sonuç burada tekrarlanmadı.
- En zayıf sınıflar: akiec (recall 0.56), melanom (precision 0.52).
