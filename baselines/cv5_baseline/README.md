# cv5_baseline (2026-11-17/18) - v2 için baseline 5-fold CV ve kilitli hedef

v2'nin ölçüm cetveli. final_v1 üyelerinin tarifi (ConvNeXt-Tiny `convnext_tiny.fb_in22k_ft_in1k_384`, 384 px,
tarif v2: 5 + 25 epoch, warmup + cosine, label smoothing 0.1), seed 42, EMA yok, 5-fold CV
(`splits/cv5_v1.csv`: train + val = 9029 görüntü, 6723 lezyon; iç test dışarıda).

- Her fold: diğer 4 fold eğitim, bu fold doğrulama. **Son epoch** alındı (en iyi epoch seçilmedi).
- OOF logit'leri fp32 (AMP'siz). Test seti tahmin edilmedi.
- Kaggle `skin-cancer-train` Version 11 (fold 0-2) ve 12 (fold 3-4), Tesla T4, ~1 saat/fold. Ortam iki koşuda aynı
  (`pip_freeze.txt`, `pip_freeze_run2.txt`). Ağırlıklar (`last_model.pt`) Kaggle çıktısında, git'te değil.
- Özet: `python -m src.evaluation.cv_summary --run-dir baselines/cv5_baseline` → `cv_summary.json`.

## Sonuç (tek model, argmax, kalibrasyon ve eşik yok)

| Fold | macro F1 | balanced accuracy | melanom recall |
|---|---|---|---|
| 0 | 0.776 | 0.749 | 0.599 |
| 1 | 0.791 | 0.795 | 0.692 |
| 2 | 0.780 | 0.773 | 0.662 |
| 3 | 0.796 | 0.792 | 0.687 |
| 4 | 0.778 | 0.787 | 0.672 |
| **Ortalama ± std** | 0.784 ± 0.009 | 0.779 ± 0.019 | 0.662 ± 0.037 |
| **Birleşik OOF (9029)** | **0.785** | **0.779** | **0.662** |

Fold'lar arası 2 × std (bilgi amaçlı gürültü eşiği): F1 0.018, BA 0.038, melanom recall 0.074.

## Kilitli hedef (`target.json`, 2026-11-18, bir kez yazıldı, değiştirilmez)

| | Baseline OOF | Hedef |
|---|---|---|
| macro F1 | 0.7848 | **≥ 0.8048** |
| balanced accuracy | 0.7793 | **≥ 0.7993** |
| melanom recall | 0.6620 | **≥ 0.662** (koruma şartı) |

**Hedef nasıl ölçülür:** bir deney ancak bu tabloyla **aynı şekilde** ölçülürse hedefle karşılaştırılabilir: aynı
5 fold, fold başına tek model, son epoch, birleşik OOF, argmax. Ensemble, kalibrasyon ya da melanom eşiği gibi karar
katmanı hileleri bu karşılaştırmaya girmez; onlar final tarifte (C4) OOF üzerinde ayrıca seçilir.

Deney kabul kriteri (her deney için ayrıca): 5 fold'un en az 4'ünde baseline'dan iyi **ve** ortalama fark > 0.01.

Not: F1 için hedef payı (0.02) fold gürültüsüyle (2 × std = 0.018) aşağı yukarı aynı büyüklükte. Yani hedefi tutan
bir deney, gürültünün ancak biraz üstünde bir iyileşme göstermiş olur.
