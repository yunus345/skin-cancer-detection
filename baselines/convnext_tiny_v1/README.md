# convnext_tiny_v1 (2026-10-10)

`baseline-v1` ile aynı pipeline (dedup fix, class-weighted CE, albumentations
augmentation, Gray-World color constancy), **tek fark mimari**: EfficientNet-B0
yerine ConvNeXt-Tiny. Temiz bir karşılaştırma için başka hiçbir şey
değiştirilmedi (aynı epoch sayısı, aynı 224px, aynı augmentation).

## Bu sonucu üreten kod

Git branch: `mimari-convnext-tiny`
`--architecture convnext_tiny` dışında `baseline-v1` ile aynı CLI bayrakları:
`--feature-extract-epochs 5 --finetune-epochs 20 --patience 6`

Eğitim tam 25 epoch'u bitirdi (5 feature-extract + 20 finetune), patience
hiç tetiklenmedi. En iyi epoch: 24 (val_f1=0.7444).

## Test seti sonuçları — baseline-v1 ile kıyaslama

| Katman | Metrik | baseline-v1 (EfficientNet-B0) | convnext_tiny_v1 |
|---|---|---|---|
| 1. Headline | macro_f1 | 0.698 (0.642–0.742) | **0.774** (0.728–0.811) |
| 2. Dış karşılaştırma | balanced_accuracy (ISIC) | 0.762 (0.716–0.804) | **0.798** (0.746–0.846) |
| 2. Dış karşılaştırma | macro_precision | 0.665 (0.605–0.723) | **0.769** (0.723–0.810) |
| 3. Klinik | mel_recall | 0.626 (0.531–0.718) | **0.766** (0.679–0.845) |
| 3. Klinik | mel_precision | 0.392 (0.313–0.464) | 0.436 (0.362–0.506) |
| (bilgi) | accuracy | 0.779 | 0.802 |

## En önemli bulgu: melanoma kaçırma hatası ciddi azaldı

Confusion matrix'te, 107 gerçek melanomadan "nv" (iyi huylu) diye yanlış
tahmin edilenler: `baseline-v1`'de **22**, burada **7**. En riskli hata
türünde (kanseri kaçırma) ~3 kat iyileşme — sadece mimari değişikliğiyle.

## Ağırlıklar (best_model.pt, 106MB)

Git'e dahil **edilmedi** — GitHub standart push 100MB üstü dosyaları
reddediyor (`baseline-v1`'in 16MB'lik EfficientNet-B0 ağırlığından farklı,
ConvNeXt-Tiny ~28M parametre ile çok daha büyük). İki yerde güvende:
- Bu klasörde yerel diskte (`.gitignore`'a özel olarak eklendi)
- Google Drive: `skin-cancer-detection-runs/latest/` (Drive Desktop ile
  bu Mac'e senkronize)

Modeli yeniden yüklemek için: `src.evaluation.evaluate --checkpoint
baselines/convnext_tiny_v1/best_model.pt`
