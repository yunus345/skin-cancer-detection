# baseline-v1 (2026-10-08)

Bu klasör, projedeki ilk "tam" sonucun kalıcı kaydı. Bundan sonra yapılacak
köklü değişiklikler (yeni mimari, farklı augmentation, ensemble vb.) bu
sayılarla kıyaslanacak — amaç, denemeler işe yaramazsa geri dönülecek,
garanti bir referans noktası tutmak.

## Bu sonucu üreten kod

Git tag: `baseline-v1`
Geri dönmek için: `git checkout baseline-v1`

İçerik: duplicate image_id düzeltmesi, albumentations tabanlı augmentation,
Gray-World color constancy ön işleme, class-weighted CrossEntropy loss,
scheduler yok (sabit LR per faz), 5 feature-extract + 20 finetune epoch
(patience=6, hiç tetiklenmedi — eğitim tam 25 epoch'u bitirdi).

## Test seti sonuçları (test_metrics.json)

| Katman | Metrik | Değer (%95 GA) |
|---|---|---|
| 1. Headline | macro_f1 | 0.698 (0.642–0.742) |
| 2. Dış karşılaştırma | balanced_accuracy (ISIC) | 0.762 (0.716–0.804) |
| 2. Dış karşılaştırma | macro_precision | 0.665 (0.605–0.723) |
| 3. Klinik | mel_recall | 0.626 (0.531–0.718) |
| 3. Klinik | mel_precision | 0.392 (0.313–0.464) |

accuracy (bilgi amaçlı, dengesiz veride yanıltıcı): 0.779

Referans: aynı ISIC 2018 verisiyle çalışan bir makalede balanced_accuracy
~0.82 raporlanmış (ensemble + CE + color constancy kullanarak).

## Dikkat çeken hata türü (confusion matrix)

107 gerçek melanomadan 22'si "nv" (iyi huylu) olarak tahmin edilmiş —
en riskli hata türü. 664 nv örneğinden 77'si "mel" tahmin edilmiş (daha az
riskli ama mel_precision'ı düşürüyor).

## Ağırlıklar (best_model.pt)

Bu klasörde doğrudan git'e dahil edildi (16MB) — Drive'a veya yerel
diske bağımlı kalmadan kalıcı, garanti bir kopya olsun diye. Modeli
yeniden yüklemek için: `src.evaluation.evaluate --checkpoint
baselines/baseline_v1/best_model.pt`.
