# convnext_tiny_384_v1 (2026-10-15)

`convnext_tiny_v1` ile aynı model ve tarif (augmentation v1); tek fark çözünürlük:
224px yerine **384px** (`--image-size 384`). Kaggle'da eğitildi (T4, ~55 dk, epoch başına
~2 dk). Kod: commit `04d411a`. 25 epoch'un tamamı, en iyi epoch 21.

## Validation (kararlar burada verildi)

| | ConvNeXt 224 | ConvNeXt 384 |
|---|---|---|
| En iyi val_f1 (eğitim sırasında) | 0.744 | **0.769** |
| Son 5 epoch ortalama val_f1 | 0.733 | **0.760** |
| val balanced accuracy | 0.761 | **0.812** |

Ensemble'a katkısı (kalibre, eşit ağırlık, `ensemble_val.py --candidate-study`):

| | val F1 | Mevcudu geçme (bootstrap) | val mel recall |
|---|---|---|---|
| Mevcut ensemble (4 model) | 0.768 | | **0.814** |
| 384, ConvNeXt 224'ün yerine | 0.772 | %65 | 0.755 |
| 384, 5. üye | 0.757 | %18 | 0.824 |

Önceden belirlenen kural (gürültüyü aşan iyileşme) sağlanmadı: **ensemble_v1 değişmedi.**

## Test seti (karar verildikten sonra, kayıt amaçlı)

| | ConvNeXt 224 | ConvNeXt 384 |
|---|---|---|
| macro_f1 | 0.774 (0.728–0.811) | 0.771 (0.725–0.807) |
| balanced_accuracy | 0.798 | 0.784 |
| mel_recall | **0.766** | 0.664 |
| mel→nv / nv→mel | 7 / 88 | 15 / 61 |

## Bulgu

Validation'daki büyük sıçrama (+0.025 F1) test setinde tekrarlanmadı (−0.003). Validation ve
test setleri küçük (nadir sınıflarda 12-14 örnek) ve tek bir model koşusu da rastgelelik
taşıyor; bu yüzden "çözünürlük yardımcı oluyor" sonucu kesin değil. Ensemble kararını test
setine bakmadan vermiş olmak burada işe yaradı.

## Ağırlıklar (111MB)

Git'te değil: yerel disk + Google Drive `archive/convnext_tiny_384_v1/` (MD5 doğrulandı).
