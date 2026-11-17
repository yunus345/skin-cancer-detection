# convnext_tiny_isic2019_v1 (2026-10-16) - ek veri denemesi, kabul edilmedi

`convnext_tiny_v1` ile aynı model ve tarif (224px, augmentation v1); tek fark eğitim verisi:
HAM10000'in 8037 train görüntüsüne ISIC 2019'un HAM dışı 14885 görüntüsü eklendi
(`src/data/add_isic2019.py`: SCC hariç, AK→akiec; val/test HAM10000'de aynen kaldı).
Train 22922 görüntü: 11982 Barselona (BCN), 819 New York (MSK), 2084 eski ISIC arşivi.
Kaggle'da eğitildi (T4, epoch başına ~6.4 dk, early stopping epoch 23, en iyi epoch 17).
Kod: commit `17902e3`, Kaggle notebook'u `USE_ISIC2019 = True`.

## Validation (karar burada verildi)

| | ConvNeXt (ek verisiz) | ConvNeXt (ek verili) |
|---|---|---|
| En iyi val_f1 (eğitim sırasında) | 0.744 | **0.760** |
| Son 5 epoch ortalama val_f1 | 0.733 | **0.742** |
| En düşük val_loss | 0.667 | **0.534** |

Ensemble'a katkısı (kalibre, `ensemble_val.py --candidate-study`): eskisinin yerine F1 0.767,
5. üye 0.761 (mevcut 0.768); gürültüyü aşan kazanç yok. Tek başına val mel recall 0.676
(ek verisiz 0.824). **Ek veri hattı durduruldu, ensemble_v1 değişmedi.**

## Test seti (karar verildikten sonra, kayıt amaçlı)

| | ConvNeXt (ek verisiz) | ConvNeXt (ek verili) |
|---|---|---|
| macro_f1 | **0.774** (0.728–0.811) | 0.744 (0.689–0.787) |
| balanced_accuracy | **0.798** | 0.751 |
| mel_recall | **0.766** | 0.626 |
| mel_precision | 0.436 | **0.549** |
| mel→nv / nv→mel | 7 / 88 | 22 / 36 |
| akiec recall | **0.73** | 0.50 |
| nv recall | 0.82 | **0.91** |

## Neden işe yaramadı (olası açıklamalar, test edilmedi)

1. **Model "nv" demeye kaydı.** Daha az yanlış melanom alarmı (36'ya karşı 88), ama çok daha
   çok kaçan melanom (22'ye karşı 7). Melanom eğitimde %11'den %19'a çıktığı için class
   weighting melanomun ağırlığını düşürdü.
2. **akiec'in anlamı değişti.** HAM10000'in akiec'i AK + Bowen hastalığı; ek veride SCC'yi
   (Bowen'ı da içerebilir) çıkardığımız için eklenen akiec'ler sadece AK. Model akiec'i daha
   dar öğrendi, HAM test setindeki Bowen örneklerini kaçırıyor (recall 0.73 → 0.50).
3. **Farklı hastane (domain shift).** Yeni görüntülerin çoğu Barselona'dan; farklı cihaz ve
   çekim koşulları (birçoğunda siyah dairesel kenar). Val/test ise tamamen Viyana.

İleride denenebilir: sınıf ağırlıklarını HAM10000 oranlarına göre sabitlemek, kaynak-dengeli
örnekleme, BCN görüntülerindeki siyah kenarları kırpmak.

## Ağırlıklar (111MB)

Git'te değil: yerel disk + Google Drive `archive/convnext_tiny_isic2019_v1/`.
