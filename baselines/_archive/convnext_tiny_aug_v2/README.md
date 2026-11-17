# convnext_tiny_aug_v2 (2026-10-15) - denendi, kabul edilmedi

`convnext_tiny_v1` ile aynı model ve tarif; tek fark eğitim augmentation'ı (`--augmentation v2`):
1. Önce kareye sıkıştır, sonra rastgele kırp. v1'de 600x450 görüntülerde RandomResizedCrop
   ~%90 ihtimalle aynı sabit orta kesime düşüyordu ve eğitim/değerlendirme farklı oranda
   sıkıştırılmış görüntü görüyordu.
2. Döndürme ±20° yerine ±180°.

Kaggle'da eğitildi (T4, ~35 dk, epoch başına ~1.1 dk). Kod: commit `916bed2`.

## Karar validation setinde verildi

| | ConvNeXt v1 | ConvNeXt v2 |
|---|---|---|
| En iyi val_f1 | **0.744** (epoch 24) | 0.731 (epoch 25) |
| Son 5 epoch ortalama val_f1 | **0.733** | 0.711 |
| En iyi epoch'ta val balanced accuracy | 0.761 | **0.774** |
| Son train_loss / val_loss | 0.184 / 0.793 | 0.227 / 0.769 |

v2 ana metrikte iyileşme göstermedi (fark küçük ve ters yönde). Daha yavaş öğreniyor ve daha
az ezberliyor (en iyi epoch sonuncusu, train_loss daha yüksek); daha uzun eğitimle farklı
olabilirdi ama bu denenmedi (validation'da "kazanana kadar varyasyon deneme" riskinden
kaçınmak için). **v1 tarifinde kalındı.** Test metrikleri Kaggle'da hesaplandı ama karar
validation'da verildiği için bakılmadı ve buraya konmadı.

Not: İki değişiklik birlikte denendiği için tek tek etkileri bilinmiyor.
Ağırlıklar indirilmedi; Kaggle'da `skin-cancer-train` notebook'unun 2. sürüm çıktısında.
