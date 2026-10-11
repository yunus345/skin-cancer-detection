# efficientnet_v2_s_v1 (2026-10-11)

`convnext_tiny_v1` ile aynı tarif (dedup fix, class-weighted CE, albumentations
augmentation, Gray-World color constancy, 5 feature-extract + 20 finetune epoch,
patience 6). Fark: mimari EfficientNetV2-S ve giriş boyutu onun ön-eğitim
boyutu olan **384px** (ConvNeXt 224px'teydi). Bu yüzden ConvNeXt'le arasındaki
farkın ne kadarının mimariden, ne kadarının çözünürlükten geldiği ayrılamaz.

Bu koşuda ilk kez hız ayarları açıktı (Colab GPU'sunda AMP, `num_workers=2`,
cudnn.benchmark); sonucu pratikte değiştirmezler.

## Bu sonucu üreten kod

`main` (commit `771dcb8` ve sonrası), Colab notebook'undaki eğitim hücresi:
`--architecture efficientnet_v2_s --feature-extract-epochs 5 --finetune-epochs 20 --patience 6`
(çözünürlük otomatik 384px). Batch size varsayılan 32: AMP sayesinde 384px'te
T4'ün belleğine sığdı. Süre ölçülmedi; gözleme göre ConvNeXt koşusuna (~2 saat)
yakın sürdü, oysa hesaplama yükü ~2 kat. Hız ayarlarının kabaca yarı yarıya
kazanç sağladığını düşündürüyor.

Eğitim 25 epoch'un tamamını bitirdi, patience tetiklenmedi. En iyi epoch: 21
(val_f1=0.7494).

## Test seti sonuçları

| Katman | Metrik | baseline-v1 (B0) | convnext_tiny_v1 | efficientnet_v2_s_v1 |
|---|---|---|---|---|
| 1. Headline | macro_f1 | 0.698 (0.642–0.742) | **0.774** (0.728–0.811) | 0.753 (0.703–0.793) |
| 2. Dış karşılaştırma | balanced_accuracy (ISIC) | 0.762 (0.716–0.804) | **0.798** (0.746–0.846) | 0.781 (0.728–0.830) |
| 2. Dış karşılaştırma | macro_precision | 0.665 | **0.769** | 0.739 |
| 3. Klinik | mel_recall | 0.626 | **0.766** | 0.710 |
| 3. Klinik | mel_precision | 0.392 | 0.436 | **0.455** |
| (bilgi) | accuracy | 0.779 | 0.802 | **0.828** |

## Bulgular

- **Tek başına ConvNeXt'i geçmedi**, ama fark küçük ve güven aralıkları büyük
  ölçüde örtüşüyor.
- **ConvNeXt'ten farklı yerlerde hata yapıyor** (ensemble için iyi haber):
  V2-S bcc/bkl/nv'de daha iyi ve daha az yanlış melanom alarmı veriyor
  (nv→mel 66, ConvNeXt'te 88); ConvNeXt akiec/df/vasc'ta daha iyi ve melanomu
  çok daha az kaçırıyor (mel→nv 7, V2-S'te 19). df (14 örnek) ve vasc (13 örnek)
  farkları küçük örneklemden dolayı gürültülü.
- **Daha fazla epoch yaramaz:** val_loss epoch 11-14 civarında ~0.68 ile dibe
  vurup 0.83'e çıktı, train_loss 0.18'e düşmeye devam etti (ezberleme başlıyor).

## Ağırlıklar (best_model.pt, 82MB)

Git'e dahil edilmedi (`.gitignore`: `baselines/*/best_model.pt`). İki yerde:
- Bu klasörde yerel diskte
- Google Drive: `skin-cancer-detection-runs/archive/efficientnet_v2_s_v1/`

Modeli yeniden yüklemek için: `src.evaluation.evaluate --checkpoint
baselines/efficientnet_v2_s_v1/best_model.pt` (384px checkpoint'ten otomatik okunur).
