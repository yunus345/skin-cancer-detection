# swin_v2_t_v1 (2026-10-13)

`convnext_tiny_v1` ve `efficientnet_v2_s_v1` ile aynı tarif (dedup fix, class-weighted
CE, albumentations augmentation, Gray-World color constancy, 5 feature-extract + 20
finetune epoch, patience 6). Fark: mimari Swin V2-Tiny (transformer) ve giriş boyutu
onun ön-eğitim boyutu olan **256px**.

İlk kez **Kaggle**'da eğitildi (T4 GPU, AMP, `num_workers=4`,
albumentations 2.0.8). Veri bölünmesi önceki koşularla birebir aynı
(train 8037 / val 992 / test 986), yani test sonuçları doğrudan karşılaştırılabilir.

## Bu sonucu üreten kod

`main` (commit `8dc0768`), [kaggle/skin-cancer-train.ipynb](../../kaggle/skin-cancer-train.ipynb)
`ARCHITECTURE = 'swin_v2_t'` ile, Mac'ten `kaggle kernels push -p kaggle` ile gönderildi.

**Early stopping tetiklendi:** epoch 21'de durdu, en iyi epoch 15 (val_f1=0.7323).
Epoch başına süre ölçüldü: Faz 1 ~1.1 dk, Faz 2 ~1.7 dk; toplam ~40 dakika (Kaggle
koşusunun tamamı, veri hazırlama dahil).

## Test seti sonuçları

| Katman | Metrik | baseline-v1 (B0) | convnext_tiny_v1 | efficientnet_v2_s_v1 | swin_v2_t_v1 |
|---|---|---|---|---|---|
| 1. Headline | macro_f1 | 0.698 | **0.774** | 0.753 | 0.755 (0.709–0.793) |
| 2. Dış karşılaştırma | balanced_accuracy (ISIC) | 0.762 | 0.798 | 0.781 | **0.813** (0.766–0.854) |
| 2. Dış karşılaştırma | macro_precision | 0.665 | **0.769** | 0.739 | 0.719 |
| 3. Klinik | mel_recall | 0.626 | **0.766** | 0.710 | 0.626 |
| 3. Klinik | mel_precision | 0.392 | 0.436 | 0.455 | **0.472** |
| (bilgi) | accuracy | 0.779 | 0.802 | **0.828** | 0.808 |

## Bulgular

- **macro_f1'de V2-S ile aynı seviyede, ConvNeXt'in biraz gerisinde**; güven aralıkları
  büyük ölçüde örtüşüyor.
- **Balanced accuracy'de şu ana kadarki en yüksek (0.813):** nadir sınıflarda çok iyi
  (akiec recall 0.80, bcc 0.90, df 0.86). Diğer modellerin zayıf olduğu akiec'te en iyisi.
- **Zayıf yanı melanom:** mel_recall 0.626 (ConvNeXt 0.766). Gerçek melanomların 17'sini
  nv sandı (ConvNeXt 7, V2-S 19). Buna karşılık en az yanlış melanom alarmı veren model
  (nv→mel 59; ConvNeXt 88, V2-S 66).
- Ensemble için: hataları ConvNeXt'ten farklı yerlerde. Swin nadir sınıflarda, ConvNeXt
  melanomda güçlü.

## Ağırlıklar (best_model.pt, 111MB)

Git'e dahil edilmedi (`.gitignore`: `baselines/*/best_model.pt`). İki yerde:
- Bu klasörde yerel diskte
- Google Drive: `skin-cancer-detection-runs/archive/swin_v2_t_v1/`

(Kaggle'daki notebook'un 1. sürümünün çıktısında da duruyor.)
