# convnext_tiny_recipe2_v1 (2026-10-17) - tarif v2, düzeltilmiş label smoothing

`convnext_tiny_v1` ile aynı model, veri (HAM10000, ek veri yok), çözünürlük (224) ve
augmentation (v1). Değişen sadece eğitim tarifi:
- finetune lr 1e-5 (sabit) → **1e-4, 2 epoch warmup + cosine** (Faz 1 aynı)
- **label smoothing 0.1** (`WeightedLabelSmoothingCE`: sınıf ağırlığı sadece gerçek sınıfa)
- 20 → **25 finetune epoch**, early stopping fiilen kapalı (patience 30)

Kaggle Version 6 (Version 5 aynı tarifin PyTorch label smoothing hatalı koşusuydu, geçersiz).
Kod: commit `8cd99e7`.

## Validation

| | Eski tarif | Tarif v2 |
|---|---|---|
| En iyi val_f1 (eğitim sırasında) | 0.744 (epoch 24) | **0.766** (epoch 16) |
| Son 5 epoch ortalama val_f1 | 0.733 | **0.745** |
| Son epoch'larda gidiş | hâlâ yükseliyor | **düz (oturmuş)** |
| Kalibrasyon T / val NLL | 1.47 / 0.588 | **0.92 / 0.497** |
| Tek model val F1 (kayıtlı tahminlerle) | 0.736 | **0.767** |

Ensemble'a katkısı (`ensemble_val.py --candidate-study`): eski ConvNeXt'in yerine 0.780
(mevcut 0.768, +0.012, bootstrap'te %78; gürültü eşiğini aşmıyor, mel recall 0.814 → 0.745),
5. üye 0.762.

## Bulgular

- Tarif v2 tek modeli belirgin şekilde iyileştirdi: tek bir ConvNeXt artık 4 modelli
  ensemble kadar iyi (val F1 0.767'ye karşı 0.768).
- **Aşırı özgüven çözüldü:** T 1.47 → 0.92 (artık hafif çekingen), NLL düştü. Label smoothing
  beklenen işi yaptı.
- Eğitim artık sonunda oturuyor (cosine): "epoch yetmiyor" sorunu bu tarifle kalktı.
- Kural gereği ensemble_v1 değişmedi. Doğru sonraki adım: diğer modelleri de tarif v2 ile
  eğitip tamamen tarif v2'li bir ensemble kurmak (tercihen 5-fold CV ile).

Test sonuçları karar verilmediği için bakılmadı. Ağırlıklar: yerel disk (111MB, git'te değil).
