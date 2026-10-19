# convnext_tiny_in22k_384_v1 (2026-10-19) - ImageNet-22k + 384px + tarif v2

ConvNeXt-Tiny, timm `convnext_tiny.fb_in22k_ft_in1k_384` (ImageNet-22k ön eğitim, 384px'te
ince ayarlı), `drop_path_rate=0.1` (torchvision ile aynı). Veri HAM10000 (ek veri yok),
augmentation v1, tarif v2 (finetune lr 1e-4 + 2 epoch warmup + cosine, label smoothing 0.1,
25 finetune epoch). Kaggle Version 9, P100, ~2.1 dk/epoch (toplam ~1 saat). Kod: commit `76a1ec2`.

Önceki ConvNeXt'lerden üç şey birden farklı (22k, 384, tarif v2); kazancın hangisinden
geldiği ayrılamaz.

## Validation

| Model | Ön eğitim | Çöz. | Tarif | En iyi val_f1 | Son 5 epoch ort. |
|---|---|---|---|---|---|
| convnext_tiny_v1 | 1k | 224 | eski | 0.744 | 0.733 |
| convnext_tiny_recipe2_v1 | 1k | 224 | v2 | 0.766 | 0.745 |
| convnext_tiny_384_v1 | 1k | 384 | eski | 0.769 | 0.760 |
| **bu model** | **22k** | **384** | **v2** | **0.803** (epoch 18) | **0.788** |

Kayıtlı tahminlerle (Mac/MPS): F1 0.794, balanced accuracy 0.793, kalibrasyon T 0.93
(NLL 0.444, şimdiye kadarki en düşük). Eğitim sonda oturuyor (son 6 epoch 0.785-0.791).

## Ensemble'a katkısı (`ensemble_val.py --candidate-study`, kalibre, eşit ağırlık)

| | val F1 | Mevcudu geçme (bootstrap) | mel recall | mel precision |
|---|---|---|---|---|
| ensemble_v1 (4 model) | 0.768 | | **0.814** | 0.542 |
| Bu model, ConvNeXt yerine | 0.783 | %93 | 0.716 | 0.598 |
| Bu model, 5. üye | 0.780 | %88 | 0.765 | 0.578 |
| Bu model tek başına | **0.794** | %94 | 0.647 | **0.635** |

## Melanom: asıl zayıf nokta

Model genel F1'de en iyisi ama melanomu daha çok kaçırıyor (val'de 102 melanom). Aynı
melanom yakalama oranında daha fazla yanlış alarm veriyor:

| | eşik | mel recall | yanlış mel alarmı | F1 |
|---|---|---|---|---|
| ensemble_v1 | normal | 0.814 | 70 | 0.768 |
| ensemble_v1 | 0.30 (tarama) | 0.882 | 127 | 0.746 |
| bu model tek | 0.10 | 0.804 | 102 | 0.764 |
| bu model tek | 0.05 | 0.882 | 147 | 0.742 |
| bu model, ConvNeXt yerine | 0.30 | 0.804 | 86 | 0.772 |

Yani melanomu diğer sınıflardan ayırmada (sıralama kalitesinde) ensemble_v1 hâlâ daha iyi;
yeni modelin kazancı diğer sınıflardan geliyor.

## Karar

Hiçbir seçenek gürültü eşiğini aşmadı (en iyisi %94, aralık sıfırı içeriyor) ve melanomda
geriliyor: **ensemble_v1 değişmedi.** Model, tarif v2'li yeni ensemble'ın (ve 5-fold CV'nin)
temel yapı taşı olarak saklanıyor. Ağırlıklar: yerel disk (111MB, git'te değil).

## Test seti (karar verildikten sonra, kayıt amaçlı; 986 görüntü)

| | ensemble_v1 (normal) | convnext_tiny_v1 (224) | **bu model** |
|---|---|---|---|
| macro_f1 | 0.783 | 0.774 | **0.818** (0.775-0.850) |
| balanced_accuracy | 0.809 | 0.798 | 0.809 (0.761-0.852) |
| macro_precision | 0.775 | 0.769 | **0.841** |
| accuracy | 0.835 | 0.802 | **0.870** |
| mel_recall | 0.692 | **0.766** | 0.664 (0.568-0.745) |
| mel_precision | 0.474 | 0.436 | **0.592** |
| mel→nv / nv→mel | 20 / 61 | 7 / 88 | 20 / **37** |

Validation'daki tablo testte de görüldü: genel F1 belirgin şekilde daha iyi (1k-384 modelinin
aksine kazanç testte kaybolmadı). Melanomda ensemble_v1 ile normal çalışma noktasında yakın
(aynı sayıda kaçırılan melanom, çok daha az yanlış alarm); 224'lük ConvNeXt'ten ise belirgin
şekilde fazla kaçırıyor. Tek test bakışıdır, karar değişmez.
