# convnext_tiny_in22k_384_seed2_ema (2026-10-20) - Tiny seed 2 + EMA gölge kopya

`convnext_tiny_in22k_384_v1`'in birebir aynısı, tek fark `--seed 2`; ayrıca `--ema-decay 0.999`
(EMA eğitimi etkilemez: aynı seed ile EMA'lı/EMA'sız koşular birebir aynı, Mac'te doğrulandı).
Amaç iki soruyu tek koşuda cevaplamak: (1) sadece seed değişince sonuç ne kadar oynuyor (ölçüm
cetveli), (2) EMA ne kazandırıyor. Kaggle Version 10, T4, 2.2 dk/epoch. Kod: commit `d3bafda`.

## Validation (hepsi kalibre; "alarm" = mel recall ≥ 0.80 için gereken yanlış melanom alarmı)

| | F1 | BAcc | mel recall | NLL | alarm | son 5 epoch F1 |
|---|---|---|---|---|---|---|
| Tiny seed 42 (önceki koşu) | 0.794 | 0.793 | 0.647 | 0.441 | 94 | 0.788 |
| **Tiny seed 2** | 0.807 | 0.821 | 0.755 | 0.422 | 80 | 0.775 |
| Tiny seed 2 EMA | 0.784 | 0.799 | 0.755 | **0.409** | 76 | 0.775 |
| Small (seed 42) | 0.801 | 0.802 | 0.608 | 0.443 | 98 | 0.780 |
| Small EMA | 0.785 | 0.798 | 0.765 | 0.424 | 73 | 0.781 |
| Tiny s42 + s2 (seed ensemble) | 0.793 | 0.795 | 0.696 | - | 73 | - |
| (ensemble_v1, 4 model) | 0.768 | 0.793 | 0.814 | - | 70 | - |

## Bulgular

1. **Ölçüm cetveli:** sadece seed değişince son 5 epoch F1 0.013, en iyi checkpoint F1 0.013,
   mel recall 0.11 (!) oynuyor. Tek koşuluk tek model kararlarında bundan küçük farklara güvenilmez;
   özellikle melanom recall tek koşuda çok gürültülü.
2. **EMA:** F1'i artırmıyor (son 5 epoch aynı), ama iki modelde de olasılık kalitesi (NLL) daha iyi
   ve melanom ayırma biraz daha iyi (Tiny 80 → 76, Small 98 → 73 alarm). Eğitimin ortasındaki
   zıplamaları neredeyse yok ediyor. Bedava olduğu için tarife ekleniyor; özellikle val'siz final
   eğitimde (son epoch alınacak) işe yarar.
3. **Small:** gürültünün içinde, 1.6 kat pahalı: kabul edilmedi.
4. **Seed ensemble** (2 Tiny): melanom ayırmada ensemble_v1 seviyesi (73'e 70 alarm), F1 daha yüksek.

Test sonuçlarına bakılmadı. Ağırlıklar yerel disk (111MB, git'te değil).
