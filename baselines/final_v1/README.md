# final_v1 (2026-11-12) - dondurulmuş final tarif

Bu klasör v1'in final modelini tanımlar. Tarif **yalnızca validation setine bakılarak** seçildi ve bu
dosya test ölçümünden **önce** commit edildi. Bu noktadan sonra tarif değişmez: test sonucu ne
çıkarsa o raporlanır.

## Tarif (`recipe.json`)

- **Modeller:** iki ConvNeXt-Tiny (timm `convnext_tiny.fb_in22k_ft_in1k_384`, ImageNet-22k, 384px,
  tarif v2), aynı ayarlar, yalnızca seed farklı:
  - `baselines/convnext_tiny_in22k_384_v1/best_model.pt` (seed 42)
  - `baselines/convnext_tiny_in22k_384_seed2_ema/best_model.pt` (seed 2, ham model; EMA kopyası değil)
- **Kalibrasyon:** her model val'de temperature scaling ile. T: seed 42 = 0.93, seed 2 = 0.89.
- **Birleştirme:** kalibre olasılıkların eşit ağırlıklı ortalaması.
- **İki çalışma noktası:**
  - Normal: argmax.
  - Tarama: ortalama melanom olasılığı ≥ **0.28** ise melanom.

Ağırlıkların MD5'leri: seed 42 `7e264e571ca56ef86524cc95d27b99cf`, seed 2
`8f0d8a42a7cdcc284a66aefeace5984e`. İki kopya: yerel disk ve Drive `archive/<klasör adı>/`.

## Seçim kuralı (önceden yazıldı)

Kural, seçim betiği çalıştırılmadan önce `PROJECT_STATUS.md`'de (denetim raporu, Adım 2) yazılıydı:

1. Adaylar: A = seed 42 tek model, B = seed 42 + seed 2 ortalaması.
2. Val macro F1 farkı gürültü cetvelinden (0.013) küçükse, val'de melanom recall ≥ 0.80 sağlayan
   noktada daha az yanlış melanom alarmı veren aday seçilir.
3. Tarama eşiği: val'de melanom recall ≥ 0.80 sağlayan en yüksek eşik (0.01 adımlı).

Tek sayı kaynağı: Mac'te (fp32) hesaplanmış `predictions.npz` dosyaları. Kod:
`python -m src.evaluation.freeze_recipe` (yalnızca `val_*` dizilerini okur).

## Validation sonuçları (992 görüntü, 102 melanom)

| Aday | F1 | BA | mel recall | Tarama eşiği | Tarama F1 | Tarama mel recall | Yanlış mel alarmı |
|---|---|---|---|---|---|---|---|
| A: seed 42 tek model | 0.794 | 0.793 | 0.647 | 0.11 | 0.767 | 0.804 | 94 |
| **B: iki seed ortalaması** | 0.793 | 0.795 | 0.696 | **0.28** | 0.781 | 0.804 | **73** |

**Seçilen: B.** F1 farkı −0.001, gürültünün çok altında. Aynı melanom yakalama oranında B 21 daha az
yanlış alarm veriyor (73'e karşı 94).

## Bilinen sınırlamalar

- Tarama eşiği 102 melanomluk bir val setinde seçildi; testte recall'ın 0.80'den sapması beklenir.
- İki modelin test sonuçlarına tek tek daha önce bakıldı (F1 0.818 ve 0.821). Ortalamalarının test
  sonucuna bakılmadı. İç test seti geliştirme boyunca birçok kez kullanıldı (denetim raporu, K1).
- En iyi checkpoint'ler val F1'e göre seçildiği için yukarıdaki val sayıları bir miktar iyimser.

## Test sonuçları

Henüz ölçülmedi. Bu dosyanın commit'inden sonra, bir kez ölçülüp buraya eklenecek.
