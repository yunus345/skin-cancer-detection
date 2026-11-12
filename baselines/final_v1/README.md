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

## İç test sonuçları (986 görüntü, tek ölçüm, 2026-11-12)

Tarif commit'i `bb0cf19`, ölçüm kodu `1529d82`; ölçüm bunlardan sonra bir kez yapıldı.
Komut: `python -m src.evaluation.ensemble_test --recipe baselines/final_v1/recipe.json`.
Parantez içi %95 bootstrap güven aralığı.

| Metrik | Normal (argmax) | Tarama (mel ≥ 0.28) |
|---|---|---|
| macro F1 | **0.829** (0.789–0.862) | 0.816 (0.773–0.849) |
| balanced accuracy | **0.832** (0.786–0.876) | 0.823 (0.774–0.867) |
| macro precision | 0.834 (0.794–0.871) | 0.826 (0.785–0.862) |
| melanom recall | 0.720 (0.632–0.800) | 0.776 (0.692–0.856) |
| melanom precision | 0.583 (0.503–0.667) | 0.503 (0.428–0.581) |
| mel→nv (kaçan melanom) / nv→mel (yanlış alarm) | 18 / 40 | 13 / 59 |
| accuracy (bilgi amaçlı) | 0.874 | 0.856 |

Eşikten bağımsız: macro AUC **0.973** (0.960–0.983), melanom AUC **0.937** (0.918–0.954).

Sınıf bazında recall (normal): akiec 0.70, bcc 0.92, bkl 0.79, df 0.86, mel 0.72, nv 0.92, vasc 0.92.

### Yorum

- Bitiş eşikleri (macro F1 ≥ 0.80, BA ≥ 0.80) tuttu.
- Tek modellerin test sonuçları F1 0.818 / 0.821 ve BA 0.809 / 0.830 idi. Ortalama F1'de ikisinin de
  biraz üstünde; farklar güven aralığının çok içinde, "ortalama daha iyi" diye iddia edilemez.
- **Tarama modu hedefini tutturamadı:** eşik val'de melanom recall 0.80 verecek şekilde seçilmişti,
  testte 0.776 çıktı (kaçan melanom 18 → 13, yanlış alarm 40 → 59). Sapma güven aralığının içinde;
  102 melanomluk bir val setinde seçilen eşiğin oynaması bekleniyordu. Kural gereği eşik değiştirilmedi.
- En zayıf sınıflar akiec (recall 0.70, 30 görüntü) ve melanom (precision 0.58).
- Bu set geliştirme boyunca birçok kez kullanıldı; sayılar iyimser kabul edilmeli.

## Dış test sonuçları (ISIC 2018 Task 3, 1512 görüntü, ikinci bakış)

**Bu setteki ikinci bakış.** İlk ölçüm 2026-10-16'da ensemble_v1 ile yapılmıştı; o sonuç görüldükten
sonra tarif v2 ve 22k modelleri geliştirildi. Bu sette hiçbir şey ayarlanmadı (T ve eşik HAM10000
val'inden), ama set artık "hiç görülmemiş" değil. Tek ölçüm, 2026-11-12.
Komut: `python -m src.evaluation.external_validation --recipe baselines/final_v1/recipe.json`.
Çıktılar: `external_isic2018_test_ikinci_bakis/`.

| Metrik | Normal (argmax) | Tarama (mel ≥ 0.28) | ensemble_v1 normal (ilk bakış) |
|---|---|---|---|
| macro F1 | **0.795** (0.761–0.825) | 0.783 (0.749–0.813) | 0.749 (0.711–0.780) |
| balanced accuracy | **0.796** (0.759–0.832) | 0.790 (0.754–0.823) | 0.765 (0.728–0.800) |
| macro precision | 0.796 | 0.786 | 0.744 |
| melanom recall | 0.737 (0.671–0.802) | 0.813 (0.754–0.867) | 0.749 |
| melanom precision | 0.640 | 0.531 | 0.520 |
| mel→nv / nv→mel | 29 / 48 | 21 / 85 | – |

Eşikten bağımsız: macro AUC **0.972** (0.962–0.980), melanom AUC **0.940** (0.922–0.957).
Tek modeller (macro F1 / BA): seed 42 0.787 / 0.776, seed 2 0.780 / 0.791.
Sınıf bazında recall (normal): akiec 0.67, bcc 0.83, bkl 0.81, df 0.73, mel 0.74, nv 0.91, vasc 0.89.

### Yorum

- **İç test yaklaşık 3.5 puan iyimser:** macro F1 0.829 → 0.795, BA 0.832 → 0.796. Genelleme için
  raporlanması gereken sayılar dış testtekiler.
- **AUC'ler iki sette neredeyse aynı** (macro 0.973 / 0.972, melanom 0.937 / 0.940). Yani modelin
  sıralama kalitesi dış sette düşmüyor; düşüş karar noktasından (argmax) ve nadir sınıflardan geliyor
  (akiec 0.67, df 0.73).
- **ensemble_v1'e göre:** macro F1 +0.046, BA +0.031, melanom precision 0.52 → 0.64. Güven aralıkları
  uçlarda örtüşüyor ve eşli test yapılmadı; "daha iyi görünüyor" denebilir, kesin değil.
- **Tarama modu burada hedefini tuttu:** melanom recall 0.813 (iç testte 0.776). İki set arasındaki bu
  oynama, eşiğin tek bir sette yargılanamayacağını gösteriyor.
- **Ortalama, iki tek modelden de iyi** (F1 0.795'e karşı 0.787 / 0.780); iç testteki tabloyla tutarlı.
- Yarışmanın ek verisiz en iyisi BA 0.845 (repodaki notlara göre): fark yaklaşık 5 puan.
