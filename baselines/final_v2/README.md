# final_v2 - kurallar (test ölçümünden önce yazıldı, 2026-11-19)

> Bu dosya seçim betiği çalıştırılmadan ve test ölçülmeden önce commit edildi. Bundan sonra kurallar değişmez.

## Neden böyle bir v2

v2 deneyleri kazanan çıkarmadı (C2 reddedildi, C3 atlandı). ROADMAP C4: "kazanan yoksa v1 tarifi kalır". Tarif (ConvNeXt-Tiny
in22k, 384 px, 5 + 25 epoch) değişmez. Değişen yalnızca **hangi modellerin** kullanıldığı ve **karar katmanının hangi veride**
seçildiği:

| | final_v1 | final_v2 |
|---|---|---|
| Modeller | 2 seed (42, 2), train setinde, en iyi epoch | baseline CV'nin 5 fold modeli (seed 42, son epoch); her biri train+val'in ~%80'inde |
| T ve eşik seçimi | val, 992 görüntü (102 melanom) | birleşik OOF, 9029 görüntü (~1000 melanom) |
| Yeni eğitim | - | yok |

Beklenen kazanç: daha fazla model (ortalama genelde biraz daha kararlı), ~9 kat büyük seçim seti (eşik daha az oynar).
Bedel: arayüzde tahmin 2.5 kat yavaş (5 model).

Not: kilitli hedef (OOF tek model F1 ≥ 0.8048) bu yolla tutturulmaz; v2 bitişi "bütçe/deney bitti" yolundan gelir.

## Tarif (önceden yazıldı)

1. **Modeller:** `models/cv5_baseline/fold{0-4}/last_model.pt` (Kaggle Version 11 ve 12; git'te değil, MD5'ler `oof_mac/` içinde).
2. **Tek sayı kaynağı:** Mac, fp32. OOF logit'leri Mac'te yeniden hesaplandı (`python -m src.evaluation.recompute_oof` →
   `oof_mac/`), çünkü Kaggle OOF'u ile Mac arasında 0.07-0.19 logit farkı var.
3. **Kalibrasyon:** her fold modeli için T, kendi OOF'unda (o modelin eğitimde görmediği görüntüler) v1'deki `fit_temperature`
   ile bulunur (NLL, 0.50-5.00, 0.01 adım).
4. **Birleştirme:** 5 kalibre olasılığın eşit ağırlıklı ortalaması.
5. **Normal mod:** argmax.
6. **Tarama modu:** melanom olasılığı ≥ t ise melanom. t = birleşik OOF'ta (kalibre, tek model olasılıkları) melanom recall
   ≥ 0.80 sağlayan **en yüksek** eşik, 0.01 adımlı (v1 kuralının aynısı).
7. **"Emin değilim" eşiği τ:** v1 kuralının aynısı, birleşik OOF'ta: τ 0.30-0.95 (0.05 adım), emin olunan tahminlerde doğruluk
   ≥ 0.95 sağlayan **en küçük** τ.

## Bilinen sınırlamalar

- OOF'ta her görüntüyü tek model tahmin etti; T, t ve τ tek model olasılıklarında seçilir, testte 5 modelin ortalamasına
  uygulanır. Ortalama olasılıklar daha yumuşak olur; tarama modunda melanom recall'ın 0.80'den sapması beklenir.
- Son epoch alındı (en iyi epoch seçilmedi): v1 üyelerinden farklı olarak val'e göre iyimserlik yok.
- İç test geliştirme boyunca birçok kez kullanıldı; dış test (ISIC 2018) bu ölçümle **üçüncü kez** görülür.

## Ölçüm (C5, bir kez)

- İç test (986) ve ISIC 2018 dış test, normal ve tarama modu, %95 bootstrap güven aralığı (v1 ile aynı betikler).
- **v1'in yerine geçme kuralı (kullanıcıyla kararlaştırıldı, 2026-11-19):** normal modda (argmax) **iç testte ve dış testte**
  macro F1 ve balanced accuracy'nin **dördü de** v1'den düşük değilse (eşit ya da yüksek) v2 arayüzdeki modelin yerine geçer
  (HF Hub'da `v2` etiketi). Biri bile düşükse v1 kalır ve bu bir sonuç olarak raporlanır. v1: iç test F1 0.829 / BA 0.832,
  dış test F1 0.795 / BA 0.796 (`baselines/final_v1/`). Karşılaştırma 3 ondalığa yuvarlanmış değerlerle yapılır.
  Fark payı yok: 986 görüntüde güven aralığı yaklaşık ±0.04, pay konursa kural pratikte hiç tutmaz; eşitlik v2 lehine
  sayılır çünkü v2'nin eşikleri ~9 kat büyük sette seçildi.

## Seçim sonucu (birleşik OOF, Mac, `recipe.json`, commit `d0c13f2`, testten önce)

- T: fold0-4 = 0.96 / 0.94 / 0.96 / 0.94 / 0.93.
- Tarama eşiği **t = 0.07** (OOF melanom recall 0.805; yanlış melanom alarmı 368 → 852). v1'de 0.28 idi: tek model
  olasılıklarında recall 0.80 için eşik çok aşağı inmek zorunda.
- τ = 0.90: vakaların %24'ü doktora; emin olunanlarda doğruluk 0.951; emin iken kaçan melanom 141 (~%13; v1 val'de 8/102).
- Mac OOF birleşik: F1 0.785, BA 0.779, melanom recall 0.665 (Kaggle OOF ile aynı; fold başına 7-10 tahmin farklı).

## Test sonucu (tek ölçüm, 2026-11-19, `python -m src.evaluation.measure_final_v2`)

| Normal mod | v2 macro F1 | v1 macro F1 | v2 BA | v1 BA | v2 mel recall | v1 mel recall |
|---|---|---|---|---|---|---|
| İç test (986) | 0.826 (0.786–0.858) | 0.829 | 0.817 (0.771–0.861) | 0.832 | 0.692 | 0.720 |
| Dış test, ISIC 2018 (1512), üçüncü bakış | **0.813** | 0.795 | 0.794 | 0.796 | 0.719 | 0.737 |

| Tarama modu | v2 (t = 0.07) mel recall / precision | v1 (t = 0.28) mel recall / precision |
|---|---|---|
| İç test | 0.841 / 0.439 | 0.776 / 0.503 |
| Dış test | 0.871 / 0.431 | 0.813 / 0.531 |

Eşikten bağımsız: macro AUC iç 0.971 / dış 0.974 (v1 0.973 / 0.972), melanom AUC iç 0.946 / dış 0.941 (v1 0.937 / 0.940).

**Karar (önceden yazılan kural): v1 kalır.** Dört koşuldan yalnızca biri tuttu (dış test F1 +0.018); iç test F1 −0.003,
iç test BA −0.015, dış test BA −0.002 (`comparison_v1.json`).

### Yorum

- Farkların hepsi güven aralıklarının çok içinde: v2 ile v1 bu ölçümle ayırt edilemiyor. Kural eşitlikte bile v2'yi
  seçecek kadar gevşekti, yine de tutmadı.
- Tarama modunda beklenen oldu: tek model OOF'unda seçilen eşik 5 modelin ortalamasına uygulanınca melanom recall
  0.80'in üstüne çıktı (0.84 / 0.87), bedeli daha çok yanlış alarm (precision ~0.44).
- AUC'ler v1 ile aynı düzeyde: 5 fold modelinin ortalaması sıralama kalitesini değiştirmedi.
