# c2_isic2019_pilot (2026-11-18/19) - C2 fold 0 pilotu: REDDEDİLDİ

v2 deneyi C2 (ek veri: ISIC 2019'un HAM dışı görüntüleri), yalnızca fold 0. Kurallar sonuçtan önce `EXPERIMENTS.md` [C2]'de yazıldı.

- Kod: branch `exp/c2-isic2019`, commit `bafa0c9` (main'e birleştirilmedi). Kaggle `skin-cancer-train` Version 13, Tesla T4.
- Tarif, seed (42), epoch (5 + 25), sınıf ağırlığı kuralı (ters frekans) baseline ile aynı. Doğrulama: fold 0 (saf HAM, 1806 görüntü,
  kimlikler baseline fold 0 ile aynı). Son epoch, argmax, fp32 logit.
- Ortam: `pip_freeze.txt` baseline'ınkiyle birebir aynı.
- Süre: ~8 dk/epoch (baseline ~2 dk), fold toplam ~4.2 saat. Ağırlıklar Kaggle çıktısında, git'te değil.

## Eklenen veri (log: `run_fold0.log`)

- ISIC 2019: 25331 görüntü. HAM ile ortak ID 10015 ve SCC 431 çıkarıldı.
- Hash kontrolü (dHash, Hamming ≤ 12 bit): HAM kopyası 3, ek veri içinde tekrar 207 çıkarıldı.
- Eklenen: **14675** (hepsi eğitime): akiec 725, bcc 2760, bkl 1497, df 122, mel 3365, nv 6096, vasc 110.
  Kaynak: BCN 11783, archive 2078, MSK4 814.

## Sonuç (fold 0, son epoch)

| | Baseline fold 0 | C2 fold 0 | Devam eşiği | Tuttu mu |
|---|---|---|---|---|
| macro F1 | 0.776 | 0.766 | ≥ 0.796 | hayır |
| balanced accuracy | 0.749 | 0.730 | ≥ 0.769 | hayır |
| melanom recall | 0.599 | 0.624 | ≥ 0.599 | evet |

Sınıf bazında recall (baseline → C2): akiec 0.746 → 0.593, bcc 0.828 → 0.796, bkl 0.807 → 0.701, df 0.600 → 0.650,
mel 0.599 → 0.624, nv 0.931 → 0.938, vasc 0.731 → 0.808.

**Karar:** pilot kuralı tutmadı (F1 ve BA eşiğin altında, hatta baseline'ın da altında). Kalan fold'lar koşulmadı, deney bitti.
