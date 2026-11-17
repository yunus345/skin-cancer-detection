# ensemble_v1 (2026-10-13)

Dört modelin birleşimi: `baseline_v1` (B0), `convnext_tiny_v1`, `efficientnet_v2_s_v1`,
`swin_v2_t_v1`. Yeniden eğitim yok; her modelin kayıtlı logit'leri (`predictions.npz`)
kullanıldı.

## Tarif (tamamen validation setinde seçildi, test setine bakmadan)

1. **Kombinasyon:** dört model birden. Validation'da en yüksek F1'i B0+V2-S+Swin verdi
   ama fark gürültü payı içindeydi (kazananın laneti riski); dört model klinik
   metriklerde en iyisiydi ve hiçbir modeli tesadüfi bir farkla dışarıda bırakmıyor.
2. **Kalibrasyon:** her model validation'da temperature scaling ile.
   T: B0 0.88, ConvNeXt 1.47, V2-S 1.11, Swin 1.41 (ConvNeXt ve Swin aşırı özgüvenliydi).
3. **Ağırlık:** eşit. Model çıkarma / B0 ağırlığı denemelerinde hiçbir seçenek eşit
   ağırlığı belirgin şekilde geçmedi. Validation'da en önemli üye V2-S, B0 da
   gerçek katkı yapıyor (çıkarınca F1 -0.015); ConvNeXt melanom recall'ını taşıyor.
4. **İki çalışma noktası** (kullanıcıyla birlikte seçildi): normal karar (argmax) ve
   "tarama modu" (ortalama mel olasılığı >= 0.30 ise mel).

Kod: `src/evaluation/ensemble_val.py` (seçim, validation), `src/evaluation/ensemble_test.py`
(test setinde tek ölçüm).

## Test seti sonuçları (tek ölçüm)

| Katman | Metrik | ConvNeXt (en iyi tek) | V2-S (val'de en iyi tek) | **Ensemble, normal** | **Ensemble, tarama 0.30** |
|---|---|---|---|---|---|
| 1. Headline | macro_f1 | 0.774 | 0.753 | **0.783** (0.736–0.823) | 0.781 (0.731–0.820) |
| 2. Dış karşılaştırma | balanced_accuracy | 0.798 | 0.781 | 0.809 (0.762–0.855) | **0.823** (0.774–0.866) |
| 2. Dış karşılaştırma | macro_precision | 0.769 | 0.739 | 0.775 | **0.778** |
| 3. Klinik | mel_recall | 0.766 | 0.710 | 0.692 | **0.888** (0.825–0.940) |
| 3. Klinik | mel_precision | 0.436 | 0.455 | **0.474** | 0.399 |
| | mel→nv (kaçan melanom) | 7 | 19 | 20 | **5** |
| | nv→mel (yanlış melanom alarmı) | 88 | 66 | **61** | 119 |

## Bulgular

- **Ensemble şimdiye kadarki en yüksek macro F1'i verdi (0.783).** Test setinde en iyi
  tek model olan ConvNeXt'e göre fark küçük (+0.009, güven aralıkları örtüşüyor).
  Validation'a göre seçseydik tek model olarak V2-S'i seçerdik; ona göre kazanç +0.030.
- **Normal kararda melanom zayıf:** recall 0.692; validation'daki 0.814'ten belirgin
  şekilde düşük ve ConvNeXt'in tek başına 0.766'sının altında (20 melanom nv sanıldı).
  Validation ile test arasındaki bu fark küçük setlerin gürültüsünü gösteriyor.
- **Tarama modu testte neredeyse bedava çıktı:** F1 sadece -0.002 düştü, melanom recall'ı
  0.888'e çıktı, kaçan melanom 20'den 5'e indi. Bedeli 58 ek yanlış melanom alarmı
  (nv→mel 61 → 119). Klinik açıdan en dengeli sonuç bu.
- Balanced accuracy (0.809 / 0.823) ISIC 2018'in ek verisiz en iyilerine (0.845)
  yaklaşıyor; test setleri farklı olduğu için kıyas yaklaşık.

Not: Test setine bu tarif için bir kez bakıldı. Bu sonuçlara göre tarifi değiştirmek
(ör. eşiği testte yeniden ayarlamak) test setini validation setine çevirir.
