# Skin Cancer Detection

HAM10000 veri setindeki dermoskopi görüntülerini 7 sınıfa ayıran bir CNN sınıflandırıcısı
(akiec, bcc, bkl, df, mel, nv, vasc). PyTorch, transfer learning. Öğrenme amaçlı bir okul projesi;
eğitim yalnızca ücretsiz Kaggle/Colab GPU'larında yapıldı.

Bu bir araştırma çalışmasıdır, tanı aracı değildir.

- Güncel durum ve sıradaki adım: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- Yol haritası: [ROADMAP.md](ROADMAP.md)
- Bütün deneyler: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem

Cilt lezyonu görüntüsünden lezyon türünü tahmin etmek. Veri çok dengesiz: görüntülerin yaklaşık
%67'si zararsız ben (nv), en nadir iki sınıfta (df, vasc) yüzer civarı görüntü var. Bu yüzden
accuracy yanıltıcıdır: her şeye "nv" diyen bir model %67 accuracy alır. Ana metrikler:

- **macro F1:** her sınıfın F1'inin eşit ağırlıklı ortalaması; model seçimi için.
- **balanced accuracy (BA):** her sınıfın recall'unun ortalaması; ISIC yarışmasıyla kıyas için.
- **melanom recall ve precision:** klinik açıdan en önemli hata, melanomu kaçırmak.

## Veri

- **HAM10000:** 10.015 görüntü, 7.470 lezyon (aynı lezyonun birden çok fotoğrafı olabiliyor).
- **Bölme:** lezyon bazında, etikete göre katmanlı, %80 / 10 / 10 (`random_state=42`):
  train 8037, val 992, test 986. Aynı lezyonun fotoğrafları asla iki farklı sete düşmez; aksi
  halde model test lezyonunu eğitimde görmüş olur (veri sızıntısı). Bölme
  [splits/ham10000_lesion_split_v1.csv](splits/ham10000_lesion_split_v1.csv) dosyasında sabit;
  `validate_manifest.py` üretilen bölme bundan saparsa hata verir.
- **Dış test:** ISIC 2018 Task 3 resmi test seti, 1512 görüntü, eğitimde hiç kullanılmadı.
  Ayrıntı: [external_test/README.md](external_test/README.md).

## Yöntem (final_v1)

- **Model:** ConvNeXt-Tiny, timm `convnext_tiny.fb_in22k_ft_in1k_384` (ImageNet-22k ön eğitimli),
  384 px, drop_path 0.1.
- **Ön işleme:** Gray-World renk düzeltme (farklı kamera ve ışık koşullarındaki renk kaymasını
  azaltır), augmentation v1 (albumentations).
- **Kayıp:** sınıf ağırlıklı label smoothing cross-entropy (smoothing 0.1, ağırlık N / (7 · n_sınıf)).
- **Eğitim, iki faz:**
  1. 5 epoch, backbone donuk, sadece sınıflandırıcı katmanı, lr 1e-4.
  2. 25 epoch, bütün model, lr 1e-4, 2 epoch warmup + cosine. AdamW, batch 32.

  En iyi epoch val macro F1'e göre seçilir.
- **Final model:** aynı tarifle iki koşu (seed 42 ve seed 2). Her model val setinde temperature
  scaling ile kalibre edilir (T 0.93 ve 0.89), olasılıkların eşit ağırlıklı ortalaması alınır.
- **İki çalışma noktası:** normal (en olası sınıf) ve tarama (ortalama melanom olasılığı ≥ 0.28 ise
  melanom). Tarama eşiği val'de melanom recall ≥ 0.80 verecek şekilde seçildi.

Tarif yalnızca val setine bakılarak donduruldu ve test ölçümünden önce commit edildi. Ayrıntı ve
seçim kuralı: [baselines/final_v1/README.md](baselines/final_v1/README.md).

## Sonuçlar

Tek ölçüm, 2026-11-12. Parantez içi %95 bootstrap güven aralığı.

| | macro F1 | balanced accuracy | melanom recall | macro AUC |
|---|---|---|---|---|
| İç test (986 görüntü) | 0.829 (0.789–0.862) | 0.832 (0.786–0.876) | 0.720 | 0.973 |
| **Dış test, ISIC 2018 (1512 görüntü)** | **0.795** (0.761–0.825) | **0.796** (0.759–0.832) | 0.737 | 0.972 |
| Dış test, tarama modu | 0.783 | 0.790 | 0.813 | – |

- **Raporlanması gereken sayı dış testtekidir.** İç test geliştirme boyunca birçok kez görüldü ve
  yaklaşık 3.5 puan iyimser çıkıyor.
- AUC iki sette neredeyse aynı: modelin sıralama kalitesi dış sette düşmüyor. Kayıp karar noktasından
  ve nadir sınıflardan geliyor (dış testte akiec recall 0.67, df 0.73).
- Kıyas: ISIC 2018'de ek veri kullanmayan en iyi yarışma sonucu BA 0.845. Aradaki fark yaklaşık 5 puan.
- Ölçüm gürültüsü: aynı tarifle sadece seed değişince val F1 yaklaşık 0.013 oynuyor. Bundan küçük
  farklar fark sayılmamalı.

Deney geçmişi (B0 224 px ile F1 0.70'ten bu modele nasıl gelindi): [EXPERIMENTS.md](EXPERIMENTS.md).

## Sınırlamalar

<!-- TASLAK: Bu bölümü proje sahibi kendi cümleleriyle yazacak (ROADMAP A5). Aşağıdaki maddeler
yalnızca hatırlatma; her birinin ayrıntısı PROJECT_STATUS.md'deki denetim raporunda (bölüm 7). -->

- Test seti geliştirme sırasında birçok kez görüldü (denetim K1).
- Dış test seti erken kullanıldı; final_v1 ölçümü ikinci bakış (O1).
- Bazı değişiklikler birlikte yapıldı, ayrı etkileri (ablation) ölçülmedi (O7).
- Veri setinde hasta kimliği yok, bölme ancak lezyon bazında yapılabildi (D1).
- Değişkenlik sadece 2 seed ile ölçüldü.

## Tekrar üretme

### Kurulum (Mac, değerlendirme için)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# HAM10000'i data/raw/ altına koy, sonra:
python -m src.data.build_manifest --data-root data --out data/processed/manifest.csv
python -m src.data.split_manifest --manifest data/processed/manifest.csv --out data/processed/manifest.csv
python -m src.data.validate_manifest --manifest data/processed/manifest.csv
```

### Eğitim (Kaggle, GPU)

[kaggle/skin-cancer-train.ipynb](kaggle/skin-cancer-train.ipynb) GitHub'daki kodu klonlayıp şu komutu
çalıştırır (seed 2 koşusu; seed 42 için `--seed 42`, EMA kopyası final modelde kullanılmıyor):

```bash
python -m src.training.train \
  --architecture convnext_tiny_in22k_384 --manifest data/processed/manifest.csv \
  --feature-extract-epochs 5 --finetune-epochs 25 --finetune-lr 1e-4 \
  --scheduler warmup_cosine --warmup-epochs 2 --label-smoothing 0.1 \
  --patience 30 --num-workers 4 --device auto --seed 2 --ema-decay 0.999
```

Bir koşu T4'te yaklaşık 1 saat sürer. Ağırlıklar (yaklaşık 110 MB) git'te değil; MD5'leri
[baselines/final_v1/README.md](baselines/final_v1/README.md)'de.

### Değerlendirme (Mac)

```bash
# Her checkpoint için val ve test logit'lerini bir kez kaydet (predictions.npz)
python -m src.evaluation.cache_predictions --checkpoints <seed42 best_model.pt> <seed2 best_model.pt>
# Tarifi val'de dondur (baselines/final_v1/recipe.json)
python -m src.evaluation.freeze_recipe
# İç test ve dış test, birer kez
python -m src.evaluation.ensemble_test --recipe baselines/final_v1/recipe.json
python -m src.evaluation.external_validation --recipe baselines/final_v1/recipe.json
```

## Dizin yapısı

- `src/data/`: manifest oluşturma (tekrar eden görüntüleri temizler), lezyon bazlı bölme, doğrulama,
  `SkinCancerDataset`.
- `src/training/train.py`: iki fazlı transfer learning.
- `src/evaluation/`: metrikler, logit önbellekleme, ensemble, tarif dondurma, iç ve dış test.
- `baselines/<deney>/`: her eğitilmiş modelin sonuçları ve README'si. `final_v1/` final tarif.
- `splits/`: sabit bölme dosyası.
- `external_test/`: dış test setinin etiketleri ve kuralları.
- `kaggle/`: eğitim notebook'u (asıl eğitim ortamı). `notebooks/`: Colab yedeği.
- `data/`: veri (git'te değil).
