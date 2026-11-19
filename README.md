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

### v2 denemesi (2026-11): v1 kaldı

v2'de önce 5-fold CV ile ölçüm cetveli kuruldu (fold dışı tahminlerde F1 0.785, BA 0.779). Ek veri deneyi (ISIC 2019'un
HAM dışı 14.675 görüntüsü) ilk fold'da kazanç göstermedi ve durduruldu; alana özel ön eğitim (PanDerm) sızıntı riski
doğrulanamadığı için denenmedi. Son olarak CV'nin 5 fold modeli, eşikleri 9029 fold dışı görüntüde seçilerek bir
ensemble'a çevrildi ve bir kez ölçüldü: iç test F1 0.826 / BA 0.817, dış test F1 0.813 / BA 0.794. Önceden yazılan
kurala göre (dört sayının hiçbiri v1'den düşük olmamalı) **v1 kaldı**; farklar güven aralıklarının içinde.
Ayrıntı: [baselines/final_v2/README.md](baselines/final_v2/README.md).

Deney geçmişi (B0 224 px ile F1 0.70'ten bu modele nasıl gelindi): [EXPERIMENTS.md](EXPERIMENTS.md).

## Sınırlamalar

**1. İç test seti tamamen bağımsız değil.**
Projenin ilk günlerinde farklı mimariler, iç test setindeki sonuçlara bakılarak
karşılaştırıldı. Validation disiplini daha sonra kuruldu ve final model yalnızca
validation verisiyle seçildi. Ancak test seti geliştirme sürecinde birkaç kez
görüldüğü için iç test skorları gerçek performansı bir miktar iyimser gösteriyor
olabilir. Bu iyimserliğin ölçülmüş bir göstergesi var: eski ensemble modelin
balanced accuracy'si iç testte 0.809 iken dış testte 0.765'e düştü.

**2. Dış test seti final model için "ikinci bakış".**
ISIC 2018 resmi test seti, final modelden önce eski bir ensemble modeli
değerlendirmek için bir kez kullanıldı. Bu nedenle final modelin dış test sonucu,
hiç kullanılmamış bir sette yapılan ilk ölçüm değil. Yine de final model bu sete
bakılarak seçilmedi, dolayısıyla projedeki en güvenilir sonuç budur. v2 karşılaştırması bu seti
üçüncü kez kullandı (2026-11-19); v2 için de bu sete bakılarak hiçbir şey değiştirilmedi.

**3. Değişikliklerin etkisi ayrı ayrı ölçülmedi (ablation yok).**
En büyük performans artışı üç değişikliğin birlikte uygulanmasıyla elde edildi:
ImageNet-22k ön eğitimi, 384 px çözünürlük ve yeni eğitim tarifi. Bunlardan
yalnızca ikisi ayrıca denendi (224 px'te yeni tarif: val F1 0.744 → 0.766;
eski tarifle 384 px: 0.744 → 0.769); ImageNet-22k'nın tek başına etkisi hiç
ölçülmedi. Bu denemeler de tek eğitimlik olduğu için kazancın ne kadarının
hangisinden geldiği bilinmiyor.

**4. Bölme hasta bazında değil, lezyon bazında yapıldı.**
Aynı lezyonun görüntüleri her zaman aynı sete düşecek şekilde bölündü. Ancak
HAM10000'de hasta kimliği bulunmadığı için, aynı hastanın farklı lezyonları
farklı setlere düşmüş olabilir. Bu da sonuçları hafifçe iyimser yapabilir.

**5. Ölçümler küçük örneklemlere dayanıyor.**
Test setinde bazı sınıfların çok az örneği var (dermatofibroma 14, vasküler
lezyon 13 görüntü). Bu sınıflarda tek bir görüntünün doğru ya da yanlış
sınıflanması balanced accuracy'yi yaklaşık 0.01 değiştiriyor. Bu yüzden
raporlanan değerler %95 güven aralıklarıyla birlikte okunmalı. Örneğin final
modelin balanced accuracy güven aralığı iç testte yaklaşık 9 puan, dış testte
yaklaşık 7 puan genişliğindedir.

**6. Rastgelelik etkisi yalnızca iki eğitimle ölçüldü.**
Aynı tarif yalnızca rastgelelik ayarı (seed) değiştirilerek iki kez eğitildi.
Validation'da macro F1 yaklaşık 0.013, melanom recall'u ise yaklaşık 0.11
değişti. İki eğitim, bu değişkenlik için ancak kaba bir tahmin veriyor.
Özellikle melanom recall'undaki dalgalanma, bu metriğe ilişkin sonuçların
dikkatli yorumlanmasını gerektiriyor.

**7. Bazı kararların test bilgisinden bağımsız olduğu doğrulanamadı.**
ConvNeXt-Small'un elenmesi ve seed 2 koşusuna dayanan kararlar verilirken bu
modellerin test sonuçları diskte hesaplanmış durumdaydı; kararların bu sonuçlar
görülmeden verildiği doğrulanamadı. Ayrıca final tarif (tek model mi, iki modelin
ortalaması mı) validation'da önceden yazılmış bir kuralla seçildi, ama o sırada iki
modelin tek tek iç test sonuçları (macro F1 0.818 ve 0.821) biliniyordu.

**8. Veri setinin kapsamı sınırlı.**
Model yalnızca dermoskopi görüntüleriyle ve büyük ölçüde açık tenli hastalardan,
iki merkezden (Avusturya ve Avustralya) toplanmış verilerle eğitildi. Dış test
seti de dermoskopi görüntülerinden oluşuyor ve ISIC arşivinden geliyor; bu yüzden
dış testteki sonuç, tamamen farklı bir klinikteki performansı garanti etmiyor.
Normal telefon fotoğraflarında, farklı cihazlarda veya koyu ten tiplerinde
performansı düşebilir. Model klinik olarak doğrulanmamıştır ve tanı aracı olarak
kullanılmamalıdır.

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

## Demo (Gradio)

Görüntü yükleme, en olası 3 sınıf, normal ve tarama modu, "emin değilim" uyarısı (val'de seçilen eşik 0.85)
ve Grad-CAM ısı haritası.

```bash
pip install gradio
python app.py            # http://127.0.0.1:7860
SHARE=1 python app.py    # geçici herkese açık link (72 saat, bilgisayar açıkken)
```

Yerelde `models/final_v1/` yoksa (`python -m src.inference.package_model` ile üretilir) model Hugging Face'ten,
sabit `v1` etiketiyle indirilir: [emreozkn/skin-lesion-convnext-v1](https://huggingface.co/emreozkn/skin-lesion-convnext-v1).

## Dizin yapısı

- `src/data/`: manifest oluşturma (tekrar eden görüntüleri temizler), lezyon bazlı bölme, doğrulama,
  `SkinCancerDataset`.
- `src/training/train.py`: iki fazlı transfer learning.
- `src/evaluation/`: metrikler, logit önbellekleme, ensemble, tarif dondurma, iç ve dış test.
- `src/inference/`: model paketleme, `Predictor` (tahmin ve Grad-CAM), paket doğrulaması, "emin değilim" eşiği.
- `app.py`, `assets/examples/`: Gradio demosu ve örnek görüntüler. `hf/`: Hugging Face model kartı ve Space betiği.
- `docs/`: bağımsız denetim raporu (2026-10-26).
- `baselines/final_v1/`: final tarif ve bütün sonuçları (ilk bakılacak yer). Yanındaki iki klasör final modelin
  üyeleri (seed 42, seed 2). `baselines/_archive/`: final modelde kullanılmayan eski deneyler.
- `splits/`: sabit bölme dosyası.
- `external_test/`: dış test setinin etiketleri ve kuralları.
- `kaggle/`: eğitim notebook'u (asıl eğitim ortamı). `notebooks/`: Colab yedeği.
- `data/`: veri (git'te değil).
