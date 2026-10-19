# Proje Durumu

(Projeye her dönüşünde önce burayı oku — sohbet geçmişini hatırlamana gerek yok.)

## Bu ne projesi?
HAM10000 veri setiyle cilt lezyonu sınıflandırma (7 sınıf: akiec, bcc, bkl,
df, mel, nv, vasc). PyTorch, transfer learning. Öğrenme amaçlı bir proje —
bkz. CLAUDE.md.

## Eğitilmiş modeller (test seti, hepsi `baselines/` altında)

| Model | macro_f1 | balanced_accuracy | mel_recall | Not |
|---|---|---|---|---|
| `baseline_v1` (EfficientNet-B0, 224px) | 0.698 | 0.762 | 0.626 | ağırlık git'te, `git checkout baseline-v1` |
| `convnext_tiny_v1` (224px) | 0.774 | 0.798 | **0.766** | melanomu en az kaçıran tek model |
| `efficientnet_v2_s_v1` (384px) | 0.753 | 0.781 | 0.710 | ConvNeXt'ten farklı yerlerde hata yapıyor |
| `swin_v2_t_v1` (256px, Kaggle) | 0.755 | 0.813 | 0.626 | nadir sınıflarda en iyi, melanomda zayıf |
| `convnext_tiny_384_v1` (384px, Kaggle) | 0.771 | 0.784 | 0.664 | val'de çok iyi, testte 224 ile aynı |
| `convnext_tiny_isic2019_v1` (224px + ISIC 2019) | 0.744 | 0.751 | 0.626 | ek veri; ensemble'a katkı yok |
| `convnext_tiny_in22k_384_v1` (22k, 384px, tarif v2) | **0.818** | 0.809 | 0.664 | **en iyi tek model**; mel→nv 20, nv→mel 37 |
| **`ensemble_v1`, normal karar** | **0.783** | 0.809 | 0.692 | 4 model, kalibre, eşit ağırlık |
| **`ensemble_v1`, tarama (mel eşiği 0.30)** | 0.781 | **0.823** | **0.888** | kaçan melanom 5, yanlış alarm 119 |

Her klasörde README (ayrıntılı sonuçlar, güven aralıkları, bulgular) var.
Ağırlıklar: `baseline_v1` git'te; diğerleri (50MB+) yerel diskte + Drive'da
`skin-cancer-detection-runs/archive/<model adı>/` altında.

## Şu an neredeyiz?
Plan "önce genişlik": birkaç farklı mimariyi aynı tarifle eğit, sonra ensemble.
**Genişlik aşaması bitti (2026-10-13).** Dört model var; üç güçlü model
(ConvNeXt, V2-S, Swin) farklı yerlerde güçlü: ConvNeXt melanomu en az kaçıran
(mel→nv 7), Swin nadir sınıflarda en iyi (balanced_accuracy 0.813) ama
melanomda zayıf (mel→nv 17), V2-S en az yanlış alarm veren ikinci model.

Altyapı: Eğitim artık **Kaggle**'da (haftada ~30 saat ücretsiz GPU, arka planda
çalışır). Mac'te `~/.kaggle/kaggle.json` kurulu; Claude notebook'u gönderip
sonucu kendisi çekiyor. Swin Kaggle'da epoch başına ~1.7 dk sürdü (toplam ~40
dk); Colab'da benzer modeller ~5 dk/epoch sürüyordu, Colab'ın 2 CPU'su veri
yüklemede darboğaz olmuş olabilir. Colab notebook'u yedek olarak duruyor.

## Sıradaki adım (tek, somut)
**Ensemble tamamlandı (2026-10-13), `baselines/ensemble_v1/`.** Augmentation v2
(kare-sonra-kırp + ±180°) ConvNeXt'te denendi, iyileşme yok, v1'de kalındı
(`baselines/convnext_tiny_aug_v2/`).

**Yol haritası (2026-10-15, kullanıcıyla kararlaştırıldı, sırayla):**
1. ~~TTA~~ (2026-10-15, **kabul edilmedi**): 4 çevirme bakışı. Tek modellere küçük katkı
   (ConvNeXt +0.008), ama kalibre 4'lü ensemble'da validation F1 0.768 → 0.757 (bootstrap'te
   %94 daha kötü). Ensemble zaten ortalama aldığı için TTA'nın katkısı üst üste binmiyor.
   `cache_predictions.py --tta`, `ensemble_val.py --tta-study`.
2. ~~Metadata füzyonu~~ (2026-10-15, **kabul edilmedi**): eğitim setinde yaş/cinsiyet/yer
   lojistik regresyonu, ensemble ile log-olasılık toplamı (`metadata_fusion.py`). Metadata tek
   başına bilgi taşıyor (val log loss 0.908, bilgisiz tahmin 1.112) ama ensemble'a ekleyince
   hiçbir alfa F1'i artırmadı (alfa ≥ 0.75'te belirgin düşüş). Muhtemel sebep: görüntü modelleri
   yaş/yer ipuçlarını görüntüden zaten alıyor, metadata aynı bilgiyi ikinci kez sayıyor.
   Uyarılar: `dx_type` leakage (teşhis yöntemi = cevap); "trunk" etiketi kaynak izi taşıyor.
3. ~~ConvNeXt @384px~~ (2026-10-15, `baselines/convnext_tiny_384_v1/`): tek model olarak
   validation'da büyük sıçrama (F1 0.744 → 0.769), ama ensemble'a girince gürültüyü aşan bir
   kazanç yok, **ensemble_v1 değişmedi**. Testte tek model F1 0.771 (224: 0.774): validation'daki
   sıçrama tekrarlanmadı.
4. ~~Dış test setini belirle ve kenara ayır~~ (2026-10-16): ISIC 2018 Task 3 resmi test seti,
   `external_test/` (kurallar README'de, `validate_manifest.py` çakışmada hata verir).
5. ~~Ek veri~~ (2026-10-16, **durduruldu**): ISIC 2019 HAM dışı 14885 görüntü sadece train'e
   (`add_isic2019.py`, `baselines/convnext_tiny_isic2019_v1/`). ConvNeXt tek model val'de +0.02,
   ensemble'a katkı yok; testte daha kötü (F1 0.744, 224'ün 0.774'ü), melanom ve akiec recall
   düştü (model "nv"ye kaydı; akiec'in anlamı değişti; farklı hastane). Kullanıcı kararıyla ek
   veri hattı şimdilik durduruldu; README'de ileride denenebilecek düzeltmeler var.
**Tarif v2 denemesi (2026-10-17, Kaggle Version 5) GEÇERSİZ:** finetune lr 1e-4 + warmup + cosine +
label smoothing 0.1. val_f1 0.657 (eski tarif 0.744). Sebep tarif değil, bir kod hatası: PyTorch'un
CrossEntropyLoss(weight, label_smoothing) sınıf ağırlıklarını smoothing payına da uyguluyor; nv görüntüsünde
df olasılığını artırmak loss'u düşürüyordu (model nadir sınıfları her yerde tahmin etti). Düzeltme:
`WeightedLabelSmoothingCE` (ağırlık sadece gerçek sınıfa). **Düzeltilmiş tarif (Version 6,
`baselines/convnext_tiny_recipe2_v1/`): tek model val F1 0.744 → 0.766, kalibrasyon T 1.47 → 0.92,
eğitim sonda oturuyor.** Ensemble'a eski ConvNeXt yerine girince 0.768 → 0.780 (gürültü eşiği altında).
**ConvNeXt-T ImageNet-22k + 384px + tarif v2 (2026-10-19, Kaggle Version 9, P100,
`baselines/convnext_tiny_in22k_384_v1/`): en iyi tek model.** Val F1 0.803 (son 5 epoch ort.
0.788; 224 tarif v2: 0.745), T 0.93. Tek başına 4'lü ensemble'dan iyi (0.794 vs 0.768, %94)
ama melanomda zayıf (mel recall 0.647; aynı recall için daha çok yanlış alarm). Gürültü eşiği
aşılmadı + melanom gerilemesi: **ensemble_v1 değişmedi.**
**Yön değişikliği (2026-10-19, kullanıcı kararı):** önce TEK modeli en iyi hale getir; 5-fold CV
pahalı, ertelendi (gerekirse ensemble yerine tek modelle devam). Final model tarif donunca
train+val ile, cosine planının tamamı (30 epoch) koşulup son epoch alınarak eğitilecek.
**0. adım (bedava, val, aynı model):** test çözünürlüğü 416/448/512 ve flip TTA hiçbirini
iyileştirmedi (416: +0.000, 448: −0.002, 512: −0.004, 384 TTA: −0.014 / %3). Çözünürlük arttıkça
mel recall düştü (0.647 → 0.529). 384'te kalınıyor, TTA yok.
**Sıradaki adım (Kaggle, paralel iki koşu):** A) Tiny seed 2 + EMA gölge kopyası (eğitim gürültüsü
cetveli + EMA etkisi, tek koşuda); B) ConvNeXt-Small 22k 384 + EMA. Sonra multi-crop / final.
**Proje 2026-10-17'de `~/Projeler/skin-cancer-detection`'a taşındı** (Masaüstü iCloud'du, kota dolu,
dosyalar okunamıyordu). Kaggle CLI: `~/.local/kaggle-cli/bin/kaggle`.

6. ~~Dış doğrulama~~ (2026-10-16, ilk ölçüm, `baselines/ensemble_v1/external_isic2018_test/`):
   ensemble_v1, ISIC 2018 resmi test seti (1512). macro_f1 0.749, balanced_accuracy 0.765
   (kendi testimizde 0.809: ~4.5 puan iyimsermiş), mel_recall 0.749; tarama modu mel_recall 0.842.
   Ensemble en iyi tek modelden +0.02 F1. Yarışmanın ek verisiz en iyisi 0.845: fark ~8 puan.
   Bu sete artık bakıldı: yeni tarifler yine validation'da seçilir, burada sadece bir kez ölçülür
   ve "daha önce bakılmış set" diye raporlanır.
7. **Arayüz** (doktor fotoğraf yükler, sonuç alır).

Her yeni tarif validation'da seçilir, test setinde bir kez ölçülür. Yeni bir model
eklenirse: `cache_predictions.py` → `ensemble_val.py` → `ensemble_test.py`.

Kaggle döngüsü (yeni bir model eğitmek gerekirse):
1. [kaggle/skin-cancer-train.ipynb](kaggle/skin-cancer-train.ipynb) ilk hücresinde
   `ARCHITECTURE`'ı değiştir, kodu GitHub'a gönder.
2. `kaggle kernels push -p kaggle` (Mac'te, SSL için `SSL_CERT_FILE` certifi'ye
   ayarlı). İlerleme: https://www.kaggle.com/code/yunusemreozkan/skin-cancer-train
3. Bitince `kaggle kernels output yunusemreozkan/skin-cancer-train -p <klasör>`
   → `baselines/<model adı>/` + Drive `archive/<model adı>/`.

Henüz başlanmayan diğer fikirler: mel-threshold ayarı, TTA (test-time
augmentation), kalibrasyon, metadata füzyonu (yaş/cinsiyet/lezyon konumu —
HAM10000'de var ama kullanılmıyor), dış veri ve ConvNeXt'i 384px'te yeniden
eğitmek (ikisi için aşağıdaki "Sonraya bırakılan konular"a bak).

**Ensemble'dan sonraki iyileştirme sırası:** Önce modelleri düz ortalama ile
birleştir → sonra **entegrasyonun kendisini** ince ayarla (ağırlıklı ortalama,
sınıf-bazlı ağırlık, kalibrasyon — hiçbiri yeniden eğitim istemiyor) →
**ancak hâlâ yetersizse ve zayıf halka netse**, son çare olarak tek bir
mimariyi yeniden eğit.

Dikkat: ensemble ağırlıkları ve hangi modellerin dahil edileceği
**validation** setinde seçilmeli, test seti sadece en sonda bir kez
kullanılmalı. Test setine bakarak seçim yaparsak sonuç iyimser çıkar
(leakage). Seçim `ensemble_val.py`'de (validation), tek ölçüm `ensemble_test.py`'de.

## Sonraya bırakılan konular (2026-10-11'de konuşuldu, karar verilmedi)
- **Kıyas hedefi (ISIC 2018, balanced accuracy):** Ek veri kullanmayanların
  en iyilerinden Zhuang ve ark. 0.845 (senin paylaştığın Sun Yat-sen raporu:
  SENet-154 + PNASNet ensemble). Birinci MetaOptima 0.885, ama ~37.800 ek
  görüntüyle. Gerçekçi hedef (tahmin): ek verisiz 0.82-0.85. Test setleri
  farklı olduğu için kıyas yaklaşık; 1-2 puanlık farklar gürültü olabilir.
- **Daha büyük model / ücretli A100:** Büyüklük ana kaldıraç değil (8 bin
  görüntüde 20-28M parametreli modellerimiz bile ezberliyor). Sıra: ensemble
  → çözünürlük (ConvNeXt-T @384, T4'te ~3 saat) → hâlâ gerekirse büyük model
  (A100'de en verimli aday EfficientNetV2-M @480). 3 saati geçecek ilk
  koşudan önce koda "kaldığı yerden devam" özelliği eklenmeli.
- **Ek veri:** Ek verisiz ensemble bittikten sonra. Önce tek modelle dene:
  ConvNeXt-T + ISIC 2019'un HAM10000 dışındaki ~15 bin görüntüsü, sonuç
  `convnext_tiny_v1` ile aynı val/test'te karşılaştırılır. **Dikkat:** ISIC
  2019 HAM10000'in tamamını içeriyor; tekrarlar temizlenmezse test
  görüntüleri eğitime girer. Yarı yolda karıştırma yok (bir model ek verili,
  diğerleri ek verisiz olmaz).
- **Dış doğrulama (projenin en sonunda, bir kez):** Kullanıcı farklı kaynaktan
  bir test seti istiyor (HAM10000 test seti aynı klinik/cihazlardan). En iyi
  aday ISIC 2018 resmi test seti (~1500 görüntü): 0.845 ile aynı sette kıyas
  sağlar. Etiketler ISIC veri sayfasında olabilir ya da tahminler ISIC canlı
  leaderboard'una yüklenir (o aşamada kontrol et). **Kural:** dış test seti
  ek veriden ÖNCE belirlenip kenara ayrılmalı; ek veri (ör. ISIC 2019) bu setle
  çakışan görüntülerden temizlenmeli. Skorların düşmesi beklenir.
- **Ensemble'da kalibrasyon:** ConvNeXt ve V2-S aşırı özgüvenli (val_loss
  dipten sonra yükseliyor). Olasılık ortalamasında daha emin konuşan model
  kazandığı için, ortalamadan önce her modeli validation setinde temperature
  scaling ile kalibre etmek değerlendirilecek.

## Drive
Google Drive masaüstü uygulaması kurulu. Yol: `~/Library/CloudStorage/
GoogleDrive-emreozkan877@gmail.com/Drive'ım/skin-cancer-detection-runs/`
- `latest/`: Colab her koşuda buraya yazar, **bir önceki koşunun üzerine**.
  Claude sonuçları buradan okur.
- `archive/<model adı>/`: büyük ağırlıkların kalıcı kopyası (adlar
  `baselines/` ile aynı). Bir sonuç kaydedilirken Claude buraya kopyalar.

Notlar: Dosyalar Mac'e bazen ancak Drive uygulaması ya da Finder'da Google
Drive açılınca iner. Komut satırından ilk erişimde "Operation timed out"
alınırsa yine Finder'dan Google Drive'a bir kez tıklamak çözüyor.

## Unutulmaması gereken kurallar
- `deri-kanseri-cnn-uygulamas.ipynb` dosyasına **asla dokunma** (senin ayrı,
  ilgisiz referans notebook'un — stage/commit edilmeyecek)
- Git'e dosya eklerken her zaman dosya adıyla (`git add <dosya>`), asla
  `git add -A` kullanılmıyor (yukarıdaki dosyayı yanlışlıkla eklememek için)
- Colab'da uzun eğitim koşarken sekmeyi arka planda bırakma / bilgisayarı
  uykuya alma — runtime koptu/sıfırlandı, birkaç kez bu yüzden ilerleme kaybettik
- Colab notebook'unu her zaman yukarıdaki GitHub linkinden aç. Drive'a ya da
  Colab'a kaydedilmiş bir kopyanın hücreleri donmuş kalır: `git clone` sayesinde
  kod güncel gelir ama eğitim komutu eski kalabilir. Bir kez yanlış mimariyle
  (sessizce EfficientNet-B0'a düşerek) eğitime başlamıştık
- 50MB üstü ağırlıklar git'e girmiyor (`.gitignore`: `baselines/*/best_model.pt`);
  yerel disk + Drive `archive/` olmak üzere iki kopya tutuluyor
