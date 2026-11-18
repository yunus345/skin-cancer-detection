# Deney Geçmişi (v1)

Bu dosya v1 döneminde (4 Ekim - 12 Kasım 2026) denenen her şeyin kaydıdır. Kaynaklar: `baselines/*/`
altındaki metrik dosyaları, git geçmişi ve 26 Ekim 2026 bağımsız denetim raporu (`docs/AUDIT_2026-10-26.md`,
bölüm 5-6). Geliştirme ajanının önceki envanteri (`AGENT_REPORT.md`) bu dosyaya taşındı. Ajanın
beyanları dosyalarla tek tek karşılaştırıldı; doğrulanamayanlar "kanıt yok" diye işaretli.

Kısaltmalar: F1 = macro F1, BA = balanced accuracy, P = macro precision, melR = melanom recall.
"Val" iç doğrulama seti (992 görüntü), "test" iç test seti (986 görüntü), "dış test" ISIC 2018 Task 3
resmi test seti (1512 görüntü). Eğitimde (Kaggle/Colab, AMP) ölçülen sayılar ile Mac'te (fp32)
`predictions.npz`'den yeniden hesaplanan sayılar arasında val F1'de 0.009'a kadar fark çıkıyor; tek
sayı kaynağı Mac değerleridir.

## Nasıl okunmalı

- **Ölçüm gürültüsü:** Aynı tarifle sadece seed değişince val F1 yaklaşık 0.013, val melanom recall
  yaklaşık 0.11 oynuyor. Bundan küçük farklar fark sayılmaz. Test setinde df 14 ve vasc 13 görüntü
  olduğundan tek bir görüntü BA'yı yaklaşık 0.010 oynatıyor; test BA'nın %95 güven aralığı yaklaşık ±0.045.
- **İç test seti güvenilir bir "hiç görülmemiş" set değil.** Geliştirme boyunca birçok kez bakıldı
  (denetim bayrağı K1). 9-13 Ekim arasında mimari karşılaştırmaları ve yön kararları test sonuçlarıyla
  verildi. Bu yüzden iç test sayıları iyimser kabul edilmeli; dış testte aynı model yaklaşık 3.5 puan geriden geliyor.
- **Dış test iki kez kullanıldı.** İlk bakış 16 Ekim'de ensemble_v1 ile, ikinci bakış 12 Kasım'da final_v1 ile.
- **Kanıt sütunu:** "dosya var" = sayı repodaki dosyadan doğrulandı. "kanıt yok" = sayı yalnızca ajanın
  beyanında veya eski durum dosyasında geçiyor, çıktısı saklanmamış.

## 1. Erken dönem (4-8 Ekim): çıktıları saklanmamış

Bu dönemin sayıları ajanın beyanıdır. Çıktı dosyası yok; 9 Ekim'de baseline_v1 ile ilk kalıcı kayıt başladı.

| Deney | Ne denendi | Beyan edilen sonuç | Kanıt |
|---|---|---|---|
| İlk yerel model | EfficientNet-B0 224px, iki fazlı transfer, sınıf ağırlıklı CE, 15 epoch | Val F1 0.572 / BA 0.709; test F1 0.633 / BA 0.725, melR 0.70 | kanıt yok |
| Cosine scheduler | Her faz için ayrı cosine | Val F1 0.513 (referans 0.572). Muhtemel sebep: 5 epoch'luk fazda lr erkenden sıfıra iniyor | kanıt yok |
| Melanom eşiği 0.256 | Val'de "melR ≥ 0.85" hedefiyle seçildi | Testte melR 0.70 → 0.81, melanom precision 0.34 → 0.27, yanlış alarm 123 → 188. Sadece recall'a bakan hedef precision'ı çok düşürdü | kanıt yok |
| Colab uzun eğitim | 20 finetune epoch | Val F1 0.709 / BA 0.780; test F1 0.722 / BA 0.785. Görüntüler Linux'ta iki kez sayıldığı için fiilen ~40 epoch | kanıt yok |
| Focal loss | Sınıf ağırlığıyla birlikte | Sonuç kayıtlı değil. İlk epoch'larda aşırı telafi (accuracy 0.17-0.33), CE'ye dönüldü | sadece kod (`0adf7cf`, `70815f2`) |
| Tekrarlanan görüntü düzeltmesi | Kaggle verisindeki çift klasör yüzünden aynı görüntü iki kez sayılıyordu; `image_id` tekrarları atıldı | Oranlar etkilenmedi | kod var (`build_manifest.py`) |

8 Ekim'de üç değişiklik aynı dakikada commit edildi (Gray-World, checkpoint ensemble, CE'ye dönüş). Checkpoint
ensemble betiği (`f5cabe8`) eklendi ve sonra silindi; ajanın beyanında yoktu.

## 2. Mimari ve ensemble (9-16 Ekim)

| # | Deney | Ayarlar | Val F1 / BA | Test F1 / BA / melR | Karar |
|---|---|---|---|---|---|
| 6 | baseline_v1 | B0 224, augmentation v1 + Gray-World + ağırlıklı CE, 5+20 epoch | 0.635 / 0.707 | 0.698 / 0.762 / 0.626 | referans. Augmentation ve Gray-World'ün ayrı etkisi ölçülmedi |
| 7 | convnext_tiny_v1 | ConvNeXt-T ImageNet-1k, 224 | 0.744 / 0.761 | 0.774 / 0.798 / 0.766 | mimari değişikliği en büyük sıçrama |
| 8 | efficientnet_v2_s_v1 | V2-S 384 | 0.749 / 0.776 | 0.753 / 0.781 / 0.710 | ensemble üyesi |
| 9 | swin_v2_t_v1 | Swin V2-T 256 | 0.732 / 0.779 | 0.755 / 0.813 / 0.626 | ensemble üyesi (nadir sınıflarda iyi, melanomda zayıf) |
| 10 | ensemble_v1, normal | 4 model, val'de temperature scaling (T: B0 0.88, ConvNeXt 1.47, V2-S 1.11, Swin 1.41), eşit ağırlık | 0.768 / 0.793, melR 0.814 | 0.783 / 0.809 / 0.692 | kabul |
| 11 | ensemble_v1, tarama | Melanom olasılığı ≥ 0.30 ise melanom | F1 0.746, melR 0.882 | 0.781 / 0.823 / 0.888, kaçan melanom 5, yanlış alarm 119 | eşik kullanıcıyla seçildi |
| 12 | Augmentation v2 | Kareye sıkıştır + kırp, ±180° döndürme, ConvNeXt 224 | 0.731 / 0.774 | bakılmadı | ret (iki değişiklik birlikte denendi, ayrı etkileri bilinmiyor) |
| 13 | Flip TTA | 4 çevirme bakışı | ensemble 0.768 → 0.757 | - | ret. Aynı TTA 22k modelinde 0.794 → 0.780 (bu ikincisi için kanıt yok) |
| 14 | Metadata füzyonu | Yaş, cinsiyet, konum, lojistik regresyon + ensemble | hiçbir ağırlık F1'i artırmadı | - | ret. `dx_type` sızıntı riski nedeniyle dışarıda bırakıldı |
| 15 | convnext_tiny_384_v1 | ConvNeXt-T 1k, 224 → 384 | 0.769 / 0.812 | 0.771 / 0.784 / 0.664 | ret. Val sıçraması testte tekrarlanmadı |
| 16 | convnext_tiny_isic2019_v1 | + 14.885 ISIC 2019 görüntüsü (sadece train; SCC hariç, AK → akiec) | 0.760 / 0.759 | 0.744 / 0.751 / 0.626 | hat durduruldu (aşağıya bak) |
| 17 | Dış test (ilk bakış) | ensemble_v1, ISIC 2018 test seti | - | normal 0.749 / 0.765 / melR 0.749; tarama 0.729 / 0.757 / 0.842 | iç test yaklaşık 4.5 puan iyimserdi |

Ek veri denemesinin (#16) beyan edilen sorunları: sınıf ağırlıkları melanomun ağırlığını düşürdü ve model
nv'ye kaydı; akiec'in anlamı değişti (Bowen yok); görüntüler farklı hastaneden (Barselona); değerlendirme
sadece iç testte yapıldı. Denemeyi doğru şekilde tekrarlamak v2 planında (ROADMAP C2).

## 3. Tarif v2 ve 22k modeli (17-20 Ekim)

| # | Deney | Ayarlar | Val F1 / BA | Test F1 / BA / melR | Karar |
|---|---|---|---|---|---|
| 18 | Tarif v2, ilk koşu | PyTorch `CrossEntropyLoss(weight, label_smoothing)` | F1 0.657 | - | geçersiz. Kod hatası: sınıf ağırlığı smoothing payına da uygulanıyordu, model nadir sınıfları her yerde tahmin etti. `WeightedLabelSmoothingCE` ile düzeltildi (metrik dosyası yok) |
| 19 | convnext_tiny_recipe2_v1 | lr 1e-4, 2 epoch warmup + cosine, label smoothing 0.1, 25 epoch, 224 | 0.766 / 0.756 | bakılmadı | kabul. Son 5 epoch ortalaması 0.733 → 0.745, T 1.47 → 0.92 |
| 20 | **convnext_tiny_in22k_384_v1** | timm `convnext_tiny.fb_in22k_ft_in1k_384`, drop_path 0.1, seed 42 | 0.803 / 0.798, son 5 ort. 0.788 (Mac F1 0.794) | 0.818 / 0.809 / 0.664 | kabul. Testte ensemble_v1'i geçen ilk tek model; melanomda zayıf |
| 21 | Test çözünürlüğü | Aynı model 416 / 448 / 512 | beyan: F1 0.795 / 0.792 / 0.790 (384: 0.794), melR düşüyor | - | ret (kanıt yok: dosya ve kod yolu yok) |
| 22 | Seed 2 | #20 ile aynı, sadece seed | 0.799 / 0.812, son 5 ort. 0.775 (Mac 0.807) | 0.821 / 0.830 / 0.729 | ölçüm cetveli: seed değişince F1 ~0.013, val melR ~0.11 |
| 23 | EMA (decay 0.999) | Ağırlıkların hareketli ortalaması, gölge kopya (eğitimi etkilemez) | Tiny 0.784 / 0.800; Small 0.794 / 0.799 | ölçülmedi (raporlanmadı) | tarife eklendi: F1'i artırmıyor, NLL ve melanom ayırma biraz iyi, zıplamaları yok ediyor |
| 24 | ConvNeXt-Small | `convnext_small.fb_in22k_ft_in1k_384`, drop_path 0.2, seed 42 | 0.801 / 0.802, son 5 ort. 0.780 | 0.816 / 0.829 / 0.682 | ret: iki Tiny seed'inin arasında, maliyet 1.6 kat (3.33 dk/epoch'a karşı 2.14) |
| 26 | 2 seed ortalaması | #20 + #22, val'de T kalibrasyonu | Mac 0.793 / 0.795, melR 0.696 | - | melR 0.80 için 73 yanlış alarm (ensemble_v1: 70, tek model: 94) → final_v1'in temeli |

Ensemble'a ek denemeler (hepsi val'de, gürültü eşiğini aşmadı, ensemble_v1 değişmedi): tarif v2 ConvNeXt'i
eski ConvNeXt yerine koymak (val F1 0.780) veya 5. üye yapmak (0.762); 22k-384 modelini eski ConvNeXt yerine
(0.783) veya 5. üye olarak (0.780) eklemek; üye çıkarma ve ağırlık çalışması (eşit ağırlığı geçen yok,
B0'ı çıkarmak F1 -0.015).

**Test zamanlaması uyarısı (#22 ve #24).** Ajan "Seed 2 ve Small'ın test sonuçlarını karardan sonra
gördüm" dedi ve bu iki koşunun README'lerinde "Test sonuçlarına bakılmadı" yazıyordu. Denetimde test
dosyalarının karar commit'inden önce diskte olduğu görüldü (Small: 20:35'ten beri, karar 21:19; seed 2:
21:12, karar yaklaşık 7 dakika sonra). Sonuçlara kararlardan önce bakılıp bakılmadığı doğrulanamıyor; bu
yüzden bu iki satır "karar sonrası görüldü" diye değil, "zamanlama doğrulanamadı" diye okunmalı.
Kararlar yalnızca val ölçütlerine dayandırılmıştı (Small: maliyet ve gürültü içinde olması; EMA: NLL).

## 4. Kapanış (12 Kasım): final_v1

Ayrıntılar `baselines/final_v1/README.md`'de. Tarif yalnızca val'e bakılarak donduruldu (commit `bb0cf19`),
sonra tek ölçüm alındı.

| Set | macro F1 | BA | melR | Not |
|---|---|---|---|---|
| İç test (986) | 0.829 | 0.832 | 0.720 | tarama modunda (eşik 0.28) 0.816 / 0.823 / 0.776 |
| Dış test, ikinci bakış (1512) | 0.795 | 0.796 | 0.737 | tarama modunda 0.783 / 0.790 / 0.813; ensemble_v1 aynı sette 0.749 / 0.765 |

AUC iki sette neredeyse aynı (macro 0.973 / 0.972, melanom 0.937 / 0.940): sıralama kalitesi dış sette
düşmüyor, kayıp karar noktasından ve nadir sınıflardan geliyor.

## 5. Kalıcı kararlar (v1)

- Ana metrik macro F1; literatürle kıyas için BA; klinik güvenlik için melanom recall / precision ve
  mel→nv sayısı. Accuracy sadece bilgi amaçlı.
- Karar yalnızca val'de (ve v2'de CV'de) verilir; test tarif başına bir kez ölçülür.
- Tek epoch yerine son 5 epoch ortalamasına bakılır; 0.013'ten küçük farklara güvenilmez.
- Tarif: 384 px, TTA yok, augmentation v1, EMA tarifte, ConvNeXt-Small ve üstü kullanılmıyor.
- Train+val ile yeniden eğitim planı iptal edildi (ROADMAP A0); final model iki seed'in ortalaması.

## 6. Konuşulup denenmeyenler

v2 planına (ROADMAP C) girenler: 5-fold CV, ek veriyi doğru şekilde eklemek, daha güçlü ön eğitim.
Girmeyenler: multi-crop, 448'de yeniden eğitim, tarif v3 (head lr, weight decay, layer decay), mixup /
CutMix, melanom sınıf ağırlığını artırmak, logit adjustment, karekök sınıf ağırlığı / oversampling,
kademeli unfreezing, V2-S ve Swin'i yeni tarifle eğitip ensemble_v2 kurmak, EfficientNetV2-M @480.
Bunlar bilinçli olarak yapılmadı; yeni fikirler `IDEAS.md`'ye yazılır, o an uygulanmaz.

# v2 deneyleri

Ölçüm cetveli: `baselines/cv5_baseline/` (5-fold CV, birleşik OOF macro F1 0.785, BA 0.779, melanom recall 0.662).
Kilitli hedef (2026-11-18): F1 ≥ 0.8048 ve BA ≥ 0.7993, melanom recall ≥ 0.662. Bütçe: baseline dışında en fazla 3 deney.

### [C2] Ek veri: ISIC 2019'un HAM dışı görüntüleri - 2026-11-18 (REDDEDİLDİ, fold 0 pilotunda)
- **Değişen tek şey:** eğitim fold'larına ISIC 2019'un HAM10000 dışındaki görüntüleri eklenir. Doğrulama fold'ları saf HAM
  kalır; tarif, seed (42), epoch sayısı (5 + 25) ve sınıf ağırlığı **kuralı** baseline ile aynı.
- **Veri kuralları (önceden):**
  - SCC ve UNK çıkarılır. AK → akiec (HAM'in akiec'i AK + Bowen; ISIC 2019'da Bowen SCC içinde: anlam kısmen kayar, bilinçli kabul).
  - HAM10000 görüntüleri ID ile, aynı görüntünün farklı ID'li kopyaları görüntü hash'i ile çıkarılır. ISIC 2018 dış testiyle örtüşme
    varsa betik durur.
  - Ek görüntüler yalnızca eğitime girer; kod eklenen sayıyı yazdırır ve 0 ise durur.
- **Sınıf ağırlığı:** ters frekans (baseline kuralı), eğitim setinin gerçek sayılarıyla yeniden hesaplanır. Böylece her sınıfın kayıptaki
  toplam payı baseline'daki gibi eşit kalır. ROADMAP'teki "karekök / üst sınırlı" seçenekleri reddedildi: karekök nv'nin payını ~%33'e
  çıkarıp df ve vasc'ı ~%5'e düşürür (nadir sınıflar ve BA zarar görür); üst sınır yeni bir ayar düğmesi getirir.
- **Hipotez:** zayıf sınıflara (mel, bcc, akiec) gelen ek örnekler OOF F1 ve BA'yı artırır. Risk: alan kayması (farklı hastane, cihaz,
  siyah kenarlar, kare görüntüler); v1'deki denemede (`baselines/_archive/convnext_tiny_isic2019_v1/`) sonuç kötüleşmişti.
- **Pilot ve durma kuralı (kullanıcıyla kararlaştırıldı, sonuçtan önce):** önce yalnızca fold 0 eğitilir (~3 saat GPU). Devam için üçü
  birden gerekir, baseline fold 0'a göre: macro F1 ≥ 0.796 (+0.02), BA ≥ 0.769 (+0.02), melanom recall ≥ 0.599. Biri tutmazsa deney
  burada biter ("pilotta kayda değer kazanç yok") ve C3'e geçilir. Bilinen bedel: gerçek etkisi ~+0.01 olan bir değişiklik elenebilir;
  böyle bir etki hedefi zaten tutturmaz.
- **Başarı kriteri (tam CV koşulursa):** 5 fold'un en az 4'ünde baseline'dan iyi ve ortalama F1 farkı > 0.01; melanom recall baseline
  altına düşmez. Kilitli hedef ayrıca kontrol edilir.
- **Değerlendirme:** CV OOF (HAM fold'ları). İç ve dış teste bakılmaz.
- **Kod:** `exp/c2-isic2019` branch'i.
- Sonuç (2026-11-19, Kaggle Version 13, `baselines/c2_isic2019_pilot/`): 14675 görüntü eklendi (HAM kopyası hash ile 3, ek veri
  içi tekrar 207 çıkarıldı). Fold 0, son epoch: **macro F1 0.766** (baseline 0.776, eşik 0.796), **BA 0.730** (baseline 0.749,
  eşik 0.769), melanom recall 0.624 (baseline 0.599, eşik 0.599). F1 ve BA eşiği tutmadı; ikisi baseline'ın da altında.
  Sınıf bazında recall: akiec 0.746 → 0.593, bkl 0.807 → 0.701 en çok düşenler; mel, vasc, df hafif arttı. Epoch başına
  süre ~4 kat (8 dk). Epoch bazında karışık (F1 30 epoch'un 14'ünde baseline'ın altında), ama son 4 epoch'un hepsinde altında.
- Yorum (kanıt değil, tek fold): akiec düşüşü önceden yazılan etiket kaymasıyla uyumlu (ISIC 2019'da Bowen SCC içinde, HAM'de
  akiec içinde). bkl düşüşü alan kaymasıyla uyumlu olabilir. v1'deki kötüleşme de doğru kurulumla tekrarlandı.
- Karar: **deney bitti, kalan fold'lar koşulmadı** (pilot kuralı). `exp/c2-isic2019` kodu main'e birleştirilmez. Bütçeden 1 deney
  harcandı (kalan: C3 + bir serbest hak).

## Deney kayıt şablonu (v2 için)

```
### [Kod] Deney adı - tarih
- Değişen tek şey:
- Hipotez:
- Başarı kriteri (önceden):
- Değerlendirme: CV / dış validation / test (hangisi, kaçıncı kullanım)
- Sonuç:
- Karar: kabul / ret / belirsiz
- Kanıt dosyaları:
```
