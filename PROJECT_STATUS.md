# Proje Durumu

(Projeye her dönüşünde önce burayı oku — sohbet geçmişini hatırlamana gerek yok.)

## Bu ne projesi?
HAM10000 veri setiyle cilt lezyonu sınıflandırma (7 sınıf: akiec, bcc, bkl,
df, mel, nv, vasc). PyTorch, transfer learning. Öğrenme amaçlı bir proje —
bkz. CLAUDE.md.

## Şu anki en iyi, güvenceli sonuç: `convnext_tiny_v1`
- balanced_accuracy: **0.798** (%95 GA 0.746–0.846)
- macro_f1: 0.774, mel_recall: 0.766
- Kod `main`'de (merge edildi), küçük sonuç dosyaları (metrik/grafik) git'te
  kalıcı. Ağırlık dosyası (106MB) GitHub'ın 100MB limitini aştığı için git'e
  giremedi — Drive + yerel diskte duruyor (bkz. README).
- Detaylı rapor: `baselines/convnext_tiny_v1/README.md`

**Önceki referans `baseline-v1`** (EfficientNet-B0, balanced_accuracy=0.762,
tamamen git'te, ağırlık dahil) hâlâ duruyor — `git checkout baseline-v1` ile
erişilebilir, kıyaslama/ensemble için saklanıyor.

## Şu an neredeyiz?
Mimari değişikliğinin tek başına büyük fark yarattığı doğrulandı
(EfficientNet-B0 → ConvNeXt-Tiny: macro_f1 0.698→0.774, gerçek melanomu
"iyi huylu" sanma 22 vakadan 7'ye). Plan "önce genişlik": 2-3 farklı
mimariyi aynı tarifle eğit, sonra ensemble. Sıradaki mimari
**EfficientNetV2-S** için kod hazır (2026-10-11), Colab'da koşulmayı bekliyor.

2026-10-11'de eklenenler:
- Hız: mixed precision (AMP, sadece Colab GPU'sunda), paralel veri yükleme
  (`--num-workers`, varsayılan 2), cudnn.benchmark. Yardımcı süreçlerin birebir
  aynı augmentation'ı üretmesi hatası bulundu (deneyle) ve düzeltildi.
- Kural koda yazıldı: `--image-size` verilmezse her mimari kendi ön-eğitim
  boyutunu kullanıyor (B0/ConvNeXt 224px, V2-S 384px). Çözünürlük checkpoint'e
  kaydediliyor; evaluate/tune_threshold/ensemble onu oradan okuyor.
- `ensemble.py` gerçek checkpoint'lerle ilk kez çalıştı (aynı modelin iki
  kopyası = tek model, doğrulandı); artık her modeli kendi çözünürlüğünde test ediyor.

## Sıradaki adım (tek, somut)
Colab'da EfficientNetV2-S koşusu. Notebook hazır, artık `main`'i klonluyor:
1. Colab → GitHub'dan `notebooks/00_project_setup.ipynb` aç (branch: **main**)
   → T4 GPU
2. Hücreleri sırayla çalıştır. İlk dakikalarda kontrol et:
   `Mixed precision (AMP): on`, `Architecture: efficientnet_v2_s, image size:
   384px`, Faz 2'de ~20 milyon parametre (Faz 1'de 8,967 görmek normal).
3. `CUDA out of memory` alırsan eğitim hücresine `--batch-size 16` ekle.
4. Bitince Drive `latest/`'ten sonucu alıp `baselines/efficientnet_v2_s_v1/`
   olarak kaydederiz, ConvNeXt ile kıyaslarız.

Sonra: Swin V2-Tiny (256px). Son katmanı `head` adında, `set_backbone_trainable`
buna göre güncellenmeli (yoksa Faz 1'de her şey donar, eğitim çöker).

Henüz başlanmayan diğer fikirler: mel-threshold ayarı, TTA (test-time
augmentation), kalibrasyon, metadata füzyonu (yaş/cinsiyet/lezyon konumu —
HAM10000'de var ama kullanılmıyor), dış veri (ISIC 2019/2020), ConvNeXt'i
384px'te yeniden eğitmek (cilalama adayı, ~3 kat süre).

**Ensemble'dan sonraki iyileştirme sırası (netleşti, 2026-10-10):** Önce 3
mimariyi de düz ortalama ile birleştir → sonra **entegrasyonun kendisini**
ince ayarla (ağırlıklı ortalama, sınıf-bazlı ağırlık, kalibrasyon — hiçbiri
yeniden eğitim istemiyor, ucuz/hızlı) → **ancak hâlâ yetersizse ve zayıf
halka netse**, en pahalı/son çare olarak tek bir mimariyi (örn. daha fazla
epoch'la) yeniden eğitmeyi düşün. Entegrasyon ayarı, tek model yeniden
eğitiminden önce gelmeli (daha ucuz, daha hızlı denenebilir).

Dikkat: ensemble ağırlıkları ve hangi modellerin dahil edileceği
**validation** setinde seçilmeli, test seti sadece en sonda bir kez
kullanılmalı. Test setine bakarak seçim yaparsak sonuç iyimser çıkar
(leakage). `ensemble.py` şu an sadece test setinde çalışıyor; ağırlık
ayarı aşamasında validation desteği eklenecek.

## Drive otomasyonu (2026-10-10 kuruldu, ÇALIŞIYOR doğrulandı)
Google Drive masaüstü uygulaması kurulu ve senkronize. Colab'ın Drive'a
kaydettiği sonuçlar (`skin-cancer-detection-runs/latest/`) bu Mac'e otomatik
iniyor — ConvNeXt koşusunda uçtan uca test edildi, hiç manuel zip indirmeye
gerek kalmadı. Yol: `~/Library/CloudStorage/GoogleDrive-
emreozkan877@gmail.com/Drive'ım/skin-cancer-detection-runs/`.

Not: komut satırından (`ls`/Python) ilk erişimde "Operation timed out"
hatası alınabilir — Finder'ı açıp Google Drive'a bir kez tıklamak
(File Provider'ı "uyandırmak") bunu çözüyor.

Basitleştirildi (2026-10-10): artık sadece tek bir `latest/` klasörü var,
tarih-damgalı arşiv kopyası kaldırıldı (kalıcı arşiv zaten git'teki
`baselines/`'ta). Eski karışık klasörler (`20261004_192356`,
`20261010_164549`, `20261010_185539`) silindi, veri kaybı yok (hepsi
önce git/yerel disk ile MD5 doğrulaması yapılarak silindi).

## Unutulmaması gereken kurallar
- `deri-kanseri-cnn-uygulamas.ipynb` dosyasına **asla dokunma** (senin ayrı,
  ilgisiz referans notebook'un — stage/commit edilmeyecek)
- Git'e dosya eklerken her zaman dosya adıyla (`git add <dosya>`), asla
  `git add -A` kullanılmıyor (yukarıdaki dosyayı yanlışlıkla eklememek için)
- Colab'da uzun eğitim koşarken sekmeyi arka planda bırakma / bilgisayarı
  uykuya alma — runtime koptu/sıfırlandı, birkaç kez bu yüzden ilerleme kaybettik
- Colab'da notebook açarken **doğru branch'i** seçtiğinden emin ol (GitHub
  dosya seçicisinde branch adını kontrol et) — bir kez `main`'i açıp yanlış
  mimariyle (sessizce EfficientNet-B0'a düşerek) eğitime başlamıştık
- 100MB üstü checkpoint'ler git'e giremiyor (GitHub limiti) — böyle
  durumlarda sadece metrik/grafik dosyaları git'e girer, ağırlık Drive'da kalır
