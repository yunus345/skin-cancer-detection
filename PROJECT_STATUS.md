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
Mimari değişikliğinin tek başına büyük fark yarattığı doğrulandı: sadece
EfficientNet-B0 → ConvNeXt-Tiny değişikliğiyle (her şey eşit kalarak)
macro_f1 0.698→0.774, mel_recall 0.626→0.766 (gerçek melanomayı "iyi huylu"
sanma hatası 22 vakadan 7'ye düştü). Proje güvencede, acil yapılacak bir
şey yok.

## Sıradaki adım (tek, somut)
Sırada: **Swin V2** ve/veya **EfficientNetV2-S** mimarilerini aynı şekilde
eğitip `convnext_tiny_v1` ile kıyaslamak — amaç tek kazanan seçmek değil,
makul çıkan hepsini **ensemble**'da birleştirmek (biri belirgin kötü
çıkarsa o dışarıda bırakılır).

Bu sıradaki koşularda, artık "saf mimari etkisi" referansımız (bu ConvNeXt
sonucu) elimizde olduğu için, ek olarak paketleyebiliriz:
- Görüntü boyutunu artırmak (224px → 320-384px, mimariye göre)
- Hız optimizasyonları: mixed precision (AMP), DataLoader `num_workers`,
  daha büyük batch size — sonucu değiştirmiyor, sadece Colab süresini kısaltır
  (ConvNeXt koşusu ~2 saat sürdü, T4'te daha büyük modellerle daha da uzayabilir)

Henüz kod tarafında hazır değil — bir sonraki oturumda `build_model`'e
Swin V2 / EfficientNetV2-S eklemekle başlanır (ConvNeXt eklerken izlenen
yöntemle aynı: her mimarinin "classifier" katman ismini kontrol et, aynı
`set_backbone_trainable` mantığı çoğunlukla değişmeden çalışıyor).

Henüz başlanmayan diğer fikirler: checkpoint ensemble (`ensemble.py` zaten
yazılı, henüz gerçek checkpoint'lerle test edilmedi), mel-threshold ayarı,
TTA (test-time augmentation), kalibrasyon, metadata füzyonu (yaş/cinsiyet/
lezyon konumu — HAM10000'de var ama kullanılmıyor), dış veri (ISIC 2019/2020).

**Ensemble'dan sonraki iyileştirme sırası (netleşti, 2026-10-10):** Önce 3
mimariyi de düz ortalama ile birleştir → sonra **entegrasyonun kendisini**
ince ayarla (ağırlıklı ortalama, sınıf-bazlı ağırlık, kalibrasyon — hiçbiri
yeniden eğitim istemiyor, ucuz/hızlı) → **ancak hâlâ yetersizse ve zayıf
halka netse**, en pahalı/son çare olarak tek bir mimariyi (örn. daha fazla
epoch'la) yeniden eğitmeyi düşün. Entegrasyon ayarı, tek model yeniden
eğitiminden önce gelmeli (daha ucuz, daha hızlı denenebilir).

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
