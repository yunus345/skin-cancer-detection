# Proje Durumu

(Projeye her dönüşünde önce burayı oku — sohbet geçmişini hatırlamana gerek yok.)

## Bu ne projesi?
HAM10000 veri setiyle cilt lezyonu sınıflandırma (7 sınıf: akiec, bcc, bkl,
df, mel, nv, vasc). PyTorch + EfficientNet-B0, transfer learning. Öğrenme
amaçlı bir proje — bkz. CLAUDE.md.

## Şu anki en iyi, güvenceli sonuç: `baseline-v1`
- balanced_accuracy: **0.762** (%95 GA 0.716–0.804)
- macro_f1: 0.698, mel_recall: 0.626
- Kod + eğitilmiş ağırlıklar + metrikler git'te kalıcı olarak duruyor,
  hiçbir dış depolamaya (Drive, Downloads) bağımlı değil.
- Bu sonuca her zaman dönebilirsin: `git checkout baseline-v1`
- Detaylı rapor: `baselines/baseline_v1/README.md`

## Şu an neredeyiz?
Proje güvencede, **acil yapılacak bir şey yok**. Sıradaki büyük adım olarak
"köklü değişiklikler" konuşuldu (farklı model mimarisi, checkpoint ensemble,
daha büyük görüntü boyutu) ama henüz hiçbirine başlanmadı — `baseline-v1`
güvenli bir referans noktası olarak dururken istediğin zaman, istediğin
hızda denemeye başlayabiliriz.

## Sıradaki adım (tek, somut)
Karar verildi: EfficientNet-B0'ın yanına **ConvNeXt-Tiny** mimarisini
ekleyeceğiz (torchvision'da pretrained hazır, ekstra bağımlılık gerekmiyor).
Seçim sebebi: ResNet50'den daha güncel/güçlü, EfficientNet'ten yeterince
farklı bir tasarım (ileride ensemble çeşitliliği için önemli).

Çalışma branch'i hazır: `mimari-convnext-tiny` (main'den ayrıldı, henüz
kod değişikliği yok). Döndüğünde: `git checkout mimari-convnext-tiny` ile
başla, `build_model`'e mimari parametresi eklemekle devam ederiz.

Henüz başlanmayan diğer fikirler (sırayla): checkpoint ensemble (aynı
mimari, farklı seed), augmentation/görüntü boyutu büyütme, mel-threshold
ayarı.

## Unutulmaması gereken kurallar
- `deri-kanseri-cnn-uygulamas.ipynb` dosyasına **asla dokunma** (senin ayrı,
  ilgisiz referans notebook'un — stage/commit edilmeyecek)
- Git'e dosya eklerken her zaman dosya adıyla (`git add <dosya>`), asla
  `git add -A` kullanılmıyor (yukarıdaki dosyayı yanlışlıkla eklememek için)
- Colab'da uzun eğitim koşarken sekmeyi arka planda bırakma / bilgisayarı
  uykuya alma — runtime koptu/sıfırlandı, iki kere bu yüzden ilerleme kaybettik
