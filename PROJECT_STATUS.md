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
Proje güvencede, **acil yapılacak bir şey yok**. "Köklü değişiklikler"
yolculuğuna başladık: ilk adım olan ConvNeXt-Tiny mimari desteği kodda
hazır (ayrı bir branch'te, `main`/`baseline-v1` dokunulmamış). Sırada bu
mimariyle gerçek bir eğitim koşusu var — henüz başlatılmadı.

## Sıradaki adım (tek, somut)
ConvNeXt-Tiny kod desteği **tamamlandı ve yerel duman testinde doğrulandı**
(branch: `mimari-convnext-tiny`, commit `c86178a`). `build_model` artık
`--architecture efficientnet_b0|convnext_tiny` ile seçim yapabiliyor,
checkpoint'ler hangi mimariyle eğitildiklerini kaydediyor.

Şimdiki adım: bu branch'te **gerçek bir tam eğitim koşusu** (Colab'da,
`--architecture convnext_tiny` ekleyerek, diğer bayraklar aynı) ve
sonucu `baseline-v1` ile kıyaslamak. Döndüğünde: `git checkout
mimari-convnext-tiny`, sonra Colab notebook'unun eğitim hücresine
`--architecture convnext_tiny` ekleyip (henüz eklenmedi) çalıştır.

Henüz başlanmayan diğer fikirler (sırayla): Swin V2 ve EfficientNetV2-S
mimarileri, checkpoint ensemble (üç mimariyi birleştirmek), augmentation/
görüntü boyutu büyütme, mel-threshold ayarı.

## Unutulmaması gereken kurallar
- `deri-kanseri-cnn-uygulamas.ipynb` dosyasına **asla dokunma** (senin ayrı,
  ilgisiz referans notebook'un — stage/commit edilmeyecek)
- Git'e dosya eklerken her zaman dosya adıyla (`git add <dosya>`), asla
  `git add -A` kullanılmıyor (yukarıdaki dosyayı yanlışlıkla eklememek için)
- Colab'da uzun eğitim koşarken sekmeyi arka planda bırakma / bilgisayarı
  uykuya alma — runtime koptu/sıfırlandı, iki kere bu yüzden ilerleme kaybettik
