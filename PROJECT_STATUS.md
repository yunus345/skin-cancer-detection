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
yolculuğuna başladık: ConvNeXt-Tiny mimari desteği hem kodda hem Colab
notebook'unda hazır (ayrı bir branch'te, `main`/`baseline-v1`
dokunulmamış). Tek eksik: Colab'da gerçek koşuyu **sen başlatman**
gerekiyor — Claude Colab'a bağlanamıyor, bu adımı otomatik yapamaz.

## Sıradaki adım (tek, somut)
Her şey hazır, sadece Colab'ı açıp çalıştırman kaldı:

1. [colab.research.google.com](https://colab.research.google.com) → Dosya
   → Not defterini aç → GitHub sekmesi → repo: `yunus345/skin-cancer-detection`
   → **branch: `mimari-convnext-tiny`** (önemli, main değil) → dosya:
   `notebooks/00_project_setup.ipynb`
2. Çalışma zamanı türü → T4 GPU
3. Hücreleri baştan sırayla çalıştır (kod zaten `--architecture convnext_tiny`
   ve doğru branch'i klonlayacak şekilde ayarlı, ekstra bir şey eklemene
   gerek yok)
4. Bitince sonucu `baseline-v1`deki sayılarla (balanced_accuracy=0.762,
   macro_f1=0.698) kıyaslarız

(branch: `mimari-convnext-tiny`, son commit `e9de811`)

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
