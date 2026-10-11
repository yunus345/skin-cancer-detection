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
| `convnext_tiny_v1` (224px) | **0.774** | **0.798** | **0.766** | en iyi tek model |
| `efficientnet_v2_s_v1` (384px) | 0.753 | 0.781 | 0.710 | ConvNeXt'ten farklı yerlerde hata yapıyor |

Her klasörde README (ayrıntılı sonuçlar, güven aralıkları, bulgular) var.
Ağırlıklar: `baseline_v1` git'te; diğerleri (50MB+) yerel diskte + Drive'da
`skin-cancer-detection-runs/archive/<model adı>/` altında.

## Şu an neredeyiz?
Plan "önce genişlik": birkaç farklı mimariyi aynı tarifle eğit, sonra ensemble.
İki güçlü model var. EfficientNetV2-S (2026-10-11) tek başına ConvNeXt'i
geçmedi, ama hataları farklı yerlerde: V2-S daha az yanlış melanom alarmı
veriyor, ConvNeXt melanomu çok daha az kaçırıyor (mel→nv 7'ye karşı 19). Bu,
ensemble için umut verici. V2-S'te val_loss epoch ~11-14'ten sonra yükseldi,
yani daha fazla epoch bu modele yaramaz.

Altyapı (2026-10-11): Colab'da mixed precision (AMP), paralel veri yükleme
(`--num-workers`, varsayılan 2), cudnn.benchmark. Çözünürlük checkpoint'e
kaydediliyor; evaluate/tune_threshold/ensemble onu oradan okuyor (ensemble her
modeli kendi çözünürlüğünde test ediyor).

## Sıradaki adım (tek, somut)
**Swin V2-Tiny** (transformer, 256px): üçlünün en farklı üyesi. Kod tarafında
`build_model`'e eklenecek; son katmanı `classifier` değil `head` adında olduğu
için `set_backbone_trainable` da güncellenmeli (yoksa Faz 1'de her şey donar,
eğitim çöker). Sonra aynı Colab döngüsü.

Colab döngüsü:
1. Notebook'u **her seferinde şu linkten aç** (GitHub'daki en güncel hali gelir):
   https://colab.research.google.com/github/yunus345/skin-cancer-detection/blob/main/notebooks/00_project_setup.ipynb
   → T4 GPU
2. İlk dakikalarda log'da `Mixed precision (AMP): on` ve doğru
   `Architecture: ..., image size: ...px` satırını kontrol et.
3. `CUDA out of memory` alırsan eğitim hücresine `--batch-size 16` ekle.
4. Bitince Drive uygulamasını ya da Finder'da Google Drive'ı aç (senkronu
   tetikler). Sonucu `baselines/<model adı>/` + Drive `archive/<model adı>/`
   olarak kaydederiz.

Henüz başlanmayan diğer fikirler: mel-threshold ayarı, TTA (test-time
augmentation), kalibrasyon, metadata füzyonu (yaş/cinsiyet/lezyon konumu —
HAM10000'de var ama kullanılmıyor), dış veri (ISIC 2019/2020), ConvNeXt'i
384px'te yeniden eğitmek (cilalama adayı, ~3 kat süre).

**Ensemble'dan sonraki iyileştirme sırası:** Önce modelleri düz ortalama ile
birleştir → sonra **entegrasyonun kendisini** ince ayarla (ağırlıklı ortalama,
sınıf-bazlı ağırlık, kalibrasyon — hiçbiri yeniden eğitim istemiyor) →
**ancak hâlâ yetersizse ve zayıf halka netse**, son çare olarak tek bir
mimariyi yeniden eğit.

Dikkat: ensemble ağırlıkları ve hangi modellerin dahil edileceği
**validation** setinde seçilmeli, test seti sadece en sonda bir kez
kullanılmalı. Test setine bakarak seçim yaparsak sonuç iyimser çıkar
(leakage). `ensemble.py` şu an sadece test setinde çalışıyor; ağırlık
ayarı aşamasında validation desteği eklenecek.

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
