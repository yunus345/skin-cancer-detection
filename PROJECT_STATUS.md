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
| `swin_v2_t_v1` (256px, Kaggle) | 0.755 | **0.813** | 0.626 | nadir sınıflarda en iyi, melanomda zayıf |

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
**Ensemble:** önce `ensemble.py`'ye validation seti desteği eklemek (model
seçimi ve ağırlıklar validation'da seçilecek, test sadece en sonda bir kez).
Sonra ConvNeXt + V2-S + Swin (+ B0?) düz ortalama → ağırlıklı ortalama /
kalibrasyon. Bu aşama eğitim istemiyor, Mac'te yapılabilir.

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
(leakage). `ensemble.py` şu an sadece test setinde çalışıyor; ağırlık
ayarı aşamasında validation desteği eklenecek.

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
