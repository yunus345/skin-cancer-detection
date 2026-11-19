# Yol Haritası: v1 Kapanış → Arayüz → v2

> Bu dosya ana geliştirme ajanı için genel plandır. Adımlar **sırayla** yapılır. Bir aşama bitmeden sonrakine geçilmez ve her aşamanın sonunda kullanıcıdan onay alınır.
> Referans: `docs/AUDIT_2026-10-26.md` (bağımsız denetim raporu).

---

## Bağlam (kısa)

- **Proje:** HAM10000 ile 7 sınıflı cilt lezyonu sınıflandırma, PyTorch, transfer learning. Öğrenme amaçlı (bkz. `CLAUDE.md`).
- **En iyi model:** ConvNeXt-Tiny, ImageNet-22k ön eğitimli, 384 px, tarif v2.
  - Seed 42: iç test F1 0.818, BA 0.809.
  - Seed 2: iç test F1 0.821, BA 0.830.
- **final_v1 (2026-11-12, dondurulmuş, tek ölçüm):** iki seed'in kalibre ortalaması, tarama eşiği 0.28.
  - İç test: F1 0.829, BA 0.832. İç test yaklaşık 3.5 puan iyimser.
  - Dış test (ISIC 2018, ikinci bakış): F1 0.795, BA 0.796. Genelleme için raporlanacak sayı bu.
  - Ayrıntı: `baselines/final_v1/README.md`.
- **Gürültü seviyesi:** Sadece seed değişince val F1 ~0.013, val melanom recall ~0.11 oynuyor. Test BA'nın %95 güven aralığı ~9 puan genişliğinde.
- **Denetimin ana bulgusu:** Modelleme fiilen bitti, eksik olan kapanış ve paketleme. Savrulmanın sebebi "bitti" tanımının ve durma kuralının olmamasıydı.

---

## Genel kurallar (bütün aşamalarda geçerli)

1. **Öğrenme modu:** `CLAUDE.md` geçerli. Her adımdan önce ne yapılacağını ve nedenini kısaca açıkla. Kavramsal kararlarda önce kullanıcıya sor.
2. **Kararlar sadece validation'da (veya CV'de) verilir.** İç test ve dış test sadece aşama sonunda, bir kez kullanılır.
3. **Tek değişiklik kuralı:** Her deneyde sadece bir şey değişir.
4. **Başarı kriteri önceden yazılır.** Deney başlamadan önce kriter `EXPERIMENTS.md`'ye yazılır. Sonuç ne çıkarsa çıksın kaydedilir.
5. **Gürültüden küçük fark, fark değildir.** Eşit durumda daha basit ve ucuz seçenek seçilir.
6. **Plan dışı fikirler `IDEAS.md`'ye yazılır,** o an uygulanmaz.
7. **Tek sayı kaynağı:** Mac'te (fp32) `predictions.npz` dosyalarından hesaplanan değerler.
8. **Git ve dosya kuralları:**
   - `git add` her zaman dosya adıyla yapılır, `-A` kullanılmaz.
   - `deri-kanseri-cnn-uygulamas.ipynb` dosyasına dokunulmaz.
   - **Sürüm ve branch düzeni (2026-11-18):** v1'in bittiği commit `v1` tag'i ile işaretli (`092bcf4`; HF modeli de `v1`).
     Ölçüm altyapısı (C0) main'de. C2'den itibaren her deney kendi branch'inde (`exp/c2-isic2019` gibi); Kaggle o branch'i
     klonlar (`git clone -b <branch>`). Kabul edilen deneyin kodu main'e birleşir; reddedilenin kodu birleşmez ama sonucu
     ve kaydı (`EXPERIMENTS.md`, sonuç dosyaları) her durumda main'e girer.
   - `.pt` dosyaları git'e girmez. Ağırlıklar en az iki yerde ve MD5'leriyle tutulur.
9. **Belge disiplini:**
   - `PROJECT_STATUS.md` kısa tutulur ve **tek bir** sonraki adımı gösterir.
   - Denetim raporu `docs/AUDIT_2026-10-26.md` dosyasında kalır ve üzerine yazılmaz.

---

# AŞAMA A: v1 Kapanış

> Yeni eğitim yok, GPU gerekmiyor. Hedef süre: 2–3 oturum.

## A0. Kararlar (kullanıcı onaylar)
- [x] Hedef: "İç testte F1 ≥ 0.80 ve BA ≥ 0.80, dış test dürüstçe raporlanır". BA ≥ 0.85 hedefi bırakılır.
- [x] Ana metrik: macro F1. BA, macro precision ve melanom recall'u da raporlanır.
- [x] Train+val ile yeniden eğitim planı iptal edilir.
- [ ] Dersin teslim beklentisi (format, tarih, demo) öğrenilir ve buraya not edilir.
- [x] Kontrol: tarif v2 iki aşamalı mı (donuk backbone + fine-tune), yoksa warmup ile tek aşamalı mı? Gerçekte ne uygulandığı koddan doğrulanıp README'ye yazılır. (Doğrulandı: iki fazlı, 5 epoch donuk + 25 epoch warmup/cosine; README "Yöntem".)

## A1. Güvenlik ve repo düzeni
- [x] Ağırlıklar Drive `archive/`'e kopyalandı, MD5'ler eşleşti ve `baselines/final_v1/README.md`'de. (2026-11-12)
- [x] Ağırlıkların üçüncü kopyası: Hugging Face Hub `emreozkn/skin-lesion-convnext-v1` (`v1`), 2026-11-17.
- [x] Denetim raporunu `docs/AUDIT_2026-10-26.md` dosyasına taşı. `PROJECT_STATUS.md`'yi kısa formata çevir. (Yapıldı 2026-11-17.)
- [x] `.gitignore`: `baselines/*/*.pt`.
- [x] Split dosyası: `splits/ham10000_lesion_split_v1.csv`, MD5 `a5e29e8b4186db15209b31715fa3a94a`. `validate_manifest.py` bölme bundan saparsa hata veriyor.
- [x] Takip dışı dosyalar:
  - Small ve seed 2'nin `test_metrics.json` dosyaları "karar sonrası görüldü" notuyla commit edilir.
  - `AGENT_REPORT.md`'nin içeriği düzeltilerek `EXPERIMENTS.md`'ye taşınır.
- [x] Klasör düzeni: `baselines/final_v1/` + iki üye üstte, 10 eski deney `baselines/_archive/`'de (2026-11-17). Kod yolları güncellendi; `ensemble_val` (calibrate, candidate, tta), `metadata_fusion`, `ensemble_test`, `freeze_recipe` çıktıları taşıma öncesiyle birebir aynı.

## A2. Final modeli dondur (sadece val) ✅
- [x] Adaylar A (seed 42) ve B (iki seed ortalaması); seçim kuralı önceden yazıldı; B seçildi, tarama eşiği 0.28.
- [x] `baselines/final_v1/README.md` ve `recipe.json` test ölçümünden önce commit'lendi (`bb0cf19`).

## A3. Tek ölçüm ✅
- [x] AUC eklendi, scriptler `--recipe` okuyor, ensemble_v1 regresyon kontrolü geçti.
- [x] İç test ve dış test (ikinci bakış), tek ölçüm. Sonuçlar `baselines/final_v1/README.md`'de (commit `1730eb4`).

## A4. Tekrarlanabilirlik
- [x] `requirements.txt` sabitlendi (2026-11-18): Mac (değerlendirme) sürümleri; Kaggle eğitim ortamı `baselines/cv5_baseline/pip_freeze.txt`'de ve dosyada yorum olarak.
- [x] Final eğitim ve değerlendirme komutları tek blok halinde README'ye yazılır.
- [x] Temiz bir klonda değerlendirme komutunun çalıştığı doğrulanır. (2026-11-17: iç test birebir yeniden üretildi; dış test görüntü gerektirdiği için klonda çalıştırılmadı.)

## A5. Belgeleme
- [x] `README.md`: problem, veri, yöntem, sonuç tablosu, sınırlamalar.
- [x] **Sınırlamalar bölümünü kullanıcı kendisi yazar.** Ajan sadece taslak madde listesi verir:
  - Test seti geliştirme sırasında görüldü (K1).
  - Dış test erken kullanıldı (O1).
  - Değişiklikler ablation'sız yapıldı (O7).
  - Hasta kimliği yok (D1).
  - Değişkenlik sadece 2 seed ile ölçüldü.
- [x] `EXPERIMENTS.md`: deney geçmişi, denetimin 5. bölümü temel alınır.
- [x] Eskimiş belgeler temizlenir (`src/README.md`, README'deki "en iyi sonuç"). `git status` temiz olmalı.

**✅ A aşaması bitti sayılır:** final_v1 dondurulmuş ve ölçülmüş, ağırlıklar iki yerde, README güncel.

---

# AŞAMA B: Arayüz (v1 modeliyle)

> v2'yi beklemeden yapılır. Model ileride değiştirilebilecek şekilde kurulur.
> **Süre sınırı: en fazla 3 oturum.** Bitmeyen kısım (B3 dahil) v2 sonrasına bırakılır. Grad-CAM tek model (seed 42) üzerinde gösterilir ve ekranda belirtilir.

## B1. Model paketleme
- [x] (2026-11-17, `src/inference/package_model.py` → `models/final_v1/`: `seed42.pt`, `seed2.pt`, `config.json`; `verify_package.py` val'de kayıtlı olasılıklarla birebir aynı) Model klasörü: `weights.pt` ve `config.json`. Config şunları içerir: mimari adı, görüntü boyutu, sınıf listesi, T, melanom eşiği, sürüm, Grad-CAM hedef katmanı.
- [x] Hugging Face Hub'a sürüm etiketiyle yüklenir (`v1`).
- [x] Ön işleme (Gray-World ve normalizasyon) `src/` içindeki koddan doğrudan import edilir, kopyalanmaz.

## B2. Gradio uygulaması (Hugging Face Spaces)
- [x] Görüntü yükleme ve HAM10000'den birkaç örnek görüntü. (`app.py`, yerelde çalışıyor, 2026-11-17)
- [x] En olası 3 sınıf ve olasılıkları.
- [x] Grad-CAM ısı haritası. (`Predictor.gradcam`, 2026-11-17; arayüze bağlanması B2b'de.)
- [x] Normal / tarama modu seçimi.
- [x] (τ = 0.85, `choose_uncertainty_threshold.py`, val) "Emin değilim" kuralı: en yüksek olasılık eşiğin altındaysa tahmin yerine "doktor incelemesi önerilir" uyarısı. Eşik val'de belirlenir.
- [x] Sabit uyarılar: "Araştırma amaçlıdır, tanı aracı değildir. Gerçek hasta görüntüsü yüklemeyin."
- [x] Ekranda model sürümü ve test metrikleri gösterilir.

## B3. (İsteğe bağlı) Giriş kontrolü
- [ ] "Dermoskopi görüntüsü mü?" ön filtresi. Basit bir yöntemle başlanır (OOD skoru veya küçük bir ikili sınıflandırıcı).

> **2026-11-17 durumu:** Gradio Space ücretsiz CPU'da PRO istiyor (HTTP 402). Kullanıcı ücret ödemek istemiyor: kalıcı ve ücretsiz seçenek v2'den sonra araştırılacak; olmazsa sunum günü `SHARE=1` geçici linki. B bu haliyle kapatıldı.

**✅ B aşaması bitti sayılır:** Spaces linki çalışıyor, README'de demo linki ve ekran görüntüsü var.

---

# AŞAMA C: v2 (geliştirme)

> v1 ve arayüz bittikten sonra başlar. Her adımın başarı kriteri önceden yazılır.
> **Durma kuralı:** Bir adım kriteri sağlamazsa sonuç kaydedilir ve sonraki adıma geçilir, aynı adım tekrar tekrar denenmez.
> **v1 her zaman kıyas noktasıdır.** v2, v1'i gürültüden büyük farkla geçmezse arayüzdeki model değişmez.
>
> **Bitiş kuralı (hangisi önce gelirse C biter):**
> - **Hedef tuttu:** OOF tahminlerinde macro F1 ve BA ikisi de baseline + 0.02. Baseline ölçülünce hedef mutlak sayıya çevrilip kilitlenir.
> - **Bütçe bitti:** baseline CV (C0) dışında en fazla 3 deney (C2, C3 ve bir serbest hak).
> - **Koruma şartı:** melanom recall baseline'ın altına düşmeyecek.
>
> **Kabul kriteri (her deney için, önceden yazılır):** 5 fold'un en az 4'ünde daha iyi **ve** ortalama fark > 0.01. Fold std'sinin 2 katı ayrıca raporlanır, bilgi amaçlı.
> **CV'de en iyi epoch seçilmez, son epoch alınır** (fold içinde seçim OOF skorunu şişirir).

## C0. Ölçüm altyapısı: 5-fold CV (önce bu)
**Neden:** 992 görüntülük val seti 0.01–0.02'lik farkları ayırt edemiyor. CV ile değerlendirme ~9.000 görüntüye dayanır.
- [x] Train+val birleştirilir. **İç test setine dokunulmaz.**
- [x] `StratifiedGroupKFold` (grup = `lesion_id`, katman = `label`) ile 5 fold oluşturulur ve `splits/cv5_v1.csv` olarak kaydedilir. (2026-11-17, MD5 `e8e35f94bd974356390d3f2aba045587`)
- [x] v1 tarifi 5 fold'da eğitilir (Kaggle'da ~5 saat). Her fold'un OOF (out-of-fold) tahminleri kaydedilir.
- [x] Çıktılar:
  - CV ortalaması ± standart sapma (F1, BA, melanom recall).
  - **Gürültü eşiği** = 2 × fold standart sapması. Bundan sonraki her deney bu eşikle değerlendirilir.
- [x] Yan ürün: 5 fold modelinin ortalaması bir ensemble olur ("v1-CV ensemble"). (C4'te kuruldu ve final_v2 oldu; testte v1'i geçmedi.)

> **C0 sonucu (2026-11-18):** baseline OOF F1 0.785 / BA 0.779 / melR 0.662; hedef F1 ≥ 0.8048, BA ≥ 0.7993 kilitli (`baselines/cv5_baseline/`).

## C1. Yeni, temiz dış test seti
> **ATLANDI (2026-11-18, kullanıcı kararı).** Gerekçe: dış teste bakarak karar verilmiyor; yeni set kurmanın maliyeti
> (lisans, örtüşme kontrolü, eksik sınıflar) kazancından büyük. Şartlar: ISIC 2018 dış testi yalnızca C5'te, v2 tarifi
> dondurulup commit'lendikten sonra **bir kez** ölçülür ve "üçüncü bakış" diye raporlanır; bu sete bakarak hiçbir şey
> değiştirilmez; README sınırlamalarına eklenir. Aşağıdaki maddeler uygulanmayacak.
**Neden:** ISIC 2018 test seti v2 sonunda üçüncü kez kullanılmış olacak.
- [ ] Aday araştırılır, örneğin Derm7pt (dermoskopi, ~1000 vaka). Erişim ve lisans doğrulanır.
- [ ] HAM10000 ve ISIC ile örtüşme kontrolü yapılır (ID ve görüntü hash'i).
- [ ] Sınıf eşleştirme tablosu yazılır. Eşleşmeyen sınıflar dışarıda bırakılır ve bu raporlanır.
- [ ] Set **en sona** saklanır. Hiçbir karar bu sete bakılarak verilmez.

## C2. Ek veri, doğru şekilde (ISIC 2019, HAM dışı kısmı)
**Önceki denemenin sorunları:** Sınıf ağırlıkları melanomu ezdi, etiket eşleştirmesi kaydı, değerlendirme sadece iç testte yapıldı.
- [x] Ek veri sadece train fold'larına eklenir, validation fold'ları saf HAM kalır (`--extra-train-source`, branch'te).
- [x] Tekrar kontrolü: HAM ile örtüşen görüntüler ID ve hash ile çıkarılır (dHash, Hamming ≤ 12).
- [x] Etiket eşleştirmesi önceden yazıldı: SCC ve UNK çıkarılır, AK → akiec.
- [x] Sınıf ağırlığı önceden seçildi (2026-11-18): **baseline kuralı (ters frekans) korunur**; karekök ve üst sınır reddedildi, gerekçe `EXPERIMENTS.md` [C2].
- [ ] ~~Alan kayması ölçümü (BCN20000 dış validation)~~: yapılmayacak (kullanıcı kararı: minimum GPU/zaman). Değerlendirme yalnızca HAM OOF.
- [x] **Başarı kriteri ve pilot kuralı önceden yazıldı** (`EXPERIMENTS.md` [C2]): önce yalnızca fold 0; F1 ve BA'da +0.02 ve melanom recall düşmüyorsa kalan fold'lar, değilse deney biter.
- [x] **Sonuç (2026-11-19): REDDEDİLDİ.** Fold 0: F1 0.766 (eşik 0.796), BA 0.730 (eşik 0.769), melanom recall 0.624. Kalan fold'lar koşulmadı, kod main'e girmedi. Ayrıntı `EXPERIMENTS.md` [C2], `baselines/c2_isic2019_pilot/`.

## C3. Daha güçlü ön eğitim (tek aday) - ATLANDI (kullanıcı kararı, 2026-11-19)
> Belgeler okundu, eğitim yapılmadı. PanDerm: ön eğitimde HAM/ISIC 2018-2019 adıyla yok, ama Avustralya klinik arşivleri
> (HAM'in Queensland kaynağıyla örtüşme) belgelerden dışlanamıyor; lisans CC-BY-NC-ND (fine-tune edilmiş modeli yayımlamak
> sorunlu). DINOv2: tıbbi kaynak yok, risk ImageNet-22k ile aynı türde; ama alana özel değil ve T4'te birkaç kat yavaş.
> Kullanıcı minimum GPU/zaman için adımı atladı. Serbest deney hakkı da kullanılmadı. Ayrıntı `EXPERIMENTS.md` [C3].
- [ ] Tek aday seçilir: DINOv2 ViT-S/B veya PanDerm-Base. Model T4'e sığmalı.
- [ ] **Sızıntı kontrolü:** Ön eğitim verisinde HAM/ISIC görüntüsü olup olmadığı belgelerden araştırılır. Doğrulanamazsa bu adım atlanır ve nedeni yazılır.
- [ ] Aynı CV ile eğitilir.
- [ ] **Başarı kriteri (önceden):** CV F1, v1-CV'yi gürültü eşiğinden büyük farkla geçmeli.

## C4. Final v2 tarifi (sadece CV/OOF verisiyle)
- [x] C2 ve C3'ten kazananlar birleştirilir. Kazanan yoksa v1 tarifi kalır. (Kazanan yok: tarif v1; v2 = baseline CV'nin 5 fold modeli.)
- [x] Fold ensemble, T kalibrasyonu ve melanom tarama eşiği OOF tahminleri üzerinde belirlenir (Mac'te yeniden hesaplanan OOF; t = 0.07, τ = 0.90).
- [x] `baselines/final_v2/README.md` yazılır ve **test ölçümünden önce** commit edilir (`2e800d9` kurallar, `d0c13f2` tarif).

## C5. Tek ölçüm ve karşılaştırma
- [x] İç testte bir kez ("geliştirme sırasında görülmüş set" notuyla): F1 0.826, BA 0.817 (v1 0.829 / 0.832).
- [x] ~~C1'deki yeni dış test setinde~~ ISIC 2018 dış testinde bir kez ("üçüncü bakış"; C1 atlandı): F1 0.813, BA 0.794 (v1 0.795 / 0.796).
- [x] v1'in aynı setteki sonucu zaten var (F1 0.795 / BA 0.796, ikinci bakış); v2 ile aynı sette karşılaştırılır.
- [x] v2, v1'i geçiyorsa arayüzdeki model `config.json` değiştirilerek güncellenir (HF Hub'da `v2` etiketi). Geçmiyorsa v1 kalır ve bu da bir sonuç olarak raporlanır. **Sonuç (2026-11-19): v1 kalır** (önceden yazılan kuralın 4 koşulundan 1'i tuttu; farklar güven aralığı içinde). `baselines/final_v2/README.md`.

---

# AŞAMA D: Sohbet paneli ve RAG (isteğe bağlı)

> Sadece v2 bittikten sonra ve vakit kalırsa. Önceki aşamaları geciktirmez.

- [ ] Arayüze "Asistana sor" paneli eklenir.
- [ ] LLM'e **görüntü gönderilmez, sadece CNN'in yapılandırılmış çıktısı** gönderilir: sınıflar, olasılıklar, mod.
- [ ] Bilgi tabanı: açık erişimli dermoskopi ve klinik kılavuz kaynakları. Lisansları kontrol edilir.
- [ ] Kurallar:
  - LLM sayıları değiştirmez ve kendi teşhisini koymaz.
  - Her iddia için kaynak gösterir.
  - Kaynakta olmayan bir şey sorulursa "kaynağım yok" der.
- [ ] Değerlendirme: küçük bir soru seti hazırlanır, cevapların kaynaklara sadakati (faithfulness) ölçülür. Örneğin RAGAS ile.

---

## v1 bitene kadar YAPILMAYACAKLAR
Yeni eğitim koşuları, yeni mimariler, ek veri, CV, TTA, metadata füzyonu, mixup/CutMix, logit adjustment, daha yüksek çözünürlük, daha büyük modeller.

## Kalıcı olarak YAPILMAYACAKLAR (kanıt aleyhte)
TTA, metadata füzyonu, test çözünürlüğünü artırmak, ConvNeXt-Small ve üstü (8 bin görüntüde kazanç yok), under/oversampling.

---

## Deney kayıt şablonu (`EXPERIMENTS.md` için)

```
### [Kod] Deney adı — tarih
- Değişen tek şey:
- Hipotez:
- Başarı kriteri (önceden):
- Değerlendirme: CV / dış validation / test (hangisi, kaçıncı kullanım)
- Sonuç:
- Karar: kabul / ret / belirsiz
- Kanıt dosyaları:
```
