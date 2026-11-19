# Yol Haritası: v1 Kapanış → v2 → Web Uygulaması → Rapor

> Bu dosya ana geliştirme ajanı için genel plandır. Adımlar **sırayla** yapılır. Bir aşama bitmeden sonrakine geçilmez ve her aşamanın sonunda kullanıcıdan onay alınır.
> Referans: `docs/AUDIT_2026-10-26.md` (bağımsız denetim raporu).
>
> **Sürüm 2 (2026-11-19) değişiklikleri:**
> - Aşama sırası değişti: A → C → **W (Web uygulaması)** → **R (Rapor, en son iş)**.
> - Eski Aşama B (Gradio) ve D (sohbet + RAG) birleştirildi: **Aşama W**, kendi web sitesi (FastAPI backend + web frontend) ve içinde asistan sohbeti.
> - C3 atlandı (gerekçe aşağıda). C4'te seçenek B (fold ensemble) seçildi.

---

## Durum özeti

| Aşama | Durum |
|---|---|
| A: v1 kapanış | Büyük ölçüde bitti. Kalanlar A-kalan listesinde. |
| C0: 5-fold CV | Bitti |
| C1: Temiz dış test seti | **Atlandı** (2026-11-18); ISIC 2018 üçüncü kullanım |
| C2: Ek veri | ❌ Reddedildi (fold 0 pilotu, 2026-11-19) |
| C3: Daha güçlü ön eğitim | **Atlandı** |
| C4: Final v2 kararı | ✅ Bitti (seçenek B: 5 fold ensemble) |
| C5: Tek ölçüm ve karşılaştırma | ✅ Bitti: **v1 kalıyor** (commit c9eee16) |
| W: Web uygulaması | **Sıradaki** (önce A-kalan kontrolü, sonra W0 tasarım) |
| R: Rapor | Bekliyor (**en son iş**) |

---

## Bağlam (kısa)

- **Proje:** HAM10000 ile 7 sınıflı cilt lezyonu sınıflandırma, PyTorch, transfer learning. Öğrenme amaçlı (bkz. `CLAUDE.md`).
- **final_v1:** ConvNeXt-Tiny, ImageNet-22k ön eğitimli, 384 px, tarif v2. Sayılar `baselines/final_v1/` içinde.
- **Gürültü:** Seed değişince val F1 ~0.013, val melanom recall ~0.11 oynuyor. C0'da hesaplanan gürültü eşiği bundan sonraki bütün karşılaştırmalarda kullanılır.
- **Denetimin ana bulgusu:** Savrulmanın sebebi "bitti" tanımının ve durma kuralının olmamasıydı.

---

## Genel kurallar (bütün aşamalarda geçerli)

1. **Öğrenme modu:** `CLAUDE.md` geçerli. Her adımdan önce ne yapılacağını ve nedenini kısaca açıkla. Kavramsal kararlarda önce kullanıcıya sor.
2. **Kararlar sadece validation'da (veya CV/OOF'ta) verilir.** İç test ve dış test sadece aşama sonunda, bir kez kullanılır.
3. **Tek değişiklik kuralı:** Her deneyde sadece bir şey değişir.
4. **Başarı kriteri önceden yazılır.** Deney başlamadan önce kriter `EXPERIMENTS.md`'ye yazılır. Sonuç ne çıkarsa çıksın kaydedilir.
5. **Gürültüden küçük fark, fark değildir.** Eşit durumda daha basit ve ucuz seçenek seçilir.
6. **Plan dışı fikirler `IDEAS.md`'ye yazılır,** o an uygulanmaz.
7. **Tek sayı kaynağı:** Mac'te (fp32) `predictions.npz` dosyalarından hesaplanan değerler.
8. **Git ve dosya kuralları:**
   - `git add` her zaman dosya adıyla yapılır, `-A` kullanılmaz.
   - `deri-kanseri-cnn-uygulamas.ipynb` dosyasına dokunulmaz.
   - `.pt` dosyaları git'e girmez. Ağırlıklar en az iki yerde ve MD5'leriyle tutulur.
   - Repo herkese açıksa tasarım referans görüntüleri (başkalarının tasarımları) commit edilmez: `docs/design/refs/` `.gitignore`'da olur.
9. **Git akışı:**
   - Kapanış ve belge işleri doğrudan `main`'de.
   - Her deney ayrı branch'te (`exp/...`). Başarısız deneyin kodu birleştirilmez, ama sonucu `EXPERIMENTS.md`'de `main`'e yazılır.
   - Web uygulaması `feature/web` branch'inde geliştirilir, çalışınca `main`'e birleştirilir.
   - Her aşama sonunda etiket: `v1.0`, `v2.0`, `web-v1`.
   - Kaggle'da eğitim yapılırken notebook'un doğru branch'i çektiği kontrol edilir.
10. **Belge disiplini:**
    - `PROJECT_STATUS.md` kısa tutulur ve **tek bir** sonraki adımı gösterir.
    - Denetim raporu `docs/AUDIT_2026-10-26.md` dosyasında kalır ve üzerine yazılmaz.

---

# AŞAMA A: v1 Kapanış (büyük ölçüde bitti)

Ayrıntılı adımlar ROADMAP sürüm 1'deydi (A0–A5). final_v1 dondurulmuş ve ölçülmüş durumda.

## A-kalan (kontrol et, eksikse kapat)
- [x] `requirements.txt` sürümleri sabitlendi mi? (Evet, 2026-11-18: Mac sürümleri; Kaggle ortamı yorumda ve `baselines/cv5_baseline/pip_freeze.txt`.)
- [x] Temiz klonda değerlendirme komutu çalıştı mı? (Evet, 2026-11-17: iç test birebir yeniden üretildi.)
- [x] Sınırlamalar bölümü kullanıcının metniyle README'de mi? (Evet, 8 madde; v2 notu eklendi.)
- [x] `git status` temiz mi, `v1.0` etiketi var mı? (Temiz. Etiket `v1` adıyla kalır (kullanıcı kararı, 2026-11-19), `092bcf4`; HF modeli de `v1`.)

---

# AŞAMA C: v2 (geliştirme)

> **Durma kuralı:** Bir adım kriteri sağlamazsa sonuç kaydedilir ve sonraki adıma geçilir, aynı adım tekrar tekrar denenmez.
> **v1 her zaman kıyas noktasıdır.** v2, v1'i gürültü eşiğinden büyük farkla geçmezse uygulamadaki model v1 olarak kalır.

## C0. 5-fold CV ✅
Bitti. Gürültü eşiği ve OOF tahminleri (9.029 görüntü) mevcut.

## C1. Temiz dış test seti
- [x] Durumu kontrol et. Yapılmadıysa: v2 karşılaştırması ISIC 2018 test setinde yapılır ve raporda "üçüncü kullanım" olarak belirtilir. (Atlandı 2026-11-18; C5'te üçüncü kullanım olarak ölçüldü, README sınırlama 2'de.)

## C2. Ek veri (ISIC 2019)
- [x] Durumu kontrol et ve sonucu `EXPERIMENTS.md`'ye işle. (Reddedildi: fold 0'da F1 0.766 / BA 0.730, eşik 0.796 / 0.769; kalan fold'lar koşulmadı. `EXPERIMENTS.md` [C2], `baselines/c2_isic2019_pilot/`.)

## C3. Daha güçlü ön eğitim ❌ Atlandı
**Gerekçe:** Beklenen kazanç düşük (literatürde dermatolojiye özel modeller iyi fine-tune edilmiş modellerle "karşılaştırılabilir" düzeyde), ön eğitim verisinde HAM/ISIC sızıntısı doğrulanamıyor, maliyet yüksek. ImageNet-22k yeterli kabul edildi. README → "Gelecek çalışmalar"a yazılır.

## C4. Final v2 tarifi: fold ensemble (seçenek B)
- [x] C0'daki 5 fold modelinin ağırlıkları indirilir (Kaggle notebook Version 11 ve 12 çıktısı). MD5'ler kaydedilir. (CLI sürüm numarasını yok saydığı için Kaggle SDK ile indirildi; `models/cv5_baseline/`, MD5'ler `recipe.json`'da.)
- [x] T kalibrasyonu ve melanom tarama eşiği OOF tahminleri üzerinde seçilir. (OOF Mac'te yeniden hesaplandı; T 0.93-0.96, t = 0.07, τ = 0.90.)
- [x] **Bilinen sınırlama:** OOF'ta her görüntüyü tek model tahmin etti, ama T ve eşik 5 modelin ortalamasına uygulanacak. README'ye yazılır. (`baselines/final_v2/README.md`)
- [x] `baselines/final_v2/README.md` **test ölçümünden önce** commit edilir (`2e800d9`, tarif `d0c13f2`). Not: gerçekte commit edilen kural bu satırdakinden daha gevşekti ("iç ve dış testte F1 ve BA'nın dördü de v1'den düşük değil", kullanıcıyla kararlaştırıldı); o da tutmadı, yani iki kurala göre de sonuç aynı: v1 kalır.
- [x] Ağırlıklar alınamazsa v2 = final_v1 olur ve bu raporlanır. (Gerekmedi, ağırlıklar alındı.)

## C5. Tek ölçüm ve karşılaştırma
- [x] İç testte bir kez ("geliştirme sırasında görülmüş set" notuyla). (F1 0.826, BA 0.817)
- [x] Dış testte bir kez (C1 varsa yeni set, yoksa ISIC 2018 "üçüncü kullanım" notuyla). (F1 0.813, BA 0.794)
- [x] v1 ile aynı setlerde karşılaştırma tablosu, güven aralıklarıyla. (`baselines/final_v2/README.md`, `comparison_v1.json`)
- [x] Sınıf bazında iç test / dış test karşılaştırma tablosu (dış testteki düşüşün hangi sınıflarda yoğunlaştığını göstermek için). (`baselines/final_v1/README.md` sonu: en büyük düşüş akiec −0.12.)
- [x] Karar: uygulamada hangi model kullanılacak (v1 / v2). `v2.0` etiketi. (Karar: v1. `v2.0` etiketi `c9eee16`'da: "v2 denendi, burada bitti".)

**✅ C aşaması bitti sayılır:** Final model kesinleşti, sayılar artık değişmiyor.

### C sonucu (2026-11-19)
- v2 (5 fold ensemble) önceden yazılmış kuralın 4 koşulundan yalnızca 1'ini sağladı (dış test F1 0.813 vs 0.795). Bütün farklar güven aralığı (±0.04) içinde.
- **Karar: uygulamada ve raporda final model v1.** v2 "denendi, ölçülebilir kazanç yok" olarak raporlanır.
- Tarama modunda v2'nin melanom recall'u daha yüksek, precision'ı daha düşük; sebebi eşiğin tek model OOF'ta seçilip ensemble ortalamasına uygulanması (önceden yazılmış sınırlama).
- Dış test (ISIC 2018) üçüncü kez görüldü; raporda belirtilir.
- Ayrıntılar: `baselines/final_v2/`, `EXPERIMENTS.md`.

---

# AŞAMA W: Web uygulaması (analiz ekranı + asistan)

> Kendi web sitesi: FastAPI backend + web frontend. Analiz ekranı ve asistan sohbeti aynı uygulamada.
> **Temel ilke:** Teşhis/sınıflandırmayı her zaman CNN yapar. LLM sadece açıklar, teşhis koymaz.
> **Kod yazmadan önce tasarım onaylanır (W0).**

## W0. Tasarım (kod yok)
- [ ] **İlham panosu:** 2–3 referans ekran, her biri için "neyi alıyoruz / neyi almıyoruz" cümlesi. Görüntüler `docs/design/refs/` (gitignore'da).
- [ ] **Yerleşim kararı (referans: üç sütunlu analitik ekran):**
  - Sol: örnek vakalar / son analizler (gerçek hasta listesi yok).
  - Orta: lezyon görüntüsü + Grad-CAM, en olası 3 sınıf, metrik kartları (risk seviyesi, en olası sınıf, mod).
  - Sağ: asistan paneli, öneri butonlarıyla.
- [ ] **Asistanın rolü (karar verildi):** Sonuç yorumlama asistanı. Modelin çıktısını açıklar, Grad-CAM'i yorumlar, kaynaklı dermoskopi kriterleri verir, rapor taslağı üretir. **Teşhis koymaz.** İş akışı işleri (hasta kaydı, not, hatırlatma, hasta listesi) **kapsam dışı**, veritabanında hasta verisi tutulmaz.
- [ ] **Ekran akışı (karar verildi):**
  1. Doktor görüntüyü yükler → CNN sonucu üretir → kartlar, olasılıklar ve Grad-CAM anında dolar.
  2. Asistan panelinde **şablonla üretilmiş** özet mesajı çıkar (LLM değil; sayılar doğrudan CNN çıktısından yerleştirilir).
  3. Altında öneri butonları: "Tarama modunda göster", "Grad-CAM'i aç/kapat", "Rapor taslağı oluştur", "Model ne kadar başarılı?" (W6'da ek olarak "Ayrıntılı açıklama iste").
  4. LLM sadece doktor soru sorduğunda veya "Ayrıntılı açıklama iste"ye bastığında çalışır.
- [ ] **Mockup** onaylanır. (Taslak: Design kanvası, "Taslak 2 · analiz + asistan".)
- [ ] **Tasarım kuralları:** renk paleti, yazı tipi, boşluklar, köşe yuvarlaklığı.
- [ ] **Bileşen listesi:** yükleme alanı, risk kartı, olasılık çubukları, Grad-CAM görüntüleyici (yoğunluk ayarlı), mod seçici, asistan paneli, öneri butonları.
- [ ] **Durumlar:** boş, yükleniyor, hata, "emin değilim", "dermoskopi görüntüsü değil".
- [ ] **Uydurma tek bir skor yok:** sadece modelin gerçek çıktıları (olasılıklar, risk seviyesi) gösterilir.
- [ ] Birebir kopya yok: yerleşim ve fikirler alınır, renkler/ikonlar/görseller kendimize ait.

## W1. API sözleşmesi
- [ ] `POST /predict` girdisi (görüntü + mod) ve çıktısı (`analysis_id`, sınıflar, olasılıklar, risk, emin değilim bayrağı, Grad-CAM, model sürümü, şablon özet metni) JSON şeması olarak `docs/api.md`'ye yazılır.
- [ ] `POST /chat` (W6 için, şimdiden tanımlanır): girdi `{analysis_id, mesaj, geçmiş}`, çıktı `{cevap, kaynaklar, eylem}`. `eylem` örnekleri: `set_mode`, `toggle_gradcam`, `generate_report`.
- [ ] Frontend ve backend bu sözleşmeye göre bağımsız geliştirilir.

## W2. Model paketleme
- [ ] Model klasörü: final_v1 ağırlıkları + `config.json` (mimari, görüntü boyutu, sınıf listesi, T, melanom eşiği, sürüm, Grad-CAM hedef katmanı).
- [ ] Hugging Face Hub'a `v1` etiketiyle yüklenir.
- [ ] Ön işleme (Gray-World, normalizasyon) `src/` içinden import edilir, kopyalanmaz.
- [ ] Model değiştirmek sadece config/ağırlık değişikliği olmalı, kod değişikliği değil.

## W3. Backend (FastAPI)
- [ ] `/predict` uç noktası, sözleşmeye uygun.
- [ ] Grad-CAM (ensemble ise: tek model mi, ortalama mı, kararı yazılır).
- [ ] "Emin değilim" kuralı: en yüksek olasılık eşiğin altındaysa uyarı. Eşik val/OOF'ta belirlenir.
- [ ] Şablon özet metni backend'de üretilir (sabit cümle + CNN sayıları).
- [ ] Analiz sonuçları `analysis_id` ile **bellekte** kısa süre tutulur (ör. 30 dk), sonra silinir. Veritabanı yok.
- [ ] Rapor taslağı uç noktası (PDF): mevcut analizden üretilir, hiçbir şey kaydedilmez.
- [ ] Gizlilik: görüntüler **diske kaydedilmez**, işlenip bellekten silinir; EXIF temizlenir; HTTPS.
- [ ] Basit güvenlik: dosya boyutu/türü kontrolü, CORS sadece kendi frontend adresine açık, istek sınırı (rate limit).
- [ ] Tahmin süresi ölçülür (ensemble CPU'da yavaşsa raporlanır).

## W4. Frontend
- [ ] Teknoloji: [React / Next.js — kullanıcı onaylar].
- [ ] W0'daki mockup ve tasarım kurallarına göre.
- [ ] Sabit uyarılar: "Araştırma amaçlıdır, tanı aracı değildir. Gerçek hasta görüntüsü yüklemeyin."
- [ ] Model sürümü ve test metrikleri "Model hakkında" bölümünde.
- [ ] Telefon genişliğinde de kullanılabilir (sütunlar alt alta).
- [ ] **Yerleşim (mockup'a göre):** sol sütun örnek vakalar + oturum bilgisi; orta sütun analiz başlığı, görüntü + Grad-CAM, en olası 3 sınıf paneli, metrik kartları (risk seviyesi, melanom olasılığı + eşik çizgisi, modelin kararlılığı); sağ sütun asistan paneli + model performans tablosu.
- [ ] **Asistan v1 (LLM'siz):** şablon özet mesajı + öneri butonları. Butonlar sabit cevaplar ve sabit eylemler (mod değiştir, Grad-CAM aç/kapat, rapor taslağı) tetikler. Serbest metin kutusu yok.

## W5. Canlıya alma
- [ ] **İlk iş: barındırma seçeneklerini güncel olarak kontrol et.** Temmuz 2026'dan beri kullanıcı raporlarına göre Hugging Face'te Gradio ve Docker Space'ler ücretsiz CPU'da PRO istiyor (17 Kasım'da Gradio Space HTTP 402 verdi). Sadece statik Space'ler ücretsiz.
- [ ] **Backend seçenekleri** (karar kullanıcıda):
  - HF PRO (ücretli, en kolay).
  - Başka bir servis (Render, Google Cloud Run vb.): ücretsiz planın RAM sınırı, uyku/soğuk başlatma ve kart şartı kontrol edilir.
  - Modeli ONNX'e çevirip ONNX Runtime ile çalıştırmak: PyTorch'a göre daha az RAM, küçük ücretsiz planlara sığabilir. Çıktıların PyTorch ile aynı olduğu (tolerans içinde) doğrulanır.
  - Tarayıcıda ONNX (backend'siz): ücretsiz statik barındırma; ama Grad-CAM zorlaşır ve W6 için yine backend gerekir.
- [ ] Seçimden önce backend'in bellek kullanımı ve tahmin süresi ölçülür.
- [ ] **Frontend:** Vercel veya Netlify.
- [ ] Ortam değişkenleri (API adresi, varsa anahtarlar) repoya girmez.
- [ ] README'ye canlı demo linki ve ekran görüntüleri eklenir. `web-v1` etiketi.

**✅ W (v1) bitti sayılır:** Canlı link çalışıyor, analiz ekranı + LLM'siz asistan hazır.

## W6. Asistan v2: LLM + RAG (isteğe bağlı)
- [ ] Akış: frontend → `POST /chat {analysis_id, mesaj, geçmiş}` → backend sonuçları bellekten alır → RAG ile 3–5 ilgili paragraf bulunur → LLM'e kurallar + sonuçlar + paragraflar + soru gönderilir → cevap kaynaklarıyla döner.
- [ ] Ekranı değiştiren istekler LLM araçlarıyla (tool calling) yapılır: `set_mode`, `toggle_gradcam`, `generate_report`.
- [ ] "Ayrıntılı açıklama iste" butonu eklenir (otomatik değil, tek tıkla).
- [ ] LLM'e **görüntü gönderilmez**, sadece CNN'in yapılandırılmış çıktısı gönderilir.
- [ ] Kartlardaki sayılar her zaman CNN'den gelir; LLM'in cevabındaki sayılar backend'in verdiği sayılarla aynı olmalı.
- [ ] LLM API anahtarı sadece backend'de durur, frontend'e asla gitmez.
- [ ] Bilgi tabanı: açık erişimli dermoskopi ve klinik kılavuz kaynakları, lisansları kontrol edilir.
- [ ] Kurallar: sayıları değiştirmez, kendi teşhisini koymaz, her iddia için kaynak gösterir, kaynakta yoksa "kaynağım yok" der.
- [ ] Değerlendirme: küçük bir soru seti, cevapların kaynaklara sadakati ölçülür (örn. RAGAS).
- [ ] Serbest metin kutusu bu aşamada açılır.

---

# AŞAMA R: Rapor (en son iş)

> Model sayıları C5'ten beri sabit. Rapor en sona bırakıldı, böylece web uygulaması (canlı demo linki, ekran görüntüleri, Grad-CAM örnekleri) da rapora girer.

- [ ] Ayrı rapor promptu kullanılır (outline önce, her sayının yanında kaynak dosya, yeni ölçüm yok).
- [ ] İçerik: problem, veri, yöntem, deney süreci, v1 / v2 sonuçları, dış test analizi, web uygulaması (mimari, ekranlar, gizlilik kararları), sınırlamalar, gelecek çalışmalar.
- [ ] Yeni eğitim gerektirmeyen ek analizler (isteğe bağlı, `IDEAS.md`'den en fazla 2–3 tane): yanlış tahmin galerisi, Grad-CAM örnekleri, kalibrasyon diyagramı, ConvNeXt-Small eğitim eğrileri karşılaştırması.

**✅ R aşaması bitti sayılır:** Rapor teslim formatında hazır. Proje biter.

---

## Proje sonunda gözden geçirilecek
Bu sürümdeki web ve asistan kararları başlangıç tasarımıdır. Proje sonunda kullanıcı beğenmediği parçaları çıkarabilir veya yeni parçalar ekleyebilir; değişiklikler bu dosyaya işlenir.

---

## Kalıcı olarak YAPILMAYACAKLAR (kanıt aleyhte)
TTA, metadata füzyonu, test çözünürlüğünü artırmak, ConvNeXt-Small ve üstü (8 bin görüntüde kazanç yok), under/oversampling, augmentation v2, dermatolojiye özel / web ölçekli ön eğitim (C3 gerekçesi).

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
