# Proje Durumu

Son güncelleme: 2026-11-17. Ayrıntılı plan: [ROADMAP.md](ROADMAP.md). Denetim raporu (2026-10-26, üzerine
yazılmaz): [docs/AUDIT_2026-10-26.md](docs/AUDIT_2026-10-26.md). Deney geçmişi: [EXPERIMENTS.md](EXPERIMENTS.md).

## Nerede kaldık

**Aşama A (v1 kapanış) neredeyse bitti.** final_v1 dondurulmuş ve bir kez ölçülmüş:
iç test F1 0.829 / BA 0.832, dış test (ISIC 2018, ikinci bakış) F1 0.795 / BA 0.796.
README, EXPERIMENTS.md ve sınırlamalar yazıldı. Temiz klonda iç test sonuçları birebir yeniden üretildi.

A'dan kalanlar (B'yi bekletmez):
- `requirements.txt` sürümleri: bir sonraki Kaggle koşusunda `pip freeze` alınacak.
- Dersin teslim beklentisi (format, tarih, demo) öğrenilecek.
- Ağırlıkların üçüncü kopyası (Hugging Face Hub): B1 ile birlikte.

## Sabit kurallar (özet; tamamı ROADMAP'te)

**v2 bitiş kuralı (hangisi önce gelirse):**
- **Hedef tuttu:** 5-fold CV'nin fold dışı (OOF) tahminlerinde macro F1 ve balanced accuracy, ikisi de baseline + 0.02. Baseline ölçülünce hedef bir kez mutlak sayıya çevrilip kilitlenir.
- **Bütçe bitti:** baseline CV dışında en fazla 3 deney.
- Koruma şartı: melanom recall baseline'ın altına düşmeyecek.

**Deney kuralı:** tek değişiklik; kabul kriteri önceden yazılır (5 fold'un en az 4'ünde daha iyi ve ortalama fark > 0.01); CV'de en iyi epoch seçilmez, son epoch alınır; sonuç ne çıkarsa kaydedilir.

**Sıra:** A (v1 kapanış) → B (arayüz, en fazla 3 oturum) → C (v2: baseline CV, temiz dış test, en fazla 3 deney, final) → D (isteğe bağlı).

## Aşama B (arayüz) durumu

- [x] B1 model paketi: `python -m src.inference.package_model` → `models/final_v1/` (git'te değil). Arayüz yalnızca
  `src.inference.predictor.Predictor`'ı kullanır. `python -m src.inference.verify_package`: val'de 992 görüntünün
  olasılıkları kayıtlı sayılarla aynı (MPS'te fark 0, CPU'da 1e-7), kararların %100'ü aynı.
- [x] B2a Grad-CAM: `Predictor.gradcam()` + `overlay()`, seed 42 üzerinde, elle yazıldı (kütüphane yok). Val'de her sınıftan
  bir görüntüde haritalar lezyonun üstünde; sınıfa göre değişiyor (mel/bkl korelasyon -0.01, mel/nv 0.53). Mac'te ~0.2 s.
- [x] B2b Gradio uygulaması (`python app.py`, yerelde): ilk 3 sınıf, karar, normal/tarama, Grad-CAM ve sınıf seçimi,
  uyarılar, metrikler. "Emin değilim" eşiği val'de τ = 0.85 (`baselines/final_v1/uncertainty.json`): vakaların %24'ü
  doktora yönlendirilir, emin olunanlarda doğruluk 0.954; ama 102 melanomdan 8'i "emin" iken kaçıyor.
- [x] Model HF Hub'da: https://huggingface.co/emreozkn/skin-lesion-convnext-v1 (`v1` etiketi, model kartı). Ağırlıkların
  üçüncü kopyası. Hub'dan indirilen model MD5 kontrolünden geçti, yereldeki paketle aynı sonucu veriyor.
- [ ] Kalıcı demo: Gradio Space ücretsiz CPU'da artık HF PRO istiyor (HTTP 402, 2026-11-17). **Karar (kullanıcı):** para
  verilmeyecek. Kalıcı ve ücretsiz bir seçenek v2'den sonra, sunumdan önce araştırılacak (ör. kendi küçük site + ücretsiz
  sunucu). Hiçbiri olmazsa sunum günü `SHARE=1 .venv/bin/python app.py` (geçici herkese açık link, 72 saat, Mac açıkken).
  Space dosyaları hazır: `hf/build_space.py`.

**Aşama B kapandı** (yerel demo çalışıyor; kalıcı link sunuma bırakıldı).

## Sıradaki tek adım

**C0a: 5-fold CV bölme dosyası.** Train+val birleştirilir (iç test dokunulmaz), `StratifiedGroupKFold` (grup `lesion_id`,
katman `label`) ile 5 fold, `splits/cv5_v1.csv`. Kaggle'a henüz gerek yok.
