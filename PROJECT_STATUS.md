# Proje Durumu

Son güncelleme: 2026-11-17. Ayrıntılı plan: [ROADMAP.md](ROADMAP.md). Denetim raporu (2026-10-26, üzerine
yazılmaz): [docs/AUDIT_2026-10-26.md](docs/AUDIT_2026-10-26.md). Deney geçmişi: [EXPERIMENTS.md](EXPERIMENTS.md).

## Nerede kaldık

**Aşama A (v1 kapanış) neredeyse bitti.** final_v1 dondurulmuş ve bir kez ölçülmüş:
iç test F1 0.829 / BA 0.832, dış test (ISIC 2018, ikinci bakış) F1 0.795 / BA 0.796.
README, EXPERIMENTS.md ve sınırlamalar yazıldı. Temiz klonda iç test sonuçları birebir yeniden üretildi.

`baselines/` düzenlendi (2026-11-17): final tarif + iki üye üstte, diğer 10 deney `baselines/_archive/`'de; bağımlı betiklerin
çıktıları taşıma öncesiyle birebir aynı.

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

## Aşama C (v2) durumu

- [x] C0a: `splits/cv5_v1.csv` (MD5 `e8e35f94bd974356390d3f2aba045587`), `python -m src.data.make_cv_folds`. Train+val = 9029 görüntü,
  6723 lezyon, 5 fold (~1806 görüntü); lezyon iki fold'da yok, test (986) dışarıda; sınıf oranları fold'lar arasında ±0.001.
- [x] C0b: `train.py --cv-folds splits/cv5_v1.csv --fold k`. CV modunda `best_model.pt` yazılmaz; `last_model.pt` (son epoch) ve
  `oof_fold{k}.npz` (fold k'nin fp32 logit'leri) yazılır; test tahmin edilmez. Mac'te küçük denemede (5×40 görüntü, 1+1 epoch)
  OOF kimlikleri = fold k, test görüntüsü 0, OOF F1 = son epoch val F1.
- [ ] C0c: Kaggle koşu 1 (fold 0, 1, 2) başladı 2026-11-17 17:04, Version 11 (https://www.kaggle.com/code/yunusemreozkan/skin-cancer-train).
  Notebook artık test setini ölçmüyor; `pip freeze` alıyor. Koşu 2: fold 3, 4 (`FOLDS = [3, 4]`).
- [x] C0d hazırlığı: `python -m src.evaluation.cv_summary --run-dir baselines/cv5_baseline [--lock-target]` (sahte veriyle test edildi;
  hedef yalnızca 5 fold tamamken ve bir kez kilitlenir).

## Sıradaki tek adım

**C0c devam:** koşu 1 bitince çıktıyı `baselines/cv5_baseline/` altına indir (önce küçük dosyalar), fold 0-2 özetine bak, sonra koşu 2'yi
(`FOLDS = [3, 4]`) gönder.
