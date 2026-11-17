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

## Sıradaki tek adım

**B1: model paketleme.** final_v1'in iki modelini arayüzün okuyacağı biçime getirmek: `weights.pt` + `config.json`
(mimari, görüntü boyutu, sınıflar, T değerleri, melanom eşiği, sürüm).
