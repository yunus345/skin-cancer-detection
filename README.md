# Skin Cancer Detection

HAM10000 veri setiyle dermoskopi görüntülerinden 7 sınıflı cilt lezyonu sınıflandırma
(akiec, bcc, bkl, df, mel, nv, vasc). PyTorch, transfer learning. Öğrenme amaçlı bir proje.

**Güncel durum, sonuçlar ve sıradaki adım:** [PROJECT_STATUS.md](PROJECT_STATUS.md)

## En iyi sonuç (test seti, 986 görüntü)

Dört modelin (EfficientNet-B0, ConvNeXt-Tiny, EfficientNetV2-S, Swin V2-Tiny) kalibre edilmiş
ensemble'ı, ayrıntılar [baselines/ensemble_v1/](baselines/ensemble_v1/):

| | macro F1 | balanced accuracy | melanom recall |
|---|---|---|---|
| Normal karar | 0.783 | 0.809 | 0.692 |
| Tarama modu (melanom eşiği 0.30) | 0.781 | 0.823 | 0.888 |

## Dizin yapısı

- `src/data/`: manifest oluşturma (tekrar eden görüntüleri temizler), lezyon bazlı
  train/val/test bölme, doğrulama, `SkinCancerDataset`
- `src/training/train.py`: iki fazlı transfer learning (önce backbone donuk, sonra tüm model)
- `src/evaluation/`: test değerlendirmesi, grafikler, logit önbellekleme ve ensemble
  (`ensemble_val.py`: validation'da seçim, `ensemble_test.py`: test setinde tek ölçüm)
- `baselines/<deneme>/`: her eğitilmiş model ve denemenin sonuçları ve README'si
  (50MB üstü ağırlıklar git'te değil; yerel disk + Google Drive `archive/`)
- `kaggle/`: Kaggle'da arka planda eğitim notebook'u (asıl eğitim ortamı)
- `notebooks/00_project_setup.ipynb`: Colab ile eğitim (yedek)
- `data/`: veri (git'te değil); `raw/` HAM10000, `processed/manifest.csv`

## Kurulum (yerel, Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# HAM10000'i data/raw/ altına koy, sonra:
python -m src.data.build_manifest --data-root data --out data/processed/manifest.csv
python -m src.data.split_manifest --manifest data/processed/manifest.csv --out data/processed/manifest.csv
python -m src.data.validate_manifest --manifest data/processed/manifest.csv
```

Eğitim GPU ister; Kaggle döngüsü için [PROJECT_STATUS.md](PROJECT_STATUS.md)'ye bak.
Ensemble ve değerlendirme Mac'te çalışır:

```bash
python -m src.evaluation.ensemble_val --calibrate   # validation'da kombinasyonlar
python -m src.evaluation.ensemble_test              # kesin tarifi test setinde bir kez ölç
```
