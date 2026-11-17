`src/` içinde proje kodu bulunur. Komutlar repo kökünden `python -m src.<modül>` şeklinde çalıştırılır.

- `data/`: manifest oluşturma (`build_manifest.py`), lezyon bazlı bölme (`split_manifest.py`),
  doğrulama ve sızıntı kontrolü (`validate_manifest.py`), `SkinCancerDataset` (`loader.py`),
  dış test ve ISIC 2019 yardımcıları.
- `training/train.py`: model kurulumu (`build_model`), dönüşümler (`build_transforms`) ve iki fazlı
  eğitim döngüsü.
- `evaluation/`: metrikler (`evaluate.py`), logit önbellekleme (`cache_predictions.py`), ensemble ve
  kalibrasyon (`ensemble_val.py`), tarif dondurma (`freeze_recipe.py`), iç test (`ensemble_test.py`),
  dış test (`external_validation.py`) ve grafikler.
