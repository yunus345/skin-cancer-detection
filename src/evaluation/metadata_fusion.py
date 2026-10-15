"""Metadata (yas, cinsiyet, lezyon yeri) ile goruntu ensemble'ini birlestirir.

Iki ayri kanit carpiliyor:
- Goruntu: kalibre 4'lu ensemble'in olasiligi p_img(sinif | goruntu)
- Metadata: sadece yas/cinsiyet/yere bakan lojistik regresyonun "bu hastada bu sinif ortalamadan
  ne kadar daha olasi" orani, p_meta(sinif | metadata) / p_egitim(sinif)

    skor = log p_img + alfa * (log p_meta - log p_egitim)

alfa = 0 bugunku ensemble. Metadata modeli EGITIM setinde egitilir (goruntu gormedigi icin
ezberleme sorunu yok); validation'da sadece alfa secilir.

dx_type KULLANILMAZ: teshisin nasil kondugunu (biyopsi/takip) soyluyor, yani cevabi neredeyse
veriyor ve tahmin aninda elde olmayacak bir bilgi (data leakage).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.ensemble_val import fit_temperature, load_split, score, softmax

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def load_metadata(split: str, image_ids) -> pd.DataFrame:
    metadata = pd.read_csv('data/raw/HAM10000_metadata.csv').set_index('image_id')
    frame = metadata.loc[list(image_ids), ['age', 'sex', 'localization', 'dx']]
    assert len(frame) == len(image_ids), f'{split}: metadata eslesmedi'
    return frame


def build_metadata_model(features):
    numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
    categorical = OneHotEncoder(handle_unknown='ignore')
    return make_pipeline(
        ColumnTransformer([('age', numeric, ['age']), ('cat', categorical, features)]),
        LogisticRegression(max_iter=2000),
    )


def image_ensemble_probs(split: str, temperatures=None):
    logits, targets, labels = load_split(split)
    if temperatures is None:
        temperatures = {name: fit_temperature(l, targets) for name, l in logits.items()}
    probs = np.mean([softmax(logits[n] / temperatures[n]) for n in logits], axis=0)
    return probs, targets, labels, temperatures


def main():
    parser = argparse.ArgumentParser(description='Metadata + goruntu ensemble birlestirmesi (validation).')
    parser.parse_args()

    val_img, val_targets, labels, _ = image_ensemble_probs('val')
    val_ids = np.load('baselines/convnext_tiny_v1/predictions.npz')['val_image_ids']
    train_ids = pd.read_csv('data/processed/manifest.csv').query("split == 'train'").image_id
    train_meta, val_meta = load_metadata('train', train_ids), load_metadata('val', val_ids)
    assert (val_meta.dx.map(labels.index).to_numpy() == val_targets).all()
    mel_idx = labels.index('mel')

    # Yer bilgisi kaynak izi tasiyabilir ("trunk" etiketi bir klinige ozgu olabilir); onsuz versiyon
    # dis veride daha guvenilir olabilir, iki versiyonu da olcuyoruz.
    variants = {'yas+cinsiyet+yer': ['sex', 'localization'], 'yas+cinsiyet': ['sex']}
    prior = train_meta.dx.value_counts(normalize=True).reindex(labels).to_numpy()
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, len(val_targets), len(val_targets)) for _ in range(1000)]
    base_pred = val_img.argmax(axis=1)
    base_f1 = np.array([f1_score(val_targets[i], base_pred[i], average='macro') for i in samples])

    for variant, features in variants.items():
        model = build_metadata_model(features).fit(train_meta, train_meta.dx)
        assert list(model.classes_) == labels
        p_meta = model.predict_proba(val_meta)
        print(f'\n=== Metadata: {variant} ===')
        print(f'Sadece metadata (validation): log loss {log_loss(val_targets, p_meta, labels=range(len(labels))):.3f} '
              f'(hicbir sey bilmeyen tahmin: {log_loss(val_targets, np.tile(prior, (len(val_targets), 1))):.3f})')
        print(f'{"alfa":>5} {"F1":>6} {"fark":>7} {"%95 aralik":>17} {"alfa=0i gecme":>14} {"BAcc":>6} {"melR":>6} {"melP":>6}')
        for alpha in ALPHAS:
            fused = np.log(val_img + 1e-12) + alpha * (np.log(p_meta + 1e-12) - np.log(prior))
            pred = fused.argmax(axis=1)
            s = score(val_targets, pred, mel_idx)
            diff = np.array([f1_score(val_targets[i], pred[i], average='macro') for i in samples]) - base_f1
            low, high = np.percentile(diff, [2.5, 97.5])
            print(f'{alpha:5.2f} {s["macro_f1"]:6.3f} {diff.mean():+7.3f} ({low:+.3f}, {high:+.3f}) '
                  f'{(diff > 0).mean() * 100:13.0f}% {s["balanced_accuracy"]:6.3f} {s["mel_recall"]:6.3f} {s["mel_precision"]:6.3f}')


if __name__ == '__main__':
    main()
