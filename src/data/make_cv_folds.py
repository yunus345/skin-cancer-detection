"""v2 icin 5-fold cross-validation bolmesi (ROADMAP C0).

Kaynak: dondurulmus v1 bolmesi (splits/ham10000_lesion_split_v1.csv), manifest degil. Train ve val
birlestirilir (9029 goruntu); test (986) hicbir fold'a girmez ve v2 boyunca dokunulmaz.

StratifiedGroupKFold:
- group = lesion_id: ayni lezyonun butun fotograflari ayni fold'a duser (aksi halde model dogrulama
  fold'undaki bir lezyonu egitimde gormus olur = veri sizintisi).
- stratify = label: her fold'da sinif oranlari benzer olur (df ve vasc gibi nadir siniflar her fold'da var).

Cikti splits/cv5_v1.csv (image_id, lesion_id, label, fold) git'e girer ve degismez: sklearn surumu
degisirse ayni kod farkli fold uretebilir, kayitli dosya uretmez.

Calistirma: python -m src.data.make_cv_folds
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SPLIT = ROOT / 'splits' / 'ham10000_lesion_split_v1.csv'
OUT = ROOT / 'splits' / 'cv5_v1.csv'
N_FOLDS = 5
SEED = 42


def main():
    split = pd.read_csv(FROZEN_SPLIT)
    pool = split[split['split'].isin(['train', 'val'])].reset_index(drop=True)
    test = split[split['split'] == 'test']

    # Stratify goruntu etiketiyle yapiliyor; bunun anlamli olmasi icin bir lezyonun butun
    # fotograflari ayni etikette olmali.
    labels_per_lesion = pool.groupby('lesion_id')['label'].nunique()
    assert (labels_per_lesion == 1).all(), 'bazi lezyonlarin fotograflari farkli etiketli'

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pool['fold'] = -1
    for fold, (_, val_idx) in enumerate(cv.split(pool, y=pool['label'], groups=pool['lesion_id'])):
        pool.loc[val_idx, 'fold'] = fold

    # Kontroller
    assert (pool['fold'] >= 0).all(), 'fold atanmamis goruntu var'
    assert (pool.groupby('lesion_id')['fold'].nunique() == 1).all(), 'bir lezyon birden fazla fold\'da'
    assert not set(pool['lesion_id']) & set(test['lesion_id']), 'test lezyonu fold\'lara girmis'
    assert not set(pool['image_id']) & set(test['image_id']), 'test goruntusu fold\'lara girmis'

    out = pool[['image_id', 'lesion_id', 'label', 'fold']].sort_values('image_id')
    out.to_csv(OUT, index=False)

    counts = pd.crosstab(pool['fold'], pool['label'], margins=True, margins_name='toplam')
    print(f'{len(pool)} goruntu, {pool["lesion_id"].nunique()} lezyon, {N_FOLDS} fold (test {len(test)} goruntu disarida)\n')
    print(counts.to_string())
    share = pd.crosstab(pool['fold'], pool['label'], normalize='index').round(3)
    print('\nSinif oranlari (her satir bir fold):')
    print(share.to_string())
    print(f'\nWrote {OUT.relative_to(ROOT)}  MD5 {hashlib.md5(OUT.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
