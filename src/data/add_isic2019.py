"""HAM10000 manifest'ine ISIC 2019 egitim setindeki ek goruntuleri SADECE train split'ine ekler.

Kararlar (2026-10-16):
- Validation ve test HAM10000'den kalir: eski modellerle karsilastirma ayni sinavda yapilir.
- ISIC 2019, HAM10000'in 10015 goruntusunu zaten iceriyor; onlar cikarilir (bizde zaten var,
  bir kismi val/test'te - eklenseydi leakage olurdu).
- AK -> akiec. SCC cikarilir: ISIC 2019 HAM'in akiec'ini AK + SCC (Bowen) diye bolmus; HAM disi
  SCC'ler bizim 7 sinifin hicbirine uymayan invaziv SCC de icerebilir ve etiketler bunlari ayirmiyor.
- Dis test seti (external_test/) ile cakisma varsa betik durur.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.external_test import assert_no_external_test_overlap

LABEL_MAP = {'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc', 'AK': 'akiec', 'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc'}


def add_isic2019(manifest: str, ground_truth: str, metadata: str, images_dir: str, out: str) -> None:
    base = pd.read_csv(manifest)
    truth = pd.read_csv(ground_truth)
    truth['label19'] = truth.drop(columns='image').idxmax(axis=1)
    meta = pd.read_csv(metadata).set_index('image')
    # Bazi eski arsiv goruntuleri "_downsampled" ekiyle geliyor; kimlik karsilastirmasi eksiz yapilir.
    truth['image_id'] = truth['image'].str.replace('_downsampled', '', regex=False)

    in_ham = truth['image_id'].isin(set(base['image_id']))
    is_scc = truth['label19'] == 'SCC'
    extra = truth[~in_ham & ~is_scc & truth['label19'].isin(LABEL_MAP)].copy()
    assert_no_external_test_overlap(extra['image_id'], 'ISIC 2019 ek verisi')

    images = Path(images_dir)
    extra['image_path'] = [str(images / f'{name}.jpg') for name in extra['image']]
    missing = [p for p in extra['image_path'] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} ISIC 2019 goruntusu bulunamadi (or. {missing[:2]})')

    lesion = meta['lesion_id'].reindex(extra['image']).fillna('').to_numpy()
    # Lezyon kimligi olmayanlar icin goruntu kimligi: her biri ayri lezyon sayilir.
    extra['lesion_id'] = [lid if lid else f'ISIC2019_{iid}' for lid, iid in zip(lesion, extra['image_id'])]
    extra['source'] = ['isic2019_' + (lid.split('_')[0].lower() if '_' in lid else 'archive') for lid in lesion]
    extra['label'] = extra['label19'].map(LABEL_MAP)
    extra['dx'] = extra['label']
    extra['split'] = 'train'

    combined = pd.concat([base, extra[base.columns]], ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f'HAM10000 manifest: {len(base)} | ISIC 2019: {len(truth)} '
          f'(HAM ile ortak {int(in_ham.sum())}, HAM disi SCC {int((~in_ham & is_scc).sum())} cikarildi)')
    print(f'Eklenen: {len(extra)} (hepsi train) | kaynaklar: {extra["source"].value_counts().to_dict()}')
    print('Yeni train sinif dagilimi:', combined[combined.split == 'train'].label.value_counts().to_dict())
    print(f'Wrote {out}')


def main():
    parser = argparse.ArgumentParser(description='ISIC 2019 ek goruntulerini manifest\'in train split\'ine ekle.')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--out', default='data/processed/manifest_isic2019.csv')
    args = parser.parse_args()
    add_isic2019(args.manifest, args.ground_truth, args.metadata, args.images_dir, args.out)


if __name__ == '__main__':
    main()
