"""HAM10000 manifest'ine ISIC 2019 egitim setindeki ek goruntuleri SADECE train split'ine ekler.

Kararlar (2026-10-16):
- Validation ve test HAM10000'den kalir: eski modellerle karsilastirma ayni sinavda yapilir.
- ISIC 2019, HAM10000'in 10015 goruntusunu zaten iceriyor; onlar cikarilir (bizde zaten var,
  bir kismi val/test'te - eklenseydi leakage olurdu).
- AK -> akiec. SCC cikarilir: ISIC 2019 HAM'in akiec'ini AK + SCC (Bowen) diye bolmus; HAM disi
  SCC'ler bizim 7 sinifin hicbirine uymayan invaziv SCC de icerebilir ve etiketler bunlari ayirmiyor.
- Dis test seti (external_test/) ile cakisma varsa betik durur.
- (2026-11-18, v2 C2) Kimlik karsilastirmasina ek olarak goruntu hash'i (dHash): ayni goruntunun farkli kimlikli
  kopyalari da cikarilir (HAM ile ortak olanlar ve ek verinin kendi icindeki tekrarlari).
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.external_test import assert_no_external_test_overlap

def dhash(path: str, size: int = 16) -> str:
    """Fark hash'i: gri, (size+1) x size'a kucultulmus goruntude her piksel sag komsusundan parlak mi?
    256 bitlik bu parmak izi yeniden sikistirma ve boyut degisimine dayanikli (MD5 degil): ayni goruntunun
    farkli kaydedilmis kopyasi da ayni hash'i verir."""
    with Image.open(path) as image:
        image.draft('L', (4 * size, 4 * size))  # buyuk JPEG'leri tam cozmeden kucult (hiz)
        gray = image.convert('L').resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    return np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes().hex()


def hash_all(paths, workers: int = 4) -> list[str]:
    with Pool(workers) as pool:
        return pool.map(dhash, list(paths), chunksize=64)


# Hamming mesafesi (256 bitten kac tanesi farkli) bu esige esit ya da kucukse "ayni goruntu". Olculdu (2026-11-18,
# HAM10000): yeniden boyutlandirilip sikistirilmis kopya 3-10, ayni lezyonun baska fotografi en az 10-12, farkli
# lezyonlarin en yakin komsusu en az 13 (medyan 60). Bilerek comert: kopya kacirmak sizinti, farkli bir ek goruntuyu
# atmak ise kucuk bir kayip.
DUPLICATE_MAX_BITS = 12


def _bit_matrix(hashes) -> np.ndarray:
    return np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in hashes]).astype(np.float32)


def near_duplicate_mask(query_hashes, reference_hashes, chunk: int = 1000, exclude_earlier_self: bool = False):
    """query'deki her hash icin, reference'ta Hamming mesafesi <= DUPLICATE_MAX_BITS olan biri var mi.
    exclude_earlier_self: query ile reference ayni listeyse, yalnizca kendinden ONCEKI elemanlara bakar
    (tekrarlarin ilki tutulur)."""
    q, r = _bit_matrix(query_hashes), _bit_matrix(reference_hashes)
    found = np.zeros(len(q), dtype=bool)
    for start in range(0, len(q), chunk):
        block = q[start:start + chunk]
        dist = block @ (1 - r).T + (1 - block) @ r.T  # (blok, referans) Hamming mesafeleri
        if exclude_earlier_self:
            rows = np.arange(start, start + len(block))[:, None]
            dist[np.arange(len(r))[None, :] >= rows] = 999
        found[start:start + len(block)] = (dist <= DUPLICATE_MAX_BITS).any(axis=1)
    return found


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

    # Hash ile tekrar ayiklama: HAM'deki bir goruntunun kopyasi olan ek goruntuler (farkli kimlikle bile) val/test
    # fold'larina sizinti demek; ek verinin kendi icindeki kopyalari da bir kez tutulur.
    ham_hashes = hash_all(base['image_path'])
    extra['dhash'] = hash_all(extra['image_path'])
    copy_of_ham = pd.Series(near_duplicate_mask(extra['dhash'], ham_hashes), index=extra.index)
    repeated = pd.Series(near_duplicate_mask(extra['dhash'], extra['dhash'], exclude_earlier_self=True),
                         index=extra.index) & ~copy_of_ham
    print(f'Hash kontrolu (Hamming <= {DUPLICATE_MAX_BITS} bit): HAM kopyasi {int(copy_of_ham.sum())}, '
          f'ek veri icinde tekrar {int(repeated.sum())} -> cikarildi')
    extra = extra[~copy_of_ham & ~repeated].copy()

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
