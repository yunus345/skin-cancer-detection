"""Dis test seti (ISIC 2018 Task 3 resmi test seti, 1512 goruntu) icin koruma.

Bu set projenin en sonunda, her sey kesinlestikten sonra BIR KEZ kullanilacak (bkz.
external_test/README.md). Egitime, validation'a ya da kendi test setimize giren hicbir
goruntu bu listede olmamali; ek veri eklerken cakisma olursa kod durmali.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXTERNAL_TEST_LABELS = Path(__file__).resolve().parents[2] / 'external_test' / 'isic2018_task3_test_labels.csv'


def load_external_test_ids() -> set[str]:
    return set(pd.read_csv(EXTERNAL_TEST_LABELS)['image_id'])


def assert_no_external_test_overlap(image_ids, context: str) -> None:
    overlap = load_external_test_ids() & {str(i).strip() for i in image_ids}
    if overlap:
        raise ValueError(
            f'{context}: {len(overlap)} goruntu dis test setinde de var (or. {sorted(overlap)[:3]}). '
            'Dis test seti egitim/degerlendirme verisine KARISAMAZ - bu goruntuleri cikar.'
        )
