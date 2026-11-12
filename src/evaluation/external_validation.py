"""Dis dogrulama: donmus bir ensemble tarifini ISIC 2018 Task 3 resmi test setinde olcer.

Tarif ensemble_test.py ile birebir ayni (resolve_recipe); HICBIR sey bu sette ayarlanmaz:
- Varsayilan ensemble_v1: dort model, her biri HAM10000 VALIDATION setinde bulunan T ile kalibre,
  esit agirlik, iki calisma noktasi (normal / mel esigi 0.30)
- --recipe ile baska bir dondurulmus tarif (or. baselines/final_v1/recipe.json)
Kurallar icin external_test/README.md. Ilk olcum 2026-10-16'da ensemble_v1 ile yapildi; bu sete o
gun bakildigi icin sonraki her olcum "ikinci bakis"tir ve ciktida oyle etiketlenir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.external_test import EXTERNAL_TEST_LABELS
from src.data.loader import SkinCancerDataset
from src.evaluation.cache_predictions import predict_logits
from src.evaluation.ensemble_test import resolve_recipe
from src.evaluation.ensemble_val import load_split, score, softmax
from src.evaluation.evaluate import apply_threshold_override, bootstrap_ci, load_checkpoint, print_report, summarize
from src.training.train import select_device

EXPECTED_LABELS_MD5 = 'c6b7b4dd8cb825be0a9338a62c73ca9b'  # external_test/README.md'de kayitli


def main():
    parser = argparse.ArgumentParser(description='Dis dogrulama (ISIC 2018 Task 3 test seti).')
    parser.add_argument('--recipe', default=None, help='recipe.json (varsayilan: ensemble_v1)')
    parser.add_argument('--images-dir', default='data/external/isic2018_test/ISIC2018_Task3_Test_Input')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    models, temperatures, operating_points, recipe_dir = resolve_recipe(args.recipe)
    # ensemble_v1 bu setteki ilk olcumdu; ondan sonraki tarifler ayri, adi acik bir klasore yazilir.
    second_look = args.recipe is not None
    OUT_DIR = recipe_dir / ('external_isic2018_test_ikinci_bakis' if second_look else 'external_isic2018_test')

    # Kenara ayirdigimiz etiket listesinin degismedigini dogrula.
    assert hashlib.md5(EXTERNAL_TEST_LABELS.read_bytes()).hexdigest() == EXPECTED_LABELS_MD5
    labels_df = pd.read_csv(EXTERNAL_TEST_LABELS)
    manifest = labels_df.assign(
        image_path=[str(Path(args.images_dir) / f'{i}.jpg') for i in labels_df.image_id],
        split='test', lesion_id='', dx=labels_df.label, source='isic2018_test',
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / 'manifest.csv'
    manifest[['image_path', 'label', 'split', 'source', 'lesion_id', 'image_id', 'dx']].to_csv(manifest_path, index=False)

    # Kalibrasyon T'leri: HAM10000 validation setinden (ensemble_test.py ile ayni).
    _, _, labels = load_split('val', models=models)
    print('HAM validation T degerleri:', {k: round(v, 2) for k, v in temperatures.items()})

    device = select_device(args.device)
    probs, targets, single_rows = {}, None, {}
    for name, npz_path in models.items():
        checkpoint = str(Path(npz_path).with_name('best_model.pt'))
        model, ckpt_labels, image_size = load_checkpoint(checkpoint, device)
        assert ckpt_labels == labels
        from src.training.train import build_transforms
        dataset = SkinCancerDataset(str(manifest_path), split='test', transform=build_transforms(image_size))
        assert dataset.labels == labels
        model_targets, logits = predict_logits(model, DataLoader(dataset, batch_size=32, shuffle=False), device)
        targets = model_targets if targets is None else targets
        assert (targets == model_targets).all()
        probs[name] = softmax(logits / temperatures[name])
        single_rows[name] = score(targets, logits.argmax(axis=1), labels.index('mel'))
        print(f'{name}: tek model macro_f1={single_rows[name]["macro_f1"]:.4f} '
              f'bacc={single_rows[name]["balanced_accuracy"]:.4f}')
    (OUT_DIR / 'single_models.json').write_text(json.dumps(single_rows, indent=2))

    avg = np.mean([probs[n] for n in models], axis=0)
    for point, threshold in operating_points.items():
        predictions = apply_threshold_override(avg, labels.index('mel'), threshold)
        results = summarize(targets, predictions, labels, probs=avg)
        results['recipe'] = {'models': list(models), 'temperatures': temperatures, 'weights': 'esit',
                             'mel_threshold': threshold, 'dataset': 'ISIC 2018 Task 3 test (1512)',
                             'look': 'ikinci bakis (ilk: ensemble_v1, 2026-10-16)' if second_look else 'ilk olcum'}
        results['bootstrap_ci'] = bootstrap_ci(targets, predictions, labels, probs=avg)
        print(f'\n===== DIS DOGRULAMA, calisma noktasi: {point} =====')
        print_report(results)
        (OUT_DIR / f'test_metrics_{point}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
