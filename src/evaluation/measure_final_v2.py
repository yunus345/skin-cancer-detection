"""final_v2'yi iç testte ve ISIC 2018 dış testinde BİR KEZ ölçer ve v1 ile önceden yazılan kurala göre karşılaştırır (C5).

Hiçbir şey burada ayarlanmaz: modeller, T'ler, tarama eşiği `baselines/final_v2/recipe.json`'dan okunur (OOF'ta seçildi).
Kural `baselines/final_v2/README.md`'de: normal modda iç ve dış testte F1 ve BA'nın dördü de v1'den düşük değilse
(3 ondalık) v2 v1'in yerine geçer.

Mac yükünü sınırlamak için modeller sırayla yüklenir, batch 16. Logit'ler `test_logits.npz`'ye yazılır; metrikler
sonradan modeller yeniden çalıştırılmadan bu dosyadan üretilebilir.

    python -m src.evaluation.measure_final_v2
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.external_test import EXTERNAL_TEST_LABELS
from src.data.loader import SkinCancerDataset
from src.evaluation.cache_predictions import predict_logits
from src.evaluation.ensemble_val import score, softmax
from src.evaluation.evaluate import apply_threshold_override, bootstrap_ci, print_report, summarize
from src.evaluation.external_validation import EXPECTED_LABELS_MD5
from src.training.train import build_model, build_transforms

RECIPE_DIR = Path('baselines/final_v2')
EXTERNAL_DIR = RECIPE_DIR / 'external_isic2018_test_ucuncu_bakis'
EXTERNAL_IMAGES = Path('data/external/isic2018_test/ISIC2018_Task3_Test_Input')
V1 = {'ic_test': Path('baselines/final_v1/test_metrics_normal.json'),
      'dis_test': Path('baselines/final_v1/external_isic2018_test_ikinci_bakis/test_metrics_normal.json')}


def external_manifest() -> Path:
    # external_validation.py ile aynı manifest; etiket dosyasının değişmediği MD5 ile doğrulanır.
    assert hashlib.md5(EXTERNAL_TEST_LABELS.read_bytes()).hexdigest() == EXPECTED_LABELS_MD5
    df = pd.read_csv(EXTERNAL_TEST_LABELS)
    df = df.assign(image_path=[str(EXTERNAL_IMAGES / f'{i}.jpg') for i in df.image_id], split='test',
                   lesion_id='', dx=df.label, source='isic2018_test')
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_DIR / 'manifest.csv'
    df[['image_path', 'label', 'split', 'source', 'lesion_id', 'image_id', 'dx']].to_csv(path, index=False)
    return path


def main():
    recipe = json.loads((RECIPE_DIR / 'recipe.json').read_text())
    labels, temperatures, threshold = recipe['labels'], recipe['temperatures'], recipe['mel_threshold']
    mel = labels.index('mel')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    transform = build_transforms(recipe['image_size'])
    sets = {
        'ic_test': SkinCancerDataset('data/processed/manifest.csv', split='test', transform=transform),
        'dis_test': SkinCancerDataset(str(external_manifest()), split='test', transform=transform),
    }
    for dataset in sets.values():
        assert dataset.labels == labels

    logits_path = RECIPE_DIR / 'test_logits.npz'
    if logits_path.exists():  # yeniden çalıştırmada modelleri tekrar koşturma; aynı sayılar
        saved = dict(np.load(logits_path, allow_pickle=True))
    else:
        saved = {}
        for name, info in recipe['models'].items():
            weights = Path(info['weights'])
            assert hashlib.md5(weights.read_bytes()).hexdigest() == info['md5'], f'{name}: MD5 tutmuyor'
            ckpt = torch.load(weights, map_location='cpu', weights_only=False)
            model = build_model(len(labels), ckpt['architecture'], pretrained=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            for set_name, dataset in sets.items():
                loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
                targets, logits = predict_logits(model, loader, device)  # fp32, TTA yok
                saved[f'{set_name}_{name}'] = logits
                saved[f'{set_name}_targets'] = targets
                print(f'{name} [{set_name}] {len(targets)} görüntü', flush=True)
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()
        for set_name, dataset in sets.items():
            saved[f'{set_name}_image_ids'] = np.array([row['image_id'] for row in dataset.manifest])
        np.savez(logits_path, **saved)

    comparison = {}
    for set_name in sets:
        out_dir = RECIPE_DIR if set_name == 'ic_test' else EXTERNAL_DIR
        targets = saved[f'{set_name}_targets']
        probs = np.mean([softmax(saved[f'{set_name}_{n}'] / temperatures[n]) for n in recipe['models']], axis=0)
        single = {n: score(targets, saved[f'{set_name}_{n}'].argmax(1), mel) for n in recipe['models']}
        (out_dir / 'single_models.json').write_text(json.dumps(single, indent=2))
        for point, t in {'normal': None, f'tarama_mel_esik_{threshold:.2f}': threshold}.items():
            predictions = apply_threshold_override(probs, mel, t)
            results = summarize(targets, predictions, labels, probs=probs)
            results['recipe'] = {'name': 'final_v2', 'models': list(recipe['models']), 'temperatures': temperatures,
                                 'weights': 'esit', 'mel_threshold': t,
                                 'look': 'ic test' if set_name == 'ic_test' else 'ISIC 2018 dis test, ucuncu bakis'}
            results['bootstrap_ci'] = bootstrap_ci(targets, predictions, labels, probs=probs)
            print(f'\n===== final_v2, {set_name}, {point} =====')
            print_report(results)
            (out_dir / f'test_metrics_{point}.json').write_text(json.dumps(results, indent=2))
            if point == 'normal':
                v1 = json.loads(V1[set_name].read_text())['overall']
                for metric in ('macro_f1', 'balanced_accuracy'):
                    a, b = round(results['overall'][metric], 3), round(v1[metric], 3)
                    comparison[f'{set_name}_{metric}'] = {'v2': a, 'v1': b, 'v2_dusuk_degil': a >= b}

    replaces = all(row['v2_dusuk_degil'] for row in comparison.values())
    comparison['karar'] = 'v2 v1in yerine gecer' if replaces else 'v1 kalir'
    (RECIPE_DIR / 'comparison_v1.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    print('\n===== v1 karşılaştırması (normal mod, 3 ondalık) =====')
    for key, row in comparison.items():
        print(key, row)


if __name__ == '__main__':
    main()
