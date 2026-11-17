"""Dondurulmus bir tarifi (recipe.json) arayuzun okuyacagi model paketine cevirir.

Cikti klasoru (varsayilan models/<tarif adi>/, git'te degil):
- <uye>.pt: yalnizca agirliklar (state_dict). Egitim checkpoint'i numpy nesneleri de tasidigi icin
  sadece weights_only=False ile aciliyor, yani dosyadaki rastgele Python kodu calisabilir. Duz tensor
  sozlugu ise torch.load(weights_only=True) ile guvenli acilir; herkese acik bir uygulamada bu onemli.
- config.json: tahmini yeniden uretmek icin gereken her sey (mimari, boyut, sinif sirasi, T'ler,
  esik, MD5'ler, olculmus metrikler). Arayuz bu dosyadan baska bir yere bakmaz.

Hicbir sayi burada yeniden hesaplanmaz: T ve esik recipe.json'dan, metrikler commit'li JSON'lardan.

Calistirma: python -m src.inference.package_model
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import torch

from src.training.train import IMAGENET_MEAN, IMAGENET_STD, TIMM_MODELS

# Kaynak checkpoint'lerin beklenen MD5'leri (baselines/final_v1/README.md). Yanlis dosya paketlenmesin.
EXPECTED_SOURCE_MD5 = {
    'seed42': '7e264e571ca56ef86524cc95d27b99cf',
    'seed2': '8f0d8a42a7cdcc284a66aefeace5984e',
}
# Grad-CAM icin ConvNeXt'in son evrisim asamasi (timm adi). 384px girdide 12x12 ozellik haritasi verir.
GRADCAM_TARGET_LAYER = 'stages.3'


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def headline(metrics_path: Path) -> dict:
    """Arayuzde gosterilecek ozet: commit'li olcum dosyasindan, yeniden hesaplamadan."""
    m = json.loads(metrics_path.read_text())
    return {
        'macro_f1': round(m['overall']['macro_f1'], 3),
        'balanced_accuracy': round(m['overall']['balanced_accuracy'], 3),
        'mel_recall': round(m['clinical']['mel_recall'], 3),
        'macro_auc': round(m['auc']['macro_auc'], 3),
    }


def main():
    parser = argparse.ArgumentParser(description='recipe.json -> models/<ad>/ (agirliklar + config.json)')
    parser.add_argument('--recipe', default='baselines/final_v1/recipe.json')
    parser.add_argument('--out-dir', default=None, help='Varsayilan: models/<tarif adi>')
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()

    recipe_path = Path(args.recipe)
    recipe = json.loads(recipe_path.read_text())
    out_dir = Path(args.out_dir or f"models/{recipe['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    members, architecture, image_size, labels = [], None, None, None
    for name, predictions_path in recipe['models'].items():
        source = Path(predictions_path).parent / 'best_model.pt'
        source_md5 = md5(source)
        if name in EXPECTED_SOURCE_MD5 and source_md5 != EXPECTED_SOURCE_MD5[name]:
            raise ValueError(f'{source} MD5 {source_md5}, beklenen {EXPECTED_SOURCE_MD5[name]}')
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)

        # Butun uyeler ayni mimari, boyut ve SINIF SIRASINI paylasmali; sira farkliysa ortalama anlamsiz olur.
        member_info = (checkpoint['architecture'], checkpoint['image_size'], checkpoint['labels'])
        if architecture is None:
            architecture, image_size, labels = member_info
        elif member_info != (architecture, image_size, labels):
            raise ValueError(f'{name} digerleriyle ayni mimari/boyut/sinif sirasinda degil: {member_info}')

        weights_file = out_dir / f'{name}.pt'
        torch.save(checkpoint['model_state_dict'], weights_file)
        members.append({
            'name': name,
            'weights': weights_file.name,
            'md5': md5(weights_file),
            'temperature': recipe['temperatures'][name],
            'source_checkpoint': str(source),
            'source_md5': source_md5,
        })
        print(f'{name}: {source} -> {weights_file}')

    recipe_dir = recipe_path.parent
    config = {
        'name': recipe['name'],
        'version': args.version,
        'packaged_on': date.today().isoformat(),
        'architecture': architecture,
        'timm_name': TIMM_MODELS.get(architecture),
        'image_size': image_size,
        'labels': labels,
        'preprocessing': {
            'code': 'src.training.train.build_transforms(image_size, train=False)',
            'steps': ['RGB', 'gray_world_color_constancy', f'resize {image_size}x{image_size}', 'normalize'],
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD,
        },
        'members': members,
        'combine': 'her uye softmax(logit / T); olasiliklarin esit agirlikli ortalamasi',
        'operating_points': {
            'normal': {'rule': 'argmax'},
            'tarama': {'rule': 'class_threshold', 'class': 'mel', 'threshold': recipe['mel_threshold']},
        },
        'gradcam': {'member': members[0]['name'], 'target_layer': GRADCAM_TARGET_LAYER},
        'metrics': {
            'internal_test': {'n': 986, **headline(recipe_dir / 'test_metrics_normal.json'),
                              'note': 'gelistirme sirasinda gorulmus set, iyimser'},
            'external_test': {'n': 1512, **headline(recipe_dir / 'external_isic2018_test_ikinci_bakis' / 'test_metrics_normal.json'),
                              'note': 'ISIC 2018 Task 3, ikinci bakis'},
        },
        'recipe': str(recipe_path),
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f'Wrote {out_dir / "config.json"}')


if __name__ == '__main__':
    main()
