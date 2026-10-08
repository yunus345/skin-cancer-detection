"""Validate image paths, labels, metadata and split leakage in a manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


def validate_manifest(manifest_path: str, check_images: bool = True) -> bool:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_file}')

    df = pd.read_csv(manifest_file).fillna('')
    required = {'image_path', 'label', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Manifest is missing columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Manifest is empty.')

    errors: list[str] = []
    repo_root = Path(__file__).resolve().parents[2]

    if df['image_path'].duplicated().any():
        errors.append(f'{int(df["image_path"].duplicated().sum())} duplicate image paths found')
    if 'image_id' in df.columns:
        nonempty_ids = df['image_id'].astype(str).str.strip()
        nonempty_ids = nonempty_ids[nonempty_ids != '']
        if nonempty_ids.duplicated().any():
            errors.append(
                f'{int(nonempty_ids.duplicated().sum())} duplicate image_id values found '
                '(same image likely present under multiple paths)'
            )
    if df['label'].astype(str).str.strip().eq('').any():
        errors.append(f'{int(df["label"].astype(str).str.strip().eq("").sum())} rows have empty labels')
    if not set(df['split']).issubset({'train', 'val', 'test'}):
        errors.append('split contains values other than train, val or test')

    if 'lesion_id' in df.columns:
        split_counts = df.groupby('lesion_id')['split'].nunique()
        leaked = split_counts[split_counts > 1]
        if not leaked.empty:
            errors.append(f'{len(leaked)} lesion_id values appear in multiple splits')

    if check_images:
        missing_paths = []
        invalid_images = []
        for image_path in df['image_path']:
            path = Path(str(image_path))
            path = path if path.is_absolute() else repo_root / path
            if not path.exists():
                missing_paths.append(str(path))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                invalid_images.append(str(path))
        if missing_paths:
            errors.append(f'{len(missing_paths)} image paths do not exist')
        if invalid_images:
            errors.append(f'{len(invalid_images)} images could not be opened')

    print(f'Rows: {len(df)}')
    print(f'Labels: {dict(Counter(df["label"]))}')
    print(f'Splits: {dict(Counter(df["split"]))}')
    if errors:
        print('Manifest validation failed:')
        for error in errors:
            print(f'- {error}')
        return False

    print('Manifest validation passed.')
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate a generated manifest.')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--skip-image-check', action='store_true')
    args = parser.parse_args()
    if not validate_manifest(args.manifest, check_images=not args.skip_image_check):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
