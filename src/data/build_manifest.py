"""Build a manifest CSV for skin lesion imaging data.

The generated manifest contains image metadata and labels needed by the training
loader. It is designed for HAM10000-like datasets where image filenames are
matched against metadata columns such as image_id and dx.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_images(data_root: Path) -> List[Path]:
    return [p for p in data_root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def read_csv_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        with path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({str(k): ('' if v is None else str(v).strip()) for k, v in row.items()})
    except Exception:
        return []
    return rows


def normalize_image_key(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ''
    return Path(value).stem.lower()


def infer_mapping_columns(columns: Iterable[str]) -> Tuple[str, str]:
    columns = list(columns)
    image_order = ['image_id', 'image', 'img', 'filename', 'file_name', 'file', 'path', 'name']
    label_order = ['dx', 'diagnosis', 'label', 'class', 'target', 'category']

    image_candidate = ''
    for token in image_order:
        for c in columns:
            lower = str(c).lower()
            if lower == token or lower.endswith('_' + token) or lower.endswith(token):
                image_candidate = str(c)
                break
        if image_candidate:
            break

    label_candidate = ''
    for token in label_order:
        for c in columns:
            lower = str(c).lower()
            if lower == token or lower.endswith('_' + token) or lower.endswith(token):
                label_candidate = str(c)
                break
        if label_candidate:
            break

    if not image_candidate:
        image_candidate = str(columns[0]) if columns else ''
    if not label_candidate:
        label_candidate = ''

    return image_candidate, label_candidate


def build_metadata_lookup(csv_path: Path) -> Dict[str, dict]:
    rows = read_csv_rows(csv_path)
    if not rows:
        return {}

    image_col, label_col = infer_mapping_columns(rows[0].keys())
    meta_map: Dict[str, dict] = {}

    for row in rows:
        image_value = row.get(image_col, '')
        if not image_value:
            continue

        image_key = normalize_image_key(image_value)
        if not image_key:
            continue

        label_value = row.get(label_col, '') if label_col else ''
        record = {
            'image_id': row.get('image_id', image_value),
            'lesion_id': row.get('lesion_id', ''),
            'dx': row.get('dx', label_value),
            'label': label_value,
        }
        meta_map[image_key] = record

    return meta_map


def resolve_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def build_manifest(data_root: str, out: str):
    dr = Path(data_root)
    if not dr.exists():
        print(f'Error: data root {dr} does not exist.', file=sys.stderr)
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[2]
    images = find_images(dr)
    output_candidate = Path(out).resolve()
    csv_files = sorted({
        p for p in dr.rglob('*.csv')
        if p.is_file()
        and p.stat().st_size > 0
        and p.resolve() != output_candidate
        and 'processed' not in p.parts
    })

    print(f'Found {len(images)} image files under {dr}')
    print(f'Found {len(csv_files)} CSV metadata files under {dr}')

    metadata_store: Dict[str, dict] = {}
    for csv_path in csv_files:
        mapping = build_metadata_lookup(csv_path)
        if mapping:
            metadata_store.update(mapping)
            print(f'  mapped {len(mapping)} entries from {csv_path.name}')
        else:
            print(f'  no usable mapping found in {csv_path.name}')

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image_path', 'label', 'split', 'source', 'lesion_id', 'image_id', 'dx'])

        labeled_count = 0
        for image_path in sorted(images):
            image_key = normalize_image_key(image_path.name)
            match = metadata_store.get(image_key, {})
            label = str(match.get('label', '') or match.get('dx', '')).strip()
            dx = str(match.get('dx', label)).strip()
            lesion_id = str(match.get('lesion_id', '')).strip()
            image_id = str(match.get('image_id', image_path.stem)).strip()
            source = 'ham10000' if 'ham10000' in str(image_path).lower() else 'unknown'

            writer.writerow([
                resolve_relative_path(image_path, repo_root),
                label,
                'unassigned',
                source,
                lesion_id,
                image_id,
                dx,
            ])
            if label:
                labeled_count += 1

    print(f'Wrote manifest to {out_path} ({len(images)} rows, {labeled_count} labeled)')


def main():
    parser = argparse.ArgumentParser(description='Build a manifest for skin lesion images and metadata.')
    parser.add_argument('--data-root', default='data', help='Root folder that contains raw/external/processed data.')
    parser.add_argument('--out', default='data/processed/manifest.csv', help='Output manifest CSV path.')
    args = parser.parse_args()
    build_manifest(args.data_root, args.out)


if __name__ == '__main__':
    main()