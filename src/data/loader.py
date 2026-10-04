"""Data loading utilities for the skin cancer project.

This module defines a lightweight manifest reader and an image dataset that can
be used with a standard PyTorch training loop.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def list_images(data_dir: str) -> List[str]:
    """Return all image paths under a directory."""
    p = Path(data_dir)
    if not p.exists():
        return []
    return [str(fp) for fp in p.rglob('*') if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS]


def read_manifest(manifest_path: str) -> List[dict]:
    """Read manifest CSV into a list of row dictionaries."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')

    rows: List[dict] = []
    with path.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: (v if v is not None else '') for k, v in row.items()})
    return rows


def resolve_image_path(image_path: str) -> str:
    """Resolve manifest-relative paths to actual filesystem paths."""
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    repo_root = Path(__file__).resolve().parents[2]
    resolved = repo_root / candidate if not candidate.is_absolute() else candidate
    if resolved.exists():
        return str(resolved)
    return str(candidate)


class SkinCancerDataset(Dataset):
    """Simple dataset wrapper for HAM10000-style image manifests."""

    def __init__(self, manifest_path: str, transform=None, target_transform=None, split: str | None = None):
        all_rows = read_manifest(manifest_path)
        self.labels = sorted({
            str(row.get('label', '')).strip()
            for row in all_rows
            if row.get('label', '').strip()
        })
        self.manifest = all_rows
        if split is not None:
            split_name = str(split).lower()
            self.manifest = [row for row in self.manifest if str(row.get('split', '')).lower() == split_name]
        self.transform = transform
        self.target_transform = target_transform
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest[idx]
        image_path = resolve_image_path(str(row.get('image_path', '')))
        img = Image.open(image_path).convert('RGB')

        label = str(row.get('label', '')).strip()
        label_idx = self.label_to_idx.get(label, 0)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0

        if self.target_transform is not None:
            label_idx = self.target_transform(label_idx)

        return img, label_idx


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='List images in a directory or inspect manifest rows')
    parser.add_argument('--data-dir', default='data/raw', help='Directory to scan for images')
    parser.add_argument('--manifest', default='data/processed/manifest.csv', help='Manifest CSV to inspect')
    args = parser.parse_args()

    imgs = list_images(args.data_dir)
    print(f'Found {len(imgs)} images in {args.data_dir}')
    for i, image in enumerate(imgs[:10], 1):
        print(f'{i}. {image}')

    manifest = read_manifest(args.manifest)
    print(f'Loaded {len(manifest)} rows from {args.manifest}')
    for row in manifest[:3]:
        print(row)
