"""Split a manifest CSV into train/val/test partitions."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_split(
    dataframe: pd.DataFrame,
    train_size: float,
    stratify_col: str | None = None,
    random_state: int = 42,
):
    if stratify_col is not None:
        counts = dataframe[stratify_col].value_counts()
        if counts.min() >= 2:
            return train_test_split(
                dataframe,
                train_size=train_size,
                stratify=dataframe[stratify_col],
                random_state=random_state,
            )
    return train_test_split(
        dataframe,
        train_size=train_size,
        random_state=random_state,
    )


def _group_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """Split all images from one lesion into the same partition."""
    group_column = 'lesion_id'
    if group_column not in dataframe.columns:
        raise ValueError('Manifest must include lesion_id for lesion-level splitting.')

    groups = dataframe[[group_column, 'label']].drop_duplicates(group_column)
    if groups[group_column].eq('').any() or groups[group_column].isna().any():
        raise ValueError('lesion_id contains empty values; cannot guarantee leakage-free splits.')

    stratify = groups['label'] if groups['label'].value_counts().min() >= 3 else None
    train_groups, temp_groups = _safe_split(
        groups,
        train_size=train_ratio,
        stratify_col='label' if stratify is not None else None,
        random_state=random_state,
    )
    val_groups, test_groups = _safe_split(
        temp_groups,
        train_size=val_ratio / (val_ratio + test_ratio),
        stratify_col='label' if stratify is not None else None,
        random_state=random_state,
    )

    split_lookup = {
        **{group: 'train' for group in train_groups[group_column]},
        **{group: 'val' for group in val_groups[group_column]},
        **{group: 'test' for group in test_groups[group_column]},
    }
    result = dataframe.copy()
    result['split'] = result[group_column].map(split_lookup)
    if result['split'].isna().any():
        raise RuntimeError('Some lesion groups could not be assigned to a split.')
    return result


def split_manifest(
    manifest_path: str,
    output_path: str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_file}')

    df = pd.read_csv(manifest_file)
    if 'label' not in df.columns:
        raise ValueError('Manifest must include a `label` column to split data.')

    ratios = [train_ratio, val_ratio, test_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')

    output_file = Path(output_path) if output_path else manifest_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if 'split' in df.columns and set(df['split'].dropna().unique()) <= {'train', 'val', 'test'}:
        existing_splits = df[['lesion_id', 'split']].drop_duplicates('lesion_id') if 'lesion_id' in df.columns else pd.DataFrame()
        if not existing_splits.empty and existing_splits['split'].nunique() > 1:
            raise ValueError('Manifest already contains split values that may violate lesion-level grouping.')
        df.to_csv(output_file, index=False)
        return df

    if 'lesion_id' in df.columns and df['lesion_id'].astype(str).str.strip().ne('').all():
        result = _group_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    else:
        stratify_target = 'label' if len(df['label'].unique()) > 1 else None
        train_df, temp_df = _safe_split(
            df,
            train_size=train_ratio,
            stratify_col=stratify_target,
            random_state=random_state,
        )
        val_df, test_df = _safe_split(
            temp_df,
            train_size=val_ratio / (val_ratio + test_ratio),
            stratify_col=stratify_target,
            random_state=random_state,
        )
        result = pd.concat([
            train_df.assign(split='train'),
            val_df.assign(split='val'),
            test_df.assign(split='test'),
        ], ignore_index=True)

    result.to_csv(output_file, index=False)
    print(f'Wrote split manifest to {output_file}')
    print(result['split'].value_counts().to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description='Split a manifest CSV into train/val/test sets.')
    parser.add_argument('--manifest', default='data/processed/manifest.csv', help='Manifest CSV path.')
    parser.add_argument('--out', default='data/processed/manifest.csv', help='Output manifest path.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split_manifest(
        manifest_path=args.manifest,
        output_path=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )


if __name__ == '__main__':
    main()
