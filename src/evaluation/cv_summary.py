"""5-fold CV sonucunu ozetler: fold bazinda metrikler, ortalama +- std ve birlesik OOF metrikleri.

Girdi: <run_dir>/fold{k}/oof_fold{k}.npz (Kaggle ciktisi, train.py --cv-folds). Her goruntu tam bir kez,
onu hic gormemis bir model tarafindan tahmin edilmis olmali; bu burada kontrol edilir.

--lock-target: baseline icin bitis hedefini (OOF macro F1 ve BA, ikisi de + 0.02) bir kez kilitler.
Dosya varsa uzerine YAZMAZ: hedef sonuclar goruldukten sonra degistirilemez (PROJECT_STATUS, bitis kurali).

Calistirma: python -m src.evaluation.cv_summary --run-dir baselines/cv5_baseline [--lock-target]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

TARGET_MARGIN = 0.02


def metrics(targets, predictions, mel_idx) -> dict:
    return {
        'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(targets, predictions)),
        'mel_recall': float(recall_score(targets == mel_idx, predictions == mel_idx, zero_division=0)),
    }


def load_oof(run_dir: Path, cv_file: Path):
    cv = pd.read_csv(cv_file)
    folds, labels = {}, None
    for path in sorted(run_dir.glob('fold*/oof_fold*.npz')):
        data = np.load(path, allow_pickle=True)
        k = int(data['fold'])
        fold_labels = [str(x) for x in data['labels']]
        assert labels is None or fold_labels == labels, f'{path}: sinif sirasi farkli'
        labels = fold_labels
        expected = set(cv.loc[cv['fold'] == k, 'image_id'])
        ids = [str(x) for x in data['image_ids']]
        assert set(ids) == expected and len(ids) == len(expected), f'{path}: goruntuler fold {k} ile ayni degil'
        folds[k] = {'logits': data['logits'], 'targets': data['targets'], 'image_ids': ids}
    if not folds:
        raise SystemExit(f'{run_dir} altinda oof_fold*.npz yok')
    return folds, labels, sorted(int(k) for k in cv['fold'].unique())  # numpy int64 JSON'a yazilamaz


def main():
    parser = argparse.ArgumentParser(description='5-fold CV OOF ozeti')
    parser.add_argument('--run-dir', default='baselines/cv5_baseline')
    parser.add_argument('--cv-file', default='splits/cv5_v1.csv')
    parser.add_argument('--lock-target', action='store_true')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    folds, labels, all_folds = load_oof(run_dir, Path(args.cv_file))
    mel = labels.index('mel')
    missing = [k for k in all_folds if k not in folds]

    rows = {k: metrics(f['targets'], f['logits'].argmax(1), mel) for k, f in sorted(folds.items())}
    table = pd.DataFrame(rows).T
    table.index.name = 'fold'
    print(table.round(4).to_string())
    mean, std = table.mean(), table.std(ddof=1) if len(table) > 1 else table.std() * np.nan
    print('\nortalama  ' + '  '.join(f'{c} {mean[c]:.4f} +- {std[c]:.4f}' for c in table.columns))

    targets = np.concatenate([f['targets'] for f in folds.values()])
    predictions = np.concatenate([f['logits'].argmax(1) for f in folds.values()])
    pooled = metrics(targets, predictions, mel)
    print(f'birlesik OOF ({len(targets)} goruntu): ' + '  '.join(f'{k} {v:.4f}' for k, v in pooled.items()))
    print('gurultu esigi (2 x fold std, bilgi amacli): ' + '  '.join(f'{c} {2 * std[c]:.4f}' for c in table.columns))

    summary = {
        'run_dir': str(run_dir), 'folds_present': sorted(folds), 'folds_missing': missing,
        'per_fold': {str(k): v for k, v in rows.items()},
        'mean': mean.to_dict(), 'std': std.to_dict(), 'pooled_oof': pooled, 'n_images': int(len(targets)),
    }
    (run_dir / 'cv_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote {run_dir / "cv_summary.json"}')

    if missing:
        print(f'\nEksik fold: {missing}. Hedef ancak 5 fold tamamken kilitlenebilir.')
        return
    if args.lock_target:
        target_path = run_dir / 'target.json'
        if target_path.exists():
            raise SystemExit(f'{target_path} zaten var; hedef bir kez kilitlenir, degistirilmez.')
        target = {
            'locked_on': date.today().isoformat(),
            'rule': f'OOF macro F1 ve BA ikisi de baseline + {TARGET_MARGIN}; melanom recall baseline altina dusmez',
            'baseline_pooled_oof': pooled,
            'target': {'macro_f1': round(pooled['macro_f1'] + TARGET_MARGIN, 4),
                       'balanced_accuracy': round(pooled['balanced_accuracy'] + TARGET_MARGIN, 4),
                       'mel_recall_min': round(pooled['mel_recall'], 4)},
        }
        target_path.write_text(json.dumps(target, indent=2))
        print(f'\nHEDEF KILITLENDI: {target["target"]}  -> {target_path}')


if __name__ == '__main__':
    main()
