"""Ensemble kombinasyonlarini VALIDATION setinde karsilastirir (test setine bakmadan).

cache_predictions.py'nin kaydettigi logit'leri okur, her kombinasyonda modellerin softmax
olasiliklarini esit agirlikla ortalar ve metrikleri hesaplar. Model secimi burada yapilir;
test seti sadece tarif kesinlestikten sonra bir kez kullanilir.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
from src.evaluation.evaluate import apply_threshold_override
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

MODELS = {
    'B0': 'baselines/baseline_v1/predictions.npz',
    'ConvNeXt': 'baselines/convnext_tiny_v1/predictions.npz',
    'V2-S': 'baselines/efficientnet_v2_s_v1/predictions.npz',
    'Swin': 'baselines/swin_v2_t_v1/predictions.npz',
}


# Mevcut ensemble'a aday yeni modeller (MODELS'e ancak validation karsilastirmasini gecerse girer).
CANDIDATES = {
    'ConvNeXt384': 'baselines/convnext_tiny_384_v1/predictions.npz',
    'ConvNeXtISIC': 'baselines/convnext_tiny_isic2019_v1/predictions.npz',
    'ConvNeXtTarif2': 'baselines/convnext_tiny_recipe2_v1/predictions.npz',
    'ConvNeXt22k384': 'baselines/convnext_tiny_in22k_384_v1/predictions.npz',
}


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, targets, temperature=1.0):
    """Ortalama negatif log-olabilirlik: emin olup yanilan tahminleri en agir cezalandiran loss."""
    probs = softmax(logits / temperature)
    return -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-12))


def fit_temperature(logits, targets):
    # Tek bir sayiyi ariyoruz; aralik taramasi hem yeterince hizli hem de ne yaptigimizi acikca gosteriyor.
    # T > 1 olasiliklari yumusatir (asiri ozguven), T < 1 keskinlestirir. argmax'i (karari) degistirmez.
    candidates = np.arange(0.5, 5.0001, 0.01)
    losses = [nll(logits, targets, t) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def load_split(split: str, tta: bool = False, models: dict | None = None):
    # tta=True: cache_predictions.py --tta ciktisi (4 cevirme bakisinin logit ortalamasi).
    data = {name: np.load(path.replace('predictions.npz', 'predictions_tta.npz') if tta else path)
            for name, path in (models or MODELS).items()}
    reference = next(iter(data.values()))
    for name, d in data.items():
        if not (d[f'{split}_image_ids'] == reference[f'{split}_image_ids']).all():
            raise ValueError(f'{name}: goruntu sirasi diger modellerden farkli')
    labels = list(reference['labels'])
    logits = {name: d[f'{split}_logits'] for name, d in data.items()}
    return logits, reference[f'{split}_targets'], labels


def score(targets, predictions, mel_idx):
    mel_true = targets == mel_idx
    mel_pred = predictions == mel_idx
    return {
        'macro_f1': f1_score(targets, predictions, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),
        'macro_precision': precision_score(targets, predictions, average='macro', zero_division=0),
        'mel_recall': recall_score(mel_true, mel_pred),
        'mel_precision': precision_score(mel_true, mel_pred, zero_division=0),
    }


def weight_study(probs, targets, mel_idx, n_bootstrap=1000, seed=0):
    """Kalibre 4'lu ensemble'a gore: her modeli cikarinca ve B0'in agirligini degistirince ne oluyor?

    Bilerek az secenek deneniyor (kazananin laneti); her fark bootstrap ile olculuyor.
    """
    names = list(probs)
    reference = {name: 1.0 for name in names}
    options = [(f'{name} cikarilinca', {**reference, name: 0.0}) for name in names]
    options += [(f'B0 agirligi {w}', {**reference, 'B0': w}) for w in (0.25, 0.5, 1.5)]

    def predict(weights):
        total = sum(weights.values())
        return (sum(weights[n] * probs[n] for n in names) / total).argmax(axis=1)

    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(targets), len(targets)) for _ in range(n_bootstrap)]
    ref_pred = predict(reference)
    ref_f1 = np.array([f1_score(targets[i], ref_pred[i], average='macro') for i in samples])
    ref_score = score(targets, ref_pred, mel_idx)
    print(f'Referans: 4 model esit agirlik, F1={ref_score["macro_f1"]:.3f} melR={ref_score["mel_recall"]:.3f}\n')
    print(f'{"Secenek":22} {"F1":>6} {"fark":>7} {"%95 aralik":>17} {"referansi gecme":>16} {"melR":>6}')
    for label, weights in options:
        pred = predict(weights)
        s = score(targets, pred, mel_idx)
        diff = np.array([f1_score(targets[i], pred[i], average='macro') for i in samples]) - ref_f1
        low, high = np.percentile(diff, [2.5, 97.5])
        print(f'{label:22} {s["macro_f1"]:6.3f} {s["macro_f1"] - ref_score["macro_f1"]:+7.3f} '
              f'({low:+.3f}, {high:+.3f}) {(diff > 0).mean() * 100:15.0f}% {s["mel_recall"]:6.3f}')


def mel_threshold_table(probs, targets, labels):
    """Kalibre 4'lu ensemble'da: 'mel olasiligi esigi gecerse mel de' kuralinin denge tablosu."""
    mel_idx, nv_idx = labels.index('mel'), labels.index('nv')
    avg = np.mean(list(probs.values()), axis=0)
    print(f'{"Esik":>8} {"F1":>6} {"melR":>6} {"melP":>6} {"yakalanan mel":>14} {"mel->nv":>8} {"mel alarmi":>11} {"yanlis alarm":>13}')
    for threshold in [None, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
        pred = apply_threshold_override(avg, mel_idx, threshold)
        s = score(targets, pred, mel_idx)
        is_mel, said_mel = targets == mel_idx, pred == mel_idx
        label = 'argmax' if threshold is None else f'{threshold:.2f}'
        print(f'{label:>8} {s["macro_f1"]:6.3f} {s["mel_recall"]:6.3f} {s["mel_precision"]:6.3f} '
              f'{f"{(is_mel & said_mel).sum()}/{is_mel.sum()}":>14} {(is_mel & (pred == nv_idx)).sum():8d} '
              f'{said_mel.sum():11d} {(said_mel & ~is_mel).sum():13d}')


def tta_study(n_bootstrap=1000, seed=0):
    """Kalibre 4'lu ensemble ve tek modeller: TTA'li ve TTA'siz (validation, bootstrap)."""
    logits, targets, labels = load_split('val')
    tta_logits, tta_targets, _ = load_split('val', tta=True)
    assert (targets == tta_targets).all()

    def calibrated(l):
        # Her durum kendi T'sini validation'da bulur (TTA olasiliklari biraz yumusatir).
        return {name: softmax(x / fit_temperature(x, targets)) for name, x in l.items()}

    plain, with_tta = calibrated(logits), calibrated(tta_logits)
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(targets), len(targets)) for _ in range(n_bootstrap)]
    mel_idx = labels.index('mel')
    print(f'{"(validation)":24} {"F1 TTAsiz":>10} {"F1 TTAli":>9} {"fark":>7} {"%95 aralik":>17} {"TTA daha iyi":>13} {"melR":>12}')
    rows = [(name, [name]) for name in plain] + [('Ensemble (4 model)', list(plain))]
    for label, names in rows:
        p0 = np.mean([plain[n] for n in names], axis=0).argmax(axis=1)
        p1 = np.mean([with_tta[n] for n in names], axis=0).argmax(axis=1)
        s0, s1 = score(targets, p0, mel_idx), score(targets, p1, mel_idx)
        diff = np.array([f1_score(targets[i], p1[i], average='macro') - f1_score(targets[i], p0[i], average='macro')
                         for i in samples])
        low, high = np.percentile(diff, [2.5, 97.5])
        print(f'{label:24} {s0["macro_f1"]:10.3f} {s1["macro_f1"]:9.3f} {s1["macro_f1"] - s0["macro_f1"]:+7.3f} '
              f'({low:+.3f}, {high:+.3f}) {(diff > 0).mean() * 100:12.0f}% {s0["mel_recall"]:.3f}->{s1["mel_recall"]:.3f}')


def candidate_study(n_bootstrap=1000, seed=0):
    """Aday model mevcut kalibre 4'lu ensemble'a ne katiyor? Secenekler oneden belirlendi:
    ayni mimarinin eski uyesinin yerine gecmek ya da yeni uye olarak eklenmek."""
    logits, targets, labels = load_split('val', models={**MODELS, **CANDIDATES})
    probs = {name: softmax(l / fit_temperature(l, targets)) for name, l in logits.items()}
    options = {'Mevcut (4 model)': list(MODELS)}
    for candidate in CANDIDATES:
        options[f'{candidate}, ConvNeXt yerine'] = [candidate if n == 'ConvNeXt' else n for n in MODELS]
        options[f'{candidate}, 5. uye'] = list(MODELS) + [candidate]
        options[f'{candidate} tek basina'] = [candidate]
    mel_idx = labels.index('mel')
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(targets), len(targets)) for _ in range(n_bootstrap)]
    preds = {k: np.mean([probs[n] for n in v], axis=0).argmax(axis=1) for k, v in options.items()}
    ref_f1 = np.array([f1_score(targets[i], preds['Mevcut (4 model)'][i], average='macro') for i in samples])
    print(f'{"(validation, kalibre)":30} {"F1":>6} {"fark":>7} {"%95 aralik":>17} {"mevcudu gecme":>14} {"BAcc":>6} {"melR":>6} {"melP":>6}')
    for label, pred in preds.items():
        s = score(targets, pred, mel_idx)
        diff = np.array([f1_score(targets[i], pred[i], average='macro') for i in samples]) - ref_f1
        low, high = np.percentile(diff, [2.5, 97.5])
        print(f'{label:30} {s["macro_f1"]:6.3f} {diff.mean():+7.3f} ({low:+.3f}, {high:+.3f}) '
              f'{(diff > 0).mean() * 100:13.0f}% {s["balanced_accuracy"]:6.3f} {s["mel_recall"]:6.3f} {s["mel_precision"]:6.3f}')


def main():
    parser = argparse.ArgumentParser(description='Ensemble kombinasyonlarini validation setinde karsilastir.')
    parser.add_argument('--calibrate', action='store_true',
                        help='Ortalamadan once her modeli validation setinde temperature scaling ile kalibre et')
    parser.add_argument('--weight-study', action='store_true',
                        help='Kalibre 4lu ensemble uzerinde model cikarma ve B0 agirligi karsilastirmasi')
    parser.add_argument('--mel-threshold-table', action='store_true',
                        help='Kalibre 4lu ensemble icin melanom esigi denge tablosu')
    parser.add_argument('--tta-study', action='store_true',
                        help='TTAli ve TTAsiz sonuclari karsilastir (once cache_predictions.py --tta)')
    parser.add_argument('--candidate-study', action='store_true',
                        help='CANDIDATES modellerinin mevcut ensemble a katkisi')
    args = parser.parse_args()
    if args.candidate_study:
        candidate_study()
        return
    if args.tta_study:
        tta_study()
        return
    if args.weight_study or args.mel_threshold_table:
        args.calibrate = True

    logits, targets, labels = load_split('val')
    mel_idx = labels.index('mel')
    temperatures = {name: 1.0 for name in logits}
    if args.calibrate:
        print(f'{"Model":10} {"T":>5} {"NLL once":>9} {"NLL sonra":>10}')
        for name, l in logits.items():
            temperatures[name] = fit_temperature(l, targets)
            print(f'{name:10} {temperatures[name]:5.2f} {nll(l, targets):9.3f} {nll(l, targets, temperatures[name]):10.3f}')
        print()
    probs = {name: softmax(l / temperatures[name]) for name, l in logits.items()}
    if args.weight_study:
        weight_study(probs, targets, mel_idx)
        return
    if args.mel_threshold_table:
        mel_threshold_table(probs, targets, labels)
        return

    rows = []
    names = list(MODELS)
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            # Duz ortalama: her model esit soz hakkina sahip.
            avg = np.mean([probs[name] for name in combo], axis=0)
            rows.append((' + '.join(combo), score(targets, avg.argmax(axis=1), mel_idx)))

    rows.sort(key=lambda row: row[1]['macro_f1'], reverse=True)
    title = 'Kombinasyon (validation, ' + ('kalibre' if args.calibrate else 'duz ortalama') + ')'
    print(f'{title:36} {"F1":>6} {"BAcc":>6} {"Prec":>6} {"melR":>6} {"melP":>6}')
    for name, s in rows:
        print(f'{name:36} {s["macro_f1"]:6.3f} {s["balanced_accuracy"]:6.3f} {s["macro_precision"]:6.3f} '
              f'{s["mel_recall"]:6.3f} {s["mel_precision"]:6.3f}')


if __name__ == '__main__':
    main()
