"""Plot the confusion matrix (from test_metrics.json) as a heatmap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # ekran olmadan (headless) PNG kaydetmek icin
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# references/palette.md'deki sequential (mavi) ramp, 100 -> 700
BLUE_SEQUENTIAL = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]
COLOR_TEXT = '#0b0b0b'
COLOR_MUTED = '#898781'


def main():
    parser = argparse.ArgumentParser(description='Plot a confusion matrix heatmap from test_metrics.json')
    parser.add_argument('--metrics', default='models/test_metrics.json')
    parser.add_argument('--output', default='models/confusion_matrix.png')
    args = parser.parse_args()

    data = json.loads(Path(args.metrics).read_text())
    labels = data['confusion_matrix']['labels']
    matrix = data['confusion_matrix']['matrix']
    n = len(labels)

    cmap = LinearSegmentedColormap.from_list('blue_sequential', BLUE_SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#fcfcfb')
    im = ax.imshow(matrix, cmap=cmap)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color=COLOR_MUTED)
    ax.set_yticklabels(labels, color=COLOR_MUTED)
    ax.set_xlabel('Model tahmini', color=COLOR_TEXT)
    ax.set_ylabel('Gerçek etiket', color=COLOR_TEXT)
    ax.set_title('Confusion matrix (test seti)', color=COLOR_TEXT)

    max_value = max(max(row) for row in matrix)
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            text_color = '#fcfcfb' if value > max_value * 0.5 else COLOR_TEXT
            ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f'Wrote confusion matrix plot to {output_path}')


if __name__ == '__main__':
    main()
