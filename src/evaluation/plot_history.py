"""Plot train/val loss and val F1 curves from a training_history.json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # ekran olmadan (headless) PNG kaydetmek icin
import matplotlib.pyplot as plt

# references/palette.md'deki kategorik slotlar: slot 1 (train) ve slot 2 (val)
COLOR_TRAIN = '#2a78d6'
COLOR_VAL = '#eb6834'
COLOR_GRID = '#e1e0d9'
COLOR_AXIS = '#c3c2b7'
COLOR_MUTED = '#898781'
COLOR_TEXT = '#0b0b0b'


def style_axis(ax):
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED)


def find_phase_boundary(history):
    for entry in history:
        if entry['phase'] == 'finetune':
            return entry['epoch']
    return None


def main():
    parser = argparse.ArgumentParser(description='Plot training/validation curves from training_history.json')
    parser.add_argument('--history', default='models/training_history.json')
    parser.add_argument('--output', default='models/training_curves.png')
    args = parser.parse_args()

    history = json.loads(Path(args.history).read_text())
    if not history:
        raise ValueError(f'{args.history} is empty — run training first.')

    epochs = [e['epoch'] for e in history]
    train_loss = [e['train_loss'] for e in history]
    val_loss = [e['loss'] for e in history]
    val_f1 = [e['f1'] for e in history]
    learning_rate = [e.get('lr') for e in history]
    boundary = find_phase_boundary(history)

    has_lr = all(lr is not None for lr in learning_rate)
    fig, axes = plt.subplots(3 if has_lr else 2, 1, figsize=(8, 10 if has_lr else 7), sharex=True)
    ax_loss, ax_f1 = axes[0], axes[1]
    fig.patch.set_facecolor('#fcfcfb')

    ax_loss.plot(epochs, train_loss, color=COLOR_TRAIN, linewidth=2, marker='o', markersize=5, label='train_loss')
    ax_loss.plot(epochs, val_loss, color=COLOR_VAL, linewidth=2, marker='o', markersize=5, label='val_loss')
    ax_loss.set_ylabel('Loss', color=COLOR_TEXT)
    ax_loss.set_title('Eğitim / doğrulama kaybı (loss)', color=COLOR_TEXT)
    ax_loss.legend(frameon=False, labelcolor=COLOR_TEXT)
    style_axis(ax_loss)

    ax_f1.plot(epochs, val_f1, color=COLOR_TRAIN, linewidth=2, marker='o', markersize=5, label='val_f1')
    ax_f1.set_ylabel('Macro F1', color=COLOR_TEXT)
    ax_f1.set_title('Doğrulama F1 skoru', color=COLOR_TEXT)
    ax_f1.legend(frameon=False, labelcolor=COLOR_TEXT)
    style_axis(ax_f1)

    axes_to_mark = [ax_loss, ax_f1]
    if has_lr:
        ax_lr = axes[2]
        ax_lr.plot(epochs, learning_rate, color=COLOR_TRAIN, linewidth=2, marker='o', markersize=5, label='learning_rate')
        ax_lr.set_ylabel('Learning rate', color=COLOR_TEXT)
        ax_lr.set_xlabel('Epoch', color=COLOR_TEXT)
        ax_lr.set_title('Cosine annealing scheduler', color=COLOR_TEXT)
        ax_lr.set_yscale('log')
        ax_lr.legend(frameon=False, labelcolor=COLOR_TEXT)
        style_axis(ax_lr)
        axes_to_mark.append(ax_lr)
    else:
        ax_f1.set_xlabel('Epoch', color=COLOR_TEXT)

    if boundary is not None:
        for ax in axes_to_mark:
            ax.axvline(boundary - 0.5, color=COLOR_MUTED, linewidth=1, linestyle='--')
        ax_loss.annotate(
            'feature_extract -> finetune', xy=(boundary - 0.5, ax_loss.get_ylim()[1]),
            xytext=(4, -4), textcoords='offset points', fontsize=8, color=COLOR_MUTED,
            va='top',
        )

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f'Wrote training curves to {output_path}')


if __name__ == '__main__':
    main()
