"""Train a lightweight EfficientNet baseline for skin lesion classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data.loader import SkinCancerDataset


def build_transforms(image_size: int, train: bool = False):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


def build_model(num_classes: int):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (trainable=False) or unfreeze the pretrained backbone; the classifier head is left untouched."""
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = trainable


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device(preferred: str = 'auto') -> torch.device:
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate(model, loader, device, criterion=None):
    model.eval()
    predictions = []
    labels = []
    losses = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_images)
            if criterion is not None:
                losses.append(criterion(logits, batch_labels).item())
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1': f1_score(labels, predictions, average='macro', zero_division=0),
        'loss': sum(losses) / len(losses) if losses else None,
    }


def run_phase(model, phase_name, num_epochs, optimizer, scheduler, train_loader, val_loader, criterion, device, out_dir, state):
    """Trains for num_epochs, mutating `state` (best_f1/best_metrics/epochs_without_improvement/global_epoch).

    Returns True if early stopping triggered during this phase.
    """
    for _ in range(num_epochs):
        state['global_epoch'] += 1
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        metrics = evaluate(model, val_loader, device, criterion)
        metrics['phase'] = phase_name
        avg_train_loss = train_loss / max(len(train_loader), 1)
        print(
            f'[{phase_name}] Epoch {state["global_epoch"]} - lr={current_lr:.2e} train_loss={avg_train_loss:.4f} '
            f'val_loss={metrics["loss"]:.4f} val_accuracy={metrics["accuracy"]:.4f} '
            f'val_balanced_accuracy={metrics["balanced_accuracy"]:.4f} val_f1={metrics["f1"]:.4f}'
        )
        history_entry = dict(metrics)
        history_entry['epoch'] = state['global_epoch']
        history_entry['train_loss'] = avg_train_loss
        history_entry['lr'] = current_lr
        state['history'].append(history_entry)
        if metrics['f1'] > state['best_f1']:
            state['best_f1'] = metrics['f1']
            state['best_metrics'] = metrics
            torch.save({
                'model_state_dict': model.state_dict(),
                'labels': state['labels'],
                'metrics': metrics,
            }, out_dir / 'best_model.pt')
            state['epochs_without_improvement'] = 0
        else:
            state['epochs_without_improvement'] += 1
            if state['epochs_without_improvement'] >= state['patience']:
                print(f'Early stopping during {phase_name} after {state["global_epoch"]} epochs.')
                return True
    return False


def train_model(
    manifest_path: str,
    output_dir: str,
    feature_extract_epochs: int,
    finetune_epochs: int,
    learning_rate: float,
    finetune_lr: float,
    batch_size: int,
    image_size: int,
    seed: int,
    device_arg: str,
    scheduler_name: str = 'none',
    patience: int = 3,
):
    torch.manual_seed(seed)
    device = select_device(device_arg)
    print(f'Using device: {device}')

    train_dataset = SkinCancerDataset(manifest_path, split='train', transform=build_transforms(image_size, train=True))
    val_dataset = SkinCancerDataset(manifest_path, split='val', transform=build_transforms(image_size))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('Manifest must contain non-empty train/val splits. Run src.data.split_manifest first.')

    model = build_model(num_classes=len(train_dataset.labels)).to(device)
    label_counts = torch.bincount(torch.tensor([
        train_dataset.label_to_idx[str(row['label']).strip()]
        for row in train_dataset.manifest
    ]), minlength=len(train_dataset.labels)).float()
    class_weights = label_counts.sum() / (len(label_counts) * label_counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'best_f1': -1.0,
        'best_metrics': None,
        'epochs_without_improvement': 0,
        'global_epoch': 0,
        'patience': patience,
        'labels': train_dataset.labels,
        'history': [],
    }

    def build_scheduler(optimizer, phase_epochs):
        if scheduler_name == 'cosine':
            # LR, fazin basindaki degerden fazin son epoch'una kadar yumusakca 0'a yaklasir.
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
        return None

    # Faz 1 - feature extraction: backbone donuk, sadece yeni classifier katmani egitiliyor.
    set_backbone_trainable(model, trainable=False)
    print(f'Phase 1 (feature_extract): {count_trainable_params(model):,} trainable params')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = build_scheduler(optimizer, feature_extract_epochs)
    stopped_early = run_phase(
        model, 'feature_extract', feature_extract_epochs, optimizer, scheduler,
        train_loader, val_loader, criterion, device, out_dir, state,
    )

    # Faz 2 - fine-tuning: butun ag aciliyor, daha dusuk bir learning rate ile devam ediyor.
    if not stopped_early:
        set_backbone_trainable(model, trainable=True)
        print(f'Phase 2 (finetune): {count_trainable_params(model):,} trainable params')
        state['epochs_without_improvement'] = 0  # unfreeze loss landscape'i degistirir, patience sayaci sifirlanir
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        scheduler = build_scheduler(optimizer, finetune_epochs)
        run_phase(
            model, 'finetune', finetune_epochs, optimizer, scheduler,
            train_loader, val_loader, criterion, device, out_dir, state,
        )

    final_metrics = state['best_metrics'] or evaluate(model, val_loader, device, criterion)
    checkpoint_path = out_dir / 'baseline_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'labels': train_dataset.labels,
        'metrics': final_metrics,
    }, checkpoint_path)

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(final_metrics, indent=2))

    history_path = out_dir / 'training_history.json'
    history_path.write_text(json.dumps(state['history'], indent=2))

    print(f'Wrote checkpoint to {checkpoint_path}')
    print(f'Wrote metrics to {metrics_path}')
    print(f'Wrote training history to {history_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Train a skin lesion model using a manifest CSV with a two-phase transfer-learning recipe '
                    '(feature extraction, then fine-tuning).'
    )
    parser.add_argument('--manifest', default='data/processed/manifest.csv', help='Manifest file with image_path,label,split columns')
    parser.add_argument('--output-dir', default='models', help='Directory to save checkpoints and metrics')
    parser.add_argument('--feature-extract-epochs', type=int, default=5, help='Epochs training only the classifier head (backbone frozen)')
    parser.add_argument('--finetune-epochs', type=int, default=10, help='Epochs training the full unfrozen network after feature extraction')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the feature-extraction phase')
    parser.add_argument('--finetune-lr', type=float, default=None, help='Learning rate for the fine-tuning phase (default: learning-rate / 10)')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto', help='Compute device; auto picks MPS > CUDA > CPU')
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default='none', help='LR scheduler per phase (none = sabit LR)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (val_f1 iyilesmeyen epoch sayisi)')
    args = parser.parse_args()

    finetune_lr = args.finetune_lr if args.finetune_lr is not None else args.learning_rate / 10

    train_model(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        feature_extract_epochs=args.feature_extract_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        finetune_lr=finetune_lr,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        device_arg=args.device,
        scheduler_name=args.scheduler,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()
