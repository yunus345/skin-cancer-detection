"""Train a lightweight EfficientNet baseline for skin lesion classification."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, get_worker_info
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    Swin_V2_T_Weights,
    convnext_tiny,
    efficientnet_b0,
    efficientnet_v2_s,
    swin_v2_t,
)

from src.data.loader import SkinCancerDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Her mimarinin ImageNet'te on-egitildigi giris boyutu; --image-size verilmezse bu kullanilir.
NATIVE_IMAGE_SIZE = {
    'efficientnet_b0': 224,
    'convnext_tiny': 224,
    'efficientnet_v2_s': 384,
    'swin_v2_t': 256,
    # timm: ImageNet-22k ile on-egitilmis, ImageNet-1k'da 384px'te ince ayarlanmis ConvNeXt-Tiny.
    'convnext_tiny_in22k_384': 384,
}

# torchvision'da olmayan agirliklar icin timm model adlari (Hugging Face'ten indirilir).
TIMM_MODELS = {
    'convnext_tiny_in22k_384': 'convnext_tiny.fb_in22k_ft_in1k_384',
}


class AlbumentationsTransform:
    """Adapts an albumentations Compose to the torchvision-style `transform(pil_image) -> tensor` call,
    so SkinCancerDataset (which calls `self.transform(img)` with a PIL image) needs no changes."""

    def __init__(self, compose: A.Compose):
        self.compose = compose

    def __call__(self, pil_image):
        return self.compose(image=np.array(pil_image))['image']


def gray_world_color_constancy(image, **kwargs):
    """Gray World varsayimi: dogal bir sahnenin ortalama rengi griye yakin olmali. Her kanali,
    kanal ortalamalari esitlenecek sekilde olcekler - cihaz/isik kaynakli renk sapmasini azaltir."""
    image = image.astype(np.float32)
    channel_means = image.reshape(-1, image.shape[2]).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / (channel_means + 1e-6)
    corrected = image * scale
    return np.clip(corrected, 0, 255).astype(np.uint8)


def build_transforms(image_size: int, train: bool = False, augmentation: str = 'v1'):
    # Color constancy: rastgele bir augmentation degil, her zaman uygulanan bir normalizasyon
    # adimi - hem train hem val/test'te aynen calisir, p=1.0.
    color_constancy = A.Lambda(image=gray_world_color_constancy, p=1.0)

    if train and augmentation == 'v2':
        # v2 (2026-10-15): v1'in iki zayif noktasini kapatiyor.
        # 1) Once val/test ile ayni sekilde kareye sikistir, sonra kirp: 600x450 goruntude v1'in
        #    RandomResizedCrop'u ~%90 ihtimalle ayni sabit orta kesime dusuyordu ve egitimde goruntu
        #    val/test'tekinden farkli oranda sikisiyordu. Kare goruntuden %80-100 alan kesmek her
        #    zaman mumkun, yani kirpma gercekten rastgele.
        # 2) Dermoskopide "yukari" yonu yok: tam tur dondurme. Kose bosluklari siyah (cogu
        #    goruntude koseler zaten koyu); ayna yansitma lezyonu kopyalayip sahte desen uretirdi.
        compose = A.Compose([
            color_constancy,
            A.Resize(image_size, image_size),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    elif train:
        # v1 - RandomResizedCrop + CoarseDropout (cutout), ColorJitter'dan daha gercekci/agresif
        # augmentation - modelin lezyonun tamamina degil, parcalarina da odaklanmasini zorlar.
        compose = A.Compose([
            color_constancy,
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        compose = A.Compose([
            color_constancy,
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return AlbumentationsTransform(compose)


def seed_augmentations_per_worker(worker_id):
    # Albumentations Compose kendi ic rastgele uretecini tutuyor ve DataLoader her yardimci
    # surece bunun ayni kopyasini gonderiyor; yeniden tohumlanmazsa tum yardimcilar ayni
    # "rastgele" augmentation dizisini uretir. worker_info.seed her yardimci icin farkli.
    worker_info = get_worker_info()
    worker_info.dataset.transform.compose.set_random_seed(worker_info.seed)


def build_model(num_classes: int, architecture: str = 'efficientnet_b0'):
    # CNN'ler son katmanini `classifier` adli bir Sequential icinde tutuyor (Linear'in indeksi farkli:
    # EfficientNet B0/V2: [1], ConvNeXt: LayerNorm+Flatten sonrasi [2]); Swin'de ise tek bir `head` Linear'i.
    if architecture == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif architecture == 'efficientnet_v2_s':
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif architecture == 'convnext_tiny':
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif architecture == 'swin_v2_t':
        model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif architecture in TIMM_MODELS:
        import timm  # sadece timm modelleri icin gerekli
        # num_classes verilince timm son katmani (head.fc) 7 sinifli yeni bir katmanla degistiriyor;
        # isim 'head' ile basladigi icin set_backbone_trainable onu donmadan birakir.
        model = timm.create_model(TIMM_MODELS[architecture], pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Bilinmeyen mimari: {architecture}')
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (trainable=False) or unfreeze the pretrained backbone; the classifier head is left untouched."""
    # Son katman CNN'lerde `classifier`, Swin'de `head` adini tasiyor; ikisi de donmadan kaliyor.
    # Yeni bir mimari eklerken son katmanin adi burada yoksa Faz 1'de tum model donar.
    for name, param in model.named_parameters():
        if not name.startswith(('classifier', 'head')):
            param.requires_grad = trainable


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """Class-weighted focal loss: modelin zaten emin oldugu (kolay) orneklerin
    loss'a katkisini (1-p)^gamma ile azaltir, kararsiz/zor orneklere odaklanir."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # dogru sinifa verilen olasilik
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class WeightedLabelSmoothingCE(nn.Module):
    """Class-weighted CrossEntropy + label smoothing; sinif agirligi SADECE gercek sinifa gore.

    PyTorch'un CrossEntropyLoss(weight=..., label_smoothing=...) agirliklari dagitilan smoothing
    kismina da uyguluyor: df agirligi 12, nv agirligi 0.2 iken bir nv goruntusunde df olasiligini
    artirmak loss'u DUSURUYOR, model nadir siniflari her yerde tahmin etmeye itiliyor
    (2026-10-16 Kaggle kosusunda val_f1 0.744 -> 0.657). Burada her ornegin loss'u kendi gercek
    sinifinin agirligiyla carpiliyor; label_smoothing=0 iken nn.CrossEntropyLoss(weight=w) ile ayni.
    """

    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits.float(), dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=1)  # kalan pay butun siniflara esit, agirliksiz
        per_sample = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        sample_weight = self.weight[targets]
        return (per_sample * sample_weight).sum() / sample_weight.sum()


def select_device(preferred: str = 'auto') -> torch.device:
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate(model, loader, device, criterion=None, use_amp=False):
    model.eval()
    predictions = []
    labels = []
    losses = []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
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


def run_phase(model, phase_name, num_epochs, optimizer, scheduler, train_loader, val_loader, criterion, device, out_dir, state, scaler):
    """Trains for num_epochs, mutating `state` (best_f1/best_metrics/epochs_without_improvement/global_epoch).

    Returns True if early stopping triggered during this phase.
    """
    use_amp = scaler.is_enabled()
    for _ in range(num_epochs):
        state['global_epoch'] += 1
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            # float16'da cok kucuk gradyanlar sifira yuvarlanabilir; scaler loss'u buyutup
            # geri-yayilimi yapar, optimizer adimindan once gradyanlari eski olcegine indirir.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        metrics = evaluate(model, val_loader, device, criterion, use_amp=use_amp)
        metrics['phase'] = phase_name
        avg_train_loss = train_loss / max(len(train_loader), 1)
        epoch_seconds = time.time() - epoch_start
        print(
            f'[{phase_name}] Epoch {state["global_epoch"]} - lr={current_lr:.2e} train_loss={avg_train_loss:.4f} '
            f'val_loss={metrics["loss"]:.4f} val_accuracy={metrics["accuracy"]:.4f} '
            f'val_balanced_accuracy={metrics["balanced_accuracy"]:.4f} val_f1={metrics["f1"]:.4f} '
            f'sure={epoch_seconds / 60:.1f}dk'
        )
        history_entry = dict(metrics)
        history_entry['epoch'] = state['global_epoch']
        history_entry['train_loss'] = avg_train_loss
        history_entry['lr'] = current_lr
        history_entry['epoch_seconds'] = epoch_seconds
        state['history'].append(history_entry)
        if metrics['f1'] > state['best_f1']:
            state['best_f1'] = metrics['f1']
            state['best_metrics'] = metrics
            torch.save({
                'model_state_dict': model.state_dict(),
                'labels': state['labels'],
                'architecture': state['architecture'],
                'image_size': state['image_size'],
                'augmentation': state['augmentation'],
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
    loss_fn: str = 'ce',
    focal_gamma: float = 2.0,
    architecture: str = 'efficientnet_b0',
    num_workers: int = 2,
    augmentation: str = 'v1',
    label_smoothing: float = 0.0,
    warmup_epochs: int = 2,
):
    torch.manual_seed(seed)
    device = select_device(device_arg)
    print(f'Using device: {device}')

    # Hiz ayarlari sadece CUDA'da (Colab GPU) acik; Mac'te (MPS) davranis eskisiyle ayni kalir.
    on_cuda = device.type == 'cuda'
    torch.backends.cudnn.benchmark = on_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=on_cuda)
    print(f'Mixed precision (AMP): {"on" if on_cuda else "off"}, num_workers={num_workers}')
    print(f'Architecture: {architecture}, image size: {image_size}px, augmentation: {augmentation}')
    print(f'Recipe: lr={learning_rate:.0e}/{finetune_lr:.0e}, scheduler={scheduler_name} (warmup {warmup_epochs}), label_smoothing={label_smoothing}, patience={patience}')

    train_dataset = SkinCancerDataset(manifest_path, split='train', transform=build_transforms(image_size, train=True, augmentation=augmentation))
    val_dataset = SkinCancerDataset(manifest_path, split='val', transform=build_transforms(image_size))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('Manifest must contain non-empty train/val splits. Run src.data.split_manifest first.')

    model = build_model(num_classes=len(train_dataset.labels), architecture=architecture).to(device)
    label_counts = torch.bincount(torch.tensor([
        train_dataset.label_to_idx[str(row['label']).strip()]
        for row in train_dataset.manifest
    ]), minlength=len(train_dataset.labels)).float()
    class_weights = label_counts.sum() / (len(label_counts) * label_counts.clamp_min(1))
    if loss_fn == 'focal':
        criterion = FocalLoss(weight=class_weights.to(device), gamma=focal_gamma)
    else:
        # label_smoothing > 0: hedef "%100 bu sinif" yerine ornegin "%90 bu sinif, kalan %10 digerlerine
        # esit". Modelin asiri emin olmasini engeller (val_loss'un yukselmesi / T=1.47 sorunu).
        # Not: loss degeri artik label_smoothing=0 kosularinin loss'uyla karsilastirilamaz.
        if label_smoothing > 0:
            criterion = WeightedLabelSmoothingCE(class_weights.to(device), label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=on_cuda,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_augmentations_per_worker,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'best_f1': -1.0,
        'best_metrics': None,
        'epochs_without_improvement': 0,
        'global_epoch': 0,
        'patience': patience,
        'labels': train_dataset.labels,
        'architecture': architecture,
        'image_size': image_size,
        'augmentation': augmentation,
        'history': [],
    }

    def build_scheduler(optimizer, phase_epochs, phase_name):
        if scheduler_name == 'cosine':
            # LR, fazin basindaki degerden fazin son epoch'una kadar yumusakca 0'a yaklasir.
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
        if scheduler_name == 'warmup_cosine' and phase_name == 'finetune':
            # Backbone yeni acildiginda buyuk LR, ImageNet bilgisini bir anda bozabilir: once birkac epoch
            # LR'yi kademeli yukselt (warmup), sonra kalan epoch'larda cosine ile sifira yaklastir.
            # Faz 1 (sadece son katman) sabit LR ile kaliyor.
            def factor(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / (warmup_epochs + 1)
                progress = (epoch - warmup_epochs) / max(1, phase_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return None

    # Faz 1 - feature extraction: backbone donuk, sadece yeni classifier katmani egitiliyor.
    set_backbone_trainable(model, trainable=False)
    print(f'Phase 1 (feature_extract): {count_trainable_params(model):,} trainable params')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = build_scheduler(optimizer, feature_extract_epochs, 'feature_extract')
    stopped_early = run_phase(
        model, 'feature_extract', feature_extract_epochs, optimizer, scheduler,
        train_loader, val_loader, criterion, device, out_dir, state, scaler,
    )

    # Faz 2 - fine-tuning: butun ag aciliyor, daha dusuk bir learning rate ile devam ediyor.
    if not stopped_early:
        set_backbone_trainable(model, trainable=True)
        print(f'Phase 2 (finetune): {count_trainable_params(model):,} trainable params')
        state['epochs_without_improvement'] = 0  # unfreeze loss landscape'i degistirir, patience sayaci sifirlanir
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        scheduler = build_scheduler(optimizer, finetune_epochs, 'finetune')
        run_phase(
            model, 'finetune', finetune_epochs, optimizer, scheduler,
            train_loader, val_loader, criterion, device, out_dir, state, scaler,
        )

    final_metrics = state['best_metrics'] or evaluate(model, val_loader, device, criterion)
    checkpoint_path = out_dir / 'baseline_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'labels': train_dataset.labels,
        'architecture': architecture,
        'image_size': image_size,
        'augmentation': augmentation,
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
    parser.add_argument('--image-size', type=int, default=None, help='Giris boyutu (varsayilan: mimarinin on-egitim boyutu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto', help='Compute device; auto picks MPS > CUDA > CPU')
    parser.add_argument('--scheduler', choices=['none', 'cosine', 'warmup_cosine'], default='none',
                        help='LR scheduler: none (sabit), cosine (her faz), warmup_cosine (sadece finetune: warmup + cosine)')
    parser.add_argument('--warmup-epochs', type=int, default=2, help='warmup_cosine icin finetune basindaki warmup epoch sayisi')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='CrossEntropy label smoothing (0 = kapali)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (val_f1 iyilesmeyen epoch sayisi)')
    parser.add_argument('--loss-fn', choices=['ce', 'focal'], default='ce', help='Loss fonksiyonu: ce (class-weighted CrossEntropy) veya focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parametresi (sadece --loss-fn focal ile)')
    parser.add_argument('--architecture', choices=list(NATIVE_IMAGE_SIZE), default='efficientnet_b0', help='Backbone mimarisi')
    parser.add_argument('--augmentation', choices=['v1', 'v2'], default='v1',
                        help='Egitim augmentation tarifi: v1 (ilk dort model) ya da v2 (kare-sonra-kirp + tam tur dondurme)')
    parser.add_argument('--num-workers', type=int, default=2, help='Veriyi arka planda hazirlayan yardimci surec sayisi (0 = ana surecte, sirayla)')
    args = parser.parse_args()

    finetune_lr = args.finetune_lr if args.finetune_lr is not None else args.learning_rate / 10
    image_size = args.image_size or NATIVE_IMAGE_SIZE[args.architecture]

    train_model(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        feature_extract_epochs=args.feature_extract_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        finetune_lr=finetune_lr,
        batch_size=args.batch_size,
        image_size=image_size,
        seed=args.seed,
        device_arg=args.device,
        scheduler_name=args.scheduler,
        patience=args.patience,
        loss_fn=args.loss_fn,
        focal_gamma=args.focal_gamma,
        architecture=args.architecture,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == '__main__':
    main()
