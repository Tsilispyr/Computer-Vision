"""
Παράδειγμα 1 - Σημασιολογική τμηματοποίηση SBD (Semantic Boundaries Dataset) με U-Net
21 κλάσεις: 0=background, 1-20=PASCAL VOC κατηγορίες
Μετρικές: pixel accuracy, mean IoU (mIoU)
3 πειράματα με διαφορετικές υπερ-παραμέτρους

Σημείωση: Το SBD γίνεται download αυτόματα (~1.5 GB).
Αν αποτύχει ο αυτόματος download, κατεβάστε το αρχείο χειροκίνητα:
  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
και αποσυμπιέστε το στον φάκελο data/sbd/ ώστε να υπάρχουν:
  data/sbd/img/, data/sbd/cls/, data/sbd/train.txt, data/sbd/val.txt
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision.datasets import SBDataset as _TorchSBD
    _HAS_TORCHVISION_SBD = True
except ImportError:
    _HAS_TORCHVISION_SBD = False

# ==== Paths ====
ROOT    = Path(__file__).parent
SBD_DIR = ROOT / 'data' / 'sbd'
RESULTS = ROOT / 'results' / 'ex1'
SBD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE    = 128
NUM_CLASSES = 21          # background + 20 PASCAL VOC classes
SUBSET_N    = 600         # train subset (full train=8498, use subset for CPU)
SUBSET_VAL  = 200

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

# ==== Dataset wrapper ========

class SBDSegDataset(Dataset):
    """Wraps torchvision SBDataset (mode='segmentation') με resize + normalize.

    SBDataset επιστρέφει (PIL Image, PIL Image)  όπου target είναι greyscale
    με τιμές 0-20 (PASCAL VOC class indices).
    """
    def __init__(self, split: str = 'train', size: int = IMG_SIZE, augment: bool = True):
        assert _HAS_TORCHVISION_SBD, 'torchvision δεν βρέθηκε'
        # skip download if already extracted
        _already = (SBD_DIR / 'img').exists() and (SBD_DIR / 'cls').exists()
        self._ds    = _TorchSBD(root=str(SBD_DIR), image_set=split,
                                mode='segmentation', download=not _already)
        self.size    = size
        self.augment = augment

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, mask = self._ds[idx]           # PIL RGB, PIL L (greyscale 0-20)

        img  = TF.resize(img,  [self.size, self.size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=Image.NEAREST)

        if self.augment and torch.rand(1).item() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        img   = TF.to_tensor(img)
        img   = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask  = torch.as_tensor(np.array(mask), dtype=torch.long).clamp(0, NUM_CLASSES - 1)
        return img, mask


# ==== U-Net ====

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Ελαφρύ U-Net (configurable base channels)."""
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES,
                 base: int = 16):
        super().__init__()
        b = base
        self.enc1 = _conv_block(in_channels, b)
        self.enc2 = _conv_block(b,    b * 2)
        self.enc3 = _conv_block(b*2,  b * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(b*4, b * 8)
        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = _conv_block(b*8, b*4)
        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = _conv_block(b*4, b*2)
        self.up1  = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = _conv_block(b*2, b)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ==== Metrics ========================

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = NUM_CLASSES,
                 ignore_index: int = 255) -> float:
    p = preds.view(-1); t = targets.view(-1)
    iou_list = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.mean(iou_list)) if iou_list else 0.0


# ==== Training helpers ====

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, n_pix = 0.0, 0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == masks).sum().item()
        n_pix    += masks.numel()
    return loss_sum / len(loader.dataset), correct / n_pix


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, n_pix = 0.0, 0, 0
    all_preds, all_masks = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        preds  = logits.argmax(1)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (preds == masks).sum().item()
        n_pix    += masks.numel()
        all_preds.append(preds.cpu()); all_masks.append(masks.cpu())
    miou = compute_miou(torch.cat(all_preds), torch.cat(all_masks))
    return loss_sum / len(loader.dataset), correct / n_pix, miou


# ==== Plotting ====================

# Pascal VOC palette (21 classes)
VOC_PALETTE = np.array([
    [  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
    [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
    [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
    [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128], [192, 128, 128],
    [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
    [  0,  64, 128],
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(min(NUM_CLASSES, len(VOC_PALETTE))):
        rgb[mask == c] = VOC_PALETTE[c]
    return rgb


def plot_curves(history: dict, exp_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eps = range(1, len(history['train_loss']) + 1)

    axes[0].plot(eps, history['train_loss'], label='Train'); axes[0].plot(eps, history['val_loss'], label='Val')
    axes[0].set(title='Loss', xlabel='Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, [a*100 for a in history['train_acc']], label='Train')
    axes[1].plot(eps, [a*100 for a in history['val_acc']],   label='Val')
    axes[1].set(title='Pixel Accuracy (%)', xlabel='Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps, [m*100 for m in history['val_miou']])
    axes[2].set(title='Val mIoU (%)', xlabel='Epoch'); axes[2].grid(True, alpha=0.3)

    plt.suptitle(exp_name, fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS / f'{exp_name}_curves.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_samples(model, loader, exp_name: str, n: int = 4):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    imgs, masks = next(iter(loader))
    model.eval()
    with torch.no_grad():
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    for i in range(n):
        img_np = np.clip(imgs[i].numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        gt_col = colorize_mask(masks[i].numpy())
        pr_col = colorize_mask(preds[i].numpy())
        axes[i, 0].imshow(img_np);  axes[i, 0].set_title('Εικόνα', fontsize=8)
        axes[i, 1].imshow(gt_col);  axes[i, 1].set_title('GT Mask', fontsize=8)
        axes[i, 2].imshow(pr_col);  axes[i, 2].set_title('Πρόβλεψη', fontsize=8)
        for ax in axes[i]: ax.axis('off')
    plt.suptitle(f'{exp_name} - Αποτελέσματα τμηματοποίησης', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS / f'{exp_name}_samples.png', dpi=100, bbox_inches='tight')
    plt.close()


# ==== Experiment runner ====================================================================================================================

def run_experiment(exp_name: str, epochs: int, lr: float,
                   base_ch: int = 16, batch_size: int = 4,
                   weight_decay: float = 1e-4) -> dict:
    print(f'\n !START! {exp_name}  epochs={epochs}  lr={lr}  base_ch={base_ch}')

    train_ds = SBDSegDataset('train',      augment=True)
    val_ds   = SBDSegDataset('val',        augment=False)

    rng     = np.random.default_rng(42)
    tr_idx  = rng.choice(len(train_ds), SUBSET_N,   replace=False)
    val_idx = rng.choice(len(val_ds),   SUBSET_VAL, replace=False)

    tr_loader  = DataLoader(Subset(train_ds, tr_idx),  batch_size=batch_size,
                            shuffle=True,  num_workers=0)
    val_loader = DataLoader(Subset(val_ds,  val_idx),  batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model     = UNet(base=base_ch, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_miou': []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta         = train_epoch(model, tr_loader, criterion, optimizer)
        vl, va, miou   = eval_epoch(model,  val_loader, criterion)
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        history['val_miou'].append(miou)
        print(f'  ep {ep:02d}/{epochs}  loss {tl:.4f}/{vl:.4f}  acc {ta:.3f}/{va:.3f}  mIoU {miou:.3f}')

    elapsed   = time.time() - t0
    best_miou = max(history['val_miou'])
    best_acc  = max(history['val_acc'])
    print(f'  -> best mIoU={best_miou:.4f}  pix_acc={best_acc:.4f}  ({elapsed/60:.1f} min)')

    plot_curves(history, exp_name)
    plot_samples(model, val_loader, exp_name)

    return {
        'epochs': epochs, 'lr': lr, 'base_channels': base_ch,
        'val_miou':      round(best_miou, 4),
        'val_pixel_acc': round(best_acc, 4),
        'elapsed_min':   round(elapsed / 60, 2),
    }


# ==== Main ========

EXPERIMENTS = [
    # (name, epochs, lr, base_channels)
    ('SBD_exp1_baseline',      5,  1e-3, 16),
    ('SBD_exp2_more_epochs',   10, 5e-4, 16),
    ('SBD_exp3_larger_model',  5,  1e-3, 32),
]

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'SBD root: {SBD_DIR}')
    print(f'Train subset: {SUBSET_N}  Val subset: {SUBSET_VAL}')

    all_results = {}
    for name, ep, lr, base in EXPERIMENTS:
        res = run_experiment(name, ep, lr, base)
        all_results[name] = res

    print('\n' + '='*72)
    print(f'{"Experiment":<30} {"Epochs":>6} {"LR":>8} {"BaseCh":>7} {"mIoU":>7} {"PixAcc":>8}')
    print('-'*72)
    for name, r in all_results.items():
        print(f'{name:<30} {r["epochs"]:>6} {r["lr"]:>8.0e} {r["base_channels"]:>7} '
              f'{r["val_miou"]:>7.4f} {r["val_pixel_acc"]:>8.4f}')

    with open(RESULTS / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # comparison plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names  = list(all_results.keys())
    mious  = [all_results[n]['val_miou'] * 100 for n in names]
    bars   = ax.bar(names, mious, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set(title='SBD - mIoU ανά πείραμα', ylabel='mIoU (%)', ylim=(0, 100))
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS / 'comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f'\nΑποτελέσματα: {RESULTS}')
