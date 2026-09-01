"""
Παράδειγμα 2 - Τμηματοποίηση Oxford-IIIT Pet με U-Net
Τριμάπες (trimaps): 1=foreground, 2=background, 3=boundary  ->  3 κλάσεις (0,1,2)
Μετρικές: pixel accuracy, mean IoU (mIoU)
3 πειράματα με διαφορετικές υπερ-παραμέτρους
"""

import json
import os
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

# == Paths ======
ROOT    = Path(__file__).parent
PET_DIR = ROOT / 'oxford-iiit-pet'
# double-nested structure after extraction
IMG_DIR     = PET_DIR / 'images'     / 'images'
ANN_DIR     = PET_DIR / 'annotations' / 'annotations'
TRIMAP_DIR  = ANN_DIR / 'trimaps'
RESULTS     = ROOT / 'results' / 'ex2'
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE   = 128         # resize for CPU feasibility
NUM_CLASSES = 3          # foreground / background / boundary
SUBSET_N   = 1200        # train subset (from ~3680)
SUBSET_VAL =  300        # val subset

# == Dataset ======

class OxfordPetDataset(Dataset):
    """Loads (image, trimap) pairs from the already-extracted Pet dataset.

    Trimap values 1,2,3 are remapped to labels 0,1,2.
    """
    def __init__(self, split: str = 'trainval', size: int = IMG_SIZE):
        assert TRIMAP_DIR.exists(), f'Trimap dir not found: {TRIMAP_DIR}'
        split_file = ANN_DIR / f'{split}.txt'
        with open(split_file) as f:
            # each line: "Name CLASS SPECIES BREED_ID"
            names = [ln.split()[0] for ln in f if not ln.startswith('#')]
        # keep only names that have both image and trimap
        self.samples = [
            (IMG_DIR / f'{n}.jpg', TRIMAP_DIR / f'{n}.png')
            for n in names
            if (IMG_DIR / f'{n}.jpg').exists() and (TRIMAP_DIR / f'{n}.png').exists()
        ]
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Resize
        img  = TF.resize(img,  [self.size, self.size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=Image.NEAREST)

        # Random horizontal flip (train augmentation)
        if torch.rand(1).item() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        img   = TF.to_tensor(img)                          # (3, H, W) float [0,1]
        img   = TF.normalize(img, [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask  = torch.as_tensor(np.array(mask), dtype=torch.long)
        # trimap: 1=fg, 2=bg, 3=boundary -> 0,1,2
        mask  = (mask - 1).clamp(0, NUM_CLASSES - 1)
        return img, mask


# == U-Net ======

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Ελαφρύ U-Net για σημασιολογική τμηματοποίηση.

    Channels: base -> 2·base -> 4·base -> 8·base (bottleneck)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES,
                 base: int = 16):
        super().__init__()
        b = base
        # Encoder
        self.enc1 = _conv_block(in_channels, b)
        self.enc2 = _conv_block(b,      b*2)
        self.enc3 = _conv_block(b*2,    b*4)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = _conv_block(b*4, b*8)
        # Decoder
        self.up3   = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3  = _conv_block(b*8, b*4)
        self.up2   = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2  = _conv_block(b*4, b*2)
        self.up1   = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1  = _conv_block(b*2, b)
        self.head  = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# == Metrics ==========================

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = NUM_CLASSES) -> float:
    p = preds.view(-1); t = targets.view(-1)
    iou_list = []
    for c in range(num_classes):
        inter = ((p == c) & (t == c)).sum().float()
        union = ((p == c) | (t == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return float(np.mean(iou_list)) if iou_list else 0.0


# ==== Training helpers ====================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, pix_correct, n_pix = 0.0, 0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        loss_sum    += loss.item() * imgs.size(0)
        pix_correct += (logits.argmax(1) == masks).sum().item()
        n_pix       += masks.numel()
    return loss_sum / len(loader.dataset), pix_correct / n_pix


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum, pix_correct, n_pix = 0.0, 0, 0
    all_preds, all_masks = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits  = model(imgs)
        loss    = criterion(logits, masks)
        preds   = logits.argmax(1)
        loss_sum    += loss.item() * imgs.size(0)
        pix_correct += (preds == masks).sum().item()
        n_pix       += masks.numel()
        all_preds.append(preds.cpu()); all_masks.append(masks.cpu())
    miou = compute_miou(torch.cat(all_preds), torch.cat(all_masks))
    return loss_sum / len(loader.dataset), pix_correct / n_pix, miou


# ==== Plotting ======

LABEL_COLORS = np.array([[255, 0, 0], [0, 128, 0], [0, 0, 255]], dtype=np.uint8)
LABEL_NAMES  = ['Foreground (pet)', 'Background', 'Boundary']

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(LABEL_COLORS):
        rgb[mask == c] = col
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
        img_np  = np.clip(imgs[i].numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        gt_col  = colorize_mask(masks[i].numpy())
        pr_col  = colorize_mask(preds[i].numpy())
        axes[i, 0].imshow(img_np);  axes[i, 0].set_title('Εικόνα', fontsize=8)
        axes[i, 1].imshow(gt_col);  axes[i, 1].set_title('GT Mask', fontsize=8)
        axes[i, 2].imshow(pr_col);  axes[i, 2].set_title('Πρόβλεψη', fontsize=8)
        for ax in axes[i]: ax.axis('off')
    plt.suptitle(f'{exp_name} - Αποτελέσματα τμηματοποίησης', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS / f'{exp_name}_samples.png', dpi=100, bbox_inches='tight')
    plt.close()


# ==== Experiment runner ==================================================

def run_experiment(exp_name: str, epochs: int, lr: float,
                   base_ch: int = 16, batch_size: int = 4,
                   weight_decay: float = 1e-4) -> dict:
    print(f'\n !START! {exp_name}  epochs={epochs}  lr={lr}  base_ch={base_ch}')

    full_ds  = OxfordPetDataset('trainval')
    rng      = np.random.default_rng(42)
    all_idx  = rng.permutation(len(full_ds))
    tr_idx   = all_idx[:SUBSET_N]
    val_idx  = all_idx[SUBSET_N: SUBSET_N + SUBSET_VAL]

    tr_loader  = DataLoader(Subset(full_ds, tr_idx),  batch_size=batch_size,
                            shuffle=True,  num_workers=0)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model     = UNet(base=base_ch).to(DEVICE)
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

    elapsed  = time.time() - t0
    best_miou = max(history['val_miou'])
    best_acc  = max(history['val_acc'])
    print(f'  -> best mIoU={best_miou:.4f}  pix_acc={best_acc:.4f}  ({elapsed/60:.1f} min)')

    plot_curves(history, exp_name)
    plot_samples(model, val_loader, exp_name)

    return {
        'epochs': epochs, 'lr': lr, 'base_channels': base_ch,
        'val_miou':       round(best_miou, 4),
        'val_pixel_acc':  round(best_acc, 4),
        'elapsed_min':    round(elapsed / 60, 2),
    }


# ==== Main ============

EXPERIMENTS = [
    # (name, epochs, lr, base_channels)
    ('PET_exp1_baseline',     5,  1e-3, 16),
    ('PET_exp2_more_epochs',  10, 5e-4, 16),
    ('PET_exp3_larger_model', 5,  1e-3, 32),
]

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Train subset: {SUBSET_N}  Val subset: {SUBSET_VAL}')

    all_results = {}
    for name, ep, lr, base in EXPERIMENTS:
        res = run_experiment(name, ep, lr, base)
        all_results[name] = res

    print('\n' + '='*70)
    print(f'{"Experiment":<28} {"Epochs":>6} {"LR":>8} {"BaseCh":>7} {"mIoU":>7} {"PixAcc":>8}')
    print('-'*70)
    for name, r in all_results.items():
        print(f'{name:<28} {r["epochs"]:>6} {r["lr"]:>8.0e} {r["base_channels"]:>7} '
              f'{r["val_miou"]:>7.4f} {r["val_pixel_acc"]:>8.4f}')

    with open(RESULTS / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # comparison plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names  = list(all_results.keys())
    mious  = [all_results[n]['val_miou'] * 100 for n in names]
    bars   = ax.bar(names, mious, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set(title='Oxford-IIIT Pet - mIoU ανά πείραμα',
           ylabel='mIoU (%)', ylim=(0, 100))
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS / 'comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f'\nΑποτελέσματα: {RESULTS}')
