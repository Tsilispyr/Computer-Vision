"""
Παράδειγμα 3 - Συγκριτική αξιολόγηση CNN vs ViT στο CIFAR-10
SmallCNN  : 3 conv blocks + FC head
TinyViT   : patch_size=4, embed_dim=64, 4 heads, 4 transformer layers
3 πειράματα για κάθε αρχιτεκτονική (διαφορετικές υπερ-παράμετροι)
"""

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# == Paths ======
ROOT    = Path(__file__).parent
DATA    = ROOT / 'data' / 'cifar'
RESULTS = ROOT / 'results' / 'ex3'
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBSET_TRAIN = 5000   # από 50 000, επαρκές για σύγκριση σε CPU
SUBSET_VAL   = 1000   # από 10 000
CLASSES      = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

# == Data ========
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_loaders(batch_size: int = 64):
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tf_val = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = torchvision.datasets.CIFAR10(DATA, train=True,  download=True, transform=tf_train)
    val_ds   = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=tf_val)

    rng       = np.random.default_rng(42)
    tr_idx    = rng.choice(len(train_ds), SUBSET_TRAIN, replace=False)
    val_idx   = rng.choice(len(val_ds),   SUBSET_VAL,   replace=False)

    tr_loader  = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(Subset(val_ds, val_idx),  batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return tr_loader, val_loader


# == Models =========

class SmallCNN(nn.Module):
    """3-block CNN με BatchNorm + Dropout."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch: int = 4, in_ch: int = 3, dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)


class TinyViT(nn.Module):
    """Ελαφρύ Vision Transformer για CIFAR-10 (32×32 εικόνες)."""
    def __init__(self, img_size: int = 32, patch: int = 4, dim: int = 64,
                 heads: int = 4, layers: int = 4, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = PatchEmbed(patch, 3, dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm        = nn.LayerNorm(dim)
        self.head        = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B  = x.size(0)
        x  = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1) + self.pos_embed
        x  = self.norm(self.transformer(x)[:, 0])
        return self.head(x)


# == Training helpers ===========

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)
    return loss_sum / n, correct / n


# == Plotting ============

def plot_curves(history: dict, exp_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'],   label='Val')
    ax1.set(title=f'{exp_name} - Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']],   label='Val')
    ax2.set(title=f'{exp_name} - Accuracy (%)', xlabel='Epoch', ylabel='Accuracy %')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(exp_name, fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS / f'{exp_name}_curves.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_samples(model, loader, exp_name: str):
    mean = np.array(CIFAR_MEAN); std = np.array(CIFAR_STD)
    imgs, labels = next(iter(loader))
    model.eval()
    with torch.no_grad():
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
    n = min(16, imgs.size(0))
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for i in range(n):
        ax = axes[i // 8, i % 8]
        img = np.clip(imgs[i].numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        ax.imshow(img)
        ok = preds[i].item() == labels[i].item()
        ax.set_title(f'{CLASSES[preds[i]]}', color='green' if ok else 'red', fontsize=7)
        ax.axis('off')
    plt.suptitle(f'{exp_name} - Δείγματα (πράσινο=σωστό)', fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS / f'{exp_name}_samples.png', dpi=100, bbox_inches='tight')
    plt.close()


# == Experiment runner ==============

def run_experiment(model_fn, exp_name: str, epochs: int, lr: float,
                   dropout: float = 0.3, batch_size: int = 64,
                   weight_decay: float = 1e-4) -> dict:
    print(f'\n !START! {exp_name}  epochs={epochs}  lr={lr}  dropout={dropout}')
    tr_loader, val_loader = get_loaders(batch_size)
    model     = model_fn(dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = train_epoch(model, tr_loader, criterion, optimizer)
        vl, va = eval_epoch(model,  val_loader, criterion)
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        print(f'  ep {ep:02d}/{epochs}  train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{va:.3f}')

    elapsed = time.time() - t0
    best_acc = max(history['val_acc'])
    print(f'  -> best val_acc={best_acc:.4f}  ({elapsed/60:.1f} min)')
    plot_curves(history, exp_name)
    plot_samples(model, val_loader, exp_name)

    return {
        'epochs': epochs, 'lr': lr, 'dropout': dropout,
        'val_acc': round(best_acc, 4),
        'final_val_acc': round(history['val_acc'][-1], 4),
        'elapsed_min': round(elapsed / 60, 2),
        'history': history,
    }


# == Main ========

EXPERIMENTS_CNN = [
    # (name, epochs, lr, dropout)
    ('CNN_exp1_baseline',  10, 1e-3,  0.3),
    ('CNN_exp2_more_epochs', 20, 5e-4, 0.3),
    ('CNN_exp3_high_dropout', 10, 1e-3, 0.5),
]
EXPERIMENTS_VIT = [
    ('ViT_exp1_baseline',  10, 1e-3,  0.1),
    ('ViT_exp2_more_epochs', 20, 5e-4, 0.1),
    ('ViT_exp3_higher_lr', 10, 2e-3,  0.1),
]


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    all_results = {}

    for name, ep, lr, dr in EXPERIMENTS_CNN:
        res = run_experiment(SmallCNN, name, ep, lr, dr)
        all_results[name] = {k: v for k, v in res.items() if k != 'history'}

    for name, ep, lr, dr in EXPERIMENTS_VIT:
        res = run_experiment(TinyViT, name, ep, lr, dr)
        all_results[name] = {k: v for k, v in res.items() if k != 'history'}

    # == Summary ===============
    print('\n' + '='*65)
    print(f'{"Experiment":<28} {"Epochs":>6} {"LR":>8} {"Drop":>5} {"ValAcc":>8}')
    print('-'*65)
    for name, r in all_results.items():
        print(f'{name:<28} {r["epochs"]:>6} {r["lr"]:>8.0e} {r["dropout"]:>5.1f} {r["val_acc"]:>8.4f}')

    with open(RESULTS / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # == Comparison bar chart ==============
    names  = list(all_results.keys())
    accs   = [all_results[n]['val_acc'] * 100 for n in names]
    colors = ['steelblue'] * 3 + ['darkorange'] * 3

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Ακρίβεια Επικύρωσης (%)')
    ax.set_title('Σύγκριση SmallCNN vs TinyViT - CIFAR-10', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='steelblue', label='SmallCNN'),
                       Patch(color='darkorange', label='TinyViT')])
    plt.tight_layout()
    fig.savefig(RESULTS / 'comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f'\nΑποτελέσματα: {RESULTS}')
