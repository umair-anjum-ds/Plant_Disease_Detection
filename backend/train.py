"""
train.py — Full training pipeline for CropDiagnosticCNN
Dataset : New Plant Diseases Dataset (Kaggle)
          https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Usage:
    python train.py --data_dir ../data --epochs 15

Folder structure expected:
    data/
      train/
        Apple___Black_rot/
        Apple___Cedar_apple_rust/
        Apple___healthy/
        Corn_(maize)___Common_rust_/
        Corn_(maize)___healthy/
        Potato___Early_blight/
        Potato___Late_blight/
        Potato___healthy/
        Tomato___Bacterial_spot/
        Tomato___Leaf_Mold/
        Tomato___healthy/
      valid/
        (same 11 folders)

Training strategy:
  Phase 1 (epochs 1-5)  : backbone frozen, train heads only
  Phase 2 (epochs 6-15) : backbone unfrozen, fine-tune end-to-end at lower LR
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import CropDiagnosticCNN, CLASSES, NUM_CLASSES, HEALTHY_INDICES

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Transforms ────────────────────────────────────────────────────────────────
# ImageNet mean/std — required because MobileNetV3 was pretrained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training: augmentation to prevent overfitting
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validation / Test: NO augmentation — only resize + normalize
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """
    Loads train/ and valid/ from data_dir using ImageFolder.

    Data split (no leakage):
      - train/  -> train_loader  (augmented)
      - valid/  -> split 50/50 into val_loader and test_loader (no augmentation)
        * val   is used during training to pick the best checkpoint
        * test  is held out completely and only evaluated AFTER training ends

    The 50/50 split uses a fixed SEED so it is always deterministic and
    class-balanced (each class contributes equally to val and test).
    """
    train_ds      = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    valid_ds_full = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_tf)

    # Verify folder names match our expected CLASSES exactly
    detected = sorted(train_ds.classes)
    expected = sorted(CLASSES)
    if detected != expected:
        print("\n WARNING: folder names do not match CLASSES in model.py!")
        print(f"   Found   : {detected}")
        print(f"   Expected: {expected}")
        print("   Fix folder names or update CLASSES in model.py\n")
    else:
        print(f"  Folder names verified OK: {detected}\n")

    # Deterministic class-balanced 50/50 split of valid -> val + test
    rng = np.random.default_rng(SEED)
    val_indices, test_indices = [], []
    targets = np.array(valid_ds_full.targets)

    for cls_idx in range(NUM_CLASSES):
        idx = np.where(targets == cls_idx)[0]
        rng.shuffle(idx)
        mid = len(idx) // 2
        val_indices.extend(idx[:mid].tolist())
        test_indices.extend(idx[mid:].tolist())

    val_ds  = Subset(valid_ds_full, val_indices)
    test_ds = Subset(valid_ds_full, test_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  Train : {len(train_ds):,} images")
    print(f"  Val   : {len(val_ds):,} images  (used to save best checkpoint)")
    print(f"  Test  : {len(test_ds):,} images  (held out, evaluated after training)")

    return train_loader, val_loader, test_loader


def get_severity_targets(labels: torch.Tensor) -> torch.Tensor:
    """
    Build severity ground-truth from class labels.
    Uses explicit HEALTHY_INDICES set from model.py — not a modulo hack.
      healthy class  -> severity 0.0
      disease class  -> severity 0.7
    """
    sev = torch.tensor(
        [0.0 if l.item() in HEALTHY_INDICES else 0.7 for l in labels],
        dtype=torch.float32,
    )
    return sev.unsqueeze(1)


def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_sev, device):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        sev_gt = get_severity_targets(labels).to(device)

        optimizer.zero_grad()
        logits, sev_out = model(imgs)

        # Combined loss: classification (main) + severity (auxiliary, low weight)
        loss_cls = criterion_cls(logits, labels)
        loss_sev = criterion_sev(sev_out, sev_gt)
        loss = loss_cls + 0.3 * loss_sev

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate on a loader. No gradients, model in eval mode.
    Returns accuracy, predicted labels, true labels.
    """
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = np.mean(all_preds == all_labels)
    return acc, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, out_path: str):
    """Save a labelled confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    # Triple underscore split to match actual folder names
    short_names = [c.split("___")[1].replace("_", " ")[:18] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {out_path}")


def save_training_curves(history, epochs, out_path: str):
    """Save loss and accuracy curves as PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = range(1, epochs + 1)

    ax1.plot(x, history["train_loss"], label="train loss", color="steelblue")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(x, history["train_acc"], label="train acc", color="steelblue")
    ax2.plot(x, history["val_acc"],   label="val acc",   color="darkorange")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curves saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CropDiagnosticCNN")
    parser.add_argument("--data_dir",   default="./data",      help="Root folder with train/ and valid/")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--out_dir",    default="./artifacts", help="Where to save model + plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {args.out_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.workers
    )

    # ── Model — Phase 1: backbone frozen ──────────────────────────────────────
    model = CropDiagnosticCNN(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_sev = nn.MSELoss()

    head_params = (
        list(model.classifier_head.parameters()) +
        list(model.severity_head.parameters())
    )
    optimizer = optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Phase 2 starts at epoch 6 (or halfway if fewer epochs)
    PHASE2_START = min(6, args.epochs // 2 + 1)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc    = 0.0
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    # ── Training loop ──────────────────────────────────────────────────────────
    print("\nTraining ...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Switch to Phase 2: unfreeze backbone at lower LR
        if epoch == PHASE2_START:
            print(f"  -> Phase 2 start (epoch {epoch}): unfreezing backbone\n")
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.features.parameters(),        "lr": args.lr / 10},
                {"params": model.classifier_head.parameters(), "lr": args.lr},
                {"params": model.severity_head.parameters(),   "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - PHASE2_START + 1
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_sev, device
        )
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}  ({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✓ Best val_acc={best_val_acc:.4f} — checkpoint saved\n")

    # ── Final test evaluation (held-out set, run only once) ───────────────────
    print(f"\nLoading best checkpoint for final test evaluation ...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, y_pred, y_true = evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  BEST VAL ACC  : {best_val_acc:.4f}  ({best_val_acc * 100:.2f}%)")
    print(f"{'='*60}\n")

    # Per-class report — triple underscore split
    short = [c.split("___")[1].replace("_", " ") for c in CLASSES]
    print("Per-class classification report:")
    print(classification_report(y_true, y_pred, target_names=short))

    # ── Save plots ─────────────────────────────────────────────────────────────
    save_confusion_matrix(
        y_true, y_pred,
        os.path.join(args.out_dir, "confusion_matrix.png")
    )
    save_training_curves(
        history, args.epochs,
        os.path.join(args.out_dir, "training_curves.png")
    )

    # ── Save metadata (read by main.py for the /model-info endpoint) ──────────
    meta = {
        "test_accuracy":     round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "num_classes":       NUM_CLASSES,
        "classes":           CLASSES,
        "epochs_trained":    args.epochs,
        "model_path":        best_model_path,
    }
    meta_path = os.path.join(args.out_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved -> {meta_path}")
    print("\nTraining complete ✓")


if __name__ == "__main__":
    main()
