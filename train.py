import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file on Jetson
import matplotlib.pyplot as plt
from dataset import LaneDataset
from model import build_model

# ── Config ─────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 30
BATCH_SIZE = 2
IMG_SIZE   = (640, 360)
LR         = 1e-4
SAVE_PATH  = "best_model.pth"
DATA_DIR   = "data"
VAL_SPLIT  = 0.2   # 80/20 train/val

def iou(pred, target, cls=1):
    pred   = (pred == cls)
    target = (target == cls)
    inter  = (pred & target).sum().float()
    union  = (pred | target).sum().float()
    return (inter / union).item() if union > 0 else 1.0

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            out  = model(images)["out"]
            loss = criterion(out, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = out.argmax(dim=1)
            total_loss += loss.item() * len(images)
            total_iou  += sum(iou(preds[i], masks[i]) for i in range(len(images)))
            n += len(images)
    return total_loss / n, total_iou / n


def main():
    # Split indices 80/20 with a fixed seed for reproducibility
    full     = LaneDataset(DATA_DIR, size=IMG_SIZE, augment=False)
    n_val    = max(1, int(VAL_SPLIT * len(full)))
    n_train  = len(full) - n_val
    train_sub, val_sub = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    # Re-wrap train indices with augmentation enabled
    train_set = Subset(LaneDataset(DATA_DIR, size=IMG_SIZE, augment=True),  train_sub.indices)
    val_set   = Subset(LaneDataset(DATA_DIR, size=IMG_SIZE, augment=False), val_sub.indices)

    # drop_last=True prevents a single-sample batch which breaks BatchNorm in ASPP
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Device: {DEVICE}")

    model     = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss,   val_iou   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss {train_loss:.4f} iou {train_iou:.3f}  |  "
              f"val loss {val_loss:.4f} iou {val_iou:.3f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → saved (best val IoU {best_iou:.3f})")

    print(f"\nDone. Best val IoU: {best_iou:.3f}  Weights: {SAVE_PATH}")

    # ── Save training curves ───────────────────────────────────────────────────
    epochs = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_ious, label="Train IoU")
    ax2.plot(epochs, val_ious,   label="Val IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU per Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")

if __name__ == "__main__":
    main()
