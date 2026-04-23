import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LaneDataset
from model import build_model

# ── Config ─────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 30
BATCH_SIZE = 4
LR         = 1e-4
SAVE_PATH  = "best_model.pth"

# Roboflow puts these folders in the zip you download
TRAIN_DIR  = "data/train"
VAL_DIR    = "data/valid"

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
    train_set = LaneDataset(TRAIN_DIR, augment=True)
    val_set   = LaneDataset(VAL_DIR,   augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Device: {DEVICE}")

    model     = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss,   val_iou   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss {train_loss:.4f} iou {train_iou:.3f}  |  "
              f"val loss {val_loss:.4f} iou {val_iou:.3f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → saved (best val IoU {best_iou:.3f})")

    print(f"\nDone. Best val IoU: {best_iou:.3f}  Weights: {SAVE_PATH}")

if __name__ == "__main__":
    main()
