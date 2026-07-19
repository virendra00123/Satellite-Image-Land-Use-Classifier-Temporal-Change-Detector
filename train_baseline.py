"""
Trains the 3-layer scratch baseline CNN (Deliverable #2).
Logs train/val loss curves and reports per-class F1.

    python -m src.train_baseline --epochs 15
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config, utils
from src.baseline_cnn import BaselineCNN
from src.data_pipeline import EuroSATDataset, spatial_block_split


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=config.BASELINE_LR)
    args = parser.parse_args()

    utils.set_seed(config.SEED)
    device = utils.get_device()

    train_idx, val_idx, _ = spatial_block_split(config.EUROSAT_DIR)
    train_ds = EuroSATDataset(config.EUROSAT_DIR, train_idx, train=True)
    val_ds = EuroSATDataset(config.EUROSAT_DIR, val_idx, train=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    model = BaselineCNN(config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f}  "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    # Final per-class F1 on val
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist(); y_pred += preds.tolist()
    per_class, macro = utils.per_class_f1(y_true, y_pred, config.EUROSAT_CLASSES)
    print(f"\nBaseline macro-F1: {macro:.4f}")
    for cls, f1 in per_class.items():
        print(f"  {cls:24s} F1={f1:.4f}")

    utils.plot_loss_curves(history, config.FIGURE_DIR / "baseline_loss_curves.png",
                            title="Baseline CNN")
    utils.plot_confusion_matrix(y_true, y_pred, config.EUROSAT_CLASSES,
                                 config.FIGURE_DIR / "baseline_confusion_matrix.png",
                                 title="Baseline CNN — EuroSAT val")
    utils.save_json({"per_class_f1": per_class, "macro_f1": macro, "history": history},
                     config.REPORT_DIR / "baseline_metrics.json")
    utils.save_checkpoint(model, config.CHECKPOINT_DIR / "baseline.pt",
                           macro_f1=macro, epochs=args.epochs)
    print(f"\nSaved checkpoint to {config.CHECKPOINT_DIR / 'baseline.pt'}")


if __name__ == "__main__":
    main()
