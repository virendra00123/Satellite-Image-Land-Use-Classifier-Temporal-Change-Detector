"""
Module 1 — Land-Use Classifier via two-phase transfer learning.

Phase 1: freeze backbone, train classifier head only, 3 epochs.
Phase 2: unfreeze last 2 conv blocks, LR / 10, 5 more epochs.

    python -m src.train_transfer --backbone resnet18 \
        --phase1-epochs 3 --phase2-epochs 5
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config, utils
from src.model import TransferNet
from src.data_pipeline import EuroSATDataset, spatial_block_split
from src.train_baseline import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--phase1-epochs", type=int, default=config.PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=config.PHASE2_EPOCHS)
    parser.add_argument("--phase1-lr", type=float, default=config.PHASE1_LR)
    parser.add_argument("--phase2-lr", type=float, default=config.PHASE2_LR)
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

    model = TransferNet(args.backbone, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    phase_boundaries = {}

    # ---------------------------------------------------------- Phase 1 ----
    print("== Phase 1: frozen backbone, classifier head only ==")
    model.freeze_backbone()
    for p in model.head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.trainable_param_groups(args.phase1_lr))
    for epoch in range(args.phase1_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        print(f"[phase1] epoch {epoch+1}/{args.phase1_epochs} "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    frozen_val_acc = history["val_acc"][-1]
    phase_boundaries["phase1_end"] = len(history["train_loss"])

    # ---------------------------------------------------------- Phase 2 ----
    print("== Phase 2: unfreeze last 2 conv blocks, LR / 10 ==")
    model.unfreeze_last_n_blocks(config.UNFREEZE_LAST_N_BLOCKS)
    optimizer = torch.optim.Adam(model.trainable_param_groups(args.phase2_lr))
    for epoch in range(args.phase2_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        print(f"[phase2] epoch {epoch+1}/{args.phase2_epochs} "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    unfrozen_val_acc = history["val_acc"][-1]

    # ------------------------------------------------------- final metrics
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist(); y_pred += preds.tolist()
    per_class, macro = utils.per_class_f1(y_true, y_pred, config.EUROSAT_CLASSES)

    print(f"\nFrozen (phase 1) val acc: {frozen_val_acc:.4f}")
    print(f"Unfrozen (phase 2) val acc: {unfrozen_val_acc:.4f}")
    print(f"Fine-tuned macro-F1: {macro:.4f}")

    utils.plot_loss_curves(history, config.FIGURE_DIR / f"transfer_{args.backbone}_curves.png",
                            title=f"Transfer learning ({args.backbone})")
    utils.plot_confusion_matrix(y_true, y_pred, config.EUROSAT_CLASSES,
                                 config.FIGURE_DIR / f"transfer_{args.backbone}_confusion_eurosat.png",
                                 title=f"{args.backbone} — EuroSAT val")
    utils.save_json({
        "backbone": args.backbone,
        "frozen_val_acc": frozen_val_acc,
        "unfrozen_val_acc": unfrozen_val_acc,
        "per_class_f1": per_class,
        "macro_f1": macro,
        "history": history,
        "phase_boundaries": phase_boundaries,
    }, config.REPORT_DIR / f"transfer_{args.backbone}_metrics.json")
    utils.save_checkpoint(model, config.CHECKPOINT_DIR / f"transfer_{args.backbone}.pt",
                           backbone=args.backbone, macro_f1=macro)
    print(f"\nSaved checkpoint to {config.CHECKPOINT_DIR / f'transfer_{args.backbone}.pt'}")


if __name__ == "__main__":
    main()
