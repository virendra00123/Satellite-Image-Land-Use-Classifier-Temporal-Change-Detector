"""
Bonus D — Class imbalance experiment.

Downsamples two classes to 20% of their original size, retrains, and
compares per-class F1 against the balanced baseline. Applies one
mitigation (weighted loss, by default) and reports before/after.

    python -m src.imbalance_experiment --checkpoint checkpoints/transfer_resnet18.pt \
        --classes River SeaLake --mitigation weighted_loss
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import config, utils
from src.model import TransferNet
from src.data_pipeline import EuroSATDataset, spatial_block_split
from src.train_baseline import run_epoch


def downsample_indices(dataset, classes_to_downsample, frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    keep = []
    per_class = {}
    for i, (_, label, _) in enumerate(dataset.samples):
        per_class.setdefault(label, []).append(i)
    target_names = {config.EUROSAT_CLASSES.index(c) for c in classes_to_downsample}
    for label, idxs in per_class.items():
        if label in target_names:
            n_keep = max(1, int(len(idxs) * frac))
            keep += list(rng.choice(idxs, size=n_keep, replace=False))
        else:
            keep += idxs
    return keep


def train_quick(model, train_ds, val_ds, device, epochs, class_weights=None, sampler=None):
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                               shuffle=(sampler is None), sampler=sampler,
                               num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PHASE2_LR)
    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"  epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_acc={val_acc:.4f}")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist(); y_pred += preds.tolist()
    return utils.per_class_f1(y_true, y_pred, config.EUROSAT_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="pretrained transfer checkpoint to fine-tune further from")
    parser.add_argument("--classes", nargs=2, default=["River", "SeaLake"])
    parser.add_argument("--mitigation", choices=["weighted_loss", "oversampling", "none"],
                         default="weighted_loss")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    utils.set_seed(config.SEED)
    device = utils.get_device()

    train_idx, val_idx, _ = spatial_block_split(config.EUROSAT_DIR)
    full_train = EuroSATDataset(config.EUROSAT_DIR, train_idx, train=True)
    val_ds = EuroSATDataset(config.EUROSAT_DIR, val_idx, train=False)

    kept = downsample_indices(full_train, args.classes, frac=0.2)
    imbalanced_train = EuroSATDataset(config.EUROSAT_DIR,
                                       [train_idx[i] for i in kept], train=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")

    print(f"== Imbalanced training (no mitigation): downsampled {args.classes} to 20% ==")
    model_a = TransferNet(backbone, config.NUM_CLASSES, pretrained=False).to(device)
    model_a.load_state_dict(ckpt["model_state"])
    per_class_before, macro_before = train_quick(model_a, imbalanced_train, val_ds, device, args.epochs)

    print(f"\n== Imbalanced training + mitigation ({args.mitigation}) ==")
    model_b = TransferNet(backbone, config.NUM_CLASSES, pretrained=False).to(device)
    model_b.load_state_dict(ckpt["model_state"])

    class_weights, sampler = None, None
    if args.mitigation == "weighted_loss":
        counts = np.bincount([label for _, label, _ in imbalanced_train.samples],
                              minlength=config.NUM_CLASSES)
        weights = 1.0 / np.maximum(counts, 1)
        class_weights = torch.tensor(weights / weights.sum() * config.NUM_CLASSES,
                                      dtype=torch.float32, device=device)
    elif args.mitigation == "oversampling":
        counts = np.bincount([label for _, label, _ in imbalanced_train.samples],
                              minlength=config.NUM_CLASSES)
        sample_weights = [1.0 / counts[label] for _, label, _ in imbalanced_train.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                         replacement=True)

    per_class_after, macro_after = train_quick(model_b, imbalanced_train, val_ds, device,
                                                args.epochs, class_weights=class_weights,
                                                sampler=sampler)

    print(f"\nmacro-F1 without mitigation: {macro_before:.4f}")
    print(f"macro-F1 with {args.mitigation}: {macro_after:.4f}")
    for c in args.classes:
        print(f"  {c}: before={per_class_before[c]:.4f}  after={per_class_after[c]:.4f}")

    analysis = (
        "Class Imbalance Experiment\n" + "=" * 26 + "\n\n"
        f"Downsampled classes: {args.classes} (kept 20% of original tiles)\n"
        f"Mitigation applied: {args.mitigation}\n\n"
        f"macro-F1 without mitigation: {macro_before:.4f}\n"
        f"macro-F1 with mitigation:    {macro_after:.4f}\n\n"
        "Per-class F1 for the downsampled classes:\n" +
        "\n".join(f"  {c}: before={per_class_before[c]:.4f} -> after={per_class_after[c]:.4f}"
                   for c in args.classes) +
        "\n\nDiscussion: downsampling the two target classes to 20% of their "
        "original size starves the model of examples for those categories, "
        "which typically shows up as a per-class F1 drop for exactly those "
        "classes while the macro-F1 average masks part of the damage. The "
        f"chosen mitigation ({args.mitigation}) rebalances the effective "
        "training signal — weighted loss upweights the gradient contribution "
        "of minority-class examples, while oversampling repeats them so each "
        "epoch sees them proportionally as often as majority classes — and "
        "should recover most, though rarely all, of the per-class F1 lost "
        "to downsampling.\n"
    )
    out_path = config.REPORT_DIR / "imbalance_experiment.txt"
    out_path.write_text(analysis)
    utils.save_json({
        "classes": args.classes, "mitigation": args.mitigation,
        "macro_f1_before": macro_before, "macro_f1_after": macro_after,
        "per_class_f1_before": per_class_before, "per_class_f1_after": per_class_after,
    }, config.REPORT_DIR / "imbalance_experiment_metrics.json")
    print(f"\n1-page analysis saved to {out_path}")


if __name__ == "__main__":
    main()
