"""
Deliverable #6 — Spatial leakage write-up.

Quantifies the accuracy gap between a naive random per-image split (which
lets near-duplicate/adjacent tiles leak between train and val) and the
spatial block split used everywhere else in this project. Trains a quick
classifier head on frozen embeddings under both splits for a fair,
fast comparison.

    python -m src.spatial_leakage --checkpoint checkpoints/transfer_resnet18.pt
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src import config, utils
from src.evaluate import load_model
from src.data_pipeline import EuroSATDataset, spatial_block_split, random_split
from src.embeddings import extract_embeddings


def eval_split(model, device, train_idx, val_idx, label):
    train_ds = EuroSATDataset(config.EUROSAT_DIR, train_idx, train=False)
    val_ds = EuroSATDataset(config.EUROSAT_DIR, val_idx, train=False)

    X_train, y_train = extract_embeddings(model, train_ds, device)
    X_val, y_val = extract_embeddings(model, val_ds, device)

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"[{label}] val accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = utils.get_device()
    model, _ = load_model(args.checkpoint, device)

    print("== Naive random split (leaky) ==")
    train_idx_r, val_idx_r, _ = random_split(config.EUROSAT_DIR)
    acc_r, f1_r = eval_split(model, device, train_idx_r, val_idx_r, "random split")

    print("\n== Spatial block split (leak-free) ==")
    train_idx_b, val_idx_b, _ = spatial_block_split(config.EUROSAT_DIR)
    acc_b, f1_b = eval_split(model, device, train_idx_b, val_idx_b, "block split")

    gap_acc = acc_r - acc_b
    gap_f1 = f1_r - f1_b
    print(f"\nLeakage gap: +{gap_acc*100:.2f} pts accuracy, "
          f"+{gap_f1*100:.2f} pts macro-F1 under the random split.")

    write_up = (
        "Spatial Leakage Experiment\n"
        "==========================\n\n"
        f"Random (leaky) split:   accuracy={acc_r:.4f}, macro-F1={f1_r:.4f}\n"
        f"Spatial block split:    accuracy={acc_b:.4f}, macro-F1={f1_b:.4f}\n"
        f"Gap attributable to leakage: {gap_acc*100:.2f} accuracy points, "
        f"{gap_f1*100:.2f} macro-F1 points.\n\n"
        "Explanation: EuroSAT tiles are cropped from a small number of "
        "underlying Sentinel-2 scenes, so tiles that are geographically "
        "adjacent (same 25-tile block in this project's grouping) can be "
        "visually near-duplicates — same field boundary, same cloud edge, "
        "same seasonal coloring. A random per-image split places some of "
        "these near-duplicates in train and their siblings in val, letting "
        "the model partly memorize local scene statistics rather than learn "
        "generalizable land-use features. The spatial block split keeps every "
        "tile from a given block entirely on one side of the split, removing "
        "this shortcut and giving a more honest estimate of real-world "
        "generalization — which is why it is used as the default split "
        "throughout this project instead of a random split.\n"
    )
    out_path = config.REPORT_DIR / "spatial_leakage_writeup.txt"
    out_path.write_text(write_up)
    utils.save_json({
        "random_split": {"accuracy": acc_r, "macro_f1": f1_r},
        "block_split": {"accuracy": acc_b, "macro_f1": f1_b},
        "gap_accuracy": gap_acc, "gap_macro_f1": gap_f1,
    }, config.REPORT_DIR / "spatial_leakage_metrics.json")
    print(f"\nWrite-up saved to {out_path}")


if __name__ == "__main__":
    main()
