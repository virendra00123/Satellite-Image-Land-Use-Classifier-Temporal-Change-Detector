"""
Deliverable #3 evaluation: per-class F1, macro-F1, confusion matrix on
both EuroSAT validation and the UC Merced holdout.

    python -m src.evaluate --checkpoint checkpoints/transfer_resnet18.pt
"""
import argparse

import torch
from torch.utils.data import DataLoader

from src import config, utils
from src.model import TransferNet
from src.data_pipeline import EuroSATDataset, UCMercedDataset, spatial_block_split


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")
    model = TransferNet(backbone, config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, backbone


def eval_eurosat(model, device):
    _, val_idx, _ = spatial_block_split(config.EUROSAT_DIR)
    val_ds = EuroSATDataset(config.EUROSAT_DIR, val_idx, train=False)
    loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist(); y_pred += preds.tolist()
    return y_true, y_pred


def eval_ucmerced(model, device):
    ds = UCMercedDataset(config.UCMERCED_DIR)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
    y_true, y_pred, unmapped = [], [], 0
    with torch.no_grad():
        for x, y, ucm_cls in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y.tolist(), preds.tolist()):
                if t == -1:
                    unmapped += 1
                    continue
                y_true.append(t); y_pred.append(p)
    return y_true, y_pred, unmapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = utils.get_device()
    model, backbone = load_model(args.checkpoint, device)

    print("== EuroSAT validation ==")
    yt, yp = eval_eurosat(model, device)
    per_class, macro = utils.per_class_f1(yt, yp, config.EUROSAT_CLASSES)
    print(f"macro-F1: {macro:.4f}")
    for cls, f1 in per_class.items():
        print(f"  {cls:24s} F1={f1:.4f}")
    utils.plot_confusion_matrix(yt, yp, config.EUROSAT_CLASSES,
                                 config.FIGURE_DIR / f"eval_{backbone}_eurosat_cm.png",
                                 title=f"{backbone} — EuroSAT val")

    print("\n== UC Merced holdout (mapped classes only) ==")
    yt2, yp2, unmapped = eval_ucmerced(model, device)
    per_class2, macro2 = utils.per_class_f1(yt2, yp2, config.EUROSAT_CLASSES)
    print(f"macro-F1: {macro2:.4f}  ({unmapped} unmapped UCM images excluded)")
    for cls, f1 in per_class2.items():
        print(f"  {cls:24s} F1={f1:.4f}")
    utils.plot_confusion_matrix(yt2, yp2, config.EUROSAT_CLASSES,
                                 config.FIGURE_DIR / f"eval_{backbone}_ucmerced_cm.png",
                                 title=f"{backbone} — UC Merced holdout")

    utils.save_json({
        "backbone": backbone,
        "eurosat": {"macro_f1": macro, "per_class_f1": per_class},
        "ucmerced": {"macro_f1": macro2, "per_class_f1": per_class2, "unmapped_excluded": unmapped},
    }, config.REPORT_DIR / f"eval_{backbone}_full.json")


if __name__ == "__main__":
    main()
