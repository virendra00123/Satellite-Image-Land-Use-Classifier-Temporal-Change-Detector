"""
Module 2 — Temporal Change Detector.

EuroSAT has no repeat imagery of the same location, so a labeled
before/after test set is simulated as region *pairs*:

  - "unchanged" pair: two different tiles of the SAME class (simulating
    the same land-use type photographed at T1 and T2)
  - "changed" pair:   two tiles of DIFFERENT classes (simulating a
    genuine land-cover change at that location between T1 and T2)

The fine-tuned backbone (classifier head stripped) embeds each tile to
512-d. Cosine similarity between the T1/T2 embedding of a pair is the
change score; a pair is flagged "changed" when similarity falls below a
threshold. A held-out labeled set of such pairs lets us draw a real ROC
curve and pick a justified operating point, exactly as we would if we
had genuine repeat imagery.

    python -m src.change_detector --checkpoint checkpoints/transfer_resnet18.pt
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc

from src import config, utils
from src.evaluate import load_model
from src.embeddings import extract_all_embeddings


def build_pairs(embeddings, labels, paths, n_pairs=2000, change_frac=0.5, seed=42):
    """Builds n_pairs (idx_a, idx_b, is_changed) tuples: same-class pairs
    are unchanged (0), different-class pairs are changed (1)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}

    pairs = []
    n_changed = int(n_pairs * change_frac)
    n_unchanged = n_pairs - n_changed

    for _ in range(n_unchanged):
        c = rng.choice(list(by_class.keys()))
        a, b = rng.choice(by_class[c], size=2, replace=False)
        pairs.append((a, b, 0))

    for _ in range(n_changed):
        c1, c2 = rng.choice(list(by_class.keys()), size=2, replace=False)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        pairs.append((a, b, 1))

    rng.shuffle(pairs)
    return pairs


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def select_operating_point(fpr, tpr, thresholds, target="balanced"):
    """Returns an index into thresholds for one of three named operating
    points, used both here and by the dashboard's multi-threshold toggle
    (Bonus B)."""
    youden = tpr - fpr
    if target == "balanced":
        return int(np.argmax(youden))
    if target == "high_recall":         # prioritize catching real changes
        idx = np.where(tpr >= 0.95)[0]
        return int(idx[0]) if len(idx) else int(np.argmax(tpr))
    if target == "high_precision":      # prioritize few false alarms
        idx = np.where(fpr <= 0.05)[0]
        return int(idx[-1]) if len(idx) else int(np.argmin(fpr))
    raise ValueError(target)


def plot_heatmap_pair(path_a, path_b, sim, changed_flag, out_path, threshold):
    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")
    diff = np.abs(np.array(img_a).astype(float) - np.array(img_b).astype(float)).sum(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(img_a); axes[0].set_title("T1 (before)"); axes[0].axis("off")
    axes[1].imshow(img_b); axes[1].set_title("T2 (after)"); axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="inferno")
    axes[2].set_title("Pixel-diff heatmap"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    flag_str = "CHANGED" if sim < threshold else "unchanged"
    truth_str = "changed" if changed_flag else "unchanged"
    fig.suptitle(f"cosine sim={sim:.3f}  |  flagged: {flag_str}  |  ground truth: {truth_str}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--n-heatmaps", type=int, default=5)
    args = parser.parse_args()

    device = utils.get_device()
    model, backbone = load_model(args.checkpoint, device)

    print("Extracting embeddings for all EuroSAT tiles...")
    embeddings, labels, paths = extract_all_embeddings(model, config.EUROSAT_DIR, device)

    print(f"Building {args.n_pairs} labeled T1/T2 region pairs...")
    pairs = build_pairs(embeddings, labels, paths, n_pairs=args.n_pairs)

    sims, y_true = [], []
    for a, b, changed in pairs:
        sims.append(cosine_sim(embeddings[a], embeddings[b]))
        y_true.append(changed)
    sims = np.array(sims)
    y_true = np.array(y_true)

    # sklearn's roc_curve wants "higher score = more positive"; our
    # positive class (changed) has LOW similarity, so score on (1 - sim)
    fpr, tpr, thresholds = roc_curve(y_true, 1 - sims)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Change Detector ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "change_detector_roc.png", dpi=150)
    plt.close(fig)

    operating_points = {}
    for name in ("high_recall", "balanced", "high_precision"):
        idx = select_operating_point(fpr, tpr, thresholds, target=name)
        sim_threshold = 1 - thresholds[idx]
        operating_points[name] = {
            "similarity_threshold": float(sim_threshold),
            "fpr": float(fpr[idx]), "tpr": float(tpr[idx]),
        }
        print(f"{name:16s} sim_threshold={sim_threshold:.3f}  "
              f"FPR={fpr[idx]:.3f}  TPR={tpr[idx]:.3f}")

    chosen = operating_points["balanced"]["similarity_threshold"]
    print(f"\nChosen operating point: 'balanced' "
          f"(maximizes TPR-FPR / Youden's J) -> similarity threshold = {chosen:.3f}. "
          f"This is the default written to config; the dashboard lets a user "
          f"toggle among all three (Bonus B).")

    # 5+ sample heatmaps, mixing changed and unchanged so both are visible
    sample_idx = list(np.where(y_true == 1)[0][:3]) + list(np.where(y_true == 0)[0][:2])
    for i, pair_i in enumerate(sample_idx[:args.n_heatmaps]):
        a, b, changed = pairs[pair_i]
        plot_heatmap_pair(paths[a], paths[b], sims[pair_i], changed,
                           config.FIGURE_DIR / f"change_heatmap_{i}.png", chosen)

    utils.save_json({
        "roc_auc": roc_auc,
        "operating_points": operating_points,
        "chosen_threshold": chosen,
        "n_pairs": args.n_pairs,
    }, config.REPORT_DIR / "change_detector_metrics.json")
    print(f"\nSaved ROC curve, {len(sample_idx[:args.n_heatmaps])} heatmaps, "
          f"and metrics to {config.FIGURE_DIR} / {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
