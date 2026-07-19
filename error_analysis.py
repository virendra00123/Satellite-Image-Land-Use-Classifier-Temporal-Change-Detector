"""
Deliverable #7 — Error analysis: top-5 misclassified pairs shown
visually, with a hypothesis for each failure.

"Misclassified pairs" here means (true_class, predicted_class) pairs
ranked by how often that confusion occurs on the UC Merced holdout (the
brief's harder, out-of-distribution evaluation) — the 5 most frequent
confusions, each illustrated with one representative example.

    python -m src.error_analysis --checkpoint checkpoints/transfer_resnet18.pt
"""
import argparse
from collections import Counter

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from src import config, utils
from src.evaluate import load_model
from src.data_pipeline import UCMercedDataset

# Fill in / edit these after inspecting the actual confusions your model
# produces — placeholders describe the *kind* of reasoning expected, not
# a fixed final answer.
HYPOTHESIS_TEMPLATES = {
    ("Residential", "Industrial"): "Dense low-rise rooftops share the same "
        "flat, rectangular, high-albedo texture as small industrial buildings "
        "at this resolution; the model likely relies on roof texture more "
        "than surrounding context.",
    ("Highway", "River"): "Both classes are long, thin, low-texture linear "
        "features cutting across a tile — a straight paved road and a "
        "straight water channel can produce a similar edge/gradient signature.",
    ("Pasture", "AnnualCrop"): "Both are flat green/brown fields; the "
        "distinguishing cue (crop rows vs. uniform grass) is a fine-grained "
        "texture detail that gets lost after downsampling to 224x224.",
    ("PermanentCrop", "AnnualCrop"): "Orchard/vineyard rows and row-cropped "
        "fields look nearly identical from directly overhead without a "
        "wider spatial context showing tree canopies.",
    ("HerbaceousVegetation", "Forest"): "Sparse shrubland and young/thin "
        "forest canopy both present as mottled green texture; the model "
        "may need canopy density cues that are ambiguous in a single tile.",
}
DEFAULT_HYPOTHESIS = ("Visually similar texture/color statistics between "
                      "these two classes at this tile resolution; likely "
                      "needs either higher resolution input or additional "
                      "spectral bands beyond RGB to disambiguate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = utils.get_device()
    model, backbone = load_model(args.checkpoint, device)

    ds = UCMercedDataset(config.UCMERCED_DIR)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

    confusions = Counter()
    examples = {}  # (true, pred) -> (path, ucm_class)
    idx = 0
    with torch.no_grad():
        for x, y, ucm_cls in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for t, p, cls_name in zip(y.tolist(), preds.tolist(), ucm_cls):
                path, _, _ = ds.samples[idx]
                idx += 1
                if t == -1 or t == p:
                    continue
                key = (config.EUROSAT_CLASSES[t], config.EUROSAT_CLASSES[p])
                confusions[key] += 1
                if key not in examples:
                    examples[key] = str(path)

    top5 = confusions.most_common(5)
    print("Top-5 misclassified (true -> predicted) pairs on UC Merced holdout:")
    for (true_c, pred_c), count in top5:
        print(f"  {true_c} -> {pred_c}: {count} occurrences")

    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4.5))
    if len(top5) == 1:
        axes = [axes]
    report_lines = ["Error Analysis — Top-5 Misclassified Pairs\n" + "=" * 43 + "\n"]
    for ax, ((true_c, pred_c), count) in zip(axes, top5):
        img_path = examples[(true_c, pred_c)]
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{true_c} -> {pred_c}\n(n={count})", fontsize=9)
        hyp = HYPOTHESIS_TEMPLATES.get((true_c, pred_c), DEFAULT_HYPOTHESIS)
        report_lines.append(f"\n{true_c} misclassified as {pred_c} ({count}x)\n"
                             f"Hypothesis: {hyp}\n")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "error_analysis_top5.png", dpi=150)
    plt.close(fig)

    out_path = config.REPORT_DIR / "error_analysis.txt"
    out_path.write_text("\n".join(report_lines))
    print(f"\nFigure + write-up saved to {config.FIGURE_DIR}/error_analysis_top5.png "
          f"and {out_path}")


if __name__ == "__main__":
    main()
