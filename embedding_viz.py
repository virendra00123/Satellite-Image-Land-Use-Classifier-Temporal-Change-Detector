"""
Bonus C — Embedding visualization.

Projects all 27,000 EuroSAT embeddings to 2D with t-SNE or UMAP, colored
by class, comparing the scratch-trained baseline's penultimate features
against the fine-tuned backbone's embeddings side by side.

    python -m src.embedding_viz --checkpoint checkpoints/transfer_resnet18.pt \
        --baseline-checkpoint checkpoints/baseline.pt --method umap
"""
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src import config, utils
from src.evaluate import load_model
from src.baseline_cnn import BaselineCNN
from src.embeddings import extract_all_embeddings


class BaselineFeatureExtractor(nn.Module):
    """Wraps BaselineCNN to expose its 128-d pre-classifier features
    through the same (x, return_embedding=True) interface as TransferNet,
    so extract_all_embeddings works unmodified on either model."""

    def __init__(self, baseline_model):
        super().__init__()
        self.features = baseline_model.features
        self.pool = baseline_model.pool

    def forward(self, x, return_embedding=True):
        return torch.flatten(self.pool(self.features(x)), 1)


def project_2d(embeddings, method="umap"):
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, init="pca")
    return reducer.fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="fine-tuned transfer model checkpoint")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    args = parser.parse_args()

    device = utils.get_device()

    print("Extracting fine-tuned embeddings...")
    ft_model, backbone = load_model(args.checkpoint, device)
    ft_emb, labels, _ = extract_all_embeddings(ft_model, config.EUROSAT_DIR, device)

    print("Extracting scratch-baseline embeddings...")
    ckpt = torch.load(args.baseline_checkpoint, map_location=device)
    base_model = BaselineCNN(config.NUM_CLASSES).to(device)
    base_model.load_state_dict(ckpt["model_state"])
    base_extractor = BaselineFeatureExtractor(base_model).to(device).eval()
    base_emb, _, _ = extract_all_embeddings(base_extractor, config.EUROSAT_DIR, device)

    print(f"Projecting to 2D with {args.method}...")
    ft_2d = project_2d(ft_emb, args.method)
    base_2d = project_2d(base_emb, args.method)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, coords, title in (
        (axes[0], base_2d, "Scratch baseline CNN embeddings"),
        (axes[1], ft_2d, f"Fine-tuned {backbone} embeddings"),
    ):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=4, alpha=0.6)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    handles, _ = scatter.legend_elements()
    fig.legend(handles, config.EUROSAT_CLASSES, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"Embedding space comparison ({args.method.upper()})")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = config.FIGURE_DIR / f"embedding_viz_{args.method}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
