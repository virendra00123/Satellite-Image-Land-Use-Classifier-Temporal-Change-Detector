"""
Bonus A — GradCAM visualization on the fine-tuned model.

Overlays a heatmap of which pixels drove each classification, for at
least 3 example tiles.

    python -m src.gradcam --checkpoint checkpoints/transfer_resnet18.pt
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src import config, utils
from src.evaluate import load_model
from src.data_pipeline import EuroSATDataset, get_transforms


class GradCAM:
    """Minimal GradCAM hooked onto the last conv block of the backbone."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # global-avg-pool grads
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(1)[0, class_idx].item()


def find_target_layer(model):
    """Last conv block, works for both supported backbones."""
    if model.backbone_name == "resnet18":
        return model.backbone.layer4[-1]
    return model.blocks[-1]  # efficientnet: last block in .features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()

    device = utils.get_device()
    model, backbone = load_model(args.checkpoint, device)
    model.eval()

    cam_engine = GradCAM(model, find_target_layer(model))

    ds = EuroSATDataset(config.EUROSAT_DIR, train=False)
    rng = np.random.RandomState(0)
    sample_positions = rng.choice(len(ds), size=args.n_examples, replace=False)

    fig, axes = plt.subplots(2, args.n_examples, figsize=(3.2 * args.n_examples, 6.4))
    for col, pos in enumerate(sample_positions):
        img_tensor, label = ds[pos]
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)

        cam, pred_class, confidence = cam_engine(x)

        path, _, _ = ds.samples[pos]
        raw_img = np.array(Image.open(path).convert("RGB").resize((224, 224))) / 255.0

        axes[0, col].imshow(raw_img)
        axes[0, col].axis("off")
        axes[0, col].set_title(f"true: {config.EUROSAT_CLASSES[label]}", fontsize=9)

        axes[1, col].imshow(raw_img)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")
        axes[1, col].set_title(
            f"pred: {config.EUROSAT_CLASSES[pred_class]} ({confidence:.2f})", fontsize=9)

    fig.suptitle("GradCAM — which pixels drove each classification")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "gradcam_examples.png", dpi=150)
    plt.close(fig)
    print(f"Saved GradCAM figure to {config.FIGURE_DIR / 'gradcam_examples.png'}")
    print(
        "\nInterpretation notes (fill in after viewing the figure):\n"
        "  - Does the heatmap concentrate on the semantically relevant region\n"
        "    (e.g. field texture for AnnualCrop, water body for River)?\n"
        "  - Or does it latch onto tile borders / compression artifacts?\n"
        "  - Compare a correct vs. an incorrect prediction's heatmap."
    )


if __name__ == "__main__":
    main()
