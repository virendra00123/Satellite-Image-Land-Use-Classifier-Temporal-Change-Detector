"""
Module 3 — Geo-Dashboard.

Accepts two satellite tile images (before/after), shows predicted
land-use class + confidence for each, cosine similarity between their
embeddings, and a side-by-side heatmap with a change flag. Includes the
Bonus B multi-threshold toggle (high recall / balanced / high precision).

Runs fully locally with no internet dependency once the checkpoint and
packages are already on disk:

    streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config, utils               # noqa: E402
from src.model import TransferNet           # noqa: E402
from src.data_pipeline import get_transforms  # noqa: E402

st.set_page_config(page_title="Satellite Land-Use & Change Dashboard", layout="wide")


@st.cache_resource
def load_model(checkpoint_path):
    device = utils.get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")
    model = TransferNet(backbone, config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, backbone


@st.cache_data
def load_operating_points():
    path = config.REPORT_DIR / "change_detector_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["operating_points"]
    # sensible fallback if change_detector.py hasn't been run yet
    return {
        "high_recall": {"similarity_threshold": 0.90},
        "balanced": {"similarity_threshold": 0.80},
        "high_precision": {"similarity_threshold": 0.60},
    }


def classify(model, device, image: Image.Image):
    transform = get_transforms(train=False)
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        embedding = model(x, return_embedding=True)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    return config.EUROSAT_CLASSES[pred_idx], float(probs[pred_idx]), embedding


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    st.title("🛰️ Satellite Land-Use Classifier & Change Detector")
    st.caption("Upload a before/after tile pair. Runs entirely on your local model — no internet calls.")

    with st.sidebar:
        st.header("Settings")
        checkpoint = st.text_input("Checkpoint path",
                                    value=str(config.CHECKPOINT_DIR / "transfer_resnet18.pt"))
        st.markdown("**Change-detection operating point** (Bonus B)")
        threshold_choice = st.radio(
            "Toggle threshold",
            ["high_recall", "balanced", "high_precision"],
            index=1,
            help="high_recall catches more real changes at the cost of more false "
                 "alarms; high_precision does the opposite; balanced maximizes TPR-FPR.",
        )

    if not Path(checkpoint).exists():
        st.warning(f"Checkpoint not found at `{checkpoint}`. Train a model first with "
                   "`python -m src.train_transfer`, then point this field at the resulting .pt file.")
        return

    model, device, backbone = load_model(checkpoint)
    operating_points = load_operating_points()
    threshold = operating_points[threshold_choice]["similarity_threshold"]

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Before (T1) tile", type=["jpg", "jpeg", "png", "tif"])
    with col2:
        after_file = st.file_uploader("After (T2) tile", type=["jpg", "jpeg", "png", "tif"])

    if before_file and after_file:
        img_before = Image.open(before_file)
        img_after = Image.open(after_file)

        class_before, conf_before, emb_before = classify(model, device, img_before)
        class_after, conf_after, emb_after = classify(model, device, img_after)
        sim = cosine_sim(emb_before, emb_after)
        changed = sim < threshold

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_before, caption="Before (T1)", use_container_width=True)
            st.metric("Predicted class", class_before, f"{conf_before:.1%} confidence")
        with c2:
            st.image(img_after, caption="After (T2)", use_container_width=True)
            st.metric("Predicted class", class_after, f"{conf_after:.1%} confidence")
        with c3:
            arr_before = np.array(img_before.convert("RGB").resize((224, 224))).astype(float)
            arr_after = np.array(img_after.convert("RGB").resize((224, 224))).astype(float)
            diff = np.abs(arr_before - arr_after).sum(axis=2)
            st.image(diff / diff.max(), caption="Pixel-diff heatmap", clamp=True,
                     use_container_width=True)

        st.divider()
        st.subheader("Change assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cosine similarity", f"{sim:.3f}")
        m2.metric("Threshold in use", f"{threshold:.3f}", threshold_choice)
        m3.metric("Change flag", "🔴 CHANGED" if changed else "🟢 unchanged")

        if changed:
            st.error(f"Flagged as **changed** — similarity ({sim:.3f}) fell below the "
                     f"'{threshold_choice}' threshold ({threshold:.3f}).")
        else:
            st.success(f"No significant change detected — similarity ({sim:.3f}) is above "
                       f"the '{threshold_choice}' threshold ({threshold:.3f}).")
    else:
        st.info("Upload both a before and an after tile to see predictions and the change assessment.")


if __name__ == "__main__":
    main()
