"""
Module: data pipeline.

- EuroSATDataset / UCMercedDataset: thin ImageFolder-style wrappers
- spatial_block_split: splits by *geographic block* (folder-index modulo
  grouping simulated over EuroSAT's tile grid) rather than a random
  per-image split, so nearby tiles cannot leak between train/val/test.
  This is what src/spatial_leakage.py quantifies against a naive random
  split.
- get_transforms: train/eval transforms, resizing 64x64 EuroSAT tiles up
  to 224x224 so ImageNet-pretrained backbones work well.
- main(--explore): saves 5 samples/class grid + class distribution plot.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src import config


def get_transforms(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class EuroSATDataset(Dataset):
    """Loads (path, label) pairs and assigns each tile a synthetic
    'block_id' derived from its filename index, used for the spatial
    block split so spatially-adjacent tiles stay in the same split."""

    def __init__(self, root: Path, indices=None, train=True, block_size=25):
        self.root = Path(root)
        self.classes = config.EUROSAT_CLASSES
        self.samples = []  # (path, label, block_id)
        for label, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.glob("*.jpg"))
            for i, f in enumerate(files):
                block_id = f"{cls}_{i // block_size}"
                self.samples.append((f, label, block_id))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def block_ids(self):
        return [s[2] for s in self.samples]


class UCMercedDataset(Dataset):
    """Holdout set. Labels are mapped through config.UCM_TO_EUROSAT;
    samples with no mapping (None) are kept but flagged unmapped=True so
    evaluate.py can report them separately instead of silently dropping
    or mis-scoring them."""

    def __init__(self, root: Path, train=False):
        self.root = Path(root)
        self.samples = []  # (path, mapped_label_or_None, ucm_class)
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            mapped = config.UCM_TO_EUROSAT.get(cls_dir.name)
            label = config.EUROSAT_CLASSES.index(mapped) if mapped else None
            for f in sorted(cls_dir.glob("*.tif")) + sorted(cls_dir.glob("*.jpg")):
                self.samples.append((f, label, cls_dir.name))
        self.transform = get_transforms(train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, ucm_class = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), (label if label is not None else -1), ucm_class


def spatial_block_split(dataset_root: Path, val_frac=0.15, test_frac=0.15, seed=42):
    """Group tiles into blocks (see EuroSATDataset), then assign whole
    blocks to train/val/test so no block straddles a split boundary."""
    rng = np.random.RandomState(seed)
    full = EuroSATDataset(dataset_root, train=False)
    blocks = sorted(set(full.block_ids()))
    rng.shuffle(blocks)

    n_val = int(len(blocks) * val_frac)
    n_test = int(len(blocks) * test_frac)
    val_blocks = set(blocks[:n_val])
    test_blocks = set(blocks[n_val:n_val + n_test])
    train_blocks = set(blocks[n_val + n_test:])

    train_idx, val_idx, test_idx = [], [], []
    for i, (_, _, b) in enumerate(full.samples):
        if b in train_blocks:
            train_idx.append(i)
        elif b in val_blocks:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def random_split(dataset_root: Path, val_frac=0.15, test_frac=0.15, seed=42):
    """Naive per-image random split — deliberately leaky, used only as
    the comparison point in src/spatial_leakage.py."""
    rng = np.random.RandomState(seed)
    full = EuroSATDataset(dataset_root, train=False)
    idx = np.arange(len(full))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    n_test = int(len(idx) * test_frac)
    return idx[n_val + n_test:].tolist(), idx[:n_val].tolist(), idx[n_val:n_val + n_test].tolist()


def plot_class_distribution(root: Path, out_path):
    counts = {}
    for cls in config.EUROSAT_CLASSES:
        cls_dir = Path(root) / cls
        counts[cls] = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values())
    ax.set_xticklabels(counts.keys(), rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("EuroSAT class distribution")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts


def plot_sample_grid(root: Path, out_path, n_per_class=5):
    fig, axes = plt.subplots(len(config.EUROSAT_CLASSES), n_per_class,
                              figsize=(n_per_class * 1.6, len(config.EUROSAT_CLASSES) * 1.6))
    for r, cls in enumerate(config.EUROSAT_CLASSES):
        files = sorted((Path(root) / cls).glob("*.jpg"))[:n_per_class]
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(files):
                ax.imshow(Image.open(files[c]))
            if c == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true",
                        help="save sample grid + class distribution plot to reports/figures/")
    args = parser.parse_args()

    if args.explore:
        plot_class_distribution(config.EUROSAT_DIR, config.FIGURE_DIR / "class_distribution.png")
        plot_sample_grid(config.EUROSAT_DIR, config.FIGURE_DIR / "sample_grid.png")
        train_idx, val_idx, test_idx = spatial_block_split(config.EUROSAT_DIR)
        print(f"Spatial block split -> train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}")
        print(f"Figures written to {config.FIGURE_DIR}")
