"""
Module 2 (part 1) — embedding extraction and T1/T2 region split.

Reuses Module 1's fine-tuned backbone as a feature extractor (classifier
head stripped) to produce 512-d embeddings for every tile. EuroSAT has no
native repeat imagery, so we simulate a two-period time series by
partitioning it into geographic "regions" (reusing the same block_id
grouping as the spatial split) and randomly assigning whole regions to a
T1 (before) or T2 (after) split, mirroring what a real before/after
acquisition would look like at the region level.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data_pipeline import EuroSATDataset


def extract_embeddings(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            emb = model(x.to(device), return_embedding=True).cpu().numpy()
            embeddings.append(emb)
            labels += y.tolist()
    return np.concatenate(embeddings, axis=0), np.array(labels)


def extract_all_embeddings(model, dataset_root, device):
    """Convenience wrapper: extract embeddings + labels for every tile in
    EuroSAT (used by change_detector.py and the bonus embedding_viz.py)."""
    ds = EuroSATDataset(dataset_root, train=False)
    embeddings, labels = extract_embeddings(model, ds, device)
    paths = [str(p) for p, _, _ in ds.samples]
    return embeddings, labels, paths
