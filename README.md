# Satellite Image Land-Use Classifier & Temporal Change Detector

A computer vision system that classifies land-use types from satellite imagery
(EuroSAT, held out on UC Merced) and detects land-cover change between two
time periods using embedding similarity, with a local Streamlit dashboard.

## Project Structure

```
satellite-project/
├── data/
│   └── download_data.py        # downloads & unpacks EuroSAT + UC Merced
├── src/
│   ├── config.py                # paths, hyperparameters, constants
│   ├── data_pipeline.py         # datasets, spatial block split, transforms
│   ├── baseline_cnn.py          # 3-layer scratch CNN
│   ├── model.py                  # ResNet-18 / EfficientNet-B0 wrapper
│   ├── train_baseline.py        # trains + logs the baseline CNN
│   ├── train_transfer.py        # two-phase fine-tuning
│   ├── embeddings.py            # embedding extraction, T1/T2 split
│   ├── change_detector.py       # cosine similarity, ROC, threshold, heatmaps
│   ├── evaluate.py              # per-class F1, macro-F1, confusion matrix
│   ├── spatial_leakage.py       # random-split vs block-split experiment
│   ├── error_analysis.py        # top-5 misclassified pairs
│   ├── gradcam.py               # Bonus A
│   ├── embedding_viz.py         # Bonus C (t-SNE/UMAP)
│   ├── imbalance_experiment.py  # Bonus D
│   └── utils.py                 # shared helpers (seed, plotting, metrics)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_change_detection.ipynb
│   └── 05_evaluation_and_analysis.ipynb
├── app/
│   └── dashboard.py              # Streamlit app (Module 3)
├── reports/
│   └── report_template.md        # fill in and export to PDF (max 8 pages)
├── checkpoints/                  # saved .pt files land here
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py    # downloads EuroSAT + UC Merced into data/
```

Everything is written to run **fully offline after this one download step** —
this satisfies the dashboard's "no internet dependency after setup" requirement.

## Running the pipeline end-to-end

```bash
# 1. Data pipeline (spatial block split, class distribution plot)
python -m src.data_pipeline --explore

# 2. Baseline CNN (the floor all results are compared against)
python -m src.train_baseline --epochs 15

# 3. Transfer learning — two-phase fine-tuning
python -m src.train_transfer --backbone resnet18 \
    --phase1-epochs 3 --phase2-epochs 5

# 4. Evaluate on EuroSAT val + UC Merced holdout
python -m src.evaluate --checkpoint checkpoints/transfer_resnet18.pt

# 5. Change detection module
python -m src.change_detector --checkpoint checkpoints/transfer_resnet18.pt

# 6. Spatial leakage write-up experiment
python -m src.spatial_leakage --checkpoint checkpoints/transfer_resnet18.pt

# 7. Error analysis (top-5 misclassified pairs)
python -m src.error_analysis --checkpoint checkpoints/transfer_resnet18.pt

# 8. Launch the dashboard
streamlit run app/dashboard.py
```

Bonus tasks (optional, each independently runnable):

```bash
python -m src.gradcam --checkpoint checkpoints/transfer_resnet18.pt          # Bonus A
# Bonus B (multi-threshold toggle) is built into app/dashboard.py — see the sidebar
python -m src.embedding_viz --checkpoint checkpoints/transfer_resnet18.pt    # Bonus C
python -m src.imbalance_experiment --checkpoint checkpoints/transfer_resnet18.pt  # Bonus D
```

## Notes on the two datasets

- **EuroSAT** (10 classes, 27,000 tiles, 64×64 Sentinel-2 RGB/MS) is the
  primary training/validation dataset and is also artificially split into
  geographic "T1"/"T2" halves to simulate a before/after time series for
  Module 2, since EuroSAT itself has no repeat imagery.
- **UC Merced Land Use** (21 classes, 2,100 images, 256×256 aerial RGB) is
  used **only as a holdout** to test generalization of the classifier — its
  extra 11 classes are mapped to the closest EuroSAT category (mapping in
  `src/config.py`) purely for the holdout evaluation table, since the brief
  only asks for classifier metrics on this set, not that the model be
  retrained on it.

## Submission checklist (from the brief)

- [ ] GitHub repository — clean, with README and requirements.txt
- [ ] All notebooks runnable top-to-bottom with no errors
- [ ] Saved model checkpoint (`.pt`) committed or linked via Git LFS
- [ ] Streamlit app tested locally before submission
- [ ] PDF report — max 8 pages, including figures
- [ ] 3-minute demo video — screen recording of the live dashboard
- [ ] Bonuses (if attempted) clearly labelled in repo and report
