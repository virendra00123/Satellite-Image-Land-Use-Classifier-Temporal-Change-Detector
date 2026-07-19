"""
Central configuration: paths, hyperparameters, and the UC Merced -> EuroSAT
class mapping used only for holdout evaluation.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EUROSAT_DIR = DATA_DIR / "EuroSAT"
UCMERCED_DIR = DATA_DIR / "UCMerced_LandUse" / "Images"
CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "reports" / "figures"

for d in (CHECKPOINT_DIR, FIGURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ constants ----
SEED = 42
IMAGE_SIZE = 64          # EuroSAT native resolution
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = "cuda"          # falls back to cpu automatically at runtime, see utils.get_device

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
NUM_CLASSES = len(EUROSAT_CLASSES)

# EuroSAT normalization stats (ImageNet-pretrained backbones expect these
# after resizing tiles up to 224x224 — see data_pipeline.get_transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------- UCM -> EuroSAT map ----
# UC Merced has 21 classes; only used to score the fine-tuned 10-way
# classifier as a holdout (not to retrain on). Each UCM class is mapped to
# its closest EuroSAT semantic equivalent. Classes with no reasonable
# equivalent are excluded from the mapped-holdout metric and reported
# separately as "unmapped" in the confusion matrix.
UCM_TO_EUROSAT = {
    "agricultural": "AnnualCrop",
    "airplane": None,
    "baseballdiamond": None,
    "beach": None,
    "buildings": "Industrial",
    "chaparral": "HerbaceousVegetation",
    "denseresidential": "Residential",
    "forest": "Forest",
    "freeway": "Highway",
    "golfcourse": "Pasture",
    "harbor": None,
    "intersection": "Highway",
    "mediumresidential": "Residential",
    "mobilehomepark": "Residential",
    "overpass": "Highway",
    "parkinglot": "Industrial",
    "river": "River",
    "runway": "Highway",
    "sparseresidential": "Residential",
    "storagetanks": "Industrial",
    "tenniscourt": None,
}

# -------------------------------------------------------- hyperparameters --
BASELINE_LR = 1e-3
PHASE1_LR = 1e-3      # frozen backbone, classifier head only
PHASE2_LR = 1e-4      # unfrozen last 2 conv blocks, LR reduced 10x
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 5
UNFREEZE_LAST_N_BLOCKS = 2

# Change detection
EMBEDDING_DIM = 512
DEFAULT_SIMILARITY_THRESHOLD = 0.80   # overwritten by ROC-selected operating point
