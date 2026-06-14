from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Data
CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
]
NUM_CLASSES = len(CLASSES)

PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST = ROOT / "test" / "trainval_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT
USE_MANIFEST = False
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"

PROMPT_ALIASES = {
    "person": ["person", "people", "pedestrian"],
    "car": ["car", "vehicle"],
    "building": ["building", "house"],
    "tree": ["tree"],
    "animal": ["animal"],
    "trash can": ["trash can", "trashbin", "garbage bin"],
    "window": ["window"],
    "door": ["door"],
    "fence": ["fence"],
    "pole_light": ["pole_light", "street light", "lamp", "light pole"],
    "motorcycle": ["motorcycle", "motorbike"],
}

VAL_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "trash can": 0.55,
    "window": 0.55,
    "door": 0.55,
    "fence": 0.55,
    "pole_light": 0.50,
    "motorcycle": 0.55,
}
CONF_FILTER = VAL_THRESHOLDS
APPLY_SCORE_FILTER = False

RARE_OVERSAMPLE = {
    "animal": 2,
    "trash can": 2,
    "window": 2,
    "door": 2,
    "pole_light": 3,
    "motorcycle": 2,
}

INCLUDE_NEGATIVE_SAMPLES = True
VAL_INCLUDE_NEGATIVE_SAMPLES = False
NEGATIVE_SAMPLE_RATIO = 0.15
NEGATIVE_SAMPLE_WEIGHT = 0.30

# Model
MODEL_TYPE = "clipseg_tiny_refine"
BASE_MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"
BASE_CHECKPOINT = ROOT / "test" / "train_output" / "clipseg_xiyou_12cls_trainval_v1" / "best.pt"
RESIDUAL_SCALE = 1.0
USE_EDGE_MAP = False

# Train
IMG_SIZE = 768
BATCH_SIZE = 4
EPOCHS = 15
WORKERS = 8
SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
AMP = True

# Optimizer
LR_REFINE = 3e-4
LR_DECODER = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 0.03
LRF = 0.01
WARMUP_EPOCHS = 1
WARMUP_START_FACTOR = 0.01

# Freeze policy
FREEZE_TEXT_ENCODER = True
FREEZE_VISION_BACKBONE = True
FREEZE_CLIPSEG_DECODER = False
TRAIN_REFINE_ONLY = False

# Loss
BCE_WEIGHT = 1.0
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.5
AUX_WEIGHT = 0.3
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LOSS_WEIGHT_FLOOR = 0.7
BOUNDARY_IGNORE_WIDTH = 1
BOUNDARY_IGNORE_MIN_AREA = 64

CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 1.0,
    "building": 1.0,
    "tree": 1.2,
    "animal": 2.0,
    "trash can": 2.0,
    "window": 1.8,
    "door": 1.5,
    "fence": 1.5,
    "pole_light": 2.5,
    "motorcycle": 2.0,
}

OLD5_CLASSES = ["person", "car", "building", "tree", "animal"]
RARE_CLASSES = ["animal", "trash can", "window", "pole_light", "motorcycle"]

# Output
PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "clipseg_tiny_refine_v1"
