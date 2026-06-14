from pathlib import Path

# =========================
# 用户配置区
# 直接改这里，然后在 IDE 里运行 train / sweep 脚本即可
# =========================
ROOT = Path(__file__).resolve().parents[2]
HANXUE_ROOT = ROOT / "src" / "hanxue"
OUTPUT_ROOT = ROOT / "test" / "train_output"
CACHE_ROOT = ROOT / "test" / "cache"

# =========================
# Class Config
# =========================
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
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}
OLD5_CLASSES = ["person", "car", "building", "tree", "animal"]
RARE_CLASSES = ["trash can", "window", "door", "fence", "pole_light", "motorcycle"]
RARE_BALANCED_OLD_CLASSES = list(OLD5_CLASSES)
RARE_BALANCED_RARE_CLASSES = list(RARE_CLASSES)

PROMPT_THRESHOLDS = {
    "person": 0.65,
    "car": 0.65,
    "building": 0.65,
    "tree": 0.55,
    "animal": 0.50,
    "trash can": 0.45,
    "window": 0.45,
    "door": 0.45,
    "fence": 0.40,
    "pole_light": 0.35,
    "motorcycle": 0.45,
}
VAL_THRESHOLDS = dict(PROMPT_THRESHOLDS)
CONF_FILTER = dict(PROMPT_THRESHOLDS)

CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 1.05,
    "building": 1.0,
    "tree": 1.0,
    "animal": 1.0,
    "trash can": 2.0,
    "window": 1.7,
    "door": 1.6,
    "fence": 1.8,
    "pole_light": 2.0,
    "motorcycle": 1.9,
}

PROMPT_PROTOTYPES = {
    "person": ["person", "people", "pedestrian", "human", "人", "行人"],
    "car": ["car", "vehicle", "automobile", "车辆", "汽车"],
    "building": ["building", "house", "architecture", "建筑", "楼"],
    "tree": ["tree", "vegetation", "树", "树木"],
    "animal": ["animal", "wild animal", "动物"],
    "trash can": ["trash can", "garbage bin", "trashbin", "rubbish bin", "垃圾桶"],
    "window": ["window", "窗户"],
    "door": ["door", "entrance", "门"],
    "fence": ["fence", "railing", "栏杆", "围栏"],
    "pole_light": ["pole_light", "pole light", "street light", "lamp", "light pole", "路灯", "灯杆"],
    "motorcycle": ["motorcycle", "motorbike", "摩托车"],
}

PROMPT_ALIASES = {
    alias.lower(): class_name
    for class_name, aliases in PROMPT_PROTOTYPES.items()
    for alias in aliases + [class_name]
}

THRESH_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_AREA_GRID = {
    "person": [16, 32, 64],
    "car": [16, 32, 64],
    "building": [64, 128, 256],
    "tree": [32, 64, 128],
    "animal": [8, 16, 32],
    "trash can": [4, 8, 16],
    "window": [4, 8, 16],
    "door": [8, 16, 32],
    "fence": [2, 4, 8],
    "pole_light": [1, 2, 4, 8],
    "motorcycle": [4, 8, 16],
}

# =========================
# Data
# =========================
IMAGE_ROOT = ROOT
TRAIN_JSON = ROOT / "test" / "clean_rare" / "label" / "train_label.json"
VAL_JSON = ROOT / "test" / "clean_rare" / "label" / "val-label.json"
ALL_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
TRAIN_LIST = ROOT / "test" / "trainval_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
TOKENIZER_DIR = HANXUE_ROOT / "weights" / "chinese_clip"
EFFICIENT_SAM_CKPT = HANXUE_ROOT / "weights" / "efficient_sam" / "efficient_sam_vitt.pt"
TEXT_CACHE_PATH = CACHE_ROOT / "text_emb_11.pt"
IMAGE_CACHE_DIR = CACHE_ROOT / "image_emb_11"

# =========================
# Train
# =========================
RUN_NAME = "ESAM-CCLIP-11-rare-balanced-mid"
MODEL_TYPE = "esam_chineseclip_decoder_11_rare_balanced_mid"
DEVICE = "cuda"
IMG_SIZE = 768
ESAM_INPUT_SIZE = 1024
BATCH_SIZE = 4
EPOCHS = 4
WORKERS = 4
SEED = 42
AMP = True

FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
TRAIN_DECODER_ONLY = True
USE_PROMPT_PROTOTYPE = True
USE_IMAGE_CACHE = False
BUILD_IMAGE_CACHE = False
REBUILD_IMAGE_CACHE = False
REBUILD_TEXT_CACHE = False

DECODER_LR = 1e-4
IMAGE_LR = 0.0
TEXT_LR = 0.0
WEIGHT_DECAY = 0.03
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.01
GRAD_CLIP = 1.0

NEGATIVE_SAMPLE_RATIO = 0.03
NEGATIVE_SAMPLE_WEIGHT = 0.20
INCLUDE_NEGATIVE_SAMPLES = True
VAL_INCLUDE_NEGATIVE_SAMPLES = False
OLD_CLASS_SAMPLE_RATIO = 0.55
RARE_CLASS_KEEP_RATIO = 1.0
RARE_OVERSAMPLE = {
    "trash can": 3,
    "window": 2,
    "door": 2,
    "fence": 3,
    "pole_light": 3,
    "motorcycle": 2,
}
PROMPT_AUG_PROB = 0.0
HFLIP_PROB = 0.5
LOSS_WEIGHT_FLOOR = 0.7

BCE_WEIGHT = 1.0
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

POSTPROCESS_DEFAULT = {
    "person": {"min_area": 16, "fill_holes": False},
    "car": {"min_area": 16, "fill_holes": False},
    "building": {"min_area": 128, "fill_holes": True},
    "tree": {"min_area": 64, "fill_holes": True},
    "animal": {"min_area": 8, "fill_holes": False},
    "trash can": {"min_area": 4, "fill_holes": False},
    "window": {"min_area": 4, "fill_holes": False},
    "door": {"min_area": 8, "fill_holes": False},
    "fence": {"min_area": 2, "fill_holes": False},
    "pole_light": {"min_area": 1, "fill_holes": False},
    "motorcycle": {"min_area": 4, "fill_holes": False},
}
