from pathlib import Path

# =========================
# Path
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
RARE_CLASSES = ["trash can", "door", "fence", "motorcycle", "window", "pole_light"]
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
    "car": 1.0,
    "building": 1.0,
    "tree": 1.0,
    "animal": 1.0,
    "trash can": 1.6,
    "window": 1.3,
    "door": 1.3,
    "fence": 1.4,
    "pole_light": 1.6,
    "motorcycle": 1.5,
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

TEXT_REALIGN_COMPLEX_ALIASES = {
    "car": [
        "草丛里的车",
        "被树遮挡的车",
        "远处的小车",
        "car in bushes",
        "car behind tree",
        "small distant car",
    ],
    "window": [
        "建筑上的窗户",
        "车窗",
        "building window",
        "car window",
        "window on building",
    ],
    "door": [
        "车门",
        "建筑门",
        "car door",
        "door of car",
    ],
    "pole_light": [
        "路边的灯杆",
        "远处的路灯",
        "street light pole",
        "light pole near road",
    ],
}


def _merge_prompt_prototypes(base_prototypes, extra_aliases):
    merged = {}
    for class_name, aliases in base_prototypes.items():
        deduped = []
        for alias in list(aliases) + list(extra_aliases.get(class_name, [])):
            alias = str(alias)
            if alias not in deduped:
                deduped.append(alias)
        merged[class_name] = deduped
    return merged


TEXT_REALIGN_PROMPT_PROTOTYPES = _merge_prompt_prototypes(PROMPT_PROTOTYPES, TEXT_REALIGN_COMPLEX_ALIASES)

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
VAL_JSON = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
ALL_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
TOKENIZER_DIR = HANXUE_ROOT / "weights" / "chinese_clip"
EFFICIENT_SAM_CKPT = HANXUE_ROOT / "weights" / "efficient_sam" / "efficient_sam_vitt.pt"
TEXT_CACHE_PATH = CACHE_ROOT / "text_emb_11.pt"
IMAGE_CACHE_DIR = CACHE_ROOT / "image_emb_11"
DEFAULT_RESUME_BEST = OUTPUT_ROOT / "efficientsam_easy_6cls_v1" / "efficientsam_easy_6cls_v1" / "best.pt"

# =========================
# Train
# =========================
RUN_NAME = "ESAM-CCLIP-11-split-control"
MODEL_TYPE = "esam_chineseclip_decoder_11_split_control"
DEVICE = "cuda"
IMG_SIZE = 768
ESAM_INPUT_SIZE = 1024
BATCH_SIZE = 8
EPOCHS = 5
WORKERS = 8
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

NEGATIVE_SAMPLE_RATIO = 0.02
NEGATIVE_SAMPLE_WEIGHT = 0.20
INCLUDE_NEGATIVE_SAMPLES = True
VAL_INCLUDE_NEGATIVE_SAMPLES = False
OLD_CLASS_SAMPLE_RATIO = 1.0
RARE_CLASS_KEEP_RATIO = 1.0
RARE_OVERSAMPLE = {
    "trash can": 2,
    "fence": 2,
    "pole_light": 2,
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


PRESET_CHOICES = (
    "base",
    "lite_fullset_polish",
    "warm3_old_anchor_lite_v2",
    "split_bridge_stage1",
    "split_bridge_stage2_fullset",
    "text_realign_1ep",
    "unfreeze_recalibrate",
    "stage1_rare_rescue",
    "rare_repair_lite",
    "partial_unfreeze_lastnorm",
    "partial_unfreeze_last1",
)

RARE_REPAIR_LITE_OVERSAMPLE = {
    "trash can": 3,
    "window": 2,
    "door": 2,
    "fence": 3,
    "pole_light": 3,
    "motorcycle": 2,
}

STAGE1_RARE_RESCUE_OVERSAMPLE = {
    "trash can": 2,
    "window": 2,
    "door": 2,
    "fence": 2,
    "pole_light": 2,
    "motorcycle": 2,
}

STAGE1_RARE_RESCUE_CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 1.0,
    "building": 1.0,
    "tree": 1.0,
    "animal": 1.0,
    "trash can": 1.7,
    "window": 1.55,
    "door": 1.45,
    "fence": 1.65,
    "pole_light": 1.75,
    "motorcycle": 1.6,
}

PRESET_CONFIGS = {
    "base": {},
    "lite_fullset_polish": {
        "run_name": "ESAM-CCLIP-11-lite-fullset-polish",
        "no_val": True,
        "no_train_split_filter": True,
        "epochs": 3,
        "decoder_lr": 5e-5,
        "warmup_epochs": 1,
        "negative_sample_ratio": 0.01,
        "negative_sample_weight": 0.10,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_lite_polish.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
    },
    "warm3_old_anchor_lite_v2": {
        # 从 3ep fullset warmup 底座开出的 old-anchor lite 分叉，不是继续纯热启动。
        # 目标：old5 稳住，rare 不被压死，避免继续强 fullset 热启动贴伪标签主分布。
        "run_name": "ESAM-CCLIP-11-warm3-old-anchor-lite-v2",
        "epochs": 1,
        "train_decoder_only": True,
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "unfreeze_image_mode": "none",
        "decoder_lr": 5e-5,
        "image_lr": 0.0,
        "text_lr": 0.0,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "min_lr_ratio": 0.20,
        "negative_sample_ratio": 0.005,
        "negative_sample_weight": 0.05,
        "old_class_sample_ratio": 0.85,
        "rare_class_keep_ratio": 1.0,
        "rare_balance_enabled": False,
        "augment_prompt": False,
        "prompt_alias_train": False,
        "prompt_alias_prob": 0.0,
        "val_augment_prompt": False,
        "use_prompt_prototype": True,
        "text_cache_path": CACHE_ROOT / "text_emb_11_warm3_old_anchor_lite_v2.pt",
        "class_weights": {
            "person": 1.03,
            "car": 1.05,
            "building": 1.05,
            "tree": 1.03,
            "animal": 1.00,
            "trash can": 1.10,
            "window": 1.05,
            "door": 1.00,
            "fence": 1.10,
            "pole_light": 1.10,
            "motorcycle": 1.05,
        },
        "rare_oversample": {
            "trash can": 2,
            "window": 1,
            "door": 1,
            "fence": 2,
            "pole_light": 2,
            "motorcycle": 1,
        },
    },
    "split_bridge_stage1": {
        "run_name": "ESAM-CCLIP-11-split-bridge-stage1",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 3,
        "decoder_lr": 5e-5,
        "warmup_epochs": 1,
        "negative_sample_ratio": 0.02,
        "negative_sample_weight": 0.15,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_split_bridge.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
    },
    "split_bridge_stage2_fullset": {
        "run_name": "ESAM-CCLIP-11-split-bridge-fullset",
        "train_json": ALL_JSON,
        "no_val": True,
        "no_train_split_filter": True,
        "epochs": 3,
        "decoder_lr": 3e-5,
        "warmup_epochs": 1,
        "negative_sample_ratio": 0.01,
        "negative_sample_weight": 0.10,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_split_bridge.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
    },
    "text_realign_1ep": {
        "run_name": "ESAM-CCLIP-11-text-realign-1ep",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 1,
        "decoder_lr": 5e-6,
        "image_lr": 0.0,
        "text_lr": 0.0,
        "warmup_epochs": 0,
        "min_lr_ratio": 0.5,
        "negative_sample_ratio": 0.005,
        "negative_sample_weight": 0.05,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "augment_prompt": True,
        "prompt_alias_train": True,
        "prompt_alias_prob": 1.0,
        "val_augment_prompt": False,
        "text_cache_path": CACHE_ROOT / "text_emb_11_text_realign_alias.pt",
        "prompt_prototype_cfg": dict(TEXT_REALIGN_PROMPT_PROTOTYPES),
        "rare_oversample": dict(RARE_OVERSAMPLE),
        "unfreeze_image_mode": "none",
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "train_decoder_only": True,
    },
    "unfreeze_recalibrate": {
        "run_name": "ESAM-CCLIP-11-unfreeze-recalibrate",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 1,
        "decoder_lr": 1e-5,
        "image_lr": 0.0,
        "text_lr": 0.0,
        "warmup_epochs": 0,
        "min_lr_ratio": 0.5,
        "negative_sample_ratio": 0.005,
        "negative_sample_weight": 0.05,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_unfreeze_recalibrate.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
        "unfreeze_image_mode": "none",
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "train_decoder_only": True,
    },
    "stage1_rare_rescue": {
        "run_name": "ESAM-CCLIP-11-stage1-rare-rescue",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 2,
        "decoder_lr": 2e-5,
        "image_lr": 0.0,
        "text_lr": 0.0,
        "unfreeze_image_mode": "none",
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "train_decoder_only": True,
        "warmup_epochs": 1,
        "min_lr_ratio": 0.3,
        "negative_sample_ratio": 0.02,
        "negative_sample_weight": 0.15,
        "old_class_sample_ratio": 0.85,
        "rare_class_keep_ratio": 1.0,
        "rare_balance_enabled": True,
        "use_prompt_prototype": True,
        "text_cache_path": CACHE_ROOT / "text_emb_11_stage1_rare_rescue.pt",
        "rare_oversample": dict(STAGE1_RARE_RESCUE_OVERSAMPLE),
        "class_weights": dict(STAGE1_RARE_RESCUE_CLASS_WEIGHTS),
    },
    "rare_repair_lite": {
        "run_name": "ESAM-CCLIP-11-rare-repair-lite",
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 3,
        "decoder_lr": 5e-5,
        "warmup_epochs": 1,
        "negative_sample_ratio": 0.01,
        "negative_sample_weight": 0.10,
        "rare_balance_enabled": True,
        "old_class_sample_ratio": 0.80,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_rare_repair_lite.pt",
        "rare_oversample": dict(RARE_REPAIR_LITE_OVERSAMPLE),
    },
    "partial_unfreeze_lastnorm": {
        "run_name": "ESAM-CCLIP-11-partial-unfreeze-lastnorm",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 2,
        "decoder_lr": 2e-5,
        "image_lr": 5e-7,
        "min_lr_ratio": 0.3,
        "negative_sample_ratio": 0.02,
        "negative_sample_weight": 0.15,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_split_bridge.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
        "unfreeze_image_mode": "last_norm",
    },
    "partial_unfreeze_last1": {
        "run_name": "ESAM-CCLIP-11-partial-unfreeze-last1",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "train_list": TRAIN_LIST,
        "val_list": VAL_LIST,
        "no_val": False,
        "no_train_split_filter": False,
        "epochs": 1,
        "decoder_lr": 1e-5,
        "image_lr": 5e-7,
        "min_lr_ratio": 0.5,
        "negative_sample_ratio": 0.02,
        "negative_sample_weight": 0.15,
        "rare_balance_enabled": False,
        "old_class_sample_ratio": 1.0,
        "rare_class_keep_ratio": 1.0,
        "text_cache_path": CACHE_ROOT / "text_emb_11_split_bridge.pt",
        "rare_oversample": dict(RARE_OVERSAMPLE),
        "unfreeze_image_mode": "last1",
    },
}


def _clone_preset_value(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def apply_preset(args, explicit_dests=None):
    explicit_dests = set(explicit_dests or [])
    preset_name = getattr(args, "preset", "base")
    preset_cfg = PRESET_CONFIGS.get(preset_name, {})
    for key, value in preset_cfg.items():
        if key not in explicit_dests:
            setattr(args, key, _clone_preset_value(value))

    if not hasattr(args, "rare_balance_enabled"):
        args.rare_balance_enabled = False
    if not hasattr(args, "rare_oversample") or args.rare_oversample is None:
        args.rare_oversample = dict(RARE_OVERSAMPLE)
    else:
        args.rare_oversample = dict(args.rare_oversample)

    preset_class_weights = getattr(args, "class_weights", None)
    if preset_class_weights is None:
        args.class_weights = dict(CLASS_WEIGHTS)
    else:
        args.class_weights = dict(preset_class_weights)
    args.old_classes = list(RARE_BALANCED_OLD_CLASSES)
    args.rare_classes = list(RARE_BALANCED_RARE_CLASSES)
    return args
