"""
CLIPSeg-11 Opt Decoder-Only + Infer-Opt 配置

说明：
1. 本文件完全自包含，不依赖 clipseg-v4 或其他训练线配置。
2. 只保留 CLIPSeg 主结构的 decoder-only 优化路线。
3. 可直接复制到其他电脑使用，只需同步项目目录结构与模型权重目录。
"""
from __future__ import annotations

from pathlib import Path


# =========================
# 基础路径
# =========================
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs"
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT
INFER_OUTPUT_ROOT = OUTPUT_ROOT / "clipseg_11_infer_opt"


def _sanitize_mapping(classes, candidates, fallback=None, default_value=None):
    output = {}
    fallback = fallback or {}
    for class_name in classes:
        if class_name in candidates:
            output[class_name] = candidates[class_name]
        elif class_name in fallback:
            output[class_name] = fallback[class_name]
        elif default_value is not None:
            output[class_name] = default_value(class_name) if callable(default_value) else default_value
    return output


def _sanitize_prompt_pool(classes, candidates):
    pool = {}
    for class_name in classes:
        aliases = candidates.get(class_name, [class_name])
        normalized = []
        seen = set()
        for alias in [class_name, *aliases]:
            alias = str(alias).strip()
            if not alias or alias in seen:
                continue
            seen.add(alias)
            normalized.append(alias)
        pool[class_name] = normalized or [class_name]
    return pool


# =========================
# 类别与数据
# =========================
CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "computer",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
]
NUM_CLASSES = len(CLASSES)

PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "val-nocomputer.json"
TRAIN_LIST = ROOT / "test" / "trainval_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT

USE_MANIFEST = False
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"

PROMPT_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "computer": 0.60,
    "trash can": 0.55,
    "window": 0.50,
    "door": 0.50,
    "fence": 0.45,
    "pole_light": 0.55,
}
APPLY_SCORE_FILTER = False
CONF_FILTER = PROMPT_THRESHOLDS


# =========================
# 输出与运行名
# =========================
RUN_NAME = "clipseg_11_opt_decoder_only"
INFER_RUN_NAME = "clipseg_11_infer_opt"
SWEEP_DIRNAME = "threshold_sweep"
EXIST_OK = True


# =========================
# 模型与冻结策略
# =========================
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"
DEFAULT_RESUME = ROOT / "test" / "train_output" / "clipseg_xiyou_12cls_trainval_v1" / "best.pt"

FREEZE_TEXT_ENCODER = True
FREEZE_IMAGE_ENCODER = True
TRAIN_DECODER_ONLY = True


# =========================
# 训练
# =========================
IMG_SIZE = 768
BATCH_SIZE = 4
EPOCHS = 8
PATIENCE = 5
WORKERS = 8
SEED = 42
DEVICE = 0
VAL_SAMPLE_IMAGE_LIMIT = 2000   # 0 表示验证全量图片
VAL_SAMPLE_SEED = 42

DECODER_LR = 1e-4
WEIGHT_DECAY = 0.03
LRF = 0.01
WARMUP_EPOCHS = 2
WARMUP_START_FACTOR = 0.01
LOSS_WEIGHT_FLOOR = 0.7


# =========================
# Loss
# =========================
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.2
FOCAL_WEIGHT = 0.05
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

NEGATIVE_SAMPLE_RATIO = 0.04
NEGATIVE_SAMPLE_WEIGHT = 0.05


# =========================
# Prompt Pool
# =========================
PROMPT_POOL = _sanitize_prompt_pool(
    CLASSES,
    {
        "person": ["human", "pedestrian"],
        "car": ["vehicle", "automobile"],
        "building": ["house", "architecture"],
        "tree": ["plant", "vegetation"],
        "animal": ["dog", "cat"],
        "computer": ["computer", "laptop", "monitor"],
        "trash can": ["garbage bin", "waste bin", "bin"],
        "window": ["glass window"],
        "door": ["gate"],
        "fence": ["railing", "barrier"],
        "pole_light": ["street light", "lamp post"],
    },
)
PROMPT_POOL_ENABLED = any(len(v) > 1 for v in PROMPT_POOL.values())
PROMPT_SAMPLE_PROB = 1.0


# =========================
# Rare Oversample
# =========================
RARE_OVERSAMPLE = _sanitize_mapping(
    CLASSES,
    {
        "animal": 2,
        "computer": 2,
        "trash can": 2,
        "window": 2,
        "door": 2,
        "fence": 2,
        "pole_light": 3,
    },
    default_value=1,
)
RARE_OVERSAMPLE = {k: int(v) for k, v in RARE_OVERSAMPLE.items() if int(v) > 1}
RARE_OVERSAMPLE_ENABLED = True


# =========================
# 阈值 / 推理增强
# =========================
THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
DEFAULT_THRESHOLDS = _sanitize_mapping(
    CLASSES,
    {
        "person": 0.55,
        "car": 0.55,
        "building": 0.60,
        "tree": 0.50,
        "animal": 0.45,
        "computer": 0.45,
        "trash can": 0.40,
        "window": 0.40,
        "door": 0.40,
        "fence": 0.40,
        "pole_light": 0.35,
    },
    fallback=PROMPT_THRESHOLDS,
    default_value=0.50,
)
VAL_THRESHOLDS = DEFAULT_THRESHOLDS
UNKNOWN_PROMPT_THRESHOLD = 0.45

PROMPT_FUSION = "weighted_mean"
PROMPT_MAIN_WEIGHT = 0.7
PROMPT_ALIAS_WEIGHT = 0.3

VAL_PROMPT_FUSION = "none"
INFER_PROMPT_FUSION = "weighted_mean"
SWEEP_PROMPT_FUSION = "weighted_mean"

VAL_TTA = "none"
INFER_TTA = "hflip"
SWEEP_TTA = "hflip"

ENABLE_POSTPROCESS = True
VAL_ENABLE_POSTPROCESS = False
INFER_ENABLE_POSTPROCESS = True
SWEEP_ENABLE_POSTPROCESS = True
ENABLE_CONFLICT_SUPPRESSION = False


# =========================
# 后处理
# =========================
MIN_AREA_BY_CLASS = _sanitize_mapping(
    CLASSES,
    {
        "person": 16,
        "car": 32,
        "building": 64,
        "tree": 32,
        "animal": 8,
        "computer": 8,
        "trash can": 8,
        "window": 8,
        "door": 8,
        "fence": 8,
        "pole_light": 4,
    },
    default_value=8,
)


# =========================
# 边界弱监督
# =========================
ENABLE_BOUNDARY_WEAK_SUPERVISION = False
BOUNDARY_IGNORE_WIDTH = 1
BOUNDARY_IGNORE_MIN_AREA = 64
TINY_PROTECT_CLASSES = {
    class_name
    for class_name in CLASSES
    if class_name in {"person", "car", "animal", "computer", "trash can", "window", "door", "fence", "pole_light"}
}
TINY_AREA_THRESHOLDS = _sanitize_mapping(
    CLASSES,
    {
        "person": 64,
        "car": 64,
        "building": 80,
        "tree": 64,
        "animal": 48,
        "computer": 32,
        "trash can": 32,
        "window": 32,
        "door": 48,
        "fence": 64,
        "pole_light": 24,
    },
    default_value=64,
)


# =========================
# 图像预处理
# =========================
HFLIP_PROB = 0.5
GRAY2RGB = True
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200
PSEUDO_COLOR_SAT_THRESH = 60.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# =========================
# 开关
# =========================
INCLUDE_NEGATIVE_SAMPLES = True
VAL_INCLUDE_NEGATIVE_SAMPLES = False
