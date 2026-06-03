"""
YOLO-World + FastSAM 两阶段端到端配置 —— 自包含。
阶段一：YOLO-World v2 检测（bbox）—— 需训练
阶段二：FastSAM 零样本分割 —— 不需训练
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

CLASSES = ["person", "car", "building", "tree", "animal", "computer"]
PRED_JSON = ROOT / "test" / "prompt_test_output" / "train_tasks" / "pred_train_tasks.json"
VAL_PRED_JSON = ROOT / "test" / "prompt_test_output" / "val_tasks1" / "pred_val_tasks1.json"
TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT

CONF_FILTER = {
    "person":   0.70,
    "car":      0.70,
    "building": 0.70,
    "tree":     0.60,
    "animal":   0.60,
    "computer": 0.55,
}
RARE_OVERSAMPLE = {"animal": 3, "computer": 2}

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World v2 检测
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset"
TRAIN_IMAGES_DIR = YOLO_DIR / "images" / "train"
TRAIN_LABELS_DIR = YOLO_DIR / "labels" / "train"
VAL_IMAGES_DIR = YOLO_DIR / "images" / "val"
VAL_LABELS_DIR = YOLO_DIR / "labels" / "val"

IMG_SIZE = 800
BATCH = 16
EPOCHS = 100
DEVICE = 0
WORKERS = 8
SEED = 42

OPTIMIZER = "AdamW"
LR0 = 2e-4
MOMENTUM = 0.937
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
COS_LR = True
PATIENCE = 10

MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0
SCALE = 0.3
DEGREES = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.15
CLOSE_MOSAIC = 0

# ══════════════════════════════════════════════════════════════════════
# 阶段二：FastSAM 零样本分割
# ══════════════════════════════════════════════════════════════════════

Fastsam_MODEL = str(ROOT / "model" / "world" / "FastSAM-x.pt")
Fastsam_IMG_SIZE = 1024
Fastsam_CONF = 0.25
Fastsam_IOU = 0.7

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yoloworld_fastsam_v1"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 推理时 prompt → 类别映射（泛化测试）
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    # person
    "human": "person", "people": "person", "行人": "person", "pedestrian": "person",
    "man": "person", "woman": "person",
    # car
    "vehicle": "car", "vehicles": "car", "automobile": "car", "truck": "car",
    "cars": "car", "汽车": "car", "轿车": "car",
    # building
    "house": "building", "houses": "building", "structure": "building",
    "建筑": "building", "房屋": "building",
    # tree
    "plant": "tree", "plants": "tree", "vegetation": "tree", "bush": "tree",
    "bushes": "tree", "树木": "tree", "植物": "tree",
    # animal
    "wildlife": "animal", "animals": "animal", "creature": "animal",
    "动物": "animal",
    # computer
    "pc": "computer", "laptop": "computer", "monitor": "computer",
    "电脑": "computer", "计算机": "computer",
}
