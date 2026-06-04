"""
YOLO-World + FastSAM v5 配置
=============================

定位:
  - 7 类全训练集“数据处理验证版”
  - 基于 v3 主线，先只验证新增 trash can 并回主线后的效果

类别组成:
  - 原 v3 六类: person / car / building / tree / animal / computer
  - 新增一类: trash can

伪标签来源:
  - 使用已合并好的 7 类伪标签文件
  - 训练: train_tasks/new_train.json
  - 验证: val_tasks1/new_val.json
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

BASE_CLASSES = ["person", "car", "building", "tree", "animal", "computer"]
EXTRA_CLASSES = ["trash can"]
CLASSES = BASE_CLASSES + EXTRA_CLASSES
WATCH_CLASSES = ["computer", "trash can"]

TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT

TRAIN_PRED_SOURCES = [
    {
        "name": "merged_7cls_train",
        "path": ROOT / "test" / "prompt_test_output" / "train_tasks" / "new_train.json",
        "classes": CLASSES,
    },
]

VAL_PRED_SOURCES = [
    {
        "name": "merged_7cls_val",
        "path": ROOT / "test" / "prompt_test_output" / "val_tasks1" / "new_val.json",
        "classes": CLASSES,
    },
]

# 伪标签分级过滤阈值
CONF_FILTER = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "computer": 0.50,
    "trash can": 0.55,
}

# 图像级过采样倍数（按图中出现的类取最大值）
RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 2,
    "trash can": 3,
}

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World 检测训练
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset_v5_7cls_trashcan"
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

# 保持 v3 的稳定设置，不把新增尾类也塞进多尺度
MULTI_SCALE_CLASSES = ["person", "animal"]
MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]
MS_NMS_IOU = 0.35
MS_CONF = 0.2

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yoloworld_fastsam_v5_7cls_trashcan_verify"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 推理时 prompt → 类别映射
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    # person
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "man": "person",
    "woman": "person",

    # car
    "vehicle": "car",
    "vehicles": "car",
    "automobile": "car",
    "truck": "car",
    "cars": "car",

    # building
    "house": "building",
    "houses": "building",
    "structure": "building",

    # tree
    "plant": "tree",
    "plants": "tree",
    "vegetation": "tree",
    "bush": "tree",
    "bushes": "tree",

    # animal
    "wildlife": "animal",
    "animals": "animal",
    "creature": "animal",
    "deer": "animal",
    "fox": "animal",
    "bird": "animal",
    "wolf": "animal",
    "lion": "animal",
    "duck": "animal",
    "monkey": "animal",
    "swan": "animal",
    "zebra": "animal",
    "wild boar": "animal",
    "bear": "animal",

    # computer
    "pc": "computer",
    "laptop": "computer",
    "monitor": "computer",
    "screen": "computer",
    "display": "computer",

    # trash can
    "bin": "trash can",
    "garbage can": "trash can",
    "dustbin": "trash can",
    "rubbish bin": "trash can",
}
