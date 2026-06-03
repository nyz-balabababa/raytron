"""
YOLOv11-seg 训练配置 —— 自包含。
CLIPSeg 实验对标此文件的参数设计。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

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

RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 10,
}

# ══════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════

MODEL = "yolo11s-seg.pt"
TASK = "segment"
YOLO_DIR = ROOT / "test" / "yolo_dataset"
TRAIN_IMAGES_DIR = YOLO_DIR / "images" / "train"
TRAIN_LABELS_DIR = YOLO_DIR / "labels" / "train"
VAL_IMAGES_DIR = YOLO_DIR / "images" / "val"
VAL_LABELS_DIR = YOLO_DIR / "labels" / "val"

# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

DEVICE = 0
WORKERS = 8
SEED = 42

# 两阶段训练（阶段二关闭 copy_paste）
STAGE1_EPOCHS = 150
STAGE1_IMG_SIZE = 640
STAGE1_BATCH = 16
STAGE1_LR0 = 1e-3

STAGE2_EPOCHS = 50
STAGE2_IMG_SIZE = 640
STAGE2_BATCH = 16
STAGE2_LR0 = 5e-5

# ══════════════════════════════════════════════════════════════════════
# 优化器
# ══════════════════════════════════════════════════════════════════════

OPTIMIZER = "AdamW"
LRF = 0.01
WEIGHT_DECAY = 0.03
WARMUP_EPOCHS = 5
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1
COS_LR = True
PATIENCE = 30                   # 早停

# ══════════════════════════════════════════════════════════════════════
# 正则化
# ══════════════════════════════════════════════════════════════════════

LABEL_SMOOTHING = 0.1
DROPOUT = 0.1

# ══════════════════════════════════════════════════════════════════════
# YOLO 专用 loss
# ══════════════════════════════════════════════════════════════════════

CLS_LOSS_GAIN = 0.3
BOX_LOSS_GAIN = 7.5

# ══════════════════════════════════════════════════════════════════════
# YOLO 专用数据增强
# ══════════════════════════════════════════════════════════════════════

MOSAIC = 0.0
COPY_PASTE = 0.3
SCALE = 0.3
DEGREES = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.15
CLOSE_MOSAIC = 0

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yolo11s_seg_v3"
EXIST_OK = True
