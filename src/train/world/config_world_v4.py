"""
YOLO-World + FastSAM v4 配置
=============================

目标:
  - 基于 data5&7 的稀有设备实验
  - 当前实验只保留 2 类 head: computer / trash can
  - 训练/验证分别使用现成的 train_d5&7 与 val_d5&7 伪标签

说明:
  - fire hydrant 不并入本次 head，后续单独做 binary few-shot
  - circuit board 已从 pred_train_d5&7_tasks.json 删除
  - 训练集: 全保留正样本图 + 按比例采空标签负样本图
  - 验证集: 保留 val_list_d5&7.txt 中的全部图
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

CLASSES = ["computer", "trash can"]
PROMPT_OUTPUT_ROOT = ROOT / "test" / "prompt_test_output"
PRED_JSON = PROMPT_OUTPUT_ROOT / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"  #训练集伪标签
VAL_PRED_JSON = PROMPT_OUTPUT_ROOT / "val_d5&7_tasks" / "pred_val_d5&7_tasks.json"  #验证集伪标签
TRAIN_LIST = ROOT / "test" / "train_list_d5&7.txt"  #训练集图像列表
VAL_LIST = ROOT / "test" / "val_list_d5&7.txt"  #验证集图像列表
IMAGE_ROOT = ROOT

# 伪标签分级过滤阈值
CONF_FILTER = {
    "computer": 0.60,
    "trash can": 0.65,
}

# 图像级过采样倍数（按图中出现的稀有类取最大值）
RARE_OVERSAMPLE = {
    "computer": 2,
    "trash can": 3,
}

# 训练集负样本采样
NEGATIVE_TO_POSITIVE_RATIO = 1.0   # 负样本图数量 = 正样本图数量 × 比例
NEGATIVE_SAMPLE_SEED = 42

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World 检测训练
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset_v4_2cls"
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

# 分类分支：Focal Loss + 手动类别权重
# 目的：降低 easy negative 对分类梯度的主导，让长尾类 computer / trash can 获得更高关注
FOCAL_LOSS_GAMMA = 1.5
FOCAL_LOSS_ALPHA = 0.25
CLASS_LOSS_WEIGHTS = {
    "computer": 2.0,
    "trash can": 3.0,
}

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

# 对更小、更稀有的目标保留多尺度
MULTI_SCALE_CLASSES = ["computer"]
MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]
MS_NMS_IOU = 0.35
MS_CONF = 0.2

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "v4_d5d7_2cls_focal"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 推理时 prompt → 类别映射
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    # computer
    "pc": "computer",
    "monitor": "computer",
    "screen": "computer",
    "display": "computer",
    "laptop": "computer",

    # trash can
    "bin": "trash can",
    "garbage can": "trash can",
    "dustbin": "trash can",
    "rubbish bin": "trash can",

}
