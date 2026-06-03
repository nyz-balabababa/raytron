"""
YOLO-World + FastSAM v3 配置
=============================
v2→v3 改动（基于 v1/v2 对比实验结论）:

  回退:
    - cls/bow/label_smoothing → 回退到 Ultralytics 默认值
      v2 实验证明 cls=1.0+label_smoothing=0.05 摧毁了 CLIP 同义词泛化
    - computer 阈值 0.4→0.5, 过采样 5×→2×
      5× 过采样未带来统计显著的提升(n=23), 回退到与 v1 一致的配置
    - tree 从多尺度移除 → 单尺度
      v2 实验显示 tree MS 反而降了 5.7% mIoU, 伪标签质量差是根因

  保留(v2 验证有效):
    - 多尺度推理 person/animal: +19.5%/+26.3% mask mIoU
    - animal 3× 过采样(独立阶段, 与 computer 解耦)
    - FastSAM 动态 imgsz、CLIP 离线缓存、GPU 日志、训练曲线图
    - 跨尺度 NMS IoU=0.35、MS_CONF=0.2
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

# 伪标签分级过滤阈值
# v3: computer 0.4→0.5 (回退, 5× 过采样对 23 样本无统计意义)
CONF_FILTER = {
    "person":   0.70,
    "car":      0.70,
    "building": 0.70,
    "tree":     0.60,
    "animal":   0.60,
    "computer": 0.50,
}

# 稀有类过采样（animal 与 computer 解耦，独立阶段）
RARE_OVERSAMPLE = {"animal": 3}
COMPUTER_EXTRA = 1     # computer 总倍数 = 1(base) + 1(extra) = 2×

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World v2 检测训练
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset_v3"
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

# ── 增强 ──
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

# v3: cls/box/label_smoothing 已删除, 使用 Ultralytics 默认值
# v2 实验证明加大这些参数会摧毁 CLIP 同义词泛化能力

# ══════════════════════════════════════════════════════════════════════
# 阶段二：FastSAM 零样本分割
# ══════════════════════════════════════════════════════════════════════

Fastsam_MODEL = str(ROOT / "model" / "world" / "FastSAM-x.pt")
Fastsam_IMG_SIZE = 1024
Fastsam_CONF = 0.25
Fastsam_IOU = 0.7

# ══════════════════════════════════════════════════════════════════════
# v3: 多尺度推理 — 仅 person/animal (v2 验证有效, +19.5%/+26.3% mIoU)
#     tree 移除 (v2 多尺度反而降 5.7%)
# ══════════════════════════════════════════════════════════════════════

MULTI_SCALE_CLASSES = ["person", "animal"]

MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]

# 跨尺度 NMS：低阈值容忍不同尺度下同一目标的框位置漂移
MS_NMS_IOU = 0.35
MS_CONF = 0.2

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yoloworld_fastsam_v3"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 推理时 prompt → 类别映射（同义词泛化测试）
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    # ── person ──
    "human": "person", "people": "person", "pedestrian": "person",
    "man": "person", "woman": "person",

    # ── car ──
    "vehicle": "car", "vehicles": "car", "automobile": "car", "truck": "car",
    "cars": "car",

    # ── building ──
    "house": "building", "houses": "building", "structure": "building",

    # ── tree ──
    "plant": "tree", "plants": "tree", "vegetation": "tree", "bush": "tree",
    "bushes": "tree",

    # ── animal ──
    "wildlife": "animal", "animals": "animal", "creature": "animal",
    "deer": "animal", "fox": "animal", "bird": "animal",
    "wolf": "animal", "lion": "animal", "duck": "animal",
    "monkey": "animal", "swan": "animal", "zebra": "animal",
    "wild boar": "animal", "bear": "animal",
    "a wild animal": "animal", "animals in the forest": "animal",
    "wildlife creature": "animal", "living creature": "animal",

    # ── computer ──
    "pc": "computer", "laptop": "computer", "monitor": "computer",
    "screen": "computer", "display": "computer",
}
