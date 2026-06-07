"""
YOLO-World + FastSAM manifest 训练配置
====================================

- 训练集可切换 `pred_json` 或 `manifest`
- manifest 模式面向当前五类:
  person / car / building / tree / animal
- 验证集先继续固定使用 `VAL_PRED_JSON`
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

CLASSES = ["person", "car", "building", "tree", "animal"]
PRED_JSON = ROOT / "test" / "prompt_test_output" / "train_tasks" / "pred_train_tasks.json"
VAL_PRED_JSON = ROOT / "test" / "prompt_test_output" / "val_tasks1" / "pred_val_tasks1.json"
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"
USE_MANIFEST = True

TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT

CONF_FILTER = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
}

# manifest 版仍保留 animal 过采样；computer 已关闭
RARE_OVERSAMPLE = {"animal": 3}

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World 检测训练
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset_manifest5_v1"
REBUILD_DATASET = True
TRAIN_IMAGES_DIR = YOLO_DIR / "images" / "train"
TRAIN_LABELS_DIR = YOLO_DIR / "labels" / "train"
VAL_IMAGES_DIR = YOLO_DIR / "images" / "val"
VAL_LABELS_DIR = YOLO_DIR / "labels" / "val"

IMG_SIZE = 800
BATCH = 16
EPOCHS = 20
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
# 多尺度推理
# ══════════════════════════════════════════════════════════════════════

MULTI_SCALE_CLASSES = ["person", "animal"]
MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]
MS_NMS_IOU = 0.35
MS_CONF = 0.2

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yoloworld_fastsam_v3"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 同义词映射
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "man": "person",
    "woman": "person",
    "person": "person",
    "a person": "person",
    "a human": "person",
    "human figure": "person",
    "人": "person",
    "行人": "person",
    "人员": "person",
    "一个人": "person",
    "vehicle": "car",
    "vehicles": "car",
    "automobile": "car",
    "truck": "car",
    "cars": "car",
    "car": "car",
    "a car": "car",
    "a vehicle": "car",
    "motor vehicle": "car",
    "车": "car",
    "汽车": "car",
    "车辆": "car",
    "一辆车": "car",
    "house": "building",
    "houses": "building",
    "structure": "building",
    "building": "building",
    "a building": "building",
    "a house": "building",
    "built structure": "building",
    "建筑": "building",
    "建筑物": "building",
    "房屋": "building",
    "房子": "building",
    "plant": "tree",
    "plants": "tree",
    "vegetation": "tree",
    "bush": "tree",
    "bushes": "tree",
    "tree": "tree",
    "a tree": "tree",
    "green plant": "tree",
    "tree canopy": "tree",
    "树": "tree",
    "树木": "tree",
    "植被": "tree",
    "灌木": "tree",
    "wildlife": "animal",
    "animals": "animal",
    "creature": "animal",
    "animal": "animal",
    "an animal": "animal",
    "wild animal": "animal",
    "living animal": "animal",
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
    "a wild animal": "animal",
    "animals in the forest": "animal",
    "wildlife creature": "animal",
    "living creature": "animal",
    "动物": "animal",
    "野生动物": "animal",
    "鸟": "animal",
    "狐狸": "animal",
    "鹿": "animal",
}
