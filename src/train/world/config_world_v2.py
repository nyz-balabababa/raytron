"""
YOLO-World + FastSAM v2 配置
相比 v1 的改动：
  - computer 阈值 0.55→0.4, 过采样 2×→5× （释放更多伪标签，长尾覆盖）
  - cls loss 0.5→1.0, box loss 7.5→10.0 （弥补无法用 Focal Loss 的限制）
  - label_smoothing=0.05（稀有类需要更确定的分类边界）
  - 多尺度推理：person/tree 小目标类用 [0.8, 1.0, 1.25, 1.5]× 融合
  - animal 同义词测试扩充
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
# v2: computer 0.55→0.4，释放低置信度样本增加稀有类覆盖（score 0.4-0.55 区间）
CONF_FILTER = {
    "person":   0.70,
    "car":      0.70,
    "building": 0.70,
    "tree":     0.60,
    "animal":   0.60,
    "computer": 0.40,   # v2: 0.55→0.4
}

# 稀有类过采样倍数（仅训练集，文件级复制）
# v2: animal 单独 3×；computer 通过 COMPUTER_EXTRA 额外 +4 达到 5×
# 两个阶段解耦，避免 max() 导致 animal 也被意外多复制
RARE_OVERSAMPLE = {"animal": 3}
COMPUTER_EXTRA = 4     # computer 总倍数 = 1(base) + 4(extra) = 5×

# ══════════════════════════════════════════════════════════════════════
# 阶段一：YOLO-World v2 检测训练
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL = str(ROOT / "model" / "world" / "yolov8s-worldv2.pt")
YOLO_TASK = "detect"
YOLO_DIR = ROOT / "test" / "yolo_world_dataset_v2"     # v2: 独立数据集目录，与原版隔离
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

# ── v2: loss 权重调整（替代 Focal Loss，Focal Loss 在 YOLO API 中不可配置）──
# cls↑: 加大分类 loss → 稀有类误分类惩罚更重，间接抑制"什么都预测为 car"
# box↑: 加大框 loss → 小目标框更精准 → FastSAM 分割 ROI 质量更高
CLS_LOSS_GAIN = 1.0       # v2: 默认 0.5→1.0
BOX_LOSS_GAIN = 10.0      # v2: 默认 7.5→10.0
LABEL_SMOOTHING = 0.05    # v2: 新增，让稀有类有更确定的分类边界

# ══════════════════════════════════════════════════════════════════════
# 阶段二：FastSAM 零样本分割
# ══════════════════════════════════════════════════════════════════════

Fastsam_MODEL = str(ROOT / "model" / "world" / "FastSAM-x.pt")
Fastsam_IMG_SIZE = 1024
Fastsam_CONF = 0.25
Fastsam_IOU = 0.7

# ══════════════════════════════════════════════════════════════════════
# v2: 多尺度推理（仅针对小目标高频类别 person / tree）
# ══════════════════════════════════════════════════════════════════════

# 需要多尺度推理的类别
MULTI_SCALE_CLASSES = ["person", "tree", "animal"]

# 多尺度因子（相对于 IMG_SIZE），>1.0 放大细节，<1.0 扩大感受野
MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]

# 跨尺度 NMS IoU 阈值。
# 不同尺度下同一目标的框有位置漂移，较低的阈值（0.35）可以容忍这种偏移，
# 避免同一目标在不同尺度下被当作两个目标（漏合并）。
MS_NMS_IOU = 0.35

# 多尺度后方检测置信度阈值（比单尺度稍低，因为小尺度可能降置信度）
MS_CONF = 0.2

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "yoloworld_fastsam_v2"
EXIST_OK = True

# ══════════════════════════════════════════════════════════════════════
# 推理时 prompt → 类别映射（泛化测试）
# v2: animal 扩充大量具体动物名 + 不同粒度描述
# ══════════════════════════════════════════════════════════════════════

PROMPT_MAP = {
    # ── person ──
    "human": "person", "people": "person", "行人": "person", "pedestrian": "person",
    "man": "person", "woman": "person",

    # ── car ──
    "vehicle": "car", "vehicles": "car", "automobile": "car", "truck": "car",
    "cars": "car", "汽车": "car", "轿车": "car",

    # ── building ──
    "house": "building", "houses": "building", "structure": "building",
    "建筑": "building", "房屋": "building",

    # ── tree ──
    "plant": "tree", "plants": "tree", "vegetation": "tree", "bush": "tree",
    "bushes": "tree", "树木": "tree", "植物": "tree",

    # ── animal (v2: 大量扩充) ──
    "wildlife": "animal", "animals": "animal", "creature": "animal",
    "动物": "animal",
    # 具体动物名（SAM3 data6 实验中激活率较高的词）
    "deer": "animal", "fox": "animal", "bird": "animal",
    "wolf": "animal", "lion": "animal", "duck": "animal",
    "monkey": "animal", "swan": "animal", "zebra": "animal",
    "wild boar": "animal", "bear": "animal",
    # 不同粒度的描述
    "a wild animal": "animal", "animals in the forest": "animal",
    "wildlife creature": "animal", "living creature": "animal",

    # ── computer ──
    "pc": "computer", "laptop": "computer", "monitor": "computer",
    "电脑": "computer", "计算机": "computer", "screen": "computer",
    "display": "computer",
}
