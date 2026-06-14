"""
SegFormer-B0 学生模型训练配置 —— 多标签11类语义分割（P0修复版）
- 修复：补上window/pole_light过采样
- 修复：补上验证集正样本标记
- 修复：Focal alpha正确配置
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

# 11类：删除computer，保留其他11类
CLASSES = ["person", "car", "building", "tree", "animal",
           "trash can", "window", "door", "fence", "pole_light", "motorcycle"]
NUM_CLASSES = 11

# 使用v3的clean_rare数据源（12类标签文件），Dataset加载时过滤computer
PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST = ROOT / "test" / "trainval_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT
USE_MANIFEST = False
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"

# teacher 伪标签输出阈值
PROMPT_THRESHOLDS = {
    "person":     0.70,
    "car":        0.70,
    "building":   0.70,
    "tree":       0.60,
    "animal":     0.60,
    "trash can":  0.55,
    "window":     0.50,
    "door":       0.50,
    "fence":      0.45,
    "pole_light": 0.55,
    "motorcycle": 0.55,
}

APPLY_SCORE_FILTER = False
CONF_FILTER = PROMPT_THRESHOLDS
VAL_THRESHOLDS = PROMPT_THRESHOLDS

# P0修复：补上window和pole_light过采样
RARE_OVERSAMPLE = {
    "animal": 2,
    "trash can": 2,
    "window": 2,        # 新增
    "pole_light": 3,    # 新增，权重更高
    "door": 2,
    "motorcycle": 2,
}

INCLUDE_NEGATIVE_SAMPLES = True
NEGATIVE_SAMPLE_RATIO = 0.15
NEGATIVE_SAMPLE_WEIGHT = 0.30    # P0修复：验证集不包含负样本（见下文）

# ══════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════

# SegFormer-B0：轻量级语义分割基线
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_DIR = ROOT / "model" / "segformer-b0"
MODEL_TYPE = "segformer_b0_multilabel_11"

# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

IMG_SIZE = 768                # 等比例缩放 + pad 正方形
BATCH = 4
EPOCHS = 25                   # P0修复：可通过命令行--epochs覆盖
DEVICE = 0
WORKERS = 8
SEED = 42

# ══════════════════════════════════════════════════════════════════════
# 优化器
# ══════════════════════════════════════════════════════════════════════

BACKBONE_LR = 1e-4             # SegFormer backbone
DECODER_HEAD_LR = 1e-3         # decoder head（收敛更快）
WEIGHT_DECAY = 0.03
LRF = 0.01
WARMUP_EPOCHS = 2
WARMUP_START_FACTOR = 0.01

# ══════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════

BCE_WEIGHT = 1.0
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.5

# 稀有类焦点损失权重 —— 按类别加权（从clip优化.md）
CLASS_WEIGHTS = {
    "person":     1.0,
    "car":        1.0,
    "building":   1.0,
    "tree":       1.2,
    "animal":     2.0,
    "trash can":  2.0,
    "window":     1.8,
    "door":       1.5,
    "fence":      1.5,
    "pole_light": 2.5,
    "motorcycle": 2.0,
}
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25             # P0修复：现在真正在focal loss中使用

# 边界弱监督
ENABLE_BOUNDARY_WEAK_SUPERVISION = True
BOUNDARY_IGNORE_WIDTH = 1
BOUNDARY_IGNORE_MIN_AREA = 64
LOSS_WEIGHT_FLOOR = 0.7        # P0修复：只对正样本使用，负样本单独处理

# ══════════════════════════════════════════════════════════════════════
# 数据增强
# ══════════════════════════════════════════════════════════════════════

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

# 图像预处理归一化 —— 与SegFormer预训练保持一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# CLIP/CLIPSeg 图像归一化
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ══════════════════════════════════════════════════════════════════════
# 蒸馏（可选）
# ══════════════════════════════════════════════════════════════════════

USE_DISTILL = False
TEACHER_SOFT_CACHE = ROOT / "test" / "train_output" / "clipseg_soft_cache_11"
DISTILL_LAMBDA = 0.15
DISTILL_LAMBDA_RARE = 0.25
DISTILL_START_EPOCH = 0
SKIP_MISSING_TEACHER = True
RESUME_WEIGHTS_ONLY = False
CLIPSEG_TEACHER_BASE_MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"
CLIPSEG_TEACHER_CHECKPOINT = ROOT / "test" / "train_output" / "clipseg_xiyou_12cls_trainval_v1" / "best.pt"

# ══════════════════════════════════════════════════════════════════════
# 验证集配置（P0修复）
# ══════════════════════════════════════════════════════════════════════

# P0修复：验证集不包含负样本，只评估正样本
VAL_INCLUDE_NEGATIVE_SAMPLES = False

# 分组统计
OLD5_CLASSES = ["person", "car", "building", "tree", "animal"]
RARE_CLASSES = ["animal", "trash can", "window", "pole_light", "motorcycle"]

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "segformer_b0_11cls_v1"  # P0修复：可通过--run_name覆盖
EXIST_OK = True
