"""
CLIPSeg 训练配置 —— 自包含，不再继承 config_common。
后续蒸馏实验（Grounding DINO 等）以此文件为模板，保持参数对齐。
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

# 伪标签分级过滤阈值
CONF_FILTER = {
    "person":   0.70,
    "car":      0.70,
    "building": 0.70,
    "tree":     0.60,
    "animal":   0.60,
    "computer": 0.55,
}

# 稀有类过采样倍数（仅训练集）
RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 2,
}

# ══════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════

MODEL_NAME = "CIDAS/clipseg-rd64-refined"
MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"

# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

IMG_SIZE = 1024                 # 等比例缩放 + pad 正方形
BATCH = 4
EPOCHS = 25                    # 单阶段 + 余弦退火
DEVICE = 0
WORKERS = 8
SEED = 42

# ══════════════════════════════════════════════════════════════════════
# 优化器
# ══════════════════════════════════════════════════════════════════════

DECODER_LR = 1e-4               # FiLM 解码器
BACKBONE_LR = 1e-5              # CLIP 视觉编码器（低 10×）
WEIGHT_DECAY = 0.03
LRF = 0.01                      # 最终 lr = lr0 * LRF
WARMUP_EPOCHS = 2               # 前 2 epoch 线性 warmup（总 200 的 1%）
WARMUP_START_FACTOR = 0.01      # warmup 起点 lr = lr0 × 0.01

# ══════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
LOSS_WEIGHT_FLOOR = 0.7          # 逐样本置信度加权下限（保护 computer/animal）

# ══════════════════════════════════════════════════════════════════════
# 数据增强
# ══════════════════════════════════════════════════════════════════════

HFLIP_PROB = 0.5                # 红外水平翻转
GRAY2RGB = True                 # 灰度复制为 3 通道
# 反色统一：黑热（热目标=暗）→ 白热（热目标=亮）
BLACKHOT_SKEW_THRESH = -0.3     # 直方图偏度 < 此值判定为黑热
BLACKHOT_MEAN_THRESH = 200      # 均值 > 此值且非 vis 图，兜底反色
# CLIP 预训练时的均值/标准差（RGB 三通道）
CLIP_MEAN = [0.48145466, 0.52048427, 0.45053169]
CLIP_STD  = [0.21028575, 0.23535925, 0.22184163]

# ══════════════════════════════════════════════════════════════════════
# Prompt 动态增强
# ══════════════════════════════════════════════════════════════════════

PROMPT_AUG_PROB = 0.5

PROMPT_AUGMENTATIONS = {
    "person": [
        "person, human, people",
        "a person walking or standing",
        "people in the scene",
    ],
    "car": [
        "cars, vehicles, trucks, or any automobiles",
        "a car or vehicle on the road",
        "vehicles including cars and trucks",
    ],
    "building": [
        "buildings, houses, or structures",
        "any building or architectural structure",
        "houses and buildings",
    ],
    "tree": [
        "trees, plants, bushes, or any vegetation",
        "trees and vegetation",
        "plants, bushes, and trees",
    ],
    "animal": [
        "animal, wildlife",
        "any animal or wildlife creature",
        "animals in the wild",
    ],
    "computer": [],
}

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "clipseg_v1"
EXIST_OK = True
