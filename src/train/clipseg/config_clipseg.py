"""
CLIPSeg 训练配置 —— 自包含，不再继承 config_common。
后续蒸馏实验（Grounding DINO 等）以此文件为模板，保持参数对齐。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

CLASSES = ["person", "car", "building", "tree", "animal","computer"]
PRED_JSON = ROOT / "test" / "sam3_label_old" / "train_tasks" / "pred_train_tasks.json"
VAL_PRED_JSON = ROOT / "test" / "sam3_label_old" / "val_tasks1" / "pred_val_tasks1.json"
TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT
USE_MANIFEST = False
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"

# teacher 伪标签输出阈值；训练验证与其保持一致
PROMPT_THRESHOLDS = {
    "person":   0.70,
    "car":      0.70,
    "building": 0.70,
    "tree":     0.60,
    "animal":   0.60,
    "computer": 0.60,
}

# 是否在训练集构建时按 score 再做一次硬过滤。
# 当前 teacher 端已经按类阈值筛过，默认关闭，避免重复丢样本。
APPLY_SCORE_FILTER = False
CONF_FILTER = PROMPT_THRESHOLDS
VAL_THRESHOLDS = PROMPT_THRESHOLDS

# 稀有类过采样倍数（仅训练集）
RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 3,
}

# 是否纳入 hit=false 的负样本；用较低比例先补拒识能力
INCLUDE_NEGATIVE_SAMPLES = True
NEGATIVE_SAMPLE_RATIO = 0.15
NEGATIVE_SAMPLE_WEIGHT = 0.30

# ══════════════════════════════════════════════════════════════════════
# 模型
# ════════════════════════════════════════════════════════════════════════

MODEL_NAME = "CIDAS/clipseg-rd64-refined"
MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"

# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

IMG_SIZE = 768                # 等比例缩放 + pad 正方形
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
LOSS_WEIGHT_FLOOR = 0.7          # 逐样本置信度加权下限（保护弱样本不被过度降权）

# 边界弱监督：对伪标签边界环带做 ignore，不把残缺边界当硬真值
ENABLE_BOUNDARY_WEAK_SUPERVISION = True
BOUNDARY_IGNORE_WIDTH = 1        # 1024 pad 尺度下的边界忽略半径（像素）
BOUNDARY_IGNORE_MIN_AREA = 64    # 太小的目标不做边界忽略，避免极小目标被抹掉

# ══════════════════════════════════════════════════════════════════════
# 数据增强
# ══════════════════════════════════════════════════════════════════════

HFLIP_PROB = 0.5                # 红外水平翻转
GRAY2RGB = True                 # 灰度复制为 3 通道
# 反色统一：黑热（热目标=暗）→ 白热（热目标=亮）
BLACKHOT_SKEW_THRESH = -0.3     # 直方图偏度 < 此值判定为黑热
BLACKHOT_MEAN_THRESH = 200      # 均值 > 此值且非 vis 图，兜底反色
PSEUDO_COLOR_SAT_THRESH = 60.0  # 与 teacher 端一致，检测伪彩图
STD_LOW = 35.0                  # teacher 端低对比增强阈值
STD_MID = 50.0
NOISE_HIGH = 12.0               # teacher 端噪声估计阈值
NOISE_MED = 8.0
BLUR_LOW = 150.0                # teacher 端清晰度阈值
BLUR_MID = 300.0
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
        # 中文 + 中英混合短语
        "行人, 人, 人员",
        "一个行走或站立的人",
        "person 行人 human 人员",
        "场景中的人",
    ],
    "car": [
        "cars, vehicles, trucks, or any automobiles",
        "a car or vehicle on the road",
        "vehicles including cars and trucks",
        # 中文 + 中英混合短语
        "车辆, 汽车, 车",
        "一辆在路上的车",
        "car 汽车 vehicle 车辆",
    ],
    "building": [
        "buildings, houses, or structures",
        "any building or architectural structure",
        "houses and buildings",
        # 中文 + 中英混合短语
        "建筑, 楼房, 房屋",
        "建筑物或房屋结构",
        "building 建筑 house 房屋",
    ],
    "tree": [
        "trees, plants, bushes, or any vegetation",
        "trees and vegetation",
        "plants, bushes, and trees",
        # 中文 + 中英混合短语
        "树, 树木, 植物, 灌木",
        "树木和植被",
        "tree 树木 plant 植物",
    ],
    "animal": [
        "animal, wildlife",
        "any animal or wildlife creature",
        "animals in the wild",
        # 中文 + 中英混合短语
        "动物, 野生动物",
        "野外出现的动物",
        "animal 动物 wildlife 野生动物",
    ],
}

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "clipseg_v1"
EXIST_OK = True
