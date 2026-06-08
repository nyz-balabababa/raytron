from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HANXUE_ROOT = Path(__file__).resolve().parent

# 数据
CLASSES = ["person", "car", "building", "tree", "animal", "computer"]
PROMPT_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "computer": 0.60,
}

TRAIN_PRED_JSON = ROOT / "test" / "sam3_label_old" / "train_tasks" / "pred_train_tasks.json"
VAL_PRED_JSON = ROOT / "test" / "sam3_label_old" / "val_tasks1" / "pred_val_tasks1.json"
TRAIN_LIST = ROOT / "test" / "train_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT
TOKENIZER_DIR = HANXUE_ROOT / "weights" / "chinese_clip"
EFFICIENT_SAM_CKPT = HANXUE_ROOT / "weights" / "efficient_sam" / "efficient_sam_vitt.pt"

# teacher 伪标签输出阈值；训练验证与其保持一致
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

# 训练
IMG_SIZE = 768
BATCH = 4
EPOCHS = 25
WORKERS = 4
SEED = 42
DEVICE = "cuda"

DECODER_LR = 1e-4
IMAGE_LR = 1e-5
WEIGHT_DECAY = 0.03
WARMUP_EPOCHS = 2
MIN_LR_RATIO = 0.01
GRAD_CLIP = 1.0

# loss / 采样
LOSS_WEIGHT_FLOOR = 0.7
HFLIP_PROB = 0.5

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
FOCAL_WEIGHT = 0.0
FOCAL_GAMMA = 1.5
FOCAL_NEG_FACTOR = 0.25
FOCAL_POS_FACTOR = 1.0
CLASS_FOCAL_WEIGHT = {
    "person": 0.25,
    "car": 0.0,
    "building": 0.55,
    "tree": 0.55,
    "animal": 0.45,
    "computer": 0.60,
}

# 图像增强/预处理
PROMPT_AUG_PROB = 0.5
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200
PSEUDO_COLOR_SAT_THRESH = 60.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0

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
    "computer": [
        "computer",
        "computer monitor",
        "screen, computer, display",
        "monitor or display screen",
        "电脑, 显示器, 屏幕",
        "电脑设备或显示器",
        "computer 电脑 monitor 显示器 screen 屏幕",
    ],
}

# 输出
PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "efficientsam_easy_6cls_v1"


REQUIRED_PATHS = [
    TRAIN_PRED_JSON,
    VAL_PRED_JSON,
    TRAIN_LIST,
    VAL_LIST,
    TOKENIZER_DIR,
    EFFICIENT_SAM_CKPT,
]

for required_path in REQUIRED_PATHS:
    if not required_path.exists():
        raise FileNotFoundError(f"hanxue 配置路径不存在: {required_path}")
