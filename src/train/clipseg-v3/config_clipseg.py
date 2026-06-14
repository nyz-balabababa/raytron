"""
CLIPSeg 训练配置 —— 自包含，不再继承 config_common。
后续蒸馏实验（Grounding DINO 等）以此文件为模板，保持参数对齐。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ══════════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════════

CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "computer",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
]
PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST = ROOT / "test" / "trainval_list.txt"
VAL_LIST = ROOT / "test" / "val_list.txt"
IMAGE_ROOT = ROOT
USE_MANIFEST = False
MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest.json"

PROMPT_THRESHOLDS = {  # teacher 伪标签输出阈值，训练验证与其保持一致
    "person":     0.70,
    "tree":       0.60,
    "building":   0.70,
    "animal":     0.60,
    "car":        0.70,
    "computer":   0.60,
    "trash can":  0.55,
    "window":     0.50,
    "door":       0.50,
    "fence":      0.45,
    "pole_light": 0.55,
    "motorcycle": 0.55,
}

APPLY_SCORE_FILTER = False  # teacher 端已筛过，默认关闭避免重复丢样本
CONF_FILTER = PROMPT_THRESHOLDS
VAL_THRESHOLDS = PROMPT_THRESHOLDS

RARE_OVERSAMPLE = {  # 稀有类过采样倍数（仅训练集）
    "animal": 2,
    "computer": 2,
    "trash can": 2,
    "door": 2,
    "motorcycle": 2,
}

TINY_PROTECT_CLASSES = {  # 极小 mask 保护，不被当噪声丢弃
    "person",
    "car",
    "animal",
    "trash can",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
}
TINY_AREA_THRESHOLDS = {  # 各类面积下限（像素），低于此值不参与训练
    "person":     64,
    "car":        64,
    "animal":     64,
    "tree":       64,
    "building":   80,
    "trash can":  64,
    "door":       80,
    "fence":      80,
    "pole_light": 64,
    "motorcycle": 96,
}

INCLUDE_NEGATIVE_SAMPLES = True  # 纳入 hit=false 负样本，补拒识能力
NEGATIVE_SAMPLE_RATIO = 0.15  # 合训版类别变多，负样本降低一点
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
EPOCHS = 25                    # train+val 合训跑 25 轮，早停可提前结束
PATIENCE = 5                   # 早停：连续 N 轮 mIoU 不提升则提前结束
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

ENABLE_BOUNDARY_WEAK_SUPERVISION = True  # 边界环带 ignore，不把残缺边界当硬真值
BOUNDARY_IGNORE_WIDTH = 1        # 1024 pad 尺度下的边界忽略半径（像素）
BOUNDARY_IGNORE_MIN_AREA = 64    # 太小的目标不做边界忽略，避免极小目标被抹掉

# ══════════════════════════════════════════════════════════════════════
# 数据增强
# ══════════════════════════════════════════════════════════════════════

HFLIP_PROB = 0.5                # 红外水平翻转
GRAY2RGB = True                 # 灰度复制为 3 通道
BLACKHOT_SKEW_THRESH = -0.3     # 直方图偏度 < 此值判定为黑热（反色统一）
BLACKHOT_MEAN_THRESH = 200      # 均值 > 此值且非 vis 图，兜底反色
PSEUDO_COLOR_SAT_THRESH = 60.0  # 与 teacher 端一致，检测伪彩图
STD_LOW = 35.0                  # teacher 端低对比增强阈值
STD_MID = 50.0
NOISE_HIGH = 12.0               # teacher 端噪声估计阈值
NOISE_MED = 8.0
BLUR_LOW = 150.0                # teacher 端清晰度阈值
BLUR_MID = 300.0
CLIP_MEAN = [0.48145466, 0.52048427, 0.45053169]  # CLIP 预训练时的均值/标准差（RGB 三通道）
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
        "行人, 人, 人员",
        "一个行走或站立的人",
        "person 行人 human 人员",
        "场景中的人",
        "路人, 行人, 步行者",
        "画面里的人",
        "红外图像中的人",
        "人形目标, 人体",
        "person 人 行人 pedestrian",
    ],
    "car": [
        "cars, vehicles, trucks, or any automobiles",
        "a car or vehicle on the road",
        "vehicles including cars and trucks",
        "车辆, 汽车, 车",
        "一辆在路上的车",
        "car 汽车 vehicle 车辆",
        "轿车, 货车, 卡车",
        "停在路边的车",
        "红外图像中的车辆",
        "机动车, 交通工具",
        "car 车 vehicle 机动车 automobile",
    ],
    "building": [
        "buildings, houses, or structures",
        "any building or architectural structure",
        "houses and buildings",
        "建筑, 楼房, 房屋",
        "建筑物或房屋结构",
        "building 建筑 house 房屋",
        "高楼, 平房, 厂房",
        "城市建筑, 居民楼",
        "红外图像中的建筑物",
        "房屋建筑, 构筑物",
        "building 建筑 structure 房屋 楼房",
    ],
    "tree": [
        "trees, plants, bushes, or any vegetation",
        "trees and vegetation",
        "plants, bushes, and trees",
        "树, 树木, 植物, 灌木",
        "树木和植被",
        "tree 树木 plant 植物",
        "大树, 小树, 树丛",
        "路边的树, 绿化带",
        "红外图像中的树木",
        "植被, 灌木丛, 树林",
        "tree 树 树木 植被 vegetation",
    ],
    "animal": [
        "animal, wildlife",
        "any animal or wildlife creature",
        "animals in the wild",
        "动物, 野生动物",
        "野外出现的动物",
        "animal 动物 wildlife 野生动物",
        "猫, 狗, 狐狸, 狼, 熊",
        "四条腿的动物, 哺乳动物",
        "红外图像中的动物",
        "野生动物, 流浪动物",
        "animal 动物 野兽 牲畜 creature",
    ],
    "computer": [
        "computer, monitor, screen, keyboard, laptop",
        "a computer or display device",
        "computers and electronic devices",
        "电脑, 计算机, 显示器, 屏幕",
        "一台电脑或显示器",
        "computer 电脑 monitor 显示器",
        "笔记本, 台式机, 一体机",
        "桌上的电脑设备",
        "电子设备, 显示屏",
        "键盘, 鼠标, 电脑配件",
        "computer 计算机 laptop 笔记本 screen 屏幕",
    ],
    "trash can": [
        "trash can, garbage bin, dustbin, waste container",
        "a trash can or garbage bin",
        "垃圾桶, 垃圾箱",
        "一个垃圾桶",
        "trash can 垃圾桶 garbage bin 垃圾箱",
        "废物箱, 果皮箱, 回收桶",
        "路边的垃圾桶",
        "公共垃圾桶, 分类垃圾桶",
        "垃圾容器, 废物容器",
        "trash can 垃圾桶 bin 垃圾箱 dustbin 废物箱",
    ],
    "window": [
        "window",
        "a window on a building",
        "窗户, 窗口",
        "building 建筑 window 窗户",
        "一扇窗户, 玻璃窗",
        "建筑物的窗户",
        "窗框, 窗台, 窗玻璃",
        "外墙上的窗户",
        "window 窗户 窗口 玻璃窗",
    ],
    "door": [
        "door",
        "a door on a building",
        "门",
        "building 建筑 door 门",
        "一扇门, 大门, 入口",
        "建筑物的门",
        "房门, 铁门, 卷帘门",
        "门框, 门槛, 门口",
        "door 门 大门 入口 entrance",
    ],
    "fence": [
        "fence, railing, barrier",
        "a fence or barrier",
        "围栏, 栅栏, 护栏",
        "fence 围栏 barrier 护栏",
        "铁栅栏, 铁丝网, 栏杆",
        "围墙, 篱笆, 护栏网",
        "路边的护栏, 隔离栏",
        "金属围栏, 木栅栏",
        "fence 围栏 栅栏 护栏 railing barrier",
    ],
    "pole_light": [
        "pole, street light, utility pole",
        "a pole or street light",
        "杆子, 路灯, 电线杆, 灯柱",
        "pole 杆子 street light 路灯",
        "路灯杆, 灯柱, 照明灯",
        "电线杆, 水泥杆, 铁杆",
        "路边的灯杆, 监控杆",
        "竖立的杆子, 立柱",
        "pole 杆子 路灯 电线杆 street light lamp post",
    ],
    "motorcycle": [
        "motorcycle, motorbike, electric scooter",
        "a motorcycle or scooter",
        "摩托车, 电动车",
        "motorcycle 摩托车 scooter 电动车",
        "电瓶车, 电动自行车, 踏板车",
        "两轮车, 机动两轮车",
        "停在路边的摩托车",
        "外卖电动车, 快递三轮车",
        "motorcycle 摩托车 电动车 scooter e-bike 电瓶车",
    ],
}

# ══════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════

PROJECT = ROOT / "test" / "train_output"
RUN_NAME = "clipseg_xiyou_12cls_trainval_v1"  # 合训版新实验名，避免续训旧 checkpoint
EXIST_OK = True
