import sys
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent        # 当前脚本所在目录
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── 输入路径 ──────────────────────────────────────────────
TRAIN_PRED_JSON  = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON    = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST       = ROOT / "test" / "trainval_list.txt"
VAL_LIST         = ROOT / "test" / "val_list.txt"
IMAGE_ROOT       = ROOT
EFFICIENT_SAM_CKPT = ROOT / "src" / "DINO" / "models" / "efficientsam" / "efficient_sam_vitt.pt"

# ── 输出路径 ──────────────────────────────────────────────
OUTPUT_DIR                = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1"
TRAIN_LOG_FILENAME        = "train.log"
INSTANCE_CACHE_DIRNAME    = ".instance_cache"
LAST_PT_FILENAME          = "last.pt"
LAST_MODEL_ONLY_FILENAME  = "last_model_only.pt"
BEST_PT_FILENAME          = "best.pt"
BEST_MODEL_ONLY_FILENAME  = "best_model_only.pt"
BEST_POS_PT_FILENAME      = "best_pos.pt"
BEST_POS_MODEL_ONLY_FILENAME = "best_pos_model_only.pt"
RESULTS_CSV_FILENAME      = "results.csv"

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from efficient_sam.efficient_sam import build_efficient_sam
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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
]

# data / split
PROMPT_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "computer": 0.60,
    "trash can": 0.55,
    "window": 0.50,
    "door": 0.50,
    "fence": 0.45,
    "pole_light": 0.55,
}
MASK_THRESHOLDS = {
    "person": 0.50,
    "car": 0.50,
    "building": 0.50,
    "tree": 0.50,
    "animal": 0.50,
    "computer": 0.50,
    "trash can": 0.45,
    "window": 0.45,
    "door": 0.45,
    "fence": 0.45,
    "pole_light": 0.45,
}
CLASS_FOCAL_WEIGHT = {
    "person": 0.25,
    "car": 0.0,
    "building": 0.55,
    "tree": 0.55,
    "animal": 0.45,
    "computer": 0.60,
    "trash can": 0.60,
    "window": 0.60,
    "door": 0.60,
    "fence": 0.65,
    "pole_light": 0.70,
}
RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 3,
    "trash can": 3,
    "window": 3,
    "door": 3,
    "fence": 3,
    "pole_light": 3,
}

# sampling / augmentation
APPLY_SCORE_FILTER = False
NEGATIVE_SAMPLE_RATIO = 0.15
NEGATIVE_SAMPLE_WEIGHT = 0.30
LOSS_WEIGHT_FLOOR = 0.7
HFLIP_PROB = 0.5
MIN_COMPONENT_AREA = 4
BOX_EXPAND_RATIO = 0.08
BOX_EXPAND_RATIO_MIN = 0.05
BOX_EXPAND_RATIO_MAX = 0.15
BOX_JITTER_PROB = 0.5
BOX_JITTER_CENTER = 0.05
BOX_JITTER_SCALE = 0.15
SMALL_OBJECT_AREA = 32

# train
IMG_SIZE = 768
BATCH_SIZE = 4
EPOCHS = 5
NUM_WORKERS = 0  # Windows 上多进程 DataLoader 会卡死，必须设为 0
DEVICE = "cuda"
SEED = 42
BALANCED_SAMPLER = True
STEPS_PER_EPOCH = 2500      # 每 epoch 迭代多少个 batch，过大可能导致训练时间过长，过小可能导致模型收敛不稳定
MAX_VAL_SAMPLES = 5000
FREEZE_IMAGE_ENCODER = True

# optimizer / scheduler
LR = 1e-4
DECODER_LR = None
IMAGE_LR = None
WEIGHT_DECAY = 0.03
MODEL_TYPE = "vitt"
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.01
GRAD_CLIP = 1.0

# loss
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
FOCAL_AUX_WEIGHT = 0.05
FOCAL_GAMMA = 1.5
FOCAL_POS_FACTOR = 1.0
FOCAL_NEG_FACTOR = 0.25

# image preprocess
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200
PSEUDO_COLOR_SAT_THRESH = 60.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_tqdm(iterable, total, desc, leave=False):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        ncols=84,
        bar_format="{l_bar}{bar:14}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def resolve_device(device_name: str):
    requested = str(device_name).lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU。")
        device = torch.device(device_name if ":" in requested else "cuda:0")
        device_index = device.index if device.index is not None else 0
        logger.info(
            f"CUDA 可用: count={torch.cuda.device_count()} "
            f"current={device_index} name={torch.cuda.get_device_name(device_index)}"
        )
        return device
    device = torch.device(device_name)
    logger.info(f"使用非 CUDA 设备: {device}")
    return device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total_params, trainable_params


def resolve_existing_path(path_value):
    path = Path(path_value)
    if path.exists():
        return path
    candidates = [
        Path.cwd() / path,
        ROOT / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def rle_to_mask(rle):
    decode_error = None
    try:
        from pycocotools import mask as mask_utils

        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return mask_utils.decode(rle_copy).astype(np.uint8)
    except Exception as exc:
        decode_error = exc

    h, w = rle["size"]
    counts = rle["counts"]
    numeric_counts = None

    if isinstance(counts, (list, tuple)):
        numeric_counts = [int(x) for x in counts]
    elif isinstance(counts, bytes):
        counts = counts.decode("utf-8")

    if numeric_counts is None and isinstance(counts, str):
        stripped = counts.strip()
        if "," in stripped and all(part.strip().isdigit() for part in stripped.split(",") if part.strip()):
            numeric_counts = [int(part) for part in stripped.split(",") if part.strip()]
        else:
            raise RuntimeError(
                "检测到 COCO compressed RLE，但 pycocotools 不可用或 decode 失败。"
                "请安装 pycocotools 以解析 compressed RLE。"
            ) from decode_error

    if numeric_counts is None:
        raise RuntimeError("无法解析 RLE counts 格式。") from decode_error

    if not numeric_counts:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in numeric_counts:
        if val == 1:
            mask[pos : pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def compute_skewness(gray):
    gray_f = gray.astype(np.float32)
    mean = float(gray_f.mean())
    std = float(gray_f.std())
    if std < 1e-6:
        return 0.0
    return float(np.mean(((gray_f - mean) / std) ** 3))


def is_pseudo_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path: Path):
    img_bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {abs_path}")
    elif is_pseudo_color(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(abs_path).replace("\\", "/")
    if "blackHot" in fname:
        img = 255 - img
    else:
        mean = float(np.mean(img))
        std = float(np.std(img))
        if std > 0:
            skew = float(np.mean(((img - mean) / std) ** 3))
            if skew < BLACKHOT_SKEW_THRESH:
                img = 255 - img
        if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            img = 255 - img

    std = float(np.std(img))
    noise_sigma = estimate_noise_sigma(img)
    blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())

    if std < STD_LOW:
        img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
    elif std < STD_MID:
        img = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(img)

    if noise_sigma > NOISE_HIGH:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > NOISE_MED:
        img = cv2.medianBlur(img, 3)

    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < BLUR_MID:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = np.clip(cv2.addWeighted(img, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)

    return img


def letterbox_image(gray, target_hw, fill_value=0):
    target_h, target_w = target_hw
    orig_h, orig_w = gray.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w), fill_value, dtype=resized.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale": scale,
        "target_h": target_h,
        "target_w": target_w,
    }
    return padded, meta


def letterbox_mask(mask, meta, fill_value=0):
    target_h = meta["target_h"]
    target_w = meta["target_w"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.full((target_h, target_w), fill_value, dtype=resized.dtype)
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return padded


def clip_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    x2 = float(np.clip(x2, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    y2 = float(np.clip(y2, 0, max(height - 1, 0)))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def expand_box_xyxy(box, width, height, ratio):
    x1, y1, x2, y2 = box
    w = max(x2 - x1 + 1.0, 1.0)
    h = max(y2 - y1 + 1.0, 1.0)
    dx = w * ratio * 0.5
    dy = h * ratio * 0.5
    return clip_box_xyxy([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def jitter_box_xyxy(box, width, height, center_ratio=0.05, scale_ratio=0.15):
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1 + 1.0, 1.0)
    box_h = max(y2 - y1 + 1.0, 1.0)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    center_x += random.uniform(-center_ratio, center_ratio) * box_w
    center_y += random.uniform(-center_ratio, center_ratio) * box_h

    scale_w = 1.0 + random.uniform(-scale_ratio, scale_ratio)
    scale_h = 1.0 + random.uniform(-scale_ratio, scale_ratio)
    new_w = max(box_w * scale_w, 1.0)
    new_h = max(box_h * scale_h, 1.0)

    jittered = [
        center_x - 0.5 * new_w,
        center_y - 0.5 * new_h,
        center_x + 0.5 * new_w,
        center_y + 0.5 * new_h,
    ]
    jittered = clip_box_xyxy(jittered, width, height)
    if jittered[2] <= jittered[0]:
        jittered[2] = min(width - 1, jittered[0] + 1.0)
    if jittered[3] <= jittered[1]:
        jittered[3] = min(height - 1, jittered[1] + 1.0)
    return clip_box_xyxy(jittered, width, height)


def letterbox_box_xyxy(box, meta):
    x1, y1, x2, y2 = box
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    return [
        x1 * scale + pad_left,
        y1 * scale + pad_top,
        x2 * scale + pad_left,
        y2 * scale + pad_top,
    ]


def hflip_box_xyxy(box, width):
    x1, y1, x2, y2 = box
    new_x1 = (width - 1) - x2
    new_x2 = (width - 1) - x1
    return [min(new_x1, new_x2), y1, max(new_x1, new_x2), y2]


def dice_loss_with_logits(logits, targets, sample_weight=None, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    loss = 1.0 - (2.0 * inter + eps) / (union + eps)
    if sample_weight is not None:
        weights = sample_weight.float()
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1e-6)
    return loss.mean()


def bce_dice_loss(logits, targets, sample_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if sample_weight is not None:
        weights = sample_weight.float().view(-1, 1, 1, 1)
        bce_loss = (bce * weights).sum() / weights.expand_as(bce).sum().clamp_min(1e-6)
    else:
        bce_loss = bce.mean()
    dice_loss = dice_loss_with_logits(logits, targets, sample_weight=sample_weight)
    total = BCE_WEIGHT * bce_loss + DICE_WEIGHT * dice_loss
    return total, bce_loss, dice_loss


def class_focal_aux_loss(logits, targets, prompts, sample_weight):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    focal_map = ((1.0 - pt).clamp(min=1e-6) ** FOCAL_GAMMA) * bce

    pos_mask = (targets > 0.5).float()
    neg_mask = 1.0 - pos_mask
    flat_focal = focal_map.flatten(1)
    flat_pos = pos_mask.flatten(1)
    flat_neg = neg_mask.flatten(1)
    pos_area = flat_pos.sum(dim=1)
    neg_area = flat_neg.sum(dim=1).clamp(min=1.0)
    pos_loss = (flat_focal * flat_pos).sum(dim=1) / pos_area.clamp(min=1.0)
    neg_loss = (flat_focal * flat_neg).sum(dim=1) / neg_area
    has_pos = pos_area > 0
    per_sample = torch.where(
        has_pos,
        FOCAL_POS_FACTOR * pos_loss + FOCAL_NEG_FACTOR * neg_loss,
        FOCAL_NEG_FACTOR * neg_loss,
    )
    cls_w = torch.tensor(
        [float(CLASS_FOCAL_WEIGHT.get(str(prompt), 0.0)) for prompt in prompts],
        device=logits.device,
        dtype=per_sample.dtype,
    )
    sw = sample_weight.float()
    sw = torch.where(has_pos, sw.clamp(min=LOSS_WEIGHT_FLOOR), sw)
    return (per_sample * cls_w * sw).mean()


@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    gt = (targets > 0.5).float()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()
    if pred_sum == 0 and gt_sum == 0:
        return {"iou": 1.0, "dice": 1.0}
    inter = (pred * gt).sum().item()
    union = (pred + gt).clamp(0, 1).sum().item()
    return {
        "iou": inter / (union + 1e-7),
        "dice": 2 * inter / (pred_sum + gt_sum + 1e-7),
    }


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("无法从 checkpoint 中解析 state_dict。")


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def build_model(model_type: str, checkpoint_path: Path):
    if model_type == "vitt":
        encoder_patch_embed_dim = 192
        encoder_num_heads = 3
    elif model_type == "vits":
        encoder_patch_embed_dim = 384
        encoder_num_heads = 6
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = build_efficient_sam(
        encoder_patch_embed_dim=encoder_patch_embed_dim,
        encoder_num_heads=encoder_num_heads,
        checkpoint=None,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"已加载 EfficientSAM 权重: {checkpoint_path}")
    if missing_keys:
        logger.warning(f"missing_keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"unexpected_keys: {len(unexpected_keys)}")
    return model


def build_optimizer(model, decoder_lr, image_lr, weight_decay):
    image_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder") or ".image_encoder" in name:
            image_params.append(param)
        elif (
            name.startswith("mask_decoder")
            or name.startswith("prompt_encoder")
            or ".mask_decoder" in name
            or ".prompt_encoder" in name
        ):
            decoder_params.append(param)
        else:
            other_params.append(param)
            logger.warning(f"[ParamGroup:other_decoder_lr] {name}")

    param_groups = []
    if image_params:
        param_groups.append({"params": image_params, "lr": image_lr, "name": "image"})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder"})
    if other_params:
        param_groups.append({"params": other_params, "lr": decoder_lr, "name": "other"})

    logger.info("[Optimizer] 参数组:")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"]) / 1e6
        logger.info(f"  {group['name']}: lr={group['lr']}, params={n_params:.2f}M")

    return torch.optim.AdamW(
        [{"params": group["params"], "lr": group["lr"], "weight_decay": weight_decay} for group in param_groups],
        weight_decay=weight_decay,
    )


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.01):
    if total_steps <= 0:
        return None

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def forward_efficient_sam(model, images, boxes):
    batch_size = images.shape[0]
    device = images.device
    input_points = torch.zeros((batch_size, 1, 2, 2), dtype=torch.float32, device=device)
    input_labels = torch.zeros((batch_size, 1, 2), dtype=torch.int64, device=device)
    input_points[:, 0, 0, 0] = boxes[:, 0]
    input_points[:, 0, 0, 1] = boxes[:, 1]
    input_points[:, 0, 1, 0] = boxes[:, 2]
    input_points[:, 0, 1, 1] = boxes[:, 3]
    input_labels[:, 0, 0] = 2
    input_labels[:, 0, 1] = 3

    outputs = model(images, input_points, input_labels)
    if not isinstance(outputs, tuple):
        raise RuntimeError("EfficientSAM forward 预期返回 (masks, iou_predictions)。")

    pred_masks, pred_ious = outputs
    if pred_masks.ndim != 5:
        raise RuntimeError(f"Unexpected EfficientSAM mask shape: {pred_masks.shape}")

    if pred_ious is not None and pred_ious.ndim == 3:
        best_idx = pred_ious.argmax(dim=2, keepdim=True)
    else:
        best_idx = torch.zeros(
            (pred_masks.size(0), pred_masks.size(1), 1),
            dtype=torch.long,
            device=pred_masks.device,
        )

    gather_index = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, pred_masks.size(-2), pred_masks.size(-1))
    best_masks = torch.gather(pred_masks, 2, gather_index).squeeze(2)
    if best_masks.ndim != 4:
        raise RuntimeError(f"Unexpected gathered mask shape: {best_masks.shape}")
    return best_masks


class EfficientSAMBoxPromptDataset(Dataset):
    def __init__(
        self,
        pred_json,
        split_txt,
        image_root,
        cache_root,
        img_size,
        class_names,
        prompt_thresholds,
        apply_score_filter=False,
        training=True,
        hflip_prob=0.0,
        min_component_area=4,
        box_expand_ratio=0.08,
        box_expand_ratio_min=0.05,
        box_expand_ratio_max=0.15,
        small_object_area=32,
        loss_weight_floor=0.7,
        negative_sample_ratio=0.0,
        negative_sample_weight=0.30,
        box_jitter_prob=0.5,
        box_jitter_center=0.05,
        box_jitter_scale=0.15,
        rare_oversample=None,
        seed=42,
    ):
        self.pred_json = resolve_existing_path(pred_json)
        self.split_txt = resolve_existing_path(split_txt)
        self.image_root = resolve_existing_path(image_root)
        self.cache_root = Path(cache_root)
        self.img_size = (int(img_size), int(img_size))
        self.class_names = set(class_names)
        self.prompt_thresholds = dict(prompt_thresholds)
        self.apply_score_filter = apply_score_filter
        self.training = training
        self.hflip_prob = hflip_prob
        self.min_component_area = int(min_component_area)
        self.box_expand_ratio = float(box_expand_ratio)
        self.box_expand_ratio_min = float(box_expand_ratio_min)
        self.box_expand_ratio_max = float(box_expand_ratio_max)
        self.small_object_area = int(small_object_area)
        self.loss_weight_floor = float(loss_weight_floor)
        self.negative_sample_ratio = float(negative_sample_ratio)
        self.negative_sample_weight = float(negative_sample_weight)
        self.box_jitter_prob = float(box_jitter_prob)
        self.box_jitter_center = float(box_jitter_center)
        self.box_jitter_scale = float(box_jitter_scale)
        self.rare_oversample = dict(rare_oversample or {})
        self.rng = random.Random(seed)
        self.allowed = self._load_allowed_set()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stats = self._init_stats()
        self.samples = self._build_samples()

    def _init_stats(self):
        return {
            "source_images": 0,
            "used_images": 0,
            "instances_raw": 0,
            "instances_final": 0,
            "per_class_instances": {class_name: 0 for class_name in sorted(self.class_names)},
            "per_class_small_instances": {class_name: 0 for class_name in sorted(self.class_names)},
            "skipped_empty_masks": 0,
            "filtered_small_components": 0,
            "missing_images": 0,
            "negative_prompts_seen": 0,
            "negative_prompts_skipped": 0,
            "score_filtered_positives": 0,
            "missing_rle": 0,
            "non_class_prompts": 0,
        }

    def _load_allowed_set(self):
        with open(self.split_txt, "r", encoding="utf-8") as handle:
            return {line.strip().replace("\\", "/") for line in handle if line.strip()}

    def _normalize_image_path(self, image_path):
        image_path = str(image_path).replace("\\", "/")
        if image_path in self.allowed:
            return image_path
        alt = image_path[5:] if image_path.startswith("test/") else f"test/{image_path}"
        if alt in self.allowed:
            return alt
        return None

    def _resolve_image_path(self, image_path):
        rel = image_path.replace("\\", "/")
        candidates = [
            self.image_root / rel,
            self.image_root / rel[5:] if rel.startswith("test/") else self.image_root / "test" / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _score_threshold(self, prompt):
        return self.prompt_thresholds.get(str(prompt), 0.60)

    def _cache_mask_path(self, image_path, prompt, component_idx, score, rle):
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8", errors="ignore")
        stable_key = json.dumps(
            [
                self.pred_json.stem,
                image_path,
                prompt,
                component_idx,
                round(float(score), 6),
                rle.get("size"),
                str(counts)[:64],
            ],
            ensure_ascii=False,
        )
        cache_name = hashlib.md5(stable_key.encode("utf-8")).hexdigest()
        return self.cache_root / f"{cache_name}.png"

    def _build_samples(self):
        with open(self.pred_json, "r", encoding="utf-8-sig") as handle:
            preds = json.load(handle)

        base_samples = []
        used_images = set()

        desc = f"Building {self.pred_json.stem}" if self.training else f"Building {self.pred_json.stem} (val)"
        for item in maybe_tqdm(preds, total=len(preds), desc=desc):
            self.stats["source_images"] += 1
            image_path = self._normalize_image_path(item.get("image_path", ""))
            if image_path is None:
                continue

            abs_path = self._resolve_image_path(image_path)
            if not abs_path.exists():
                self.stats["missing_images"] += 1
                continue

            for prompt, prompt_info in item.get("prompts", {}).items():
                prompt = str(prompt)
                if prompt not in self.class_names:
                    self.stats["non_class_prompts"] += 1
                    continue

                hit = bool(prompt_info.get("hit"))
                score = float(prompt_info.get("score", 1.0))

                if not hit:
                    self.stats["negative_prompts_seen"] += 1
                    if self.negative_sample_ratio > 0 and self.rng.random() < self.negative_sample_ratio:
                        self.stats["negative_prompts_skipped"] += 1
                    continue

                if self.apply_score_filter and score < self._score_threshold(prompt):
                    self.stats["score_filtered_positives"] += 1
                    continue

                rle = prompt_info.get("rle")
                if rle is None:
                    self.stats["missing_rle"] += 1
                    continue

                union_mask = rle_to_mask(rle)
                union_mask = (union_mask > 0).astype(np.uint8)
                if union_mask.sum() <= 0:
                    self.stats["skipped_empty_masks"] += 1
                    continue

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
                kept_any = False

                for component_idx in range(1, num_labels):
                    area = int(stats[component_idx, cv2.CC_STAT_AREA])
                    if area < self.min_component_area:
                        self.stats["filtered_small_components"] += 1
                        continue

                    left = int(stats[component_idx, cv2.CC_STAT_LEFT])
                    top = int(stats[component_idx, cv2.CC_STAT_TOP])
                    width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
                    height = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
                    bbox = [left, top, left + width - 1, top + height - 1]

                    instance_mask = (labels == component_idx).astype(np.uint8)
                    if instance_mask.sum() <= 0:
                        self.stats["skipped_empty_masks"] += 1
                        continue

                    cache_path = self._cache_mask_path(image_path, prompt, component_idx, score, rle)
                    if not cache_path.exists():
                        cv2.imwrite(str(cache_path), instance_mask * 255)

                    sample_weight = max(score, self.loss_weight_floor)
                    base_samples.append(
                        {
                            "image_path": image_path,
                            "mask_path": cache_path,
                            "bbox": bbox,
                            "prompt": prompt,
                            "sample_weight": sample_weight,
                            "instance_area": area,
                        }
                    )
                    self.stats["instances_raw"] += 1
                    self.stats["per_class_instances"][prompt] += 1
                    if area < self.small_object_area:
                        self.stats["per_class_small_instances"][prompt] += 1
                    kept_any = True
                    used_images.add(image_path)

                if not kept_any:
                    self.stats["skipped_empty_masks"] += 1

        if self.training and self.rare_oversample:
            final_samples = []
            for sample in base_samples:
                repeat = int(self.rare_oversample.get(sample["prompt"], 1))
                for _ in range(max(repeat, 1)):
                    final_samples.append(sample.copy())
        else:
            final_samples = base_samples

        self.stats["used_images"] = len(used_images)
        self.stats["instances_final"] = len(final_samples)
        return final_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample["image_path"])
        gray = load_teacher_aligned_gray(image_path)
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"实例 mask 不存在或无法读取: {sample['mask_path']}")
        mask = (mask > 0).astype(np.float32)

        orig_h, orig_w = gray.shape[:2]
        if self.training:
            expand_ratio = self.rng.uniform(self.box_expand_ratio_min, self.box_expand_ratio_max)
        else:
            expand_ratio = self.box_expand_ratio
        box = expand_box_xyxy(sample["bbox"], orig_w, orig_h, expand_ratio)

        gray, meta = letterbox_image(gray, self.img_size, fill_value=0)
        mask = letterbox_mask(mask, meta, fill_value=0)
        box = letterbox_box_xyxy(box, meta)
        box = clip_box_xyxy(box, self.img_size[1], self.img_size[0])

        if self.training and self.rng.random() < self.box_jitter_prob:
            box = jitter_box_xyxy(
                box,
                self.img_size[1],
                self.img_size[0],
                center_ratio=self.box_jitter_center,
                scale_ratio=self.box_jitter_scale,
            )

        if self.training and self.rng.random() < self.hflip_prob:
            gray = np.fliplr(gray)
            mask = np.fliplr(mask)
            box = hflip_box_xyxy(box, self.img_size[1])
            box = clip_box_xyxy(box, self.img_size[1], self.img_size[0])

        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        mask = (mask > 0.5).astype(np.float32)[None, ...]

        return {
            "image": torch.from_numpy(rgb).float(),
            "mask": torch.from_numpy(mask).float(),
            "box": torch.tensor(box, dtype=torch.float32),
            "sample_weight": torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
            "prompt": sample["prompt"],
            "class_name": sample["prompt"],
            "image_path": sample["image_path"],
            "instance_area": sample["instance_area"],
        }


def summarize_dataset(name, dataset):
    stats = dataset.stats
    logger.info(
        f"[{name}] images={stats['used_images']} instances_raw={stats['instances_raw']} "
        f"instances_final={stats['instances_final']}"
    )
    logger.info(f"[{name}] per_class_instances={stats['per_class_instances']}")
    logger.info(f"[{name}] per_class_small_instances={stats['per_class_small_instances']}")
    logger.info(
        f"[{name}] skipped_empty_masks={stats['skipped_empty_masks']} "
        f"filtered_small_components={stats['filtered_small_components']} "
        f"missing_images={stats['missing_images']}"
    )
    logger.info(
        f"[{name}] negative_prompts_seen={stats['negative_prompts_seen']} "
        f"negative_prompts_skipped={stats['negative_prompts_skipped']} "
        f"score_filtered_positives={stats['score_filtered_positives']} "
        f"missing_rle={stats['missing_rle']}"
    )
    logger.info("hit=false negative prompts are counted but not used for EfficientSAM box-prompt training.")


def smoke_test_forward(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    boxes = batch["box"].to(device, non_blocking=True)

    with torch.no_grad():
        logits = forward_efficient_sam(model, images, boxes)

    logger.info(f"[SmokeTest] images: {tuple(images.shape)}")
    logger.info(f"[SmokeTest] masks:  {tuple(masks.shape)}")
    logger.info(f"[SmokeTest] boxes:  {tuple(boxes.shape)}")
    logger.info(f"[SmokeTest] logits: {tuple(logits.shape)}")
    logger.info(
        f"[SmokeTest] image value min/max: {images.min().item():.4f} / {images.max().item():.4f}"
    )
    logger.info(f"[SmokeTest] first box: {boxes[0].tolist()}")

    assert logits.ndim == 4, f"logits 应为 4D [B,1,H,W]，当前 {logits.shape}"
    assert logits.shape[1] == 1, f"logits channel 应为 1，当前 {logits.shape}"


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_focal_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    valid_batches = 0

    progress = maybe_tqdm(loader, total=len(loader), desc=f"Train {epoch}/{epochs}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        prompts = batch["class_name"]

        optimizer.zero_grad(set_to_none=True)
        logits = forward_efficient_sam(model, images, boxes)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        main_loss, bce_loss, dice_loss = bce_dice_loss(logits, masks, sample_weight=sample_weight)
        focal_loss = class_focal_aux_loss(logits, masks, prompts, sample_weight) if FOCAL_AUX_WEIGHT > 0 else torch.zeros((), device=device)
        loss = main_loss + FOCAL_AUX_WEIGHT * focal_loss

        if not torch.isfinite(loss):
            logger.warning(f"Epoch {epoch}: loss 非有限值，跳过一个 batch。")
            continue

        loss.backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad and param.grad is not None],
                GRAD_CLIP,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_main_loss += float(main_loss.item())
        total_focal_loss += float(focal_loss.item())
        total_bce_loss += float(bce_loss.item())
        total_dice_loss += float(dice_loss.item())
        valid_batches += 1

    if valid_batches == 0:
        raise RuntimeError(f"Epoch {epoch} 没有任何有效 batch。")

    return {
        "loss": total_loss / valid_batches,
        "main": total_main_loss / valid_batches,
        "focal": total_focal_loss / valid_batches,
        "bce": total_bce_loss / valid_batches,
        "dice": total_dice_loss / valid_batches,
    }


@torch.no_grad()
def validate(model, loader, device, epoch, epochs):
    model.eval()
    all_metrics = defaultdict(list)
    total_loss = 0.0
    total_main_loss = 0.0
    total_focal_loss = 0.0
    valid_batches = 0

    progress = maybe_tqdm(loader, total=len(loader), desc=f"Val {epoch}/{epochs}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        prompts = batch["class_name"]

        logits = forward_efficient_sam(model, images, boxes)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        main_loss, _, _ = bce_dice_loss(logits, masks, sample_weight=sample_weight)
        focal_loss = class_focal_aux_loss(logits, masks, prompts, sample_weight) if FOCAL_AUX_WEIGHT > 0 else torch.zeros((), device=device)
        loss = main_loss + FOCAL_AUX_WEIGHT * focal_loss
        total_loss += float(loss.item())
        total_main_loss += float(main_loss.item())
        total_focal_loss += float(focal_loss.item())
        valid_batches += 1

        for idx in range(images.size(0)):
            threshold = MASK_THRESHOLDS.get(prompts[idx], 0.5)
            metrics = compute_metrics(logits[idx : idx + 1], masks[idx : idx + 1], threshold=threshold)
            all_metrics["iou/overall"].append(metrics["iou"])
            all_metrics["dice/overall"].append(metrics["dice"])
            all_metrics["iou/pos_only"].append(metrics["iou"])
            all_metrics["dice/pos_only"].append(metrics["dice"])
            all_metrics[f"iou/{prompts[idx]}"].append(metrics["iou"])

    if valid_batches == 0:
        raise RuntimeError("验证集没有有效 batch。")

    metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
    metrics["loss/overall"] = total_loss / valid_batches
    metrics["loss/main"] = total_main_loss / valid_batches
    metrics["loss/focal"] = total_focal_loss / valid_batches
    return metrics


def save_checkpoint(path, model_only_path, epoch, model, optimizer, scheduler, history, best_overall_iou, best_pos_iou):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "history": dict(history),
        "best_overall_iou": best_overall_iou,
        "best_pos_iou": best_pos_iou,
    }
    torch.save(checkpoint, path)
    torch.save(model.state_dict(), model_only_path)


def write_results_csv(results_csv, history):
    keys = [
        "train_loss",
        "train_main",
        "train_bce",
        "train_dice",
        "train_focal",
        "val_loss",
        "val_miou",
        "val_pos_miou",
        "val_dice",
        "val_pos_dice",
        "lr",
    ] + [f"iou/{class_name}" for class_name in CLASSES]
    with open(results_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in keys])


def load_resume_checkpoint(model, optimizer, scheduler, resume_path, device, history):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    for key, values in checkpoint.get("history", {}).items():
        history[key] = list(values)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_overall_iou = float(checkpoint.get("best_overall_iou", -1.0))
    best_pos_iou = float(checkpoint.get("best_pos_iou", -1.0))
    return start_epoch, best_overall_iou, best_pos_iou


def freeze_image_encoder(model):
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("image_encoder") or ".image_encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
    logger.info(f"[Freeze] image_encoder frozen params={frozen / 1e6:.2f}M")


def build_balanced_sampler(dataset, num_samples):
    class_counts = defaultdict(int)
    for sample in dataset.samples:
        class_counts[sample["prompt"]] += 1

    weights = []
    for sample in dataset.samples:
        cls = sample["prompt"]
        weights.append(1.0 / max(class_counts[cls], 1))

    logger.info(f"[Sampler] class_counts={dict(class_counts)}")
    logger.info(f"[Sampler] balanced num_samples={num_samples}")

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
    )


def limit_dataset_stratified(dataset, max_samples, seed=42, name="dataset"):
    if max_samples <= 0 or len(dataset.samples) <= max_samples:
        return

    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in dataset.samples:
        groups[sample["prompt"]].append(sample)

    for cls in groups:
        rng.shuffle(groups[cls])

    classes = sorted(groups.keys())
    base_quota = max_samples // max(len(classes), 1)

    selected = []
    leftovers = []

    for cls in classes:
        cls_samples = groups[cls]
        take = min(base_quota, len(cls_samples))
        selected.extend(cls_samples[:take])
        leftovers.extend(cls_samples[take:])

    rng.shuffle(leftovers)
    remain = max_samples - len(selected)
    if remain > 0:
        selected.extend(leftovers[:remain])

    rng.shuffle(selected)
    dataset.samples = selected

    final_counts = defaultdict(int)
    for sample in dataset.samples:
        final_counts[sample["prompt"]] += 1

    logger.info(f"[Limit] {name} samples limited to {len(dataset.samples)}")
    logger.info(f"[Limit] {name} class_counts={dict(final_counts)}")


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientSAM box-to-mask fine-tune with CLIPSeg-aligned pseudo labels.")
    parser.add_argument(
        "--train-pred-json",
        default=str(TRAIN_PRED_JSON),
    )
    parser.add_argument(
        "--val-pred-json",
        default=str(VAL_PRED_JSON),
    )
    parser.add_argument("--train-list", default=str(TRAIN_LIST))
    parser.add_argument("--val-list", default=str(VAL_LIST))
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument(
        "--efficient-sam-ckpt",
        default=str(EFFICIENT_SAM_CKPT),
    )
    parser.add_argument("--model-type", default=MODEL_TYPE, choices=["vitt", "vits"])
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    parser.add_argument("--image-lr", type=float, default=IMAGE_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-area", type=int, default=MIN_COMPONENT_AREA)
    parser.add_argument("--box-expand-ratio", type=float, default=BOX_EXPAND_RATIO)
    parser.add_argument("--box-expand-ratio-min", type=float, default=BOX_EXPAND_RATIO_MIN)
    parser.add_argument("--box-expand-ratio-max", type=float, default=BOX_EXPAND_RATIO_MAX)
    parser.add_argument("--negative-sample-ratio", type=float, default=NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--negative-sample-weight", type=float, default=NEGATIVE_SAMPLE_WEIGHT)
    parser.add_argument("--balanced-sampler", action="store_true", default=BALANCED_SAMPLER)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--max-val-samples", type=int, default=MAX_VAL_SAMPLES)
    parser.add_argument("--freeze-image-encoder", action="store_true", default=FREEZE_IMAGE_ENCODER)
    return parser.parse_args()


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / TRAIN_LOG_FILENAME)

    set_seed(args.seed)
    device = resolve_device(args.device)
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.lr
    image_lr = args.image_lr if args.image_lr is not None else 1e-5

    logger.info("========== EfficientSAM CLIPSeg Flow ==========")
    logger.info(f"classes: {CLASSES}")
    logger.info(f"train_pred_json: {args.train_pred_json}")
    logger.info(f"val_pred_json: {args.val_pred_json}")
    logger.info(f"train_list: {args.train_list}")
    logger.info(f"val_list: {args.val_list}")
    logger.info(f"image_root: {args.image_root}")
    logger.info(f"img_size: {args.img_size}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"epochs: {args.epochs}")
    logger.info(f"decoder_lr: {decoder_lr}")
    logger.info(f"image_lr: {image_lr}")
    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"apply_score_filter: {APPLY_SCORE_FILTER}")
    logger.info(f"prompt_thresholds(score filter): {PROMPT_THRESHOLDS}")
    logger.info(f"mask_thresholds(eval binarization): {MASK_THRESHOLDS}")
    logger.info(f"negative_sample_ratio(reference): {args.negative_sample_ratio}")
    logger.info(f"negative_sample_weight(reference): {args.negative_sample_weight}")
    logger.info(
        f"box_expand_ratio(val fixed)={args.box_expand_ratio}, "
        f"train_range=({args.box_expand_ratio_min}, {args.box_expand_ratio_max})"
    )
    logger.info(
        f"box_jitter: prob={BOX_JITTER_PROB} center={BOX_JITTER_CENTER} scale={BOX_JITTER_SCALE}"
    )
    logger.info(f"checkpoint_dir: {output_dir}")
    logger.info(f"num_workers: {args.num_workers}")
    logger.info(f"balanced_sampler: {args.balanced_sampler}")
    logger.info(f"steps_per_epoch: {args.steps_per_epoch}")
    logger.info(f"max_val_samples: {args.max_val_samples}")
    logger.info(f"freeze_image_encoder: {args.freeze_image_encoder}")

    model = build_model(args.model_type, resolve_existing_path(args.efficient_sam_ckpt)).to(device)
    if args.freeze_image_encoder:
        freeze_image_encoder(model)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"总参数量: {total_params:.2f} M")
    logger.info(f"可训练参数量: {trainable_params:.2f} M")

    cache_root = output_dir / INSTANCE_CACHE_DIRNAME
    train_dataset = EfficientSAMBoxPromptDataset(
        pred_json=args.train_pred_json,
        split_txt=args.train_list,
        image_root=args.image_root,
        cache_root=cache_root / "train",
        img_size=args.img_size,
        class_names=CLASSES,
        prompt_thresholds=PROMPT_THRESHOLDS,
        apply_score_filter=APPLY_SCORE_FILTER,
        training=True,
        hflip_prob=HFLIP_PROB,
        min_component_area=args.min_area,
        box_expand_ratio=args.box_expand_ratio,
        box_expand_ratio_min=args.box_expand_ratio_min,
        box_expand_ratio_max=args.box_expand_ratio_max,
        small_object_area=SMALL_OBJECT_AREA,
        loss_weight_floor=LOSS_WEIGHT_FLOOR,
        negative_sample_ratio=args.negative_sample_ratio,
        negative_sample_weight=args.negative_sample_weight,
        box_jitter_prob=BOX_JITTER_PROB,
        box_jitter_center=BOX_JITTER_CENTER,
        box_jitter_scale=BOX_JITTER_SCALE,
        rare_oversample=RARE_OVERSAMPLE,
        seed=args.seed,
    )
    val_dataset = EfficientSAMBoxPromptDataset(
        pred_json=args.val_pred_json,
        split_txt=args.val_list,
        image_root=args.image_root,
        cache_root=cache_root / "val",
        img_size=args.img_size,
        class_names=CLASSES,
        prompt_thresholds=PROMPT_THRESHOLDS,
        apply_score_filter=False,
        training=False,
        hflip_prob=0.0,
        min_component_area=args.min_area,
        box_expand_ratio=args.box_expand_ratio,
        box_expand_ratio_min=args.box_expand_ratio_min,
        box_expand_ratio_max=args.box_expand_ratio_max,
        small_object_area=SMALL_OBJECT_AREA,
        loss_weight_floor=LOSS_WEIGHT_FLOOR,
        negative_sample_ratio=0.0,
        negative_sample_weight=args.negative_sample_weight,
        box_jitter_prob=0.0,
        box_jitter_center=BOX_JITTER_CENTER,
        box_jitter_scale=BOX_JITTER_SCALE,
        rare_oversample=None,
        seed=args.seed,
    )
    summarize_dataset("train", train_dataset)
    summarize_dataset("val", val_dataset)

    # 验证集按类分层限量，不随机截断
    limit_dataset_stratified(
        val_dataset,
        max_samples=args.max_val_samples,
        seed=args.seed,
        name="val",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("训练集实例数为 0。")
    if len(val_dataset) == 0:
        raise RuntimeError("验证集实例数为 0。")

    # 类别均衡采样器 —— 完整 samples 保留，每 epoch 按反频率加权采样
    train_sampler = None
    if args.balanced_sampler:
        if args.steps_per_epoch <= 0:
            raise ValueError("--balanced-sampler 开启时必须设置 --steps-per-epoch > 0")
        train_sampler = build_balanced_sampler(
            train_dataset,
            num_samples=args.steps_per_epoch * args.batch_size,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    logger.info(f"train images={train_dataset.stats['used_images']} train instances={len(train_dataset)}")
    logger.info(f"val images={val_dataset.stats['used_images']} val instances={len(val_dataset)}")
    logger.info(f"effective train steps per epoch: {len(train_loader)}")
    logger.info(f"effective val steps per epoch: {len(val_loader)}")

    smoke_test_forward(model, train_loader, device)

    optimizer = build_optimizer(model, decoder_lr, image_lr, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, MIN_LR_RATIO)

    history = defaultdict(list)
    best_overall_iou = -1.0
    best_pos_iou = -1.0
    start_epoch = 1

    last_pt = output_dir / LAST_PT_FILENAME
    last_model_only_pt = output_dir / LAST_MODEL_ONLY_FILENAME
    best_pt = output_dir / BEST_PT_FILENAME
    best_model_only_pt = output_dir / BEST_MODEL_ONLY_FILENAME
    best_pos_pt = output_dir / BEST_POS_PT_FILENAME
    best_pos_model_only_pt = output_dir / BEST_POS_MODEL_ONLY_FILENAME
    results_csv = output_dir / RESULTS_CSV_FILENAME

    resume_path = Path(args.resume) if args.resume else (last_pt if last_pt.exists() else None)
    if resume_path is not None and resume_path.exists():
        logger.info(f"恢复 checkpoint: {resume_path}")
        start_epoch, best_overall_iou, best_pos_iou = load_resume_checkpoint(
            model, optimizer, scheduler, resume_path, device, history
        )
        if start_epoch > args.epochs:
            logger.info(f"已完成训练: start_epoch={start_epoch}, epochs={args.epochs}")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{args.epochs}")
        start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs)
        val_metrics = validate(model, val_loader, device, epoch, args.epochs)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_min = (time.time() - start_time) / 60.0

        history["train_loss"].append(train_stats["loss"])
        history["train_main"].append(train_stats["main"])
        history["train_bce"].append(train_stats["bce"])
        history["train_dice"].append(train_stats["dice"])
        history["train_focal"].append(train_stats["focal"])
        history["val_loss"].append(val_metrics.get("loss/overall", 0.0))
        history["val_miou"].append(val_metrics.get("iou/overall", 0.0))
        history["val_pos_miou"].append(val_metrics.get("iou/pos_only", 0.0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
        history["val_pos_dice"].append(val_metrics.get("dice/pos_only", 0.0))
        history["lr"].append(current_lr)
        for class_name in CLASSES:
            history[f"iou/{class_name}"].append(val_metrics.get(f"iou/{class_name}", 0.0))

        logger.info(
            f"TrainLoss={train_stats['loss']:.4f} Main={train_stats['main']:.4f} "
            f"BCE={train_stats['bce']:.4f} Dice={train_stats['dice']:.4f} "
            f"Focal={train_stats['focal']:.4f} ValLoss={val_metrics.get('loss/overall', 0.0):.4f} "
            f"mIoU={val_metrics.get('iou/overall', 0.0):.4f} pos_mIoU={val_metrics.get('iou/pos_only', 0.0):.4f} "
            f"Dice={val_metrics.get('dice/overall', 0.0):.4f} LR={current_lr:.2e} {elapsed_min:.1f}min"
        )
        logger.info("Per-class IoU: " + "  ".join(f"{class_name}={val_metrics.get(f'iou/{class_name}', 0.0):.3f}" for class_name in CLASSES))

        current_overall_iou = val_metrics.get("iou/overall", 0.0)
        current_pos_iou = val_metrics.get("iou/pos_only", 0.0)
        is_best = current_overall_iou > best_overall_iou
        is_best_pos = current_pos_iou > best_pos_iou
        if is_best:
            best_overall_iou = current_overall_iou
        if is_best_pos:
            best_pos_iou = current_pos_iou

        save_checkpoint(
            last_pt,
            last_model_only_pt,
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_overall_iou,
            best_pos_iou,
        )
        logger.info(f"已保存 last checkpoint: {last_pt}")
        write_results_csv(results_csv, history)

        if is_best:
            save_checkpoint(
                best_pt,
                best_model_only_pt,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_overall_iou,
                best_pos_iou,
            )
            logger.info(f"★ best.pt 更新: {best_pt}")
            logger.info(f"★ best_model_only.pt 更新: {best_model_only_pt}")
        if is_best_pos:
            save_checkpoint(
                best_pos_pt,
                best_pos_model_only_pt,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_overall_iou,
                best_pos_iou,
            )
            logger.info(f"★ best_pos.pt 更新: {best_pos_pt}")
            logger.info(f"★ best_pos_model_only.pt 更新: {best_pos_model_only_pt}")

    write_results_csv(results_csv, history)

    logger.info(f"训练完成，best.pt={best_pt}")
    logger.info(f"results.csv={results_csv}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
