#!/usr/bin/env python3
"""
SegFormer-B0 学生模型训练脚本 —— 支持B0-hard和B0+D蒸馏
- 默认模式：B0-hard（纯hard labels，无蒸馏）
- 蒸馏模式：B0+D（hard + CLIPSeg soft mask蒸馏）
- 命令行参数控制蒸馏开关和缓存路径
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

from transformers import SegformerForSemanticSegmentation

from config_clipseg import (
    MODEL_NAME, MODEL_DIR, MODEL_TYPE, CLASSES, NUM_CLASSES,
    PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST, IMAGE_ROOT,
    USE_MANIFEST, MANIFEST_JSON,
    IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    BACKBONE_LR, DECODER_HEAD_LR, WEIGHT_DECAY, LRF,
    WARMUP_EPOCHS, WARMUP_START_FACTOR,
    BCE_WEIGHT, DICE_WEIGHT, FOCAL_WEIGHT,
    CLASS_WEIGHTS, FOCAL_GAMMA, FOCAL_ALPHA,
    ENABLE_BOUNDARY_WEAK_SUPERVISION, BOUNDARY_IGNORE_WIDTH, BOUNDARY_IGNORE_MIN_AREA,
    LOSS_WEIGHT_FLOOR,
    HFLIP_PROB, GRAY2RGB,
    BLACKHOT_SKEW_THRESH, BLACKHOT_MEAN_THRESH,
    PSEUDO_COLOR_SAT_THRESH, STD_LOW, STD_MID,
    NOISE_HIGH, NOISE_MED, BLUR_LOW, BLUR_MID,
    IMAGENET_MEAN, IMAGENET_STD, PROJECT, RUN_NAME,
    APPLY_SCORE_FILTER, CONF_FILTER, VAL_THRESHOLDS,
    RARE_OVERSAMPLE, INCLUDE_NEGATIVE_SAMPLES,
    NEGATIVE_SAMPLE_RATIO, NEGATIVE_SAMPLE_WEIGHT,
    TEACHER_SOFT_CACHE, DISTILL_LAMBDA, DISTILL_LAMBDA_RARE,
    DISTILL_START_EPOCH, SKIP_MISSING_TEACHER, RESUME_WEIGHTS_ONLY,
    OLD5_CLASSES, RARE_CLASSES, VAL_INCLUDE_NEGATIVE_SAMPLES,
)

# ── 全局蒸馏配置（通过命令行override） ──
_use_distill = False
_teacher_soft_cache = TEACHER_SOFT_CACHE
_distill_lambda = DISTILL_LAMBDA
_distill_lambda_rare = DISTILL_LAMBDA_RARE
_distill_start_epoch = DISTILL_START_EPOCH
_skip_missing_teacher = SKIP_MISSING_TEACHER
_resume_weights_only = RESUME_WEIGHTS_ONLY
_run_name = RUN_NAME
_total_epochs = EPOCHS
_resume_ckpt_override = None
_auto_resume = True
_reset_history = False


def _device_for_torch_load():
    if isinstance(DEVICE, int):
        return f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"
    return DEVICE

# ── 日志 ──
LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def reconfigure_file_logger(run_name: str) -> None:
    global LOG_FILE
    new_log_file = PROJECT / f"{run_name}.log"
    if LOG_FILE == new_log_file:
        return
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    file_handler = logging.FileHandler(new_log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(file_handler)
    LOG_FILE = new_log_file

# ── 固定随机种子 ──
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════
# 1. 辅助函数（同v4，无改动）
# ══════════════════════════════════════════════════════════════════════

def rle_to_mask(rle):
    try:
        from pycocotools import mask as maskUtils
        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return maskUtils.decode(rle_cp).astype(np.float32)
    except ImportError:
        pass
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes): counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.float32)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = val = 0
    for run_len in counts:
        if val == 1: mask[pos:pos + run_len] = 1
        pos += run_len; val = 1 - val
    return mask.reshape((h, w), order="F").astype(np.float32)


def stable_hash(parts):
    import hashlib
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def is_pseudo_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path):
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


TINY_PROTECT_CLASSES = {"person", "car", "animal", "trash can", "door", "fence", "pole_light", "motorcycle"}
TINY_AREA_THRESHOLDS = {
    "person": 64, "car": 64, "animal": 64, "tree": 64,
    "building": 80, "trash can": 64, "door": 80, "fence": 80,
    "pole_light": 64, "motorcycle": 96, "window": 64,
}


def compute_mask_area_stats(mask: np.ndarray):
    binary = (mask > 0.5).astype(np.uint8)
    total_area = int(binary.sum())
    if total_area <= 0:
        return {"component_count": 0, "max_component_area": 0, "total_area": 0}
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return {"component_count": 0, "max_component_area": total_area, "total_area": total_area}
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "component_count": int(len(areas)),
        "max_component_area": int(areas.max()) if len(areas) else 0,
        "total_area": total_area,
    }


def choose_boundary_ignore_width(mask: np.ndarray, prompt: str) -> int:
    default_width = max(int(BOUNDARY_IGNORE_WIDTH), 0)
    if default_width <= 0:
        return 0
    stats = compute_mask_area_stats(mask)
    total_area = stats["total_area"]
    max_component_area = stats["max_component_area"]
    tiny_thresh = int(TINY_AREA_THRESHOLDS.get(prompt, 64))
    is_tiny = bool(max_component_area < tiny_thresh or total_area < tiny_thresh * 2)
    if prompt in TINY_PROTECT_CLASSES:
        if max_component_area <= tiny_thresh or total_area <= tiny_thresh * 2:
            return 0
        if max_component_area <= tiny_thresh * 2:
            return min(default_width, 1)
        return default_width
    if prompt == "tree":
        if is_tiny:
            return min(default_width, 1)
        return default_width
    return default_width


def build_boundary_valid_mask(mask: np.ndarray, prompt: str) -> np.ndarray:
    if not ENABLE_BOUNDARY_WEAK_SUPERVISION or BOUNDARY_IGNORE_WIDTH <= 0:
        return np.ones_like(mask, dtype=np.float32)
    binary = (mask > 0.5).astype(np.uint8)
    if int(binary.sum()) < BOUNDARY_IGNORE_MIN_AREA:
        return np.ones_like(mask, dtype=np.float32)
    ignore_width = choose_boundary_ignore_width(mask, prompt)
    if ignore_width <= 0:
        return np.ones_like(mask, dtype=np.float32)
    k = ignore_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    ignore_band = (dilated != eroded).astype(np.float32)
    valid = 1.0 - ignore_band
    if float(valid.sum()) < 1.0:
        return np.ones_like(mask, dtype=np.float32)
    return valid.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 2. 改造 Dataset —— 支持加载soft mask
# ══════════════════════════════════════════════════════════════════════

class SegFormerMultiLabelDataset(Dataset):

    def __init__(
        self,
        pred_json,
        image_list_path,
        oversample=None,
        include_negatives=False,
        negative_ratio=0.0,
        negative_weight=0.0,
        apply_score_filter=False,
        use_manifest=False,
        teacher_soft_cache=None,  # 新增：soft mask缓存目录
    ):
        with open(pred_json, encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(image_list_path, encoding="utf-8") as f:
            allowed = {l.strip().replace("\\", "/") for l in f if l.strip()}

        self.samples = []
        self.use_manifest = use_manifest
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.teacher_soft_cache = Path(teacher_soft_cache) if teacher_soft_cache else None

        skipped_empty = skipped_low_conf = skipped_missing = skipped_computer = 0
        negative_kept = 0
        rng = random.Random(SEED)
        manifest_grade_counts = defaultdict(int)

        cache_dir = PROJECT / ".mask_cache" / pred_json.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  解码 RLE → PNG 缓存（{cache_dir}）...")
        if use_manifest:
            records = raw_data.get("records", []) if isinstance(raw_data, dict) else []
            for record in tqdm(records, desc="  解码Manifest"):
                img_path = record["image_path"].replace("\\", "/")
                if img_path not in allowed:
                    alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
                    if alt not in allowed:
                        continue
                    img_path = alt
                abs_img_path = IMAGE_ROOT / img_path
                if not abs_img_path.exists():
                    skipped_missing += 1
                    continue

                prompt = record.get("prompt")
                if prompt == "computer":
                    skipped_computer += 1
                    continue
                if prompt not in CLASSES:
                    continue
                if not record.get("include_in_train"):
                    continue
                if not record.get("selected_hit"):
                    continue
                rle = record.get("rle")
                if rle is None:
                    continue

                grade = str(record.get("grade", ""))
                sample_weight = float(record.get("sample_weight", 1.0))
                ann_id = record.get("ann_id", None)  # 新增：用于查找soft mask

                cache_name = stable_hash([img_path, prompt, grade])
                cache_path = cache_dir / f"{cache_name}.png"
                if not cache_path.exists():
                    mask = rle_to_mask(rle)
                    if mask.sum() == 0:
                        skipped_empty += 1
                        continue
                    cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))

                meta = {"grade": grade, "is_positive": True, "ann_id": ann_id}
                self.samples.append((img_path, prompt, cache_path, sample_weight, meta))
                manifest_grade_counts[grade] += 1
        else:
            preds = raw_data
            for p in tqdm(preds, desc="  解码RLE"):
                img_path = p["image_path"].replace("\\", "/")
                if img_path not in allowed:
                    alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
                    if alt not in allowed:
                        continue
                    img_path = alt
                abs_img_path = IMAGE_ROOT / img_path
                if not abs_img_path.exists():
                    skipped_missing += 1
                    continue
                for prompt, v in p["prompts"].items():
                    if prompt == "computer":
                        skipped_computer += 1
                        continue
                    if prompt not in CLASSES:
                        continue
                    if not v.get("hit"):
                        if include_negatives and negative_ratio > 0 and rng.random() < negative_ratio:
                            meta = {"is_positive": False, "ann_id": None}
                            self.samples.append((img_path, prompt, None, float(negative_weight), meta))
                            negative_kept += 1
                        continue
                    rle = v.get("rle")
                    if rle is None:
                        continue
                    thresh = CONF_FILTER.get(prompt, 0.7) if isinstance(CONF_FILTER, dict) else CONF_FILTER
                    if apply_score_filter and v.get("score", 1.0) < thresh:
                        skipped_low_conf += 1
                        continue
                    cache_name = stable_hash([img_path, prompt, "teacher"])
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() == 0:
                            skipped_empty += 1
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                    ann_id = p.get("ann_id", None)  # 新增
                    meta = {"is_positive": True, "ann_id": ann_id}
                    self.samples.append((img_path, prompt, cache_path, v.get("score", 1.0), meta))

        base = len(self.samples)
        oversampled = defaultdict(int)
        if oversample:
            extra = []
            for s in self.samples:
                if not s[4]["is_positive"]:
                    continue
                mult = oversample.get(s[1], 1)
                for _ in range(mult - 1):
                    extra.append(s)
                    oversampled[s[1]] += 1
            self.samples.extend(extra)
        if oversampled:
            extra_str = ", ".join(f"{c}×{oversample[c]}" for c in oversampled)
            logger.info(f"  {pred_json.name}: {base} → {len(self.samples)} 样本 (过采样: {extra_str})")
        else:
            logger.info(f"  {pred_json.name}: {base} 样本 (无过采样)")
        logger.info(
            f"  负样本保留 {negative_kept}, 跳过空掩码 {skipped_empty}, "
            f"低置信度 {skipped_low_conf}, 缺失图片 {skipped_missing}, 过滤computer {skipped_computer}"
        )

    def __len__(self): return len(self.samples)

    def _safe_class_name(self, class_name: str) -> str:
        """将class_name转换为文件安全名称"""
        return class_name.replace(" ", "_").replace("/", "_")

    def _align_soft_mask_to_training_canvas(self, soft_prob: np.ndarray, orig_hw) -> np.ndarray:
        """将 teacher soft mask 按训练同款规则 resize + pad 到 IMG_SIZE。"""
        orig_h, orig_w = orig_hw
        if soft_prob.shape == (IMG_SIZE, IMG_SIZE):
            return soft_prob.astype(np.float32, copy=False)

        soft_prob = np.asarray(soft_prob, dtype=np.float32)
        scale = IMG_SIZE / max(orig_h, orig_w)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        resized = cv2.resize(soft_prob, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = IMG_SIZE - new_h
        pad_w = IMG_SIZE - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        aligned = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )
        return aligned.astype(np.float32, copy=False)

    def _load_teacher_soft_mask(self, img_path: str, class_name: str, ann_id=None):
        """加载CLIPSeg生成的soft probability map"""
        if not self.teacher_soft_cache or not self.teacher_soft_cache.exists():
            return None

        # 优先用ann_id查找
        if ann_id is not None:
            safe_cls_name = self._safe_class_name(class_name)
            for candidate in [
                self.teacher_soft_cache / f"{ann_id}__{safe_cls_name}.npy",
                self.teacher_soft_cache / f"{ann_id}.npy",
            ]:
                if candidate.exists():
                    try:
                        soft_prob = np.load(candidate)
                        if soft_prob.dtype == np.float16:
                            soft_prob = soft_prob.astype(np.float32)
                        return soft_prob
                    except Exception as e:
                        logger.warning(f"加载soft mask失败 {candidate}: {e}")
                        return None

        # 退化为文件名查找
        safe_img_name = self._safe_class_name(Path(img_path).stem)
        safe_cls_name = self._safe_class_name(class_name)
        soft_path = self.teacher_soft_cache / f"{safe_img_name}__{safe_cls_name}.npy"
        if soft_path.exists():
            try:
                soft_prob = np.load(soft_path)
                if soft_prob.dtype == np.float16:
                    soft_prob = soft_prob.astype(np.float32)
                return soft_prob
            except Exception as e:
                logger.warning(f"加载soft mask失败 {soft_path}: {e}")
                return None

        return None

    def __getitem__(self, idx):
        img_path, class_name, cache_path, score, sample_meta = self.samples[idx]
        img = load_teacher_aligned_gray(IMAGE_ROOT / img_path)

        if sample_meta["is_positive"]:
            mask_img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"掩码缓存不存在或无法读取: {cache_path}")
            mask = mask_img.astype(np.float32) / 255.0
        else:
            mask = np.zeros(img.shape, dtype=np.float32)

        h, w = img.shape
        scale = IMG_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        valid_mask = build_boundary_valid_mask(mask, class_name) if sample_meta["is_positive"] else np.ones_like(mask, dtype=np.float32)

        # 新增：加载teacher soft mask
        teacher_soft_prob = None
        if _use_distill and sample_meta["is_positive"]:
            teacher_soft_prob = self._load_teacher_soft_mask(img_path, class_name, sample_meta.get("ann_id"))
            if teacher_soft_prob is not None:
                teacher_soft_prob = self._align_soft_mask_to_training_canvas(teacher_soft_prob, (h, w))
                if teacher_soft_prob.shape != mask.shape:
                    logger.warning(
                        f"soft mask 对齐后形状异常: {teacher_soft_prob.shape} vs {mask.shape} "
                        f"(img={img_path}, class={class_name})"
                    )
                    teacher_soft_prob = cv2.resize(
                        teacher_soft_prob, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR
                    )

        if GRAY2RGB: img = np.stack([img] * 3, axis=-1)
        else: img = img[:, :, None]

        if random.random() < HFLIP_PROB:
            img = np.fliplr(img); mask = np.fliplr(mask); valid_mask = np.fliplr(valid_mask)
            if teacher_soft_prob is not None:
                teacher_soft_prob = np.fliplr(teacher_soft_prob)

        img = img.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std

        img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        valid_mask = torch.from_numpy(valid_mask.copy()).unsqueeze(0)
        if teacher_soft_prob is not None:
            teacher_soft_prob = torch.from_numpy(teacher_soft_prob.copy()).unsqueeze(0)
        else:
            teacher_soft_prob = torch.zeros_like(mask)

        class_idx = self.class_to_idx[class_name]
        is_positive = sample_meta["is_positive"]
        return img, class_idx, mask, valid_mask, class_name, img_path, score, teacher_soft_prob, is_positive


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    class_indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    masks = torch.stack([b[2] for b in batch], dim=0)
    valid_masks = torch.stack([b[3] for b in batch], dim=0)
    class_names = [b[4] for b in batch]
    img_paths = [b[5] for b in batch]
    weights = torch.tensor([b[6] for b in batch], dtype=torch.float32)
    teacher_soft_probs = torch.stack([b[7] for b in batch], dim=0)
    is_positives = [b[8] for b in batch]
    return imgs, class_indices, masks, valid_masks, class_names, img_paths, weights, teacher_soft_probs, is_positives


# ══════════════════════════════════════════════════════════════════════
# 3. Loss 函数（同v4，无改动）
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target, valid_mask=None):
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        if valid_mask is not None:
            valid_mask = valid_mask.view(valid_mask.size(0), -1)
            pred = pred * valid_mask
            target = target * valid_mask
        intersection = (pred * target).sum(dim=1)
        return 1 - (2. * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)

    def per_sample(self, pred, target, valid_mask=None):
        return self.forward(pred, target, valid_mask=valid_mask)


class RareClassFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weights is not None:
            if isinstance(class_weights, dict):
                self.class_weights = torch.tensor([class_weights.get(c, 1.0) for c in CLASSES], dtype=torch.float32)
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)

    def to(self, device):
        super().to(device)
        self.class_weights = self.class_weights.to(device)
        return self

    def forward(self, logits, targets, class_indices, valid_mask=None):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if valid_mask is not None:
            bce_loss = bce_loss * valid_mask

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        cls_weights = self.class_weights[class_indices].view(-1, 1, 1, 1)
        focal_loss = alpha_t * focal_weight * bce_loss * cls_weights

        if valid_mask is not None:
            area = valid_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
            loss_per_sample = focal_loss.sum(dim=(1, 2, 3)) / area
        else:
            loss_per_sample = focal_loss.view(focal_loss.size(0), -1).mean(dim=1)
        return loss_per_sample


# ══════════════════════════════════════════════════════════════════════
# 4. 指标（改造：支持分组）
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    gt = gt_mask.float()
    pred_sum = pred_bin.sum().item()
    gt_sum = gt.sum().item()
    intersection = (pred_bin * gt).sum().item()
    union = (pred_bin + gt).clamp(0, 1).sum().item()
    if gt_sum == 0 and pred_sum == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}
    tp = intersection
    fp = (pred_bin * (1 - gt)).sum().item()
    fn = ((1 - pred_bin) * gt).sum().item()
    return {
        "iou": intersection / (union + 1e-7),
        "dice": 2 * intersection / (pred_sum + gt_sum + 1e-7),
        "precision": tp / (tp + fp + 1e-7),
        "recall": tp / (tp + fn + 1e-7),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. 模型加载（同v4，无改动）
# ══════════════════════════════════════════════════════════════════════

def load_model():
    if MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        logger.info(f"从本地加载: {MODEL_DIR}")
        model = SegformerForSemanticSegmentation.from_pretrained(str(MODEL_DIR))
    else:
        logger.info(f"从 HuggingFace 加载: {MODEL_NAME}")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    in_channels = model.decode_head.classifier.in_channels
    model.decode_head.classifier = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)

    model = model.to(DEVICE)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": DECODER_HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")
    logger.info(f"  Backbone lr={BACKBONE_LR}, Decoder head lr={DECODER_HEAD_LR}")
    return model, optimizer


# ══════════════════════════════════════════════════════════════════════
# 6. 训练一个 epoch（改造：支持hard+soft loss）
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, dice_loss, focal_loss, scheduler, epoch, total_epochs):
    model.train()
    total_loss = 0
    total_hard_loss = 0
    total_distill_loss = 0
    effective_weight_floor = 0.1 if USE_MANIFEST else LOSS_WEIGHT_FLOOR
    teacher_hit_count = 0
    teacher_missing_count = 0

    pbar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for imgs, class_indices, masks, valid_masks, class_names, _, weights, teacher_soft_probs, is_positives in pbar:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        valid_masks = valid_masks.to(DEVICE)
        weights = weights.to(DEVICE)
        class_indices = class_indices.to(DEVICE)
        teacher_soft_probs = teacher_soft_probs.to(DEVICE)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        # P0修复：正负样本分别处理权重
        for i, is_pos in enumerate(is_positives):
            if is_pos:
                weights[i] = max(weights[i].item(), effective_weight_floor)
            else:
                weights[i] = NEGATIVE_SAMPLE_WEIGHT

        loss_bce_per_sample = []
        loss_dice_per_sample = []
        loss_focal_per_sample = []
        loss_distill_per_sample = []

        for b in range(len(class_indices)):
            cls_idx = class_indices[b].item()
            cls_name = class_names[b]
            logit_b = logits[b:b+1, cls_idx:cls_idx+1, :, :]
            mask_b = masks[b:b+1]
            valid_mask_b = valid_masks[b:b+1]
            teacher_soft_b = teacher_soft_probs[b:b+1]

            # Hard loss
            bce_map = F.binary_cross_entropy_with_logits(logit_b, mask_b, reduction="none")
            valid_area = valid_mask_b.sum().clamp(min=1.0)
            bce_per_sample = (bce_map * valid_mask_b).sum() / valid_area
            loss_bce_per_sample.append(bce_per_sample)

            dice_per_sample = dice_loss.per_sample(logit_b, mask_b, valid_mask=valid_mask_b).mean()
            loss_dice_per_sample.append(dice_per_sample)

            focal_per_sample = focal_loss(logit_b, mask_b, class_indices[b:b+1], valid_mask=valid_mask_b).mean()
            loss_focal_per_sample.append(focal_per_sample)

            # Distill loss（新增）
            distill_per_sample = torch.tensor(0.0, device=DEVICE)
            if _use_distill and epoch >= _distill_start_epoch:
                has_teacher = bool((teacher_soft_b.max() > 0).item() or (teacher_soft_b.sum().abs() > 1e-6).item())
                if has_teacher:
                    teacher_hit_count += 1
                    # teacher_soft_b是概率 [0, 1]，需要转换为logits进行BCEWithLogits
                    # 或者直接用BCE：bce(sigmoid(logit), teacher_prob)
                    # 这里用BCEWithLogits更稳定
                    distill_loss_map = F.binary_cross_entropy_with_logits(logit_b, teacher_soft_b, reduction="none")
                    if valid_mask_b.sum() > 0:
                        distill_per_sample = (distill_loss_map * valid_mask_b).sum() / valid_area
                    else:
                        distill_per_sample = distill_loss_map.mean()
                else:
                    teacher_missing_count += 1
            loss_distill_per_sample.append(distill_per_sample)

        loss_bce_per_sample = torch.stack(loss_bce_per_sample)
        loss_dice_per_sample = torch.stack(loss_dice_per_sample)
        loss_focal_per_sample = torch.stack(loss_focal_per_sample).squeeze()
        loss_distill_per_sample = torch.stack(loss_distill_per_sample)

        hard_loss_per_sample = BCE_WEIGHT * loss_bce_per_sample + DICE_WEIGHT * loss_dice_per_sample + FOCAL_WEIGHT * loss_focal_per_sample
        loss_hard = (hard_loss_per_sample * weights).mean()

        # 蒸馏loss加权
        if _use_distill and epoch >= _distill_start_epoch:
            distill_lambdas = []
            for cls_name in class_names:
                if cls_name in RARE_CLASSES:
                    distill_lambdas.append(_distill_lambda_rare)
                else:
                    distill_lambdas.append(_distill_lambda)
            distill_lambdas = torch.tensor(distill_lambdas, device=DEVICE)
            loss_distill = (loss_distill_per_sample * weights * distill_lambdas).mean()
        else:
            loss_distill = torch.tensor(0.0, device=DEVICE)

        loss = loss_hard + loss_distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_hard_loss += loss_hard.item()
        total_distill_loss += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else loss_distill

        loss_str = f"{loss.item():.4f}"
        if _use_distill and epoch >= _distill_start_epoch:
            loss_str += f" (H:{loss_hard.item():.4f} D:{loss_distill.item():.4f})"
        pbar.set_postfix(loss=loss_str)

    teacher_stats = {
        "teacher_hit": teacher_hit_count,
        "teacher_missing": teacher_missing_count,
    }
    return total_loss / len(loader), total_hard_loss / len(loader), total_distill_loss / len(loader), teacher_stats


# ══════════════════════════════════════════════════════════════════════
# 7. 验证（改造：支持分组mIoU）
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_metrics = defaultdict(list)
    vis_samples = []

    for imgs, class_indices, masks, valid_masks, class_names, _, _, _, is_positives in tqdm(loader, desc="  Val"):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        class_indices = class_indices.to(DEVICE)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        for b in range(len(class_indices)):
            cls_idx = class_indices[b].item()
            cls_name = class_names[b]
            logit_b = logits[b:b+1, cls_idx:cls_idx+1, :, :]
            mask_b = masks[b:b+1]

            pred_threshold = VAL_THRESHOLDS.get(cls_name, 0.5) if isinstance(VAL_THRESHOLDS, dict) else 0.5
            m = compute_metrics(logit_b, mask_b, threshold=pred_threshold)

            for k, v in m.items():
                all_metrics[f"{k}/{cls_name}"].append(v)
            all_metrics["iou/overall"].append(m["iou"])
            all_metrics["dice/overall"].append(m["dice"])

            # 分组统计
            if cls_name in OLD5_CLASSES:
                all_metrics["iou/old5"].append(m["iou"])
            if cls_name in RARE_CLASSES:
                all_metrics["iou/rare"].append(m["iou"])

            if len(vis_samples) < 8:
                vis_samples.append({
                    "image": imgs[b].cpu(), "mask_gt": mask_b[0, 0].cpu(),
                    "mask_pred": (torch.sigmoid(logit_b[0, 0]) > pred_threshold).float().cpu(),
                    "class_name": cls_name,
                })

    summary = {k: np.mean(v) for k, v in all_metrics.items()}
    return summary, vis_samples


# ══════════════════════════════════════════════════════════════════════
# 8. 绘图（同v4）
# ══════════════════════════════════════════════════════════════════════

def plot_results(history, vis_samples, save_path, epoch):
    fig = plt.figure(figsize=(16, 8))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs_range, history["train_loss"], "b-", linewidth=1, label="Total")
    if _use_distill and "train_hard_loss" in history:
        ax1.plot(epochs_range, history["train_hard_loss"], "g--", linewidth=1, alpha=0.7, label="Hard")
        ax1.plot(epochs_range, history["train_distill_loss"], "r--", linewidth=1, alpha=0.7, label="Distill")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("1. Training Loss")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs_range, history["val_miou_all11"], "g-", linewidth=1.5, label="All11")
    if "val_miou_old5" in history:
        ax2.plot(epochs_range, history["val_miou_old5"], "b--", linewidth=1.5, label="Old5")
    if "val_miou_rare" in history:
        ax2.plot(epochs_range, history["val_miou_rare"], "r--", linewidth=1.5, label="Rare")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU"); ax2.set_title("2. Val mIoU (Grouped)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs_range, history["val_dice"], "m-", linewidth=1.5)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Dice"); ax3.set_title("3. Val mDice")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    for cls_name in CLASSES:
        k = f"iou/{cls_name}"
        if k in history and history[k]:
            ax4.plot(epochs_range, history[k], "-", label=cls_name, linewidth=1, alpha=0.8)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("IoU"); ax4.set_title("4. Per-Class IoU")
    ax4.legend(fontsize=5); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    if history.get("lr"):
        ax5.plot(epochs_range, history["lr"], "r-", linewidth=1)
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("LR"); ax5.set_title("5. Learning Rate")
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax6 = fig.add_subplot(2, 3, 6)
    if vis_samples:
        s = vis_samples[0]
        img_np = s["image"].permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 3: img_np = img_np[:, :, 0]
        h, w = img_np.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3): overlay[:, :, c] = np.clip(img_np, 0, 1)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + s["mask_gt"].numpy() * 0.4, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + s["mask_pred"].numpy() * 0.5, 0, 1)
        ax6.imshow(overlay)
        ax6.set_title(f"6. {s['class_name']}\nRed=GT Cyan=Pred", fontsize=8)
        ax6.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 9. 改进的 checkpoint 加载（支持从hard继续finetune）
# ══════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path, model, optimizer=None):
    """加载checkpoint，支持多种格式"""
    logger.info(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=_device_for_torch_load(), weights_only=False)

    # 兼容多种模型权重key
    if "model" in ckpt:
        model_state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        model_state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        model_state = ckpt["state_dict"]
    else:
        model_state = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    start_epoch = ckpt.get("epoch", 0)
    best_metrics = {
        "best_all11": ckpt.get("best_miou", -1),
        "best_old5": ckpt.get("best_miou_old5", -1),
        "best_rare": ckpt.get("best_miou_rare", -1),
    }
    history = defaultdict(list)
    for k, v in ckpt.get("history", {}).items():
        history[k] = list(v)

    if optimizer is not None and not _resume_weights_only and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("已恢复optimizer状态")

    return start_epoch, best_metrics, history


# ══════════════════════════════════════════════════════════════════════
# 10. 主训练（改造：支持蒸馏 + 多个best checkpoint）
# ══════════════════════════════════════════════════════════════════════

def train():
    run_dir = PROJECT / _run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_all11_pt = run_dir / "best_all11.pt"
    best_old5_pt = run_dir / "best_old5.pt"
    best_rare_pt = run_dir / "best_rare.pt"
    last_pt = run_dir / "last.pt"
    results_png = run_dir / "results.png"

    model, optimizer = load_model()

    train_ds = SegFormerMultiLabelDataset(
        MANIFEST_JSON if USE_MANIFEST else PRED_JSON,
        TRAIN_LIST,
        oversample=RARE_OVERSAMPLE,
        include_negatives=(False if USE_MANIFEST else INCLUDE_NEGATIVE_SAMPLES),
        negative_ratio=NEGATIVE_SAMPLE_RATIO,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=APPLY_SCORE_FILTER,
        use_manifest=USE_MANIFEST,
        teacher_soft_cache=_teacher_soft_cache if _use_distill else None,
    )
    val_ds = SegFormerMultiLabelDataset(
        VAL_PRED_JSON,
        VAL_LIST,
        oversample=None,
        include_negatives=VAL_INCLUDE_NEGATIVE_SAMPLES,
        negative_ratio=1.0 if VAL_INCLUDE_NEGATIVE_SAMPLES else 0.0,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=APPLY_SCORE_FILTER,
        use_manifest=False,
        teacher_soft_cache=None,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
                            collate_fn=collate_fn, pin_memory=True)
    logger.info(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    dice_loss = DiceLoss()
    focal_loss = RareClassFocalLoss(class_weights=CLASS_WEIGHTS, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    focal_loss = focal_loss.to(DEVICE)

    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    total_steps = _total_epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return WARMUP_START_FACTOR + (1 - WARMUP_START_FACTOR) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LRF + (1 - LRF) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    history = defaultdict(list)
    best_metrics = {"best_all11": -1, "best_old5": -1, "best_rare": -1}

    resume_ckpt = None
    if _resume_ckpt_override is not None:
        resume_ckpt = Path(_resume_ckpt_override)
    elif _auto_resume:
        if last_pt.exists():
            resume_ckpt = last_pt
        elif best_all11_pt.exists():
            resume_ckpt = best_all11_pt

    if resume_ckpt and resume_ckpt.exists():
        start_epoch, best_metrics, history = load_checkpoint(resume_ckpt, model, optimizer)
        if _reset_history:
            logger.info("  重置 history / best metrics / epoch，只保留模型初始化权重")
            start_epoch = 0
            history = defaultdict(list)
            best_metrics = {"best_all11": -1, "best_old5": -1, "best_rare": -1}
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif _resume_weights_only:
            logger.info("  仅加载模型权重，重新初始化optimizer/scheduler")
            start_epoch = 0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            if start_epoch >= _total_epochs:
                logger.info(f"已完成 ({start_epoch}/{_total_epochs})，跳过"); return best_all11_pt
            logger.info(f"从 epoch {start_epoch + 1}/{_total_epochs} 续训, best_all11={best_metrics['best_all11']:.4f}")
    elif _resume_ckpt_override is not None:
        raise FileNotFoundError(f"--resume 指定的 checkpoint 不存在: {_resume_ckpt_override}")

    for epoch in range(start_epoch + 1, _total_epochs + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{_total_epochs}")
        t0 = time.time()

        train_loss, train_hard_loss, train_distill_loss, teacher_stats = train_epoch(
            model, train_loader, optimizer, dice_loss, focal_loss, scheduler, epoch, _total_epochs
        )
        val_metrics, vis_samples = validate(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        if _use_distill:
            history["train_hard_loss"].append(train_hard_loss)
            history["train_distill_loss"].append(train_distill_loss)
        history["val_miou_all11"].append(val_metrics.get("iou/overall", 0))
        history["val_miou_old5"].append(val_metrics.get("iou/old5", 0))
        history["val_miou_rare"].append(val_metrics.get("iou/rare", 0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0))
        history["lr"].append(current_lr)
        for c in CLASSES:
            history[f"iou/{c}"].append(val_metrics.get(f"iou/{c}", 0))

        elapsed = time.time() - t0
        logger.info(
            f"  Loss={train_loss:.4f}  mIoU_all11={val_metrics.get('iou/overall',0):.4f}  "
            f"mIoU_old5={val_metrics.get('iou/old5',0):.4f}  "
            f"mIoU_rare={val_metrics.get('iou/rare',0):.4f}  mDice={val_metrics.get('dice/overall',0):.4f}  "
            f"LR={current_lr:.2e}  {elapsed/60:.1f}min"
        )
        if _use_distill and epoch >= _distill_start_epoch:
            teacher_total = teacher_stats["teacher_hit"] + teacher_stats["teacher_missing"]
            teacher_hit_rate = (teacher_stats["teacher_hit"] / teacher_total) if teacher_total > 0 else 0.0
            logger.info(
                f"  Teacher cache hit: {teacher_stats['teacher_hit']}  "
                f"missing: {teacher_stats['teacher_missing']}  "
                f"hit_rate={teacher_hit_rate:.2%}"
            )
        cls_str = "  ".join(f"{c}={val_metrics.get(f'iou/{c}', 0):.3f}" for c in CLASSES)
        logger.info(f"  Per-class IoU: {cls_str}")

        plot_results(history, vis_samples, results_png, epoch)

        # 分别保存三个best版本
        miou_all11 = val_metrics.get("iou/overall", 0)
        improved_all11 = False
        if miou_all11 > best_metrics["best_all11"]:
            best_metrics["best_all11"] = miou_all11
            improved_all11 = True

        miou_old5 = val_metrics.get("iou/old5", 0)
        improved_old5 = False
        if miou_old5 > best_metrics["best_old5"]:
            best_metrics["best_old5"] = miou_old5
            improved_old5 = True

        miou_rare = val_metrics.get("iou/rare", 0)
        improved_rare = False
        if miou_rare > best_metrics["best_rare"]:
            best_metrics["best_rare"] = miou_rare
            improved_rare = True

        save_dict = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": dict(history),
            "run_name": _run_name,
            "classes": CLASSES,
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "val_thresholds": VAL_THRESHOLDS,
            "model_type": MODEL_TYPE,
            "best_miou": best_metrics["best_all11"],
            "best_miou_old5": best_metrics["best_old5"],
            "best_miou_rare": best_metrics["best_rare"],
        }
        torch.save(save_dict, last_pt)

        if improved_all11:
            torch.save(save_dict, best_all11_pt)
            logger.info(f"  ★ 新best_all11 (mIoU={best_metrics['best_all11']:.4f})")
        if improved_old5:
            torch.save(save_dict, best_old5_pt)
            logger.info(f"  ★ 新best_old5 (mIoU={best_metrics['best_old5']:.4f})")
        if improved_rare:
            torch.save(save_dict, best_rare_pt)
            logger.info(f"  ★ 新best_rare (mIoU={best_metrics['best_rare']:.4f})")

    csv_path = run_dir / "results.csv"
    keys = ["train_loss", "val_miou_all11", "val_miou_old5", "val_miou_rare", "val_dice", "lr"] + [f"iou/{c}" for c in CLASSES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] if i < len(history[k]) else "" for k in keys])

    logger.info(f"完成, best_all11={best_metrics['best_all11']:.4f}, best_old5={best_metrics['best_old5']:.4f}, best_rare={best_metrics['best_rare']:.4f}")
    return best_all11_pt


# ══════════════════════════════════════════════════════════════════════
# main + 命令行参数
# ══════════════════════════════════════════════════════════════════════

def main():
    global _use_distill, _teacher_soft_cache, _distill_lambda, _distill_lambda_rare
    global _distill_start_epoch, _skip_missing_teacher, _resume_weights_only
    global _run_name, _total_epochs, _resume_ckpt_override, _auto_resume, _reset_history

    parser = argparse.ArgumentParser(description="SegFormer-B0学生模型训练 (B0-hard或B0+D蒸馏)")
    parser.add_argument("--use_distill", action="store_true", help="启用CLIPSeg蒸馏")
    parser.add_argument("--teacher_soft_cache", type=str, default=None, help="soft mask缓存目录")
    parser.add_argument("--distill_lambda", type=float, default=None, help="普通类蒸馏权重")
    parser.add_argument("--distill_lambda_rare", type=float, default=None, help="稀有类蒸馏权重")
    parser.add_argument("--distill_start_epoch", type=int, default=0, help="从第N个epoch开始蒸馏")
    parser.add_argument("--skip_missing_teacher", action="store_true", default=True, help="缺失soft mask时跳过蒸馏")
    parser.add_argument("--resume_weights_only", action="store_true", default=RESUME_WEIGHTS_ONLY, help="从checkpoint只加载权重")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径（优先级最高）")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--run_name", type=str, default=None, help="运行名称（影响输出目录）")
    parser.add_argument("--no_auto_resume", action="store_true", help="禁用自动查找checkpoint")
    parser.add_argument("--reset_history", action="store_true", help="重置训练历史")

    args = parser.parse_args()

    _use_distill = args.use_distill
    if args.teacher_soft_cache:
        _teacher_soft_cache = Path(args.teacher_soft_cache)
    if args.distill_lambda is not None:
        _distill_lambda = args.distill_lambda
    if args.distill_lambda_rare is not None:
        _distill_lambda_rare = args.distill_lambda_rare
    _distill_start_epoch = args.distill_start_epoch
    _skip_missing_teacher = args.skip_missing_teacher
    _resume_weights_only = args.resume_weights_only
    _resume_ckpt_override = Path(args.resume) if args.resume else None
    _total_epochs = int(args.epochs) if args.epochs is not None else EPOCHS
    _run_name = args.run_name if args.run_name else RUN_NAME
    _auto_resume = not args.no_auto_resume
    _reset_history = args.reset_history
    reconfigure_file_logger(_run_name)

    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)
    mode_str = "B0+D (Hard+Distill)" if _use_distill else "B0-Hard"
    logger.info(f"SegFormer-B0 训练模式: {mode_str}")
    logger.info(f"  运行名: {_run_name}")
    logger.info(f"  模型: {MODEL_NAME}  分辨率: {IMG_SIZE}  Batch: {BATCH}  Epochs: {_total_epochs}")
    logger.info(f"  类别数: {NUM_CLASSES}  类别: {CLASSES}")
    logger.info(f"  Backbone lr: {BACKBONE_LR}  Decoder head lr: {DECODER_HEAD_LR}")
    logger.info(f"  Loss: {BCE_WEIGHT}*BCE + {DICE_WEIGHT}*Dice + {FOCAL_WEIGHT}*RareClassFocal")
    logger.info(f"  Focal gamma: {FOCAL_GAMMA}  alpha: {FOCAL_ALPHA}")
    logger.info(f"  Val include negatives: {VAL_INCLUDE_NEGATIVE_SAMPLES}")
    logger.info(f"  Auto resume: {_auto_resume}  Resume override: {_resume_ckpt_override}")
    logger.info(f"  Resume weights only: {_resume_weights_only}  Reset history: {_reset_history}")

    if _use_distill:
        logger.info(f"  [蒸馏配置]")
        logger.info(f"    Teacher soft cache: {_teacher_soft_cache}")
        logger.info(f"    Distill lambda: {_distill_lambda}  (rare: {_distill_lambda_rare})")
        logger.info(f"    Distill start epoch: {_distill_start_epoch}")
        logger.info(f"    Skip missing teacher: {_skip_missing_teacher}")
        logger.info(f"    Resume weights only: {_resume_weights_only}")
    logger.info("=" * 60)

    ckpt = train()
    logger.info(f"最终模型: {ckpt}")
    logger.info(f"日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
