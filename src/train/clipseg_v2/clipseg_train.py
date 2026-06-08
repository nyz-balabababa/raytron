#!/usr/bin/env python3
"""
CLIPSeg 训练脚本 —— SAM3 蒸馏实验第一个模型
- 图像 + 文本 prompt → 单通道分割 mask
- 动态 prompt 增强 + 水平翻转 + warmup + 余弦退火
- 稀有类过采样 + 分级置信度过滤
- 断点续训 + 每 epoch 验证 + 实时 results.png（6 指标）
"""
import csv
import json
import logging
import math
import os
import random
import shutil
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

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import config_clipseg as cfg

from config_clipseg import (
    MODEL_NAME, MODEL_DIR, CLASSES,
    PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST, IMAGE_ROOT,
    USE_MANIFEST, MANIFEST_JSON,
    IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    DECODER_LR, BACKBONE_LR, WEIGHT_DECAY, LRF,
    WARMUP_EPOCHS, WARMUP_START_FACTOR,
    BCE_WEIGHT, DICE_WEIGHT, LOSS_WEIGHT_FLOOR,
    ENABLE_BOUNDARY_WEAK_SUPERVISION, BOUNDARY_IGNORE_WIDTH, BOUNDARY_IGNORE_MIN_AREA,
    PROMPT_AUG_PROB, PROMPT_AUGMENTATIONS,
    HFLIP_PROB, GRAY2RGB,
    BLACKHOT_SKEW_THRESH, BLACKHOT_MEAN_THRESH,
    PSEUDO_COLOR_SAT_THRESH, STD_LOW, STD_MID,
    NOISE_HIGH, NOISE_MED, BLUR_LOW, BLUR_MID,
    CLIP_MEAN, CLIP_STD, PROJECT, RUN_NAME,
    APPLY_SCORE_FILTER, CONF_FILTER, VAL_THRESHOLDS,
    RARE_OVERSAMPLE, INCLUDE_NEGATIVE_SAMPLES,
    NEGATIVE_SAMPLE_RATIO, NEGATIVE_SAMPLE_WEIGHT,
)

INIT_CHECKPOINT = getattr(cfg, "INIT_CHECKPOINT", None)
MANIFEST_LOSS_WEIGHT_FLOOR = float(getattr(cfg, "MANIFEST_LOSS_WEIGHT_FLOOR", 0.1))
MANIFEST_NEGATIVE_RATIO = float(getattr(cfg, "MANIFEST_NEGATIVE_RATIO", 0.25))
MANIFEST_CACHE_TAG = str(getattr(cfg, "MANIFEST_CACHE_TAG", "train_manifest_v1"))
FORCE_REBUILD_MASK_CACHE = bool(getattr(cfg, "FORCE_REBUILD_MASK_CACHE", False))
VAL_APPLY_SCORE_FILTER = bool(getattr(cfg, "VAL_APPLY_SCORE_FILTER", True))

# ── 日志 ──
LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 固定随机种子 ──
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def resolve_runtime_device(device_cfg):
    if isinstance(device_cfg, torch.device):
        return device_cfg
    if isinstance(device_cfg, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_cfg}")
        return torch.device("cpu")
    if isinstance(device_cfg, str):
        if device_cfg.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


RUNTIME_DEVICE = resolve_runtime_device(DEVICE)
RUNTIME_DEVICE_STR = str(RUNTIME_DEVICE)
GPU_INDEX = RUNTIME_DEVICE.index if RUNTIME_DEVICE.type == "cuda" and RUNTIME_DEVICE.index is not None else 0
TQDM_NCOLS = 88
TRAIN_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
VAL_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


# ══════════════════════════════════════════════════════════════════════
# 1. RLE 解码
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


TINY_PROTECT_CLASSES = {"person", "car", "animal"}
TINY_AREA_THRESHOLDS = {
    "person": 64,
    "car": 64,
    "animal": 64,
    "tree": 64,
    "building": 80,
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

    # person / car / animal: tiny 时优先保留正样本监督，不靠大边界带弱化
    if prompt in TINY_PROTECT_CLASSES:
        if max_component_area <= tiny_thresh or total_area <= tiny_thresh * 2:
            return 0
        if max_component_area <= tiny_thresh * 2:
            return min(default_width, 1)
        return default_width

    # tree: 中性处理，小目标只轻微缩小 ignore band
    if prompt == "tree":
        if is_tiny:
            return min(default_width, 1)
        return default_width

    # building: 不靠 ignore band 保护 tiny，是否训练交给 manifest/policy
    return default_width


def build_boundary_valid_mask(mask: np.ndarray, prompt: str) -> np.ndarray:
    """
    对伪标签边界做 ignore band。
    valid=1 的区域参与监督，valid=0 的边界环带不参与损失。
    """
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
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════

class CLIPSegDataset(Dataset):

    def __init__(
        self,
        pred_json,
        image_list_path,
        augment_prompt=False,
        oversample=None,
        include_negatives=False,
        negative_ratio=0.0,
        negative_weight=0.0,
        apply_score_filter=False,
        use_manifest=False,
    ):
        with open(pred_json, encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(image_list_path, encoding="utf-8") as f:
            allowed = {l.strip().replace("\\", "/") for l in f if l.strip()}

        self.samples = []
        self.augment_prompt = augment_prompt
        self.use_manifest = use_manifest
        skipped_empty = skipped_low_conf = skipped_missing = 0
        negative_kept = 0
        rng = random.Random(SEED)
        manifest_grade_counts = defaultdict(int)
        manifest_source_counts = defaultdict(int)
        manifest_tiny_counts = defaultdict(int)
        manifest_positive_count = 0
        manifest_negative_raw_count = 0
        manifest_negative_kept_count = 0
        manifest_weight_values = []
        manifest_positive_samples = []
        manifest_negative_samples = []

        # 磁盘缓存目录：RLE 解码为 PNG 存盘，避免每 epoch 重复解码
        cache_dir = PROJECT / ".mask_cache" / (MANIFEST_CACHE_TAG if use_manifest else pred_json.stem)
        if use_manifest and FORCE_REBUILD_MASK_CACHE and cache_dir.exists():
            logger.info(f"  强制重建 manifest cache: {cache_dir}")
            shutil.rmtree(cache_dir)
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
                if prompt not in CLASSES:
                    continue
                if not record.get("include_in_train"):
                    continue

                grade = str(record.get("grade", ""))
                mask_source = str(record.get("mask_source", "manifest"))
                is_tiny = bool(record.get("is_tiny", False))
                selected_hit = bool(record.get("selected_hit"))
                rle = record.get("rle")
                is_positive = selected_hit and rle is not None
                default_weight = 1.0 if is_positive else negative_weight
                sample_weight = float(record.get("sample_weight", default_weight))
                if sample_weight <= 0:
                    continue
                cache_path = None

                if is_positive:
                    rle_size = rle.get("size") if isinstance(rle, dict) else None
                    rle_counts = rle.get("counts") if isinstance(rle, dict) else None
                    cache_name = stable_hash([
                        img_path,
                        prompt,
                        grade,
                        mask_source,
                        selected_hit,
                        rle_size,
                        stable_hash([rle_counts]),
                    ])
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() == 0:
                            skipped_empty += 1
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))

                meta = {
                    "mode": "manifest",
                    "grade": grade,
                    "mask_source": mask_source,
                    "is_tiny": is_tiny,
                    "flags": list(record.get("flags", [])),
                    "disabled_reason": record.get("disabled_reason"),
                    "component_count": int(record.get("component_count", 0)),
                    "max_component_area": int(record.get("max_component_area", 0)),
                    "total_mask_area": int(record.get("total_mask_area", 0)),
                    "sample_weight": sample_weight,
                    "is_positive": is_positive,
                }
                sample_tuple = (img_path, prompt, cache_path, sample_weight, is_positive, meta)
                if is_positive:
                    manifest_positive_count += 1
                    manifest_positive_samples.append(sample_tuple)
                else:
                    manifest_negative_raw_count += 1
                    manifest_negative_samples.append(sample_tuple)
                manifest_grade_counts[grade] += 1
                manifest_source_counts[mask_source] += 1
                if is_tiny:
                    manifest_tiny_counts[prompt] += 1
            max_negatives = int(len(manifest_positive_samples) * MANIFEST_NEGATIVE_RATIO)
            if max_negatives <= 0:
                kept_negative_samples = []
            elif len(manifest_negative_samples) > max_negatives:
                kept_negative_samples = rng.sample(manifest_negative_samples, max_negatives)
            else:
                kept_negative_samples = manifest_negative_samples
            manifest_negative_kept_count = len(kept_negative_samples)
            self.samples.extend(manifest_positive_samples)
            self.samples.extend(kept_negative_samples)
            manifest_weight_values = [float(s[3]) for s in self.samples]
        else:
            preds = raw_data
            for p in tqdm(preds, desc="  解码RLE"):
                img_path = p["image_path"].replace("\\", "/")
                if img_path not in allowed:
                    alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
                    if alt not in allowed: continue
                    img_path = alt
                abs_img_path = IMAGE_ROOT / img_path
                if not abs_img_path.exists():
                    skipped_missing += 1
                    continue
                for prompt, v in p["prompts"].items():
                    if prompt not in CLASSES:
                        continue
                    if not v.get("hit"):
                        if include_negatives and negative_ratio > 0 and rng.random() < negative_ratio:
                            meta = {
                                "mode": "pred_json",
                                "grade": None,
                                "mask_source": "stable_negative",
                                "is_tiny": False,
                                "flags": [],
                                "disabled_reason": None,
                                "component_count": 0,
                                "max_component_area": 0,
                                "total_mask_area": 0,
                            }
                            self.samples.append((img_path, prompt, None, float(negative_weight), False, meta))
                            negative_kept += 1
                        continue
                    rle = v.get("rle")
                    if rle is None:
                        continue
                    thresh = CONF_FILTER.get(prompt, 0.7) if isinstance(CONF_FILTER, dict) else CONF_FILTER
                    if apply_score_filter and v.get("score", 1.0) < thresh:
                        skipped_low_conf += 1
                        continue
                    cache_name = stable_hash([img_path, prompt, "pred_json", "teacher"])
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() == 0:
                            skipped_empty += 1
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                    meta = {
                        "mode": "pred_json",
                        "grade": None,
                        "mask_source": "teacher",
                        "is_tiny": False,
                        "flags": [],
                        "disabled_reason": None,
                        "component_count": 0,
                        "max_component_area": 0,
                        "total_mask_area": 0,
                    }
                    self.samples.append((img_path, prompt, cache_path, v.get("score", 1.0), True, meta))

        base = len(self.samples)
        oversampled = defaultdict(int)
        if oversample:
            extra = []
            for s in self.samples:
                if not s[4]:
                    continue
                mult = oversample.get(s[1], 1)
                for _ in range(mult - 1):
                    extra.append(s)
                    oversampled[s[1]] += 1
            self.samples.extend(extra)
        dataset_name = MANIFEST_CACHE_TAG if use_manifest else pred_json.name
        if oversampled:
            extra_str = ", ".join(f"{c}×{oversample[c]}" for c in oversampled)
            logger.info(f"  {dataset_name}: {base} → {len(self.samples)} 样本 (过采样: {extra_str})")
        else:
            logger.info(f"  {dataset_name}: {base} 样本 (无过采样)")
        logger.info(
            f"  负样本保留 {negative_kept}, 跳过空掩码 {skipped_empty}, "
            f"低置信度 {skipped_low_conf}, 缺失图片 {skipped_missing}"
        )
        if use_manifest:
            logger.info(f"  Manifest positives: {manifest_positive_count}")
            logger.info(f"  Manifest negatives raw: {manifest_negative_raw_count}")
            logger.info(f"  Manifest negatives kept: {manifest_negative_kept_count}")
            logger.info(f"  Manifest negative ratio: {MANIFEST_NEGATIVE_RATIO}")
            logger.info(f"  Manifest grade 分布: {dict(sorted(manifest_grade_counts.items()))}")
            logger.info(f"  Manifest mask_source 分布: {dict(sorted(manifest_source_counts.items()))}")
            logger.info(f"  Manifest tiny 分布: {dict(sorted(manifest_tiny_counts.items()))}")
            if manifest_weight_values:
                logger.info(
                    "  Manifest sample_weight: "
                    f"min={min(manifest_weight_values):.3f} "
                    f"mean={float(np.mean(manifest_weight_values)):.3f} "
                    f"max={max(manifest_weight_values):.3f}"
                )
            logger.info(f"  Manifest 训练样本总数: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def _augment(self, prompt):
        opts = PROMPT_AUGMENTATIONS.get(prompt, [])
        return random.choice(opts) if opts else prompt

    def __getitem__(self, idx):
        img_path, orig_prompt, cache_path, score, is_positive, sample_meta = self.samples[idx]
        prompt = self._augment(orig_prompt) if (self.augment_prompt and random.random() < PROMPT_AUG_PROB) else orig_prompt

        img = load_teacher_aligned_gray(IMAGE_ROOT / img_path)

        # 从 PNG 加载预解码的 mask（cv2.imread 是 C 实现，远快于 RLE 解码）
        if is_positive:
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

        # pad 正方形
        ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        valid_mask = build_boundary_valid_mask(mask, orig_prompt) if is_positive else np.ones_like(mask, dtype=np.float32)

        if GRAY2RGB: img = np.stack([img] * 3, axis=-1)
        else: img = img[:, :, None]

        # 水平翻转
        if random.random() < HFLIP_PROB:
            img = np.fliplr(img); mask = np.fliplr(mask); valid_mask = np.fliplr(valid_mask)

        img = img.astype(np.float32) / 255.0

        # CLIP 标准化（与预训练时一致）
        mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(CLIP_STD,  dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std

        img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        valid_mask = torch.from_numpy(valid_mask.copy()).unsqueeze(0)
        return img, prompt, mask, valid_mask, orig_prompt, img_path, score, sample_meta


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    prompts = [b[1] for b in batch]
    masks = torch.stack([b[2] for b in batch], dim=0)
    valid_masks = torch.stack([b[3] for b in batch], dim=0)
    orig_prompts = [b[4] for b in batch]
    img_paths = [b[5] for b in batch]
    weights = torch.tensor([b[6] for b in batch], dtype=torch.float32)
    sample_meta = [b[7] for b in batch]
    return imgs, prompts, masks, valid_masks, orig_prompts, img_paths, weights, sample_meta


def ensure_channel_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


# ══════════════════════════════════════════════════════════════════════
# 3. Loss
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
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
        """返回逐样本的 Dice loss，不取均值。"""
        return self.forward(pred, target, valid_mask=valid_mask)


# ══════════════════════════════════════════════════════════════════════
# 4. 指标
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
# 5. 模型
# ══════════════════════════════════════════════════════════════════════

def _extract_model_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj:
            return ckpt_obj["model"]
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_model(init_checkpoint=None):
    if MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        logger.info(f"从本地加载: {MODEL_DIR}")
        processor = CLIPSegProcessor.from_pretrained(str(MODEL_DIR))
        model = CLIPSegForImageSegmentation.from_pretrained(str(MODEL_DIR))
    else:
        logger.info(f"从 HuggingFace 加载: {MODEL_NAME}")
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)

    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint)
        if init_checkpoint.exists():
            ckpt = torch.load(init_checkpoint, map_location=RUNTIME_DEVICE, weights_only=False)
            state_dict = _extract_model_state_dict(ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"从 INIT_CHECKPOINT 初始化模型参数: {init_checkpoint}")
            if missing:
                logger.info(f"  缺失参数: {len(missing)}")
            if unexpected:
                logger.info(f"  额外参数: {len(unexpected)}")
        else:
            logger.warning(f"INIT_CHECKPOINT 不存在，跳过初始化: {init_checkpoint}")

    model = model.to(RUNTIME_DEVICE)
    decoder_params, vision_params = [], []
    for name, param in model.named_parameters():
        if "clip.visual" in name: vision_params.append(param)
        elif "clip.text" in name or "clip.logit_scale" in name: param.requires_grad = False
        else: decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": DECODER_LR},
        {"params": vision_params, "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")
    logger.info(f"  解码器 lr={DECODER_LR}, 视觉编码器 lr={BACKBONE_LR}, 文本编码器: 冻结")
    return model, processor, optimizer


# ══════════════════════════════════════════════════════════════════════
# 6. 训练一个 epoch
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, processor, loader, optimizer, dice_loss, scheduler, epoch, total_epochs):
    model.train()
    total_loss = 0
    effective_weight_floor = MANIFEST_LOSS_WEIGHT_FLOOR if USE_MANIFEST else LOSS_WEIGHT_FLOOR
    pbar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}",
                bar_format=TRAIN_BAR_FORMAT, ncols=TQDM_NCOLS, leave=False)
    for imgs, prompts, masks, valid_masks, _, _, weights, _ in pbar:
        imgs = imgs.to(RUNTIME_DEVICE); masks = masks.to(RUNTIME_DEVICE); valid_masks = valid_masks.to(RUNTIME_DEVICE)
        weights = weights.to(RUNTIME_DEVICE)
        text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].to(RUNTIME_DEVICE)
        attention_mask = text_inputs["attention_mask"].to(RUNTIME_DEVICE)
        logits = model(pixel_values=imgs, input_ids=input_ids, attention_mask=attention_mask).logits
        logits = ensure_channel_dim(logits)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        # manifest 模式使用 sample_weight，floor 仅做极低值保护，不再用高 floor 强抬权重
        weights = weights.clamp(min=effective_weight_floor)
        bce_map = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
        valid_area = valid_masks.sum(dim=(1, 2, 3)).clamp(min=1.0)
        bce_per_sample = (bce_map * valid_masks).sum(dim=(1, 2, 3)) / valid_area
        loss_bce = (bce_per_sample * weights).mean()
        d_per_sample = dice_loss.per_sample(logits, masks, valid_mask=valid_masks)
        loss_dice = (d_per_sample * weights).mean()
        loss = BCE_WEIGHT * loss_bce + DICE_WEIGHT * loss_dice
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════
# 7. 验证
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, processor, loader):
    model.eval()
    all_metrics = defaultdict(list)
    vis_samples = []
    for imgs, prompts, masks, valid_masks, orig_prompts, _, _, _ in tqdm(
        loader,
        desc="  Val",
        bar_format=VAL_BAR_FORMAT,
        ncols=TQDM_NCOLS,
        leave=False,
    ):
        imgs = imgs.to(RUNTIME_DEVICE); masks = masks.to(RUNTIME_DEVICE)
        text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        logits = model(pixel_values=imgs, input_ids=text_inputs["input_ids"].to(RUNTIME_DEVICE),
                       attention_mask=text_inputs["attention_mask"].to(RUNTIME_DEVICE)).logits
        logits = ensure_channel_dim(logits)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(len(imgs)):
            pred_threshold = VAL_THRESHOLDS.get(orig_prompts[i], 0.5) if isinstance(VAL_THRESHOLDS, dict) else VAL_THRESHOLDS
            m = compute_metrics(logits[i:i+1], masks[i:i+1], threshold=pred_threshold)
            for k, v in m.items(): all_metrics[f"{k}/{orig_prompts[i]}"].append(v)
            all_metrics["iou/overall"].append(m["iou"]); all_metrics["dice/overall"].append(m["dice"])
            if len(vis_samples) < 8:
                vis_samples.append({
                    "image": imgs[i].cpu(), "mask_gt": masks[i, 0].cpu(),
                    "mask_pred": (torch.sigmoid(logits[i, 0]) > pred_threshold).float().cpu(),
                    "valid_mask": valid_masks[i, 0].cpu(),
                    "prompt": prompts[i],
                })
    summary = {k: np.mean(v) for k, v in all_metrics.items()}
    return summary, vis_samples


# ══════════════════════════════════════════════════════════════════════
# 8. 绘图（6 指标）
# ══════════════════════════════════════════════════════════════════════

def plot_results(history, vis_samples, save_path, epoch):
    fig = plt.figure(figsize=(16, 10))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs_range, history["train_loss"], "b-", linewidth=1)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("1. Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs_range, history["val_miou"], "g-", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU"); ax2.set_title("2. Val mIoU")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs_range, history["val_dice"], "m-", linewidth=1.5)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Dice"); ax3.set_title("3. Val Dice")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    for cls_name in CLASSES:
        k = f"iou/{cls_name}"
        if k in history and history[k]:
            ax4.plot(epochs_range, history[k], "-", label=cls_name, linewidth=1, alpha=0.8)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("IoU"); ax4.set_title("4. Per-Class IoU")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

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
        ignore_vis = 1.0 - s["valid_mask"].numpy()
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + ignore_vis * 0.35, 0, 1)
        ax6.imshow(overlay)
        ax6.set_title(f"6. {s['prompt'][:40]}\nRed=GT Cyan=Pred Green=Ignore", fontsize=8)
        ax6.axis("off")

    plt.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 9. 主训练
# ══════════════════════════════════════════════════════════════════════

def train():
    run_dir = PROJECT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    best_pt = run_dir / "best.pt"; last_pt = run_dir / "last.pt"
    results_png = run_dir / "results.png"

    resume_ckpt = None
    if last_pt.exists():
        resume_ckpt = last_pt
    elif best_pt.exists():
        resume_ckpt = best_pt

    init_ckpt = None if resume_ckpt is not None else INIT_CHECKPOINT
    model, processor, optimizer = load_model(init_checkpoint=init_ckpt)

    # DataLoader 必须在 scheduler 之前创建（scheduler 依赖 len(train_loader)）
    train_ds = CLIPSegDataset(
        MANIFEST_JSON if USE_MANIFEST else PRED_JSON,
        TRAIN_LIST,
        augment_prompt=True,
        oversample=RARE_OVERSAMPLE,
        include_negatives=(False if USE_MANIFEST else INCLUDE_NEGATIVE_SAMPLES),
        negative_ratio=NEGATIVE_SAMPLE_RATIO,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=(False if USE_MANIFEST else APPLY_SCORE_FILTER),
        use_manifest=USE_MANIFEST,
    )
    val_ds = CLIPSegDataset(
        VAL_PRED_JSON,
        VAL_LIST,
        augment_prompt=False,
        oversample=None,
        include_negatives=INCLUDE_NEGATIVE_SAMPLES,
        negative_ratio=1.0 if INCLUDE_NEGATIVE_SAMPLES else 0.0,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=VAL_APPLY_SCORE_FILTER,
        use_manifest=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
                            collate_fn=collate_fn, pin_memory=True)
    logger.info(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    dice_loss = DiceLoss()

    # warmup + cosine scheduler（per-step）
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    total_steps = EPOCHS * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return WARMUP_START_FACTOR + (1 - WARMUP_START_FACTOR) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LRF + (1 - LRF) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 断点续训
    start_epoch = 0; history = defaultdict(list); best_miou = -1

    if resume_ckpt:
        logger.info(f"检测到 checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=RUNTIME_DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info("  已恢复 scheduler 状态（warmup/cosine 衔接正确）")
        else:
            # 旧 checkpoint 无 scheduler，手动推进
            start_epoch_ckpt = ckpt.get("epoch", 0)
            for _ in range(start_epoch_ckpt * len(train_loader)): scheduler.step()
            logger.info("  旧 checkpoint 无 scheduler，手动推进 step")
        start_epoch = ckpt.get("epoch", 0); best_miou = ckpt.get("best_miou", -1)
        for k, v in ckpt.get("history", {}).items(): history[k] = list(v)
        if start_epoch >= EPOCHS:
            logger.info(f"已完成 ({start_epoch}/{EPOCHS})，跳过"); return best_pt
        logger.info(f"从 epoch {start_epoch + 1}/{EPOCHS} 续训, best_mIoU={best_miou:.4f}")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_loss = train_epoch(model, processor, train_loader, optimizer,
                                 dice_loss, scheduler, epoch, EPOCHS)
        val_metrics, vis_samples = validate(model, processor, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_miou"].append(val_metrics.get("iou/overall", 0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0))
        history["lr"].append(current_lr)
        for c in CLASSES: history[f"iou/{c}"].append(val_metrics.get(f"iou/{c}", 0))

        elapsed = time.time() - t0
        logger.info(f"  Loss={train_loss:.4f}  mIoU={val_metrics.get('iou/overall',0):.4f}  "
                    f"Dice={val_metrics.get('dice/overall',0):.4f}  LR={current_lr:.2e}  {elapsed/60:.1f}min")
        cls_str = "  ".join(f"{c}={val_metrics.get(f'iou/{c}', 0):.3f}" for c in CLASSES)
        logger.info(f"  Per-class IoU: {cls_str}")

        plot_results(history, vis_samples, results_png, epoch)

        save_dict = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "history": dict(history), "best_miou": best_miou}
        torch.save(save_dict, last_pt)

        current_miou = val_metrics.get("iou/overall", 0)
        if current_miou > best_miou:
            best_miou = current_miou; save_dict["best_miou"] = best_miou
            torch.save(save_dict, best_pt)
            logger.info(f"  ★ 新最佳模型 (mIoU={best_miou:.4f})")

    csv_path = run_dir / "results.csv"
    keys = ["train_loss", "val_miou", "val_dice", "lr"] + [f"iou/{c}" for c in CLASSES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch"] + keys)
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] if i < len(history[k]) else "" for k in keys])
    logger.info(f"完成, best mIoU={best_miou:.4f}, best.pt={best_pt}")
    return best_pt


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    if RUNTIME_DEVICE.type != "cuda":
        logger.error("未检测到 CUDA"); sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(GPU_INDEX)}")
    logger.info("=" * 60)
    logger.info("CLIPSeg 训练 v2")
    logger.info(f"  模型: {MODEL_NAME}")
    logger.info(f"  TrainMode: {'manifest' if USE_MANIFEST else 'pred_json'}")
    logger.info(f"  IMG_SIZE: {IMG_SIZE}  Batch: {BATCH}  Epochs: {EPOCHS}")
    logger.info(f"  RuntimeDevice: {RUNTIME_DEVICE_STR}")
    logger.info(f"  LR(decoder/backbone): {DECODER_LR} / {BACKBONE_LR}")
    logger.info(f"  INIT_CHECKPOINT: {INIT_CHECKPOINT}")
    if USE_MANIFEST:
        logger.info(f"  MANIFEST_JSON: {MANIFEST_JSON}")
        logger.info(f"  MANIFEST_LOSS_WEIGHT_FLOOR: {MANIFEST_LOSS_WEIGHT_FLOOR}")
        logger.info(f"  MANIFEST_NEGATIVE_RATIO: {MANIFEST_NEGATIVE_RATIO}")
        logger.info(f"  MANIFEST_CACHE_TAG: {MANIFEST_CACHE_TAG}")
        logger.info(f"  FORCE_REBUILD_MASK_CACHE: {FORCE_REBUILD_MASK_CACHE}")
    logger.info(f"  VAL_APPLY_SCORE_FILTER: {VAL_APPLY_SCORE_FILTER}")
    logger.info(f"  Warmup: {WARMUP_EPOCHS}ep  HFlip: {HFLIP_PROB}  PromptAug: {PROMPT_AUG_PROB}")
    logger.info(f"  类别: {CLASSES}  过采样: {RARE_OVERSAMPLE}")
    logger.info(
        f"  ScoreFilter: {APPLY_SCORE_FILTER}  "
        f"Negatives: {INCLUDE_NEGATIVE_SAMPLES} (train ratio={NEGATIVE_SAMPLE_RATIO})"
    )
    logger.info(
        f"  BoundaryWeakSup: {ENABLE_BOUNDARY_WEAK_SUPERVISION} "
        f"(default_ignore_width={BOUNDARY_IGNORE_WIDTH}, min_area={BOUNDARY_IGNORE_MIN_AREA}, adaptive_by_target_size=True)"
    )
    logger.info("=" * 60)

    ckpt = train()
    logger.info(f"最终模型: {ckpt}")
    logger.info(f"日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
