#!/usr/bin/env python3
"""
CLIPSeg v5 共享数据与工具脚本。

用途：
1. 读取配置
2. 构建训练/验证 Dataset
3. 提供图像预处理、RLE 编解码、prompt pool、后处理
4. 提供 checkpoint 兼容加载、推理增强和评估工具
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_opt.py"
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))


def load_config(config_path: Optional[Path] = None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    spec = importlib.util.spec_from_file_location(f"clipseg_v5_cfg_{config_path.stem}", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载配置: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG_PATH = config_path.resolve()
    return module


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_override=None, default_device=0, logger=None):
    requested = device_override if device_override is not None else default_device
    if isinstance(requested, int):
        requested = f"cuda:{requested}"
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if logger:
            logger.warning("请求 CUDA 但当前不可用，自动回退到 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def stable_hash(parts: Iterable[object]) -> str:
    import hashlib
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "model_state_dict", "state_dict", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            if any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
    raise RuntimeError("checkpoint 中未找到可用的 state_dict")


def smart_load_model_state(model, checkpoint_state: dict):
    model_state = model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    skipped_incompatible = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_incompatible.append(
                {
                    "key": key,
                    "checkpoint_shape": tuple(value.shape),
                    "model_shape": tuple(model_state[key].shape),
                }
            )
            continue
        compatible_state[key] = value
    if not compatible_state:
        raise RuntimeError("checkpoint 中没有任何 shape 兼容的权重可加载")
    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    model.load_state_dict(compatible_state, strict=False)
    return missing_keys, unexpected_keys, skipped_incompatible


def load_checkpoint_into_model(model, checkpoint_path: Path, logger=None, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    missing_keys, unexpected_keys, skipped_incompatible = smart_load_model_state(model, state_dict)
    if logger:
        logger.info("loaded checkpoint path: %s", checkpoint_path)
        logger.info("missing keys: %s", missing_keys if missing_keys else [])
        logger.info("unexpected keys: %s", unexpected_keys if unexpected_keys else [])
        logger.info("skipped incompatible keys: %s", skipped_incompatible[:20] if skipped_incompatible else [])
    return checkpoint, missing_keys, unexpected_keys, skipped_incompatible


def load_clipseg_model(cfg, device: torch.device, logger=None):
    local_model_ok = cfg.MODEL_DIR.exists() and (
        (cfg.MODEL_DIR / "pytorch_model.bin").exists()
        or (cfg.MODEL_DIR / "model.safetensors").exists()
        or (cfg.MODEL_DIR / "config.json").exists()
    )
    if local_model_ok:
        if logger:
            logger.info("从本地加载 CLIPSeg: %s", cfg.MODEL_DIR)
        processor = CLIPSegProcessor.from_pretrained(str(cfg.MODEL_DIR), local_files_only=True)
        model = CLIPSegForImageSegmentation.from_pretrained(str(cfg.MODEL_DIR), local_files_only=True)
    else:
        if logger:
            logger.info("从 HuggingFace 加载 CLIPSeg: %s", cfg.MODEL_NAME)
        processor = CLIPSegProcessor.from_pretrained(cfg.MODEL_NAME)
        model = CLIPSegForImageSegmentation.from_pretrained(cfg.MODEL_NAME)
    return model.to(device), processor


def classify_parameter_groups(model):
    text_params = []
    image_params = []
    decoder_params = []
    text_names = []
    image_names = []
    decoder_names = []
    for name, param in model.named_parameters():
        lname = name.lower()
        if "clip.text_model" in lname or "clip.text" in lname or "text_model" in lname:
            text_params.append(param)
            text_names.append(name)
        elif "clip.vision_model" in lname or "clip.visual" in lname or "vision_model" in lname:
            image_params.append(param)
            image_names.append(name)
        else:
            decoder_params.append(param)
            decoder_names.append(name)
    return {
        "text": (text_params, text_names),
        "image": (image_params, image_names),
        "decoder": (decoder_params, decoder_names),
    }


def freeze_for_decoder_only(model, cfg, logger=None):
    groups = classify_parameter_groups(model)
    for param in groups["text"][0]:
        param.requires_grad = not cfg.FREEZE_TEXT_ENCODER
    for param in groups["image"][0]:
        param.requires_grad = not cfg.FREEZE_IMAGE_ENCODER
    for param in groups["decoder"][0]:
        param.requires_grad = bool(cfg.TRAIN_DECODER_ONLY or (not cfg.FREEZE_TEXT_ENCODER and not cfg.FREEZE_IMAGE_ENCODER))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_text = sum(p.numel() for p in groups["text"][0] if not p.requires_grad)
    frozen_image = sum(p.numel() for p in groups["image"][0] if not p.requires_grad)
    trainable_decoder = sum(p.numel() for p in groups["decoder"][0] if p.requires_grad)
    if logger:
        logger.info("freeze text encoder: %s", cfg.FREEZE_TEXT_ENCODER)
        logger.info("freeze image encoder: %s", cfg.FREEZE_IMAGE_ENCODER)
        logger.info("train decoder only: %s", cfg.TRAIN_DECODER_ONLY)
        logger.info("total params: %d", total_params)
        logger.info("trainable params: %d", trainable_params)
        logger.info("frozen text params: %d", frozen_text)
        logger.info("frozen image params: %d", frozen_image)
        logger.info("trainable decoder params: %d", trainable_decoder)
        logger.info("text params preview: %s", groups["text"][1][:8])
        logger.info("image params preview: %s", groups["image"][1][:8])
        logger.info("decoder params preview: %s", groups["decoder"][1][:12])
    return groups, total_params, trainable_params


def rle_to_mask(rle):
    try:
        from pycocotools import mask as mask_utils
        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return mask_utils.decode(rle_copy).astype(np.float32)
    except ImportError:
        pass

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.split(",") if x.strip()]
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    value = 0
    for run_len in counts:
        if value == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        value = 1 - value
    return mask.reshape((h, w), order="F").astype(np.float32)


def simple_rle_encode(binary_mask: np.ndarray):
    flat = binary_mask.astype(np.uint8).flatten(order="F")
    runs = []
    prev = 0
    count = 0
    for value in flat:
        value = int(value)
        if value == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = value
    runs.append(count)
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": ",".join(str(x) for x in runs),
    }


def mask_to_rle(binary_mask: np.ndarray):
    binary_mask = binary_mask.astype(np.uint8)
    try:
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        if isinstance(rle, list):
            rle = rle[0]
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
        }
    except ImportError:
        return simple_rle_encode(binary_mask)


def build_empty_rle(height: int, width: int):
    return mask_to_rle(np.zeros((height, width), dtype=np.uint8))


def get_prompt_pool(cfg, class_name: str):
    pool = cfg.PROMPT_POOL.get(class_name, [class_name])
    return pool if pool else [class_name]


def resolve_prompt_to_class(cfg, prompt_text: str):
    prompt_text = str(prompt_text).strip()
    for class_name in cfg.CLASSES:
        if prompt_text in get_prompt_pool(cfg, class_name):
            return class_name
    return prompt_text


def choose_prompt_for_train(cfg, class_name: str):
    pool = get_prompt_pool(cfg, class_name)
    if len(pool) == 1 or random.random() > getattr(cfg, "PROMPT_SAMPLE_PROB", 1.0):
        return pool[0]
    return random.choice(pool)


def is_pseudo_color(img_bgr, cfg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > cfg.PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path: Path, cfg):
    img_bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        gray = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {abs_path}")
    elif is_pseudo_color(img_bgr, cfg):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(abs_path).replace("\\", "/")
    if "blackHot" in fname:
        gray = 255 - gray
    else:
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if std > 0:
            skew = float(np.mean(((gray - mean) / std) ** 3))
            if skew < cfg.BLACKHOT_SKEW_THRESH:
                gray = 255 - gray
        if mean > cfg.BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            gray = 255 - gray

    std = float(np.std(gray))
    noise_sigma = estimate_noise_sigma(gray)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if std < cfg.STD_LOW:
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    elif std < cfg.STD_MID:
        gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(gray)

    if noise_sigma > cfg.NOISE_HIGH:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > cfg.NOISE_MED:
        gray = cv2.medianBlur(gray, 3)

    if blur_score < cfg.BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = np.clip(cv2.filter2D(gray, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < cfg.BLUR_MID:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = np.clip(cv2.addWeighted(gray, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)
    return gray


def resize_and_pad_gray_mask(gray: np.ndarray, mask: Optional[np.ndarray], cfg):
    h, w = gray.shape
    scale = cfg.IMG_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = cfg.IMG_SIZE - nh, cfg.IMG_SIZE - nw
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    gray = cv2.copyMakeBorder(gray, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    if mask is not None:
        mask = cv2.copyMakeBorder(mask, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    return gray, mask, (nh, nw, pt, pb, pl, pr)


def preprocess_image_only(gray: np.ndarray, cfg):
    if cfg.GRAY2RGB:
        img = np.stack([gray] * 3, axis=-1)
    else:
        img = gray[:, :, None]
    img = img.astype(np.float32) / 255.0
    mean = np.array(cfg.CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.CLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std


def preprocess_for_infer(image_path: Path, cfg):
    gray = load_teacher_aligned_gray(image_path, cfg)
    orig_h, orig_w = gray.shape
    gray, _, resize_meta = resize_and_pad_gray_mask(gray, None, cfg)
    image = preprocess_image_only(gray, cfg)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w), resize_meta


def restore_to_original(binary_mask: np.ndarray, orig_hw: Tuple[int, int], resize_meta):
    orig_h, orig_w = orig_hw
    nh, nw, pt, pb, pl, pr = resize_meta
    cropped = binary_mask[pt:pt + nh, pl:pl + nw]
    restored = cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (restored > 0).astype(np.uint8)


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


def choose_boundary_ignore_width(mask: np.ndarray, class_name: str, cfg) -> int:
    default_width = max(int(cfg.BOUNDARY_IGNORE_WIDTH), 0)
    if default_width <= 0:
        return 0
    stats = compute_mask_area_stats(mask)
    total_area = stats["total_area"]
    max_component_area = stats["max_component_area"]
    tiny_thresh = int(cfg.TINY_AREA_THRESHOLDS.get(class_name, 64))
    if class_name in cfg.TINY_PROTECT_CLASSES:
        if max_component_area <= tiny_thresh or total_area <= tiny_thresh * 2:
            return 0
        if max_component_area <= tiny_thresh * 2:
            return min(default_width, 1)
        return default_width
    if class_name == "tree" and (max_component_area < tiny_thresh or total_area < tiny_thresh * 2):
        return min(default_width, 1)
    return default_width


def build_boundary_valid_mask(mask: np.ndarray, class_name: str, cfg):
    if not cfg.ENABLE_BOUNDARY_WEAK_SUPERVISION or cfg.BOUNDARY_IGNORE_WIDTH <= 0:
        return np.ones_like(mask, dtype=np.float32)
    binary = (mask > 0.5).astype(np.uint8)
    if int(binary.sum()) < cfg.BOUNDARY_IGNORE_MIN_AREA:
        return np.ones_like(mask, dtype=np.float32)
    ignore_width = choose_boundary_ignore_width(mask, class_name, cfg)
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


def apply_connected_component_filter(binary_mask: np.ndarray, min_area: int):
    if min_area <= 1 or binary_mask.sum() == 0:
        return binary_mask.astype(np.uint8)
    binary_mask = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    filtered = np.zeros_like(binary_mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
            filtered[labels == label_idx] = 1
    return filtered


def postprocess_probability_map(prob_map: np.ndarray, class_name: str, threshold: float, cfg, enable_postprocess: bool = True):
    pred = (prob_map > float(threshold)).astype(np.uint8)
    if not enable_postprocess:
        return pred
    min_area = int(cfg.MIN_AREA_BY_CLASS.get(class_name, 0))
    return apply_connected_component_filter(pred, min_area=min_area)


def suppress_class_conflicts(prob_maps: Dict[str, np.ndarray], class_names: Sequence[str]):
    if not prob_maps:
        return {}
    stack = np.stack([prob_maps[name] for name in class_names], axis=0)
    winners = np.argmax(stack, axis=0)
    output = {}
    for idx, class_name in enumerate(class_names):
        output[class_name] = prob_maps[class_name] * (winners == idx).astype(np.float32)
    return output


def tokenize_prompts(processor, prompts: Sequence[str], device: torch.device):
    tokenized = processor.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


def forward_logits_for_prompt(model, processor, images: torch.Tensor, prompt: str):
    input_ids, attention_mask = tokenize_prompts(processor, [prompt] * images.size(0), images.device)
    logits = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask).logits
    if logits.ndim == 4:
        logits = logits[:, 0]
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return logits


def apply_tta_prob(model, processor, images: torch.Tensor, prompt: str, tta: str):
    logits = forward_logits_for_prompt(model, processor, images, prompt)
    prob = torch.sigmoid(logits)
    if tta != "hflip":
        return prob
    flipped_images = torch.flip(images, dims=[3])
    flipped_logits = forward_logits_for_prompt(model, processor, flipped_images, prompt)
    flipped_prob = torch.sigmoid(flipped_logits)
    flipped_prob = torch.flip(flipped_prob, dims=[2])
    return 0.5 * prob + 0.5 * flipped_prob


@torch.no_grad()
def predict_probability_batch(
    model,
    processor,
    images: torch.Tensor,
    prompt_keys: Sequence[str],
    cfg,
    prompt_fusion: str = "none",
    tta: str = "none",
):
    prompt_fusion = prompt_fusion or "none"
    device = images.device
    batch_size, height, width = images.shape[0], images.shape[2], images.shape[3]
    output = torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
    groups = defaultdict(list)
    for idx, prompt_key in enumerate(prompt_keys):
        groups[prompt_key].append(idx)

    for prompt_key, indices in groups.items():
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        chunk = images.index_select(0, index_tensor)
        prompt_pool = get_prompt_pool(cfg, prompt_key) if prompt_key in cfg.CLASSES else [prompt_key]

        main_prob = apply_tta_prob(model, processor, chunk, prompt_pool[0], tta=tta)
        if prompt_fusion == "none" or len(prompt_pool) == 1:
            fused = main_prob
        else:
            alias_probs = [apply_tta_prob(model, processor, chunk, alias, tta=tta) for alias in prompt_pool[1:]]
            if prompt_fusion == "max":
                fused = torch.stack([main_prob, *alias_probs], dim=0).max(dim=0).values
            else:
                alias_mean = torch.stack(alias_probs, dim=0).mean(dim=0)
                fused = cfg.PROMPT_MAIN_WEIGHT * main_prob + cfg.PROMPT_ALIAS_WEIGHT * alias_mean
        output.index_copy_(0, index_tensor, fused)
    return output


def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    intersection = float(np.logical_and(pred_bin > 0, gt_bin > 0).sum())
    union = float(np.logical_or(pred_bin > 0, gt_bin > 0).sum())
    if pred_sum == 0 and gt_sum == 0:
        return {
            "iou": 1.0,
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "pred_positive": False,
            "gt_positive": False,
        }
    fp = float(np.logical_and(pred_bin > 0, gt_bin <= 0).sum())
    fn = float(np.logical_and(pred_bin <= 0, gt_bin > 0).sum())
    return {
        "iou": intersection / (union + 1e-7),
        "dice": 2 * intersection / (pred_sum + gt_sum + 1e-7),
        "precision": intersection / (intersection + fp + 1e-7),
        "recall": intersection / (intersection + fn + 1e-7),
        "pred_positive": pred_sum > 0,
        "gt_positive": gt_sum > 0,
    }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def per_sample(self, logits, target, valid_mask=None):
        pred = torch.sigmoid(logits).view(logits.size(0), -1)
        target = target.view(target.size(0), -1)
        if valid_mask is not None:
            valid_mask = valid_mask.view(valid_mask.size(0), -1)
            pred = pred * valid_mask
            target = target * valid_mask
        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        return 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def per_sample(self, logits, target, valid_mask=None):
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_map = alpha_t * torch.pow((1 - pt).clamp(min=1e-6), self.gamma) * bce_map
        if valid_mask is not None:
            valid_area = valid_mask.sum(dim=(1, 2)).clamp(min=1.0)
            return (focal_map * valid_mask).sum(dim=(1, 2)) / valid_area
        return focal_map.mean(dim=(1, 2))


class ClipSegDataset(Dataset):
    def __init__(
        self,
        cfg,
        pred_json: Path,
        image_list_path: Path,
        *,
        is_train: bool,
        include_negatives: bool,
        negative_ratio: float,
        negative_weight: float,
        oversample: Optional[dict],
        apply_score_filter: bool,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.is_train = bool(is_train)
        self.logger = logger
        self.samples = []

        with open(pred_json, encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(image_list_path, encoding="utf-8") as f:
            allowed = {line.strip().replace("\\", "/") for line in f if line.strip()}

        rng = random.Random(cfg.SEED)
        pred_json = Path(pred_json)
        cache_key = stable_hash([str(pred_json.resolve()), pred_json.stat().st_mtime, pred_json.stat().st_size])
        cache_dir = cfg.OUTPUT_ROOT / ".mask_cache_clipseg_opt" / f"{pred_json.stem}_{cache_key[:8]}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        skipped_empty = 0
        skipped_low_conf = 0
        skipped_low_conf_by_class = Counter()
        skipped_missing = 0
        negative_kept = 0
        class_counts_before = Counter()

        iterator = raw_data.get("records", []) if isinstance(raw_data, dict) else raw_data
        desc = "TrainData" if is_train else "ValData"
        for item in tqdm(iterator, desc=f"  Parse {desc}"):
            image_path = str(item.get("image_path", "")).replace("\\", "/")
            if image_path not in allowed:
                alt = image_path[5:] if image_path.startswith("test/") else f"test/{image_path}"
                if alt not in allowed:
                    continue
                image_path = alt
            abs_img_path = cfg.IMAGE_ROOT / image_path
            if not abs_img_path.exists():
                skipped_missing += 1
                continue

            prompts_dict = item.get("prompts")
            ann_id = item.get("ann_id", -1)
            if isinstance(prompts_dict, dict):
                for class_name, info in prompts_dict.items():
                    if class_name not in cfg.CLASSES:
                        continue
                    if not info.get("hit"):
                        if include_negatives and negative_ratio > 0 and rng.random() < negative_ratio:
                            meta = {"ann_id": ann_id, "is_positive": False, "source": "negative"}
                            self.samples.append((image_path, class_name, None, float(negative_weight), meta))
                            negative_kept += 1
                        continue
                    rle = info.get("rle")
                    if rle is None:
                        continue
                    threshold = cfg.CONF_FILTER.get(class_name, 0.5) if isinstance(cfg.CONF_FILTER, dict) else float(cfg.CONF_FILTER)
                    score = float(info.get("score", 1.0))
                    if apply_score_filter and score < threshold:
                        skipped_low_conf += 1
                        skipped_low_conf_by_class[class_name] += 1
                        continue
                    cache_name = stable_hash([image_path, class_name, "pred_json", "teacher"])
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() <= 0:
                            skipped_empty += 1
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                    meta = {"ann_id": ann_id, "is_positive": True, "source": "teacher", "score": score}
                    self.samples.append((image_path, class_name, cache_path, score, meta))
                    class_counts_before[class_name] += 1
            else:
                class_name = str(item.get("prompt", "")).strip()
                if class_name not in cfg.CLASSES:
                    continue
                if not item.get("include_in_train", True) or not item.get("selected_hit", True):
                    continue
                rle = item.get("rle")
                if rle is None:
                    continue
                cache_name = stable_hash([image_path, class_name, "manifest"])
                cache_path = cache_dir / f"{cache_name}.png"
                if not cache_path.exists():
                    mask = rle_to_mask(rle)
                    if mask.sum() <= 0:
                        skipped_empty += 1
                        continue
                    cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                weight = float(item.get("sample_weight", 1.0))
                meta = {"ann_id": ann_id, "is_positive": True, "source": "manifest", "score": weight}
                self.samples.append((image_path, class_name, cache_path, weight, meta))
                class_counts_before[class_name] += 1

        class_counts_after = Counter(class_counts_before)
        if oversample:
            extras = []
            for sample in tqdm(self.samples, desc=f"  Oversample {desc}"):
                image_path, class_name, cache_path, weight, meta = sample
                if not meta["is_positive"]:
                    continue
                mult = int(oversample.get(class_name, 1))
                for _ in range(mult - 1):
                    extras.append((image_path, class_name, cache_path, weight, dict(meta)))
                    class_counts_after[class_name] += 1
            self.samples.extend(extras)

        if logger:
            logger.info("%s parsed from %s", desc, pred_json.name)
            logger.info("negative kept: %d", negative_kept)
            logger.info("skipped empty: %d", skipped_empty)
            logger.info("skipped low conf: %d", skipped_low_conf)
            logger.info("skipped low conf by class: %s", dict(sorted(skipped_low_conf_by_class.items())))
            logger.info("skipped missing image: %d", skipped_missing)
            logger.info("class counts before oversample: %s", dict(sorted(class_counts_before.items())))
            logger.info("class counts after oversample: %s", dict(sorted(class_counts_after.items())))
            logger.info("total samples: %d", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name, cache_path, score, meta = self.samples[idx]
        prompt_text = choose_prompt_for_train(self.cfg, class_name) if self.is_train else get_prompt_pool(self.cfg, class_name)[0]
        gray = load_teacher_aligned_gray(self.cfg.IMAGE_ROOT / image_path, self.cfg)
        if meta["is_positive"]:
            mask_img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"掩码缓存读取失败: {cache_path}")
            mask = mask_img.astype(np.float32) / 255.0
        else:
            mask = np.zeros(gray.shape, dtype=np.float32)
        gray, mask, _ = resize_and_pad_gray_mask(gray, mask, self.cfg)
        valid_mask = build_boundary_valid_mask(mask, class_name, self.cfg) if meta["is_positive"] else np.ones_like(mask, dtype=np.float32)

        if self.is_train and random.random() < self.cfg.HFLIP_PROB:
            gray = np.fliplr(gray)
            mask = np.fliplr(mask)
            valid_mask = np.fliplr(valid_mask)

        image = preprocess_image_only(gray, self.cfg)
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1),
            "mask": torch.from_numpy(mask.copy()),
            "valid_mask": torch.from_numpy(valid_mask.copy()),
            "class_name": class_name,
            "prompt_text": prompt_text,
            "score": float(score),
            "image_path": image_path,
            "is_positive": bool(meta["is_positive"]),
            "ann_id": int(meta.get("ann_id", -1)),
        }


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "valid_masks": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "class_names": [item["class_name"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
        "scores": torch.tensor([item["score"] for item in batch], dtype=torch.float32),
        "image_paths": [item["image_path"] for item in batch],
        "is_positive": torch.tensor([item["is_positive"] for item in batch], dtype=torch.bool),
        "ann_ids": [item["ann_id"] for item in batch],
    }
