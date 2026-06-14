#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BertTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
HANXUE_ROOT = ROOT / "src" / "hanxue"
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(SCRIPT_DIR / ".hf_cache" / "hub"))

for candidate in [str(HANXUE_ROOT.resolve()), str((ROOT / "src").resolve())]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hanxue.models.fusion_decoder import FiLMFusionDecoder  # noqa: E402
from hanxue.models.image_encoder import PureImageEncoder  # noqa: E402
from hanxue.models.text_encoder import FreezeTextEncoder  # noqa: E402

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


DEFAULT_TASKS = ROOT / "test" / "test_tasks.json"
DEFAULT_IMAGE_ROOT = ROOT
DEFAULT_OUTPUT = ROOT / "test" / "predictions_dino_esam_hybrid.json"
DEFAULT_CHECKPOINT = ROOT / "test" / "train_output" / "ESAM-CCLIP-11-fastfinetune" / "final_fullset.pt"
DEFAULT_TOKENIZER_DIR = ROOT / "src" / "hanxue" / "weights" / "chinese_clip"
DEFAULT_EFFICIENT_SAM_CKPT = ROOT / "src" / "hanxue" / "weights" / "efficient_sam" / "efficient_sam_vitt.pt"
DEFAULT_TEXT_CACHE = ROOT / "test" / "cache" / "text_emb_11.pt"
DEFAULT_GD_CONFIG = ROOT / "src" / "DINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_GD_CKPT = ROOT / "src" / "DINO" / "models" / "groundingdino" / "groundingdino_swint_ogc.pth"
DEFAULT_HF_MODEL_ID = "IDEA-Research/grounding-dino-base"

DEFAULT_CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
]

DEFAULT_PROMPT_PROTOTYPES = {
    "person": ["person", "people", "pedestrian", "human", "人", "行人"],
    "car": ["car", "vehicle", "automobile", "车辆", "汽车"],
    "building": ["building", "house", "architecture", "建筑", "楼"],
    "tree": ["tree", "vegetation", "树", "树木"],
    "animal": ["animal", "wild animal", "动物"],
    "trash can": ["trash can", "garbage bin", "trashbin", "rubbish bin", "垃圾桶"],
    "window": ["window", "窗户"],
    "door": ["door", "entrance", "门"],
    "fence": ["fence", "railing", "栏杆", "围栏"],
    "pole_light": ["pole_light", "street light", "lamp", "light pole", "路灯", "灯杆"],
    "motorcycle": ["motorcycle", "motorbike", "摩托车"],
}

DEFAULT_ESAM_THRESHOLDS = {
    "person": 0.65,
    "car": 0.65,
    "building": 0.65,
    "tree": 0.55,
    "animal": 0.50,
    "trash can": 0.45,
    "window": 0.45,
    "door": 0.45,
    "fence": 0.40,
    "pole_light": 0.35,
    "motorcycle": 0.45,
}

DEFAULT_POSTPROCESS = {
    "person": {"min_area": 16, "fill_holes": False},
    "car": {"min_area": 16, "fill_holes": False},
    "building": {"min_area": 128, "fill_holes": True},
    "tree": {"min_area": 64, "fill_holes": True},
    "animal": {"min_area": 8, "fill_holes": False},
    "trash can": {"min_area": 4, "fill_holes": False},
    "window": {"min_area": 4, "fill_holes": False},
    "door": {"min_area": 8, "fill_holes": False},
    "fence": {"min_area": 2, "fill_holes": False},
    "pole_light": {"min_area": 1, "fill_holes": False},
    "motorcycle": {"min_area": 4, "fill_holes": False},
}

DEFAULT_DINO_BOX_THRESHOLDS = {
    "person": 0.25,
    "car": 0.25,
    "building": 0.20,
    "tree": 0.20,
    "animal": 0.18,
    "trash can": 0.18,
    "window": 0.18,
    "door": 0.18,
    "fence": 0.16,
    "pole_light": 0.16,
    "motorcycle": 0.18,
}

DEFAULT_IMG_SIZE = 768
DEFAULT_DINO_BOX_THRESHOLD = 0.18
DEFAULT_DINO_TEXT_THRESHOLD = 0.18
DEFAULT_MAX_BOXES_PER_PROMPT = 20
DEFAULT_BOX_EXPAND_RATIO = 0.12
DEFAULT_DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MAX_TEXT_LEN = 15

PSEUDO_COLOR_SAT_THRESH = 60.0
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0

LOGGER = logging.getLogger("DINO_ESAM_HYBRID")


def configure_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    LOGGER.addHandler(handler)


def maybe_tqdm(iterable: Iterable, total: int, desc: str, leave: bool = False) -> Iterable:
    try:
        from tqdm.auto import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=leave,
            ncols=96,
            bar_format="{l_bar}{bar:18}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        )
    except Exception:
        return iterable


def resolve_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    for candidate in [Path.cwd() / path, ROOT / path, SCRIPT_DIR / path]:
        if candidate.exists():
            return candidate
    return ROOT / path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
        device = torch.device(device_name if ":" in requested else "cuda:0")
        device_index = device.index if device.index is not None else 0
        LOGGER.info(
            "CUDA available: count=%d current=%d name=%s",
            torch.cuda.device_count(),
            device_index,
            torch.cuda.get_device_name(device_index),
        )
        return device
    device = torch.device(device_name)
    LOGGER.info("使用非 CUDA 设备: %s", device)
    return device


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ["tasks", "annotations", "data"]:
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("任务文件必须是 JSON 数组或包含 tasks/annotations/data 数组的对象。")
    tasks = []
    for idx, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"第 {idx} 条任务不是 JSON 对象。")
        if "ann_id" not in task or "image_path" not in task:
            raise ValueError(f"第 {idx} 条任务缺少 ann_id 或 image_path。")
        if not any(key in task for key in ["text_prompt", "prompt", "text"]):
            raise ValueError(f"第 {idx} 条任务缺少文本提示字段。")
        tasks.append(task)
    return tasks


def get_prompt_text(task: Dict[str, Any]) -> str:
    for key in ["text_prompt", "prompt", "text"]:
        value = task.get(key)
        if value is not None:
            prompt = str(value).strip()
            if prompt:
                return prompt
    raise KeyError(f"任务缺少有效 prompt 字段: {list(task.keys())}")


def group_tasks_by_image(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task["image_path"])].append(task)
    return grouped


def resolve_image_path(image_root: Path, image_rel_path: str) -> Path:
    image_rel_path = str(image_rel_path).replace("\\", "/")
    if Path(image_rel_path).is_absolute():
        return Path(image_rel_path)
    candidates = [image_root / image_rel_path]
    if image_rel_path.startswith("test/"):
        candidates.append(image_root / image_rel_path[5:])
    else:
        candidates.append(image_root / "test" / image_rel_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_tokenizer(tokenizer_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    except Exception:
        return BertTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def normalize_text_feature(text_feature: torch.Tensor) -> torch.Tensor:
    return text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def build_prompt_aliases(prompt_prototypes: Dict[str, List[str]], classes: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for class_name in classes:
        for alias in prompt_prototypes.get(class_name, [class_name]) + [class_name]:
            aliases[str(alias).strip().lower()] = class_name
    return aliases


def validate_text_cache(payload: Dict[str, Any], classes: List[str]) -> None:
    if set(payload.get("classes", [])) != set(classes):
        raise RuntimeError("text cache 与当前 classes 不一致。")
    for class_name in classes:
        if class_name not in payload.get("embeddings", {}):
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}")


def build_text_cache(
    model: "ESAMCCLIPModel",
    tokenizer,
    classes: List[str],
    prompt_prototypes: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    embeddings = {}
    aliases_payload = {}
    iterator = maybe_tqdm(classes, total=len(classes), desc="BuildTextCache", leave=False)
    for class_name in iterator:
        aliases = prompt_prototypes.get(class_name, [class_name])
        feature_list = []
        for alias in aliases:
            encoded = tokenizer(
                alias,
                padding="max_length",
                truncation=True,
                max_length=DEFAULT_MAX_TEXT_LEN,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                feature = model.encode_text(input_ids, attention_mask)
            feature = normalize_text_feature(feature)[0].detach().cpu()
            feature_list.append(feature)
        merged = torch.stack(feature_list, dim=0).mean(dim=0)
        merged = normalize_text_feature(merged.unsqueeze(0))[0].detach().cpu()
        embeddings[class_name] = merged
        aliases_payload[class_name] = aliases
    return {
        "classes": list(classes),
        "embeddings": embeddings,
        "aliases": aliases_payload,
        "use_prompt_prototype": True,
    }


def load_or_build_text_cache(
    model: "ESAMCCLIPModel",
    tokenizer,
    cache_path: Path,
    classes: List[str],
    prompt_prototypes: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_text_cache(payload, classes)
        LOGGER.info("loaded text cache: %s", cache_path)
        return payload
    payload = build_text_cache(model, tokenizer, classes, prompt_prototypes, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    save_json(cache_path.with_suffix(".json"), {"classes": classes, "aliases": payload["aliases"]})
    LOGGER.info("text cache saved: %s", cache_path)
    return payload


def encode_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = np.asarray(binary_mask)
    if binary_mask.ndim == 3:
        if binary_mask.shape[0] == 1:
            binary_mask = binary_mask[0]
        elif binary_mask.shape[-1] == 1:
            binary_mask = binary_mask[..., 0]
    if binary_mask.ndim != 2:
        raise ValueError(f"mask 必须是二维，当前: {binary_mask.shape}")
    binary_mask = (binary_mask > 0).astype(np.uint8)
    h, w = binary_mask.shape
    if mask_utils is not None:
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]
        return rle

    pixels = binary_mask.flatten(order="F")
    counts: List[int] = []
    prev = 0
    count = 0
    for pix in pixels:
        pix = int(pix)
        if pix == prev:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev = pix
    counts.append(count)
    return {"size": [int(h), int(w)], "counts": counts}


def empty_mask_rle(height: int, width: int) -> Dict[str, Any]:
    return encode_mask_to_rle(np.zeros((int(height), int(width)), dtype=np.uint8))


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {image_path}")
    elif is_pseudo_color(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(image_path).replace("\\", "/")
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


def letterbox_image(gray: np.ndarray, target_hw: Tuple[int, int], fill_value: int = 0) -> Tuple[np.ndarray, Dict[str, int]]:
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
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "scale": float(scale),
    }
    return padded, meta


def preprocess_gray_to_tensor(gray: np.ndarray, target_size: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    gray_pad, meta = letterbox_image(gray, (target_size, target_size), fill_value=0)
    rgb = np.stack([gray_pad, gray_pad, gray_pad], axis=-1).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(rgb).float(), meta


def logits_to_mask(logits: torch.Tensor, meta: Dict[str, int], threshold: float, target_size: int) -> Tuple[np.ndarray, float]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.ndim == 3:
        logits = logits.unsqueeze(1)
    logits = F.interpolate(logits, size=(target_size, target_size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    top = meta["pad_top"]
    left = meta["pad_left"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    prob = prob[top : top + new_h, left : left + new_w]
    prob = cv2.resize(prob, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)
    score = float(prob.max()) if prob.size > 0 else 0.0
    return (prob >= threshold).astype(np.uint8), score


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return (mask > 0).astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= int(min_area):
            filtered[labels == label_idx] = 1
    return filtered


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    canvas = binary.copy()
    cv2.drawContours(canvas, contours, -1, 1, thickness=cv2.FILLED)
    return canvas.astype(np.uint8)


def apply_postprocess(mask: np.ndarray, min_area: int = 0, fill_holes: bool = False) -> np.ndarray:
    processed = remove_small_components(mask, min_area=min_area)
    if fill_holes:
        processed = fill_small_holes(processed)
    return processed.astype(np.uint8)


class ESAMCCLIPModel(torch.nn.Module):
    def __init__(self, tokenizer_dir: Path, efficient_sam_ckpt: Path, freeze_image: bool = True, freeze_text: bool = True):
        super().__init__()
        self.img_encoder = PureImageEncoder(checkpoint_path=str(efficient_sam_ckpt), freeze=freeze_image)
        self.txt_encoder = FreezeTextEncoder(model_dir=str(tokenizer_dir))
        self.decoder = FiLMFusionDecoder(image_dim=256, text_dim=768)
        if freeze_text:
            for param in self.txt_encoder.parameters():
                param.requires_grad = False
            self.txt_encoder.eval()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.img_encoder(images)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, pooled = self.txt_encoder(input_ids, attention_mask)
        return pooled

    def decode(self, image_features: torch.Tensor, text_features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        return self.decoder(image_features=image_features, text_global_features=text_features, target_size=target_size)


def load_state_dict_flexible(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict。")


def _strip_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    for old_prefix, new_prefix in {
        "image_encoder.": "img_encoder.",
        "text_encoder.": "txt_encoder.",
        "fusion_decoder.": "decoder.",
    }.items():
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def load_checkpoint_flexible(model: ESAMCCLIPModel, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = load_state_dict_flexible(checkpoint)
    model_state = model.state_dict()
    matched_state = {}
    decoder_matched = 0
    missing_keys = []
    unexpected_keys = []
    for raw_key, value in state_dict.items():
        key = _strip_prefix(raw_key)
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            unexpected_keys.append(f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}")
            continue
        matched_state[key] = value
        if key.startswith("decoder."):
            decoder_matched += 1
    for key in model_state.keys():
        if key not in matched_state:
            missing_keys.append(key)
    if decoder_matched == 0:
        raise RuntimeError("decoder_matched_keys=0，无法加载 ESAM decoder 权重。")
    model.load_state_dict(matched_state, strict=False)
    LOGGER.info(
        "checkpoint loaded: %s matched=%d decoder_matched=%d missing=%d unexpected=%d",
        checkpoint_path,
        len(matched_state),
        decoder_matched,
        len(missing_keys),
        len(unexpected_keys),
    )
    return checkpoint


def map_prompt_to_known_class(prompt_text: str, prompt_aliases: Dict[str, str]) -> Optional[str]:
    return prompt_aliases.get(str(prompt_text).strip().lower())


def build_text_feature_for_prompt(
    model: ESAMCCLIPModel,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[str]]:
    mapped_class = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if mapped_class is not None and mapped_class in text_cache_payload["embeddings"]:
        feature = text_cache_payload["embeddings"][mapped_class].to(device)
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        return feature[0], mapped_class
    encoded = tokenizer(
        prompt_text,
        padding="max_length",
        truncation=True,
        max_length=DEFAULT_MAX_TEXT_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        feature = model.encode_text(input_ids, attention_mask)
    feature = normalize_text_feature(feature)[0]
    return feature, None


def get_prompt_threshold(
    prompt_text: str,
    mapped_class: Optional[str],
    thresholds: Dict[str, float],
    default_threshold: float,
) -> float:
    if mapped_class is not None:
        return float(thresholds.get(mapped_class, default_threshold))
    return float(thresholds.get(prompt_text, default_threshold))


def get_prompt_postprocess(
    prompt_text: str,
    mapped_class: Optional[str],
    postprocess_cfg: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if mapped_class is not None:
        return dict(postprocess_cfg.get(mapped_class, {"min_area": 0, "fill_holes": False}))
    return dict(postprocess_cfg.get(prompt_text, {"min_area": 0, "fill_holes": False}))


def get_prompt_dino_box_threshold(
    prompt_text: str,
    mapped_class: Optional[str],
    dino_thresholds: Dict[str, float],
    default_threshold: float,
) -> float:
    if mapped_class is not None:
        return float(dino_thresholds.get(mapped_class, default_threshold))
    return float(dino_thresholds.get(prompt_text, default_threshold))


def choose_tokenizer_dir(explicit: Optional[Path], model_dir: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if model_dir is not None:
        candidates.extend([model_dir, model_dir / "chinese_clip"])
    candidates.append(DEFAULT_TOKENIZER_DIR)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        files = {item.name for item in candidate.iterdir() if item.is_file()}
        if "config.json" in files and ({"vocab.txt", "tokenizer.json"} & files):
            return candidate
    raise FileNotFoundError("未找到 tokenizer 目录，请显式传入 --tokenizer_dir。")


def choose_text_cache_path(explicit: Optional[Path], checkpoint_preview: Dict[str, Any], model_dir: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    ckpt_cache = checkpoint_preview.get("text_cache_path")
    if ckpt_cache:
        candidates.append(resolve_path(ckpt_cache) or Path(ckpt_cache))
    if model_dir is not None:
        candidates.append(model_dir / "text_emb_11.pt")
    candidates.append(DEFAULT_TEXT_CACHE)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return candidates[0]


def choose_efficient_sam_ckpt(explicit: Optional[Path], model_dir: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if model_dir is not None:
        candidates.append(model_dir / "efficient_sam_vitt.pt")
    candidates.append(DEFAULT_EFFICIENT_SAM_CKPT)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return candidates[0]


def load_esam_resources(args, device: torch.device):
    checkpoint_path = resolve_path(args.checkpoint) or DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    model_dir = resolve_path(args.model_dir) if args.model_dir else None
    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint_preview.get("classes", DEFAULT_CLASSES))
    prompt_prototypes = checkpoint_preview.get("prompt_prototypes", DEFAULT_PROMPT_PROTOTYPES)
    thresholds = checkpoint_preview.get("val_thresholds", DEFAULT_ESAM_THRESHOLDS)
    postprocess_cfg = checkpoint_preview.get("postprocess_cfg", DEFAULT_POSTPROCESS)

    tokenizer_dir = choose_tokenizer_dir(resolve_path(args.tokenizer_dir), model_dir)
    efficient_sam_ckpt = choose_efficient_sam_ckpt(resolve_path(args.efficient_sam_ckpt), model_dir)
    text_cache_path = choose_text_cache_path(resolve_path(args.text_cache_path), checkpoint_preview, model_dir)

    model = ESAMCCLIPModel(
        tokenizer_dir=tokenizer_dir,
        efficient_sam_ckpt=efficient_sam_ckpt,
        freeze_image=True,
        freeze_text=True,
    ).to(device)
    checkpoint = load_checkpoint_flexible(model, checkpoint_path, device)
    tokenizer = load_tokenizer(tokenizer_dir)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=text_cache_path,
        classes=classes,
        prompt_prototypes=prompt_prototypes,
        device=device,
    )
    prompt_aliases = build_prompt_aliases(prompt_prototypes, classes)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "text_cache_payload": text_cache_payload,
        "prompt_aliases": prompt_aliases,
        "classes": classes,
        "thresholds": thresholds,
        "postprocess_cfg": postprocess_cfg,
        "model_input_size": int(checkpoint.get("img_size", DEFAULT_IMG_SIZE)),
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "tokenizer_dir": tokenizer_dir,
        "efficient_sam_ckpt": efficient_sam_ckpt,
        "text_cache_path": text_cache_path,
    }


def clip_box_xyxy(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    x2 = float(np.clip(x2, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    y2 = float(np.clip(y2, 0, max(height - 1, 0)))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def expand_box_xyxy(box: List[float], width: int, height: int, ratio: float) -> List[float]:
    x1, y1, x2, y2 = box
    w = max(x2 - x1 + 1.0, 1.0)
    h = max(y2 - y1 + 1.0, 1.0)
    dx = w * ratio * 0.5
    dy = h * ratio * 0.5
    return clip_box_xyxy([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def _convert_box(box: List[float], width: int, height: int, fmt: str) -> List[float]:
    arr = [float(v) for v in box]
    if fmt == "xyxy_abs":
        return clip_box_xyxy(arr, width, height)
    if fmt == "cxcywh_abs":
        cx, cy, w, h = arr
        return clip_box_xyxy([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], width, height)
    if fmt == "cxcywh_norm":
        cx, cy, w, h = arr
        return clip_box_xyxy(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ],
            width,
            height,
        )
    if fmt == "auto":
        if max(abs(v) for v in arr) <= 1.5:
            return _convert_box(arr, width, height, "cxcywh_norm")
        return _convert_box(arr, width, height, "xyxy_abs")
    raise ValueError(f"不支持的 box 格式: {fmt}")


def run_grounding_dino_hf(
    image_path: Path,
    prompt_text: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
    model_id: str,
    hf_box_format: str,
) -> List[Dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not hasattr(run_grounding_dino_hf, "_processor"):
        LOGGER.info("[DINO] loading HF backend: %s", model_id)
        run_grounding_dino_hf._processor = AutoProcessor.from_pretrained(model_id)
        run_grounding_dino_hf._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    processor = run_grounding_dino_hf._processor
    model = run_grounding_dino_hf._model
    image_pil = Image.open(image_path).convert("RGB")
    prompt = str(prompt_text).strip()
    if not prompt.endswith("."):
        prompt = prompt + "."
    inputs = processor(images=image_pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    result = results[0]
    boxes = result["boxes"].detach().cpu().tolist() if "boxes" in result else []
    scores = result["scores"].detach().cpu().tolist() if "scores" in result else []
    labels = result["labels"] if "labels" in result else []
    detections = []
    width, height = image_pil.size
    for box, score, label in zip(boxes, scores, labels):
        xyxy = _convert_box(box, width, height, hf_box_format)
        detections.append({"phrase": str(label), "score": float(score), "box": xyxy})
    return detections


def run_grounding_dino_local(
    image_path: Path,
    prompt_text: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
    config_path: Path,
    ckpt_path: Path,
    local_box_format: str,
) -> List[Dict[str, Any]]:
    from groundingdino.util.inference import Model as GDModel
    from groundingdino.util.inference import predict as gd_predict

    if not hasattr(run_grounding_dino_local, "_model"):
        LOGGER.info("[DINO] loading local backend: config=%s ckpt=%s", config_path, ckpt_path)
        run_grounding_dino_local._model = GDModel(
            model_config_path=str(config_path),
            model_checkpoint_path=str(ckpt_path),
            device=str(device),
        )

    image_pil = Image.open(image_path).convert("RGB")
    prompt = str(prompt_text).strip()
    if not prompt.endswith("."):
        prompt = prompt + "."
    boxes, logits, phrases = gd_predict(
        model=run_grounding_dino_local._model,
        image=str(image_path),
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    width, height = image_pil.size
    detections = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        score = float(logit.sigmoid().item() if hasattr(logit, "sigmoid") else logit)
        xyxy = _convert_box([float(v) for v in box], width, height, local_box_format)
        detections.append({"phrase": str(phrase), "score": score, "box": xyxy})
    return detections


def run_grounding_dino(image_path: Path, prompt_text: str, args, device: torch.device, box_threshold: float) -> List[Dict[str, Any]]:
    backends = [args.dino_backend] if args.dino_backend != "auto" else ["hf", "local"]
    last_error = None
    for backend in backends:
        try:
            if backend == "hf":
                return run_grounding_dino_hf(
                    image_path=image_path,
                    prompt_text=prompt_text,
                    box_threshold=box_threshold,
                    text_threshold=args.dino_text_threshold,
                    device=device,
                    model_id=args.hf_model_id,
                    hf_box_format=args.hf_box_format,
                )
            if backend == "local":
                config_path = resolve_path(args.groundingdino_config)
                ckpt_path = resolve_path(args.groundingdino_ckpt)
                if config_path is None or not config_path.exists():
                    raise FileNotFoundError("未找到 GroundingDINO config。")
                if ckpt_path is None or not ckpt_path.exists():
                    raise FileNotFoundError("未找到 GroundingDINO ckpt。")
                return run_grounding_dino_local(
                    image_path=image_path,
                    prompt_text=prompt_text,
                    box_threshold=box_threshold,
                    text_threshold=args.dino_text_threshold,
                    device=device,
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    local_box_format=args.local_box_format,
                )
        except Exception as exc:
            last_error = exc
            LOGGER.warning("[DINO] backend=%s failed: %s", backend, exc)
    raise RuntimeError(f"GroundingDINO 所有后端均失败，最后错误: {last_error}")


@torch.inference_mode()
def segment_gray_patch(
    gray_patch: np.ndarray,
    prompt_feature: torch.Tensor,
    model: ESAMCCLIPModel,
    model_input_size: int,
    threshold: float,
    postprocess_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    image_tensor, meta = preprocess_gray_to_tensor(gray_patch, model_input_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    text_tensor = prompt_feature.unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    logits = model.decode(
        image_features=image_features,
        text_features=text_tensor,
        target_size=(model_input_size, model_input_size),
    )
    mask, score = logits_to_mask(logits, meta, threshold, model_input_size)
    mask = apply_postprocess(
        mask,
        min_area=int(postprocess_cfg.get("min_area", 0)),
        fill_holes=bool(postprocess_cfg.get("fill_holes", False)),
    )
    return mask, score


@torch.inference_mode()
def segment_prompt_with_dino(
    gray_full: np.ndarray,
    image_path: Path,
    prompt_text: str,
    mapped_class: Optional[str],
    prompt_feature: torch.Tensor,
    model: ESAMCCLIPModel,
    model_input_size: int,
    threshold: float,
    postprocess_cfg: Dict[str, Any],
    args,
    device: torch.device,
    dino_box_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    detections = run_grounding_dino(
        image_path=image_path,
        prompt_text=prompt_text,
        args=args,
        device=device,
        box_threshold=dino_box_threshold,
    )
    detections = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    detections = detections[: int(args.max_boxes_per_prompt)]
    orig_h, orig_w = gray_full.shape[:2]
    merged_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    kept_boxes = []
    scores = []

    for det in detections:
        box = expand_box_xyxy(det["box"], orig_w, orig_h, float(args.box_expand_ratio))
        x1, y1, x2, y2 = box
        x1_i = max(int(math.floor(x1)), 0)
        y1_i = max(int(math.floor(y1)), 0)
        x2_i = min(int(math.ceil(x2)), orig_w - 1)
        y2_i = min(int(math.ceil(y2)), orig_h - 1)
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        gray_patch = gray_full[y1_i : y2_i + 1, x1_i : x2_i + 1]
        if gray_patch.size == 0:
            continue
        roi_mask, roi_score = segment_gray_patch(
            gray_patch=gray_patch,
            prompt_feature=prompt_feature,
            model=model,
            model_input_size=model_input_size,
            threshold=threshold,
            postprocess_cfg=postprocess_cfg,
        )
        if roi_mask.sum() == 0:
            continue
        merged_mask[y1_i : y2_i + 1, x1_i : x2_i + 1] = np.maximum(
            merged_mask[y1_i : y2_i + 1, x1_i : x2_i + 1],
            roi_mask.astype(np.uint8),
        )
        kept_boxes.append([x1_i, y1_i, x2_i, y2_i])
        scores.append(float(max(det["score"], roi_score)))

    if merged_mask.sum() == 0 and args.fallback_full_image:
        merged_mask, full_score = segment_gray_patch(
            gray_patch=gray_full,
            prompt_feature=prompt_feature,
            model=model,
            model_input_size=model_input_size,
            threshold=threshold,
            postprocess_cfg=postprocess_cfg,
        )
        if merged_mask.sum() > 0:
            scores.append(float(full_score))

    merged_mask = apply_postprocess(
        merged_mask,
        min_area=int(postprocess_cfg.get("min_area", 0)),
        fill_holes=bool(postprocess_cfg.get("fill_holes", False)),
    )
    debug_info = {
        "prompt": prompt_text,
        "mapped_class": mapped_class,
        "dino_raw_boxes": len(detections),
        "dino_kept_boxes": len(kept_boxes),
        "boxes": kept_boxes,
        "scores": scores,
        "threshold": float(threshold),
        "mask_area": int(merged_mask.sum()),
    }
    return merged_mask, debug_info


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GroundingDINO -> ROI ESAM hybrid inference.")
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--image_root", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--efficient_sam_ckpt", type=str, default=None)
    parser.add_argument("--text_cache_path", type=str, default=None)

    parser.add_argument("--dino_backend", choices=["auto", "hf", "local"], default="auto")
    parser.add_argument("--hf_model_id", type=str, default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--groundingdino_config", type=str, default=str(DEFAULT_GD_CONFIG))
    parser.add_argument("--groundingdino_ckpt", type=str, default=str(DEFAULT_GD_CKPT))
    parser.add_argument("--hf_box_format", choices=["xyxy_abs", "cxcywh_abs", "cxcywh_norm"], default="xyxy_abs")
    parser.add_argument("--local_box_format", choices=["auto", "xyxy_abs", "cxcywh_abs", "cxcywh_norm"], default="auto")

    parser.add_argument("--dino_box_threshold", type=float, default=DEFAULT_DINO_BOX_THRESHOLD)
    parser.add_argument("--dino_text_threshold", type=float, default=DEFAULT_DINO_TEXT_THRESHOLD)
    parser.add_argument("--max_boxes_per_prompt", type=int, default=DEFAULT_MAX_BOXES_PER_PROMPT)
    parser.add_argument("--box_expand_ratio", type=float, default=DEFAULT_BOX_EXPAND_RATIO)
    parser.add_argument("--default_mask_threshold", type=float, default=DEFAULT_DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--fallback_full_image", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--save_debug_json", action="store_true", default=False)
    return parser


def main() -> None:
    configure_logging()
    parser = build_argparser()
    args = parser.parse_args()

    tasks_path = resolve_path(args.tasks) or DEFAULT_TASKS
    image_root = resolve_path(args.image_root) or DEFAULT_IMAGE_ROOT
    output_path = resolve_path(args.output) or DEFAULT_OUTPUT
    device = resolve_device(args.device if torch.cuda.is_available() else "cpu")

    LOGGER.info("tasks=%s", tasks_path)
    LOGGER.info("image_root=%s", image_root)
    LOGGER.info("output=%s", output_path)
    LOGGER.info("checkpoint=%s", args.checkpoint)
    LOGGER.info("dino_backend=%s", args.dino_backend)
    LOGGER.info("fallback_full_image=%s", args.fallback_full_image)

    esam = load_esam_resources(args, device)
    model = esam["model"]
    tokenizer = esam["tokenizer"]
    text_cache_payload = esam["text_cache_payload"]
    prompt_aliases = esam["prompt_aliases"]
    thresholds = esam["thresholds"]
    postprocess_cfg = esam["postprocess_cfg"]
    model_input_size = esam["model_input_size"]

    dino_thresholds = dict(DEFAULT_DINO_BOX_THRESHOLDS)
    for class_name in esam["classes"]:
        dino_thresholds.setdefault(class_name, args.dino_box_threshold)

    tasks = load_tasks(tasks_path)
    tasks_by_image = group_tasks_by_image(tasks)
    LOGGER.info("task_count=%d image_count=%d", len(tasks), len(tasks_by_image))

    task_to_rle: Dict[Any, Dict[str, Any]] = {}
    debug_records: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    start_time = time.time()

    iterator = maybe_tqdm(tasks_by_image.items(), total=len(tasks_by_image), desc="HybridInfer", leave=False)
    for image_rel_path, image_tasks in iterator:
        image_abs_path = resolve_image_path(image_root, image_rel_path)
        try:
            gray_full = load_teacher_aligned_gray(image_abs_path)
            height, width = gray_full.shape[:2]
            results_by_prompt: Dict[str, Dict[str, Any]] = {}
            unique_prompts = list(dict.fromkeys(get_prompt_text(task) for task in image_tasks))

            for prompt_text in unique_prompts:
                prompt_feature, mapped_class = build_text_feature_for_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    text_cache_payload=text_cache_payload,
                    prompt_text=prompt_text,
                    prompt_aliases=prompt_aliases,
                    device=device,
                )
                threshold = get_prompt_threshold(prompt_text, mapped_class, thresholds, args.default_mask_threshold)
                post_cfg = get_prompt_postprocess(prompt_text, mapped_class, postprocess_cfg)
                dino_box_threshold = get_prompt_dino_box_threshold(
                    prompt_text,
                    mapped_class,
                    dino_thresholds,
                    args.dino_box_threshold,
                )
                mask, debug_info = segment_prompt_with_dino(
                    gray_full=gray_full,
                    image_path=image_abs_path,
                    prompt_text=prompt_text,
                    mapped_class=mapped_class,
                    prompt_feature=prompt_feature,
                    model=model,
                    model_input_size=model_input_size,
                    threshold=threshold,
                    postprocess_cfg=post_cfg,
                    args=args,
                    device=device,
                    dino_box_threshold=dino_box_threshold,
                )
                if args.save_debug_json:
                    debug_info["image_path"] = str(image_rel_path)
                    debug_records.append(debug_info)
                results_by_prompt[prompt_text] = {
                    "rle": encode_mask_to_rle(mask) if mask.sum() > 0 else empty_mask_rle(height, width)
                }

            for task in image_tasks:
                ann_id = task["ann_id"]
                prompt_text = get_prompt_text(task)
                task_to_rle[ann_id] = results_by_prompt[prompt_text]["rle"]

        except Exception as exc:
            if args.strict:
                raise
            LOGGER.warning("image failed: %s err=%s", image_rel_path, exc)
            width, height = 1, 1
            if image_abs_path.exists():
                try:
                    with Image.open(image_abs_path) as img:
                        width, height = img.size
                except Exception:
                    pass
            fallback_rle = empty_mask_rle(height, width)
            for task in image_tasks:
                task_to_rle[task["ann_id"]] = fallback_rle
                failed_items.append(
                    {"ann_id": task["ann_id"], "image_path": str(image_rel_path), "error": str(exc)}
                )

    predictions = [{"ann_id": task["ann_id"], "rle": task_to_rle[task["ann_id"]]} for task in tasks]
    output_payload = {"predictions": predictions}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    LOGGER.info("done output=%s task_count=%d elapsed=%.2fs", output_path, len(tasks), elapsed)

    if args.save_debug_json:
        debug_path = output_path.with_name(f"{output_path.stem}_debug.json")
        save_json(
            debug_path,
            {
                "model_input_size": model_input_size,
                "checkpoint": str(esam["checkpoint_path"]),
                "tokenizer_dir": str(esam["tokenizer_dir"]),
                "efficient_sam_ckpt": str(esam["efficient_sam_ckpt"]),
                "text_cache_path": str(esam["text_cache_path"]),
                "timing": {"elapsed_seconds": float(elapsed), "image_count": len(tasks_by_image), "task_count": len(tasks)},
                "failed_items": failed_items,
                "records": debug_records,
            },
        )
        LOGGER.info("debug saved: %s", debug_path)


if __name__ == "__main__":
    main()
