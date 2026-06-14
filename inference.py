#!/usr/bin/env python3
"""
DINO + ESAM 混合提交推理脚本

固定接口要求：
1. 本文件顶部的 DEFAULT_* 常量不可修改
2. 运行 `python inference.py`
3. 从 /raytron/test/test_tasks.json 读取任务
4. 输出 /raytron/test/predictions.json
5. 模型主 checkpoint 固定为 /raytron/code/model/sam3.pt
"""

from __future__ import annotations

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


DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test/"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_MODEL_DIR = "/raytron/code/model"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"
DEFAULT_MASK_THRESHOLD = 0.5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR_PATH = Path(DEFAULT_MODEL_DIR)
PROJECT_SRC = ROOT / "src"
HANXUE_ROOT = PROJECT_SRC / "hanxue"

for candidate in [str(PROJECT_SRC), str(HANXUE_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hanxue.models.fusion_decoder import FiLMFusionDecoder  # noqa: E402
from hanxue.models.image_encoder import PureImageEncoder  # noqa: E402
from hanxue.models.text_encoder import FreezeTextEncoder  # noqa: E402

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("警告: pycocotools 未安装，将使用备用 RLE 编码方法")
    maskUtils = None


LOGGER = logging.getLogger("DINO_ESAM_SUBMIT")
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(SCRIPT_DIR / ".hf_cache" / "hub"))

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

DEFAULT_DINO_TEXT_THRESHOLD = 0.18
DEFAULT_MAX_BOXES_PER_PROMPT = 20
DEFAULT_BOX_EXPAND_RATIO = 0.12
DEFAULT_FALLBACK_FULL_IMAGE = True
DEFAULT_MAX_TEXT_LEN = 15
DEFAULT_SUBMIT_IMG_SIZE = 768
DEFAULT_HF_MODEL_ID = "IDEA-Research/grounding-dino-base"

PSEUDO_COLOR_SAT_THRESH = 60.0
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0


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


def count_model_params(model) -> Dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "device": DEVICE,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def resolve_local_path(path: str | Path) -> Path:
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


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ["tasks", "annotations", "data"]:
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("任务文件必须是 JSON 数组或包含 tasks/annotations/data 数组的对象")

    tasks: List[Dict[str, Any]] = []
    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"第 {index} 条任务不是 JSON 对象")
        if "ann_id" not in task or "image_path" not in task:
            raise ValueError(f"第 {index} 条任务缺少 ann_id 或 image_path")
        if not any(key in task for key in ["text_prompt", "prompt", "text"]):
            raise ValueError(f"第 {index} 条任务缺少文本提示字段")
        tasks.append(task)
    return tasks


def get_prompt_text(task: Dict[str, Any]) -> str:
    for key in ["text_prompt", "prompt", "text"]:
        value = task.get(key)
        if value is not None:
            prompt = str(value).strip()
            if prompt:
                return prompt
    raise KeyError(f"任务中找不到有效文本提示字段: {list(task.keys())}")


def group_tasks_by_image(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task["image_path"])].append(task)
    return grouped


def resolve_image_path(image_root: str, image_rel_path: str) -> str:
    image_rel_path = image_rel_path.replace("\\", "/")
    direct = os.path.join(image_root, image_rel_path)
    if os.path.exists(direct):
        return direct
    if image_rel_path.startswith("test/"):
        stripped = os.path.join(image_root, image_rel_path[5:])
        if os.path.exists(stripped):
            return stripped
    prefixed = os.path.join(image_root, "test", image_rel_path)
    if os.path.exists(prefixed):
        return prefixed
    return direct


def build_prompt_aliases(prompt_prototypes: Dict[str, List[str]], classes: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for class_name in classes:
        for alias in prompt_prototypes.get(class_name, [class_name]) + [class_name]:
            aliases[str(alias).strip().lower()] = class_name
    return aliases


def normalize_text_feature(text_feature: torch.Tensor) -> torch.Tensor:
    return text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def validate_text_cache(payload: Dict[str, Any], classes: List[str]) -> None:
    if set(payload.get("classes", [])) != set(classes):
        raise RuntimeError("text cache 与当前 classes 不一致")
    for class_name in classes:
        if class_name not in payload.get("embeddings", {}):
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}")


def build_text_cache(
    model: "ESAMCCLIPModel",
    tokenizer,
    classes: List[str],
    prompt_prototypes: Dict[str, List[str]],
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
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
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
) -> Dict[str, Any]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_text_cache(payload, classes)
        LOGGER.info("loaded text cache: %s", cache_path)
        return payload
    payload = build_text_cache(model, tokenizer, classes, prompt_prototypes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    save_json(cache_path.with_suffix(".json"), {"classes": classes, "aliases": payload["aliases"]})
    LOGGER.info("text cache saved: %s", cache_path)
    return payload


def load_tokenizer(tokenizer_dir: str | Path):
    tokenizer_dir = str(tokenizer_dir)
    try:
        return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    except Exception:
        return BertTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def simple_rle_encode(binary_mask: np.ndarray) -> Dict[str, Any]:
    flat = binary_mask.astype(np.uint8).flatten(order="F")
    runs: List[int] = []
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
        "counts": ",".join(map(str, runs)),
    }


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = binary_mask.astype(np.uint8)
    if maskUtils is None:
        return simple_rle_encode(binary_mask)
    rle = maskUtils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle, list):
        rle = rle[0]
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
    }


def build_empty_rle(height: int, width: int) -> Dict[str, Any]:
    return mask_to_rle(np.zeros((height, width), dtype=np.uint8))


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    padded[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
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
    prob = prob[top:top + new_h, left:left + new_w]
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
    def __init__(self, tokenizer_dir: str, efficient_sam_ckpt: str, freeze_image: bool = True, freeze_text: bool = True):
        super().__init__()
        self.img_encoder = PureImageEncoder(checkpoint_path=efficient_sam_ckpt, freeze=freeze_image)
        self.txt_encoder = FreezeTextEncoder(model_dir=tokenizer_dir)
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

    def decode(self, image_features: torch.Tensor, text_features: torch.Tensor, target_size):
        return self.decoder(image_features=image_features, text_global_features=text_features, target_size=target_size)


def load_state_dict_flexible(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict")


def _strip_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("model."):
        key = key[len("model."):]
    for old_prefix, new_prefix in {
        "image_encoder.": "img_encoder.",
        "text_encoder.": "txt_encoder.",
        "fusion_decoder.": "decoder.",
    }.items():
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def load_checkpoint_flexible(model: ESAMCCLIPModel, checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
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
        raise RuntimeError("decoder_matched_keys=0，无法加载 ESAM decoder 权重")

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


def choose_tokenizer_dir(model_dir: Path) -> Path:
    candidates = [
        model_dir,
        model_dir / "chinese_clip",
        ROOT / "src" / "hanxue" / "weights" / "chinese_clip",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        files = {item.name for item in candidate.iterdir() if item.is_file()}
        if "config.json" in files and ({"vocab.txt", "tokenizer.json"} & files):
            return candidate
    raise FileNotFoundError("未找到 tokenizer 目录，请将 Chinese-CLIP tokenizer 放到 /raytron/code/model 或 /raytron/code/model/chinese_clip")


def choose_efficient_sam_ckpt(model_dir: Path) -> Path:
    candidates = [
        model_dir / "efficient_sam_vitt.pt",
        model_dir / "efficient_sam" / "efficient_sam_vitt.pt",
        model_dir / "DINO" / "efficientsam" / "efficient_sam_vitt.pt",
        ROOT / "src" / "hanxue" / "weights" / "efficient_sam" / "efficient_sam_vitt.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 EfficientSAM backbone 权重，请放到 /raytron/code/model/efficient_sam_vitt.pt")


def choose_text_cache_path(model_dir: Path, checkpoint_preview: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    ckpt_cache = checkpoint_preview.get("text_cache_path")
    if ckpt_cache:
        candidates.append(resolve_local_path(ckpt_cache))
    candidates.extend([
        model_dir / "text_emb_11.pt",
        model_dir / "cache" / "text_emb_11.pt",
        ROOT / "test" / "cache" / "text_emb_11.pt",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def choose_dino_config_path(model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "GroundingDINO_SwinT_OGC.py",
        model_dir / "groundingdino" / "GroundingDINO_SwinT_OGC.py",
        model_dir / "DINO" / "groundingdino" / "GroundingDINO_SwinT_OGC.py",
        ROOT / "src" / "DINO" / "GroundingDINO_SwinT_OGC.py",
        ROOT / "src" / "DINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def choose_dino_ckpt_path(model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "groundingdino_swint_ogc.pth",
        model_dir / "groundingdino" / "groundingdino_swint_ogc.pth",
        model_dir / "DINO" / "groundingdino" / "groundingdino_swint_ogc.pth",
        ROOT / "src" / "DINO" / "models" / "groundingdino" / "groundingdino_swint_ogc.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def map_prompt_to_known_class(prompt_text: str, prompt_aliases: Dict[str, str]) -> Optional[str]:
    return prompt_aliases.get(str(prompt_text).strip().lower())


def build_text_feature_for_prompt(
    model: ESAMCCLIPModel,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    prompt_text: str,
    prompt_aliases: Dict[str, str],
) -> Tuple[torch.Tensor, Optional[str]]:
    mapped_class = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if mapped_class is not None and mapped_class in text_cache_payload.get("embeddings", {}):
        feature = text_cache_payload["embeddings"][mapped_class].to(DEVICE)
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
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        feature = model.encode_text(input_ids, attention_mask)
    feature = normalize_text_feature(feature)[0]
    return feature, None


def get_prompt_threshold(prompt_text: str, mapped_class: Optional[str], thresholds: Dict[str, float], default_threshold: float) -> float:
    if mapped_class is not None:
        return float(thresholds.get(mapped_class, default_threshold))
    return float(thresholds.get(prompt_text, default_threshold))


def get_prompt_postprocess(prompt_text: str, mapped_class: Optional[str], postprocess_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if mapped_class is not None:
        return dict(postprocess_cfg.get(mapped_class, {"min_area": 0, "fill_holes": False}))
    return dict(postprocess_cfg.get(prompt_text, {"min_area": 0, "fill_holes": False}))


def get_prompt_dino_box_threshold(prompt_text: str, mapped_class: Optional[str], thresholds: Dict[str, float]) -> float:
    if mapped_class is not None:
        return float(thresholds.get(mapped_class, DEFAULT_DINO_BOX_THRESHOLDS.get(mapped_class, DEFAULT_MASK_THRESHOLD)))
    return float(thresholds.get(prompt_text, DEFAULT_MASK_THRESHOLD))


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


def run_grounding_dino_local(
    image_path: str,
    prompt_text: str,
    box_threshold: float,
    text_threshold: float,
    config_path: Path,
    ckpt_path: Path,
) -> List[Dict[str, Any]]:
    from groundingdino.util.inference import Model as GDModel
    from groundingdino.util.inference import predict as gd_predict

    if not hasattr(run_grounding_dino_local, "_model"):
        LOGGER.info("[DINO] 使用本地后端: config=%s ckpt=%s", config_path, ckpt_path)
        run_grounding_dino_local._model = GDModel(
            model_config_path=str(config_path),
            model_checkpoint_path=str(ckpt_path),
            device=DEVICE,
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
        xyxy = _convert_box([float(v) for v in box], width, height, "auto")
        detections.append({"phrase": str(phrase), "score": score, "box": xyxy})
    return detections


def run_grounding_dino_hf(
    image_path: str,
    prompt_text: str,
    box_threshold: float,
    text_threshold: float,
) -> List[Dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not hasattr(run_grounding_dino_hf, "_processor"):
        LOGGER.info("[DINO] 使用 HF 后端: %s", DEFAULT_HF_MODEL_ID)
        run_grounding_dino_hf._processor = AutoProcessor.from_pretrained(DEFAULT_HF_MODEL_ID)
        run_grounding_dino_hf._model = AutoModelForZeroShotObjectDetection.from_pretrained(DEFAULT_HF_MODEL_ID).to(DEVICE)

    processor = run_grounding_dino_hf._processor
    model = run_grounding_dino_hf._model
    image_pil = Image.open(image_path).convert("RGB")
    prompt = str(prompt_text).strip()
    if not prompt.endswith("."):
        prompt = prompt + "."

    inputs = processor(images=image_pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]], device=DEVICE)
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
        xyxy = _convert_box(box, width, height, "xyxy_abs")
        detections.append({"phrase": str(label), "score": float(score), "box": xyxy})
    return detections


def run_grounding_dino(image_path: str, prompt_text: str, box_threshold: float, text_threshold: float, model_dir: Path) -> List[Dict[str, Any]]:
    config_path = choose_dino_config_path(model_dir)
    ckpt_path = choose_dino_ckpt_path(model_dir)

    if config_path is not None and ckpt_path is not None:
        try:
            return run_grounding_dino_local(image_path, prompt_text, box_threshold, text_threshold, config_path, ckpt_path)
        except Exception as exc:
            LOGGER.warning("[DINO] 本地后端失败，回退 HF: %s", exc)

    try:
        return run_grounding_dino_hf(image_path, prompt_text, box_threshold, text_threshold)
    except Exception as exc:
        LOGGER.warning("[DINO] HF 后端失败，回退 ESAM 全图分割: %s", exc)
        return []


@torch.inference_mode()
def segment_gray_patch(
    gray_patch: np.ndarray,
    prompt_feature: torch.Tensor,
    model: ESAMCCLIPModel,
    model_input_size: int,
    threshold: float,
    postprocess_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    image_tensor, meta = preprocess_gray_to_tensor(gray_patch, model_input_size)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    text_tensor = prompt_feature.unsqueeze(0).to(DEVICE)
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
def do_inference(
    image_path: str,
    prompt_text: str,
    model: ESAMCCLIPModel,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    prompt_aliases: Dict[str, str],
    model_input_size: int,
    thresholds: Dict[str, float],
    postprocess_cfg: Dict[str, Dict[str, Any]],
    model_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any], int, int]:
    gray_full = load_teacher_aligned_gray(image_path)
    orig_h, orig_w = gray_full.shape[:2]
    prompt_feature, mapped_class = build_text_feature_for_prompt(
        model=model,
        tokenizer=tokenizer,
        text_cache_payload=text_cache_payload,
        prompt_text=prompt_text,
        prompt_aliases=prompt_aliases,
    )
    threshold = get_prompt_threshold(prompt_text, mapped_class, thresholds, DEFAULT_MASK_THRESHOLD)
    post_cfg = get_prompt_postprocess(prompt_text, mapped_class, postprocess_cfg)
    dino_box_threshold = get_prompt_dino_box_threshold(prompt_text, mapped_class, DEFAULT_DINO_BOX_THRESHOLDS)

    detections = run_grounding_dino(
        image_path=image_path,
        prompt_text=prompt_text,
        box_threshold=dino_box_threshold,
        text_threshold=DEFAULT_DINO_TEXT_THRESHOLD,
        model_dir=model_dir,
    )
    detections = sorted(detections, key=lambda item: float(item["score"]), reverse=True)[:DEFAULT_MAX_BOXES_PER_PROMPT]

    merged_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    kept_boxes = []
    scores = []

    for det in detections:
        box = expand_box_xyxy(det["box"], orig_w, orig_h, DEFAULT_BOX_EXPAND_RATIO)
        x1, y1, x2, y2 = box
        x1_i = max(int(math.floor(x1)), 0)
        y1_i = max(int(math.floor(y1)), 0)
        x2_i = min(int(math.ceil(x2)), orig_w - 1)
        y2_i = min(int(math.ceil(y2)), orig_h - 1)
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        gray_patch = gray_full[y1_i:y2_i + 1, x1_i:x2_i + 1]
        if gray_patch.size == 0:
            continue

        roi_mask, roi_score = segment_gray_patch(
            gray_patch=gray_patch,
            prompt_feature=prompt_feature,
            model=model,
            model_input_size=model_input_size,
            threshold=threshold,
            postprocess_cfg=post_cfg,
        )
        if roi_mask.sum() == 0:
            continue

        merged_mask[y1_i:y2_i + 1, x1_i:x2_i + 1] = np.maximum(
            merged_mask[y1_i:y2_i + 1, x1_i:x2_i + 1],
            roi_mask.astype(np.uint8),
        )
        kept_boxes.append([x1_i, y1_i, x2_i, y2_i])
        scores.append(float(max(det["score"], roi_score)))

    if merged_mask.sum() == 0 and DEFAULT_FALLBACK_FULL_IMAGE:
        full_mask, full_score = segment_gray_patch(
            gray_patch=gray_full,
            prompt_feature=prompt_feature,
            model=model,
            model_input_size=model_input_size,
            threshold=threshold,
            postprocess_cfg=post_cfg,
        )
        if full_mask.sum() > 0:
            merged_mask = full_mask
            scores.append(float(full_score))

    merged_mask = apply_postprocess(
        merged_mask,
        min_area=int(post_cfg.get("min_area", 0)),
        fill_holes=bool(post_cfg.get("fill_holes", False)),
    )

    debug_info = {
        "prompt": prompt_text,
        "mapped_class": mapped_class,
        "threshold": float(threshold),
        "dino_box_threshold": float(dino_box_threshold),
        "dino_raw_boxes": len(detections),
        "dino_kept_boxes": len(kept_boxes),
        "boxes": kept_boxes,
        "scores": scores,
        "mask_area": int(merged_mask.sum()),
    }
    return merged_mask, debug_info, orig_w, orig_h


def load_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT_PATH,
) -> Tuple[ESAMCCLIPModel, Any, Dict[str, Any], Dict[str, str], Dict[str, float], Dict[str, Dict[str, Any]], int, Dict[str, Any], Path]:
    model_dir_path = Path(model_dir)
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"缺少模型权重文件: {checkpoint_path}")

    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint_preview.get("classes", DEFAULT_CLASSES))
    prompt_prototypes = checkpoint_preview.get("prompt_prototypes", DEFAULT_PROMPT_PROTOTYPES)
    thresholds = checkpoint_preview.get("val_thresholds", DEFAULT_ESAM_THRESHOLDS)
    postprocess_cfg = checkpoint_preview.get("postprocess_cfg", DEFAULT_POSTPROCESS)
    model_input_size = int(checkpoint_preview.get("img_size", DEFAULT_SUBMIT_IMG_SIZE))

    tokenizer_dir = choose_tokenizer_dir(model_dir_path)
    efficient_sam_ckpt = choose_efficient_sam_ckpt(model_dir_path)
    text_cache_path = choose_text_cache_path(model_dir_path, checkpoint_preview)

    model = ESAMCCLIPModel(
        tokenizer_dir=str(tokenizer_dir),
        efficient_sam_ckpt=str(efficient_sam_ckpt),
        freeze_image=True,
        freeze_text=True,
    ).to(DEVICE)
    checkpoint = load_checkpoint_flexible(model, checkpoint_path)
    tokenizer = load_tokenizer(tokenizer_dir)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=text_cache_path,
        classes=classes,
        prompt_prototypes=prompt_prototypes,
    )
    prompt_aliases = build_prompt_aliases(prompt_prototypes, classes)
    model.eval()
    return (
        model,
        tokenizer,
        text_cache_payload,
        prompt_aliases,
        thresholds,
        postprocess_cfg,
        model_input_size,
        checkpoint,
        model_dir_path,
    )


def process_tasks(
    tasks: List[Dict[str, Any]],
    image_root: str,
    model_dir: str,
    checkpoint_path: Optional[str],
    output_path: str,
    mask_threshold: float,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    (
        model,
        tokenizer,
        text_cache_payload,
        prompt_aliases,
        thresholds,
        postprocess_cfg,
        model_input_size,
        checkpoint,
        model_dir_path,
    ) = load_model(model_dir=model_dir, checkpoint_path=checkpoint_path)

    model_info = count_model_params(model)
    model_info["model_type"] = "GroundingDINO+ESAM(ROI hybrid)"
    model_info["checkpoint_path"] = checkpoint_path
    model_info["default_mask_threshold"] = float(mask_threshold)
    model_info["model_input_size"] = int(model_input_size)
    model_info["class_mask_thresholds"] = {key: float(value) for key, value in sorted(thresholds.items())}
    if isinstance(checkpoint, dict):
        for key in ["epoch", "run_name", "model_type"]:
            if key in checkpoint:
                model_info[f"checkpoint_{key}"] = checkpoint[key]

    tasks_by_image = group_tasks_by_image(tasks)
    print(f"任务总数: {len(tasks)}")
    print(f"图片总数: {len(tasks_by_image)}")

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}
    failed_items: List[Dict[str, Any]] = []

    pbar = maybe_tqdm(tasks_by_image.items(), total=len(tasks_by_image), desc="推理进度", leave=False)
    for image_rel_path, image_tasks in pbar:
        image_abs_path = resolve_image_path(image_root, image_rel_path)
        image_start = time.time()

        try:
            unique_prompts = list(dict.fromkeys(get_prompt_text(task) for task in image_tasks))
            results_by_prompt: Dict[str, Dict[str, Any]] = {}

            for prompt_text in unique_prompts:
                mask, debug_info, width, height = do_inference(
                    image_path=image_abs_path,
                    prompt_text=prompt_text,
                    model=model,
                    tokenizer=tokenizer,
                    text_cache_payload=text_cache_payload,
                    prompt_aliases=prompt_aliases,
                    model_input_size=model_input_size,
                    thresholds=thresholds,
                    postprocess_cfg=postprocess_cfg,
                    model_dir=model_dir_path,
                )
                rle = mask_to_rle(mask) if mask.sum() > 0 else build_empty_rle(height, width)
                results_by_prompt[prompt_text] = {"rle": rle, "debug": debug_info}

            for task in image_tasks:
                ann_id = task["ann_id"]
                prompt_text = get_prompt_text(task)
                task_to_rle[ann_id] = results_by_prompt[prompt_text]["rle"]

        except Exception as exc:
            width, height = 1, 1
            if os.path.exists(image_abs_path):
                try:
                    with Image.open(image_abs_path) as img:
                        width, height = img.size
                except Exception:
                    pass
            fallback_rle = build_empty_rle(height, width)
            for task in image_tasks:
                task_to_rle[task["ann_id"]] = fallback_rle
                failed_items.append(
                    {
                        "ann_id": task["ann_id"],
                        "image_path": image_rel_path,
                        "error": str(exc),
                    }
                )
            print(f"警告: 图片推理失败，已回退为空掩码: {image_rel_path} err={exc}")

        elapsed = time.time() - image_start
        inference_total_time += elapsed
        processed_images += 1

    predictions_output = []
    for task in tasks:
        ann_id = task["ann_id"]
        if ann_id not in task_to_rle:
            raise RuntimeError(f"ann_id {ann_id} 未生成结果")
        predictions_output.append({"ann_id": ann_id, "rle": task_to_rle[ann_id]})

    avg_inference_time = inference_total_time / processed_images if processed_images > 0 else 0.0
    output_data = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_inference_time),
            "processed_images": processed_images,
            "total_tasks": len(tasks),
            "failed_count": len(failed_items),
        },
        "predictions": predictions_output,
    }

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)

    print("\n推理完成!")
    print(f"输出文件: {output_path_obj}")
    print(f"纯推理总耗时: {inference_total_time:.2f}s")
    if processed_images > 0:
        print(f"平均每张图推理耗时: {avg_inference_time:.2f}s")
    if failed_items:
        debug_path = output_path_obj.with_name(f"{output_path_obj.stem}_debug_failed.json")
        save_json(debug_path, {"failed_items": failed_items})
        print(f"失败样本记录已保存: {debug_path}")


def main() -> None:
    configure_logging()
    print("=" * 60)
    print("DINO + ESAM 混合推理配置")
    print("=" * 60)
    print(f"任务文件: {DEFAULT_TASKS}")
    print(f"图片根目录: {DEFAULT_IMAGE_ROOT}")
    print(f"输出路径: {DEFAULT_OUTPUT_PATH}")
    print(f"模型目录: {DEFAULT_MODEL_DIR}")
    print(f"模型检查点: {DEFAULT_CHECKPOINT_PATH}")
    print(f"默认掩码阈值: {DEFAULT_MASK_THRESHOLD}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    tasks = load_tasks(DEFAULT_TASKS)
    process_tasks(
        tasks=tasks,
        image_root=DEFAULT_IMAGE_ROOT,
        model_dir=DEFAULT_MODEL_DIR,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        mask_threshold=DEFAULT_MASK_THRESHOLD,
    )


if __name__ == "__main__":
    main()
