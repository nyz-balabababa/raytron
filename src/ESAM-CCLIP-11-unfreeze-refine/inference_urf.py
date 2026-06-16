#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BertTokenizer, ChineseCLIPTextConfig, ChineseCLIPTextModel

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_MODEL_DIR = "/raytron/code/model"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"

DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_SUBMIT_IMG_SIZE = 768
PROMPT_BATCH_SIZE = 16
MAX_TEXT_LEN = 15
DEFAULT_PROMPT_FUSION_MODE = "prototype"
DEFAULT_RAW_PROMPT_WEIGHT = 0.0
DEFAULT_PROMPT_MATCH_MODE = "exact"

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
DEFAULT_OLD5_CLASSES = ["person", "car", "building", "tree", "animal"]
DEFAULT_RARE6_CLASSES = ["trash can", "window", "door", "fence", "pole_light", "motorcycle"]
DEFAULT_ROUTE_MAP = {
    "person": "lite",
    "car": "lite",
    "building": "lite",
    "tree": "lite",
    "animal": "lite",
    "trash can": "stage2",
    "window": "stage2",
    "door": "stage2",
    "fence": "stage2",
    "pole_light": "stage2",
    "motorcycle": "stage2",
}
DEFAULT_ROUTER_MAX_RUNTIME_PARAMS = 300_000_000

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

DEFAULT_THRESHOLDS = {
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

RARE_FALLBACK_CLASSES = {
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
}
RARE_EMPTY_FALLBACK_DELTA = 0.05
RARE_EMPTY_FALLBACK_CFG = {
    "trash can": {"min_area": 2, "topk_components": 1},
    "window": {"min_area": 2, "topk_components": 2},
    "door": {"min_area": 4, "topk_components": 1},
    "fence": {"min_area": 1, "topk_components": 3},
    "pole_light": {"min_area": 1, "topk_components": 2},
    "motorcycle": {"min_area": 2, "topk_components": 1},
}
LARGE_AREA_DEBUG_THRESHOLDS = {
    "person": 0.35,
    "car": 0.40,
    "building": 0.85,
    "tree": 0.70,
    "animal": 0.20,
}

PSEUDO_COLOR_SAT_THRESH = 60.0
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0

LOGGER = logging.getLogger("ESAM_CCLIP_11_SUBMIT")


def configure_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    LOGGER.addHandler(handler)


def ensure_hf_cache() -> None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = str(Path(__file__).resolve().parent / ".hf_cache")
        os.environ["HF_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "hub"))


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    cwd_path = Path.cwd() / path_obj
    if cwd_path.exists():
        return str(cwd_path)
    file_dir_path = Path(__file__).resolve().parent / path_obj
    if file_dir_path.exists():
        return str(file_dir_path)
    root_path = Path(__file__).resolve().parents[2] / path_obj
    return str(root_path)


class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = torch.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Attention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_num, channel_num = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, token_num, 3, self.num_heads, channel_num // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_num, channel_num)
        return self.proj(x)


class Mlp(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer=torch.nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        act_layer=torch.nn.GELU,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]) -> torch.Tensor:
    height = hw[0]
    width = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    if size * size != xy_num:
        raise ValueError(f"绝对位置编码长度异常: {xy_num}")
    if size != height or size != width:
        resized = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, height, width, -1)


class ImageEncoderViT(torch.nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.image_embedding_size = img_size // (patch_size if patch_size > 0 else 1)
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, patch_embed_dim))
        self.blocks = torch.nn.ModuleList(
            [Block(patch_embed_dim, num_heads, mlp_ratio, True, act_layer=act_layer) for _ in range(depth)]
        )
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(patch_embed_dim, neck_dims[0], kernel_size=1, bias=False),
            LayerNorm2d(neck_dims[0]),
            torch.nn.Conv2d(neck_dims[0], neck_dims[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dims[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            raise ValueError("input image size must match self.img_size")
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        num_patches = x.shape[1]
        if x.shape[2] != num_patches:
            raise ValueError(f"ViT patch 网格异常: {tuple(x.shape)}")
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        return self.neck(x.permute(0, 3, 1, 2))


def build_submission_image_encoder() -> ImageEncoderViT:
    return ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        patch_embed_dim=192,
        normalization_type="layer_norm",
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        neck_dims=[256, 256],
        act_layer=torch.nn.GELU,
    )


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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def count_parameters(model: torch.nn.Module) -> Tuple[float, float]:
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6
    return total_params, trainable_params


def count_model_params(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    total_params, trainable_params = count_parameters(model)
    return {
        "device": str(device),
        "total_params_m": float(total_params),
        "trainable_params_m": float(trainable_params),
    }


def count_tensor_params(state_dict: Dict[str, Any]) -> int:
    return int(sum(value.numel() for value in state_dict.values() if torch.is_tensor(value)))


def normalize_prompt_key(text: str) -> str:
    normalized = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


COMPLEX_PROMPT_ALIASES: Dict[str, List[str]] = {
    "car": [
        "truck",
        "bus",
        "车辆",
        "汽车",
        "车",
        "小车",
        "远处的车",
        "远处的小车",
        "草丛里的车",
        "被树遮挡的车",
        "被遮挡的车",
        "car behind tree",
        "car in bushes",
        "partially occluded car",
        "small distant car",
        "vehicle on road",
    ],
    "window": [
        "窗户",
        "窗",
        "车窗",
        "汽车窗户",
        "car window",
        "building window",
        "window on building",
        "建筑窗户",
        "楼上的窗户",
        "建筑上的窗户",
        "glass window",
        "window of car",
    ],
    "door": [
        "门",
        "车门",
        "汽车门",
        "car door",
        "building door",
        "door of car",
        "建筑门",
        "door on vehicle",
    ],
    "pole_light": [
        "pole light",
        "streetlight",
        "street light",
        "lamp post",
        "utility pole",
        "street light pole",
        "light pole",
        "路灯",
        "灯杆",
        "路边的路灯",
        "路边的灯杆",
        "远处的灯杆",
        "street light beside road",
    ],
    "motorcycle": [
        "motor bike",
        "scooter",
        "motorbike",
        "摩托车",
        "远处的摩托车",
        "small motorcycle",
    ],
    "trash can": [
        "trashcan",
        "trash can",
        "garbage can",
        "waste bin",
        "dustbin",
        "garbage bin",
        "垃圾桶",
        "路边的垃圾桶",
        "rubbish bin",
    ],
}
ENGLISH_TARGET_RELATIONS = [" in ", " on ", " under ", " behind ", " near ", " beside ", " among ", " with ", " inside ", " at "]
PROMPT_MATCH_SELFTEST_CASES: List[Tuple[str, Optional[str]]] = [
    ("草丛里的车", "car"),
    ("被树遮挡的车", "car"),
    ("远处的小车", "car"),
    ("建筑上的窗户", "window"),
    ("车窗", "window"),
    ("车门", "door"),
    ("路边的灯杆", "pole_light"),
    ("路边的垃圾桶", "trash can"),
    ("car in bushes", "car"),
    ("car behind tree", "car"),
    ("window on building", "window"),
    ("door of car", "door"),
    ("car window", "window"),
    ("street light pole", "pole_light"),
    ("cart", None),
    ("street", None),
]


def build_complex_prompt_alias_map(classes: List[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for class_name in classes:
        for alias in COMPLEX_PROMPT_ALIASES.get(class_name, []):
            alias_map[normalize_prompt_key(alias)] = class_name
    return alias_map


def build_prompt_aliases(prompt_prototypes: Dict[str, List[str]], classes: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for class_name in classes:
        for alias in prompt_prototypes.get(class_name, [class_name]) + [class_name]:
            aliases[normalize_prompt_key(alias)] = class_name
    aliases.update(build_complex_prompt_alias_map(classes))
    return aliases


def normalize_prompt_prototypes(
    prompt_prototypes: Dict[str, List[str]],
    classes: List[str],
) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for class_name in classes:
        aliases = prompt_prototypes.get(class_name, [class_name])
        normalized[class_name] = [str(alias).strip() for alias in aliases]
    return normalized


def build_prompt_prototypes_signature(
    prompt_prototypes: Dict[str, List[str]],
    classes: List[str],
) -> Dict[str, List[str]]:
    return normalize_prompt_prototypes(prompt_prototypes, classes)


def resolve_image_path(image_root: str, image_rel_path: str) -> str:
    image_rel_path = str(image_rel_path).replace("\\", "/")
    if os.path.isabs(image_rel_path):
        return image_rel_path
    candidates = [os.path.join(image_root, image_rel_path)]
    if image_rel_path.startswith("test/"):
        candidates.append(os.path.join(image_root, image_rel_path[5:]))
    else:
        candidates.append(os.path.join(image_root, "test", image_rel_path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)
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


def load_tokenizer(tokenizer_dir: str):
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    except Exception:
        return BertTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def normalize_text_feature(text_feature: torch.Tensor) -> torch.Tensor:
    return text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(image_path: str | Path) -> np.ndarray:
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
    padded[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return padded, meta


def preprocess_image(image_path: str, input_size: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    gray = load_teacher_aligned_gray(Path(image_path))
    gray_pad, meta = letterbox_image(gray, (input_size, input_size), fill_value=0)
    rgb = np.stack([gray_pad, gray_pad, gray_pad], axis=-1).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(rgb).float(), meta


def logits_to_probability_map(
    logits: torch.Tensor,
    meta: Dict[str, int],
    input_size: int,
) -> Tuple[np.ndarray, float]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.ndim == 3:
        logits = logits.unsqueeze(1)

    if logits.shape[-2:] != (input_size, input_size):
        logits = F.interpolate(
            logits,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

    top = meta["pad_top"]
    left = meta["pad_left"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    prob = prob[top:top + new_h, left:left + new_w]
    prob = cv2.resize(prob, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)

    score = float(prob.max()) if prob.size > 0 else 0.0
    return prob, score


def logits_to_mask(
    logits: torch.Tensor,
    meta: Dict[str, int],
    threshold: float,
    input_size: int,
) -> Tuple[np.ndarray, float]:
    prob, score = logits_to_probability_map(logits, meta, input_size)
    return (prob >= threshold).astype(np.uint8), score


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
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


def keep_top_k_components(mask: np.ndarray, topk_components: Optional[int]) -> np.ndarray:
    if topk_components is None or int(topk_components) <= 0:
        return (mask > 0).astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    components: List[Tuple[int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, label_idx))
    if not components:
        return np.zeros_like(binary)
    components.sort(reverse=True)
    keep_labels = {label_idx for _, label_idx in components[: int(topk_components)]}
    filtered = np.zeros_like(binary)
    for label_idx in keep_labels:
        filtered[labels == label_idx] = 1
    return filtered


def apply_postprocess(
    mask: np.ndarray,
    min_area: int = 0,
    fill_holes: bool = False,
    topk_components: Optional[int] = None,
) -> np.ndarray:
    processed = remove_small_components(mask, min_area=min_area)
    processed = keep_top_k_components(processed, topk_components=topk_components)
    if fill_holes:
        processed = fill_small_holes(processed)
    return processed.astype(np.uint8)


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


def encode_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = binary_mask.astype(np.uint8)
    if mask_utils is None:
        return simple_rle_encode(binary_mask)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle, list):
        rle = rle[0]
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
    }


def empty_mask_rle(height: int, width: int) -> Dict[str, Any]:
    return encode_mask_to_rle(np.zeros((height, width), dtype=np.uint8))


def validate_text_cache(
    payload: Dict[str, Any],
    classes: List[str],
    prompt_prototypes: Optional[Dict[str, List[str]]] = None,
) -> None:
    expected_classes = list(classes)
    cache_classes = list(payload.get("classes", []))
    if set(cache_classes) != set(expected_classes):
        raise RuntimeError("text cache 与当前类别集合不一致，请重新生成")
    embeddings = payload.get("embeddings", {})
    for class_name in expected_classes:
        if class_name not in embeddings:
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}")
        embedding = embeddings[class_name]
        if not torch.is_tensor(embedding):
            raise RuntimeError(f"text cache embedding 不是 tensor: {class_name}")
        if embedding.ndim != 1:
            raise RuntimeError(f"text cache embedding 维度异常: {class_name} shape={tuple(embedding.shape)}")
    if prompt_prototypes is not None:
        expected_signature = build_prompt_prototypes_signature(prompt_prototypes, expected_classes)
        cache_signature = payload.get("prompt_prototypes_signature")
        cache_aliases = payload.get("aliases")
        if cache_signature is not None and cache_signature != expected_signature:
            raise RuntimeError("text cache 的 prompt prototype 签名与当前权重不一致，请重新生成")
        if cache_aliases is not None and cache_aliases != expected_signature:
            raise RuntimeError("text cache 的 aliases 与当前权重不一致，请重新生成")


def build_text_cache_payload(
    model,
    tokenizer,
    classes: List[str],
    prompt_prototypes: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    embeddings: Dict[str, torch.Tensor] = {}
    aliases_used: Dict[str, List[str]] = {}
    iterator = maybe_tqdm(classes, total=len(classes), desc="BuildTextCache", leave=False)
    for class_name in iterator:
        aliases = prompt_prototypes.get(class_name, [class_name])
        aliases_used[class_name] = aliases
        emb_list: List[torch.Tensor] = []
        for alias in aliases:
            encoded = tokenizer(
                alias,
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LEN,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            text_feature = model.encode_text(input_ids, attention_mask)
            text_feature = normalize_text_feature(text_feature).detach().cpu()
            emb_list.append(text_feature[0])
        prototype = normalize_text_feature(torch.stack(emb_list, dim=0).mean(dim=0, keepdim=True))[0]
        embeddings[class_name] = prototype.cpu()
    return {
        "classes": list(classes),
        "use_prompt_prototype": True,
        "embeddings": embeddings,
        "aliases": aliases_used,
        "prompt_prototypes_signature": build_prompt_prototypes_signature(prompt_prototypes, classes),
    }


def load_or_build_text_cache(
    model,
    tokenizer,
    cache_path: Optional[Path],
    classes: List[str],
    prompt_prototypes: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            validate_text_cache(payload, classes, prompt_prototypes=prompt_prototypes)
            LOGGER.info("loaded text cache: %s", cache_path)
            return payload
        except Exception as exc:
            LOGGER.warning("text cache 校验失败，将重建: %s reason=%s", cache_path, exc)
    elif cache_path is None:
        LOGGER.info("未提供本地 text cache，将基于 checkpoint prompt_prototypes 直接构建内存 cache。")
    else:
        LOGGER.info("本地 text cache 不存在，将基于 checkpoint prompt_prototypes 自动重建: %s", cache_path)

    payload = build_text_cache_payload(
        model=model,
        tokenizer=tokenizer,
        classes=classes,
        prompt_prototypes=prompt_prototypes,
        device=device,
    )
    validate_text_cache(payload, classes, prompt_prototypes=prompt_prototypes)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
            save_json(
                cache_path.with_suffix(".json"),
                {
                    "classes": classes,
                    "aliases": payload["aliases"],
                    "prompt_prototypes_signature": payload["prompt_prototypes_signature"],
                },
            )
            LOGGER.info("text cache saved: %s", cache_path)
        except Exception as exc:
            LOGGER.warning("text cache 保存失败，将继续使用内存 cache，不中断推理: %s reason=%s", cache_path, exc)
    return payload


def move_text_cache_to_device(
    payload: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    embeddings = payload.get("embeddings", {})
    payload["device_embeddings"] = {
        class_name: embedding.to(device, non_blocking=True)
        for class_name, embedding in embeddings.items()
    }
    return payload


class SubmissionTextEncoder(torch.nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        config = ChineseCLIPTextConfig.from_pretrained(str(model_dir), local_files_only=True)
        self.text_model = ChineseCLIPTextModel(config)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.text_model.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.text_model.eval()
        with torch.no_grad():
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            if pooler_output is None:
                pooler_output = last_hidden_state[:, 0, :]
        return last_hidden_state, pooler_output


class SubmissionImageEncoder(torch.nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.image_encoder = build_submission_image_encoder().eval()
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self._resize_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"输入图像应为 [B, 3, H, W]，当前形状: {x.shape}")
        if x.size(1) != 3:
            raise ValueError(f"输入图像通道数应为 3，当前为: {x.size(1)}")

        expected_size = int(self.image_encoder.img_size)
        if x.size(2) != expected_size or x.size(3) != expected_size:
            if not self._resize_warned:
                LOGGER.warning(
                    "输入尺寸 %s 与 EfficientSAM 期望尺寸 (%d, %d) 不一致，将自动插值到 backbone 输入尺寸。",
                    tuple(x.shape[-2:]),
                    expected_size,
                    expected_size,
                )
                self._resize_warned = True
            x = F.interpolate(x, size=(expected_size, expected_size), mode="bilinear", align_corners=False)

        x = (x - self.pixel_mean.to(device=x.device, dtype=x.dtype)) / self.pixel_std.to(
            device=x.device,
            dtype=x.dtype,
        )
        image_feats = self.image_encoder(x)
        if isinstance(image_feats, (tuple, list)):
            image_feats = image_feats[0]
        if not torch.is_tensor(image_feats):
            raise TypeError(f"image_encoder 输出不是 tensor，当前类型: {type(image_feats)}")
        if image_feats.ndim != 4:
            raise ValueError(f"image_encoder 输出应为 [B, C, H, W]，当前形状: {image_feats.shape}")
        return image_feats


class FiLMFusionDecoder(torch.nn.Module):
    def __init__(self, image_dim: int = 256, text_dim: int = 768):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.film_projection = torch.nn.Linear(text_dim, image_dim * 2)
        torch.nn.init.zeros_(self.film_projection.weight)
        torch.nn.init.zeros_(self.film_projection.bias)
        self.conv_after_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(image_dim, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.mask_head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, image_features: torch.Tensor, text_global_features: torch.Tensor, target_size) -> torch.Tensor:
        if image_features.ndim != 4:
            raise ValueError(f"image_features 应为 [B, C, H, W]，当前形状: {image_features.shape}")
        if text_global_features.ndim == 3:
            text_global_features = text_global_features[:, 0, :]
        if text_global_features.ndim != 2:
            raise ValueError(f"text_global_features 应为 [B, D]，当前形状: {text_global_features.shape}")
        if image_features.size(0) != text_global_features.size(0):
            raise ValueError(f"图像 batch 和文本 batch 不一致: {image_features.size(0)} vs {text_global_features.size(0)}")
        if image_features.size(1) != self.image_dim:
            raise ValueError(f"image_features 通道数应为 {self.image_dim}，当前为 {image_features.size(1)}")
        if torch.is_tensor(target_size):
            target_size = target_size.detach().cpu().tolist()
        if isinstance(target_size, (list, tuple)):
            target_size = (int(target_size[0]), int(target_size[1]))
        else:
            raise ValueError(f"target_size 格式错误，应为 (H, W)，当前为: {target_size}")

        film_params = self.film_projection(text_global_features)
        gamma, beta = torch.split(film_params, image_features.size(1), dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fused_features = image_features * (1.0 + gamma) + beta

        x = self.conv_after_fusion(fused_features)
        x = self.mask_head(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class ZeroInitResidualRefineHead(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 4)
        self.conv1 = torch.nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.act1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.act2 = torch.nn.GELU()
        self.out = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, bias=True)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, coarse_logits: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != coarse_logits.shape[-2:]:
            images = F.interpolate(images, size=coarse_logits.shape[-2:], mode="bilinear", align_corners=False)
        gray = images.mean(dim=1, keepdim=True)
        gray_min = gray.amin(dim=(-2, -1), keepdim=True)
        gray_max = gray.amax(dim=(-2, -1), keepdim=True)
        gray = (gray - gray_min) / (gray_max - gray_min).clamp_min(1e-6)
        coarse_prob = torch.sigmoid(coarse_logits)
        refine_input = torch.cat([gray, coarse_logits, coarse_prob], dim=1)
        hidden = self.act1(self.conv1(refine_input))
        hidden = self.act2(self.conv2(hidden))
        return self.out(hidden)


def validate_refine_head_structure(model: torch.nn.Module) -> Dict[str, Any]:
    refine_head = getattr(model, "refine_head", None)
    if refine_head is None:
        raise RuntimeError("model 缺少 refine_head。")
    if getattr(refine_head.conv1, "in_channels", None) != 3:
        raise RuntimeError(f"refine_head.conv1.in_channels 必须为 3，当前={refine_head.conv1.in_channels}")
    if int(torch.count_nonzero(refine_head.out.weight.detach()).item()) != 0:
        raise RuntimeError("refine_head.out.weight 不是 zero-init。")
    if int(torch.count_nonzero(refine_head.out.bias.detach()).item()) != 0:
        raise RuntimeError("refine_head.out.bias 不是 zero-init。")
    return {
        "conv1_in_channels": int(refine_head.conv1.in_channels),
        "out_weight_all_zero": True,
        "out_bias_all_zero": True,
    }


class ESAMCCLIPModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer_dir: str,
        freeze_image: bool = True,
        freeze_text: bool = True,
        use_refine_head: bool = False,
        refine_head_hidden_dim: int = 16,
    ):
        super().__init__()
        self.img_encoder = SubmissionImageEncoder(freeze=freeze_image)
        self.txt_encoder = SubmissionTextEncoder(model_dir=tokenizer_dir)
        self.decoder = FiLMFusionDecoder(image_dim=256, text_dim=768)
        self.refine_head = ZeroInitResidualRefineHead(hidden_dim=refine_head_hidden_dim)
        self.use_refine_head = bool(use_refine_head)
        self._refine_head_structure_info = validate_refine_head_structure(self)
        if freeze_text:
            for param in self.txt_encoder.parameters():
                param.requires_grad = False
            self.txt_encoder.eval()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.img_encoder(images)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, pooled = self.txt_encoder(input_ids, attention_mask)
        return pooled

    def set_refine_head_enabled(self, enabled: bool) -> None:
        self.use_refine_head = bool(enabled)

    def decode(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        target_size,
        images: Optional[torch.Tensor] = None,
        use_refine_head: Optional[bool] = None,
    ) -> torch.Tensor:
        coarse_logits = self.decoder(
            image_features=image_features,
            text_global_features=text_features,
            target_size=target_size,
        )
        refine_enabled = self.use_refine_head if use_refine_head is None else bool(use_refine_head)
        if not refine_enabled:
            return coarse_logits
        if images is None:
            raise RuntimeError("use_refine_head=True 但 decode 未收到 images。")
        residual_logits = self.refine_head(coarse_logits=coarse_logits, images=images)
        return coarse_logits + residual_logits


def load_state_dict_flexible(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict")


KEY_PREFIX_REMAP = {
    "image_encoder.": "img_encoder.",
    "text_encoder.": "txt_encoder.",
    "fusion_decoder.": "decoder.",
}


def strip_common_prefixes(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("model."):
        key = key[len("model."):]
    return key


def remap_key(key: str) -> str:
    key = strip_common_prefixes(key)
    for old_prefix, new_prefix in KEY_PREFIX_REMAP.items():
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def load_checkpoint_payload(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint 不是 dict，无法读取 metadata: {checkpoint_path}")
    return checkpoint


def load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint: Dict[str, Any],
    checkpoint_path: str,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    state_dict = load_state_dict_flexible(checkpoint)
    model_state = model.state_dict()
    matched_state: Dict[str, torch.Tensor] = {}
    matched_keys: List[str] = []
    decoder_matched_keys: List[str] = []
    refine_head_matched_keys: List[str] = []
    unexpected_keys: List[str] = []
    shape_mismatch_keys: List[str] = []

    for raw_key, value in state_dict.items():
        mapped_key = remap_key(raw_key)
        if mapped_key not in model_state:
            unexpected_keys.append(mapped_key)
            continue
        if model_state[mapped_key].shape != value.shape:
            shape_mismatch_keys.append(
                f"{mapped_key}: ckpt={tuple(value.shape)} model={tuple(model_state[mapped_key].shape)}"
            )
            continue
        matched_state[mapped_key] = value
        matched_keys.append(mapped_key)
        if mapped_key.startswith("decoder."):
            decoder_matched_keys.append(mapped_key)
        elif mapped_key.startswith("refine_head."):
            refine_head_matched_keys.append(mapped_key)

    missing_keys = [key for key in model_state.keys() if key not in matched_state]
    unexpected_keys.extend(shape_mismatch_keys)
    metadata_use_refine_head = bool(checkpoint.get("use_refine_head", False))
    checkpoint_has_refine_keys = any(remap_key(str(raw_key)).startswith("refine_head.") for raw_key in state_dict.keys())
    checkpoint_use_refine_head = bool(metadata_use_refine_head or checkpoint_has_refine_keys)
    checkpoint_claims_refine = bool(checkpoint_use_refine_head or checkpoint_has_refine_keys)
    refine_shape_mismatch_keys = [key for key in shape_mismatch_keys if key.startswith("refine_head.")]
    missing_refine_keys = [key for key in missing_keys if key.startswith("refine_head.")]
    model_use_refine_head = bool(getattr(model, "use_refine_head", False))
    if not metadata_use_refine_head and checkpoint_has_refine_keys:
        LOGGER.warning("checkpoint contains refine_head keys but metadata use_refine_head=False, enabling refine_head automatically")
    if len(decoder_matched_keys) == 0:
        raise RuntimeError("decoder_matched_keys=0，说明 decoder 没有成功加载，不能提交")
    if checkpoint_claims_refine:
        if len(refine_head_matched_keys) == 0:
            raise RuntimeError("checkpoint 声明/包含 refine_head，但推理时 refine_head_matched_keys=0，拒绝提交")
        if refine_shape_mismatch_keys:
            raise RuntimeError(
                "checkpoint refine_head 形状不匹配，拒绝提交: "
                f"{refine_shape_mismatch_keys[:10]}"
            )
    model.load_state_dict(matched_state, strict=False)
    if checkpoint_use_refine_head:
        if not bool(getattr(model, "use_refine_head", False)):
            raise RuntimeError("checkpoint_use_refine_head=True，但 model.use_refine_head=False，拒绝提交")
        if len(refine_head_matched_keys) <= 0:
            raise RuntimeError("checkpoint_use_refine_head=True，但 refine_head_matched_keys=0，拒绝提交")
        if refine_shape_mismatch_keys:
            raise RuntimeError("checkpoint_use_refine_head=True，但 refine_head 存在 shape mismatch，拒绝提交")
    model._last_checkpoint_load_info = {
        "checkpoint_path": str(checkpoint_path),
        "matched_keys": matched_keys,
        "decoder_matched_keys": decoder_matched_keys,
        "refine_head_matched_keys": refine_head_matched_keys,
        "missing_keys": missing_keys,
        "missing_refine_keys": missing_refine_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
        "refine_shape_mismatch_keys": refine_shape_mismatch_keys,
        "checkpoint_use_refine_head": checkpoint_use_refine_head,
        "metadata_use_refine_head": metadata_use_refine_head,
        "checkpoint_has_refine_keys": checkpoint_has_refine_keys,
        "checkpoint_claims_refine": checkpoint_claims_refine,
        "model_use_refine_head": model_use_refine_head,
    }
    LOGGER.info("checkpoint loaded: %s", checkpoint_path)
    LOGGER.info(
        "matched_keys=%d decoder_matched_keys=%d refine_head_matched_keys=%d missing_keys=%d unexpected_keys=%d",
        len(matched_state),
        len(decoder_matched_keys),
        len(refine_head_matched_keys),
        len(missing_keys),
        len(unexpected_keys),
    )
    LOGGER.info(
        "checkpoint_use_refine_head=%s checkpoint_has_refine_keys=%s checkpoint_claims_refine=%s model_use_refine_head=%s refine_head_matched_count=%d refine_shape_mismatch_count=%d",
        checkpoint_use_refine_head,
        checkpoint_has_refine_keys,
        checkpoint_claims_refine,
        model_use_refine_head,
        len(refine_head_matched_keys),
        len(refine_shape_mismatch_keys),
    )
    if not checkpoint_claims_refine and missing_refine_keys and model_use_refine_head:
        LOGGER.info("old checkpoint has no refine_head, keep zero-init refine_head")
    if missing_refine_keys:
        LOGGER.info("missing_refine_keys_preview=%s", missing_refine_keys[:20])
    if refine_shape_mismatch_keys:
        LOGGER.info("refine_shape_mismatch_keys_preview=%s", refine_shape_mismatch_keys[:20])
    return checkpoint, missing_keys, unexpected_keys


def resolve_checkpoint_use_refine_head(checkpoint: Dict[str, Any]) -> bool:
    if bool(checkpoint.get("use_refine_head", False)):
        return True
    state_dict = load_state_dict_flexible(checkpoint)
    return any(remap_key(str(raw_key)).startswith("refine_head.") for raw_key in state_dict.keys())


def normalize_decoder_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = remap_key(str(raw_key))
        if key.startswith("decoder."):
            key = key[len("decoder.") :]
        normalized[key] = value
    return normalized


def validate_router_route_map(route_map: Dict[str, str], classes: List[str]) -> Dict[str, str]:
    normalized = {str(key): str(value) for key, value in route_map.items()}
    missing_classes = [class_name for class_name in classes if class_name not in normalized]
    if missing_classes:
        raise RuntimeError(f"router route_map 缺少类别: {missing_classes}")
    invalid_routes = {class_name: route for class_name, route in normalized.items() if route not in {"lite", "stage2"}}
    if invalid_routes:
        raise RuntimeError(f"router route_map 存在非法 route: {invalid_routes}")
    return normalized


def resolve_prompt_route(
    mapped_class: Optional[str],
    route_map: Optional[Dict[str, str]],
) -> str:
    if mapped_class is None or route_map is None:
        return "lite"
    return route_map.get(mapped_class, "lite")


def load_decoder_module_state(
    decoder_module: torch.nn.Module,
    decoder_state_dict: Dict[str, torch.Tensor],
    decoder_name: str,
) -> None:
    normalized = normalize_decoder_state_dict_keys(decoder_state_dict)
    if len(normalized) == 0:
        raise RuntimeError(f"{decoder_name} load_state_dict strict=True failed: key_count=0")
    preview_keys = list(normalized.keys())[:5]
    LOGGER.info("%s key_count=%d first_keys=%s", decoder_name, len(normalized), preview_keys)
    try:
        decoder_module.load_state_dict(normalized, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{decoder_name} load_state_dict strict=True failed: {exc}") from exc


def validate_router_runtime_configs(
    classes: List[str],
    route_map: Dict[str, str],
    thresholds: Dict[str, float],
    postprocess_cfg: Dict[str, Dict[str, Any]],
) -> None:
    missing_route = [class_name for class_name in classes if class_name not in route_map]
    missing_threshold = [class_name for class_name in classes if class_name not in thresholds]
    missing_post = [class_name for class_name in classes if class_name not in postprocess_cfg]
    if missing_route or missing_threshold or missing_post:
        raise RuntimeError(
            "router config incomplete: "
            f"missing_route={missing_route}, "
            f"missing_threshold={missing_threshold}, "
            f"missing_postprocess={missing_post}"
        )


def build_router_runtime(
    model: ESAMCCLIPModel,
    checkpoint: Dict[str, Any],
    classes: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    router_enabled = bool(checkpoint.get("router_enabled", False))
    router_info: Dict[str, Any] = {
        "enabled": router_enabled,
        "router_type": str(checkpoint.get("router_type", "")) if router_enabled else None,
        "route_mode": str(checkpoint.get("route_mode", "")) if router_enabled else None,
        "route_map": None,
        "decoder_lite": model.decoder,
        "decoder_stage2": None,
        "base_model_params": None,
        "stage2_decoder_params": 0,
        "total_runtime_params": None,
    }
    if not router_enabled:
        return router_info

    route_map_raw = checkpoint.get("route_map")
    if not isinstance(route_map_raw, dict):
        raise RuntimeError("router_enabled=True 但 checkpoint 缺少 route_map")
    route_map = validate_router_route_map(route_map_raw, classes)

    decoder_lite = model.decoder
    decoder_lite_state_dict = checkpoint.get("decoder_lite_state_dict")
    if isinstance(decoder_lite_state_dict, dict):
        load_decoder_module_state(decoder_lite, decoder_lite_state_dict, "decoder_lite")
    decoder_stage2_state_dict = checkpoint.get("decoder_stage2_state_dict")
    if not isinstance(decoder_stage2_state_dict, dict) or not decoder_stage2_state_dict:
        raise RuntimeError("router_enabled=True 但 checkpoint 缺少 decoder_stage2_state_dict")
    decoder_stage2 = copy.deepcopy(model.decoder)
    load_decoder_module_state(decoder_stage2, decoder_stage2_state_dict, "decoder_stage2")

    decoder_lite.to(device).eval()
    decoder_stage2.to(device).eval()

    base_model_state_dict = load_state_dict_flexible(checkpoint)
    base_model_params = count_tensor_params(base_model_state_dict)
    stage2_decoder_params = count_tensor_params(decoder_stage2_state_dict)
    total_runtime_params = int(base_model_params + stage2_decoder_params)
    LOGGER.info("router_enabled=%s", router_enabled)
    LOGGER.info("router_type=%s", checkpoint.get("router_type"))
    LOGGER.info("route_mode=%s", checkpoint.get("route_mode"))
    LOGGER.info("route_map=%s", route_map)
    LOGGER.info("base_model_params=%d", base_model_params)
    LOGGER.info("stage2_decoder_params=%d", stage2_decoder_params)
    LOGGER.info("total_runtime_params=%d", total_runtime_params)
    if total_runtime_params > DEFAULT_ROUTER_MAX_RUNTIME_PARAMS:
        raise RuntimeError(
            f"router runtime params 超限: total={total_runtime_params} max={DEFAULT_ROUTER_MAX_RUNTIME_PARAMS}"
        )
    LOGGER.info("Router selfcheck:")
    for class_name in classes:
        LOGGER.info("%s -> %s", class_name, route_map.get(class_name))

    router_info.update(
        {
            "route_map": route_map,
            "decoder_lite": decoder_lite,
            "decoder_stage2": decoder_stage2,
            "base_model_params": int(base_model_params),
            "stage2_decoder_params": int(stage2_decoder_params),
            "total_runtime_params": int(total_runtime_params),
        }
    )
    return router_info


def map_prompt_to_known_class(prompt_text: str, prompt_aliases: Dict[str, str]) -> Optional[str]:
    return prompt_aliases.get(normalize_prompt_key(prompt_text))


def is_short_english_alias(alias_norm: str) -> bool:
    if not alias_norm:
        return False
    if any(ord(ch) > 127 for ch in alias_norm):
        return False
    compact = alias_norm.replace(" ", "")
    return len(compact) < 3


def contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in str(text))


def token_sequence_in_prompt(prompt_norm: str, alias_norm: str) -> bool:
    prompt_tokens = prompt_norm.split()
    alias_tokens = alias_norm.split()
    if not prompt_tokens or not alias_tokens or len(alias_tokens) > len(prompt_tokens):
        return False
    window = len(alias_tokens)
    for start in range(0, len(prompt_tokens) - window + 1):
        if prompt_tokens[start:start + window] == alias_tokens:
            return True
    return False


def extract_chinese_target_phrase(prompt_text: str) -> Optional[str]:
    prompt_norm = normalize_prompt_key(prompt_text)
    if "的" not in prompt_norm:
        return None
    target_phrase = prompt_norm.rsplit("的", 1)[-1].strip()
    return target_phrase or None


def extract_english_target_phrase(prompt_norm: str) -> Optional[str]:
    prompt_norm = normalize_prompt_key(prompt_norm)
    if not prompt_norm:
        return None
    if " of " in prompt_norm:
        target_phrase = prompt_norm.split(" of ", 1)[0].strip()
        if target_phrase:
            return target_phrase
    for relation_token in ENGLISH_TARGET_RELATIONS:
        if relation_token in prompt_norm:
            target_phrase = prompt_norm.split(relation_token, 1)[0].strip()
            if target_phrase:
                return target_phrase
    return None


def collect_soft_match_candidates(
    prompt_norm: str,
    alias_to_class: Dict[str, str],
    classes: List[str],
    complex_aliases: Optional[Dict[str, str]] = None,
) -> List[Tuple[int, int, int, int, str]]:
    rare_classes = RARE_FALLBACK_CLASSES.intersection(set(classes))
    complex_aliases = complex_aliases or {}
    candidates: List[Tuple[int, int, int, int, str]] = []
    for alias_norm, class_name in alias_to_class.items():
        if not alias_norm or alias_norm == prompt_norm:
            continue
        if is_short_english_alias(alias_norm):
            continue
        if contains_non_ascii(alias_norm):
            matched = alias_norm in prompt_norm
        else:
            matched = token_sequence_in_prompt(prompt_norm, alias_norm)
        if not matched:
            continue
        is_complex = 1 if alias_norm in complex_aliases else 0
        token_count = len(alias_norm.split())
        rare_priority = 1 if class_name in rare_classes else 0
        candidates.append((is_complex, token_count, len(alias_norm), rare_priority, class_name))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return candidates


def match_target_phrase_to_class(
    target_phrase: str,
    prompt_aliases: Dict[str, str],
    complex_prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    target_norm = normalize_prompt_key(target_phrase)
    if not target_norm:
        return None
    exact_complex = complex_prompt_aliases.get(target_norm)
    if exact_complex is not None:
        return exact_complex
    exact = prompt_aliases.get(target_norm)
    if exact is not None:
        return exact
    candidates = collect_soft_match_candidates(
        prompt_norm=target_norm,
        alias_to_class=prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    return candidates[0][4] if candidates else None


def map_prompt_to_known_class_soft(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    exact = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if exact is not None:
        return exact
    prompt_norm = normalize_prompt_key(prompt_text)
    if not prompt_norm:
        return None
    complex_prompt_aliases = build_complex_prompt_alias_map(classes)
    candidates = collect_soft_match_candidates(
        prompt_norm=prompt_norm,
        alias_to_class=prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    return candidates[0][4] if candidates else None


def map_prompt_to_known_class_target_soft(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    exact = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if exact is not None:
        return exact
    prompt_norm = normalize_prompt_key(prompt_text)
    if not prompt_norm:
        return None
    complex_prompt_aliases = build_complex_prompt_alias_map(classes)
    complex_candidates = collect_soft_match_candidates(
        prompt_norm=prompt_norm,
        alias_to_class=complex_prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    if complex_candidates:
        return complex_candidates[0][4]
    chinese_target = extract_chinese_target_phrase(prompt_text)
    if chinese_target:
        matched = match_target_phrase_to_class(chinese_target, prompt_aliases, complex_prompt_aliases, classes)
        if matched is not None:
            return matched
    english_target = extract_english_target_phrase(prompt_norm)
    if english_target:
        matched = match_target_phrase_to_class(english_target, prompt_aliases, complex_prompt_aliases, classes)
        if matched is not None:
            return matched
    return map_prompt_to_known_class_soft(prompt_text, prompt_aliases, classes)


def run_prompt_match_selfcheck(
    prompt_aliases: Dict[str, str],
    classes: List[str],
    prompt_match_mode: str,
) -> List[Tuple[str, Optional[str]]]:
    return [
        (prompt_text, resolve_prompt_mapped_class(prompt_text, prompt_aliases, classes, prompt_match_mode))
        for prompt_text, _ in PROMPT_MATCH_SELFTEST_CASES
    ]


def resolve_prompt_mapped_class(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
    prompt_match_mode: str = DEFAULT_PROMPT_MATCH_MODE,
) -> Optional[str]:
    if prompt_match_mode == "target_soft":
        return map_prompt_to_known_class_target_soft(prompt_text, prompt_aliases, classes)
    if prompt_match_mode == "soft":
        return map_prompt_to_known_class_soft(prompt_text, prompt_aliases, classes)
    return map_prompt_to_known_class(prompt_text, prompt_aliases)


def resolve_prompt_inference_config(
    checkpoint: Optional[Dict[str, Any]],
    cli_prompt_fusion_mode: Optional[str],
    cli_raw_prompt_weight: Optional[float],
    cli_prompt_match_mode: Optional[str],
) -> Tuple[str, float, str]:
    prompt_fusion_mode = cli_prompt_fusion_mode
    if prompt_fusion_mode is None:
        prompt_fusion_mode = (
            str(checkpoint.get("prompt_fusion_mode"))
            if isinstance(checkpoint, dict) and checkpoint.get("prompt_fusion_mode") is not None
            else DEFAULT_PROMPT_FUSION_MODE
        )
    raw_prompt_weight = cli_raw_prompt_weight
    if raw_prompt_weight is None:
        raw_prompt_weight = (
            float(checkpoint.get("raw_prompt_weight"))
            if isinstance(checkpoint, dict) and checkpoint.get("raw_prompt_weight") is not None
            else DEFAULT_RAW_PROMPT_WEIGHT
        )
    prompt_match_mode = cli_prompt_match_mode
    if prompt_match_mode is None:
        prompt_match_mode = (
            str(checkpoint.get("prompt_match_mode"))
            if isinstance(checkpoint, dict) and checkpoint.get("prompt_match_mode") is not None
            else DEFAULT_PROMPT_MATCH_MODE
        )
    prompt_fusion_mode = prompt_fusion_mode if prompt_fusion_mode in {"prototype", "raw", "blend"} else DEFAULT_PROMPT_FUSION_MODE
    prompt_match_mode = prompt_match_mode if prompt_match_mode in {"exact", "soft", "target_soft"} else DEFAULT_PROMPT_MATCH_MODE
    raw_prompt_weight = min(max(float(raw_prompt_weight), 0.0), 1.0)
    return prompt_fusion_mode, raw_prompt_weight, prompt_match_mode


def build_text_feature_for_prompt(
    model: ESAMCCLIPModel,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    device: torch.device,
    prompt_feature_cache: Optional[Dict[str, Tuple[torch.Tensor, Optional[str]]]] = None,
    prompt_fusion_mode: str = DEFAULT_PROMPT_FUSION_MODE,
    raw_prompt_weight: float = DEFAULT_RAW_PROMPT_WEIGHT,
    prompt_match_mode: str = DEFAULT_PROMPT_MATCH_MODE,
) -> Tuple[torch.Tensor, Optional[str]]:
    prompt_text = str(prompt_text).strip()
    raw_prompt_weight = min(max(float(raw_prompt_weight), 0.0), 1.0)
    cache_key = f"{prompt_text}||{prompt_fusion_mode}||{raw_prompt_weight:.6f}||{prompt_match_mode}"
    if prompt_feature_cache is not None and cache_key in prompt_feature_cache:
        cached_feature, cached_mapped_class = prompt_feature_cache[cache_key]
        return cached_feature.to(device, non_blocking=True), cached_mapped_class

    mapped_class = resolve_prompt_mapped_class(
        prompt_text=prompt_text,
        prompt_aliases=prompt_aliases,
        classes=list(text_cache_payload.get("classes", DEFAULT_CLASSES)),
        prompt_match_mode=prompt_match_mode,
    )
    device_embeddings = text_cache_payload.get("device_embeddings", {})

    def encode_raw_prompt_feature(text: str) -> torch.Tensor:
        raw_cache_key = f"__raw__::{text}"
        if prompt_feature_cache is not None and raw_cache_key in prompt_feature_cache:
            cached_feature, _ = prompt_feature_cache[raw_cache_key]
            return cached_feature.to(device, non_blocking=True)
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        feature = model.encode_text(input_ids, attention_mask).detach()
        feature = normalize_text_feature(feature)
        if feature.ndim == 2:
            feature = feature[0]
        feature = normalize_text_feature(feature.unsqueeze(0))[0]
        if prompt_feature_cache is not None:
            prompt_feature_cache[raw_cache_key] = (feature.detach().cpu(), None)
        return feature

    if mapped_class is not None and mapped_class in device_embeddings:
        prototype_feature = device_embeddings[mapped_class]
        if prototype_feature.ndim == 2:
            prototype_feature = prototype_feature[0]
        prototype_feature = normalize_text_feature(prototype_feature.unsqueeze(0))[0]
        if prompt_fusion_mode == "prototype":
            final_feature = prototype_feature
        elif prompt_fusion_mode == "raw":
            final_feature = encode_raw_prompt_feature(prompt_text)
        else:
            raw_prompt_feature = encode_raw_prompt_feature(prompt_text)
            blended = (1.0 - raw_prompt_weight) * prototype_feature + raw_prompt_weight * raw_prompt_feature
            final_feature = normalize_text_feature(blended.unsqueeze(0))[0]
        resolved = normalize_text_feature(final_feature.unsqueeze(0))[0], mapped_class
        if prompt_feature_cache is not None:
            prompt_feature_cache[cache_key] = (resolved[0].detach().cpu(), resolved[1])
        return resolved

    feature = encode_raw_prompt_feature(prompt_text)
    resolved = feature, None
    if prompt_feature_cache is not None:
        prompt_feature_cache[cache_key] = (resolved[0].detach().cpu(), resolved[1])
    return resolved


def get_prompt_threshold(
    prompt_text: str,
    mapped_class: Optional[str],
    thresholds: Dict[str, float],
    default_threshold: float,
) -> float:
    key = mapped_class if mapped_class is not None else prompt_text
    return float(thresholds.get(key, thresholds.get(str(key), default_threshold)))


def get_prompt_postprocess(
    prompt_text: str,
    mapped_class: Optional[str],
    postprocess_cfg: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    key = mapped_class if mapped_class is not None else prompt_text
    return postprocess_cfg.get(key, {"min_area": 0, "fill_holes": False})


def apply_rare_empty_fallback(
    prob: np.ndarray,
    class_name: Optional[str],
    threshold: float,
    base_post_cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    if class_name not in RARE_FALLBACK_CLASSES:
        return None
    fallback_cfg = RARE_EMPTY_FALLBACK_CFG.get(class_name or "")
    if fallback_cfg is None:
        return None
    fallback_threshold = max(0.0, float(threshold) - RARE_EMPTY_FALLBACK_DELTA)
    fallback_mask = (prob >= fallback_threshold).astype(np.uint8)
    fallback_mask = apply_postprocess(
        fallback_mask,
        min_area=int(fallback_cfg.get("min_area", 0)),
        fill_holes=bool(base_post_cfg.get("fill_holes", False)),
        topk_components=int(fallback_cfg.get("topk_components", 0)),
    )
    if int(fallback_mask.sum()) <= 0:
        return None
    return fallback_mask


def choose_tokenizer_dir(model_dir: Path, tokenizer_path: Optional[str]) -> str:
    if tokenizer_path:
        return resolve_path(tokenizer_path) or tokenizer_path
    candidates = [
        model_dir,
        model_dir / "chinese_clip",
    ]
    required_names = {"config.json"}
    optional_tokenizer_names = {"vocab.txt", "tokenizer.json"}
    for candidate in candidates:
        if not candidate.exists():
            continue
        existing = {item.name for item in candidate.iterdir() if item.is_file()}
        if required_names.issubset(existing) and existing.intersection(optional_tokenizer_names):
            return str(candidate)
    raise FileNotFoundError(
        f"未找到 tokenizer 目录。请确保 {model_dir} 下至少包含 config.json 和 vocab.txt/tokenizer.json。"
    )


def resolve_submission_resources(
    model_dir: str,
    checkpoint_path: Optional[str],
    tokenizer_path: Optional[str],
    text_cache_path: Optional[str],
) -> Tuple[str, str, str, Optional[str]]:
    resolved_model_dir = resolve_path(model_dir) or model_dir
    model_dir_path = Path(resolved_model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {resolved_model_dir}")

    resolved_checkpoint = resolve_path(checkpoint_path) if checkpoint_path else None
    if resolved_checkpoint is None:
        resolved_checkpoint = str(model_dir_path / "sam3.pt")
    if not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(f"缺少模型权重文件: {resolved_checkpoint}")

    resolved_tokenizer = choose_tokenizer_dir(model_dir_path, tokenizer_path)

    resolved_cache = resolve_path(text_cache_path) if text_cache_path else None
    return resolved_model_dir, resolved_checkpoint, resolved_tokenizer, resolved_cache


def resolve_checkpoint_classes(checkpoint: Dict[str, Any]) -> List[str]:
    return list(checkpoint.get("classes", DEFAULT_CLASSES))


def resolve_checkpoint_prompt_prototypes(
    checkpoint: Dict[str, Any],
    classes: List[str],
) -> Dict[str, List[str]]:
    merged_prototypes = normalize_prompt_prototypes(DEFAULT_PROMPT_PROTOTYPES, classes)
    if isinstance(checkpoint, dict):
        ckpt_prototypes = checkpoint.get("prompt_prototypes")
        if isinstance(ckpt_prototypes, dict):
            for class_name, aliases in ckpt_prototypes.items():
                if str(class_name) not in merged_prototypes:
                    continue
                merged_prototypes[str(class_name)] = [str(alias) for alias in aliases]
            return normalize_prompt_prototypes(merged_prototypes, classes)
    return merged_prototypes


def resolve_checkpoint_thresholds(checkpoint: Dict[str, Any]) -> Dict[str, float]:
    raw_thresholds = (
        checkpoint.get("val_thresholds")
        or checkpoint.get("prompt_thresholds")
        or checkpoint.get("thresholds")
        or checkpoint.get("class_mask_thresholds")
        or {}
    )
    thresholds = {str(key): float(value) for key, value in DEFAULT_THRESHOLDS.items()}
    thresholds.update({str(key): float(value) for key, value in raw_thresholds.items()})
    return thresholds


def resolve_checkpoint_postprocess(checkpoint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_postprocess = (
        checkpoint.get("postprocess_cfg")
        or checkpoint.get("postprocess")
        or {}
    )
    normalized: Dict[str, Dict[str, Any]] = {
        str(key): {
            "min_area": int(cfg.get("min_area", 0)),
            "fill_holes": bool(cfg.get("fill_holes", False)),
            **({"topk_components": int(cfg.get("topk_components", 0))} if cfg.get("topk_components") is not None else {}),
        }
        for key, cfg in DEFAULT_POSTPROCESS.items()
    }
    flat_min_area = checkpoint.get("min_area") if isinstance(checkpoint, dict) else None
    if isinstance(flat_min_area, dict):
        for key, value in flat_min_area.items():
            class_name = str(key)
            normalized.setdefault(class_name, {"min_area": 0, "fill_holes": False})
            normalized[class_name]["min_area"] = int(value)
    for key, cfg in raw_postprocess.items():
        class_name = str(key)
        normalized_cfg = {
            "min_area": int(cfg.get("min_area", 0)),
            "fill_holes": bool(cfg.get("fill_holes", False)),
        }
        if "topk_components" in cfg and cfg.get("topk_components") is not None:
            normalized_cfg["topk_components"] = int(cfg.get("topk_components", 0))
        normalized[class_name] = normalized_cfg
    return normalized


def load_model(
    model_dir: str,
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    text_cache_path: Optional[str] = None,
) -> Tuple[ESAMCCLIPModel, Any, Dict[str, Any], int, Dict[str, Any], List[str], Dict[str, str], Dict[str, Any]]:
    ensure_hf_cache()
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, checkpoint_path, tokenizer_path, text_cache_path = resolve_submission_resources(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        text_cache_path=text_cache_path,
    )

    checkpoint_preview = load_checkpoint_payload(checkpoint_path, device)
    classes = resolve_checkpoint_classes(checkpoint_preview)
    prompt_prototypes = resolve_checkpoint_prompt_prototypes(checkpoint_preview, classes)
    checkpoint_has_refine_keys = any(
        remap_key(str(raw_key)).startswith("refine_head.")
        for raw_key in load_state_dict_flexible(checkpoint_preview).keys()
    )
    checkpoint_use_refine_head = resolve_checkpoint_use_refine_head(checkpoint_preview)
    checkpoint_claims_refine = bool(checkpoint_use_refine_head or checkpoint_has_refine_keys)
    prompt_prototypes_source = (
        str(checkpoint_preview.get("prompt_prototypes_source", "default"))
        if isinstance(checkpoint_preview, dict)
        else "default"
    )
    LOGGER.info("prompt_prototypes_source=%s", prompt_prototypes_source)
    prompt_aliases = build_prompt_aliases(prompt_prototypes, classes)

    model = ESAMCCLIPModel(
        tokenizer_dir=tokenizer_path,
        freeze_image=True,
        freeze_text=True,
        use_refine_head=checkpoint_use_refine_head,
        refine_head_hidden_dim=int(checkpoint_preview.get("refine_head_hidden_dim", 16)) if isinstance(checkpoint_preview, dict) else 16,
    ).to(device)

    checkpoint, missing_keys, unexpected_keys = load_checkpoint_flexible(
        model,
        checkpoint_preview,
        checkpoint_path,
    )
    model.set_refine_head_enabled(checkpoint_use_refine_head)
    load_info = getattr(model, "_last_checkpoint_load_info", {})
    LOGGER.info(
        "checkpoint_use_refine_head=%s checkpoint_has_refine_keys=%s checkpoint_claims_refine=%s model_use_refine_head=%s refine_head_matched_keys_count=%d",
        checkpoint_use_refine_head,
        checkpoint_has_refine_keys,
        checkpoint_claims_refine,
        bool(getattr(model, "use_refine_head", False)),
        len(load_info.get("refine_head_matched_keys", [])),
    )
    if missing_keys:
        LOGGER.warning("checkpoint missing_keys=%d，前10项: %s", len(missing_keys), missing_keys[:10])
    if unexpected_keys:
        LOGGER.warning("checkpoint unexpected_keys=%d，前10项: %s", len(unexpected_keys), unexpected_keys[:10])

    tokenizer = load_tokenizer(tokenizer_path)
    cache_path_obj = Path(text_cache_path) if text_cache_path else None
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=cache_path_obj,
        classes=classes,
        prompt_prototypes=prompt_prototypes,
        device=device,
    )
    text_cache_payload = move_text_cache_to_device(text_cache_payload, device)
    text_cache_payload["prompt_prototypes_source"] = prompt_prototypes_source
    text_cache_payload["cache_path"] = str(cache_path_obj) if cache_path_obj is not None else None
    router_info = build_router_runtime(model, checkpoint, classes, device)
    model.eval()
    model_input_size = int(checkpoint.get("img_size", DEFAULT_SUBMIT_IMG_SIZE)) if isinstance(checkpoint, dict) else DEFAULT_SUBMIT_IMG_SIZE
    return model, tokenizer, text_cache_payload, model_input_size, checkpoint, classes, prompt_aliases, router_info


@torch.inference_mode()
def do_inference(
    image_path: str,
    text_prompts: List[str],
    model: ESAMCCLIPModel,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    model_input_size: int,
    thresholds: Dict[str, float],
    postprocess_cfg: Dict[str, Dict[str, Any]],
    prompt_aliases: Dict[str, str],
    default_mask_threshold: float,
    rare_empty_fallback: bool = False,
    prompt_feature_cache: Optional[Dict[str, Tuple[torch.Tensor, Optional[str]]]] = None,
    prompt_fusion_mode: str = DEFAULT_PROMPT_FUSION_MODE,
    raw_prompt_weight: float = DEFAULT_RAW_PROMPT_WEIGHT,
    prompt_match_mode: str = DEFAULT_PROMPT_MATCH_MODE,
    router_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], int, int, Dict[str, Any]]:
    device = next(model.parameters()).device
    image_tensor, meta = preprocess_image(image_path, model_input_size)
    image_embedding = model.encode_image(image_tensor.unsqueeze(0).to(device, non_blocking=True))
    results_by_prompt: Dict[str, Dict[str, Any]] = {}
    debug_stats: Dict[str, Any] = {
        "fallback_hit_count": 0,
        "fallback_by_class": defaultdict(int),
    }

    for start in range(0, len(text_prompts), PROMPT_BATCH_SIZE):
        prompt_batch = text_prompts[start:start + PROMPT_BATCH_SIZE]
        feature_list: List[torch.Tensor] = []
        mapped_classes: List[Optional[str]] = []
        routes: List[str] = []
        for prompt_text in prompt_batch:
            feature, mapped_class = build_text_feature_for_prompt(
                model=model,
                tokenizer=tokenizer,
                text_cache_payload=text_cache_payload,
                prompt_text=prompt_text,
                prompt_aliases=prompt_aliases,
                device=device,
                prompt_feature_cache=prompt_feature_cache,
                prompt_fusion_mode=prompt_fusion_mode,
                raw_prompt_weight=raw_prompt_weight,
                prompt_match_mode=prompt_match_mode,
            )
            feature_list.append(feature)
            mapped_classes.append(mapped_class)
            route_map = router_info.get("route_map") if isinstance(router_info, dict) else None
            routes.append(resolve_prompt_route(mapped_class, route_map))

        if not feature_list:
            continue

        logits_by_index: List[Optional[torch.Tensor]] = [None] * len(prompt_batch)
        router_enabled = bool(router_info and router_info.get("enabled"))
        if router_enabled:
            grouped_indices: Dict[str, List[int]] = {"lite": [], "stage2": []}
            for idx, route in enumerate(routes):
                grouped_indices.setdefault(route, []).append(idx)
            for route_name, indices in grouped_indices.items():
                if not indices:
                    continue
                route_features = torch.stack([feature_list[idx] for idx in indices], dim=0).to(device, non_blocking=True)
                route_image_batch = image_embedding.expand(route_features.size(0), -1, -1, -1).contiguous()
                decoder = router_info["decoder_lite"] if route_name == "lite" else router_info["decoder_stage2"]
                route_logits = decoder(
                    image_features=route_image_batch,
                    text_global_features=route_features,
                    target_size=(model_input_size, model_input_size),
                )
                if route_logits.ndim != 4 or route_logits.shape[1] != 1:
                    raise RuntimeError(f"{route_name} logits 形状异常，当前: {tuple(route_logits.shape)}")
                for local_idx, batch_idx in enumerate(indices):
                    logits_by_index[batch_idx] = route_logits[local_idx]
        else:
            text_features = torch.stack(feature_list, dim=0).to(device, non_blocking=True)
            image_batch = image_embedding.expand(text_features.size(0), -1, -1, -1).contiguous()
            logits = model.decode(
                image_features=image_batch,
                text_features=text_features,
                target_size=(model_input_size, model_input_size),
                images=image_tensor.unsqueeze(0).to(device, non_blocking=True).expand(text_features.size(0), -1, -1, -1).contiguous(),
                use_refine_head=getattr(model, "use_refine_head", False),
            )
            if logits.ndim != 4 or logits.shape[1] != 1:
                raise RuntimeError(f"logits 形状异常，期望 [B, 1, H, W]，当前: {tuple(logits.shape)}")
            for idx, prompt_logits in enumerate(logits):
                logits_by_index[idx] = prompt_logits

        if any(prompt_logits is None for prompt_logits in logits_by_index):
            raise RuntimeError("router decode 未能还原完整 logits 顺序")

        for prompt_text, mapped_class, route_name, prompt_logits in zip(prompt_batch, mapped_classes, routes, logits_by_index):
            threshold = get_prompt_threshold(prompt_text, mapped_class, thresholds, default_mask_threshold)
            post_cfg = get_prompt_postprocess(prompt_text, mapped_class, postprocess_cfg)
            prob, score = logits_to_probability_map(prompt_logits, meta, model_input_size)
            mask = (prob >= threshold).astype(np.uint8)
            mask = apply_postprocess(
                mask,
                min_area=int(post_cfg.get("min_area", 0)),
                fill_holes=bool(post_cfg.get("fill_holes", False)),
                topk_components=post_cfg.get("topk_components"),
            )
            if mask.sum() == 0 and rare_empty_fallback:
                fallback_mask = apply_rare_empty_fallback(prob, mapped_class, threshold, post_cfg)
                if fallback_mask is not None and int(fallback_mask.sum()) > 0:
                    mask = fallback_mask
                    debug_stats["fallback_hit_count"] += 1
                    debug_stats["fallback_by_class"][mapped_class or prompt_text] += 1
            if mask.sum() == 0:
                continue
            results_by_prompt[prompt_text] = {
                "prompt": prompt_text,
                "mapped_class": mapped_class,
                "route": route_name,
                "score": float(score),
                "threshold": float(threshold),
                "mask_area": int(mask.sum()),
                "rle": encode_mask_to_rle(mask),
            }

    return results_by_prompt, int(meta["orig_w"]), int(meta["orig_h"]), debug_stats


def process_tasks(
    tasks: List[Dict[str, Any]],
    image_root: str,
    model_dir: str,
    checkpoint_path: Optional[str],
    output_path: str,
    mask_threshold: float,
    tokenizer_path: Optional[str] = None,
    text_cache_path: Optional[str] = None,
    threshold_json: Optional[str] = None,
    postprocess_json: Optional[str] = None,
    fail_safe: bool = True,
    save_debug_json: bool = False,
    rare_empty_fallback: bool = False,
    prompt_fusion_mode: Optional[str] = None,
    raw_prompt_weight: Optional[float] = None,
    prompt_match_mode: Optional[str] = None,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, text_cache_payload, model_input_size, checkpoint, classes, prompt_aliases, router_info = load_model(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        text_cache_path=text_cache_path,
    )
    device = next(model.parameters()).device
    prompt_fusion_mode, raw_prompt_weight, prompt_match_mode = resolve_prompt_inference_config(
        checkpoint,
        prompt_fusion_mode,
        raw_prompt_weight,
        prompt_match_mode,
    )
    LOGGER.info("final prompt_fusion_mode=%s", prompt_fusion_mode)
    LOGGER.info("final raw_prompt_weight=%.3f", raw_prompt_weight)
    LOGGER.info("final prompt_match_mode=%s", prompt_match_mode)
    for prompt_text, expected in PROMPT_MATCH_SELFTEST_CASES:
        resolved = resolve_prompt_mapped_class(prompt_text, prompt_aliases, classes, prompt_match_mode)
        LOGGER.info(
            "prompt selfcheck[%s]: %s -> %s (expected=%s)",
            prompt_match_mode,
            prompt_text,
            resolved,
            expected,
        )
    checkpoint_thresholds = resolve_checkpoint_thresholds(checkpoint) if isinstance(checkpoint, dict) else DEFAULT_THRESHOLDS
    checkpoint_postprocess = resolve_checkpoint_postprocess(checkpoint) if isinstance(checkpoint, dict) else DEFAULT_POSTPROCESS
    threshold_source = "DEFAULT"
    if threshold_json is not None:
        LOGGER.info("使用 threshold_json: %s", threshold_json)
        threshold_source = "threshold_json"
    elif isinstance(checkpoint, dict) and "val_thresholds" in checkpoint:
        LOGGER.info("使用 checkpoint 内 val_thresholds")
        threshold_source = "val_thresholds"
    elif isinstance(checkpoint, dict) and "prompt_thresholds" in checkpoint:
        LOGGER.info("使用 checkpoint 内 prompt_thresholds")
        threshold_source = "prompt_thresholds"
    elif isinstance(checkpoint, dict) and "thresholds" in checkpoint:
        LOGGER.info("使用 checkpoint 内 thresholds")
        threshold_source = "thresholds"
    elif isinstance(checkpoint, dict) and "class_mask_thresholds" in checkpoint:
        LOGGER.info("使用 checkpoint 内 class_mask_thresholds")
        threshold_source = "class_mask_thresholds"
    else:
        LOGGER.warning("checkpoint 内没有 val_thresholds，使用 DEFAULT_THRESHOLDS")
    postprocess_source = "DEFAULT"

    if postprocess_json is not None:
        LOGGER.info("使用 postprocess_json: %s", postprocess_json)
        postprocess_source = "postprocess_json"
    elif isinstance(checkpoint, dict) and "postprocess_cfg" in checkpoint:
        LOGGER.info("使用 checkpoint 内 postprocess_cfg")
        postprocess_source = "postprocess_cfg"
    elif isinstance(checkpoint, dict) and "postprocess" in checkpoint:
        LOGGER.info("使用 checkpoint 内 postprocess")
        postprocess_source = "postprocess"
    elif isinstance(checkpoint, dict) and "min_area" in checkpoint:
        LOGGER.info("使用 checkpoint 内 min_area + DEFAULT_POSTPROCESS")
        postprocess_source = "min_area"
    else:
        LOGGER.warning("checkpoint 内没有 postprocess_cfg，使用 DEFAULT_POSTPROCESS")

    thresholds = checkpoint_thresholds if threshold_json is None else json.loads(Path(threshold_json).read_text(encoding="utf-8"))
    postprocess_cfg = checkpoint_postprocess if postprocess_json is None else json.loads(Path(postprocess_json).read_text(encoding="utf-8"))
    LOGGER.info("final threshold source=%s", threshold_source)
    LOGGER.info("final postprocess source=%s", postprocess_source)
    route_map = router_info.get("route_map") if router_info else None
    if router_info and router_info.get("enabled"):
        validate_router_runtime_configs(classes, route_map or {}, thresholds, postprocess_cfg)
    for class_name in classes:
        post_cfg = postprocess_cfg.get(class_name, {})
        LOGGER.info(
            "class cfg: %s route=%s threshold=%.4f min_area=%s topk_components=%s fill_holes=%s",
            class_name,
            resolve_prompt_route(class_name, route_map),
            float(thresholds.get(class_name, mask_threshold)),
            post_cfg.get("min_area", 0),
            post_cfg.get("topk_components"),
            bool(post_cfg.get("fill_holes", False)),
        )
    prompt_feature_cache: Dict[str, Tuple[torch.Tensor, Optional[str]]] = {}

    model_info = count_model_params(model, device)
    model_info["model_type"] = "EfficientSAM+ChineseCLIP+PromptPrototype+FiLMDecoder(11cls)"
    model_info["model_dir"] = str(model_dir)
    model_info["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
    model_info["default_mask_threshold"] = float(mask_threshold)
    model_info["model_input_size"] = int(model_input_size)
    model_info["class_mask_thresholds"] = {key: float(value) for key, value in sorted(thresholds.items())}
    model_info["classes"] = list(classes)
    model_info["use_prompt_prototype"] = bool(text_cache_payload.get("use_prompt_prototype", True))
    model_info["text_cache_classes"] = list(text_cache_payload.get("classes", classes))
    model_info["text_cache_signature"] = text_cache_payload.get("prompt_prototypes_signature")
    model_info["prompt_prototypes_source"] = text_cache_payload.get("prompt_prototypes_source", "default")
    model_info["prompt_fusion_mode"] = prompt_fusion_mode
    model_info["raw_prompt_weight"] = float(raw_prompt_weight)
    model_info["prompt_match_mode"] = prompt_match_mode
    model_info["use_refine_head"] = bool(getattr(model, "use_refine_head", False))
    model_info["threshold_source"] = threshold_source
    model_info["postprocess_source"] = postprocess_source
    model_info["rare_empty_fallback_enabled"] = bool(rare_empty_fallback)
    model_info["router_enabled"] = bool(router_info.get("enabled")) if router_info else False
    model_info["router_type"] = router_info.get("router_type") if router_info else None
    model_info["route_mode"] = router_info.get("route_mode") if router_info else None
    model_info["route_map"] = route_map
    model_info["base_model_params"] = router_info.get("base_model_params") if router_info else None
    model_info["stage2_decoder_params"] = router_info.get("stage2_decoder_params") if router_info else None
    model_info["total_runtime_params"] = router_info.get("total_runtime_params") if router_info else None
    model_info["thresholds"] = {key: float(value) for key, value in sorted(thresholds.items())}
    model_info["postprocess_cfg"] = {
        key: {
            "min_area": int(value.get("min_area", 0)),
            "fill_holes": bool(value.get("fill_holes", False)),
            **({"topk_components": int(value.get("topk_components", 0))} if value.get("topk_components") is not None else {}),
        }
        for key, value in sorted(postprocess_cfg.items())
    }
    if isinstance(checkpoint, dict):
        for key in ["epoch", "run_name", "model_type", "val_thresholds", "prompt_thresholds", "sweep_mode", "sweep_metric"]:
            if key in checkpoint:
                model_info[f"checkpoint_{key}"] = checkpoint[key]

    tasks_by_image = group_tasks_by_image(tasks)
    LOGGER.info("任务总数: %d", len(tasks))
    LOGGER.info("图片总数: %d", len(tasks_by_image))

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}
    failed_items: List[Dict[str, Any]] = []
    empty_mask_count = 0
    class_pred_count: Dict[str, int] = defaultdict(int)
    class_empty_count: Dict[str, int] = defaultdict(int)
    pred_area_ratio_values: Dict[str, List[float]] = defaultdict(list)
    large_area_count_by_class: Dict[str, int] = defaultdict(int)
    fallback_hit_count = 0
    fallback_by_class: Dict[str, int] = defaultdict(int)

    for image_rel_path, image_tasks in maybe_tqdm(
        tasks_by_image.items(),
        total=len(tasks_by_image),
        desc="推理进度",
        leave=False,
    ):
        image_abs_path = resolve_image_path(image_root, image_rel_path)
        image_start = time.time()
        try:
            unique_prompts = list(dict.fromkeys(get_prompt_text(task) for task in image_tasks))
            results_by_prompt, width, height, image_debug_stats = do_inference(
                image_path=image_abs_path,
                text_prompts=unique_prompts,
                model=model,
                tokenizer=tokenizer,
                text_cache_payload=text_cache_payload,
                model_input_size=model_input_size,
                thresholds=thresholds,
                postprocess_cfg=postprocess_cfg,
                prompt_aliases=prompt_aliases,
                default_mask_threshold=mask_threshold,
                rare_empty_fallback=rare_empty_fallback,
                prompt_feature_cache=prompt_feature_cache,
                prompt_fusion_mode=prompt_fusion_mode,
                raw_prompt_weight=raw_prompt_weight,
                prompt_match_mode=prompt_match_mode,
                router_info=router_info,
            )
            fallback_hit_count += int(image_debug_stats.get("fallback_hit_count", 0))
            for class_name, hit_count in image_debug_stats.get("fallback_by_class", {}).items():
                fallback_by_class[str(class_name)] += int(hit_count)
            empty_rle = empty_mask_rle(height, width)
            image_area = max(int(width) * int(height), 1)
            for task in image_tasks:
                ann_id = task["ann_id"]
                prompt_text = get_prompt_text(task)
                prompt_class_key = resolve_prompt_mapped_class(
                    prompt_text,
                    prompt_aliases,
                    classes,
                    prompt_match_mode=prompt_match_mode,
                ) or prompt_text
                prediction = results_by_prompt.get(prompt_text)
                if prediction is None:
                    task_to_rle[ann_id] = empty_rle
                    empty_mask_count += 1
                    class_empty_count[prompt_class_key] += 1
                else:
                    task_to_rle[ann_id] = prediction["rle"]
                    pred_class_key = prediction.get("mapped_class") or prompt_class_key
                    class_pred_count[pred_class_key] += 1
                    mask_area = int(prediction.get("mask_area", 0))
                    area_ratio = float(mask_area / image_area)
                    pred_area_ratio_values[pred_class_key].append(area_ratio)
                    if area_ratio >= float(LARGE_AREA_DEBUG_THRESHOLDS.get(pred_class_key, 1.01)):
                        large_area_count_by_class[pred_class_key] += 1
        except Exception as exc:
            if not fail_safe:
                raise
            width, height = 1, 1
            if os.path.exists(image_abs_path):
                try:
                    with Image.open(image_abs_path) as image_obj:
                        width, height = image_obj.size
                except Exception:
                    pass
            fallback_rle = empty_mask_rle(height, width)
            for task in image_tasks:
                task_to_rle[task["ann_id"]] = fallback_rle
                empty_mask_count += 1
                failed_items.append(
                    {
                        "ann_id": task["ann_id"],
                        "image_path": image_rel_path,
                        "error": str(exc),
                    }
                )
                prompt_text = get_prompt_text(task)
                prompt_class_key = resolve_prompt_mapped_class(
                    prompt_text,
                    prompt_aliases,
                    classes,
                    prompt_match_mode=prompt_match_mode,
                ) or prompt_text
                class_empty_count[prompt_class_key] += 1
        inference_total_time += time.time() - image_start
        processed_images += 1

    predictions_output = [{"ann_id": task["ann_id"], "rle": task_to_rle[task["ann_id"]]} for task in tasks]
    if len(predictions_output) != len(tasks):
        raise RuntimeError(f"predictions 数量不匹配: predictions={len(predictions_output)}, tasks={len(tasks)}")

    avg_time_per_image = inference_total_time / max(processed_images, 1)
    predictions_payload = {"predictions": predictions_output}
    debug_payload = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_time_per_image),
            "avg_inference_seconds_per_task": float(inference_total_time / max(len(tasks), 1)),
            "processed_images": int(processed_images),
            "total_tasks": int(len(tasks)),
            "empty_mask_count": int(empty_mask_count),
            "failed_count": int(len(failed_items)),
        },
        "class_prediction_stats": {
            "pred_count": {key: int(value) for key, value in sorted(class_pred_count.items())},
            "empty_count": {key: int(value) for key, value in sorted(class_empty_count.items())},
            "pred_area_ratio_by_class": {
                key: float(sum(values) / max(len(values), 1))
                for key, values in sorted(pred_area_ratio_values.items())
            },
            "large_area_count_by_class": {
                key: int(value) for key, value in sorted(large_area_count_by_class.items())
            },
        },
        "rare_fallback_stats": {
            "rare_empty_fallback_enabled": bool(rare_empty_fallback),
            "fallback_hit_count": int(fallback_hit_count),
            "fallback_by_class": {
                key: int(value) for key, value in sorted(fallback_by_class.items())
            },
        },
    }
    if failed_items:
        debug_payload["failed_items"] = failed_items

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(predictions_payload, file_obj, ensure_ascii=False, separators=(",", ":"))
    if save_debug_json:
        debug_path = output_path_obj.with_name(f"{output_path_obj.stem}_debug.json")
        save_json(debug_path, debug_payload)
        LOGGER.info("debug 信息已保存: %s", debug_path)

    LOGGER.info("推理完成，输出: %s", output_path_obj)
    LOGGER.info("纯推理总耗时: %.2fs", inference_total_time)
    LOGGER.info("平均每张图耗时: %.2fs", avg_time_per_image)
    LOGGER.info("平均每个 task 耗时: %.4fs", inference_total_time / max(len(tasks), 1))
    LOGGER.info("处理图片数: %d", processed_images)
    LOGGER.info("处理 task 数: %d", len(tasks))
    LOGGER.info("empty_mask_count: %d", empty_mask_count)
    LOGGER.info("failed_count: %d", len(failed_items))
    LOGGER.info("rare_empty_fallback_enabled: %s", rare_empty_fallback)
    LOGGER.info("fallback_hit_count: %d", fallback_hit_count)
    LOGGER.info("prompt_fusion_mode: %s", prompt_fusion_mode)
    LOGGER.info("raw_prompt_weight: %.3f", raw_prompt_weight)
    LOGGER.info("prompt_match_mode: %s", prompt_match_mode)


def main() -> None:
    configure_logging()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--images_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--text_cache_path", type=str, default=None)
    parser.add_argument("--threshold_json", type=str, default=None)
    parser.add_argument("--postprocess_json", type=str, default=None)
    parser.add_argument("--mask_threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save_debug_json", action="store_true")
    parser.add_argument("--rare_empty_fallback", action="store_true")
    parser.add_argument("--prompt_fusion_mode", choices=["prototype", "raw", "blend"], default=None)
    parser.add_argument("--raw_prompt_weight", type=float, default=None)
    parser.add_argument("--prompt_match_mode", choices=["exact", "soft", "target_soft"], default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("EfficientSAM + ChineseCLIP 提交推理 (11类优化路线)")
    print("=" * 60)
    print(f"任务文件: {args.tasks}")
    print(f"图片根目录: {args.images_root}")
    print(f"输出路径: {args.output}")
    print(f"模型目录: {args.model_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"默认输入尺寸: {DEFAULT_SUBMIT_IMG_SIZE} (若 checkpoint 含 img_size 则优先使用)")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"rare empty fallback: {args.rare_empty_fallback}")
    print(f"prompt fusion mode(cli): {args.prompt_fusion_mode}")
    print(f"raw prompt weight(cli): {args.raw_prompt_weight}")
    print(f"prompt match mode(cli): {args.prompt_match_mode}")
    print("=" * 60)

    tasks = load_tasks(resolve_path(args.tasks) or args.tasks)
    process_tasks(
        tasks=tasks,
        image_root=resolve_path(args.images_root) or args.images_root,
        model_dir=resolve_path(args.model_dir) or args.model_dir,
        checkpoint_path=resolve_path(args.checkpoint) or args.checkpoint,
        output_path=resolve_path(args.output) or args.output,
        mask_threshold=args.mask_threshold,
        tokenizer_path=args.tokenizer_path,
        text_cache_path=args.text_cache_path,
        threshold_json=resolve_path(args.threshold_json) if args.threshold_json else None,
        postprocess_json=resolve_path(args.postprocess_json) if args.postprocess_json else None,
        fail_safe=not args.strict,
        save_debug_json=args.save_debug_json,
        rare_empty_fallback=args.rare_empty_fallback,
        prompt_fusion_mode=args.prompt_fusion_mode,
        raw_prompt_weight=args.raw_prompt_weight,
        prompt_match_mode=args.prompt_match_mode,
    )


if __name__ == "__main__":
    main()
