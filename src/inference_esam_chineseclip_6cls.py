#!/usr/bin/env python3
"""
EfficientSAM image encoder + ChineseCLIP text encoder + FiLM decoder
提交推理脚本（6 类）。

设计目标：
1. 提交协议对齐 clip 路线：tasks 读取、按图聚合、RLE、predictions.json 结构一致
2. 数据处理对齐 hanxue 训练路线：gray -> 3 通道 -> /255，训练尺寸 768，阈值按验证配置
3. 单文件可提交：不依赖 src/hanxue 下的额外 Python 模块，也不依赖外部基础权重目录
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(__file__).resolve().parent / ".hf_cache" / "hub"))

from transformers import (
    AutoTokenizer,
    BertTokenizer,
    ChineseCLIPTextConfig,
    ChineseCLIPTextModel,
)

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("警告: pycocotools 未安装，将使用备用 RLE 编码方法")
    maskUtils = None


DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test/"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_MODEL_DIR = "/raytron/code/model"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"

CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "computer",
]

DEFAULT_MASK_THRESHOLD = 0.5
CLASS_MASK_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
    "computer": 0.60,
}

DEFAULT_INPUT_SIZE = 768
EFFICIENT_SAM_BACKBONE_SIZE = 1024
PROMPT_BATCH_SIZE = 8
MAX_TEXT_LEN = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200.0
PSEUDO_COLOR_SAT_THRESH = 60.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0


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


def preprocess_image(image_path: str, input_size: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    gray = load_teacher_aligned_gray(image_path)
    orig_h, orig_w = gray.shape
    scale = input_size / max(orig_h, orig_w)
    resized_h = max(1, int(round(orig_h * scale)))
    resized_w = max(1, int(round(orig_w * scale)))
    gray = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h = input_size - resized_h
    pad_w = input_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    gray = cv2.copyMakeBorder(
        gray,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return tensor, meta


def logits_to_mask(
    logits: torch.Tensor,
    meta: Dict[str, int],
    threshold: float,
    input_size: int,
) -> Tuple[np.ndarray, float]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.ndim == 3:
        logits = logits.unsqueeze(1)

    logits = F.interpolate(logits, size=(input_size, input_size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

    top = meta["pad_top"]
    left = meta["pad_left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    prob = prob[top:top + resized_h, left:left + resized_w]
    prob = cv2.resize(prob, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)

    score = float(prob.max()) if prob.size > 0 else 0.0
    return (prob >= threshold).astype(np.uint8), score


def get_mask_threshold(prompt: str, default_threshold: float) -> float:
    return float(CLASS_MASK_THRESHOLDS.get(prompt, default_threshold))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]) -> torch.Tensor:
    height, width = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    token_count = abs_pos.shape[1]
    grid_size = int(round(token_count ** 0.5))
    if grid_size * grid_size != token_count:
        raise RuntimeError(f"非法 position embedding token 数量: {token_count}")
    if grid_size != height or grid_size != width:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, height, width, -1)


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = EFFICIENT_SAM_BACKBONE_SIZE,
        patch_size: int = 16,
        in_chans: int = 3,
        patch_embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        neck_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        neck_dims = neck_dims or [256, 256]
        self.img_size = img_size
        self.image_embedding_size = img_size // patch_size
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True

        pretrain_img_size = 224
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=patch_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, patch_embed_dim))
        self.blocks = nn.ModuleList([Block(dim=patch_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.neck = nn.Sequential(
            nn.Conv2d(patch_embed_dim, neck_dims[0], kernel_size=1, bias=False),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(neck_dims[0], neck_dims[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dims[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"input image size must be {(self.img_size, self.img_size)}, 当前为 {tuple(x.shape[-2:])}")
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        grid_size = x.shape[1]
        if x.shape[2] != grid_size:
            raise RuntimeError(f"图像 token 网格不是正方形: {tuple(x.shape)}")
        x = x.reshape(x.shape[0], grid_size * grid_size, x.shape[3])
        for block in self.blocks:
            x = block(x)
        x = x.reshape(x.shape[0], grid_size, grid_size, x.shape[2])
        return self.neck(x.permute(0, 3, 1, 2))


class PureImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = ImageEncoderViT()
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
        self._resize_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_size = int(self.image_encoder.img_size)
        if x.shape[-2:] != (expected_size, expected_size):
            if not self._resize_warned:
                print(
                    f"[PureImageEncoder] 输入尺寸 {tuple(x.shape[-2:])} 与 backbone 期望 {(expected_size, expected_size)} 不一致，将自动插值。"
                )
                self._resize_warned = True
            x = F.interpolate(x, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
        x = (x - self.pixel_mean.to(device=x.device, dtype=x.dtype)) / self.pixel_std.to(
            device=x.device, dtype=x.dtype
        )
        return self.image_encoder(x)


class FreezeTextEncoder(nn.Module):
    def __init__(self, text_config: ChineseCLIPTextConfig) -> None:
        super().__init__()
        self.text_model = ChineseCLIPTextModel(text_config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        if pooler_output is None:
            pooler_output = last_hidden_state[:, 0, :]
        return last_hidden_state, pooler_output


class FiLMFusionDecoder(nn.Module):
    def __init__(self, image_dim: int = 256, text_dim: int = 768) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.film_projection = nn.Linear(text_dim, image_dim * 2)
        self.conv_after_fusion = nn.Sequential(
            nn.Conv2d(image_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_global_features: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        film_params = self.film_projection(text_global_features)
        gamma, beta = torch.split(film_params, image_features.size(1), dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fused = image_features * (1.0 + gamma) + beta
        x = self.conv_after_fusion(fused)
        x = self.mask_head(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class CustomSAMWorldModel(nn.Module):
    def __init__(self, text_config: ChineseCLIPTextConfig) -> None:
        super().__init__()
        self.img_encoder = PureImageEncoder()
        self.txt_encoder = FreezeTextEncoder(text_config)
        self.decoder = FiLMFusionDecoder(image_dim=256, text_dim=int(text_config.hidden_size))

    @staticmethod
    def _get_text_global_feature(text_encoder_output: Any) -> torch.Tensor:
        if torch.is_tensor(text_encoder_output):
            return text_encoder_output
        if isinstance(text_encoder_output, (list, tuple)):
            if len(text_encoder_output) >= 2:
                return text_encoder_output[1]
            if len(text_encoder_output) == 1:
                return text_encoder_output[0]
        if isinstance(text_encoder_output, dict):
            if text_encoder_output.get("pooler_output") is not None:
                return text_encoder_output["pooler_output"]
            if "last_hidden_state" in text_encoder_output:
                return text_encoder_output["last_hidden_state"][:, 0, :]
        if hasattr(text_encoder_output, "pooler_output") and text_encoder_output.pooler_output is not None:
            return text_encoder_output.pooler_output
        if hasattr(text_encoder_output, "last_hidden_state"):
            return text_encoder_output.last_hidden_state[:, 0, :]
        raise RuntimeError("无法从 text_encoder 输出中解析文本全局特征。")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if target_size is None:
            target_size = (int(images.size(2)), int(images.size(3)))
        image_features = self.img_encoder(images)
        text_output = self.txt_encoder(input_ids, attention_mask)
        text_global_feat = self._get_text_global_feature(text_output)
        return self.decoder(image_features=image_features, text_global_features=text_global_feat, target_size=target_size)


def count_model_params(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "device": DEVICE,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model", "model_state_dict", "state_dict", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_like = [torch.is_tensor(v) for v in checkpoint.values()]
            if tensor_like and any(tensor_like):
                return checkpoint
    raise RuntimeError("无法从 checkpoint 中解析 state_dict。")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = value
    return cleaned


def load_local_tokenizer(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc:
        print(f"[Tokenizer 警告] AutoTokenizer 加载失败，回退 BertTokenizer。错误: {repr(exc)}")
        return BertTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def load_text_config(model_dir: Path, state_dict: Dict[str, torch.Tensor]) -> ChineseCLIPTextConfig:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"缺少 ChineseCLIP 配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config_data = json.load(file_obj)

    text_cfg_dict = dict(config_data.get("text_config") or {})
    if not text_cfg_dict:
        raise RuntimeError(f"{config_path} 中缺少 text_config，无法构造 ChineseCLIPTextModel。")

    vocab_key = "txt_encoder.text_model.embeddings.word_embeddings.weight"
    if vocab_key in state_dict and hasattr(state_dict[vocab_key], "shape"):
        text_cfg_dict["vocab_size"] = int(state_dict[vocab_key].shape[0])
    return ChineseCLIPTextConfig(**text_cfg_dict)


def load_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT_PATH,
) -> Tuple[CustomSAMWorldModel, Any, int]:
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir_path}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"缺少模型权重文件: {checkpoint_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(extract_state_dict(checkpoint))
    text_config = load_text_config(model_dir_path, state_dict)
    tokenizer = load_local_tokenizer(model_dir_path)

    model = CustomSAMWorldModel(text_config=text_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"警告: checkpoint 缺少 {len(missing)} 个参数，前20个: {missing[:20]}")
    if unexpected:
        print(f"警告: checkpoint 多出 {len(unexpected)} 个参数，前20个: {unexpected[:20]}")
    if len(missing) > 30 or len(unexpected) > 30:
        print("[WARN] missing/unexpected keys 较多，请确认导出权重和模型结构一致。")

    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer, int(DEFAULT_INPUT_SIZE)


@torch.inference_mode()
def do_inference(
    image_path: str,
    text_prompts: List[str],
    model: CustomSAMWorldModel,
    tokenizer,
    model_input_size: int,
    default_mask_threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    image_tensor, meta = preprocess_image(image_path, model_input_size)
    results_by_prompt: Dict[str, Dict[str, Any]] = {}

    for start in range(0, len(text_prompts), PROMPT_BATCH_SIZE):
        prompt_batch = text_prompts[start:start + PROMPT_BATCH_SIZE]
        valid_prompts = [prompt for prompt in prompt_batch if prompt in CLASSES]
        if not valid_prompts:
            continue

        tokenized = tokenizer(
            valid_prompts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)
        images = image_tensor.unsqueeze(0).repeat(len(valid_prompts), 1, 1, 1).to(DEVICE)

        logits = model(images, input_ids, attention_mask, target_size=(model_input_size, model_input_size))
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"logits 形状异常，期望 [B, 1, H, W]，当前: {tuple(logits.shape)}")

        for prompt, prompt_logits in zip(valid_prompts, logits):
            threshold = get_mask_threshold(prompt, default_mask_threshold)
            mask, score = logits_to_mask(prompt_logits, meta, threshold, model_input_size)
            if mask.sum() == 0:
                continue
            results_by_prompt[prompt] = {
                "prompt": prompt,
                "score": score,
                "threshold": threshold,
                "rle": mask_to_rle(mask),
            }

    return results_by_prompt, meta["orig_w"], meta["orig_h"]


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as file_obj:
        tasks = json.load(file_obj)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"第 {index} 条任务不是 JSON 对象")
        if "ann_id" not in task or "image_path" not in task or "text_prompt" not in task:
            raise ValueError(f"第 {index} 条任务缺少 ann_id/image_path/text_prompt")
    return tasks


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

    model, tokenizer, model_input_size = load_model(model_dir=model_dir, checkpoint_path=checkpoint_path)
    model_info = count_model_params(model)
    model_info["model_type"] = "EfficientSAMImageEncoder+ChineseCLIPText+FiLMDecoder"
    model_info["model_dir"] = model_dir
    model_info["default_mask_threshold"] = float(mask_threshold)
    model_info["model_input_size"] = int(model_input_size)
    model_info["class_mask_thresholds"] = {key: float(value) for key, value in sorted(CLASS_MASK_THRESHOLDS.items())}
    if checkpoint_path:
        model_info["checkpoint_path"] = checkpoint_path

    tasks_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        tasks_by_image[task["image_path"]].append(task)

    print(f"任务总数: {len(tasks)}")
    print(f"图片总数: {len(tasks_by_image)}")

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}

    for image_rel_path, image_tasks in tqdm(
        tasks_by_image.items(),
        total=len(tasks_by_image),
        desc="推理进度",
        unit="img",
    ):
        image_abs_path = resolve_image_path(image_root, image_rel_path)
        if not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        unique_prompts = list(
            dict.fromkeys(
                task.get("text_prompt", "").strip()
                for task in image_tasks
                if task.get("text_prompt", "").strip()
            )
        )

        image_start = time.time()
        results_by_prompt, width, height = do_inference(
            image_path=image_abs_path,
            text_prompts=unique_prompts,
            model=model,
            tokenizer=tokenizer,
            model_input_size=model_input_size,
            default_mask_threshold=mask_threshold,
        )
        empty_rle = build_empty_rle(height, width)

        for task in image_tasks:
            ann_id = task["ann_id"]
            prompt = task.get("text_prompt", "").strip()
            prediction = results_by_prompt.get(prompt)
            task_to_rle[ann_id] = prediction["rle"] if prediction is not None else empty_rle

        inference_total_time += time.time() - image_start
        processed_images += 1

    predictions_output = [{"ann_id": task["ann_id"], "rle": task_to_rle[task["ann_id"]]} for task in tasks]
    if len(predictions_output) != len(tasks):
        raise RuntimeError(f"predictions 数量不匹配: predictions={len(predictions_output)}, tasks={len(tasks)}")

    avg_time = inference_total_time / max(processed_images, 1)
    output_data = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_time),
            "processed_images": processed_images,
            "total_tasks": len(tasks),
        },
        "predictions": predictions_output,
    }

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)

    print(f"\n推理完成! 输出: {output_path_obj}")
    print(f"纯推理总耗时: {inference_total_time:.2f}s, 平均每张图: {avg_time:.2f}s")


def main() -> None:
    print("=" * 60)
    print("EfficientSAM + ChineseCLIP 提交推理 (6类)")
    print("=" * 60)
    print(f"任务文件: {DEFAULT_TASKS}")
    print(f"图片根目录: {DEFAULT_IMAGE_ROOT}")
    print(f"输出路径: {DEFAULT_OUTPUT_PATH}")
    print(f"模型目录: {DEFAULT_MODEL_DIR}")
    print(f"模型检查点: {DEFAULT_CHECKPOINT_PATH}")
    print(f"输入尺寸: {DEFAULT_INPUT_SIZE}")
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
