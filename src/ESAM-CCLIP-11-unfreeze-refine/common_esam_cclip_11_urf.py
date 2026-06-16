import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HANXUE_ROOT = ROOT / "src" / "hanxue"
candidate_roots = [
    ROOT / "src" / "hanxue",
    Path(__file__).resolve().parent / "src" / "hanxue",
    Path.cwd() / "src" / "hanxue",
    HANXUE_ROOT,
]
for candidate_root in candidate_roots:
    if candidate_root.exists():
        candidate_str = str(candidate_root.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        HANXUE_ROOT = candidate_root.resolve()
        break

from data.dataset import (  # noqa: E402
    letterbox_image,
    letterbox_mask,
    load_teacher_aligned_gray,
    load_tokenizer,
    resolve_existing_path,
    rle_to_mask,
)
from models.fusion_decoder import FiLMFusionDecoder  # noqa: E402
from models.image_encoder import PureImageEncoder  # noqa: E402
from models.text_encoder import FreezeTextEncoder  # noqa: E402
from utils.format_utils import empty_mask_rle, encode_mask_to_rle  # noqa: E402
from utils.losses import DiceLoss, FocalLoss  # noqa: E402

LOGGER = logging.getLogger("ESAM_CCLIP_11")


def maybe_tqdm(iterable, total, desc, leave=False):
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


def resolve_device(device_name: str):
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


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total_params, trainable_params


def compute_metrics(logits, targets, threshold):
    pred = (torch.sigmoid(logits) > threshold).float()
    gt = (targets > 0.5).float()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()
    if pred_sum == 0 and gt_sum == 0:
        return {
            "iou": 1.0,
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "pred_area": 0.0,
            "gt_area": 0.0,
        }
    inter = (pred * gt).sum().item()
    union = (pred + gt).clamp(0, 1).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    return {
        "iou": inter / (union + 1e-7),
        "dice": 2 * inter / (pred_sum + gt_sum + 1e-7),
        "precision": inter / (inter + fp + 1e-7),
        "recall": inter / (inter + fn + 1e-7),
        "pred_area": pred_sum,
        "gt_area": gt_sum,
    }


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


def apply_postprocess(mask: np.ndarray, min_area: int = 0, fill_holes: bool = False) -> np.ndarray:
    processed = remove_small_components(mask, min_area=min_area)
    if fill_holes:
        processed = fill_small_holes(processed)
    return processed.astype(np.uint8)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def normalize_text_feature(text_feature: torch.Tensor) -> torch.Tensor:
    return text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def load_state_dict_flexible(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict")


class ESAMCCLIPModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer_dir,
        efficient_sam_ckpt,
        freeze_image=True,
        freeze_text=True,
        use_refine_head: bool = False,
        refine_head_hidden_dim: int = 16,
    ):
        super().__init__()
        self.img_encoder = PureImageEncoder(checkpoint_path=str(efficient_sam_ckpt), freeze=freeze_image)
        self.txt_encoder = FreezeTextEncoder(model_dir=str(tokenizer_dir))
        self.decoder = FiLMFusionDecoder(image_dim=256, text_dim=768)
        self.refine_head = ZeroInitResidualRefineHead(hidden_dim=refine_head_hidden_dim)
        self.use_refine_head = bool(use_refine_head)
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
    ):
        coarse_logits = self.decoder(
            image_features=image_features,
            text_global_features=text_features,
            target_size=target_size,
        )
        refine_enabled = self.use_refine_head if use_refine_head is None else bool(use_refine_head)
        if not refine_enabled:
            return coarse_logits
        if images is None:
            raise RuntimeError("use_refine_head=True 时必须向 decode 传入 images。")
        residual_logits = self.refine_head(coarse_logits=coarse_logits, images=images)
        return coarse_logits + residual_logits

    def forward(self, images, input_ids, attention_mask, target_size):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        return self.decode(image_features, text_features, target_size, images=images)


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
        if images.dim() != 4:
            raise RuntimeError(f"refine_head expects BCHW images, got shape={tuple(images.shape)}")
        if images.shape[-2:] != coarse_logits.shape[-2:]:
            images = torch.nn.functional.interpolate(
                images,
                size=coarse_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gray = images.mean(dim=1, keepdim=True)
        gray_min = gray.amin(dim=(-2, -1), keepdim=True)
        gray_max = gray.amax(dim=(-2, -1), keepdim=True)
        gray = (gray - gray_min) / (gray_max - gray_min).clamp_min(1e-6)
        coarse_prob = torch.sigmoid(coarse_logits)
        refine_input = torch.cat([gray, coarse_logits, coarse_prob], dim=1)
        hidden = self.act1(self.conv1(refine_input))
        hidden = self.act2(self.conv2(hidden))
        return self.out(hidden)


def group_tasks_by_image(tasks):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[str(task["image_path"])].append(task)
    return grouped
