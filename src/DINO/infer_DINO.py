#!/usr/bin/env python3
"""
DINO+EfficientSAM 推理闭环脚本
===============================
流程：test image → GroundingDINO 产 box → 训练好的 EfficientSAM 产 mask → 后处理 → 输出 JSON

与训练脚本保持一致：
  - 图像预处理（灰度、blackHot 反色、CLAHE、去噪、锐化、letterbox 768）
  - EfficientSAM forward（box prompt → mask logits）
  - RLE 编解码

输出 JSON 格式对齐项目现有 clipseg/Hanxue 推理格式。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── 路径配置 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent        # 当前脚本所在目录
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── 输入路径 ──────────────────────────────────────────────
IMAGE_ROOT        = ROOT
TEST_TASK_JSON    = ROOT / "test" / "test_tasks.json"
TEST_LIST         = ROOT / "test" / "test_list.txt"

# ── 输出路径 ──────────────────────────────────────────────
OUTPUT_JSON       = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1" / "pred_test_dino_efficientsam.json"

# ── 模型权重 ──────────────────────────────────────────────
EFFICIENT_SAM_CKPT = ROOT / "src" / "DINO" / "models" / "efficientsam" / "efficient_sam_vitt.pt"
STUDENT_CKPT       = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1" / "best_model_only.pt"

# ── GroundingDINO 配置 ───────────────────────────────────
# 如果 config .py 文件不存在，适配层会自动回退到 HuggingFace transformers 方式
_GD_CONFIG_CANDIDATE = (
    ROOT / "src" / "DINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
)
GROUNDINGDINO_CONFIG = str(_GD_CONFIG_CANDIDATE) if _GD_CONFIG_CANDIDATE.exists() else None
GROUNDINGDINO_CKPT = (
    ROOT / "src" / "DINO" / "models" / "groundingdino" / "groundingdino_swint_ogc.pth"
)

# ── 类别配置（与训练脚本完全一致）─────────────────────────
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

# ── DINO 检测阈值（按类，初始不要太高）───────────────────
DINO_THRESHOLDS = {
    "person": 0.25,
    "car": 0.25,
    "building": 0.20,
    "tree": 0.20,
    "animal": 0.18,
    "computer": 0.18,
    "trash can": 0.18,
    "window": 0.18,
    "door": 0.18,
    "fence": 0.16,
    "pole_light": 0.16,
}

# ── DINO 每类最多保留 box 数 ──────────────────────────────
MAX_BOXES_PER_CLASS = {
    "person": 80,
    "car": 120,
    "building": 80,
    "tree": 100,
    "animal": 80,
    "computer": 40,
    "trash can": 40,
    "window": 60,
    "door": 60,
    "fence": 80,
    "pole_light": 80,
}

# ── EfficientSAM mask 二值化阈值（按类）──────────────────
MASK_THRESHOLDS = {
    "person": 0.50,
    "car": 0.50,
    "building": 0.50,
    "tree": 0.50,
    "animal": 0.45,
    "computer": 0.45,
    "trash can": 0.45,
    "window": 0.45,
    "door": 0.45,
    "fence": 0.45,
    "pole_light": 0.45,
}

# ── 后处理：每类最小连通域面积 ────────────────────────────
MIN_AREA = {
    "person": 8,
    "car": 8,
    "building": 16,
    "tree": 8,
    "animal": 4,
    "computer": 4,
    "trash can": 4,
    "window": 4,
    "door": 4,
    "fence": 4,
    "pole_light": 2,
}

# ── 后处理：同类 mask NMS IoU 阈值 ────────────────────────
MASK_NMS_IOU = 0.65

# ── 推理参数 ──────────────────────────────────────────────
IMG_SIZE = 768
DEVICE = "cuda"
BATCH_SIZE = 32  # 同一张图的多个 box 一次喂给 EfficientSAM 的最大 batch

# ── image preprocess 参数（与训练完全一致）───────────────
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


# ═══════════════════════════════════════════════════════════
# 复用的工具函数（与训练脚本一致或为其变体）
# ═══════════════════════════════════════════════════════════

def setup_logging(log_path: Path | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=handlers,
        force=True,
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
    """灰度读取 + 极性统一 + CLAHE + 降噪 + 锐化（与训练完全一致）"""
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
    """与训练脚本完全一致"""
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
        "scale": scale,
        "target_h": target_h,
        "target_w": target_w,
    }
    return padded, meta


def clip_box_xyxy(box, width, height):
    """与训练脚本完全一致"""
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    x2 = float(np.clip(x2, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    y2 = float(np.clip(y2, 0, max(height - 1, 0)))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def letterbox_box_xyxy(box, meta):
    """与训练脚本完全一致"""
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


def unletterbox_mask(mask_768, meta):
    """
    将 768×768 letterbox 空间的 mask 还原回原图尺寸。
    mask_768: np.uint8 [768, 768] or [H, W]
    meta: letterbox_image 返回的 meta dict
    返回: np.uint8 [orig_h, orig_w]
    """
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    # 裁掉 padding
    cropped = mask_768[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # resize 回原图
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return restored


def mask_to_rle(binary_mask):
    """
    将二值 mask (np.uint8, [H, W]) 编码为 COCO RLE 格式。
    需要 pycocotools。返回 {"size": [H, W], "counts": "..."}
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError(
            "mask_to_rle 需要 pycocotools。请安装: pip install pycocotools"
        )

    binary_mask = np.asarray(binary_mask)
    if binary_mask.ndim != 2:
        if binary_mask.ndim == 3 and binary_mask.shape[0] == 1:
            binary_mask = binary_mask[0]
        elif binary_mask.ndim == 3 and binary_mask.shape[-1] == 1:
            binary_mask = binary_mask[..., 0]
        else:
            raise ValueError(f"mask_to_rle 只接受二维 mask，当前形状: {binary_mask.shape}")

    binary_mask = (binary_mask > 0).astype(np.uint8)
    mask_fortran = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(mask_fortran)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]
    return rle


def mask_iou(mask1, mask2):
    """计算两个二值 mask 的 IoU。"""
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_nms(masks, scores, class_names, iou_threshold):
    """
    简单的 mask-level NMS（按 score 降序，贪心消除）。
    返回保留的 indices。
    """
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = [False] * len(masks)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or j == i:
                continue
            # 只对同类做 NMS
            if class_names[i] != class_names[j]:
                continue
            if mask_iou(masks[i], masks[j]) > iou_threshold:
                suppressed[j] = True

    return keep


# ═══════════════════════════════════════════════════════════
# GroundingDINO 适配层
# ═══════════════════════════════════════════════════════════

def run_grounding_dino(image_path: str | Path, class_names: list[str]):
    """
    对单张图片运行 GroundingDINO，返回检测到的 boxes。

    返回:
        list of dict: [
            {
                "class_name": str,
                "score": float,
                "box": [x1, y1, x2, y2]  # 原图坐标 xyxy
            },
            ...
        ]

    适配层说明：
      优先尝试 huggingface transformers 版 GroundingDINO（`from transformers import
      GroundingDinoProcessor, GroundingDinoForObjectDetection`）。
      其次尝试独立 GroundingDINO 仓库的 API。
      如果都无法导入，请根据你的实际 DINO 安装情况调整此函数。
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    detections = []

    # ── 方式 A：HuggingFace transformers GroundingDINO ──
    try:
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        # 全局缓存模型，避免每张图重复加载
        if not hasattr(run_grounding_dino, "_hf_processor"):
            model_id = "IDEA-Research/grounding-dino-base"
            logger.info(f"[DINO] 加载 HF 模型: {model_id}")
            run_grounding_dino._hf_processor = AutoProcessor.from_pretrained(model_id)
            run_grounding_dino._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(run_grounding_dino._hf_device if hasattr(run_grounding_dino, "_hf_device") else "cpu")

        processor = run_grounding_dino._hf_processor
        model = run_grounding_dino._hf_model
        device = next(model.parameters()).device

        image_pil = Image.open(image_path).convert("RGB")

        # 合并 11 类 prompt
        text_prompt = ". ".join(class_names) + "."

        inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.0,  # 先不过滤，交给后续阈值
            target_sizes=target_sizes,
            text_threshold=0.0,
        )

        result = results[0]
        boxes = result["boxes"].cpu().tolist() if "boxes" in result else []
        scores = result["scores"].cpu().tolist() if "scores" in result else []
        labels = result["labels"] if "labels" in result else []

        for box, score, label in zip(boxes, scores, labels):
            # label 可能是 str 或 int
            cls_name = str(label).lower() if isinstance(label, str) else class_names[label] if isinstance(label, int) and label < len(class_names) else str(label)
            # 统一映射到 CLASSES
            for cn in class_names:
                if cn.lower() in cls_name:
                    cls_name = cn
                    break
            if cls_name not in class_names:
                continue
            # HF 返回 [cx, cy, w, h] → 转 [x1, y1, x2, y2]
            cx, cy, w, h = box
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            detections.append({
                "class_name": cls_name,
                "score": float(score),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })

        logger.debug(f"[DINO] {image_path.name}: {len(detections)} raw detections")
        return detections

    except ImportError:
        pass  # 尝试下一种方式
    except Exception as e:
        logger.warning(f"[DINO] HF 方式失败: {e}，尝试下一种方式...")

    # ── 方式 B：独立 GroundingDINO 仓库 ──
    if GROUNDINGDINO_CONFIG is not None and Path(GROUNDINGDINO_CONFIG).exists():
        try:
            from groundingdino.util.inference import Model as GDModel
            from groundingdino.util.inference import predict as gd_predict

            if not hasattr(run_grounding_dino, "_gd_model"):
                config_path = str(GROUNDINGDINO_CONFIG)
                ckpt_path = str(resolve_existing_path(GROUNDINGDINO_CKPT))
                logger.info(f"[DINO] 加载独立仓库模型: config={config_path}, ckpt={ckpt_path}")
                run_grounding_dino._gd_model = GDModel(
                    model_config_path=config_path,
                    model_checkpoint_path=ckpt_path,
                    device=run_grounding_dino._gd_device if hasattr(run_grounding_dino, "_gd_device") else "cpu",
                )

            gd_model = run_grounding_dino._gd_model

            # 合并 prompt
            text_prompt = ". ".join(class_names) + "."

            boxes, logits, phrases = gd_predict(
                model=gd_model,
                image=str(image_path),
                caption=text_prompt,
                box_threshold=0.0,
                text_threshold=0.0,
            )

            for box, logit, phrase in zip(boxes, logits, phrases):
                score = float(logit.sigmoid().item() if hasattr(logit, 'sigmoid') else logit)
                cls_name = str(phrase).strip().lower()
                for cn in class_names:
                    if cn.lower() in cls_name:
                        cls_name = cn
                        break
                if cls_name not in class_names:
                    continue
                # 已经是 xyxy
                detections.append({
                    "class_name": cls_name,
                    "score": score,
                    "box": [float(coord) for coord in box],
                })

            logger.debug(f"[DINO] {image_path.name}: {len(detections)} raw detections")
            return detections

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[DINO] 独立仓库方式失败: {e}")

    # ── 如果所有方式都失败 ──
    raise RuntimeError(
        "无法加载 GroundingDINO。请确保以下之一可用：\n"
        "  1. `pip install transformers` (HuggingFace GroundingDINO)\n"
        "  2. 安装独立 GroundingDINO 并确保 import groundingdino 可用\n"
        "你也可以修改 `run_grounding_dino` 函数适配你本地的 DINO 调用方式。"
    )


def filter_dino_detections(detections, class_names, thresholds, max_per_class):
    """按类过滤：阈值 + top-k 限制。返回过滤后的 detections 列表。"""
    groups = defaultdict(list)
    for det in detections:
        cls_name = det["class_name"]
        thresh = thresholds.get(cls_name, 0.25)
        if det["score"] >= thresh:
            groups[cls_name].append(det)

    filtered = []
    for cls_name in class_names:
        items = sorted(groups[cls_name], key=lambda x: x["score"], reverse=True)
        max_n = max_per_class.get(cls_name, 100)
        filtered.extend(items[:max_n])

    return filtered


# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════

def build_model(model_type: str, checkpoint_path: Path):
    """与训练脚本 build_model 完全一致（但不加载预训练权重，只搭架构）"""
    from efficient_sam.efficient_sam import build_efficient_sam

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
    logger.info(f"已加载 EfficientSAM 基础权重: {checkpoint_path}")
    if missing_keys:
        logger.warning(f"missing_keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"unexpected_keys: {len(unexpected_keys)}")
    return model


def load_student_model(student_ckpt_path: Path, efficient_sam_ckpt: Path, device):
    """
    加载完整 student 模型：
    1. 用基础 EfficientSAM 权重初始化
    2. 再加载 student checkpoint（兼容纯 state_dict 和带 wrapper 的 dict）
    """
    model = build_model("vitt", efficient_sam_ckpt)

    # 加载 student checkpoint
    checkpoint = torch.load(str(student_ckpt_path), map_location="cpu")
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"已加载 student 权重: {student_ckpt_path}")
    if missing_keys:
        logger.warning(f"missing_keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"unexpected_keys: {len(unexpected_keys)}")

    model.eval().to(device)
    return model


def forward_efficient_sam(model, images, boxes):
    """
    与训练脚本 forward_efficient_sam 完全一致。
    images: [B, 3, 768, 768]
    boxes:  [B, 4] xyxy（letterbox 坐标）
    返回: logits [B, 1, 768, 768]
    """
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

    gather_index = best_idx.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, pred_masks.size(-2), pred_masks.size(-1)
    )
    best_masks = torch.gather(pred_masks, 2, gather_index).squeeze(2)
    if best_masks.ndim != 4:
        raise RuntimeError(f"Unexpected gathered mask shape: {best_masks.shape}")
    return best_masks  # [B, 1, 768, 768]


# ═══════════════════════════════════════════════════════════
# 单图推理
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def infer_one_image(
    model,
    image_path: Path,
    dino_detections: list[dict],
    device: torch.device,
    img_size: int = 768,
    sam_batch_size: int = 32,
) -> list[dict]:
    """
    对一张图的 DINO detections，跑 EfficientSAM 产 mask。

    返回:
        list of dict: [
            {
                "class_name": str,
                "score": float,
                "bbox": [x1, y1, x2, y2],  # 原图坐标
                "area": int,
                "rle": dict,
            },
            ...
        ]
    """
    if not dino_detections:
        return []

    # 1. 灰度预处理 + letterbox
    gray = load_teacher_aligned_gray(image_path)
    orig_h, orig_w = gray.shape[:2]
    gray_pad, meta = letterbox_image(gray, (img_size, img_size), fill_value=0)

    # 2. 将所有 DINO boxes 转成 letterbox 坐标
    lb_boxes = []
    for det in dino_detections:
        box = clip_box_xyxy(det["box"], orig_w, orig_h)
        lb_box = letterbox_box_xyxy(box, meta)
        lb_box = clip_box_xyxy(lb_box, img_size, img_size)
        lb_boxes.append(lb_box)

    # 3. 准备图像 tensor（所有 box 共享同一张图）
    rgb = np.stack([gray_pad, gray_pad, gray_pad], axis=-1).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # [3, 768, 768]
    img_tensor = torch.from_numpy(rgb).float()

    # 4. Batch 跑 EfficientSAM
    all_masks = []
    for start in range(0, len(lb_boxes), sam_batch_size):
        end = min(start + sam_batch_size, len(lb_boxes))
        batch_boxes = lb_boxes[start:end]
        bsz = len(batch_boxes)

        images = img_tensor.unsqueeze(0).repeat(bsz, 1, 1, 1).to(device, non_blocking=True)
        boxes_t = torch.tensor(batch_boxes, dtype=torch.float32, device=device)

        logits = forward_efficient_sam(model, images, boxes_t)  # [bsz, 1, 768, 768]

        for i in range(bsz):
            mask_logit = logits[i, 0]  # [768, 768]
            cls_name = dino_detections[start + i]["class_name"]
            threshold = MASK_THRESHOLDS.get(cls_name, 0.50)
            mask_prob = torch.sigmoid(mask_logit).cpu().numpy()
            mask_bin = (mask_prob > threshold).astype(np.uint8)
            all_masks.append(mask_bin)

    # 5. unletterbox 回原图尺寸 + 后处理
    # 先尝试 import 依赖
    try:
        from scipy.ndimage import label as _ndi_label
    except ImportError:
        _ndi_label = None
    try:
        from pycocotools import mask as _mask_utils
    except ImportError:
        raise ImportError("pycocotools 是必需的，请安装: pip install pycocotools")

    instances = []
    for idx, (det, mask_768) in enumerate(zip(dino_detections, all_masks)):
        cls_name = det["class_name"]
        score = det["score"]

        # 还原到原图
        mask_orig = unletterbox_mask(mask_768, meta)  # [orig_h, orig_w]

        # 小连通域过滤
        min_area = MIN_AREA.get(cls_name, 8)
        if min_area > 0 and mask_orig.sum() > 0 and _ndi_label is not None:
            labeled, num_features = _ndi_label(mask_orig)
            if num_features > 0:
                clean = np.zeros_like(mask_orig)
                kept_any = False
                for lbl in range(1, num_features + 1):
                    component = (labeled == lbl).astype(np.uint8)
                    if component.sum() >= min_area:
                        clean = np.where(component > 0, 1, clean)
                        kept_any = True
                if not kept_any:
                    continue
                mask_orig = clean

        if mask_orig.sum() <= 0:
            continue

        # RLE 编码
        try:
            rle = mask_to_rle(mask_orig)
        except Exception as e:
            logger.warning(f"RLE 编码失败 for {image_path.name} {cls_name}: {e}")
            continue

        # bbox（从 mask 计算）
        ys, xs = np.where(mask_orig > 0)
        if len(ys) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        area = int(mask_orig.sum())

        instances.append({
            "class_name": cls_name,
            "score": float(score),
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "rle": rle,
        })

    # 6. 同类 mask NMS
    if len(instances) > 1:
        masks_np = []
        scores_np = []
        cls_names = []
        for inst in instances:
            try:
                rle_copy = dict(inst["rle"])
                if isinstance(rle_copy["counts"], str):
                    rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
                m = _mask_utils.decode(rle_copy).astype(np.uint8)
                masks_np.append(m)
                scores_np.append(inst["score"])
                cls_names.append(inst["class_name"])
            except Exception:
                masks_np.append(np.ones((1, 1), dtype=np.uint8))
                scores_np.append(inst["score"])
                cls_names.append(inst["class_name"])

        keep_indices = mask_nms(masks_np, scores_np, cls_names, MASK_NMS_IOU)
        instances = [instances[i] for i in keep_indices]

    return instances


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def load_test_image_list(test_tasks_json: Path | None, test_list: Path):
    """加载测试图片列表。优先用 test_tasks.json，其次用 test_list.txt。"""
    if test_tasks_json is not None and test_tasks_json.exists():
        with open(test_tasks_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("tasks", data.get("annotations", data.get("data", [])))
        else:
            items = []

        # 提取唯一 image_path
        image_paths = []
        seen = set()
        for item in items:
            imp = str(item.get("image_path", "")).replace("\\", "/")
            if imp and imp not in seen:
                seen.add(imp)
                image_paths.append(imp)
        logger.info(f"从 {test_tasks_json.name} 读取到 {len(image_paths)} 张测试图")
        return image_paths

    # Fallback: 用 test_list.txt
    if test_list.exists():
        with open(test_list, "r", encoding="utf-8") as f:
            lines = [line.strip().replace("\\", "/") for line in f if line.strip()]
        logger.info(f"从 {test_list.name} 读取到 {len(lines)} 张测试图")
        return lines

    raise FileNotFoundError(f"找不到测试任务文件: {test_tasks_json} 或 {test_list}")


def resolve_image_path(image_root: Path, image_rel_path: str) -> Path:
    """解析图片绝对路径，兼容多种相对路径格式。"""
    image_rel_path = image_rel_path.replace("\\", "/")
    candidates = [
        image_root / image_rel_path,
    ]
    if image_rel_path.startswith("test/"):
        candidates.append(image_root / image_rel_path[len("test/"):])
    if not image_rel_path.startswith("test/"):
        candidates.append(image_root / "test" / image_rel_path)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def main():
    args = parse_args()

    setup_logging(args.output_json.parent / "infer_dino_efficientsam.log" if args.output_json else None)

    device = resolve_device(args.device)

    # ═══ 打印配置 ═══
    logger.info("========== DINO + EfficientSAM 推理 ==========")
    logger.info(f"device: {device}")
    logger.info(f"classes: {CLASSES}")
    logger.info(f"image_root: {args.image_root}")
    logger.info(f"test_list: {args.test_list}")
    logger.info(f"test_tasks_json: {args.test_tasks_json}")
    logger.info(f"efficient_sam_ckpt: {args.efficient_sam_ckpt}")
    logger.info(f"student_ckpt: {args.student_ckpt}")
    logger.info(f"output_json: {args.output_json}")
    logger.info(f"img_size: {args.img_size}")
    logger.info(f"DINO thresholds: {DINO_THRESHOLDS}")
    logger.info(f"GD config: {GROUNDINGDINO_CONFIG}")
    logger.info(f"GD ckpt: {GROUNDINGDINO_CKPT}")
    logger.info(f"Mask thresholds: {MASK_THRESHOLDS}")
    logger.info(f"MIN_AREA: {MIN_AREA}")
    logger.info(f"MASK_NMS_IOU: {MASK_NMS_IOU}")

    # 设置 DINO 模型使用的 device
    run_grounding_dino._hf_device = device
    run_grounding_dino._gd_device = str(device)

    # ═══ 加载 EfficientSAM student 模型 ═══
    logger.info("加载 EfficientSAM student 模型...")
    model = load_student_model(
        student_ckpt_path=resolve_existing_path(args.student_ckpt),
        efficient_sam_ckpt=resolve_existing_path(args.efficient_sam_ckpt),
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"总参数量: {total_params:.2f}M, 可训练参数量: {trainable_params:.2f}M")

    # ═══ 加载测试图列表 ═══
    test_tasks_path = resolve_existing_path(args.test_tasks_json) if args.test_tasks_json else None
    test_list_path = resolve_existing_path(args.test_list)
    test_image_paths = load_test_image_list(test_tasks_path, test_list_path)
    logger.info(f"测试图数量: {len(test_image_paths)}")

    # ═══ 逐图推理 ═══
    results = []
    stats = {
        "total_images": len(test_image_paths),
        "total_instances": 0,
        "per_class_instances": defaultdict(int),
        "per_class_scores": defaultdict(list),
        "empty_images": 0,
        "missing_images": 0,
        "dino_empty_images": 0,
    }
    start_time = time.time()
    img_root = Path(args.image_root)

    for img_idx, image_rel_path in enumerate(test_image_paths):
        abs_path = resolve_image_path(img_root, image_rel_path)

        if not abs_path.exists():
            logger.warning(f"[{img_idx + 1}/{len(test_image_paths)}] 图片不存在: {abs_path}")
            stats["missing_images"] += 1
            results.append({
                "image_path": image_rel_path,
                "instances": [],
            })
            continue

        # Step 1: GroundingDINO → boxes
        try:
            raw_detections = run_grounding_dino(abs_path, CLASSES)
        except Exception as e:
            logger.warning(f"[{img_idx + 1}/{len(test_image_paths)}] DINO 推理失败 {abs_path.name}: {e}")
            stats["dino_empty_images"] += 1
            results.append({
                "image_path": image_rel_path,
                "instances": [],
            })
            continue

        # 按类过滤
        filtered = filter_dino_detections(raw_detections, CLASSES, DINO_THRESHOLDS, MAX_BOXES_PER_CLASS)

        if not filtered:
            stats["dino_empty_images"] += 1
            results.append({
                "image_path": image_rel_path,
                "instances": [],
            })
            if (img_idx + 1) % 100 == 0 or img_idx == 0:
                logger.info(f"[{img_idx + 1}/{len(test_image_paths)}] {abs_path.name}: DINO 未检测到有效框")
            continue

        # Step 2: EfficientSAM → masks
        try:
            instances = infer_one_image(
                model=model,
                image_path=abs_path,
                dino_detections=filtered,
                device=device,
                img_size=args.img_size,
                sam_batch_size=BATCH_SIZE,
            )
        except Exception as e:
            logger.warning(f"[{img_idx + 1}/{len(test_image_paths)}] SAM 推理失败 {abs_path.name}: {e}")
            instances = []

        if not instances:
            stats["empty_images"] += 1

        results.append({
            "image_path": image_rel_path,
            "instances": instances,
        })

        # 统计
        for inst in instances:
            stats["total_instances"] += 1
            stats["per_class_instances"][inst["class_name"]] += 1
            stats["per_class_scores"][inst["class_name"]].append(inst["score"])

        # 进度打印
        if (img_idx + 1) % 50 == 0 or img_idx == 0 or img_idx == len(test_image_paths) - 1:
            elapsed = time.time() - start_time
            speed = (img_idx + 1) / max(elapsed, 1)
            logger.info(
                f"[{img_idx + 1}/{len(test_image_paths)}] {abs_path.name}: "
                f"{len(instances)} instances, {elapsed:.0f}s elapsed, {speed:.1f} img/s"
            )

    elapsed_total = time.time() - start_time
    logger.info(f"推理完成，总耗时: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")

    # ═══ 保存输出 ═══
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": {
                    "student_ckpt": str(args.student_ckpt),
                    "efficient_sam_ckpt": str(args.efficient_sam_ckpt),
                    "classes": CLASSES,
                    "dino_thresholds": DINO_THRESHOLDS,
                    "mask_thresholds": MASK_THRESHOLDS,
                    "mask_nms_iou": MASK_NMS_IOU,
                },
                "timing": {
                    "inference_seconds": float(elapsed_total),
                    "avg_seconds_per_image": float(elapsed_total / max(len(test_image_paths), 1)),
                },
                "stats": {
                    "total_images": stats["total_images"],
                    "total_instances": stats["total_instances"],
                    "per_class_instances": dict(stats["per_class_instances"]),
                    "per_class_avg_score": {
                        cls: float(np.mean(scores)) if scores else 0.0
                        for cls, scores in stats["per_class_scores"].items()
                    },
                    "empty_images": stats["empty_images"],
                    "missing_images": stats["missing_images"],
                    "dino_empty_images": stats["dino_empty_images"],
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ═══ 打印最终统计 ═══
    logger.info("========== 最终统计 ==========")
    logger.info(f"总图片数:       {stats['total_images']}")
    logger.info(f"总 Instance 数:  {stats['total_instances']}")
    logger.info(f"空结果图片数:    {stats['empty_images']}")
    logger.info(f"缺失图片数:      {stats['missing_images']}")
    logger.info(f"DINO 空检图片数: {stats['dino_empty_images']}")
    logger.info("每类 Instance 数:")
    for cls in CLASSES:
        n = stats["per_class_instances"][cls]
        avg_s = float(np.mean(stats["per_class_scores"][cls])) if stats["per_class_scores"][cls] else 0.0
        logger.info(f"  {cls}: {n} instances, avg_score={avg_s:.4f}")
    logger.info(f"输出文件: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINO + EfficientSAM 推理闭环：test image → DINO box → SAM mask → JSON"
    )
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument("--test-tasks-json", default=str(TEST_TASK_JSON) if TEST_TASK_JSON.exists() else "")
    parser.add_argument("--test-list", default=str(TEST_LIST))
    parser.add_argument("--student-ckpt", default=str(STUDENT_CKPT))
    parser.add_argument("--efficient-sam-ckpt", default=str(EFFICIENT_SAM_CKPT))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--groundingdino-config", default=GROUNDINGDINO_CONFIG or "")
    parser.add_argument("--groundingdino-ckpt", default=str(GROUNDINGDINO_CKPT))
    return parser.parse_args()


if __name__ == "__main__":
    main()
