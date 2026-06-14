#!/usr/bin/env python3
"""
CLIPSeg 教师 soft mask 缓存脚本。

修复点：
1. teacher 正确按 base_model_dir + checkpoint 加载
2. 不依赖 clipseg_train_base，脚本自包含
3. 缓存文件名统一为 {ann_id}__{class_name}.npy
4. 直接保存训练对齐后的 768x768 soft mask，避免蒸馏错位
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from config_clipseg import (
    BLACKHOT_MEAN_THRESH,
    BLACKHOT_SKEW_THRESH,
    BLUR_LOW,
    BLUR_MID,
    CLASSES,
    CLIPSEG_TEACHER_BASE_MODEL_DIR,
    CLIPSEG_TEACHER_CHECKPOINT,
    GRAY2RGB,
    IMAGE_ROOT,
    IMG_SIZE,
    MANIFEST_JSON,
    NOISE_HIGH,
    NOISE_MED,
    PRED_JSON,
    PSEUDO_COLOR_SAT_THRESH,
    STD_LOW,
    STD_MID,
    TRAIN_LIST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

CLASSES_11 = [cls_name for cls_name in CLASSES if cls_name != "computer"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_class_name(class_name: str) -> str:
    return class_name.replace(" ", "_").replace("/", "_")


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model", "model_state_dict", "state_dict", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_like = [torch.is_tensor(v) for v in checkpoint.values()]
            if tensor_like and any(tensor_like):
                return checkpoint
    raise RuntimeError("无法从 teacher checkpoint 解析 state_dict。")


def load_clipseg_teacher(
    base_model_dir: Path,
    teacher_checkpoint: Optional[Path] = None,
) -> Tuple[CLIPSegForImageSegmentation, CLIPSegProcessor]:
    if not base_model_dir.exists():
        raise FileNotFoundError(f"CLIPSeg base_model_dir 不存在: {base_model_dir}")

    logger.info(f"从 base model 加载 CLIPSeg: {base_model_dir}")
    processor = CLIPSegProcessor.from_pretrained(str(base_model_dir))
    model = CLIPSegForImageSegmentation.from_pretrained(str(base_model_dir))

    if teacher_checkpoint is not None:
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"teacher checkpoint 不存在: {teacher_checkpoint}")
        logger.info(f"加载 teacher checkpoint: {teacher_checkpoint}")
        ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"teacher checkpoint missing keys: {missing[:10]}")
        if unexpected:
            logger.warning(f"teacher checkpoint unexpected keys: {unexpected[:10]}")

    model = model.to(DEVICE).eval()
    return model, processor


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path: Path) -> np.ndarray:
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


def build_teacher_canvas(image_path: Path) -> np.ndarray:
    gray = load_teacher_aligned_gray(image_path)
    orig_h, orig_w = gray.shape
    scale = IMG_SIZE / max(orig_h, orig_w)
    resized_h = max(1, int(round(orig_h * scale)))
    resized_w = max(1, int(round(orig_w * scale)))
    gray = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h = IMG_SIZE - resized_h
    pad_w = IMG_SIZE - resized_w
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

    if GRAY2RGB:
        rgb = np.stack([gray] * 3, axis=-1)
    else:
        rgb = gray[:, :, None]
    return rgb.astype(np.uint8, copy=False)


@torch.no_grad()
def generate_soft_mask_from_path(
    model: CLIPSegForImageSegmentation,
    processor: CLIPSegProcessor,
    img_path: Path,
    class_name: str,
) -> Optional[np.ndarray]:
    try:
        rgb_pad = build_teacher_canvas(img_path)
        inputs = processor(
            text=[class_name],
            images=rgb_pad,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(DEVICE)

        logits = model(**inputs).logits
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        soft_prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy().astype(np.float32)
        return soft_prob
    except Exception as exc:
        logger.warning(f"生成 soft mask 失败 {img_path} {class_name}: {exc}")
        return None


def iter_manifest_records(raw_data: Any, allowed: set[str]) -> Iterable[Tuple[str, str, str]]:
    records = raw_data.get("records", []) if isinstance(raw_data, dict) else []
    for record in records:
        img_path = record["image_path"].replace("\\", "/")
        if img_path not in allowed:
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allowed:
                continue
            img_path = alt
        prompt = record.get("prompt")
        ann_id = record.get("ann_id")
        if prompt == "computer" or prompt not in CLASSES_11:
            continue
        if not record.get("include_in_train") or not record.get("selected_hit"):
            continue
        cache_key = str(int(ann_id)) if ann_id is not None else Path(img_path).stem
        yield img_path, prompt, cache_key


def iter_pred_records(raw_data: Any, allowed: set[str]) -> Iterable[Tuple[str, str, str]]:
    preds = raw_data
    for item in preds:
        img_path = item["image_path"].replace("\\", "/")
        if img_path not in allowed:
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allowed:
                continue
            img_path = alt
        ann_id = item.get("ann_id")
        cache_key = str(int(ann_id)) if ann_id is not None else Path(img_path).stem
        for prompt, prompt_info in item.get("prompts", {}).items():
            if prompt == "computer" or prompt not in CLASSES_11:
                continue
            if not prompt_info.get("hit"):
                continue
            yield img_path, prompt, cache_key


def generate_cache(
    output_dir: Path,
    pred_json: Path,
    train_list: Path,
    use_manifest: bool = False,
    base_model_dir: Optional[Path] = None,
    teacher_checkpoint: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_clipseg_teacher(
        base_model_dir=base_model_dir or CLIPSEG_TEACHER_BASE_MODEL_DIR,
        teacher_checkpoint=teacher_checkpoint or CLIPSEG_TEACHER_CHECKPOINT,
    )

    with open(pred_json, encoding="utf-8") as file_obj:
        raw_data = json.load(file_obj)
    with open(train_list, encoding="utf-8") as file_obj:
        allowed = {line.strip().replace("\\", "/") for line in file_obj if line.strip()}

    iterator = iter_manifest_records(raw_data, allowed) if use_manifest else iter_pred_records(raw_data, allowed)
    records = list(iterator)

    logger.info(f"开始生成 soft mask 缓存到: {output_dir}")
    logger.info(f"teacher base model: {base_model_dir or CLIPSEG_TEACHER_BASE_MODEL_DIR}")
    logger.info(f"teacher checkpoint: {teacher_checkpoint or CLIPSEG_TEACHER_CHECKPOINT}")
    logger.info("缓存格式: {ann_id_or_image_stem}__{class_name}.npy")
    logger.info(f"保存尺寸: {IMG_SIZE}x{IMG_SIZE}, dtype=float16")
    logger.info(f"overwrite: {overwrite}")

    cached_count = 0
    skipped_count = 0
    failed_count = 0
    failed_items: List[Dict[str, Any]] = []

    pbar = tqdm(records, desc="生成 soft mask", total=len(records))
    for img_path, prompt, cache_key in pbar:
        abs_img_path = IMAGE_ROOT / img_path
        if not abs_img_path.exists():
            skipped_count += 1
            continue

        cache_path = output_dir / f"{cache_key}__{safe_class_name(prompt)}.npy"
        if cache_path.exists() and not overwrite:
            skipped_count += 1
            pbar.set_postfix(cached=cached_count, skipped=skipped_count, failed=failed_count)
            continue

        soft_prob = generate_soft_mask_from_path(model, processor, abs_img_path, prompt)
        if soft_prob is None:
            failed_items.append({"image_path": img_path, "class": prompt, "cache_key": cache_key})
            failed_count += 1
        else:
            np.save(cache_path, soft_prob.astype(np.float16))
            cached_count += 1

        pbar.set_postfix(cached=cached_count, skipped=skipped_count, failed=failed_count)

    logger.info(f"缓存生成完成: cached={cached_count}, skipped={skipped_count}, failed={failed_count}")
    if failed_items:
        failed_json = output_dir / "failed_items.json"
        with open(failed_json, "w", encoding="utf-8") as file_obj:
            json.dump(failed_items, file_obj, ensure_ascii=False, indent=2)
        logger.warning(f"失败样本列表已写入: {failed_json}")

    summary = {
        "teacher_checkpoint": str(teacher_checkpoint or CLIPSEG_TEACHER_CHECKPOINT),
        "base_model_dir": str(base_model_dir or CLIPSEG_TEACHER_BASE_MODEL_DIR),
        "pred_json": str(pred_json),
        "train_list": str(train_list),
        "num_cached": cached_count,
        "num_skipped": skipped_count,
        "num_failed": failed_count,
        "classes": CLASSES_11,
        "img_size": IMG_SIZE,
        "device": DEVICE,
        "overwrite": overwrite,
    }
    with open(output_dir / "cache_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CLIPSeg teacher soft mask 缓存")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "test" / "train_output" / "clipseg_soft_cache_11",
        help="输出缓存目录",
    )
    parser.add_argument("--pred_json", type=Path, default=PRED_JSON, help="训练标签 JSON")
    parser.add_argument("--train_list", type=Path, default=TRAIN_LIST, help="训练集图片列表")
    parser.add_argument("--use_manifest", action="store_true", help="使用 manifest records 格式")
    parser.add_argument(
        "--base_model_dir",
        type=Path,
        default=CLIPSEG_TEACHER_BASE_MODEL_DIR,
        help="CLIPSeg HuggingFace base model 目录",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=Path,
        default=CLIPSEG_TEACHER_CHECKPOINT,
        help="teacher checkpoint 路径",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的缓存文件")
    args = parser.parse_args()

    generate_cache(
        output_dir=args.output_dir,
        pred_json=args.pred_json,
        train_list=args.train_list,
        use_manifest=args.use_manifest,
        base_model_dir=args.base_model_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
