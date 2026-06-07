#!/usr/bin/env python3
"""
YOLO-World + FastSAM 两阶段推理脚本。

用途：
1. 读取 task json，按 image + prompt 批量推理
2. 输出提交格式 predictions.json（ann_id -> rle）
3. 可选导出 prompt 风格 pred json，便于与 teacher/manifest 对比
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("ULTRALYTICS_DOWNLOADS", "false")

from config_world_new import (
    CLASSES,
    DEVICE,
    Fastsam_CONF,
    Fastsam_IMG_SIZE,
    Fastsam_IOU,
    Fastsam_MODEL,
    IMAGE_ROOT,
    IMG_SIZE,
    MS_CONF,
    MS_NMS_IOU,
    MULTI_SCALE_CLASSES,
    MULTI_SCALES,
    PROJECT,
    PROMPT_MAP,
    RUN_NAME,
    YOLO_MODEL,
)

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


DEFAULT_TASKS = ROOT / "test" / "json" / "val_tasks.json"
DEFAULT_IMAGE_ROOT = IMAGE_ROOT
DEFAULT_CHECKPOINT = PROJECT / RUN_NAME / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "inference_eval" / RUN_NAME
DEFAULT_DET_CONF = 0.25
DEFAULT_MAX_IMAGES = 0
TORCH_DEVICE = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════
# 用户配置区
# ══════════════════════════════════════════════════════════════════════

INFER_TASKS = DEFAULT_TASKS
INFER_IMAGE_ROOT = DEFAULT_IMAGE_ROOT
INFER_CHECKPOINT = DEFAULT_CHECKPOINT
INFER_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
INFER_DET_CONF = DEFAULT_DET_CONF
INFER_MAX_IMAGES = DEFAULT_MAX_IMAGES
INFER_DEVICE = DEVICE
INFER_CLASSES = CLASSES

_LOCAL_CLIP_DIR = ROOT / "model" / "clip"
_DEFAULT_CLIP_CACHE = Path.home() / ".cache" / "clip"


def _ensure_local_clip() -> None:
    _DEFAULT_CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    target = _DEFAULT_CLIP_CACHE / "ViT-B-32.pt"
    if target.exists():
        return

    local_pt = _LOCAL_CLIP_DIR / "ViT-B-32.pt"
    if local_pt.exists():
        shutil.copy2(str(local_pt), str(target))
        return

    import clip as _clip

    _LOCAL_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    _clip.load("ViT-B/32", download_root=str(_LOCAL_CLIP_DIR))
    local_pt = _LOCAL_CLIP_DIR / "ViT-B-32.pt"
    if local_pt.exists():
        shutil.copy2(str(local_pt), str(target))


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
    if mask_utils is None:
        return simple_rle_encode(binary_mask)

    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle, list):
        rle = rle[0]
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("utf-8")
        if isinstance(rle["counts"], bytes)
        else rle["counts"],
    }


def build_empty_rle(height: int, width: int) -> Dict[str, Any]:
    return mask_to_rle(np.zeros((height, width), dtype=np.uint8))


def unify_polarity(gray: np.ndarray, fname: str) -> np.ndarray:
    if "blackHot" in fname:
        return 255 - gray
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < -0.3:
            return 255 - gray
    if mean > 200 and "vis" not in fname.lower():
        return 255 - gray
    return gray


def load_rgb_for_world(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"图片不存在或无法读取: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = unify_polarity(img_gray, image_path.name)
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def compute_iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def align_imgsz(value: int, stride: int = 32) -> int:
    return max(stride, ((int(value) + stride - 1) // stride) * stride)


def resolve_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    return PROMPT_MAP.get(prompt, prompt)


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    return tasks


def load_models(checkpoint_path: Path):
    _ensure_local_clip()
    from ultralytics import FastSAM, YOLO

    det_model = YOLO(str(checkpoint_path), task="detect")
    device_str = f"cuda:{INFER_DEVICE}" if isinstance(INFER_DEVICE, int) and torch.cuda.is_available() else str(TORCH_DEVICE)
    det_model.to(device_str)
    seg_model = FastSAM(Fastsam_MODEL)
    return det_model, seg_model


def multi_scale_detect(det_model, img_rgb: np.ndarray, prompt: str) -> List[Tuple[List[float], float]]:
    all_boxes: List[Tuple[List[float], float]] = []
    det_model.set_classes([prompt])
    device = f"cuda:{INFER_DEVICE}" if isinstance(INFER_DEVICE, int) and torch.cuda.is_available() else str(TORCH_DEVICE)
    for scale in MULTI_SCALES:
        imgsz = align_imgsz(int(IMG_SIZE * scale))
        results = det_model.predict(
            img_rgb,
            imgsz=imgsz,
            conf=MS_CONF,
            device=device,
            verbose=False,
        )
        if results and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy.cpu().numpy().tolist()
            confs = results[0].boxes.conf.cpu().numpy().tolist()
            for box, score in zip(xyxy, confs):
                all_boxes.append((box, float(score)))

    if not all_boxes:
        return []

    all_boxes.sort(key=lambda item: item[1], reverse=True)
    kept: List[Tuple[List[float], float]] = []
    for box, score in all_boxes:
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(compute_iou_xyxy(tuple(box), tuple(prev_box)) > MS_NMS_IOU for prev_box, _ in kept):
            continue
        kept.append((box, score))
    return kept


def single_scale_detect(det_model, img_rgb: np.ndarray, prompt: str, conf: float) -> List[Tuple[List[float], float]]:
    det_model.set_classes([prompt])
    device = f"cuda:{INFER_DEVICE}" if isinstance(INFER_DEVICE, int) and torch.cuda.is_available() else str(TORCH_DEVICE)
    results = det_model.predict(
        img_rgb,
        imgsz=IMG_SIZE,
        conf=conf,
        device=device,
        verbose=False,
    )
    if not results or len(results[0].boxes) == 0:
        return []
    xyxy = results[0].boxes.xyxy.cpu().numpy().tolist()
    confs = results[0].boxes.conf.cpu().numpy().tolist()
    return [(box, float(score)) for box, score in zip(xyxy, confs)]


def segment_boxes(seg_model, img_rgb: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    device = f"cuda:{INFER_DEVICE}" if isinstance(INFER_DEVICE, int) and torch.cuda.is_available() else str(TORCH_DEVICE)

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            continue
        crop = img_rgb[y1:y2, x1:x2]
        dynamic_imgsz = max(256, min(Fastsam_IMG_SIZE, max(crop_h, crop_w)))
        dynamic_imgsz = ((dynamic_imgsz + 31) // 32) * 32
        fs_results = seg_model.predict(
            source=crop,
            imgsz=dynamic_imgsz,
            conf=Fastsam_CONF,
            iou=Fastsam_IOU,
            device=device,
            verbose=False,
            retina_masks=True,
        )
        if fs_results and fs_results[0].masks is not None:
            for cm_tensor in fs_results[0].masks.data:
                cm = cm_tensor.cpu().numpy().astype(np.uint8)
                cm = cv2.resize(cm, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                pred_mask[y1:y2, x1:x2] |= cm
    return pred_mask


def infer_prompt(det_model, seg_model, img_rgb: np.ndarray, prompt: str, det_conf: float) -> Dict[str, Any]:
    resolved_prompt = resolve_prompt(prompt)
    use_ms = resolved_prompt in MULTI_SCALE_CLASSES
    detections = (
        multi_scale_detect(det_model, img_rgb, resolved_prompt)
        if use_ms
        else single_scale_detect(det_model, img_rgb, resolved_prompt, det_conf)
    )
    boxes = [box for box, _ in detections]
    if not boxes:
        return {"hit": False}

    mask = segment_boxes(seg_model, img_rgb, boxes)
    if int(mask.sum()) <= 0:
        return {"hit": False}

    scores = [score for _, score in detections]
    return {
        "hit": True,
        "score": round(float(max(scores)), 4),
        "instances": len(boxes),
        "rle": mask_to_rle(mask),
        "det_prompt": resolved_prompt,
        "multi_scale": use_ms,
    }


def run_inference(args: argparse.Namespace) -> None:
    if isinstance(INFER_DEVICE, int) and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，world 推理脚本默认要求 GPU")

    tasks = load_tasks(args.tasks)
    det_model, seg_model = load_models(args.checkpoint)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = f"cuda:{INFER_DEVICE}" if isinstance(INFER_DEVICE, int) and torch.cuda.is_available() else str(TORCH_DEVICE)
    print("=" * 60)
    print("YOLO-World + FastSAM 推理配置")
    print("=" * 60)
    print(f"tasks:      {args.tasks}")
    print(f"image_root: {args.image_root}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print(f"classes:    {INFER_CLASSES}")
    print(f"device:     {device_str}")
    print(f"det_conf:   {args.det_conf}")
    print(f"max_images: {args.max_images}")
    print("=" * 60)

    tasks_by_image = defaultdict(list)
    for task in tasks:
        image_path = str(task["image_path"]).replace("\\", "/")
        tasks_by_image[image_path].append(task)

    predictions_output = []
    prompt_style_output = []
    timing = {"processed_images": 0, "total_tasks": len(tasks), "inference_seconds": 0.0}

    image_items = list(tasks_by_image.items())
    if args.max_images and args.max_images > 0:
        image_items = image_items[: args.max_images]

    for image_rel_path, image_tasks in tqdm(image_items, desc="world infer", unit="img"):
        image_abs_path = args.image_root / image_rel_path
        if not image_abs_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        img_rgb = load_rgb_for_world(image_abs_path)
        orig_h, orig_w = img_rgb.shape[:2]
        prompts = list(
            dict.fromkeys(
                str(task.get("text_prompt", "")).strip()
                for task in image_tasks
                if str(task.get("text_prompt", "")).strip()
            )
        )

        t0 = time.time()
        prompt_iter = tqdm(prompts, desc=f"  prompts {Path(image_rel_path).name}", unit="prompt", leave=False)
        results_by_prompt = {
            prompt: infer_prompt(det_model, seg_model, img_rgb, prompt, args.det_conf)
            for prompt in prompt_iter
        }
        elapsed = time.time() - t0
        timing["processed_images"] += 1
        timing["inference_seconds"] += elapsed

        empty_rle = build_empty_rle(orig_h, orig_w)
        prompt_style_output.append({"image_path": image_rel_path, "prompts": results_by_prompt})

        for task in image_tasks:
            ann_id = int(task["ann_id"])
            prompt = str(task.get("text_prompt", "")).strip()
            pred = results_by_prompt.get(prompt)
            rle = pred["rle"] if pred and pred.get("hit") else empty_rle
            predictions_output.append({"ann_id": ann_id, "rle": rle})

    avg_time = timing["inference_seconds"] / max(timing["processed_images"], 1)
    timing["avg_inference_seconds_per_image"] = float(avg_time)

    predictions_path = output_dir / f"{args.tasks.stem}_predictions.json"
    prompt_path = output_dir / f"pred_{args.tasks.stem}_world.json"
    meta_path = output_dir / f"{args.tasks.stem}_meta.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": {
                    "checkpoint": str(args.checkpoint),
                    "run_name": RUN_NAME,
                    "classes": CLASSES,
                    "yolo_model": str(args.checkpoint),
                    "fastsam_model": Fastsam_MODEL,
                    "imgsz": IMG_SIZE,
                    "multi_scale_classes": MULTI_SCALE_CLASSES,
                    "multi_scales": MULTI_SCALES,
                },
                "timing": timing,
                "predictions": predictions_output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_style_output, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": str(args.tasks),
                "image_root": str(args.image_root),
                "checkpoint": str(args.checkpoint),
                "output_dir": str(output_dir),
                "processed_images": timing["processed_images"],
                "total_tasks": timing["total_tasks"],
                "avg_inference_seconds_per_image": timing["avg_inference_seconds_per_image"],
                "det_conf": args.det_conf,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"完成: images={timing['processed_images']}, tasks={timing['total_tasks']}")
    print(f"predictions: {predictions_path}")
    print(f"prompt_json:  {prompt_path}")
    print(f"meta:         {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-World + FastSAM 两阶段推理脚本")
    parser.add_argument("--tasks", type=Path, default=INFER_TASKS, help="任务 json，例如 test/json/val_tasks1.json")
    parser.add_argument("--image-root", type=Path, default=INFER_IMAGE_ROOT, help="图片根目录")
    parser.add_argument("--checkpoint", type=Path, default=INFER_CHECKPOINT, help="YOLO-World best.pt 路径")
    parser.add_argument("--output-dir", type=Path, default=INFER_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--det-conf", type=float, default=INFER_DET_CONF, help="单尺度检测置信度阈值")
    parser.add_argument("--max-images", type=int, default=INFER_MAX_IMAGES, help="可选，仅推理前 N 张图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
