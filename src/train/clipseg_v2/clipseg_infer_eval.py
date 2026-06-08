#!/usr/bin/env python3
"""
CLIPSeg 验证集/任务集推理脚本。

用途：
1. 读取 task json，按 image + prompt 批量推理
2. 输出提交格式 predictions.json（ann_id -> rle）
3. 可选导出 prompt 风格 pred json，便于与 teacher 伪标签直接对比

说明：
- 复用当前 clipseg 训练配置中的五类阈值、图像预处理和模型结构
- 默认加载 test/train_output/clipseg_v1/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    CLIP_MEAN,
    CLIP_STD,
    DEVICE,
    GRAY2RGB,
    IMAGE_ROOT,
    IMG_SIZE,
    MODEL_DIR,
    MODEL_NAME,
    PROJECT,
    PSEUDO_COLOR_SAT_THRESH,
    RUN_NAME,
    STD_LOW,
    STD_MID,
    NOISE_HIGH,
    NOISE_MED,
    PROMPT_THRESHOLDS,
)

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


DEFAULT_TASKS = ROOT / "test" / "json" / "val_tasks1.json"
DEFAULT_IMAGE_ROOT = IMAGE_ROOT
DEFAULT_CHECKPOINT = PROJECT / RUN_NAME / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "inference_eval" / RUN_NAME
TORCH_DEVICE = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")


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


def load_clipseg_model(checkpoint_path: Path):
    if MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        processor = CLIPSegProcessor.from_pretrained(str(MODEL_DIR))
        model = CLIPSegForImageSegmentation.from_pretrained(str(MODEL_DIR))
    else:
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)

    ckpt = torch.load(checkpoint_path, map_location=TORCH_DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(TORCH_DEVICE)
    model.eval()
    return model, processor, ckpt


def is_pseudo_color(img_bgr):
    import cv2

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    import cv2

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path: Path):
    import cv2

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


def preprocess_for_clipseg(image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    gray = load_teacher_aligned_gray(image_path)
    orig_h, orig_w = gray.shape

    scale = IMG_SIZE / max(orig_h, orig_w)
    nh, nw = int(orig_h * scale), int(orig_w * scale)
    img = np.array(gray, copy=False)
    import cv2

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)

    if GRAY2RGB:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = img[:, :, None]

    img = img.astype(np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(CLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w)


def postprocess_mask(
    logits: torch.Tensor,
    orig_hw: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    import cv2

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8)
    orig_h, orig_w = orig_hw

    scale = IMG_SIZE / max(orig_h, orig_w)
    nh, nw = int(orig_h * scale), int(orig_w * scale)
    ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2

    pred = pred[pt : pt + nh, pl : pl + nw]
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (pred > 0).astype(np.uint8)


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    return tasks


@torch.inference_mode()
def infer_image_prompts(
    image_path: Path,
    prompts: List[str],
    model,
    processor,
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[int, int]]:
    pixel_values, orig_hw = preprocess_for_clipseg(image_path)
    pixel_values = pixel_values.to(TORCH_DEVICE)

    results = {}
    for prompt in prompts:
        tokenized = processor.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
        logits = model(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"].to(TORCH_DEVICE),
            attention_mask=tokenized["attention_mask"].to(TORCH_DEVICE),
        ).logits
        if logits.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        threshold = float(PROMPT_THRESHOLDS.get(prompt, 0.5))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        score = float(prob.max())
        mask = postprocess_mask(logits, orig_hw, threshold)

        if mask.sum() > 0:
            results[prompt] = {
                "hit": True,
                "score": round(score, 4),
                "instances": 1,
                "rle": mask_to_rle(mask),
            }
        else:
            results[prompt] = {"hit": False}
    return results, orig_hw


def run_eval(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.tasks)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, ckpt = load_clipseg_model(args.checkpoint)
    tasks_by_image = defaultdict(list)
    for task in tasks:
        image_path = str(task["image_path"]).replace("\\", "/")
        tasks_by_image[image_path].append(task)

    predictions_output = []
    prompt_style_output = []
    timing = {"processed_images": 0, "total_tasks": len(tasks), "inference_seconds": 0.0}

    for image_rel_path, image_tasks in tasks_by_image.items():
        image_abs_path = args.image_root / image_rel_path
        if not image_abs_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        prompts = list(
            dict.fromkeys(
                str(task.get("text_prompt", "")).strip()
                for task in image_tasks
                if str(task.get("text_prompt", "")).strip()
            )
        )

        t0 = time.time()
        results_by_prompt, (orig_h, orig_w) = infer_image_prompts(
            image_path=image_abs_path,
            prompts=prompts,
            model=model,
            processor=processor,
        )
        elapsed = time.time() - t0
        timing["processed_images"] += 1
        timing["inference_seconds"] += elapsed

        empty_rle = build_empty_rle(orig_h, orig_w)
        prompt_entry = {"image_path": image_rel_path, "prompts": results_by_prompt}
        prompt_style_output.append(prompt_entry)

        for task in image_tasks:
            ann_id = int(task["ann_id"])
            prompt = str(task.get("text_prompt", "")).strip()
            pred = results_by_prompt.get(prompt)
            rle = pred["rle"] if pred and pred.get("hit") else empty_rle
            predictions_output.append({"ann_id": ann_id, "rle": rle})

    avg_time = timing["inference_seconds"] / max(timing["processed_images"], 1)
    timing["avg_inference_seconds_per_image"] = float(avg_time)

    predictions_path = output_dir / f"{args.tasks.stem}_predictions.json"
    prompt_path = output_dir / f"pred_{args.tasks.stem}_clipseg.json"
    meta_path = output_dir / f"{args.tasks.stem}_meta.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": {
                    "checkpoint": str(args.checkpoint),
                    "run_name": RUN_NAME,
                    "classes": CLASSES,
                    "prompt_thresholds": PROMPT_THRESHOLDS,
                    "epoch": ckpt.get("epoch"),
                    "best_miou": ckpt.get("best_miou"),
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"完成: images={timing['processed_images']}, tasks={timing['total_tasks']}")
    print(f"predictions: {predictions_path}")
    print(f"prompt_json:  {prompt_path}")
    print(f"meta:         {meta_path}")

    if args.reference:
        print("\n接下来可直接跑分析：")
        print(
            f"D:\\anconda3\\envs\\rayton\\python.exe src\\prompt\\analyze_prompt_predictions.py "
            f"--tasks {args.tasks} "
            f"--predictions {predictions_path} "
            f"--reference {args.reference} "
            f"--image-root {args.image_root}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="CLIPSeg 任务集推理评估脚本")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="任务 json，例如 test/json/val_tasks1.json")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="图片根目录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练输出 best.pt 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--reference", type=Path, default=None, help="可选，teacher prompt json，用于后续分析提醒")
    return parser.parse_args()


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
