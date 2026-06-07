#!/usr/bin/env python3
"""
基于官方 SAM3 推理代码生成 prompt_test 风格伪标签。

设计原则：
1. 尽量复用官方推理链路：build_sam3_image_model + Sam3Processor
2. 保留你之前在 prompt_test 中验证过的红外预处理
3. 输出兼容现有训练流程的 pred_*.json / stats_*.csv / kept_images.txt

说明：
- 本脚本不是提交推理入口，而是 teacher 伪标签生成工具
- 结果格式对齐 test/prompt_test_output 下既有伪标签格式
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path(r"D:\nyz\sam3\选手示例工程\to_Df - 官方")
DEFAULT_TASKS = ROOT / "test" / "json" / "train_tasks.json"
DEFAULT_IMAGE_ROOT = ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "test" / "prompt_test_output"
DEFAULT_CHECKPOINT = DEFAULT_OFFICIAL_ROOT / "model" / "sam3.pt"
DEFAULT_PROMPT_THRESHOLD = 0.60
DEFAULT_PROCESSOR_CONF_THRESHOLD = 0.50
DEFAULT_RAW_STATS_THRESHOLD = 0.01
DEFAULT_SKIP_LIST = ROOT / "noisy_data" / "unified_denylist.txt"
DEFAULT_OUTPUT_THRESHOLDS = {
    "person": 0.70,
    "car": 0.70,
    "building": 0.70,
    "tree": 0.60,
    "animal": 0.60,
}

# 预处理参数，沿用 prompt_test 的经验值
PSEUDO_COLOR_SAT_THRESH = 60.0
BLACKHOT_SKEW_THRESH = -0.3
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def import_official_sam3(official_root: Path):
    if not official_root.exists():
        raise FileNotFoundError(f"官方 SAM3 根目录不存在: {official_root}")

    official_root_str = str(official_root)
    if official_root_str not in sys.path:
        sys.path.insert(0, official_root_str)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    return build_sam3_image_model, Sam3Processor


try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


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


def normalize_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def aggregate_prompt_prediction(
    state: Dict[str, Any], prompt: str, conf_threshold: float
) -> Optional[Dict[str, Any]]:
    """
    直接改写自官方 inference.py 的聚合逻辑。
    """
    masks = state.get("masks")
    scores = state.get("scores")

    if masks is None or len(masks) == 0:
        return None

    if scores is None or len(scores) == 0:
        scores_np = np.zeros(len(masks), dtype=np.float32)
    else:
        scores_np = scores.detach().cpu().numpy().reshape(-1)

    selected_indices = [
        index for index, score in enumerate(scores_np) if float(score) >= conf_threshold
    ]
    if not selected_indices:
        return None

    merged_mask: Optional[np.ndarray] = None
    for index in selected_indices:
        mask_np = normalize_mask(masks[index])
        if merged_mask is None:
            merged_mask = mask_np
        else:
            merged_mask = np.logical_or(merged_mask, mask_np).astype(np.uint8)

    if merged_mask is None:
        return None

    return {
        "prompt": prompt,
        "score": float(np.max(scores_np[selected_indices])) if len(scores_np) > 0 else 0.0,
        "instance_count": len(selected_indices),
        "rle": mask_to_rle(merged_mask),
    }


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def should_invert(gray: np.ndarray, fname: str) -> bool:
    if "blackHot" in fname:
        return True
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < BLACKHOT_SKEW_THRESH:
            return True
    if mean > 200 and "vis" not in fname.lower():
        return True
    return False


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def preprocess_image(abs_path: str) -> Tuple[Image.Image, Dict[str, float]]:
    """
    沿用 prompt_test 的自适应预处理，但直接返回 PIL.Image，
    避免写临时文件。
    """
    img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {abs_path}")
    elif is_pseudo_color(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if should_invert(img, os.path.basename(abs_path)):
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

    rgb = np.stack([img] * 3, axis=-1)
    pil_image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    metrics = {
        "std": round(std, 3),
        "noise_sigma": round(noise_sigma, 3),
        "blur_score": round(blur_score, 3),
    }
    return pil_image, metrics


def load_image(abs_path: str, enable_preprocess: bool) -> Tuple[Image.Image, Dict[str, float]]:
    if enable_preprocess:
        return preprocess_image(abs_path)
    image = Image.open(abs_path).convert("RGB")
    return image, {}


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    return tasks


def load_skip_set(skip_list_path: Optional[Path]) -> set[str]:
    if skip_list_path is None or not skip_list_path.exists():
        return set()
    with open(skip_list_path, "r", encoding="utf-8") as f:
        return {
            line.strip().replace("\\", "/")
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }


def get_output_threshold(prompt: str, default_threshold: float, overrides: Dict[str, float]) -> float:
    return float(overrides.get(prompt, default_threshold))


def load_threshold_overrides(raw_json: Optional[str]) -> Dict[str, float]:
    if not raw_json:
        return dict(DEFAULT_OUTPUT_THRESHOLDS)
    parsed = json.loads(raw_json)
    return {str(k): float(v) for k, v in parsed.items()}


def load_model(
    official_root: Path,
    checkpoint_path: Path,
    processor_conf_threshold: float,
):
    build_sam3_image_model, Sam3Processor = import_official_sam3(official_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")

    model = build_sam3_image_model(checkpoint_path=str(checkpoint_path), device=DEVICE)
    processor = Sam3Processor(
        model=model,
        device=DEVICE,
        confidence_threshold=processor_conf_threshold,
    )
    return model, processor


@torch.inference_mode()
def infer_prompts_for_image(
    image: Image.Image,
    prompts: Sequence[str],
    processor,
    raw_stats_threshold: float,
    default_threshold: float,
    threshold_overrides: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    state = processor.set_image(image)
    raw_results: Dict[str, Dict[str, Any]] = {}
    filtered_results: Dict[str, Dict[str, Any]] = {}

    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        raw_prediction = aggregate_prompt_prediction(state, prompt, raw_stats_threshold)
        if raw_prediction is not None:
            raw_results[prompt] = raw_prediction

        output_threshold = get_output_threshold(prompt, default_threshold, threshold_overrides)
        filtered_prediction = aggregate_prompt_prediction(state, prompt, output_threshold)
        if filtered_prediction is not None:
            filtered_results[prompt] = filtered_prediction

    return raw_results, filtered_results


def write_stats_csv(
    stats_path: Path,
    prompts: Sequence[str],
    prompt_seen: Dict[str, int],
    raw_prompt_hits: Dict[str, List[Dict[str, Any]]],
    raw_prompt_misses: Dict[str, int],
    filtered_prompt_hits: Dict[str, List[Dict[str, Any]]],
    filtered_prompt_misses: Dict[str, int],
) -> None:
    with open(stats_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt",
                "total_images",
                "raw_hits",
                "raw_misses",
                "raw_activation_rate",
                "raw_avg_score",
                "raw_avg_instances",
                "raw_total_instances",
                "filtered_hits",
                "filtered_misses",
                "filtered_activation_rate",
                "filtered_avg_score",
                "filtered_avg_instances",
                "filtered_total_instances",
            ]
        )
        for prompt in prompts:
            total_img = prompt_seen[prompt]
            raw_hits = len(raw_prompt_hits.get(prompt, []))
            raw_scores = [h["score"] for h in raw_prompt_hits.get(prompt, [])]
            raw_instances = [h["instances"] for h in raw_prompt_hits.get(prompt, [])]
            filtered_hits = len(filtered_prompt_hits.get(prompt, []))
            filtered_scores = [h["score"] for h in filtered_prompt_hits.get(prompt, [])]
            filtered_instances = [h["instances"] for h in filtered_prompt_hits.get(prompt, [])]

            writer.writerow(
                [
                    prompt,
                    total_img,
                    raw_hits,
                    raw_prompt_misses.get(prompt, 0),
                    round(raw_hits / max(total_img, 1), 4),
                    round(float(np.mean(raw_scores)), 4) if raw_scores else 0,
                    round(float(np.mean(raw_instances)), 2) if raw_instances else 0,
                    sum(raw_instances),
                    filtered_hits,
                    filtered_prompt_misses.get(prompt, 0),
                    round(filtered_hits / max(total_img, 1), 4),
                    round(float(np.mean(filtered_scores)), 4) if filtered_scores else 0,
                    round(float(np.mean(filtered_instances)), 2) if filtered_instances else 0,
                    sum(filtered_instances),
                ]
            )


def run_tasks(args: argparse.Namespace) -> None:
    if args.require_cuda and DEVICE != "cuda":
        raise RuntimeError("当前未检测到 CUDA，已停止，避免在 CPU 上误跑")

    tasks = load_tasks(args.tasks)
    tasks_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_prompts: List[str] = []
    for task in tasks:
        image_path = task["image_path"].replace("\\", "/")
        tasks_by_image[image_path].append(task)
        prompt = task.get("text_prompt", "").strip()
        if prompt and prompt not in all_prompts:
            all_prompts.append(prompt)

    output_dir = args.output_root / args.tasks.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_overrides = load_threshold_overrides(args.output_thresholds_json)
    skip_set = load_skip_set(args.skip_list)

    _, processor = load_model(
        official_root=args.official_root,
        checkpoint_path=args.checkpoint,
        processor_conf_threshold=args.processor_conf_threshold,
    )

    prompt_seen = defaultdict(int)
    raw_prompt_hits = defaultdict(list)
    raw_prompt_misses = defaultdict(int)
    filtered_prompt_hits = defaultdict(list)
    filtered_prompt_misses = defaultdict(int)

    predictions = []
    kept_images: List[str] = []
    skipped_images: List[Dict[str, Any]] = []

    t0 = time.time()
    pbar = tqdm(sorted(tasks_by_image.items()), desc=args.tasks.name, unit="img")

    for rel_path, image_tasks in pbar:
        if rel_path in skip_set:
            skipped_images.append({"image_path": rel_path, "reason": "skip_list"})
            continue

        abs_path = str((args.image_root / rel_path).resolve())
        if not os.path.exists(abs_path):
            skipped_images.append({"image_path": rel_path, "reason": "missing_file"})
            continue

        try:
            image, preprocess_metrics = load_image(abs_path, args.enable_preprocess)
            unique_prompts = list(
                dict.fromkeys(
                    task.get("text_prompt", "").strip()
                    for task in image_tasks
                    if task.get("text_prompt", "").strip()
                )
            )
            raw_results, filtered_results = infer_prompts_for_image(
                image=image,
                prompts=unique_prompts,
                processor=processor,
                raw_stats_threshold=args.raw_stats_threshold,
                default_threshold=args.default_threshold,
                threshold_overrides=threshold_overrides,
            )
        except Exception as exc:
            skipped_images.append({"image_path": rel_path, "reason": f"inference_error: {exc}"})
            continue

        for prompt in unique_prompts:
            prompt_seen[prompt] += 1
            raw_pred = raw_results.get(prompt)
            if raw_pred is not None and raw_pred.get("score", 0) > 0:
                raw_prompt_hits[prompt].append(
                    {
                        "score": raw_pred["score"],
                        "instances": raw_pred.get("instance_count", 0),
                        "image": rel_path,
                    }
                )
            else:
                raw_prompt_misses[prompt] += 1

            filtered_pred = filtered_results.get(prompt)
            if filtered_pred is not None and filtered_pred.get("score", 0) > 0:
                filtered_prompt_hits[prompt].append(
                    {
                        "score": filtered_pred["score"],
                        "instances": filtered_pred.get("instance_count", 0),
                        "image": rel_path,
                    }
                )
            else:
                filtered_prompt_misses[prompt] += 1

        pred_entry = {
            "image_path": rel_path,
            "prompts": {},
        }
        if preprocess_metrics:
            pred_entry["preprocess_metrics"] = preprocess_metrics

        for prompt in unique_prompts:
            pred = filtered_results.get(prompt)
            if pred is not None and pred.get("score", 0) > 0:
                pred_entry["prompts"][prompt] = {
                    "hit": True,
                    "score": round(pred["score"], 4),
                    "instances": pred.get("instance_count", 0),
                    "rle": pred.get("rle"),
                }
            else:
                pred_entry["prompts"][prompt] = {"hit": False}

        predictions.append(pred_entry)
        kept_images.append(rel_path)
        pbar.set_postfix(hits=sum(1 for x in pred_entry["prompts"].values() if x.get("hit")))

    elapsed = time.time() - t0

    pred_path = output_dir / f"pred_{args.tasks.stem}.json"
    stats_path = output_dir / f"stats_{args.tasks.stem}.csv"
    kept_path = output_dir / f"kept_{args.tasks.stem}.txt"
    skipped_path = output_dir / f"skipped_{args.tasks.stem}.json"
    meta_path = output_dir / f"meta_{args.tasks.stem}.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    write_stats_csv(
        stats_path=stats_path,
        prompts=all_prompts,
        prompt_seen=prompt_seen,
        raw_prompt_hits=raw_prompt_hits,
        raw_prompt_misses=raw_prompt_misses,
        filtered_prompt_hits=filtered_prompt_hits,
        filtered_prompt_misses=filtered_prompt_misses,
    )

    with open(kept_path, "w", encoding="utf-8") as f:
        for path in kept_images:
            f.write(path + "\n")

    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped_images, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": str(args.tasks),
                "image_root": str(args.image_root),
                "official_root": str(args.official_root),
                "checkpoint": str(args.checkpoint),
                "device": DEVICE,
                "enable_preprocess": bool(args.enable_preprocess),
                "processor_conf_threshold": float(args.processor_conf_threshold),
                "raw_stats_threshold": float(args.raw_stats_threshold),
                "default_threshold": float(args.default_threshold),
                "output_threshold_overrides": threshold_overrides,
                "skip_list": str(args.skip_list) if args.skip_list else None,
                "kept_images": len(kept_images),
                "skipped_images": len(skipped_images),
                "elapsed_seconds": round(elapsed, 3),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"完成: kept={len(kept_images)}, skipped={len(skipped_images)}, elapsed={elapsed:.1f}s")
    print(f"pred json:   {pred_path}")
    print(f"stats csv:   {stats_path}")
    print(f"kept list:   {kept_path}")
    print(f"skipped log: {skipped_path}")
    print(f"meta json:   {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于官方 SAM3 生成伪标签")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="任务 JSON 路径")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="图片根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument(
        "--official-root",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT,
        help="官方 to_Df - 官方 根目录",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM3 权重路径")
    parser.add_argument(
        "--processor-conf-threshold",
        type=float,
        default=DEFAULT_PROCESSOR_CONF_THRESHOLD,
        help="Sam3Processor 内部 proposal 保留阈值，默认沿用官方 0.5",
    )
    parser.add_argument(
        "--raw-stats-threshold",
        type=float,
        default=DEFAULT_RAW_STATS_THRESHOLD,
        help="统计用阈值，只影响 raw_hits 汇总",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_PROMPT_THRESHOLD,
        help="未单独配置类别时的输出阈值",
    )
    parser.add_argument(
        "--output-thresholds-json",
        type=str,
        default=None,
        help='按类阈值 JSON 字符串，如 {"person":0.7,"trash can":0.65}',
    )
    parser.add_argument(
        "--skip-list",
        type=str,
        default=str(DEFAULT_SKIP_LIST) if DEFAULT_SKIP_LIST.exists() else "",
        help='坏图列表，每行一个相对 image_path；传空字符串 "" 可关闭',
    )
    parser.add_argument(
        "--disable-preprocess",
        action="store_true",
        help="关闭你之前验证过的红外预处理",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="要求必须跑在 CUDA 上",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.enable_preprocess = not args.disable_preprocess
    args.skip_list = Path(args.skip_list) if args.skip_list else None
    run_tasks(args)


if __name__ == "__main__":
    main()
