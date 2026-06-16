from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


DEFAULT_COLORS = [
    (255, 64, 64),
    (64, 200, 255),
    (80, 220, 120),
    (255, 180, 60),
    (180, 100, 255),
    (255, 120, 180),
    (80, 255, 220),
    (220, 220, 80),
]


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, dict):
        for key in ["tasks", "annotations", "data"]:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("tasks 文件必须是 list，或包含 tasks/annotations/data 的 dict")
    tasks: List[Dict[str, Any]] = []
    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"第 {index} 条 task 不是 dict")
        if "ann_id" not in task or "image_path" not in task:
            raise ValueError(f"第 {index} 条 task 缺少 ann_id 或 image_path")
        tasks.append(task)
    return tasks


def load_predictions(predictions_path: str) -> List[Dict[str, Any]]:
    with open(predictions_path, "r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, dict):
        for key in ["predictions", "annotations", "results"]:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("predictions 文件必须是 list，或包含 predictions/annotations/results 的 dict")
    predictions: List[Dict[str, Any]] = []
    for index, pred in enumerate(payload):
        if not isinstance(pred, dict):
            raise ValueError(f"第 {index} 条 prediction 不是 dict")
        if "ann_id" not in pred:
            raise ValueError(f"第 {index} 条 prediction 缺少 ann_id")
        if "segmentation" in pred:
            predictions.append(pred)
        elif "rle" in pred:
            predictions.append({"ann_id": pred["ann_id"], "segmentation": pred["rle"]})
        else:
            raise ValueError(f"第 {index} 条 prediction 缺少 segmentation/rle")
    return predictions


def get_prompt_text(task: Dict[str, Any]) -> str:
    for key in ["text_prompt", "prompt", "text"]:
        value = task.get(key)
        if value is not None:
            prompt = str(value).strip()
            if prompt:
                return prompt
    return ""


def resolve_image_path(image_root: str, image_rel_path: str) -> str:
    image_rel_path = str(image_rel_path).replace("\\", "/")
    if os.path.isabs(image_rel_path):
        return image_rel_path
    candidates = [os.path.join(image_root, image_rel_path)]
    if image_rel_path.startswith("test/"):
        candidates.append(os.path.join(image_root, image_rel_path[5:]))
    candidates.append(os.path.join(image_root, "test", image_rel_path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def simple_rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"RLE size 非法: {size}")
    if not isinstance(counts, str):
        raise ValueError("simple RLE counts 必须是逗号分隔字符串")
    height, width = int(size[0]), int(size[1])
    runs = [int(x) for x in counts.split(",") if str(x).strip()]
    flat = np.zeros(height * width, dtype=np.uint8)
    value = 0
    index = 0
    for run_length in runs:
        if run_length < 0:
            raise ValueError(f"RLE run_length 非法: {run_length}")
        if value == 1 and run_length > 0:
            flat[index:index + run_length] = 1
        index += run_length
        value = 1 - value
    if index != flat.size:
        raise ValueError(f"RLE 长度不匹配: decoded={index} expected={flat.size}")
    return np.reshape(flat, (height, width), order="F").astype(np.uint8)


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    if not isinstance(rle, dict):
        raise ValueError("RLE 必须是 dict")
    counts = rle.get("counts")
    if mask_utils is not None:
        try:
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return (mask > 0).astype(np.uint8)
        except Exception:
            pass
    if isinstance(counts, str) and "," in counts:
        return simple_rle_decode(rle)
    raise RuntimeError("RLE decode 失败，既不是可用 pycocotools RLE，也不是 simple comma RLE")


def sanitize_text(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "empty"
    return cleaned[:max_len]


def ensure_color_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"不支持的图片形状: {image.shape}")


def draw_text_block(image: np.ndarray, lines: List[str], origin: Tuple[int, int] = (10, 24)) -> np.ndarray:
    canvas = image.copy()
    x, y = origin
    line_height = 22
    box_width = max(260, min(canvas.shape[1] - 10, max(len(line) for line in lines) * 11 + 20))
    box_height = 10 + line_height * len(lines)
    cv2.rectangle(canvas, (x - 6, y - 18), (x - 6 + box_width, y - 18 + box_height), (0, 0, 0), thickness=-1)
    cv2.rectangle(canvas, (x - 6, y - 18), (x - 6 + box_width, y - 18 + box_height), (255, 255, 255), thickness=1)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    canvas = ensure_color_image(image)
    if mask.shape[:2] != canvas.shape[:2]:
        raise ValueError(f"mask shape 与 image 不一致: mask={mask.shape} image={canvas.shape}")
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return canvas
    overlay = canvas.copy()
    overlay[mask_bool] = color
    blended = cv2.addWeighted(overlay, float(alpha), canvas, float(1.0 - alpha), 0.0)
    canvas[mask_bool] = blended[mask_bool]
    return canvas


def save_per_ann_visualization(
    image: np.ndarray,
    task: Dict[str, Any],
    mask: np.ndarray,
    out_path: Path,
    alpha: float,
) -> None:
    prompt = get_prompt_text(task)
    image_name = Path(str(task.get("image_path", ""))).name
    vis = overlay_mask(image, mask, color=(64, 64, 255), alpha=alpha)
    lines = [
        f"ann_id: {task['ann_id']}",
        f"prompt: {prompt}",
        f"mask_area: {int(mask.sum())}",
        f"image: {image_name}",
    ]
    vis = draw_text_block(vis, lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def save_by_image_visualization(
    image: np.ndarray,
    records: List[Dict[str, Any]],
    out_path: Path,
    alpha: float,
) -> None:
    vis = ensure_color_image(image)
    legend_lines: List[str] = []
    for idx, record in enumerate(records):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        vis = overlay_mask(vis, record["mask"], color=color, alpha=alpha)
        legend_lines.append(
            f"[{idx + 1}] {record['prompt'][:28]} | ann={record['ann_id']} | area={int(record['mask'].sum())}"
        )
    vis = draw_text_block(vis, legend_lines[:14])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=str)
    parser.add_argument("--predictions", required=True, type=str)
    parser.add_argument("--image_root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--max_vis", default=100, type=int)
    parser.add_argument("--alpha", default=0.45, type=float)
    parser.add_argument("--only_non_empty", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    predictions = load_predictions(args.predictions)
    if len(predictions) != len(tasks):
        print(f"WARNING: predictions 数量({len(predictions)}) != tasks 数量({len(tasks)})")

    task_by_ann_id = {str(task["ann_id"]): task for task in tasks}
    pred_by_ann_id = {str(pred["ann_id"]): pred for pred in predictions}

    out_dir = Path(args.out_dir)
    per_ann_dir = out_dir / "per_ann"
    by_image_dir = out_dir / "by_image"
    per_ann_dir.mkdir(parents=True, exist_ok=True)
    by_image_dir.mkdir(parents=True, exist_ok=True)

    matched_predictions = 0
    missing_task_count = 0
    decode_failed_count = 0
    empty_mask_count = 0
    non_empty_mask_count = 0
    saved_count = 0
    by_image_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    image_cache: Dict[str, np.ndarray] = {}

    for index, pred in enumerate(predictions):
        if saved_count >= int(args.max_vis):
            break
        ann_id = str(pred["ann_id"])
        task = task_by_ann_id.get(ann_id)
        if task is None:
            missing_task_count += 1
            print(f"WARNING: 找不到 ann_id 对应 task: {ann_id}")
            continue
        matched_predictions += 1
        image_rel_path = str(task["image_path"])
        image_abs_path = resolve_image_path(args.image_root, image_rel_path)
        try:
            if image_abs_path not in image_cache:
                image = cv2.imread(image_abs_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f"图片不存在或无法读取: {image_abs_path}")
                image_cache[image_abs_path] = image
            image = image_cache[image_abs_path]
            mask = decode_rle(pred["segmentation"])
        except Exception as exc:
            decode_failed_count += 1
            print(f"WARNING: ann_id={ann_id} decode/读取失败: {exc}")
            continue

        mask_area = int(mask.sum())
        if mask_area <= 0:
            empty_mask_count += 1
            if args.only_non_empty:
                continue
        else:
            non_empty_mask_count += 1

        prompt = get_prompt_text(task)
        safe_prompt = sanitize_text(prompt)
        per_ann_path = per_ann_dir / f"{index + 1:06d}_ann{ann_id}_{safe_prompt}.jpg"
        save_per_ann_visualization(image=image, task=task, mask=mask, out_path=per_ann_path, alpha=float(args.alpha))
        saved_count += 1

        by_image_records[image_abs_path].append(
            {
                "ann_id": ann_id,
                "prompt": prompt,
                "mask": mask,
                "image_path": image_rel_path,
            }
        )

    for image_abs_path, records in by_image_records.items():
        if not records:
            continue
        image = image_cache.get(image_abs_path)
        if image is None:
            continue
        image_name = sanitize_text(Path(image_abs_path).stem, max_len=80)
        save_by_image_visualization(
            image=image,
            records=records,
            out_path=by_image_dir / f"{image_name}.jpg",
            alpha=float(args.alpha),
        )

    summary = {
        "total_tasks": int(len(tasks)),
        "total_predictions": int(len(predictions)),
        "matched_predictions": int(matched_predictions),
        "missing_task_count": int(missing_task_count),
        "decode_failed_count": int(decode_failed_count),
        "empty_mask_count": int(empty_mask_count),
        "non_empty_mask_count": int(non_empty_mask_count),
        "saved_count": int(saved_count),
        "out_dir": str(out_dir),
        "only_non_empty": bool(args.only_non_empty),
        "alpha": float(args.alpha),
        "max_vis": int(args.max_vis),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(f"total_tasks={summary['total_tasks']}")
    print(f"total_predictions={summary['total_predictions']}")
    print(f"matched_predictions={summary['matched_predictions']}")
    print(f"missing_task_count={summary['missing_task_count']}")
    print(f"decode_failed_count={summary['decode_failed_count']}")
    print(f"empty_mask_count={summary['empty_mask_count']}")
    print(f"non_empty_mask_count={summary['non_empty_mask_count']}")
    print(f"saved_count={summary['saved_count']}")
    print(f"summary_path={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
