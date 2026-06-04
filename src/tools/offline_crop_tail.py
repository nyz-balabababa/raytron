#!/usr/bin/env python3
"""
Offline crop augmentation for tail classes.

Goals:
  - Generate crop-based positive augmentations for `trash can` and `computer`
  - Keep the output format compatible with both YOLO-World and later CLIPSeg
  - Export crop-only samples, merged pred_json, and a train_list ready for v4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_FG_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"
DEFAULT_TRAIN_LIST = ROOT / "test" / "train_list_d5&7.txt"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "crop_tail"
DEFAULT_D5D7_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"
DEFAULT_D5D7_TRAIN_LIST = ROOT / "test" / "train_list_d5&7.txt"
DEFAULT_D5D7_MERGED_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks_with_crop.json"
DEFAULT_D5D7_MERGED_LIST = ROOT / "test" / "train_list_d5&7_with_crop.txt"
DEFAULT_YOLO_CLASS_ORDER = ["computer", "trash can"]

TARGET_SPECS = {
    "trash can": {
        "num_samples": 150,
        "min_score": 0.65,
        "min_area_px": 120,
        "max_area_ratio": 0.12,
        "min_side_px": 10,
        "expand_range": (2.0, 3.0),
        "center_jitter": 0.12,
        "target_keep_ratio": 0.70,
        "min_crop_area_ratio": 0.08,
        "min_crop_size": 96,
    },
    "computer": {
        "num_samples": 80,
        "min_score": 0.60,
        "min_area_px": 100,
        "max_area_ratio": 0.08,
        "min_side_px": 8,
        "expand_range": (3.0, 4.5),
        "center_jitter": 0.08,
        "target_keep_ratio": 0.75,
        "min_crop_area_ratio": 0.04,
        "min_crop_size": 128,
    },
}

RNG_SEED = 42
MIN_MARGIN = 4
MAX_SAMPLE_ATTEMPTS = 12
MIN_MASK_PIXELS = 16
CONTEXT_KEEP_RATIO = 0.60
NEGATIVE_CROP_AREA_RANGE = (0.08, 0.20)
NEGATIVE_ASPECT_RANGE = (0.75, 1.33)


def load_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_file_list(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip().replace("\\", "/") for line in f if line.strip()}


def load_file_list_ordered(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


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


def read_gray(image_path: str) -> np.ndarray | None:
    abs_path = ROOT / image_path
    gray = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return unify_polarity(gray, Path(image_path).name)


def rle_to_mask(rle: dict) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils

        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return mask_utils.decode(rle_cp).astype(np.uint8)
    except ImportError:
        pass

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def mask_to_rle(mask: np.ndarray) -> dict:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle, list):
            rle = rle[0]
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
        }
    except ImportError:
        pass

    h, w = mask.shape
    flat = mask.astype(np.uint8).flatten(order="F")
    runs = []
    prev = 0
    cnt = 0
    for v in flat:
        v = int(v)
        if v == prev:
            cnt += 1
        else:
            runs.append(str(cnt))
            cnt = 1
            prev = v
    runs.append(str(cnt))
    return {"size": [h, w], "counts": ",".join(runs)}


def mask_to_bboxes(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    bboxes = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        bboxes.append((cx, cy, bw / w, bh / h))
    return bboxes


def prepare_output_dir(output_dir: Path):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"
    if output_dir.exists():
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir, preview_dir


def infer_prompt_order(records: list[dict], target_classes: list[str]) -> list[str]:
    prompt_order = list(records[0]["prompts"].keys())
    for cls_name in target_classes:
        if cls_name not in prompt_order:
            prompt_order.append(cls_name)
    return prompt_order


def resolve_yolo_class_order(prompt_order: list[str]) -> list[str]:
    class_order = [cls_name for cls_name in DEFAULT_YOLO_CLASS_ORDER if cls_name in prompt_order]
    if not class_order:
        raise RuntimeError("no YOLO classes available in prompt order")
    return class_order


def save_rgb_jpg(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def mask_overlay(rgb: np.ndarray, prompt_masks: dict[str, np.ndarray], target: str | None = None) -> np.ndarray:
    out = rgb.copy()
    color_map = {
        "computer": (255, 80, 80),
        "trash can": (80, 220, 120),
    }
    default_color = (255, 210, 80)
    for cls_name, mask in prompt_masks.items():
        if mask is None or mask.size == 0:
            continue
        color = color_map.get(cls_name, default_color)
        alpha = 0.45 if cls_name == target else 0.28
        color_img = np.zeros_like(out)
        color_img[..., 0] = color[0]
        color_img[..., 1] = color[1]
        color_img[..., 2] = color[2]
        binary = mask.astype(bool)
        out[binary] = np.clip(out[binary] * (1.0 - alpha) + color_img[binary] * alpha, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2 if cls_name == target else 1)
    return out


def save_preview_image(
    preview_path: Path,
    src_gray: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    crop_rgb: np.ndarray,
    prompt_masks: dict[str, np.ndarray],
    target: str,
    sample_type: str,
):
    left, top, crop_w, crop_h = crop_rect
    src_rgb = np.stack([src_gray] * 3, axis=-1)
    src_view = src_rgb.copy()
    rect_color = (255, 80, 80) if sample_type == "positive" else (80, 180, 255)
    cv2.rectangle(src_view, (left, top), (left + crop_w, top + crop_h), rect_color, 2)
    cv2.putText(
        src_view,
        f"{sample_type}: {target}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        rect_color,
        2,
        cv2.LINE_AA,
    )

    crop_view = mask_overlay(crop_rgb, prompt_masks, target=target if sample_type == "positive" else None)
    cv2.putText(
        crop_view,
        f"crop {crop_w}x{crop_h}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        rect_color,
        2,
        cv2.LINE_AA,
    )

    target_h = max(src_view.shape[0], crop_view.shape[0])

    def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        top_pad = pad // 2
        bottom_pad = pad - top_pad
        return cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    src_view = _pad_to_height(src_view, target_h)
    crop_view = _pad_to_height(crop_view, target_h)
    canvas = np.concatenate([src_view, crop_view], axis=1)
    save_rgb_jpg(preview_path, canvas)


def write_yolo_label(label_path: Path, class_order: list[str], prompt_masks: dict[str, np.ndarray]):
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(class_order)}
    lines = []
    for cls_name in class_order:
        mask = prompt_masks.get(cls_name)
        if mask is None:
            continue
        for cx, cy, bw, bh in mask_to_bboxes(mask):
            lines.append(f"{class_to_id[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def merge_d5d7_training_assets(
    d5d7_json_path: Path,
    d5d7_train_list_path: Path,
    synthetic_records: list[dict],
    out_json_path: Path,
    out_train_list_path: Path,
):
    d5d7_records = load_json(d5d7_json_path)
    merged_records = d5d7_records + synthetic_records
    write_json(out_json_path, merged_records)

    base_list = load_file_list_ordered(d5d7_train_list_path)
    merged_list = []
    seen = set()
    for image_path in base_list + [x["image_path"].replace("\\", "/") for x in synthetic_records]:
        norm = image_path.replace("\\", "/")
        if norm in seen:
            continue
        merged_list.append(norm)
        seen.add(norm)

    out_train_list_path.parent.mkdir(parents=True, exist_ok=True)
    out_train_list_path.write_text("\n".join(merged_list) + "\n", encoding="utf-8")
    return len(d5d7_records), len(merged_records), len(merged_list)


def crop_rect_from_bbox(
    rng: random.Random,
    img_h: int,
    img_w: int,
    bbox: tuple[int, int, int, int],
    expand_range: tuple[float, float],
    center_jitter: float,
    min_crop_size: int,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    expand = rng.uniform(*expand_range)

    jitter_x = rng.uniform(-center_jitter, center_jitter) * bw
    jitter_y = rng.uniform(-center_jitter, center_jitter) * bh
    cx += jitter_x
    cy += jitter_y

    crop_w = max(min_crop_size, int(round(bw * expand)))
    crop_h = max(min_crop_size, int(round(bh * expand)))
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))
    left = max(0, min(left, img_w - crop_w))
    top = max(0, min(top, img_h - crop_h))
    return left, top, crop_w, crop_h


def crop_mask_with_component_filter(
    full_mask: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    min_keep_ratio: float,
) -> np.ndarray:
    left, top, crop_w, crop_h = crop_rect
    out_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask.astype(np.uint8), connectivity=8)
    for cc_idx in range(1, num_labels):
        area = int(stats[cc_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_mask = (labels == cc_idx).astype(np.uint8)
        crop_component = component_mask[top:top + crop_h, left:left + crop_w].astype(np.uint8)
        crop_area = int(crop_component.sum())
        if crop_area < MIN_MASK_PIXELS:
            continue
        keep_ratio = crop_area / float(area)
        if keep_ratio < min_keep_ratio:
            continue
        out_mask |= crop_component
    return out_mask


def build_negative_record_pool(records: list[dict], allow_set: set[str]) -> list[dict]:
    pool = []
    for record in records:
        image_path = record["image_path"].replace("\\", "/")
        if image_path not in allow_set:
            continue
        if any(bool(value.get("hit")) for value in record["prompts"].values()):
            continue
        pool.append({"image_path": image_path, "record": record})
    return pool


def sample_negative_crop_rect(
    rng: random.Random,
    img_h: int,
    img_w: int,
    min_crop_size: int,
) -> tuple[int, int, int, int] | None:
    if img_h < min_crop_size or img_w < min_crop_size:
        return None
    img_area = float(img_h * img_w)
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        area_ratio = rng.uniform(*NEGATIVE_CROP_AREA_RANGE)
        aspect = rng.uniform(*NEGATIVE_ASPECT_RANGE)
        area = img_area * area_ratio
        crop_w = int(round((area * aspect) ** 0.5))
        crop_h = int(round((area / aspect) ** 0.5))
        if crop_w < min_crop_size or crop_h < min_crop_size:
            continue
        if crop_w > img_w or crop_h > img_h:
            continue
        left = rng.randint(0, img_w - crop_w)
        top = rng.randint(0, img_h - crop_h)
        return left, top, crop_w, crop_h
    return None


def build_negative_prompts(prompt_order: list[str]) -> dict[str, dict]:
    return {cls_name: {"hit": False} for cls_name in prompt_order}


def extract_target_pool(records: list[dict], allow_set: set[str], target_specs: dict[str, dict]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    pools = defaultdict(list)
    record_index = {}
    for record in tqdm(records, desc="index positives", unit="img"):
        image_path = record["image_path"].replace("\\", "/")
        if image_path not in allow_set:
            continue
        record_index[image_path] = record
        for target, spec in target_specs.items():
            value = record["prompts"].get(target, {})
            if not value.get("hit") or value.get("rle") is None:
                continue
            score = float(value.get("score", 0.0))
            if score < spec["min_score"]:
                continue
            mask = rle_to_mask(value["rle"])
            h, w = mask.shape
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for cc_idx in range(1, num_labels):
                x, y, bw, bh, area = stats[cc_idx].tolist()
                if area < spec["min_area_px"]:
                    continue
                if bw < spec["min_side_px"] or bh < spec["min_side_px"]:
                    continue
                if area / float(h * w) > spec["max_area_ratio"]:
                    continue
                if x <= MIN_MARGIN or y <= MIN_MARGIN or x + bw >= w - MIN_MARGIN or y + bh >= h - MIN_MARGIN:
                    continue
                component_mask = (labels == cc_idx).astype(np.uint8)
                pools[target].append(
                    {
                        "image_path": image_path,
                        "target": target,
                        "score": score,
                        "bbox": (x, y, bw, bh),
                        "area": int(area),
                        "component_rle": mask_to_rle(component_mask),
                    }
                )
    return pools, record_index


def build_cropped_prompts(
    record: dict,
    prompt_order: list[str],
    crop_rect: tuple[int, int, int, int],
    target: str,
    target_score: float,
    target_keep_ratio: float,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    left, top, crop_w, crop_h = crop_rect
    prompts = {}
    prompt_masks = {}
    base_prompts = record["prompts"]

    for cls_name in prompt_order:
        value = base_prompts.get(cls_name, {})
        if value.get("hit") and value.get("rle") is not None:
            full_mask = rle_to_mask(value["rle"])
            keep_ratio = target_keep_ratio if cls_name == target else CONTEXT_KEEP_RATIO
            crop_mask = crop_mask_with_component_filter(full_mask, crop_rect, min_keep_ratio=keep_ratio)
            if int(crop_mask.sum()) >= MIN_MASK_PIXELS:
                prompt_masks[cls_name] = crop_mask
                prompts[cls_name] = {
                    "hit": True,
                    "score": round(float(value.get("score", target_score)), 4),
                    "instances": int(cv2.connectedComponents(crop_mask)[0] - 1),
                    "rle": mask_to_rle(crop_mask),
                }
                continue
        prompts[cls_name] = {"hit": False}
    return prompts, prompt_masks


def run_crop(args):
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    images_dir, labels_dir, preview_dir = prepare_output_dir(output_dir)

    fg_records = load_json(Path(args.fg_json))
    allow_set = load_file_list(Path(args.train_list))

    target_specs = {name: dict(TARGET_SPECS[name]) for name in args.targets}
    if args.num_trash_can is not None and "trash can" in target_specs:
        target_specs["trash can"]["num_samples"] = int(args.num_trash_can)
    if args.num_computer is not None and "computer" in target_specs:
        target_specs["computer"]["num_samples"] = int(args.num_computer)

    prompt_order = infer_prompt_order(fg_records, args.targets)
    yolo_class_order = resolve_yolo_class_order(prompt_order)
    target_pools, record_index = extract_target_pool(fg_records, allow_set, target_specs)
    negative_pool = build_negative_record_pool(fg_records, allow_set)

    for target in args.targets:
        if not target_pools[target]:
            raise RuntimeError(f"empty target pool: {target}")
    if args.num_neg_samples > 0 and not negative_pool:
        raise RuntimeError("empty negative crop pool")

    synthetic_records = []
    manifest_rows = []
    source_usage = Counter()
    source_meta = {}
    img_counter = 0
    preview_saved = 0
    min_neg_crop_size = min(spec["min_crop_size"] for spec in target_specs.values())

    for target in args.targets:
        spec = target_specs[target]
        pbar = tqdm(total=spec["num_samples"], desc=f"crop {target}", unit="img")
        success = 0
        attempts = 0
        max_attempts = max(spec["num_samples"] * MAX_SAMPLE_ATTEMPTS, spec["num_samples"])
        while success < spec["num_samples"] and attempts < max_attempts:
            attempts += 1
            sample = rng.choice(target_pools[target])
            record = record_index.get(sample["image_path"])
            if record is None:
                continue

            gray = read_gray(sample["image_path"])
            if gray is None:
                continue

            crop_rect = crop_rect_from_bbox(
                rng,
                gray.shape[0],
                gray.shape[1],
                sample["bbox"],
                spec["expand_range"],
                spec["center_jitter"],
                spec["min_crop_size"],
            )
            left, top, crop_w, crop_h = crop_rect

            component_mask = rle_to_mask(sample["component_rle"])
            target_crop_mask = component_mask[top:top + crop_h, left:left + crop_w].astype(np.uint8)
            full_area = max(int(component_mask.sum()), 1)
            keep_ratio = float(target_crop_mask.sum()) / full_area
            crop_area_ratio = float(target_crop_mask.sum()) / float(max(crop_w * crop_h, 1))
            if int(target_crop_mask.sum()) < MIN_MASK_PIXELS:
                continue
            if keep_ratio < spec["target_keep_ratio"]:
                continue
            if crop_area_ratio < spec["min_crop_area_ratio"]:
                continue

            crop_gray = gray[top:top + crop_h, left:left + crop_w]
            crop_rgb = np.stack([crop_gray] * 3, axis=-1)

            prompts, prompt_masks = build_cropped_prompts(
                record,
                prompt_order,
                crop_rect,
                target,
                sample["score"],
                spec["target_keep_ratio"],
            )
            if not prompts.get(target, {}).get("hit"):
                continue

            stem = f"crop_{target.replace(' ', '_')}_{img_counter:05d}"
            rel_image_path = (output_dir / "images" / f"{stem}.jpg").relative_to(ROOT).as_posix()
            save_rgb_jpg(images_dir / f"{stem}.jpg", crop_rgb)
            write_yolo_label(labels_dir / f"{stem}.txt", yolo_class_order, prompt_masks)
            if preview_saved < args.num_preview:
                save_preview_image(
                    preview_dir / f"{stem}.jpg",
                    gray,
                    crop_rect,
                    crop_rgb,
                    prompt_masks,
                    target,
                    "positive",
                )
                preview_saved += 1

            usage_key = (target, sample["image_path"], *sample["bbox"])
            source_usage[usage_key] += 1
            source_meta[usage_key] = {
                "target": target,
                "src_image": sample["image_path"],
                "bbox": f"{sample['bbox'][0]},{sample['bbox'][1]},{sample['bbox'][2]},{sample['bbox'][3]}",
                "score": round(float(sample["score"]), 4),
                "target_area_px": int(sample["area"]),
            }

            synthetic_records.append({"image_path": rel_image_path, "prompts": prompts})
            manifest_rows.append(
                {
                    "sample_type": "positive",
                    "image_path": rel_image_path,
                    "target": target,
                    "src_image": sample["image_path"],
                    "score": round(float(sample["score"]), 4),
                    "expand": round(float(max(crop_w / max(sample["bbox"][2], 1), crop_h / max(sample["bbox"][3], 1))), 4),
                    "crop_box": f"{left},{top},{crop_w},{crop_h}",
                    "target_area_px": int(sample["area"]),
                    "keep_ratio": round(keep_ratio, 4),
                    "crop_area_ratio": round(crop_area_ratio, 4),
                    "source_use_index": source_usage[usage_key],
                }
            )
            img_counter += 1
            success += 1
            pbar.update(1)
            pbar.set_postfix(done=success, attempts=attempts)
        pbar.close()
        if success < spec["num_samples"]:
            print(
                f"warning: target={target} generated {success}/{spec['num_samples']} samples "
                f"after {attempts} attempts"
            )

    neg_success = 0
    if args.num_neg_samples > 0:
        pbar = tqdm(total=args.num_neg_samples, desc="crop background", unit="img")
        attempts = 0
        max_attempts = max(args.num_neg_samples * MAX_SAMPLE_ATTEMPTS, args.num_neg_samples)
        while neg_success < args.num_neg_samples and attempts < max_attempts:
            attempts += 1
            sample = rng.choice(negative_pool)
            gray = read_gray(sample["image_path"])
            if gray is None:
                continue
            crop_rect = sample_negative_crop_rect(rng, gray.shape[0], gray.shape[1], min_neg_crop_size)
            if crop_rect is None:
                continue
            left, top, crop_w, crop_h = crop_rect
            crop_gray = gray[top:top + crop_h, left:left + crop_w]
            crop_rgb = np.stack([crop_gray] * 3, axis=-1)
            prompts = build_negative_prompts(prompt_order)

            stem = f"crop_background_{img_counter:05d}"
            rel_image_path = (output_dir / "images" / f"{stem}.jpg").relative_to(ROOT).as_posix()
            save_rgb_jpg(images_dir / f"{stem}.jpg", crop_rgb)
            write_yolo_label(labels_dir / f"{stem}.txt", yolo_class_order, {})
            if preview_saved < args.num_preview:
                save_preview_image(
                    preview_dir / f"{stem}.jpg",
                    gray,
                    crop_rect,
                    crop_rgb,
                    {},
                    "background",
                    "negative",
                )
                preview_saved += 1

            synthetic_records.append({"image_path": rel_image_path, "prompts": prompts})
            manifest_rows.append(
                {
                    "sample_type": "negative",
                    "image_path": rel_image_path,
                    "target": "background",
                    "src_image": sample["image_path"],
                    "score": "",
                    "expand": "",
                    "crop_box": f"{left},{top},{crop_w},{crop_h}",
                    "target_area_px": 0,
                    "keep_ratio": "",
                    "crop_area_ratio": "",
                    "source_use_index": "",
                }
            )
            img_counter += 1
            neg_success += 1
            pbar.update(1)
            pbar.set_postfix(done=neg_success, attempts=attempts)
        pbar.close()
        if neg_success < args.num_neg_samples:
            print(
                f"warning: background generated {neg_success}/{args.num_neg_samples} samples "
                f"after {attempts} attempts"
            )

    only_json = output_dir / "pred_crop_only.json"
    merged_json = output_dir / "pred_crop_merged.json"
    manifest_csv = output_dir / "manifest.csv"
    source_usage_csv = output_dir / "source_usage.csv"

    write_json(only_json, synthetic_records)
    merged_records = fg_records + synthetic_records
    write_json(merged_json, merged_records)

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_type",
                "image_path",
                "target",
                "src_image",
                "score",
                "expand",
                "crop_box",
                "target_area_px",
                "keep_ratio",
                "crop_area_ratio",
                "source_use_index",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(source_usage_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["target", "src_image", "bbox", "score", "target_area_px", "use_count"],
        )
        writer.writeheader()
        for usage_key, use_count in sorted(source_usage.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            row = dict(source_meta[usage_key])
            row["use_count"] = int(use_count)
            writer.writerow(row)

    d5d7_base_count, d5d7_merged_count, d5d7_train_count = merge_d5d7_training_assets(
        Path(args.d5d7_json),
        Path(args.d5d7_train_list),
        synthetic_records,
        Path(args.d5d7_out_json),
        Path(args.d5d7_out_train_list),
    )

    print(f"output_dir: {output_dir}")
    print(f"synthetic samples: {len(synthetic_records)}")
    for target in args.targets:
        count = sum(1 for x in synthetic_records if x["prompts"].get(target, {}).get("hit"))
        print(f"  {target}: {count}")
    if args.num_neg_samples > 0:
        print(f"  background: {neg_success}")
    print(f"labels: {labels_dir}")
    print(f"crop-only json: {only_json}")
    print(f"crop-merged json: {merged_json}")
    print(f"manifest: {manifest_csv}")
    print(f"source usage: {source_usage_csv}")
    print(f"preview: {preview_dir} ({preview_saved})")
    print(f"d5&7 merged json: {args.d5d7_out_json} ({d5d7_base_count} -> {d5d7_merged_count})")
    print(f"d5&7 merged train_list: {args.d5d7_out_train_list} ({d5d7_train_count})")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline crop augmentation for trash can / computer")
    parser.add_argument("--fg-json", default=str(DEFAULT_FG_JSON), help="source pred_json")
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST), help="allowed train list")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    parser.add_argument("--d5d7-json", default=str(DEFAULT_D5D7_JSON), help="base d5&7 pred_json")
    parser.add_argument("--d5d7-train-list", default=str(DEFAULT_D5D7_TRAIN_LIST), help="base d5&7 train list")
    parser.add_argument("--d5d7-out-json", default=str(DEFAULT_D5D7_MERGED_JSON), help="merged d5&7 pred_json output")
    parser.add_argument("--d5d7-out-train-list", default=str(DEFAULT_D5D7_MERGED_LIST), help="merged d5&7 train list output")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="random seed")
    parser.add_argument("--num-neg-samples", type=int, default=0, help="additional pure background crop samples")
    parser.add_argument("--num-preview", type=int, default=12, help="number of preview images to export")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["trash can", "computer"],
        choices=sorted(TARGET_SPECS.keys()),
        help="targets to crop",
    )
    parser.add_argument("--num-trash-can", type=int, default=None, help="override trash can sample count")
    parser.add_argument("--num-computer", type=int, default=None, help="override computer sample count")
    return parser.parse_args()


if __name__ == "__main__":
    run_crop(parse_args())
