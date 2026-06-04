#!/usr/bin/env python3
"""
Offline crop augmentation for tiny person instances in data1.

Goals:
  - Only enhance `person` in `data1`
  - Only crop sufficiently small person instances
  - Keep contextual square crops instead of tight person-only crops
  - Export crop-only samples plus merged pred_json/train_list for training only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_FG_JSON = ROOT / "test" / "prompt_test_output" / "train(person)" / "pred_train(person).json"
DEFAULT_TRAIN_LIST = ROOT / "test" / "train_list_d1.txt"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "crop_person_data1"
DEFAULT_MERGED_JSON = ROOT / "test" / "prompt_test_output" / "train(person)" / "pred_train(person)_with_crop.json"
DEFAULT_MERGED_LIST = ROOT / "test" / "train_list_d1_with_crop.txt"

TARGET_PROMPT = "person"
YOLO_CLASS_ORDER = [TARGET_PROMPT]

SPEC = {
    "num_samples": 400,
    "min_score": 0.70,
    "min_area_px": 8,
    "min_side_px": 2,
    "small_bbox_ratio_max": 0.001,
    "small_max_side_px": 20,
    "expand_range": (2.5, 3.5),
    "center_jitter": 0.06,
    "target_keep_ratio": 0.80,
    "context_keep_ratio": 0.60,
    "min_crop_area_ratio": 0.015,
    "min_crop_size": 48,
    "patch_size": 160,
}

RNG_SEED = 42
MIN_MARGIN = 4
MAX_SAMPLE_ATTEMPTS = 12
MIN_MASK_PIXELS = 12


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
        for folder in (images_dir, labels_dir, preview_dir):
            if folder.exists():
                shutil.rmtree(folder)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir, preview_dir


def save_rgb_jpg(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_yolo_label(label_path: Path, prompt_masks: dict[str, np.ndarray]):
    lines = []
    mask = prompt_masks.get(TARGET_PROMPT)
    if mask is not None:
        for cx, cy, bw, bh in mask_to_bboxes(mask):
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def merge_training_assets(
    base_json_path: Path,
    base_train_list_path: Path,
    synthetic_records: list[dict],
    out_json_path: Path,
    out_train_list_path: Path,
):
    base_records = load_json(base_json_path)
    merged_records = base_records + synthetic_records
    write_json(out_json_path, merged_records)

    base_list = load_file_list_ordered(base_train_list_path)
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
    return len(base_records), len(merged_records), len(merged_list)


def square_crop_rect_from_bbox(
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
    side = max(min_crop_size, int(round(max(bw, bh) * expand)))
    side = min(side, img_w, img_h)

    cx += rng.uniform(-center_jitter, center_jitter) * bw
    cy += rng.uniform(-center_jitter, center_jitter) * bh

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, img_w - side))
    top = max(0, min(top, img_h - side))
    return left, top, side, side


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


def resize_patch_and_mask(crop_rgb: np.ndarray, crop_mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    out_rgb = cv2.resize(crop_rgb, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    out_mask = cv2.resize(crop_mask.astype(np.uint8), (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    out_mask = (out_mask > 0).astype(np.uint8)
    return out_rgb, out_mask


def mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    if mask.size == 0:
        return out
    binary = mask.astype(bool)
    color = np.zeros_like(out)
    color[..., 0] = 255
    color[..., 1] = 90
    color[..., 2] = 90
    out[binary] = np.clip(out[binary] * 0.55 + color[binary] * 0.45, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 90, 90), 2)
    return out


def save_preview_image(
    preview_path: Path,
    src_gray: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    patch_rgb: np.ndarray,
    patch_mask: np.ndarray,
):
    left, top, crop_w, crop_h = crop_rect
    src_rgb = np.stack([src_gray] * 3, axis=-1)
    src_view = src_rgb.copy()
    cv2.rectangle(src_view, (left, top), (left + crop_w, top + crop_h), (255, 90, 90), 2)
    cv2.putText(src_view, "person@data1", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 90, 90), 2, cv2.LINE_AA)

    patch_view = mask_overlay(patch_rgb, patch_mask)
    cv2.putText(patch_view, f"patch {patch_rgb.shape[1]}x{patch_rgb.shape[0]}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 90, 90), 2, cv2.LINE_AA)

    target_h = max(src_view.shape[0], patch_view.shape[0])

    def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        top_pad = pad // 2
        bottom_pad = pad - top_pad
        return cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    canvas = np.concatenate([
        _pad_to_height(src_view, target_h),
        _pad_to_height(patch_view, target_h),
    ], axis=1)
    save_rgb_jpg(preview_path, canvas)


def extract_small_person_pool(records: list[dict], allow_set: set[str], spec: dict) -> tuple[list[dict], dict[str, dict]]:
    pool = []
    record_index = {}
    for record in tqdm(records, desc="index small person", unit="img"):
        image_path = record["image_path"].replace("\\", "/")
        if image_path not in allow_set or not image_path.startswith("test/data1/"):
            continue
        value = record["prompts"].get(TARGET_PROMPT, {})
        if not value.get("hit") or value.get("rle") is None:
            continue
        score = float(value.get("score", 0.0))
        if score < spec["min_score"]:
            continue
        record_index[image_path] = record
        mask = rle_to_mask(value["rle"])
        h, w = mask.shape
        img_area = float(h * w)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for cc_idx in range(1, num_labels):
            x, y, bw, bh, area = stats[cc_idx].tolist()
            if area < spec["min_area_px"]:
                continue
            if bw < spec["min_side_px"] or bh < spec["min_side_px"]:
                continue
            if x <= MIN_MARGIN or y <= MIN_MARGIN or x + bw >= w - MIN_MARGIN or y + bh >= h - MIN_MARGIN:
                continue
            bbox_ratio = (bw * bh) / img_area
            max_side = max(bw, bh)
            if bbox_ratio >= spec["small_bbox_ratio_max"]:
                continue
            if max_side >= spec["small_max_side_px"]:
                continue
            component_mask = (labels == cc_idx).astype(np.uint8)
            pool.append(
                {
                    "image_path": image_path,
                    "score": score,
                    "bbox": (x, y, bw, bh),
                    "area": int(area),
                    "bbox_ratio": float(bbox_ratio),
                    "max_side": int(max_side),
                    "component_rle": mask_to_rle(component_mask),
                }
            )
    return pool, record_index


def run_crop(args):
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    images_dir, labels_dir, preview_dir = prepare_output_dir(output_dir)

    fg_records = load_json(Path(args.fg_json))
    allow_set = load_file_list(Path(args.train_list))
    spec = dict(SPEC)
    if args.num_samples is not None:
        spec["num_samples"] = int(args.num_samples)

    target_pool, record_index = extract_small_person_pool(fg_records, allow_set, spec)
    if not target_pool:
        raise RuntimeError("empty small-person pool")

    synthetic_records = []
    manifest_rows = []
    source_usage = Counter()
    source_meta = {}
    img_counter = 0
    preview_saved = 0

    pbar = tqdm(total=spec["num_samples"], desc="crop person@data1", unit="img")
    success = 0
    attempts = 0
    max_attempts = max(spec["num_samples"] * MAX_SAMPLE_ATTEMPTS, spec["num_samples"])
    while success < spec["num_samples"] and attempts < max_attempts:
        attempts += 1
        sample = rng.choice(target_pool)
        record = record_index.get(sample["image_path"])
        if record is None:
            continue

        gray = read_gray(sample["image_path"])
        if gray is None:
            continue

        crop_rect = square_crop_rect_from_bbox(
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

        full_mask = rle_to_mask(record["prompts"][TARGET_PROMPT]["rle"])
        crop_mask = crop_mask_with_component_filter(full_mask, crop_rect, spec["context_keep_ratio"])
        if int(crop_mask.sum()) < MIN_MASK_PIXELS:
            continue

        crop_gray = gray[top:top + crop_h, left:left + crop_w]
        crop_rgb = np.stack([crop_gray] * 3, axis=-1)
        patch_rgb, patch_mask = resize_patch_and_mask(crop_rgb, crop_mask, spec["patch_size"])
        target_patch_mask = cv2.resize(target_crop_mask.astype(np.uint8), (spec["patch_size"], spec["patch_size"]), interpolation=cv2.INTER_NEAREST)
        target_patch_mask = (target_patch_mask > 0).astype(np.uint8)
        target_patch_area_ratio = float(target_patch_mask.sum()) / float(spec["patch_size"] * spec["patch_size"])
        if target_patch_area_ratio < spec["min_crop_area_ratio"]:
            continue

        stem = f"crop_person_data1_{img_counter:05d}"
        rel_image_path = (output_dir / "images" / f"{stem}.jpg").relative_to(ROOT).as_posix()
        save_rgb_jpg(images_dir / f"{stem}.jpg", patch_rgb)
        write_yolo_label(labels_dir / f"{stem}.txt", {TARGET_PROMPT: patch_mask})

        prompts = {
            TARGET_PROMPT: {
                "hit": True,
                "score": round(float(record["prompts"][TARGET_PROMPT].get("score", sample["score"])), 4),
                "instances": int(cv2.connectedComponents(patch_mask)[0] - 1),
                "rle": mask_to_rle(patch_mask),
            }
        }
        synthetic_records.append({"image_path": rel_image_path, "prompts": prompts})

        usage_key = (sample["image_path"], *sample["bbox"])
        source_usage[usage_key] += 1
        source_meta[usage_key] = {
            "src_image": sample["image_path"],
            "bbox": f"{sample['bbox'][0]},{sample['bbox'][1]},{sample['bbox'][2]},{sample['bbox'][3]}",
            "score": round(float(sample["score"]), 4),
            "target_area_px": int(sample["area"]),
            "bbox_ratio": round(float(sample["bbox_ratio"]), 6),
            "max_side": int(sample["max_side"]),
        }

        manifest_rows.append(
            {
                "image_path": rel_image_path,
                "src_image": sample["image_path"],
                "score": round(float(sample["score"]), 4),
                "expand": round(float(crop_w / max(sample["bbox"][2], sample["bbox"][3], 1)), 4),
                "crop_box": f"{left},{top},{crop_w},{crop_h}",
                "target_area_px": int(sample["area"]),
                "bbox_ratio": round(float(sample["bbox_ratio"]), 6),
                "max_side": int(sample["max_side"]),
                "keep_ratio": round(keep_ratio, 4),
                "target_crop_area_ratio": round(crop_area_ratio, 4),
                "target_patch_area_ratio": round(target_patch_area_ratio, 4),
                "source_use_index": source_usage[usage_key],
            }
        )

        if preview_saved < args.num_preview:
            save_preview_image(preview_dir / f"{stem}.jpg", gray, crop_rect, patch_rgb, patch_mask)
            preview_saved += 1

        img_counter += 1
        success += 1
        pbar.update(1)
        pbar.set_postfix(done=success, attempts=attempts)
    pbar.close()

    if success < spec["num_samples"]:
        print(f"warning: generated {success}/{spec['num_samples']} samples after {attempts} attempts")

    only_json = output_dir / "pred_crop_only.json"
    merged_json = output_dir / "pred_crop_merged.json"
    manifest_csv = output_dir / "manifest.csv"
    source_usage_csv = output_dir / "source_usage.csv"

    write_json(only_json, synthetic_records)
    write_json(merged_json, fg_records + synthetic_records)

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "src_image",
                "score",
                "expand",
                "crop_box",
                "target_area_px",
                "bbox_ratio",
                "max_side",
                "keep_ratio",
                "target_crop_area_ratio",
                "target_patch_area_ratio",
                "source_use_index",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(source_usage_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_image", "bbox", "score", "target_area_px", "bbox_ratio", "max_side", "use_count"],
        )
        writer.writeheader()
        for usage_key, use_count in sorted(source_usage.items(), key=lambda x: (-x[1], x[0][0])):
            row = dict(source_meta[usage_key])
            row["use_count"] = int(use_count)
            writer.writerow(row)

    base_count, merged_count, merged_train_count = merge_training_assets(
        Path(args.fg_json),
        Path(args.train_list),
        synthetic_records,
        Path(args.out_json),
        Path(args.out_train_list),
    )

    print(f"output_dir: {output_dir}")
    print(f"small-person pool: {len(target_pool)}")
    print(f"synthetic samples: {len(synthetic_records)}")
    print(f"crop-only json: {only_json}")
    print(f"crop-merged json: {merged_json}")
    print(f"manifest: {manifest_csv}")
    print(f"source usage: {source_usage_csv}")
    print(f"preview: {preview_dir} ({preview_saved})")
    print(f"merged json: {args.out_json} ({base_count} -> {merged_count})")
    print(f"merged train_list: {args.out_train_list} ({merged_train_count})")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline crop augmentation for tiny person instances in data1")
    parser.add_argument("--fg-json", default=str(DEFAULT_FG_JSON), help="data1 person pred_json")
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST), help="data1 train list")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    parser.add_argument("--out-json", default=str(DEFAULT_MERGED_JSON), help="merged pred_json output")
    parser.add_argument("--out-train-list", default=str(DEFAULT_MERGED_LIST), help="merged train list output")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="random seed")
    parser.add_argument("--num-samples", type=int, default=None, help="override generated sample count")
    parser.add_argument("--num-preview", type=int, default=12, help="number of preview images to export")
    return parser.parse_args()


if __name__ == "__main__":
    run_crop(parse_args())
