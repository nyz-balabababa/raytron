#!/usr/bin/env python3
"""
离线 Copy-Paste 尾类增强
========================

目标:
  - 从 d5&7 伪标签中抽取高质量前景实例（当前默认只生成 trash can）
  - 从全训练集伪标签中抽取“不含该类”的同域背景图
  - 生成增强图、YOLO bbox 标签、synthetic-only pred_json
  - 额外生成 merged pred_json，便于后续接入 CLIPSeg

默认输入:
  - 前景伪标签: test/prompt_test_output/train_d5&7_tasks/pred_train_d5&7_tasks.json
  - 背景伪标签: test/prompt_test_output/train_tasks/new_train.json
  - 训练列表:   test/train_list.txt

默认输出:
  - test/copy_paste_tail/
    ├── images/
    ├── labels/
    ├── pred_copy_paste_only.json
    ├── pred_copy_paste_merged.json
    └── manifest.csv

额外导出:
  - test/prompt_test_output/train_d5&7_tasks/pred_train_d5&7_tasks_with_copy_paste.json
  - test/train_list_d5&7_with_copy_paste.txt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_FG_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"
DEFAULT_BG_JSON = ROOT / "test" / "prompt_test_output" / "train_tasks" / "new_train.json"
DEFAULT_TRAIN_LIST = ROOT / "test" / "train_list.txt"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "copy_paste_tail"
DEFAULT_D5D7_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"
DEFAULT_D5D7_TRAIN_LIST = ROOT / "test" / "train_list_d5&7.txt"
DEFAULT_D5D7_MERGED_JSON = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks_with_copy_paste.json"
DEFAULT_D5D7_MERGED_LIST = ROOT / "test" / "train_list_d5&7_with_copy_paste.txt"

TARGET_SPECS = {
    "trash can": {
        "num_samples": 120,
        "min_score": 0.65,
        "min_area_px": 120,
        "max_area_ratio": 0.12,
        "min_side_px": 10,
        "scale_range": (0.85, 1.20),
        "same_source_only": False,
        "max_overlap_ratio": 0.05,
    },
    "computer": {
        "num_samples": 150,
        "min_score": 0.60,
        "min_area_px": 100,
        "max_area_ratio": 0.08,
        "min_side_px": 8,
        "scale_range": (0.90, 1.10),
        "same_source_only": True,
        "max_overlap_ratio": 0.03,
    },
}

RNG_SEED = 42
MAX_PLACEMENT_ATTEMPTS = 40
EDGE_BLUR_SIGMA = 1.2
MIN_MARGIN = 4


def load_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


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


def infer_class_order(bg_records: list[dict], target_classes: list[str]) -> list[str]:
    class_order = list(bg_records[0]["prompts"].keys())
    for cls_name in target_classes:
        if cls_name not in class_order:
            class_order.append(cls_name)
    return class_order


def get_source_name(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[1] if len(parts) >= 2 else "unknown"


def prepare_output_dir(output_dir: Path):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    if output_dir.exists():
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def extract_foreground_pool(fg_records: list[dict], target_specs: dict) -> dict[str, list[dict]]:
    pools = defaultdict(list)
    for record in tqdm(fg_records, desc="提取前景池", unit="img"):
        image_path = record["image_path"].replace("\\", "/")
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
                comp_mask = (labels[y:y + bh, x:x + bw] == cc_idx).astype(np.uint8)
                pools[target].append(
                    {
                        "target": target,
                        "image_path": image_path,
                        "score": score,
                        "bbox": (x, y, bw, bh),
                        "mask_rle": mask_to_rle(comp_mask),
                        "area": int(area),
                        "source": get_source_name(image_path),
                    }
                )
    return pools


def build_background_pools(bg_records: list[dict], allow_set: set[str], target_classes: list[str]) -> dict[str, list[dict]]:
    pools = {target: [] for target in target_classes}
    by_path = {}
    for record in bg_records:
        image_path = record["image_path"].replace("\\", "/")
        if image_path not in allow_set:
            continue
        by_path[image_path] = record
        for target in target_classes:
            value = record["prompts"].get(target, {})
            if not value.get("hit"):
                pools[target].append(
                    {
                        "image_path": image_path,
                        "record": record,
                        "source": get_source_name(image_path),
                    }
                )
    return pools


def decode_background_prompt_masks(record: dict) -> dict[str, np.ndarray]:
    masks = {}
    for prompt, value in record["prompts"].items():
        if value.get("hit") and value.get("rle") is not None:
            masks[prompt] = rle_to_mask(value["rle"])
    return masks


def choose_background(rng: random.Random, target: str, fg_item: dict, background_pool: dict[str, list[dict]], spec: dict) -> dict:
    candidates = background_pool[target]
    if spec["same_source_only"]:
        same_source = [x for x in candidates if x["source"] == fg_item["source"]]
        if same_source:
            return rng.choice(same_source)
    return rng.choice(candidates)


def resize_patch(patch: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(scale - 1.0) < 1e-6:
        return patch, mask
    new_w = max(2, int(round(patch.shape[1] * scale)))
    new_h = max(2, int(round(patch.shape[0] * scale)))
    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return patch, (mask > 0).astype(np.uint8)


def maybe_flip(rng: random.Random, patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        patch = cv2.flip(patch, 1)
        mask = cv2.flip(mask, 1)
    return patch, mask


def build_occupied_mask(prompt_masks: dict[str, np.ndarray]) -> np.ndarray:
    if not prompt_masks:
        return None
    shape = next(iter(prompt_masks.values())).shape
    occupied = np.zeros(shape, dtype=np.uint8)
    for mask in prompt_masks.values():
        occupied |= (mask > 0).astype(np.uint8)
    return occupied


def place_patch(
    rng: random.Random,
    bg_h: int,
    bg_w: int,
    patch_mask: np.ndarray,
    occupied_mask: np.ndarray | None,
    max_overlap_ratio: float,
) -> tuple[int, int] | None:
    ph, pw = patch_mask.shape
    if ph >= bg_h - 2 * MIN_MARGIN or pw >= bg_w - 2 * MIN_MARGIN:
        return None

    max_y = bg_h - ph - MIN_MARGIN
    max_x = bg_w - pw - MIN_MARGIN
    if max_x <= MIN_MARGIN or max_y <= MIN_MARGIN:
        return None

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        y = rng.randint(MIN_MARGIN, max_y)
        x = rng.randint(MIN_MARGIN, max_x)
        if occupied_mask is None:
            return x, y
        overlap = occupied_mask[y:y + ph, x:x + pw]
        denom = max(int(patch_mask.sum()), 1)
        overlap_ratio = float((overlap.astype(bool) & patch_mask.astype(bool)).sum()) / denom
        if overlap_ratio <= max_overlap_ratio:
            return x, y
    return None


def paste_patch(bg_rgb: np.ndarray, patch_gray: np.ndarray, patch_mask: np.ndarray, x: int, y: int):
    ph, pw = patch_mask.shape
    patch_rgb = np.stack([patch_gray] * 3, axis=-1).astype(np.float32)
    alpha = patch_mask.astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=EDGE_BLUR_SIGMA, sigmaY=EDGE_BLUR_SIGMA)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    roi = bg_rgb[y:y + ph, x:x + pw].astype(np.float32)
    blended = roi * (1.0 - alpha) + patch_rgb * alpha
    bg_rgb[y:y + ph, x:x + pw] = np.clip(blended, 0, 255).astype(np.uint8)


def build_synthetic_prompts(bg_record: dict, class_order: list[str], target: str, target_mask: np.ndarray, score: float) -> dict:
    prompts = {}
    bg_prompts = bg_record["prompts"]
    for cls_name in class_order:
        value = copy.deepcopy(bg_prompts.get(cls_name, {"hit": False}))
        prompts[cls_name] = value
    prompts[target] = {
        "hit": True,
        "score": round(float(score), 4),
        "instances": int(cv2.connectedComponents(target_mask.astype(np.uint8))[0] - 1),
        "rle": mask_to_rle(target_mask),
    }
    return prompts


def write_yolo_label(label_path: Path, class_order: list[str], prompt_masks: dict[str, np.ndarray]):
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(class_order)}
    lines = []
    for cls_name in class_order:
        value = prompt_masks.get(cls_name)
        if value is None:
            continue
        for cx, cy, bw, bh in mask_to_bboxes(value):
            lines.append(f"{class_to_id[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_rgb_jpg(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_json(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


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
        image_path = image_path.replace("\\", "/")
        if image_path in seen:
            continue
        merged_list.append(image_path)
        seen.add(image_path)

    out_train_list_path.parent.mkdir(parents=True, exist_ok=True)
    out_train_list_path.write_text("\n".join(merged_list) + "\n", encoding="utf-8")
    return len(d5d7_records), len(merged_records), len(merged_list)


def run_copy_paste(args):
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    images_dir, labels_dir = prepare_output_dir(output_dir)

    fg_records = load_json(Path(args.fg_json))
    bg_records = load_json(Path(args.bg_json))
    allow_set = load_file_list(Path(args.train_list))

    target_specs = {name: TARGET_SPECS[name] for name in args.targets}
    if args.num_trash_can is not None and "trash can" in target_specs:
        target_specs["trash can"]["num_samples"] = int(args.num_trash_can)
    if args.num_computer is not None and "computer" in target_specs:
        target_specs["computer"]["num_samples"] = int(args.num_computer)

    class_order = infer_class_order(bg_records, args.targets)
    fg_pools = extract_foreground_pool(fg_records, target_specs)
    bg_pools = build_background_pools(bg_records, allow_set, args.targets)

    for target in args.targets:
        if not fg_pools[target]:
            raise RuntimeError(f"前景池为空: {target}")
        if not bg_pools[target]:
            raise RuntimeError(f"背景池为空: {target}")

    synthetic_records = []
    manifest_rows = []
    img_counter = 0

    for target in args.targets:
        spec = target_specs[target]
        pbar = tqdm(range(spec["num_samples"]), desc=f"生成 {target}", unit="img")
        for _ in pbar:
            fg_item = rng.choice(fg_pools[target])
            bg_item = choose_background(rng, target, fg_item, bg_pools, spec)

            fg_gray = read_gray(fg_item["image_path"])
            bg_gray = read_gray(bg_item["image_path"])
            if fg_gray is None or bg_gray is None:
                continue

            x, y, bw, bh = fg_item["bbox"]
            comp_mask = rle_to_mask(fg_item["mask_rle"])
            fg_patch = fg_gray[y:y + bh, x:x + bw]
            if fg_patch.shape[:2] != comp_mask.shape:
                continue

            fg_patch, comp_mask = maybe_flip(rng, fg_patch, comp_mask)
            scale = rng.uniform(*spec["scale_range"])
            fg_patch, comp_mask = resize_patch(fg_patch, comp_mask, scale)

            bg_prompt_masks = decode_background_prompt_masks(bg_item["record"])
            occupied_mask = build_occupied_mask(bg_prompt_masks)
            placement = place_patch(
                rng,
                bg_gray.shape[0],
                bg_gray.shape[1],
                comp_mask,
                occupied_mask,
                spec["max_overlap_ratio"],
            )
            if placement is None:
                continue

            px, py = placement
            out_rgb = np.stack([bg_gray] * 3, axis=-1)
            paste_patch(out_rgb, fg_patch, comp_mask, px, py)

            target_mask = np.zeros(bg_gray.shape, dtype=np.uint8)
            ph, pw = comp_mask.shape
            target_mask[py:py + ph, px:px + pw] = comp_mask

            prompts = build_synthetic_prompts(bg_item["record"], class_order, target, target_mask, fg_item["score"])
            final_prompt_masks = {}
            for cls_name, value in prompts.items():
                if value.get("hit") and value.get("rle") is not None:
                    final_prompt_masks[cls_name] = rle_to_mask(value["rle"])

            stem = f"cp_{target.replace(' ', '_')}_{img_counter:05d}"
            rel_image_path = (output_dir / "images" / f"{stem}.jpg").relative_to(ROOT).as_posix()
            save_rgb_jpg(images_dir / f"{stem}.jpg", out_rgb)
            write_yolo_label(labels_dir / f"{stem}.txt", class_order, final_prompt_masks)

            synthetic_records.append(
                {
                    "image_path": rel_image_path,
                    "prompts": prompts,
                }
            )
            manifest_rows.append(
                {
                    "image_path": rel_image_path,
                    "target": target,
                    "fg_image": fg_item["image_path"],
                    "bg_image": bg_item["image_path"],
                    "score": round(float(fg_item["score"]), 4),
                    "scale": round(float(scale), 4),
                    "fg_area_px": int(fg_item["area"]),
                }
            )
            img_counter += 1
            pbar.set_postfix(done=img_counter)

    only_json = output_dir / "pred_copy_paste_only.json"
    merged_json = output_dir / "pred_copy_paste_merged.json"
    manifest_csv = output_dir / "manifest.csv"

    write_json(only_json, synthetic_records)

    merged_records = bg_records + synthetic_records
    write_json(merged_json, merged_records)

    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "target", "fg_image", "bg_image", "score", "scale", "fg_area_px"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    d5d7_base_count, d5d7_merged_count, d5d7_train_count = merge_d5d7_training_assets(
        Path(args.d5d7_json),
        Path(args.d5d7_train_list),
        synthetic_records,
        Path(args.d5d7_out_json),
        Path(args.d5d7_out_train_list),
    )

    print(f"输出目录: {output_dir}")
    print(f"synthetic 样本: {len(synthetic_records)}")
    for target in args.targets:
        count = sum(1 for x in synthetic_records if x["prompts"][target]["hit"])
        print(f"  {target}: {count}")
    print(f"YOLO bbox 标签: {labels_dir}")
    print(f"synthetic-only json: {only_json}")
    print(f"merged json: {merged_json}")
    print(f"manifest: {manifest_csv}")
    print(f"d5&7 merged json: {args.d5d7_out_json} ({d5d7_base_count} -> {d5d7_merged_count})")
    print(f"d5&7 merged train_list: {args.d5d7_out_train_list} ({d5d7_train_count})")


def parse_args():
    parser = argparse.ArgumentParser(description="离线 Copy-Paste 生成尾类增强图、bbox 和 pred_json")
    parser.add_argument("--fg-json", default=str(DEFAULT_FG_JSON), help="前景伪标签 JSON")
    parser.add_argument("--bg-json", default=str(DEFAULT_BG_JSON), help="背景伪标签 JSON")
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST), help="训练列表 txt")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--d5d7-json", default=str(DEFAULT_D5D7_JSON), help="d5&7 原始训练伪标签 JSON")
    parser.add_argument("--d5d7-train-list", default=str(DEFAULT_D5D7_TRAIN_LIST), help="d5&7 原始训练列表 txt")
    parser.add_argument("--d5d7-out-json", default=str(DEFAULT_D5D7_MERGED_JSON), help="导出的 d5&7+copy_paste 训练 JSON")
    parser.add_argument("--d5d7-out-train-list", default=str(DEFAULT_D5D7_MERGED_LIST), help="导出的 d5&7+copy_paste 训练列表 txt")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="随机种子")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["trash can"],
        choices=sorted(TARGET_SPECS.keys()),
        help="要增强的类别",
    )
    parser.add_argument("--num-trash-can", type=int, default=None, help="单独指定 trash can 生成数量")
    parser.add_argument("--num-computer", type=int, default=None, help="单独指定 computer 生成数量")
    return parser.parse_args()


if __name__ == "__main__":
    run_copy_paste(parse_args())
