#!/usr/bin/env python3
"""
teacher 伪标签分档脚本。

输入：
- 一份主 teacher 伪标签 JSON（prompt 风格）
- 可选多份 variant 伪标签 JSON（用于一致性筛查）

输出到 label_analysis/：
- pseudo_A.json
- pseudo_B.json
- pseudo_C.json
- review_list.txt
- denylist.txt
- refine_summary.json

分档目标：
- A：高质量，可直接强监督
- B：可疑但可用，建议降权/边界弱监督
- C：高风险，建议剔除或人工复核
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm 缺失时退化为普通迭代
    tqdm = None


ROOT = Path(__file__).resolve().parents[2]

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
REFINE_INPUT_JSON = ROOT / "test" / "label_analysis" / "稀有类" / "val-out" / "pred_val_changwei.json"
REFINE_VARIANT_JSONS: list[Path] = []
REFINE_IMAGE_ROOT = ROOT
REFINE_OUTPUT_DIR = ROOT / "test" / "label_analysis" / "label_analysis_ABC(changwei-val)"
# ─────────────────────────────────────────────────────────────────────


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_path(path_str: str) -> str:
    return str(path_str).replace("\\", "/")


def load_prompt_json(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是 prompt 风格 JSON list")

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in data:
        image_path = normalize_path(item["image_path"])
        out[image_path] = item.get("prompts", {})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils

        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return mask_utils.decode(rle_copy).astype(np.uint8)
    except Exception:
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
            mask[pos: pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def resolve_image_path(image_root: Path, rel_path: str) -> Path:
    rel_path = normalize_path(rel_path)
    candidates = [
        image_root / rel_path,
        image_root / (rel_path[5:] if rel_path.startswith("test/") else f"test/{rel_path}"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def compute_image_quality(image_path: Path) -> Dict[str, Optional[float]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"brightness": None, "contrast_std": None, "laplacian": None}
    return {
        "brightness": float(img.mean()),
        "contrast_std": float(img.std()),
        "laplacian": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def mask_connected_components(mask: np.ndarray) -> Tuple[int, float]:
    if int(mask.sum()) == 0:
        return 0, 0.0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([])
    if len(areas) == 0:
        return 0, 0.0
    return int(len(areas)), float(areas.max()) / max(float(mask.sum()), 1.0)


def touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def safe_quantile(values: List[float], q: float, default: float) -> float:
    if not values:
        return default
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def build_prompt_stats(base_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    by_prompt = defaultdict(lambda: defaultdict(list))
    for image_path, prompts in progress(
        base_data.items(),
        total=len(base_data),
        desc="Build prompt stats",
        unit="img",
    ):
        for prompt, info in prompts.items():
            if not info.get("hit") or info.get("rle") is None:
                continue
            mask = rle_to_mask(info["rle"])
            if int(mask.sum()) == 0:
                continue
            area_ratio = float(mask.sum()) / float(mask.size)
            num_cc, max_comp_ratio = mask_connected_components(mask)
            by_prompt[prompt]["score"].append(float(info.get("score", 0.0)))
            by_prompt[prompt]["area_ratio"].append(area_ratio)
            by_prompt[prompt]["num_cc"].append(float(num_cc))
            by_prompt[prompt]["max_comp_ratio"].append(max_comp_ratio)

    stats: Dict[str, Dict[str, float]] = {}
    for prompt, vals in by_prompt.items():
        stats[prompt] = {
            "score_q25": safe_quantile(vals["score"], 0.25, 0.0),
            "score_q50": safe_quantile(vals["score"], 0.50, 0.0),
            "area_q05": safe_quantile(vals["area_ratio"], 0.05, 0.0),
            "area_q95": safe_quantile(vals["area_ratio"], 0.95, 1.0),
            "cc_q75": safe_quantile(vals["num_cc"], 0.75, 1.0),
            "cc_q90": safe_quantile(vals["num_cc"], 0.90, 2.0),
            "max_comp_q10": safe_quantile(vals["max_comp_ratio"], 0.10, 0.3),
        }
    return stats


def classify_sample(
    prompt: str,
    info: Dict[str, Any],
    prompt_stats: Dict[str, Dict[str, float]],
    quality: Dict[str, Optional[float]],
    variant_infos: List[Optional[Dict[str, Any]]],
) -> Tuple[str, List[str], Dict[str, Any]]:
    prompt_stat = prompt_stats.get(prompt, {})
    hit = bool(info.get("hit")) and info.get("rle") is not None
    score = float(info.get("score", 0.0))

    variant_hit_flags = []
    variant_ious = []

    if hit:
        base_mask = rle_to_mask(info["rle"])
        area_ratio = float(base_mask.sum()) / max(float(base_mask.size), 1.0)
        num_cc, max_comp_ratio = mask_connected_components(base_mask)
        border_flag = touches_border(base_mask)
        base_area = int(base_mask.sum())
    else:
        base_mask = None
        area_ratio = 0.0
        num_cc = 0
        max_comp_ratio = 0.0
        border_flag = False
        base_area = 0

    for variant_info in variant_infos:
        v_hit = bool(variant_info and variant_info.get("hit") and variant_info.get("rle") is not None)
        variant_hit_flags.append(v_hit)
        if hit and v_hit:
            variant_mask = rle_to_mask(variant_info["rle"])
            variant_ious.append(compute_mask_iou(base_mask, variant_mask))

    reasons_b: List[str] = []
    reasons_c: List[str] = []

    if hit:
        if score < prompt_stat.get("score_q25", 0.0):
            reasons_b.append("low_score_q25")
        if score < prompt_stat.get("score_q25", 0.0) * 0.95:
            reasons_c.append("very_low_score")

        if area_ratio < prompt_stat.get("area_q05", 0.0) * 0.7 and base_area >= 64:
            reasons_b.append("too_small_area")
        if prompt_stat.get("area_q95", 1.0) > 0 and area_ratio > prompt_stat.get("area_q95", 1.0) * 1.4:
            reasons_c.append("too_large_area")

        if num_cc > prompt_stat.get("cc_q75", 1.0) + 2:
            reasons_b.append("fragmented")
        if num_cc > max(prompt_stat.get("cc_q90", 2.0) + 3, 6):
            reasons_c.append("highly_fragmented")

        if max_comp_ratio < min(prompt_stat.get("max_comp_q10", 0.3), 0.35) and num_cc >= 3:
            reasons_b.append("weak_main_component")
        if max_comp_ratio < 0.18 and num_cc >= 4:
            reasons_c.append("very_weak_main_component")

        if border_flag and area_ratio > max(prompt_stat.get("area_q95", 1.0), 0.02):
            reasons_b.append("touch_border_large")

        contrast_std = quality.get("contrast_std")
        if contrast_std is not None and contrast_std < 18 and score < prompt_stat.get("score_q50", 0.0):
            reasons_b.append("low_contrast_low_score")

        if variant_infos:
            hit_switch = any(v != hit for v in variant_hit_flags)
            if hit_switch:
                reasons_c.append("variant_hit_flip")
            if variant_ious:
                mean_iou = float(np.mean(variant_ious))
                if mean_iou < 0.65:
                    reasons_b.append("variant_iou_low")
                if mean_iou < 0.40:
                    reasons_c.append("variant_iou_very_low")
            elif any(variant_hit_flags):
                reasons_c.append("variant_unmatched_shape")
    else:
        if variant_infos and any(variant_hit_flags):
            reasons_c.append("negative_not_stable")

    if reasons_c:
        label = "C"
    elif reasons_b:
        label = "B"
    else:
        label = "A"

    metrics = {
        "hit": hit,
        "score": score,
        "area_ratio": area_ratio,
        "num_cc": num_cc,
        "max_comp_ratio": max_comp_ratio,
        "touch_border": border_flag,
        "contrast_std": quality.get("contrast_std"),
        "laplacian": quality.get("laplacian"),
        "variant_hit_flags": variant_hit_flags,
        "variant_mean_iou": float(np.mean(variant_ious)) if variant_ious else None,
    }
    return label, reasons_c + reasons_b, metrics


def append_prompt(output_map: Dict[str, Dict[str, Any]], image_path: str, prompt: str, info: Dict[str, Any]) -> None:
    if image_path not in output_map:
        output_map[image_path] = {"image_path": image_path, "prompts": {}}
    output_map[image_path]["prompts"][prompt] = info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按形态 + 一致性对 teacher 伪标签做 A/B/C 分档")
    parser.add_argument("--input-json", type=Path, default=REFINE_INPUT_JSON, help="主 teacher 伪标签 JSON")
    parser.add_argument(
        "--variant-json",
        nargs="*",
        type=Path,
        default=REFINE_VARIANT_JSONS,
        help="可选，多份 variant 伪标签 JSON，用于一致性筛查",
    )
    parser.add_argument("--image-root", type=Path, default=REFINE_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REFINE_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading base pseudo labels: {args.input_json}")
    base_data = load_prompt_json(args.input_json)
    variant_datas = []
    for path in args.variant_json:
        print(f"Loading variant pseudo labels: {path}")
        variant_datas.append(load_prompt_json(path))
    prompt_stats = build_prompt_stats(base_data)

    image_quality_cache: Dict[str, Dict[str, Optional[float]]] = {}
    out_a: Dict[str, Dict[str, Any]] = {}
    out_b: Dict[str, Dict[str, Any]] = {}
    out_c: Dict[str, Dict[str, Any]] = {}
    review_rows: List[str] = []
    row_details: List[Dict[str, Any]] = []
    image_c_counter = defaultdict(int)
    image_prompt_counter = defaultdict(int)

    total_prompt_rows = sum(len(prompts) for prompts in base_data.values())
    row_bar = None
    if tqdm is not None:
        row_bar = tqdm(total=total_prompt_rows, desc="Classify prompts", unit="prompt")

    for image_path, prompts in progress(
        base_data.items(),
        total=len(base_data),
        desc="Refine images",
        unit="img",
    ):
        if image_path not in image_quality_cache:
            image_quality_cache[image_path] = compute_image_quality(resolve_image_path(args.image_root, image_path))
        quality = image_quality_cache[image_path]

        for prompt, info in prompts.items():
            variant_infos = [variant_data.get(image_path, {}).get(prompt) for variant_data in variant_datas]
            label, reasons, metrics = classify_sample(prompt, info, prompt_stats, quality, variant_infos)

            if label == "A":
                append_prompt(out_a, image_path, prompt, info)
            elif label == "B":
                append_prompt(out_b, image_path, prompt, info)
                review_rows.append(f"{image_path}\t{prompt}\tB\t{','.join(reasons)}")
            else:
                append_prompt(out_c, image_path, prompt, info)
                review_rows.append(f"{image_path}\t{prompt}\tC\t{','.join(reasons)}")
                image_c_counter[image_path] += 1

            image_prompt_counter[image_path] += 1
            row_details.append(
                {
                    "image_path": image_path,
                    "prompt": prompt,
                    "label": label,
                    "reasons": reasons,
                    **metrics,
                }
            )
            if row_bar is not None:
                row_bar.update(1)

    if row_bar is not None:
        row_bar.close()

    denylist_images = sorted(
        image_path
        for image_path, c_count in image_c_counter.items()
        if c_count >= 2 or c_count == image_prompt_counter[image_path]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "pseudo_A.json", list(out_a.values()))
    dump_json(args.output_dir / "pseudo_B.json", list(out_b.values()))
    dump_json(args.output_dir / "pseudo_C.json", list(out_c.values()))

    with open(args.output_dir / "review_list.txt", "w", encoding="utf-8") as f:
        f.write("# image_path\tprompt\tlabel\treasons\n")
        for line in review_rows:
            f.write(line + "\n")

    with open(args.output_dir / "denylist.txt", "w", encoding="utf-8") as f:
        for image_path in denylist_images:
            f.write(image_path + "\n")

    summary = {
        "input_json": str(args.input_json),
        "variant_jsons": [str(p) for p in args.variant_json],
        "output_dir": str(args.output_dir),
        "images_total": len(base_data),
        "prompt_stats": prompt_stats,
        "counts": {
            "A_prompts": sum(len(v["prompts"]) for v in out_a.values()),
            "B_prompts": sum(len(v["prompts"]) for v in out_b.values()),
            "C_prompts": sum(len(v["prompts"]) for v in out_c.values()),
            "review_rows": len(review_rows),
            "denylist_images": len(denylist_images),
        },
        "rows": row_details,
    }
    dump_json(args.output_dir / "refine_summary.json", summary)

    print(f"images_total: {len(base_data)}")
    print(f"A prompts:    {summary['counts']['A_prompts']}")
    print(f"B prompts:    {summary['counts']['B_prompts']}")
    print(f"C prompts:    {summary['counts']['C_prompts']}")
    print(f"review rows:  {summary['counts']['review_rows']}")
    print(f"denylist img: {summary['counts']['denylist_images']}")
    print(f"output_dir:   {args.output_dir}")


if __name__ == "__main__":
    main()
