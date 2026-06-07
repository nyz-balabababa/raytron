#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def rle_to_mask(rle):
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
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_image_path(image_path):
    image_path = str(image_path).replace("\\", "/")
    return image_path


def load_tasks(task_json):
    data = load_json(task_json)
    if not isinstance(data, list):
        raise ValueError(f"task json 必须是 list，当前为 {type(data).__name__}")

    tasks = []
    for item in data:
        tasks.append(
            {
                "ann_id": int(item["ann_id"]),
                "image_path": normalize_image_path(item["image_path"]),
                "prompt": str(item.get("text_prompt", item.get("prompt", item.get("text")))),
            }
        )
    return tasks


def load_predictions(pred_json):
    data = load_json(pred_json)
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        pred_list = data["predictions"]
    elif isinstance(data, list):
        pred_list = data
    else:
        raise ValueError("predictions json 格式不支持")

    pred_by_ann = {}
    for item in pred_list:
        ann_id = int(item["ann_id"])
        rle = item.get("rle")
        pred_by_ann[ann_id] = rle
    return pred_by_ann


def load_reference_prompts(ref_json):
    data = load_json(ref_json)
    if not isinstance(data, list):
        raise ValueError("reference json 必须是 prompt_test 输出的 list")

    ref = {}
    for item in data:
        image_path = normalize_image_path(item["image_path"])
        ref[image_path] = item["prompts"]
    return ref


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_iou(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    inter = int((pred_mask & gt_mask).sum())
    union = int((pred_mask | gt_mask).sum())
    return safe_div(inter, union), inter


def compute_image_quality(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "brightness": None,
            "contrast_std": None,
            "laplacian": None,
        }
    return {
        "brightness": float(img.mean()),
        "contrast_std": float(img.std()),
        "laplacian": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }


def quantile_bins(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float32)
    if len(arr) == 0:
        return None
    q1 = float(np.quantile(arr, 1 / 3))
    q2 = float(np.quantile(arr, 2 / 3))
    return q1, q2


def bucketize(value, bins):
    if value is None or bins is None:
        return "unknown"
    q1, q2 = bins
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def main():
    parser = argparse.ArgumentParser(description="按 prompt 统计分割预测错误模式")
    parser.add_argument("--tasks", required=True, help="任务 json，例如 test/json/val_tasks1.json")
    parser.add_argument("--predictions", required=True, help="提交格式 predictions.json")
    parser.add_argument("--reference", required=True, help="参考 prompt json，例如 pred_val_tasks1.json")
    parser.add_argument("--image-root", required=True, help="图片根目录")
    parser.add_argument("--out-json", default="", help="可选，导出详细分析 json")
    parser.add_argument("--coverage-thresh", type=float, default=0.6, help="低覆盖判定阈值")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="整图错误判定 IoU 阈值")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    pred_by_ann = load_predictions(args.predictions)
    ref_by_image = load_reference_prompts(args.reference)
    image_root = Path(args.image_root)

    image_quality = {}
    contrast_values = []
    for task in tasks:
        image_path = task["image_path"]
        if image_path not in image_quality:
            quality = compute_image_quality(image_root / image_path)
            image_quality[image_path] = quality
            contrast_values.append(quality["contrast_std"])
    contrast_bins = quantile_bins(contrast_values)

    per_class = defaultdict(lambda: defaultdict(float))
    per_class_lists = defaultdict(lambda: defaultdict(list))
    per_image_results = defaultdict(list)
    detailed_rows = []

    for task in tasks:
        ann_id = task["ann_id"]
        image_path = task["image_path"]
        prompt = task["prompt"]

        ref_prompts = ref_by_image.get(image_path, {})
        ref_info = ref_prompts.get(prompt)
        if ref_info is None:
            continue

        gt_rle = ref_info.get("rle")
        gt_hit = bool(ref_info.get("hit")) and gt_rle is not None
        gt_mask = rle_to_mask(gt_rle) if gt_hit else None

        pred_rle = pred_by_ann.get(ann_id)
        pred_hit = pred_rle is not None
        pred_mask = rle_to_mask(pred_rle) if pred_hit else None

        gt_area = int(gt_mask.sum()) if gt_mask is not None else 0
        pred_area = int(pred_mask.sum()) if pred_mask is not None else 0
        iou = 0.0
        inter = 0
        gt_coverage = 0.0
        pred_precision = 0.0

        if gt_mask is not None and pred_mask is not None:
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            iou, inter = compute_iou(pred_mask, gt_mask)
            gt_coverage = safe_div(inter, gt_area)
            pred_precision = safe_div(inter, pred_area)

        quality = image_quality[image_path]
        contrast_bucket = bucketize(quality["contrast_std"], contrast_bins)

        row = {
            "ann_id": ann_id,
            "image_path": image_path,
            "folder": Path(image_path).parts[1] if len(Path(image_path).parts) > 1 else "unknown",
            "prompt": prompt,
            "gt_hit": gt_hit,
            "pred_hit": pred_hit and pred_area > 0,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "iou": iou,
            "gt_coverage": gt_coverage,
            "pred_precision": pred_precision,
            "brightness": quality["brightness"],
            "contrast_std": quality["contrast_std"],
            "laplacian": quality["laplacian"],
            "contrast_bucket": contrast_bucket,
        }

        if gt_hit:
            per_class[prompt]["gt_positive"] += 1
            per_class_lists[prompt]["iou_pos"].append(iou)
            per_class_lists[prompt]["coverage_pos"].append(gt_coverage)
            per_class_lists[prompt][f"contrast_{contrast_bucket}_iou"].append(iou)
            per_class_lists[prompt][f"contrast_{contrast_bucket}_coverage"].append(gt_coverage)

            if not row["pred_hit"]:
                per_class[prompt]["miss_count"] += 1
                row["error_type"] = "miss"
            elif gt_coverage < args.coverage_thresh:
                per_class[prompt]["under_coverage_count"] += 1
                row["error_type"] = "under_coverage"
            else:
                row["error_type"] = "ok_positive"
        else:
            per_class[prompt]["gt_negative"] += 1
            if row["pred_hit"]:
                per_class[prompt]["false_positive_count"] += 1
                row["error_type"] = "false_positive"
            else:
                row["error_type"] = "ok_negative"

        per_image_results[image_path].append(row)
        detailed_rows.append(row)

    image_all_wrong = []
    for image_path, rows in per_image_results.items():
        wrong_flags = []
        for row in rows:
            if row["gt_hit"]:
                wrong_flags.append((not row["pred_hit"]) or (row["iou"] < args.iou_thresh))
            else:
                wrong_flags.append(row["pred_hit"])
        if wrong_flags and all(wrong_flags):
            image_all_wrong.append(
                {
                    "image_path": image_path,
                    "folder": rows[0]["folder"],
                    "brightness": rows[0]["brightness"],
                    "contrast_std": rows[0]["contrast_std"],
                    "laplacian": rows[0]["laplacian"],
                    "prompts": [r["prompt"] for r in rows],
                }
            )

    summary = {
        "tasks_total": len(detailed_rows),
        "images_total": len(per_image_results),
        "contrast_bins": contrast_bins,
        "per_class": {},
        "all_prompts_wrong_images": image_all_wrong,
    }

    print("\n=== Per-class Summary ===")
    for cls_name in sorted(per_class.keys()):
        stats = per_class[cls_name]
        lists = per_class_lists[cls_name]

        gt_pos = int(stats["gt_positive"])
        gt_neg = int(stats["gt_negative"])
        miss = int(stats["miss_count"])
        fp = int(stats["false_positive_count"])
        under = int(stats["under_coverage_count"])

        info = {
            "gt_positive": gt_pos,
            "gt_negative": gt_neg,
            "false_positive_rate": safe_div(fp, gt_neg),
            "miss_rate": safe_div(miss, gt_pos),
            "under_coverage_rate": safe_div(under, gt_pos),
            "mean_iou_on_positive": float(np.mean(lists["iou_pos"])) if lists["iou_pos"] else 0.0,
            "mean_coverage_on_positive": float(np.mean(lists["coverage_pos"])) if lists["coverage_pos"] else 0.0,
            "contrast": {},
        }

        for bucket in ["low", "mid", "high", "unknown"]:
            ious = lists.get(f"contrast_{bucket}_iou", [])
            covers = lists.get(f"contrast_{bucket}_coverage", [])
            if ious or covers:
                info["contrast"][bucket] = {
                    "count": len(ious),
                    "mean_iou": float(np.mean(ious)) if ious else 0.0,
                    "mean_coverage": float(np.mean(covers)) if covers else 0.0,
                }

        summary["per_class"][cls_name] = info

        print(
            f"{cls_name:16s} "
            f"FP={info['false_positive_rate']:.3f} "
            f"MISS={info['miss_rate']:.3f} "
            f"UNDER={info['under_coverage_rate']:.3f} "
            f"IOU={info['mean_iou_on_positive']:.3f} "
            f"COV={info['mean_coverage_on_positive']:.3f}"
        )
        for bucket in ["low", "mid", "high"]:
            bucket_info = info["contrast"].get(bucket)
            if bucket_info:
                print(
                    f"  - contrast={bucket:4s} "
                    f"n={bucket_info['count']:4d} "
                    f"iou={bucket_info['mean_iou']:.3f} "
                    f"cov={bucket_info['mean_coverage']:.3f}"
                )

    print("\n=== All-prompts-wrong Images ===")
    print(f"count = {len(image_all_wrong)}")
    for item in image_all_wrong[:30]:
        print(
            f"{item['image_path']} | std={item['contrast_std']:.1f} "
            f"| lap={item['laplacian']:.1f} | prompts={','.join(item['prompts'])}"
        )
    if len(image_all_wrong) > 30:
        print(f"... 还有 {len(image_all_wrong) - 30} 张")

    if args.out_json:
        out = {
            "summary": summary,
            "rows": detailed_rows,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
