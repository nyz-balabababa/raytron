#!/usr/bin/env python3
"""
新旧伪标签质量对比脚本。

只做对比不做判断——告诉你两版差异在哪、有多大，不声称谁更准。

可以直接运行（改下面的路径），也可以用命令行覆盖:
  python compare_pseudo_labels.py
  python compare_pseudo_labels.py --old other_old.json --new other_new.json
"""

# ══════════════════════════════════════════════════════════════════════
# 配置路径 —— 改这里就行
# ══════════════════════════════════════════════════════════════════════

# 旧版伪标签 JSON（参照系）
OLD_PRED_JSON = "test/sam3_label_output/pred_val_tasks.json"

# 新版伪标签 JSON（待评估的新生成结果）
NEW_PRED_JSON = "test/sam3_label_output/pred_val_label(low).json"

# 图片根目录（项目根目录即可，脚本会自动兼容 test/ 前缀）
IMAGE_ROOT = "."

# 输出目录（保存对比报告 + 分歧样本列表）
OUT_DIR = "test/pseudo_label_compare"

# 指定要对比的 prompt 列表，None 表示自动收集两版所有 prompt
# 例如: ["person", "car", "building", "tree", "animal"]
PROMPTS = None

# ══════════════════════════════════════════════════════════════════════

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# RLE 解码（兼容 pycocotools / 纯 Python 回退）
# ══════════════════════════════════════════════════════════════════════


def rle_to_mask(rle):
    """RLE → numpy uint8 mask。"""
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
            mask[pos : pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


# ══════════════════════════════════════════════════════════════════════
# JSON 加载
# ══════════════════════════════════════════════════════════════════════


def load_pseudo_label_json(path):
    """
    加载 prompt_test 输出的伪标签 JSON。
    格式: [{"image_path": ..., "prompts": {"car": {"hit":..., "score":..., "rle":...}, ...}}, ...]
    返回: {normalized_image_path: {"prompts": {...}}}
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"伪标签 JSON 必须是 list，当前类型: {type(data).__name__}")

    result = {}
    for item in tqdm(data, desc="  解析JSON条目", unit="图"):
        image_path = str(item["image_path"]).replace("\\", "/")
        result[image_path] = item.get("prompts", {})
    return result


# ══════════════════════════════════════════════════════════════════════
# 图像质量
# ══════════════════════════════════════════════════════════════════════


def compute_image_quality(image_path):
    """返回 brightness / contrast_std / laplacian_var。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"brightness": None, "contrast_std": None, "laplacian": None}
    return {
        "brightness": float(img.mean()),
        "contrast_std": float(img.std()),
        "laplacian": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }


def resolve_image_path(image_root, rel_path):
    """兼容 test/data1/xxx.jpg 和 data1/xxx.jpg。"""
    rel_path = str(rel_path).replace("\\", "/")
    candidates = [
        image_root / rel_path,
        image_root / (rel_path[5:] if rel_path.startswith("test/") else "test/" + rel_path),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


# ══════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_mask_iou(mask_a, mask_b):
    """两个 uint8 mask 的 IoU。"""
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_a = (mask_a > 0).astype(np.uint8)
    mask_b = (mask_b > 0).astype(np.uint8)
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    if union == 0:
        return 1.0  # 两版都没检出
    return inter / union


def mask_connected_components(mask):
    """返回连通域数量和最大连通域占比。"""
    if mask.sum() == 0:
        return 0, 0.0
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # 去掉背景 (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([])
    if len(areas) == 0:
        return 0, 0.0
    max_area = float(areas.max())
    total_area = float(mask.sum())
    return int(len(areas)), max_area / max(total_area, 1.0)


# ══════════════════════════════════════════════════════════════════════
# 主对比逻辑
# ══════════════════════════════════════════════════════════════════════


def collect_all_prompts(old_data, new_data):
    """收集两版中出现的所有 prompt 名称。"""
    prompts = set()
    for image_path, prompts_dict in old_data.items():
        prompts.update(prompts_dict.keys())
    for image_path, prompts_dict in new_data.items():
        prompts.update(prompts_dict.keys())
    return sorted(prompts)


def compute_per_class_stats(data, prompts):
    """计算每个 prompt 的激活率、avg_score、样本数统计。"""
    stats = {}
    for p in prompts:
        total = 0
        hit_count = 0
        scores = []
        for image_path, prompts_dict in tqdm(data.items(), desc=f"  {p:16s}", unit="图", leave=False):
            info = prompts_dict.get(p)
            if info is None:
                continue
            total += 1
            if info.get("hit"):
                hit_count += 1
                score = info.get("score")
                if score is not None:
                    scores.append(float(score))
        stats[p] = {
            "total_samples": total,
            "hit_count": hit_count,
            "activation_rate": safe_div(hit_count, total),
            "avg_score": float(np.mean(scores)) if scores else None,
            "median_score": float(np.median(scores)) if scores else None,
            "score_std": float(np.std(scores)) if scores else None,
            "score_p25": float(np.percentile(scores, 25)) if scores else None,
            "score_p75": float(np.percentile(scores, 75)) if scores else None,
        }
    return stats


def build_pairwise(
    old_data,
    new_data,
    image_root,
    prompts,
    mask_cache_dir,
):
    """
    遍历两版共同存在的 (image, prompt)，构建对比记录。

    返回:
        pairs:  [(image_path, prompt, row_dict), ...]
        old_only: {(image, prompt)}  仅在旧版中
        new_only: {(image, prompt)}  仅在新版中
    """
    pairs = []
    old_only = set()
    new_only = set()
    quality_cache = {}

    all_images = set(old_data.keys()) | set(new_data.keys())

    for image_path in tqdm(sorted(all_images), desc="  逐张对比图片", unit="图"):
        old_prompts = old_data.get(image_path, {})
        new_prompts = new_data.get(image_path, {})

        # 图像质量缓存
        if image_path not in quality_cache:
            abs_path = resolve_image_path(image_root, image_path)
            quality_cache[image_path] = compute_image_quality(abs_path) if abs_path.exists() else {
                "brightness": None, "contrast_std": None, "laplacian": None
            }

        for prompt in prompts:
            in_old = prompt in old_prompts
            in_new = prompt in new_prompts

            if in_old and not in_new:
                old_only.add((image_path, prompt))
                continue
            if in_new and not in_old:
                new_only.add((image_path, prompt))
                continue
            if not in_old and not in_new:
                continue

            # 两版都有
            old_info = old_prompts[prompt]
            new_info = new_prompts[prompt]
            old_hit = bool(old_info.get("hit"))
            new_hit = bool(new_info.get("hit"))
            old_score = float(old_info.get("score", 0))
            new_score = float(new_info.get("score", 0))
            old_rle = old_info.get("rle") if old_hit else None
            new_rle = new_info.get("rle") if new_hit else None

            # IoU 只在两版都 hit 时计算
            iou = None
            if old_hit and new_hit and old_rle is not None and new_rle is not None:
                try:
                    old_mask = _decode_cached(old_data, image_path, prompt, old_rle, mask_cache_dir, "old")
                    new_mask = _decode_cached(new_data, image_path, prompt, new_rle, mask_cache_dir, "new")
                    iou = compute_mask_iou(old_mask, new_mask)
                except Exception:
                    iou = None

            # 老版掩码形态
            old_cc = old_max_comp = None
            if old_hit and old_rle is not None:
                try:
                    old_mask = _decode_cached(old_data, image_path, prompt, old_rle, mask_cache_dir, "old")
                    old_cc, old_max_comp = mask_connected_components(old_mask)
                    old_area_ratio = safe_div(int(old_mask.sum()), old_mask.size)
                except Exception:
                    old_area_ratio = None
            else:
                old_area_ratio = None

            # 新版掩码形态
            new_cc = new_max_comp = None
            if new_hit and new_rle is not None:
                try:
                    new_mask = _decode_cached(new_data, image_path, prompt, new_rle, mask_cache_dir, "new")
                    new_cc, new_max_comp = mask_connected_components(new_mask)
                    new_area_ratio = safe_div(int(new_mask.sum()), new_mask.size)
                except Exception:
                    new_area_ratio = None
            else:
                new_area_ratio = None

            row = {
                "image_path": image_path,
                "prompt": prompt,
                "old_hit": old_hit,
                "new_hit": new_hit,
                "old_score": old_score,
                "new_score": new_score,
                "score_delta": new_score - old_score,
                "hit_agree": old_hit == new_hit,
                "iou": iou,
                "old_area_ratio": old_area_ratio,
                "new_area_ratio": new_area_ratio,
                "old_num_cc": old_cc,
                "new_num_cc": new_cc,
                "old_max_comp_ratio": old_max_comp,
                "new_max_comp_ratio": new_max_comp,
                "brightness": quality_cache[image_path]["brightness"],
                "contrast_std": quality_cache[image_path]["contrast_std"],
                "laplacian": quality_cache[image_path]["laplacian"],
            }
            pairs.append((image_path, prompt, row))

    return pairs, old_only, new_only


def _decode_cached(data, image_path, prompt, rle, cache_dir, tag):
    """RLE 解码为 mask，缓存到磁盘避免重复解码。"""
    cache_name = hashlib.md5(f"{tag}|{image_path}|{prompt}".encode()).hexdigest()
    cache_path = cache_dir / f"{cache_name}.png"
    if cache_path.exists():
        mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
    mask = rle_to_mask(rle)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), mask)
    return mask


def contrast_bucket(std_val, bins):
    if std_val is None or bins is None:
        return "unknown"
    q1, q2 = bins
    if std_val <= q1:
        return "low"
    if std_val <= q2:
        return "mid"
    return "high"


def build_contrast_bins(pairs):
    stds = [r["contrast_std"] for _, _, r in pairs if r["contrast_std"] is not None]
    if not stds:
        return None
    arr = np.array(stds, dtype=np.float32)
    q1 = float(np.quantile(arr, 1 / 3))
    q2 = float(np.quantile(arr, 2 / 3))
    return q1, q2


# ══════════════════════════════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════════════════════════════


def summarize(pairs, old_stats, new_stats, prompts, contrast_bins, old_only, new_only):
    """生成可打印和导出的汇总 report dict。"""

    # ── per-class cross-IoU ──
    per_class_iou = defaultdict(list)
    per_class_score_delta = defaultdict(list)
    per_class_hit_agree = defaultdict(lambda: {"agree": 0, "old_only_hit": 0, "new_only_hit": 0, "both_miss": 0})

    hit_flip_samples = []  # 命中状态翻转的样本

    for _, prompt, row in pairs:
        if row["iou"] is not None:
            per_class_iou[prompt].append(row["iou"])
        per_class_score_delta[prompt].append(row["score_delta"])

        if row["hit_agree"]:
            per_class_hit_agree[prompt]["agree"] += 1
            if not row["old_hit"]:
                per_class_hit_agree[prompt]["both_miss"] += 1
        else:
            if row["old_hit"] and not row["new_hit"]:
                per_class_hit_agree[prompt]["old_only_hit"] += 1
            else:
                per_class_hit_agree[prompt]["new_only_hit"] += 1
            hit_flip_samples.append(row)

    # ── cross-IoU 全局分布 ──
    all_ious = [iou for ious in per_class_iou.values() for iou in ious]
    iou_dist = {}
    if all_ious:
        iou_dist = {
            "count": len(all_ious),
            "mean": float(np.mean(all_ious)),
            "median": float(np.median(all_ious)),
            "std": float(np.std(all_ious)),
            "p25": float(np.percentile(all_ious, 25)),
            "p75": float(np.percentile(all_ious, 75)),
            "above_08_ratio": safe_div(sum(1 for x in all_ious if x > 0.8), len(all_ious)),
            "below_03_ratio": safe_div(sum(1 for x in all_ious if x < 0.3), len(all_ious)),
        }

    # ── per-class iou summary ──
    per_class_iou_summary = {}
    for cls_name in prompts:
        ious = per_class_iou.get(cls_name, [])
        per_class_iou_summary[cls_name] = {
            "iou_count": len(ious),
            "iou_mean": float(np.mean(ious)) if ious else None,
            "iou_median": float(np.median(ious)) if ious else None,
            "above_08": safe_div(sum(1 for x in ious if x > 0.8), len(ious)) if ious else None,
            "below_03": safe_div(sum(1 for x in ious if x < 0.3), len(ious)) if ious else None,
        }

    # ── 掩码形态变化 ──
    morph_changes = defaultdict(list)
    for _, prompt, row in pairs:
        if row["old_area_ratio"] is not None and row["new_area_ratio"] is not None:
            morph_changes[prompt].append({
                "area_delta": row["new_area_ratio"] - row["old_area_ratio"],
                "cc_delta": (row["new_num_cc"] or 0) - (row["old_num_cc"] or 0),
            })

    per_class_morph = {}
    for cls_name in prompts:
        changes = morph_changes.get(cls_name, [])
        if changes:
            area_deltas = [c["area_delta"] for c in changes]
            cc_deltas = [c["cc_delta"] for c in changes]
            per_class_morph[cls_name] = {
                "mean_area_delta": float(np.mean(area_deltas)),
                "median_area_delta": float(np.median(area_deltas)),
                "mean_cc_delta": float(np.mean(cc_deltas)),
                "fragmented_more": sum(1 for d in cc_deltas if d > 0),
                "fragmented_less": sum(1 for d in cc_deltas if d < 0),
            }

    # ── 对比度分桶下的 IoU ──
    iou_by_contrast = defaultdict(list)
    for _, _, row in pairs:
        if row["iou"] is not None and row["contrast_std"] is not None:
            bucket = contrast_bucket(row["contrast_std"], contrast_bins)
            iou_by_contrast[bucket].append(row["iou"])

    contrast_summary = {}
    for bucket in ["low", "mid", "high"]:
        ious = iou_by_contrast.get(bucket, [])
        contrast_summary[bucket] = {
            "count": len(ious),
            "mean_iou": float(np.mean(ious)) if ious else None,
        }

    # ── 分歧最大样本 Top-N ──
    divergent = [
        row for _, _, row in pairs
        if (row["iou"] is not None and row["iou"] < 0.3) or (not row["hit_agree"])
    ]
    divergent.sort(key=lambda r: (r["iou"] if r["iou"] is not None else -1))  # IoU 小的排前面

    # ── 组装 ──
    report = {
        "summary": {
            "common_pairs": len(pairs),
            "only_in_old": len(old_only),
            "only_in_new": len(new_only),
            "prompts_compared": len(prompts),
            "hit_flip_samples": len(hit_flip_samples),
            "cross_iou_global": iou_dist,
            "per_class_iou": per_class_iou_summary,
            "per_class_hit_agreement": dict(per_class_hit_agree),
            "per_class_morphology": per_class_morph,
            "contrast_buckets": contrast_summary,
        },
        "class_stats": {
            "old": old_stats,
            "new": new_stats,
        },
        "divergent_top200": [
            {
                "image_path": r["image_path"],
                "prompt": r["prompt"],
                "old_hit": r["old_hit"],
                "new_hit": r["new_hit"],
                "old_score": round(r["old_score"], 4),
                "new_score": round(r["new_score"], 4),
                "iou": round(r["iou"], 4) if r["iou"] is not None else None,
                "contrast_std": round(r["contrast_std"], 1) if r["contrast_std"] is not None else None,
            }
            for r in divergent[:200]
        ],
    }
    return report


# ══════════════════════════════════════════════════════════════════════
# 打印
# ══════════════════════════════════════════════════════════════════════


def print_report(report, prompts):
    s = report["summary"]
    cls_stats_old = report["class_stats"]["old"]
    cls_stats_new = report["class_stats"]["new"]

    print()
    print("=" * 80)
    print("  新旧伪标签质量对比报告")
    print("=" * 80)

    # ── 1. 样本量对比 ──
    print(f"\n{'─' * 60}")
    print("  1. 样本量对比")
    print(f"{'─' * 60}")
    print(f"  共同 (image, prompt) 对: {s['common_pairs']}")
    print(f"  仅在旧版: {s['only_in_old']}")
    print(f"  仅在新版: {s['only_in_new']}")
    if s["only_in_old"] or s["only_in_new"]:
        print(f"  ⚠ 两版覆盖范围不一致，请确认 prompt 列表或图片范围是否相同")

    # ── 2. Per-class 统计 ──
    print(f"\n{'─' * 60}")
    print("  2. Per-class: 激活率 & 置信度")
    print(f"{'─' * 60}")
    header = f"  {'Prompt':16s} | {'激活率(old→new)':18s} | {'avg_score(old→new)':22s} | {'median_score(old→new)':24s} | {'score_std(old→new)'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in prompts:
        old_s = cls_stats_old.get(p, {})
        new_s = cls_stats_new.get(p, {})
        ar_old = old_s.get("activation_rate", 0)
        ar_new = new_s.get("activation_rate", 0)
        ar_delta = ar_new - ar_old
        ar_sign = "+" if ar_delta > 0 else ""

        avg_old = old_s.get("avg_score")
        avg_new = new_s.get("avg_score")
        med_old = old_s.get("median_score")
        med_new = new_s.get("median_score")
        std_old = old_s.get("score_std")
        std_new = new_s.get("score_std")

        ar_str = f"{ar_old:.3f}→{ar_new:.3f} ({ar_sign}{ar_delta:+.3f})"
        avg_str = f"{avg_old:.3f}→{avg_new:.3f}" if avg_old is not None and avg_new is not None else "N/A"
        med_str = f"{med_old:.3f}→{med_new:.3f}" if med_old is not None and med_new is not None else "N/A"
        std_str = f"{std_old:.3f}→{std_new:.3f}" if std_old is not None and std_new is not None else "N/A"

        print(f"  {p:16s} | {ar_str:18s} | {avg_str:22s} | {med_str:24s} | {std_str}")

        # 信号提示
        if ar_delta > 0.15:
            print(f"  {'':16s}   ⚠ 激活率大幅上升，可能 prompt 太泛导致 SAM3 乱打")
        elif ar_delta < -0.15:
            print(f"  {'':16s}   ⚠ 激活率大幅下降，可能 prompt 太窄导致漏检增多")

    # ── 3. Cross-IoU 分布 ──
    iou = s["cross_iou_global"]
    if iou:
        print(f"\n{'─' * 60}")
        print("  3. 新老掩码 Cross-IoU 分布（两版都 hit 时）")
        print(f"{'─' * 60}")
        print(f"  可比样本数: {iou['count']}")
        print(f"  mean={iou['mean']:.3f}  median={iou['median']:.3f}  std={iou['std']:.3f}")
        print(f"  p25={iou['p25']:.3f}  p75={iou['p75']:.3f}")
        print(f"  IoU>0.8 (高度一致): {iou['above_08_ratio']:.1%}  → 这些样本大概率两版都对")
        print(f"  IoU<0.3 (严重分歧): {iou['below_03_ratio']:.1%}  → 这些样本必须人工抽查")

        if iou["above_08_ratio"] > 0.70:
            print(f"  ✓ 两版高度一致，prompt 变更相当于措辞微调")
        elif iou["below_03_ratio"] > 0.15:
            print(f"  ⚠ 分歧比例较高，prompt 变更导致了实质性的语义漂移")
    else:
        print(f"\n{'─' * 60}")
        print("  3. Cross-IoU: 无可比样本（两版同时 hit 的对数为 0）")

    # ── 4. Per-class IoU ──
    print(f"\n{'─' * 60}")
    print("  4. Per-class Cross-IoU")
    print(f"{'─' * 60}")
    per_iou = s["per_class_iou"]
    for p in prompts:
        info = per_iou.get(p, {})
        if info.get("iou_mean") is not None:
            print(
                f"  {p:16s}  IoU={info['iou_mean']:.3f} (median={info['iou_median']:.3f})  "
                f">0.8={info['above_08']:.1%}  <0.3={info['below_03']:.1%}  n={info['iou_count']}"
            )
        else:
            print(f"  {p:16s}  无共同 hit 样本")

    # ── 5. Hit 一致性 ──
    print(f"\n{'─' * 60}")
    print("  5. Hit/未命中 一致性")
    print(f"{'─' * 60}")
    hit = s["per_class_hit_agreement"]
    for p in prompts:
        info = hit.get(p, {})
        total = sum(info.values())
        if total > 0:
            print(
                f"  {p:16s}  一致={info['agree']:5d} ({safe_div(info['agree'], total):.1%})  "
                f"仅旧版命中={info['old_only_hit']:4d}  仅新版命中={info['new_only_hit']:4d}  "
                f"均未命中={info['both_miss']:4d}"
            )

    print(f"\n  全局: 命中状态翻转的样本共 {s['hit_flip_samples']} 个")
    if s["hit_flip_samples"] > 0:
        print(f"    → 若翻转数多，说明新 prompt 改变了 SAM3 的 '有没有' 判断，需重点抽查")

    # ── 6. 掩码形态变化 ──
    morph = s["per_class_morphology"]
    if morph:
        print(f"\n{'─' * 60}")
        print("  6. 掩码形态变化 (new - old)")
        print(f"{'─' * 60}")
        for p in prompts:
            m = morph.get(p, {})
            if m:
                frag = f"碎片化↑{m.get('fragmented_more',0)} 碎片化↓{m.get('fragmented_less',0)}"
                print(
                    f"  {p:16s}  area_delta(median)={m.get('median_area_delta', 0):+.4f}  "
                    f"cc_delta(mean)={m.get('mean_cc_delta', 0):+.1f}  {frag}"
                )

    # ── 7. 对比度分桶 ──
    print(f"\n{'─' * 60}")
    print("  7. 按图像对比度分桶的 Cross-IoU")
    print(f"{'─' * 60}")
    for bucket in ["low", "mid", "high"]:
        info = s["contrast_buckets"].get(bucket, {})
        iou_val = info.get("mean_iou")
        if iou_val is not None:
            print(f"  contrast={bucket:4s}  n={info['count']:5d}  mean_IoU={iou_val:.3f}")
        else:
            print(f"  contrast={bucket:4s}  无数据")
    print(f"  → 如果低对比度桶 IoU 明显低于高对比度桶，说明伪标签在低质图上更不稳定")

    # ── 8. 分歧最大样本 ──
    divergent = report["divergent_top200"]
    print(f"\n{'─' * 60}")
    print(f"  8. 分歧最大样本 Top-20（共 {len(divergent)}）")
    print(f"{'─' * 60}")
    print(f"  {'Image':50s} | {'Prompt':16s} | old_hit | new_hit | {'IoU':6s} | contrast")
    print("  " + "-" * 108)
    for r in divergent[:20]:
        iou_str = f"{r['iou']:.3f}" if r["iou"] is not None else "N/A"
        contrast_str = f"{r['contrast_std']:.0f}" if r["contrast_std"] is not None else "?"
        print(
            f"  {r['image_path']:50s} | {r['prompt']:16s} | {str(r['old_hit']):7s} | {str(r['new_hit']):7s} | {iou_str:6s} | {contrast_str}"
        )

    # ── 结论提示 ──
    print(f"\n{'─' * 60}")
    print("  9. 综合判断提示")
    print(f"{'─' * 60}")
    hints = []
    if iou and iou.get("above_08_ratio", 0) > 0.70:
        hints.append("✓ 新老伪标签高度一致，新 prompt 等价于措辞微调，可直接替换")
    if iou and iou.get("below_03_ratio", 0) > 0.15:
        hints.append("⚠ IoU<0.3 占比偏高，需人工抽查分歧样本确认哪个版本更准")
    if s["hit_flip_samples"] > s["common_pairs"] * 0.10:
        hints.append("⚠ 超过 10% 样本命中状态翻转，新 prompt 语义可能漂移了")
    for p in prompts:
        old_ar = cls_stats_old.get(p, {}).get("activation_rate", 0)
        new_ar = cls_stats_new.get(p, {}).get("activation_rate", 0)
        if abs(new_ar - old_ar) > 0.20:
            hints.append(f"⚠ {p}: 激活率变化 >20%，请抽查该类的命中变化是否合理")

    if not hints:
        hints.append("无明显异常，但仍建议抽查分歧 Top-20 样本做最终确认")

    for h in hints:
        print(f"  {h}")

    print()
    print("=" * 80)
    print("  注意: 本报告只描述差异，不判断真伪。最终质量判定需人工抽查分歧样本。")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="新旧伪标签质量对比 —— 告诉你两版差异在哪、有多大，不做真伪判定",
    )
    parser.add_argument("--old", default=OLD_PRED_JSON, help=f"旧版伪标签 JSON（默认: {OLD_PRED_JSON}）")
    parser.add_argument("--new", default=NEW_PRED_JSON, help=f"新版伪标签 JSON（默认: {NEW_PRED_JSON}）")
    parser.add_argument("--image-root", default=IMAGE_ROOT, help=f"图片根目录（默认: {IMAGE_ROOT}）")
    parser.add_argument("--out-dir", default=OUT_DIR, help=f"输出目录（默认: {OUT_DIR}）")
    parser.add_argument("--prompts", nargs="*", default=PROMPTS, help="指定要对比的 prompt（默认自动收集两版所有 prompt）")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    image_root = Path(args.image_root)

    if not old_path.exists():
        print(f"[ERROR] 旧版文件不存在: {old_path}"); sys.exit(1)
    if not new_path.exists():
        print(f"[ERROR] 新版文件不存在: {new_path}"); sys.exit(1)

    print(f"旧版: {old_path}")
    print(f"新版: {new_path}")
    print(f"图片根目录: {image_root}")

    # 加载
    print("\n加载旧版伪标签...")
    old_data = load_pseudo_label_json(old_path)
    print(f"  旧版: {len(old_data)} 张图")

    print("加载新版伪标签...")
    new_data = load_pseudo_label_json(new_path)
    print(f"  新版: {len(new_data)} 张图")

    prompts = args.prompts if args.prompts else collect_all_prompts(old_data, new_data)
    print(f"对比 prompt: {prompts}")

    # 缓存目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = old_path.parent.parent / "pseudo_label_compare"
    mask_cache = out_dir / ".mask_cache"
    mask_cache.mkdir(parents=True, exist_ok=True)

    # Per-class 统计
    print("\n计算 per-class 统计...")
    old_stats = compute_per_class_stats(old_data, prompts)
    new_stats = compute_per_class_stats(new_data, prompts)

    # 逐样本 pairwise 对比
    print("逐样本对比中...")
    pairs, old_only, new_only = build_pairwise(
        old_data, new_data, image_root, prompts, mask_cache,
    )
    print(f"  可比对: {len(pairs)}  仅旧版: {len(old_only)}  仅新版: {len(new_only)}")

    # 对比度分桶
    contrast_bins = build_contrast_bins(pairs)
    if contrast_bins:
        print(f"  对比度分桶: low≤{contrast_bins[0]:.1f}  mid≤{contrast_bins[1]:.1f}  high>{contrast_bins[1]:.1f}")

    # 汇总
    print("\n汇总...")
    report = summarize(pairs, old_stats, new_stats, prompts, contrast_bins, old_only, new_only)

    # 打印
    print_report(report, prompts)

    # 保存
    if args.out_dir or True:  # 始终保存
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "compare_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: {report_path}")

        # 单独导出分歧样本列表，方便人工抽查
        divergent_txt = out_dir / "divergent_samples.txt"
        with open(divergent_txt, "w", encoding="utf-8") as f:
            f.write("# 新老伪标签分歧最大样本（建议人工抽查）\n")
            f.write("# 格式: image_path | prompt | old_hit | new_hit | IoU | contrast_std\n")
            for row in report["divergent_top200"]:
                f.write(
                    f"{row['image_path']} | {row['prompt']} | "
                    f"{row['old_hit']} | {row['new_hit']} | "
                    f"{row['iou'] if row['iou'] is not None else 'N/A'} | "
                    f"{row['contrast_std'] if row['contrast_std'] is not None else '?'}\n"
                )
        print(f"分歧样本列表已保存: {divergent_txt} ({len(report['divergent_top200'])} 条)")


if __name__ == "__main__":
    main()
