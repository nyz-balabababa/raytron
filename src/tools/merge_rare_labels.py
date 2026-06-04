#!/usr/bin/env python3
"""
合并稀有类伪标签 → 6 类方案 (5 common + equipment)

输入:
  - test/prompt_test_output/train_tasks/pred_train_tasks.json  (全量 6 类, 39K)
  - test/prompt_test_output/train_d5&7_tasks/pred_train_d5&7_tasks.json  (data5&7 5 类, 8K)

合并规则:
  equipment = trash can OR cable connector OR computer  (RLE 并集)
  丢弃: fire hydrant, circuit board, 原 computer 类

输出:
  - test/prompt_test_output/train_tasks/pred_train_tasks_v2.json  (6 类: person/car/building/tree/animal/equipment)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
ORIG = ROOT / "test" / "prompt_test_output" / "train_tasks" / "new_train.json"
NEW = ROOT / "test" / "prompt_test_output" / "train_d5&7_tasks" / "pred_train_d5&7_tasks.json"
OUT = ROOT / "test" / "prompt_test_output" / "train_tasks" / "pred_train_tasks_v2.json"

COMMON_CLASSES = ["person", "car", "building", "tree", "animal"]
MERGE_SOURCES = ["trash can", "cable connector", "computer"]  # → equipment
SKIP_CLASSES = ["fire hydrant", "circuit board"]


def rle_to_mask(rle: dict) -> np.ndarray:
    try:
        from pycocotools import mask as maskUtils
        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return maskUtils.decode(rle_cp).astype(np.uint8)
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
    pos = val = 0
    for run_len in counts:
        if val == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def mask_to_rle(mask: np.ndarray) -> dict:
    try:
        from pycocotools import mask as maskUtils
        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
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


def merge_equipment(prompts: dict) -> dict:
    """合并 trash can + cable connector + computer → equipment (RLE OR)"""
    masks = []
    best_score = 0.0
    best_instances = 0

    for src_name in MERGE_SOURCES:
        v = prompts.get(src_name, {})
        if v.get("hit") and v.get("rle"):
            masks.append(rle_to_mask(v["rle"]))
            best_score = max(best_score, v.get("score", 0))
            best_instances = max(best_instances, v.get("instances") or 0)

    if not masks:
        return {"hit": False}

    merged = masks[0].copy()
    for m in masks[1:]:
        merged = np.logical_or(merged, m).astype(np.uint8)

    return {
        "hit": True,
        "score": best_score,
        "instances": best_instances,
        "rle": mask_to_rle(merged),
    }


def main():
    print("加载原伪标签...")
    with open(ORIG, encoding="utf-8") as f:
        orig = json.load(f)
    print(f"  原: {len(orig)} 条 ({COMMON_CLASSES} + computer)")

    print("加载新伪标签 (data5&7)...")
    with open(NEW, encoding="utf-8") as f:
        new = json.load(f)
    print(f"  新: {len(new)} 条 ({MERGE_SOURCES + SKIP_CLASSES})")

    # 新数据按 image_path 索引
    new_by_path = {}
    for rec in tqdm(new, desc="构建新标签索引", unit="img"):
        new_by_path[rec["image_path"]] = rec["prompts"]

    merged = []
    equip_hit = 0
    equip_total = 0

    for rec in tqdm(orig, desc="合并 equipment 标签", unit="img"):
        img_path = rec["image_path"]
        orig_prompts = rec["prompts"]
        new_prompts = new_by_path.get(img_path, {})

        # 保留 5 个常见类
        out_prompts = {}
        for cls_name in COMMON_CLASSES:
            if cls_name in orig_prompts:
                out_prompts[cls_name] = orig_prompts[cls_name]

        # equipment: 优先用新数据合并，回退到原 computer
        equip = merge_equipment(new_prompts)
        equip_total += 1
        if equip["hit"]:
            equip_hit += 1
        else:
            # 新数据没命中 → 回退用原 computer 伪标签
            comp = orig_prompts.get("computer", {})
            if comp.get("hit") and comp.get("rle"):
                equip = comp
                equip_hit += 1
        out_prompts["equipment"] = equip

        merged.append({"image_path": img_path, "prompts": out_prompts})

    # 统计
    print(f"\n合并完成: {len(merged)} 条")
    print(f"  类别: {COMMON_CLASSES + ['equipment']}")
    print()

    stats = defaultdict(lambda: {"hit": 0, "total": 0, "scores": []})
    for rec in merged:
        for prompt, v in rec["prompts"].items():
            stats[prompt]["total"] += 1
            if v.get("hit"):
                stats[prompt]["hit"] += 1
                stats[prompt]["scores"].append(v.get("score", 0))

    print(f"{'prompt':15s} {'命中':>6s} {'总数':>6s} {'激活率':>7s} {'avg_score':>8s}")
    for prompt in sorted(stats.keys()):
        s = stats[prompt]
        rate = s["hit"] / s["total"] * 100
        avg_s = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        print(f"{prompt:15s} {s['hit']:6d} {s['total']:6d} {rate:6.1f}% {avg_s:8.3f}")

    # 写入
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n输出: {OUT}")
    print(f"文件大小: {OUT.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
