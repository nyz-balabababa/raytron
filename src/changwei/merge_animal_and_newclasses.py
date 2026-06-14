#!/usr/bin/env python3
"""
合成最终训练 label：animal 映射版为主 + 新增类映射版补充新类别。

用法示例：

  # 填好顶部「用户配置区」后直接运行
  python src/changwei/merge_animal_and_newclasses.py

  # 也可以用命令行覆盖
  python src/changwei/merge_animal_and_newclasses.py \
      --animal-label data/animal_mapped.json \
      --newclass-label data/newclass_mapped.json \
      --output data/final_label.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# 路径
ANIMAL_LABEL = ROOT / "test" / "clean_rare" / "animal-val" / "merged_base_plus_rare_animal.json"
"""animal 映射版 label（主 base）"""

NEWCLASS_LABEL = ROOT / "test" / "clean_rare" / "xiyou-val" / "merged_label_cleaned.json"
"""新增类映射版 label"""

OUTPUT = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
"""合并后的最终 label"""

SUMMARY_OUTPUT = ROOT / "test" / "clean_rare" / "label" / "val_merge_summary.json"
"""统计 summary，None=不保存"""

# 从新增类映射版拷贝的类别
COPY_FROM_NEWCLASS_LABELS = {
    "car",
    "computer",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
}

# 输出控制
VERBOSE = True
# ── 用户配置区结束 ──────────────────────────────────────────────────────


def load_label_json(path: Path) -> List[Dict[str, Any]]:
    """加载 label JSON，兼容 list / entries / annotations / records."""
    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("annotations") or data.get("records") or [data]
        if isinstance(entries, list):
            return entries
    print(f"[ERROR] 无法识别格式: {path}", file=sys.stderr)
    sys.exit(1)


def run(args: argparse.Namespace) -> None:
    t0 = time.time()

    # 1. 加载
    animal_entries = load_label_json(args.animal_label)
    newclass_entries = load_label_json(args.newclass_label)
    print(f"[INFO] animal label:  {args.animal_label}  ({len(animal_entries)} entries)")
    print(f"[INFO] newclass label: {args.newclass_label}  ({len(newclass_entries)} entries)")

    # 2. 建索引
    animal_index: Dict[str, Dict[str, Any]] = {}
    for e in animal_entries:
        ip = str(e.get("image_path", "")).replace("\\", "/")
        animal_index[ip] = e

    newclass_index: Dict[str, Dict[str, Any]] = {}
    for e in newclass_entries:
        ip = str(e.get("image_path", "")).replace("\\", "/")
        newclass_index[ip] = e

    # 3. 统计初始化
    summary: Dict[str, Any] = {
        "animal_label": str(args.animal_label),
        "newclass_label": str(args.newclass_label),
        "output": str(args.output),
        "animal_entries": len(animal_entries),
        "newclass_entries": len(newclass_entries),
        "copied_from_newclass_total": 0,
        "copied_from_newclass_per_class": defaultdict(int),
        "newclass_extra_entries_added": 0,
        "animal_preserved_count": 0,
        "blocked_labels_from_newclass": defaultdict(int),
    }

    # 4. 以 animal 为主，逐个 entry 合并
    output: List[Dict[str, Any]] = []
    for ip, animal_entry in animal_index.items():
        out_entry = copy.deepcopy(animal_entry)
        out_prompts = out_entry.setdefault("prompts", {})

        # 记录 animal 保留
        if "animal" in out_prompts:
            summary["animal_preserved_count"] += 1

        # 从 newclass 拷贝允许的类别
        new_entry = newclass_index.get(ip)
        if new_entry is not None:
            new_prompts = new_entry.get("prompts", {})
            if isinstance(new_prompts, dict):
                for cls in COPY_FROM_NEWCLASS_LABELS:
                    info = new_prompts.get(cls)
                    if isinstance(info, dict) and info.get("hit") is True and info.get("rle") is not None:
                        out_prompts[cls] = copy.deepcopy(info)
                        summary["copied_from_newclass_total"] += 1
                        summary["copied_from_newclass_per_class"][cls] += 1

                # 统计被阻止的类别
                for cls in ("animal", "person", "tree", "building"):
                    if cls in new_prompts:
                        summary["blocked_labels_from_newclass"][cls] += 1

        output.append(out_entry)

    # 5. newclass 中有但 animal 中没有的图片
    for ip, new_entry in newclass_index.items():
        if ip in animal_index:
            continue
        new_prompts = new_entry.get("prompts", {})
        if not isinstance(new_prompts, dict):
            continue
        extra_prompts = {}
        for cls in COPY_FROM_NEWCLASS_LABELS:
            info = new_prompts.get(cls)
            if isinstance(info, dict) and info.get("hit") is True and info.get("rle") is not None:
                extra_prompts[cls] = copy.deepcopy(info)
                summary["copied_from_newclass_total"] += 1
                summary["copied_from_newclass_per_class"][cls] += 1
        if extra_prompts:
            output.append({"image_path": ip, "prompts": extra_prompts})
            summary["newclass_extra_entries_added"] += 1

    # 6. 按 image_path 排序
    output.sort(key=lambda e: e["image_path"])
    summary["output_entries"] = len(output)
    summary["copied_from_newclass_per_class"] = dict(summary["copied_from_newclass_per_class"])
    summary["blocked_labels_from_newclass"] = dict(summary["blocked_labels_from_newclass"])

    # 7. 写输出
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {out_path}  ({len(output)} entries)")

    # 8. Summary
    print(f"\n{'='*60}")
    print(f"合并完成  ({time.time() - t0:.1f}s)")
    print(f"  animal entries:              {summary['animal_entries']}")
    print(f"  newclass entries:            {summary['newclass_entries']}")
    print(f"  output entries:              {summary['output_entries']}")
    print(f"  animal preserved:            {summary['animal_preserved_count']}")
    print(f"  copied total:                {summary['copied_from_newclass_total']}")
    print(f"  newclass extra entries:      {summary['newclass_extra_entries_added']}")
    print(f"  per class:")
    for cls, n in sorted(summary["copied_from_newclass_per_class"].items()):
        print(f"    {cls}: {n}")
    blocked = summary["blocked_labels_from_newclass"]
    if any(blocked.values()):
        print(f"  blocked (not copied):")
        for cls, n in sorted(blocked.items()):
            if n > 0:
                print(f"    {cls}: {n}")

    if args.summary:
        sp = Path(args.summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] summary: {sp}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合成 animal 映射版 + 新增类映射版 → 最终训练 label")
    p.add_argument("--animal-label", type=Path, default=ANIMAL_LABEL)
    p.add_argument("--newclass-label", type=Path, default=NEWCLASS_LABEL)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--summary", type=Path, default=SUMMARY_OUTPUT)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
