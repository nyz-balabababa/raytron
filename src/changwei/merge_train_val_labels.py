#!/usr/bin/env python3
"""
把训练集 label 和验证集 label 合并成全集训练用的 label。
只做 JSON 层面合并，不解码 RLE，不改 mask，不改 prompt。

用法示例：

  # 填好顶部「用户配置区」后直接运行
  python src/changwei/merge_train_val_labels.py

  # 也可以用命令行覆盖
  python src/changwei/merge_train_val_labels.py ^
      --train-label data/train_label.json ^
      --val-label data/val_label.json ^
      --output data/trainval_label.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

TRAIN_LABEL = ROOT / "test" / "clean_rare" / "label" / "train_label.json"  # 训练集最终 label
VAL_LABEL = ROOT / "test" / "clean_rare" / "label" / "val_label.json"  # 验证集最终 label
OUTPUT = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"  # 全集训练 label
SUMMARY_OUTPUT = None  # 合并统计，None=不保存
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
    # 1. 加载
    train_entries = load_label_json(args.train_label)
    val_entries = load_label_json(args.val_label)
    print(f"train entries: {len(train_entries)}")
    print(f"val entries:   {len(val_entries)}")

    # 2. 合并 + 去重检查
    seen: set = set()
    output: List[Dict[str, Any]] = []
    duplicate_paths: List[str] = []

    for e in train_entries:
        ip = str(e.get("image_path", "")).replace("\\", "/")
        e["image_path"] = ip
        output.append(e)
        seen.add(ip)

    for e in val_entries:
        ip = str(e.get("image_path", "")).replace("\\", "/")
        if ip in seen:
            duplicate_paths.append(ip)
            continue
        e["image_path"] = ip
        output.append(e)
        seen.add(ip)

    duplicate_count = len(duplicate_paths)
    print(f"output entries: {len(output)}")
    if duplicate_count > 0:
        print(f"⚠ 发现 {duplicate_count} 条重复 image_path（已保留 train 版）")
        for ip in duplicate_paths[:20]:
            print(f"  {ip}")

    # 3. 统计
    per_class_prompt_count: Dict[str, int] = defaultdict(int)
    per_class_hit_true_count: Dict[str, int] = defaultdict(int)
    unknown_prompts: Dict[str, int] = defaultdict(int)

    known = {
        "person", "tree", "building", "animal", "car", "computer",
        "trash can", "window", "door", "fence", "pole_light", "motorcycle",
    }
    for e in output:
        prompts = e.get("prompts", {})
        if not isinstance(prompts, dict):
            continue
        for cls, info in prompts.items():
            per_class_prompt_count[cls] += 1
            if isinstance(info, dict) and info.get("hit") is True:
                per_class_hit_true_count[cls] += 1
            if cls not in known:
                unknown_prompts[cls] += 1

    print(f"\nper class hit=true count:")
    for cls in sorted(known):
        n = per_class_hit_true_count.get(cls, 0)
        print(f"  {cls:20s} {n}")
    if unknown_prompts:
        print(f"\nunknown prompts: {dict(unknown_prompts)}")

    # 4. 写输出
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OUTPUT] {out_path}")

    # 5. Summary
    summary = {
        "train_label": str(args.train_label),
        "val_label": str(args.val_label),
        "output": str(args.output),
        "train_entries": len(train_entries),
        "val_entries": len(val_entries),
        "output_entries": len(output),
        "duplicate_count": duplicate_count,
        "duplicate_examples": duplicate_paths[:20],
        "per_class_prompt_count": dict(per_class_prompt_count),
        "per_class_hit_true_count": dict(per_class_hit_true_count),
    }
    if unknown_prompts:
        summary["unknown_prompts"] = dict(unknown_prompts)

    if args.summary:
        sp = Path(args.summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] summary: {sp}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并 train + val label → 全集训练 label")
    p.add_argument("--train-label", type=Path, default=TRAIN_LABEL)
    p.add_argument("--val-label", type=Path, default=VAL_LABEL)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--summary", type=Path, default=SUMMARY_OUTPUT)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
