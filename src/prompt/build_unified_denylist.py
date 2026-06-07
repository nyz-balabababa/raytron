#!/usr/bin/env python3
"""合成统一坏图名单。

输入来源：
1. test/clean_outliers.csv
   - 按历史文档规则，只取 is_color 或 is_text_overlay 为 True 的图片
2. noisy_data/
   - 视作人工 denylist，对应路径映射回 test/
3. 伪标签质量分析输出
   - 支持 analyze_prompt_predictions.py 产出的 JSON
   - 读取 all_prompts_wrong_images[].image_path

输出：
- noisy_data/unified_denylist.txt
- noisy_data/unified_denylist_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[2]

# ── 用户配置区 ────────────────────────────────────────────────────────
CLEAN_OUTLIERS_CSV = ROOT / "test" / "clean_outliers.csv"
MANUAL_NOISY_DIR = ROOT / "noisy_data"
ANALYSIS_JSONS: list[Path] = []
OUTPUT_TXT = ROOT / "noisy_data" / "unified_denylist.txt"
OUTPUT_META = ROOT / "noisy_data" / "unified_denylist_meta.json"
# ─────────────────────────────────────────────────────────────────────


def normalize_path(path_str: str) -> str:
    return path_str.strip().replace("\\", "/")


def load_clean_outliers(csv_path: Path) -> Set[str]:
    items: Set[str] = set()
    if not csv_path.exists():
        return items

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = normalize_path(row.get("path", ""))
            if not path:
                continue
            is_color = str(row.get("is_color", "")).lower() == "true"
            is_text_overlay = str(row.get("is_text_overlay", "")).lower() == "true"
            if is_color or is_text_overlay:
                items.add(path)
    return items


def load_manual_noisy(noisy_dir: Path) -> Set[str]:
    items: Set[str] = set()
    if not noisy_dir.exists():
        return items

    for file_path in noisy_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(noisy_dir)
        items.add(normalize_path(str(Path("test") / rel)))
    return items


def load_analysis_bad_images(json_paths: Iterable[Path]) -> Set[str]:
    items: Set[str] = set()
    for json_path in json_paths:
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        bad_list = data.get("all_prompts_wrong_images", [])
        for entry in bad_list:
            image_path = normalize_path(str(entry.get("image_path", "")))
            if image_path:
                items.add(image_path)
    return items


def infer_source_folder(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return "unknown"
    if parts[0] == "test" and len(parts) > 1:
        return parts[1]
    return parts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合成统一坏图 denylist")
    parser.add_argument("--clean-outliers-csv", type=Path, default=CLEAN_OUTLIERS_CSV)
    parser.add_argument("--manual-noisy-dir", type=Path, default=MANUAL_NOISY_DIR)
    parser.add_argument(
        "--analysis-json",
        nargs="*",
        type=Path,
        default=ANALYSIS_JSONS,
        help="analyze_prompt_predictions.py 输出的 JSON，可传多个",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_TXT)
    parser.add_argument("--meta-output", type=Path, default=OUTPUT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean_items = load_clean_outliers(args.clean_outliers_csv)
    manual_items = load_manual_noisy(args.manual_noisy_dir)
    analysis_items = load_analysis_bad_images(args.analysis_json)

    merged = sorted(clean_items | manual_items | analysis_items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(item + "\n")

    folder_counts = Counter(infer_source_folder(path) for path in merged)
    meta = {
        "clean_outliers_csv": str(args.clean_outliers_csv),
        "manual_noisy_dir": str(args.manual_noisy_dir),
        "analysis_jsons": [str(p) for p in args.analysis_json],
        "counts": {
            "clean_outliers_items": len(clean_items),
            "manual_noisy_items": len(manual_items),
            "analysis_items": len(analysis_items),
            "merged_total": len(merged),
        },
        "folder_counts": dict(sorted(folder_counts.items())),
    }
    with open(args.meta_output, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"clean_outliers: {len(clean_items)}")
    print(f"manual_noisy:   {len(manual_items)}")
    print(f"analysis_bad:   {len(analysis_items)}")
    print(f"merged_total:   {len(merged)}")
    print(f"output:         {args.output}")
    print(f"meta:           {args.meta_output}")


if __name__ == "__main__":
    main()
