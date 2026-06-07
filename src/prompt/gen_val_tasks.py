#!/usr/bin/env python3
"""从图片列表生成任务 JSON：每张图 × 每个 prompt = 一条任务。

保留现有任务格式：
[
  {"ann_id": 1, "image_path": "...", "text_prompt": "..."},
  ...
]

相比旧版补充：
1. CLI 参数，不再把路径和 prompt 硬编码进脚本
2. 路径规范化、去重、缺失文件检查
3. 可选接入坏图 skip list，和前期数据清洗结果打通
4. 输出 sidecar metadata，便于核对数据分布
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
TASK_IMAGE_LIST = ROOT / "test" / "val_list.txt"
TASK_OUTPUT_JSON = ROOT / "test" / "json" / "val_tasks1.json"
TASK_IMAGE_ROOT = ROOT
TASK_PROMPTS = ["person", "car", "building", "tree", "animal"]
TASK_SKIP_LIST = ROOT / "noisy_data" / "unified_denylist.txt"
TASK_ALLOW_MISSING = False
TASK_ANN_ID_START = 1
# ──────────────────────────────────────────────────────────────────────


def normalize_path(path_str: str) -> str:
    return path_str.strip().replace("\\", "/")


def load_lines(txt_path: Path) -> List[str]:
    with open(txt_path, encoding="utf-8") as f:
        return [normalize_path(line) for line in f if line.strip()]


def load_skip_set(skip_list_path: Path | None) -> set[str]:
    if skip_list_path is None or not skip_list_path.exists():
        return set()
    return set(load_lines(skip_list_path))


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def infer_source_folder(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return "unknown"
    if parts[0] == "test" and len(parts) > 1:
        return parts[1]
    return parts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从图片列表生成任务 JSON")
    parser.add_argument(
        "--list",
        type=Path,
        default=TASK_IMAGE_LIST,
        help="图片列表 txt，每行一个相对路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK_OUTPUT_JSON,
        help="输出任务 JSON 路径",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=TASK_IMAGE_ROOT,
        help="图片根目录，用于校验文件是否存在",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=TASK_PROMPTS,
        help="prompt 列表，例如 --prompts person car building",
    )
    parser.add_argument(
        "--skip-list",
        type=str,
        default=str(TASK_SKIP_LIST) if TASK_SKIP_LIST else "",
        help='坏图列表 txt；传空字符串 "" 表示不启用',
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=TASK_ALLOW_MISSING,
        help="允许图片文件不存在，只做路径展开不报错",
    )
    parser.add_argument(
        "--ann-id-start",
        type=int,
        default=TASK_ANN_ID_START,
        help="ann_id 起始值，默认 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths_raw = load_lines(args.list)
    image_paths = dedupe_preserve_order(image_paths_raw)
    skip_list_path = Path(args.skip_list) if args.skip_list else None
    skip_set = load_skip_set(skip_list_path)

    prompts = dedupe_preserve_order([prompt.strip() for prompt in args.prompts if prompt.strip()])
    if not prompts:
        raise ValueError("prompts 不能为空")

    kept_images = []
    skipped_missing = []
    skipped_denylist = []

    for rel_path in image_paths:
        if rel_path in skip_set:
            skipped_denylist.append(rel_path)
            continue

        abs_path = args.image_root / rel_path
        if not args.allow_missing and not abs_path.exists():
            skipped_missing.append(rel_path)
            continue

        kept_images.append(rel_path)

    if not kept_images:
        raise RuntimeError("没有可用图片。请检查列表、skip list 和 image_root")

    tasks = []
    ann_id = args.ann_id_start
    for img in kept_images:
        for prompt in prompts:
            tasks.append(
                {
                    "ann_id": ann_id,
                    "image_path": img,
                    "text_prompt": prompt,
                }
            )
            ann_id += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    folder_counts = Counter(infer_source_folder(path) for path in kept_images)
    meta = {
        "list_path": str(args.list),
        "output_path": str(args.output),
        "image_root": str(args.image_root),
        "prompts": prompts,
        "duplicates_removed": len(image_paths_raw) - len(image_paths),
        "unique_images_input": len(image_paths),
        "unique_images_kept": len(kept_images),
        "tasks_total": len(tasks),
        "skipped_denylist": len(skipped_denylist),
        "skipped_missing": len(skipped_missing),
        "folder_counts": dict(sorted(folder_counts.items())),
        "skip_list": str(skip_list_path) if skip_list_path else None,
    }

    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"输入图片数(去重后): {len(image_paths)}")
    print(f"保留图片数: {len(kept_images)}")
    print(f"prompt 数: {len(prompts)} -> {prompts}")
    print(f"任务总数: {len(tasks)}")
    print(f"skip denylist: {len(skipped_denylist)}")
    print(f"skip missing:  {len(skipped_missing)}")
    print(f"输出: {args.output}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
