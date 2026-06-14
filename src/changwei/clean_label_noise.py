#!/usr/bin/env python3
"""
清洗伪标签 JSON 中的噪声。当前处理 building / tree 小连通域过滤。
不做 rare 映射，不做类别合并，不做训练。

用法示例：

  # 填好顶部「用户配置区」后直接运行
  python src/changwei/clean_label_noise.py

  # 也可以用命令行覆盖
  python src/changwei/clean_label_noise.py \
      --input data/label.json \
      --output data/label_cleaned.json \
      --building-min-area 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
"""项目根目录"""

# 路径
INPUT = ROOT / "test" / "clean_rare" / "xiyou-val" / "merged_label.json"
"""输入 label JSON"""

OUTPUT = ROOT / "test" / "clean_rare" / "xiyou-val" / "merged_label_cleaned.json"
"""清洗后的输出 JSON"""

SUMMARY_OUTPUT = ROOT / "test" / "clean_rare" / "xiyou-val" / "clean_summary.json"
"""清洗统计 JSON，None=不保存"""

# 清洗规则：每个 class 的最小连通域面积（低于此值的连通域删除）
CLASS_MIN_COMPONENT_AREA = {
    "building": 350,
    "tree": 200,
}

# 输出控制
BUILDING_MIN_AREA = 350
"""building 小连通域最小面积阈值（CLI 可覆盖）"""

TREE_MIN_AREA = 200
"""tree 小连通域最小面积阈值（CLI 可覆盖）"""

VERBOSE = True

USE_GPU = True
# ── 用户配置区结束 ──────────────────────────────────────────────────────

import argparse
import copy
import json
import sys
import time
from collections import defaultdict

import numpy as np

# ---- tqdm ----
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    HAS_TQDM = False


def _progress(iterable, desc: str = "", unit: str = "it", **kwargs):
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, unit=unit, **kwargs)


# ---- pycocotools ----
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    mask_utils = None
    HAS_PYCOCOTOOLS = False

# ---- cv2 ----
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# ---- GPU ----
try:
    import torch
    HAS_TORCH = True
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False
    TORCH_DEVICE = None

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# ── RLE 工具 ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def decode_rle(rle: Dict[str, Any]) -> Optional[np.ndarray]:
    """解码 COCO RLE -> 二值 mask (H, W) uint8."""
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError("Please install pycocotools to decode/encode RLE.")
    if rle is None:
        return None
    size = rle.get("size")
    counts = rle.get("counts")
    if size is None or counts is None:
        return None
    h, w = int(size[0]), int(size[1])
    try:
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        coco_rle = {"size": [h, w], "counts": counts}
        mask = mask_utils.decode(coco_rle)
        return mask.astype(np.uint8)
    except Exception:
        return None


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """编码二值 mask -> COCO RLE dict."""
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError("Please install pycocotools to decode/encode RLE.")
    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_f)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


# ═══════════════════════════════════════════════════════════════════════════════
# ── 数据加载 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def load_label_json(path: Path) -> List[Dict[str, Any]]:
    """加载 label JSON，自动识别 list / entries / annotations / records 格式."""
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
    print(f"[ERROR] 无法识别 JSON 格式: {path}", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 小连通域过滤 ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def filter_small_components(
    mask: np.ndarray, min_area: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    删除面积 < min_area 的连通域。
    返回 (cleaned_mask, stats_dict)。
    """
    if not HAS_CV2:
        raise RuntimeError("Please install opencv-python.")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    area_before = int(mask.sum())
    removed_components = 0
    removed_area = 0
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        comp_area = int(stats[i, cv2.CC_STAT_AREA])
        if comp_area >= min_area:
            cleaned[labels == i] = 1
        else:
            removed_components += 1
            removed_area += comp_area

    area_after = int(cleaned.sum())
    comp_count_before = num_labels - 1
    comp_count_after = comp_count_before - removed_components

    return cleaned, {
        "component_count_before": comp_count_before,
        "component_count_after": comp_count_after,
        "removed_components": removed_components,
        "removed_area": removed_area,
        "area_before": area_before,
        "area_after": area_after,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── 通用清洗函数 ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def clean_one_class_mask(
    entry: Dict[str, Any],
    class_name: str,
    min_area: int,
    summary: Dict[str, Any],
    change_examples: List[Dict[str, Any]],
) -> None:
    """清洗 entry 中指定 class 的小连通域噪声。原地修改 entry."""
    prompts = entry.get("prompts", {})
    if not isinstance(prompts, dict):
        return

    info = prompts.get(class_name)
    if not isinstance(info, dict):
        return
    if info.get("hit") is not True:
        return

    rle = info.get("rle")
    if rle is None:
        return

    summary[f"{class_name}_hit_before"] = summary.get(f"{class_name}_hit_before", 0) + 1

    # 解码
    try:
        mask = decode_rle(rle)
    except Exception:
        summary["decode_fail"] = summary.get("decode_fail", 0) + 1
        return
    if mask is None:
        summary["decode_fail"] = summary.get("decode_fail", 0) + 1
        return

    # 过滤小连通域
    cleaned_mask, stats = filter_small_components(mask, min_area)

    if stats["removed_components"] > 0:
        summary[f"{class_name}_masks_cleaned"] = summary.get(f"{class_name}_masks_cleaned", 0) + 1
        summary[f"{class_name}_removed_components_total"] = summary.get(f"{class_name}_removed_components_total", 0) + stats["removed_components"]
        summary[f"{class_name}_removed_area_total"] = summary.get(f"{class_name}_removed_area_total", 0) + stats["removed_area"]
        if len(change_examples) < 30:
            change_examples.append({
                "image_path": entry.get("image_path", ""),
                "class_name": class_name,
                **stats,
            })
    else:
        summary[f"{class_name}_masks_unchanged"] = summary.get(f"{class_name}_masks_unchanged", 0) + 1

    if stats["area_after"] > 0:
        try:
            info["rle"] = encode_rle(cleaned_mask)
        except Exception:
            summary["encode_fail"] = summary.get("encode_fail", 0) + 1
            return
        info["cleaned_by"] = "remove_small_components"
        info["min_component_area"] = min_area
        summary[f"{class_name}_hit_after"] = summary.get(f"{class_name}_hit_after", 0) + 1
    else:
        info["hit"] = False
        info["rle"] = None
        info["cleaned_by"] = "remove_small_components"
        info["disabled_reason"] = "empty_after_small_component_filter"
        summary[f"{class_name}_disabled_empty"] = summary.get(f"{class_name}_disabled_empty", 0) + 1


# ═══════════════════════════════════════════════════════════════════════════════
# ── 主流程 ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def run(args: argparse.Namespace) -> None:
    global BUILDING_MIN_AREA, VERBOSE

    # CLI 覆盖 building 阈值
    if hasattr(args, "building_min_area"):
        CLASS_MIN_COMPONENT_AREA["building"] = args.building_min_area

    # GPU 状态
    backend = "cpu"
    if HAS_CUPY and USE_GPU:
        backend = "cupy"
    elif HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda" and USE_GPU:
        backend = "torch"
    print(f"[SYS] GPU: {backend}  pycocotools: {'✓' if HAS_PYCOCOTOOLS else '✗'}  "
          f"cv2: {'✓' if HAS_CV2 else '✗'}  tqdm: {'✓' if HAS_TQDM else '✗'}")

    t0 = time.time()

    # 1. 加载
    entries = load_label_json(args.input)
    print(f"[INFO] 加载: {args.input}  ({len(entries)} entries)")

    # 2. 深拷贝，只改配置中指定的类
    cleaned = copy.deepcopy(entries)

    # 3. summary 初始化
    summary: Dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "entries_total": len(entries),
        "decode_fail": 0,
        "encode_fail": 0,
    }
    for class_name, min_area in CLASS_MIN_COMPONENT_AREA.items():
        summary[f"{class_name}_min_area"] = min_area
        summary[f"{class_name}_hit_before"] = 0
        summary[f"{class_name}_hit_after"] = 0
        summary[f"{class_name}_masks_cleaned"] = 0
        summary[f"{class_name}_masks_unchanged"] = 0
        summary[f"{class_name}_disabled_empty"] = 0
        summary[f"{class_name}_removed_components_total"] = 0
        summary[f"{class_name}_removed_area_total"] = 0

    change_examples: List[Dict[str, Any]] = []

    # 4. 遍历清洗
    for entry in _progress(cleaned, desc="清洗 label 噪声", unit="entry"):
        prompts = entry.get("prompts", {})
        if not isinstance(prompts, dict):
            continue
        for class_name, min_area in CLASS_MIN_COMPONENT_AREA.items():
            clean_one_class_mask(
                entry=entry,
                class_name=class_name,
                min_area=min_area,
                summary=summary,
                change_examples=change_examples,
            )

    summary["change_examples"] = change_examples

    # 5. 写输出
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {output_path}  ({len(cleaned)} entries)")

    # 6. Summary 终端打印
    print(f"\n{'='*60}")
    print(f"清洗完成  ({time.time() - t0:.1f}s)")
    print(f"  entries_total:  {summary['entries_total']}")
    for class_name in CLASS_MIN_COMPONENT_AREA:
        print(f"\n  [class {class_name}]")
        print(f"    hit_before:                {summary.get(f'{class_name}_hit_before', 0)}")
        print(f"    hit_after:                 {summary.get(f'{class_name}_hit_after', 0)}")
        print(f"    masks_cleaned:             {summary.get(f'{class_name}_masks_cleaned', 0)}")
        print(f"    masks_unchanged:           {summary.get(f'{class_name}_masks_unchanged', 0)}")
        print(f"    disabled_empty:            {summary.get(f'{class_name}_disabled_empty', 0)}")
        print(f"    removed_components_total:  {summary.get(f'{class_name}_removed_components_total', 0)}")
        print(f"    removed_area_total:        {summary.get(f'{class_name}_removed_area_total', 0)}")
    print(f"\n  decode_fail:  {summary.get('decode_fail', 0)}")
    print(f"  encode_fail:  {summary.get('encode_fail', 0)}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗伪标签 JSON 噪声（building / tree 小连通域过滤）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="所有参数的默认值见脚本顶部「用户配置区」。",
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT,
        help=f"输入 label JSON (默认 {INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT,
        help=f"清洗后的输出 JSON (默认 {OUTPUT})",
    )
    parser.add_argument(
        "--summary", type=Path, default=SUMMARY_OUTPUT,
        help=f"清洗统计 JSON (默认 {SUMMARY_OUTPUT})",
    )
    parser.add_argument(
        "--building-min-area", type=int, default=BUILDING_MIN_AREA,
        help=f"building 小连通域最小面积阈值 (默认 {BUILDING_MIN_AREA})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
