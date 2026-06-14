#!/usr/bin/env python3
"""
把 rare prompt 伪标签清洗后合并到 base label。

用法示例：

  python src/changwei/xiyou.py \
      --base-label data/base_val_6class.json \
      --rare-label data/rare_pred.json \
      --output-label data/new_val_label.json \
      --output-summary data/merge_summary.json

默认只输出 merged label 和 summary 两个文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── 路径 ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BASE_LABEL = ROOT / "test" / "sam3_label_old" / "train_tasks" / "pred_train_tasks.json"
RARE_LABEL = ROOT / "test" / "label_analysis" / "稀有类" /"train-out" / "pred_train_changwei.json"
OUTPUT_LABEL = ROOT / "test" / "clean_rare" / "xiyou-train" / "merged_label.json"
OUTPUT_SUMMARY = ROOT / "test" / "clean_rare" / "xiyou-train" / "merge_summary.json"

# ── 直接丢弃的 prompt ──────────────────────────────────────────────────────
DROP_PROMPTS = {
    "traffic light",
    "sign",
    "screen",
    "bicycle",
    "backpack",
    "animal",
    "wildlife",
    "sliding door",
    "tunnel light",
    "transmission tower",
    "electrical cabinet",
}
"""这些 prompt 直接丢弃。animal 是 rare 里的 animal，不参与合并（base animal 原样保留）"""

# ── 映射到已有主类的 prompt ────────────────────────────────────────────────
MERGE_TO_CLASS = {
    # car 系
    "truck": "car",
    "bus": "car",
    # trash can 系
    "dustbin": "trash can",
    "garbage bin": "trash can",
    # computer 系
    "laptop": "computer",
    "monitor": "computer",
    "keyboard": "computer",
}
"""rare prompt -> base 主类。只补充到目标类，不覆盖 base"""

# ── 保留为独立 rare 类的 prompt ────────────────────────────────────────────
KEEP_AS_RARE = [
    "window",
    "door",
    "fence",
    "motorcycle",
]
"""这些 prompt 作为独立类写入 base label"""

NEW_INDEPENDENT_CLASSES = [
    "window",
    "door",
    "fence",
    "motorcycle",
    "pole_light",
]
"""最终新增的独立目标类"""

RARE_TRAIN_LABEL_OVERRIDE = {
    # "window": "building",   # 示例：后续改成映射到主类时取消注释
}
"""可选覆盖：把 KEEP_AS_RARE 里的 prompt 映射到主类"""

# ── 需要减去 car 重合区域的 prompt ────────────────────────────────────────
SUBTRACT_CAR_FROM = {
    "window",
    "door",
}
"""这些 prompt 的 mask 会先减去同图 base label 里 car 的 union mask"""

# ── window/door 差集后最多保留组件数 ──────────────────────────────────────
MAX_COMPONENTS_AFTER_SUBTRACT = {
    "window": 12,
    "door": 8,
}
"""window/door 减 car 后只保留面积最大的前 N 个连通域，OR 成一个 mask，不再拆成多个 candidate"""

# ── 跨 prompt 合并组 ───────────────────────────────────────────────────────
CROSS_PROMPT_MERGE_GROUPS = {
    "pole_light": ["pole", "street light", "utility pole"],
}
"""同一组内的 prompt 先做跨 prompt 去重，再统一 train_label 为 key"""

# ── 安全白名单 ─────────────────────────────────────────────────────────────
def _build_allowed_source_prompts() -> set:
    allowed = set()
    allowed.update(DROP_PROMPTS)
    allowed.update(MERGE_TO_CLASS.keys())
    allowed.update(KEEP_AS_RARE)
    for src_list in CROSS_PROMPT_MERGE_GROUPS.values():
        allowed.update(src_list)
    return allowed

ALLOWED_SOURCE_PROMPTS = _build_allowed_source_prompts()
"""白名单：不在其中的 prompt 一律丢弃，不允许自动当新类写入"""

ALLOWED_TARGET_LABELS_TO_MODIFY = {
    "car",
    "computer",
    "trash can",
    "window",
    "door",
    "fence",
    "motorcycle",
    "pole_light",
}
"""允许修改/新增的 target label。person/tree/building/animal 不在此列，禁止改动"""

# ── 过滤参数 ───────────────────────────────────────────────────────────────
MIN_AREA = 64
"""全局最小 mask 面积（像素）"""

IOU_THRESH = 0.6
"""IoU 去重阈值"""

CONTAINMENT_THRESH = 0.8
"""containment 去重阈值（intersection / min(area_a, area_b)）"""

SMOOTHNESS_THRESH = 60.0
"""边界平滑度阈值：perimeter² / area。超过此值认为边界毛糙，丢弃"""

FENCE_MAX_SMOOTHNESS = 100.0
"""fence 放宽后的平滑度上限（长条形物体不能用圆紧致度）"""

FENCE_MAX_FRAGMENT_RATIO = 0.3
"""fence 最大碎片比例：小连通域总面积 / 总面积"""

# ── 各 prompt 独立阈值 ─────────────────────────────────────────────────────
PROMPT_SCORE_THRESHOLDS = {
    "window": 0.50,
    "door": 0.50,
    "fence": 0.50,
    "motorcycle": 0.55,
    "pole": 0.55,
    "street light": 0.55,
    "utility pole": 0.55,
    "truck": 0.60,
    "bus": 0.60,
    "dustbin": 0.58,
    "garbage bin": 0.58,
    "laptop": 0.58,
    "monitor": 0.70,
    "keyboard": 0.55,
}
"""每个 prompt 的最低 score 阈值。monitor 未可视化，阈值较高"""

DEFAULT_SCORE_THRESHOLD = 0.50

PROMPT_MIN_AREAS = {
    "pole": 32,
    "street light": 32,
    "utility pole": 32,
}
"""每个 prompt 的最小面积覆盖"""

# ── 每图上限（两阶段） ────────────────────────────────────────────────────
# 第一阶段：按 image_path + source_prompt 限制
MAX_PER_IMAGE_BY_PROMPT = {
    "monitor": 2,
    "keyboard": 2,
    "laptop": 3,

    "dustbin": 3,
    "garbage bin": 3,

    "truck": 5,
    "bus": 3,

    # pole_light 系：不再用 smoothness 误杀，改用每图 top-k 控制数量
    "pole": 5,
    "street light": 5,
    "utility pole": 5,

    "window": 12,
    "door": 8,
}
"""第一阶段：同图同 source_prompt 上限"""

# 第二阶段：按 image_path + train_label 限制
MAX_PER_IMAGE_BY_LABEL = {
    "window": 12,
    "door": 8,
    "fence": 5,
    "motorcycle": 5,
    "pole_light": 10,
    "car": 8,
    "trash can": 5,
    "computer": 3,
}
"""第二阶段：同图同 train_label 上限"""

MAX_PER_IMAGE_FALLBACK = 5
"""兜底：不在上述配置中的类别默认上限"""

# ── 边界质量参数 ───────────────────────────────────────────────────────────
MORPH_CLOSE_KERNEL = 3
MORPH_OPEN_KERNEL = 3

# ── 输出控制 ───────────────────────────────────────────────────────────────
VERBOSE = True

# ═══════════════════════════════════════════════════════════════════════════════
# ── 依赖导入 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
import json
import os
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


def _xp():
    """返回当前最优的数组模块 (cupy > numpy)。"""
    if HAS_CUPY:
        return cp
    return np


def _gpu_mean(arr: List[float]) -> float:
    """GPU 加速均值（小列表用 numpy）。"""
    if len(arr) == 0:
        return 0.0
    if len(arr) < 1000:
        return float(np.mean(arr))
    xp_mod = _xp()
    return float(xp_mod.mean(xp_mod.asarray(arr, dtype=xp_mod.float32)))


# ═══════════════════════════════════════════════════════════════════════════════
# ── RLE 工具 ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def decode_rle(rle: Dict[str, Any]) -> Optional[np.ndarray]:
    """解码 RLE -> 二值 mask (H, W) uint8."""
    if rle is None:
        return None
    size = rle.get("size")
    counts = rle.get("counts")
    if size is None or counts is None:
        return None
    h, w = int(size[0]), int(size[1])

    if HAS_PYCOCOTOOLS and mask_utils is not None:
        try:
            coco_rle = {
                "size": [h, w],
                "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
            }
            mask = mask_utils.decode(coco_rle)
            return mask.astype(np.uint8)
        except Exception:
            pass

    if isinstance(counts, str) and "," in counts:
        try:
            runs = [int(x) for x in counts.split(",")]
            total = h * w
            mask = np.zeros(total, dtype=np.uint8)
            pos, val = 0, 0
            for run_len in runs:
                if run_len > 0:
                    if val == 1:
                        mask[pos: pos + run_len] = 1
                    pos += run_len
                val = 1 - val
            return mask.reshape((h, w), order="F")
        except Exception:
            pass
    return None


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """编码二值 mask -> RLE dict."""
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError("encode_rle requires pycocotools")
    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_f)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# ── Mask 运算 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """两个二值 mask 的 IoU."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """containment = intersection / min(area_a, area_b)."""
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    if min(area_a, area_b) == 0:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / min(area_a, area_b))


def compute_perimeter_squared_over_area(mask: np.ndarray) -> float:
    """边界质量指标: perimeter² / area。越小越平滑。"""
    if not HAS_CV2:
        return 0.0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    area = max(mask.sum(), 1)
    return (total_perimeter ** 2) / area


def compute_fragment_ratio(mask: np.ndarray, min_component_area: int = 16) -> float:
    """小碎片比例：面积 < min_component_area 的连通域总面积 / 总面积."""
    if not HAS_CV2:
        return 0.0
    total_area = max(mask.sum(), 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    fragment_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            fragment_area += stats[i, cv2.CC_STAT_AREA]
    return float(fragment_area / total_area)


def subtract_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """mask_a 减去 mask_b 的重合部分。省内存版，不创建 int32 临时大数组。"""
    if mask_a.shape[:2] != mask_b.shape[:2]:
        mask_b = resize_mask_to_shape(mask_b, mask_a.shape[:2])
    result = np.logical_and(mask_a > 0, mask_b == 0)
    return result.astype(np.uint8)


def mask_union(masks: List[np.ndarray]) -> Optional[np.ndarray]:
    """多 mask 取像素级 OR。若尺寸不一致，后续 mask 对齐到第一个 mask 的尺寸."""
    if not masks:
        return None
    target_shape = masks[0].shape[:2]
    result = np.zeros(target_shape, dtype=np.uint8)
    for m in masks:
        if m.shape[:2] != target_shape:
            m = resize_mask_to_shape(m, target_shape)
        result = np.logical_or(result, m)
    return result.astype(np.uint8)


def mask_union_area(masks: List[np.ndarray]) -> int:
    """多 mask OR 后面积."""
    u = mask_union(masks)
    if u is None:
        return 0
    return int(u.sum())


def get_entry_reference_shape(entry: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """从 base entry 的 prompts 里找一个 hit=true 且有 rle 的 mask size 作为参考尺寸 (H,W)."""
    if not isinstance(entry, dict):
        return None
    prompts = entry.get("prompts", {})
    if not isinstance(prompts, dict):
        return None
    for _, info in prompts.items():
        if not isinstance(info, dict):
            continue
        if not info.get("hit"):
            continue
        rle = info.get("rle")
        if not isinstance(rle, dict):
            continue
        size = rle.get("size")
        if isinstance(size, list) and len(size) == 2:
            return int(size[0]), int(size[1])
    return None


def resize_mask_to_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """把 mask resize 到 target_shape=(H,W)，使用最近邻，保持二值."""
    if mask.shape[:2] == target_shape:
        return mask.astype(np.uint8)
    if not HAS_CV2 or cv2 is None:
        raise RuntimeError(
            f"mask shape mismatch {mask.shape} -> {target_shape}, but cv2 is not available"
        )
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def align_candidates_to_base_shape(
    candidates: List[Dict[str, Any]],
    base_index: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """把 rare candidate 的 mask 统一对齐到同图 base entry 的参考尺寸."""
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        ip = c.get("image_path")
        base_entry = base_index.get(ip)
        if base_entry is None:
            kept.append(c)
            continue
        ref_shape = get_entry_reference_shape(base_entry)
        if ref_shape is None:
            kept.append(c)
            continue
        mask = c.get("mask")
        if mask is None:
            kept.append(c)
            continue
        if mask.shape[:2] != ref_shape:
            old_shape = mask.shape[:2]
            mask = resize_mask_to_shape(mask, ref_shape)
            c["mask"] = mask
            c["area"] = mask_area(mask)
            summary["resized_to_base_shape"] = summary.get("resized_to_base_shape", 0) + 1
            key = f"resized_{c.get('source_prompt', 'unknown')}_{old_shape[0]}x{old_shape[1]}_to_{ref_shape[0]}x{ref_shape[1]}"
            summary[key] = summary.get(key, 0) + 1
            if c["area"] <= 0:
                summary["dropped_empty_after_resize"] = summary.get("dropped_empty_after_resize", 0) + 1
                continue
        kept.append(c)
    return kept


def morphology_close(mask: np.ndarray, kernel_size: int = MORPH_CLOSE_KERNEL) -> np.ndarray:
    """形态学闭运算."""
    if not HAS_CV2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def morphology_open(mask: np.ndarray, kernel_size: int = MORPH_OPEN_KERNEL) -> np.ndarray:
    """形态学开运算."""
    if not HAS_CV2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def get_connected_components(
    mask: np.ndarray, min_area: int = 0, max_components: int = 0
) -> List[np.ndarray]:
    """提取连通域，返回每个连通域的独立 mask 列表。可按面积排序取 top-k."""
    if not HAS_CV2:
        return [mask]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area > 0 and area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        components.append((area, comp))
    # 按面积降序
    components.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in components]
    if max_components > 0 and len(result) > max_components:
        result = result[:max_components]
    return result


def filter_small_components(mask: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    """去掉面积 < min_area 的连通域."""
    comps = get_connected_components(mask)
    kept = [c for c in comps if mask_area(c) >= min_area]
    if not kept:
        return np.zeros_like(mask)
    u = mask_union(kept)
    if u is None:
        return np.zeros_like(mask)
    return u


def filter_small_components_compact(
    mask: np.ndarray,
    min_area: int = MIN_AREA,
    max_components: Optional[int] = None,
) -> np.ndarray:
    """
    省内存版连通域过滤：
    - 不返回多个 full-size component mask；
    - 直接返回一个清理后的单个 mask；
    - 可选只保留面积最大的前 max_components 个连通域。
    """
    if not HAS_CV2:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append((i, area))
    comps.sort(key=lambda x: x[1], reverse=True)
    if max_components is not None and len(comps) > max_components:
        comps = comps[:max_components]
    cleaned = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i, _ in comps:
        cleaned[labels == i] = 1
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# ── 数据加载 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def load_label_json(path: Path) -> Optional[List[Dict[str, Any]]]:
    """加载伪标签 JSON。自动识别 entries / annotations / list 格式."""
    if path is None or not path.exists():
        print(f"[WARN] 文件不存在: {path}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("annotations") or data.get("records") or [data]
    else:
        return None
    if VERBOSE:
        print(f"[INFO] 加载: {path}  ({len(entries)} entries)")
    return entries


def build_image_index(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按 image_path 建索引."""
    idx: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        ip = str(e.get("image_path", "")).replace("\\", "/")
        idx[ip] = e
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# ── Rare candidate 提取 ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def extract_rare_candidates(
    rare_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从 rare pred JSON 中提取所有 hit=true 的 prompt 实例为 candidate 列表。
    兼容两种格式：
      A: {"hit":true, "score":0.8, "rle":{...}}
      B: {"hit":true, "instances":[{"score":0.8,"rle":{...}}, ...]}
    """
    candidates: List[Dict[str, Any]] = []
    for entry in _progress(rare_entries, desc="提取 rare candidates", unit="entry"):
        image_path = str(entry.get("image_path", "")).replace("\\", "/")
        prompts = entry.get("prompts", {})
        if not isinstance(prompts, dict):
            continue
        for prompt, pdata in prompts.items():
            if not isinstance(pdata, dict):
                continue
            if pdata.get("hit") is False:
                continue

            instances = pdata.get("instances")
            # 格式 B：instances 是 list
            if isinstance(instances, list):
                for idx, inst in enumerate(instances):
                    rle = inst.get("rle") if isinstance(inst, dict) else None
                    if rle is None:
                        continue
                    score = float(inst.get("score", 0.5))
                    candidates.append({
                        "image_path": image_path,
                        "source_prompt": prompt,
                        "score": score,
                        "instances": 1,
                        "instance_index": idx,
                        "rle": rle,
                    })
            else:
                # 格式 A：单个 rle
                rle = pdata.get("rle")
                if rle is None:
                    continue
                score = float(pdata.get("score", 0.5))
                inst_count = int(instances) if instances is not None else 1
                candidates.append({
                    "image_path": image_path,
                    "source_prompt": prompt,
                    "score": score,
                    "instances": inst_count,
                    "rle": rle,
                })
    if VERBOSE:
        print(f"[INFO] 提取 rare candidates: {len(candidates)}")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# ── 基础过滤 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def get_prompt_score_threshold(prompt: str) -> float:
    """获取 prompt 的 score 阈值."""
    return PROMPT_SCORE_THRESHOLDS.get(prompt, DEFAULT_SCORE_THRESHOLD)


def get_prompt_min_area(prompt: str) -> int:
    """获取 prompt 的最小面积."""
    return PROMPT_MIN_AREAS.get(prompt, MIN_AREA)


def basic_filter(
    candidates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """基础过滤：drop prompt、score 阈值、面积、decode 失败."""
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = defaultdict(int)

    for c in _progress(candidates, desc="基础过滤", unit="cand"):
        sp = c["source_prompt"]

        # 白名单检查：未配置 prompt 一律丢弃
        if sp not in ALLOWED_SOURCE_PROMPTS:
            dropped[f"unknown_prompt_{sp}"] += 1
            continue

        # 直接丢弃的 prompt
        if sp in DROP_PROMPTS:
            dropped[f"dropped_prompt_{sp}"] += 1
            continue

        # decode mask
        mask = decode_rle(c["rle"])
        if mask is None:
            summary["decode_fail"] = summary.get("decode_fail", 0) + 1
            dropped["decode_fail"] += 1
            continue

        # score 过滤
        threshold = get_prompt_score_threshold(sp)
        if c["score"] < threshold:
            dropped[f"low_score_{sp}"] += 1
            continue

        # 面积过滤
        min_a = get_prompt_min_area(sp)
        area = mask_area(mask)
        if area < min_a:
            dropped[f"too_small_{sp}"] += 1
            continue

        c["mask"] = mask
        c["area"] = area
        kept.append(c)

    summary["candidates_raw"] = len(candidates)
    summary["candidates_after_basic_filter"] = len(kept)
    if VERBOSE:
        print(f"[INFO] 基础过滤: {len(candidates)} → {len(kept)}")
    return kept, dict(dropped)


def get_train_label(candidate: Dict[str, Any]) -> str:
    """统一获取 candidate 的目标 train_label."""
    if "train_label" in candidate:
        return candidate["train_label"]
    sp = candidate["source_prompt"]
    if sp in MERGE_TO_CLASS:
        return MERGE_TO_CLASS[sp]
    if sp in RARE_TRAIN_LABEL_OVERRIDE:
        return RARE_TRAIN_LABEL_OVERRIDE[sp]
    return sp


# ═══════════════════════════════════════════════════════════════════════════════
# ── 特殊处理 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def subtract_car_from_candidates(
    candidates: List[Dict[str, Any]],
    base_index: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """对 SUBTRACT_CAR_FROM 里的 prompt，减去同图 base car union mask。
    差集后重新提取连通域，每个连通域成为独立 candidate（一个输入可能变成多个输出）。"""
    kept: List[Dict[str, Any]] = []
    for c in _progress(candidates, desc="window/door 减 car", unit="cand"):
        sp = c["source_prompt"]
        if sp not in SUBTRACT_CAR_FROM:
            kept.append(c)
            continue

        ip = c["image_path"]
        base_entry = base_index.get(ip)
        car_union_mask = None
        if base_entry is not None:
            base_prompts = base_entry.get("prompts", {})
            if isinstance(base_prompts, dict):
                car_info = base_prompts.get("car", {})
                if isinstance(car_info, dict) and car_info.get("hit"):
                    car_rle = car_info.get("rle")
                    if car_rle is not None:
                        car_union_mask = decode_rle(car_rle)

        # 只要有 car 就做差集（不设 overlap_ratio 门槛）
        if car_union_mask is not None and car_union_mask.sum() > 0:
            summary[f"{sp}_car_subtract_input"] = \
                summary.get(f"{sp}_car_subtract_input", 0) + 1
            new_mask = subtract_mask(c["mask"], car_union_mask)
            # 省内存：不再拆成多个 full-size candidate，
            # 而是过滤小连通域后仍保留为一个整体 mask
            min_a = get_prompt_min_area(sp)
            max_comp = MAX_COMPONENTS_AFTER_SUBTRACT.get(sp, None)
            new_mask = filter_small_components_compact(
                new_mask, min_area=min_a, max_components=max_comp,
            )
            new_area = mask_area(new_mask)
            if new_area <= 0:
                summary[f"{sp}_dropped_after_car_subtract"] = \
                    summary.get(f"{sp}_dropped_after_car_subtract", 0) + 1
                continue
            new_c = dict(c)
            new_c["mask"] = new_mask
            new_c["area"] = new_area
            kept.append(new_c)
            summary[f"{sp}_car_subtract_kept"] = \
                summary.get(f"{sp}_car_subtract_kept", 0) + 1
            summary[f"{sp}_car_subtract_compact"] = \
                summary.get(f"{sp}_car_subtract_compact", 0) + 1
        else:
            kept.append(c)
    return kept


def process_fence_candidates(
    candidates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """fence 特殊处理：形态学平滑、去小面积、边界质量过滤."""
    kept: List[Dict[str, Any]] = []
    for c in _progress(candidates, desc="fence 清洗", unit="cand"):
        if c["source_prompt"] != "fence":
            kept.append(c)
            continue

        mask = c["mask"]

        # 形态学平滑
        mask = morphology_close(mask)
        mask = morphology_open(mask)

        # 去小面积连通域
        mask = filter_small_components(mask, get_prompt_min_area("fence"))
        if mask.sum() == 0:
            summary["fence_dropped_empty_after_morph"] = \
                summary.get("fence_dropped_empty_after_morph", 0) + 1
            continue

        # 边界质量检查
        p2a = compute_perimeter_squared_over_area(mask)
        frag = compute_fragment_ratio(mask)

        if p2a > FENCE_MAX_SMOOTHNESS:
            summary["fence_dropped_rough_boundary"] = \
                summary.get("fence_dropped_rough_boundary", 0) + 1
            continue
        if frag > FENCE_MAX_FRAGMENT_RATIO:
            summary["fence_dropped_too_fragmented"] = \
                summary.get("fence_dropped_too_fragmented", 0) + 1
            continue

        c["mask"] = mask
        c["area"] = mask_area(mask)
        kept.append(c)
    return kept


def process_general_boundary_filter(
    candidates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """通用边界质量过滤（非 fence 的普通 prompt）."""
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        sp = c["source_prompt"]
        # fence 已在 process_fence_candidates 里处理过
        if sp == "fence":
            kept.append(c)
            continue
        # 对 pole/street_light/motorcycle 做边界检查
        # 只对 motorcycle 做通用边界质量过滤
        # pole/street light/utility pole 是细长目标，不适合用 perimeter²/area 过滤
        if sp in ("motorcycle",):
            p2a = compute_perimeter_squared_over_area(c["mask"])
            if p2a > SMOOTHNESS_THRESH:
                summary[f"{sp}_dropped_rough_boundary"] = \
                    summary.get(f"{sp}_dropped_rough_boundary", 0) + 1
                continue
        kept.append(c)
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# ── 去重 ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def _is_duplicate(
    mask_a: np.ndarray, mask_b: np.ndarray,
    iou_thresh: float = IOU_THRESH,
    containment_thresh: float = CONTAINMENT_THRESH,
) -> bool:
    """判断两个 mask 是否重复。尺寸不一致时把 mask_b 对齐到 mask_a."""
    if mask_a.shape != mask_b.shape:
        mask_b = resize_mask_to_shape(mask_b, mask_a.shape)
    iou = compute_iou(mask_a, mask_b)
    if iou > iou_thresh:
        return True
    containment = compute_containment(mask_a, mask_b)
    if containment > containment_thresh:
        return True
    return False


def _candidate_quality_key(c: Dict[str, Any]) -> Tuple[float, int]:
    """轻量排序：score 高优先，面积大优先。避免排序阶段反复调用 cv2。"""
    return (-float(c.get("score", 0.0)), -int(c.get("area", 0)))


def dedup_same_class(
    candidates: List[Dict[str, Any]],
    base_index: Dict[str, Dict[str, Any]],
    target_class: str,
    summary: Dict[str, Any],
    is_merge_to_class: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    对指定 target_class 的 rare candidates 做同图同类去重。
    is_merge_to_class=True: 和 base 对应类也去重，优先保留 base。
    """
    kept: List[Dict[str, Any]] = []
    dedup_stats: Dict[str, int] = {"rare_base_dup": 0, "rare_rare_dup": 0}

    # 按 image_path 分组
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_image[c["image_path"]].append(c)

    for ip, items in _progress(
        list(by_image.items()), desc=f"  去重 {target_class}", unit="img", leave=False
    ):
        # 加载 base mask（如果有）
        existing_masks: List[Tuple[np.ndarray, str]] = []  # (mask, source_label)
        if is_merge_to_class and ip in base_index:
            base_entry = base_index[ip]
            base_prompts = base_entry.get("prompts", {})
            if isinstance(base_prompts, dict):
                tc_info = base_prompts.get(target_class, {})
                if isinstance(tc_info, dict) and tc_info.get("hit"):
                    tc_rle = tc_info.get("rle")
                    if tc_rle is not None:
                        tc_mask = decode_rle(tc_rle)
                        if tc_mask is not None:
                            existing_masks.append((tc_mask, "base"))

        # 按质量排序
        items.sort(key=_candidate_quality_key)

        for item in items:
            is_dup = False
            for em, em_src in existing_masks:
                if _is_duplicate(item["mask"], em, IOU_THRESH, CONTAINMENT_THRESH):
                    is_dup = True
                    if em_src == "base":
                        dedup_stats["rare_base_dup"] += 1
                    else:
                        dedup_stats["rare_rare_dup"] += 1
                    break

            if not is_dup:
                kept.append(item)
                existing_masks.append((item["mask"], item["source_prompt"]))

    summary[f"{target_class}_dedup_base"] = dedup_stats["rare_base_dup"]
    summary[f"{target_class}_dedup_rare"] = dedup_stats["rare_rare_dup"]
    return kept, dedup_stats


def merge_cross_prompt_group(
    candidates: List[Dict[str, Any]],
    group_name: str,
    source_prompts: List[str],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """跨 prompt 去重并统一 train_label。
    例如 pole + street light → pole_light."""
    sp_set = set(source_prompts)
    group_cands = [c for c in candidates if c["source_prompt"] in sp_set]
    other_cands = [c for c in candidates if c["source_prompt"] not in sp_set]

    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in group_cands:
        by_image[c["image_path"]].append(c)

    kept_group: List[Dict[str, Any]] = []
    dup_count = 0
    input_count = len(group_cands)

    for ip, items in _progress(
        list(by_image.items()), desc=f"  合并 {group_name}", unit="img", leave=False
    ):
        items.sort(key=_candidate_quality_key)
        existing_masks: List[np.ndarray] = []
        for item in items:
            is_dup = False
            for em in existing_masks:
                if _is_duplicate(item["mask"], em):
                    is_dup = True
                    dup_count += 1
                    break
            if not is_dup:
                # 统一 train_label
                item["train_label"] = group_name
                item["merged_group"] = group_name
                kept_group.append(item)
                existing_masks.append(item["mask"])

    summary[f"{group_name}_input"] = input_count
    summary[f"{group_name}_cross_dedup"] = dup_count
    summary[f"{group_name}_output"] = len(kept_group)
    if VERBOSE:
        print(f"[INFO] {group_name}: {input_count} → {len(kept_group)} "
              f"(dedup {dup_count})")
    return other_cands + kept_group


def apply_max_per_image_limits(
    candidates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """两阶段上限截断。
    阶段1：按 image_path + source_prompt 限制（MAX_PER_IMAGE_BY_PROMPT）
    阶段2：按 image_path + train_label 限制（MAX_PER_IMAGE_BY_LABEL）
    """
    trimmed_prompt = 0
    trimmed_label = 0

    # ── 阶段1：按 source_prompt ──
    by_image_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        key = f"{c['image_path']}|||{c['source_prompt']}"
        by_image_prompt[key].append(c)
    stage1_kept: List[Dict[str, Any]] = []
    for key, items in by_image_prompt.items():
        sp = items[0]["source_prompt"]
        limit = MAX_PER_IMAGE_BY_PROMPT.get(sp, MAX_PER_IMAGE_FALLBACK * 2)
        if len(items) <= limit:
            stage1_kept.extend(items)
        else:
            items.sort(key=_candidate_quality_key)
            stage1_kept.extend(items[:limit])
            trimmed_prompt += len(items) - limit

    # ── 阶段2：按 train_label ──
    by_image_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in stage1_kept:
        tl = get_train_label(c)
        key = f"{c['image_path']}|||{tl}"
        by_image_label[key].append(c)
    stage2_kept: List[Dict[str, Any]] = []
    for key, items in by_image_label.items():
        tl = get_train_label(items[0])
        limit = MAX_PER_IMAGE_BY_LABEL.get(tl, MAX_PER_IMAGE_FALLBACK)
        if len(items) <= limit:
            stage2_kept.extend(items)
        else:
            items.sort(key=_candidate_quality_key)
            stage2_kept.extend(items[:limit])
            trimmed_label += len(items) - limit

    if trimmed_prompt:
        summary["trimmed_by_prompt_limit"] = trimmed_prompt
    if trimmed_label:
        summary["trimmed_by_label_limit"] = trimmed_label
    if VERBOSE and (trimmed_prompt or trimmed_label):
        print(f"[INFO] 上限截断: prompt={trimmed_prompt}, label={trimmed_label}")
    return stage2_kept


# ═══════════════════════════════════════════════════════════════════════════════
# ── 合并输出 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def merge_into_base(
    base_entries: List[Dict[str, Any]],
    kept_candidates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    将清洗后的 rare candidates 合并进 base label。
    - MERGE_TO_CLASS 映射的 prompt：追加到对应主类（重复则保留 base）
    - KEEP_AS_RARE / CROSS_PROMPT_MERGE_GROUPS 的 prompt：作为独立类写入
    """
    import copy
    merged = copy.deepcopy(base_entries)
    base_index = build_image_index(merged)

    final_per_class: Dict[str, int] = defaultdict(int)

    for c in _progress(kept_candidates, desc="写回 base label", unit="cand"):
        ip = c["image_path"]
        sp = c["source_prompt"]
        train_label = get_train_label(c)

        # 安全检查：只允许修改白名单内的 label
        if train_label not in ALLOWED_TARGET_LABELS_TO_MODIFY:
            summary[f"blocked_target_label_{train_label}"] = \
                summary.get(f"blocked_target_label_{train_label}", 0) + 1
            continue

        entry = base_index.get(ip)
        if entry is None:
            entry = {"image_path": ip, "prompts": {}}
            merged.append(entry)
            base_index[ip] = entry

        prompts = entry.setdefault("prompts", {})
        if not isinstance(prompts, dict):
            prompts = {}
            entry["prompts"] = prompts

        # 对 MERGE_TO_CLASS 类：检查是否和 base 同类重复
        if sp in MERGE_TO_CLASS:
            existing_info = prompts.get(train_label, {})
            if isinstance(existing_info, dict) and existing_info.get("hit"):
                existing_rle = existing_info.get("rle")
                if existing_rle is not None:
                    existing_mask = decode_rle(existing_rle)
                    if existing_mask is not None:
                        if _is_duplicate(c["mask"], existing_mask):
                            summary[f"{sp}_skipped_dup_base_on_write"] = \
                                summary.get(f"{sp}_skipped_dup_base_on_write", 0) + 1
                            continue

        # 写入：同 label 已有 hit → 做 OR
        existing_info = prompts.get(train_label, {})
        existing_mask = None
        existing_sources = []
        if isinstance(existing_info, dict) and existing_info.get("hit"):
            existing_rle = existing_info.get("rle")
            if existing_rle is not None:
                existing_mask = decode_rle(existing_rle)
            existing_sources = existing_info.get("source_prompts", [])
            if not existing_sources and existing_info.get("source_prompt"):
                existing_sources = [existing_info["source_prompt"]]

        masks_to_union = []
        if existing_mask is not None:
            masks_to_union.append(existing_mask)
        masks_to_union.append(c["mask"])

        u = mask_union(masks_to_union)
        final_rle = encode_rle(u) if u is not None else encode_rle(c["mask"])

        # source_prompts 合并去重
        new_sources = list(set(existing_sources + [sp]))

        prompts[train_label] = {
            "hit": True,
            "score": round(max(float(existing_info.get("score", 0))
                                if isinstance(existing_info, dict) else 0,
                                c["score"]), 4),
            "rle": final_rle,
            "merged_from": "rare_clean",
            "source_prompts": new_sources,
        }
        final_per_class[train_label] += 1

    summary["final_per_class"] = dict(final_per_class)
    if VERBOSE:
        print(f"[INFO] 合并完成: {len(merged)} entries")
        for label, count in sorted(final_per_class.items()):
            print(f"  {label}: +{count}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# ── Summary 输出 ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def write_summary(summary: Dict[str, Any], output_path: Path) -> None:
    """输出 summary JSON."""
    # 加入规则说明
    summary["_rules"] = {
        "drop_prompts": sorted(DROP_PROMPTS),
        "merge_to_class": dict(MERGE_TO_CLASS),
        "keep_as_rare": list(KEEP_AS_RARE),
        "new_independent_classes": list(NEW_INDEPENDENT_CLASSES),
        "cross_prompt_merge_groups": {k: list(v) for k, v in CROSS_PROMPT_MERGE_GROUPS.items()},
        "subtract_car_from": sorted(SUBTRACT_CAR_FROM),
        "allowed_target_labels_to_modify": sorted(ALLOWED_TARGET_LABELS_TO_MODIFY),
        "min_area": MIN_AREA,
        "iou_thresh": IOU_THRESH,
        "containment_thresh": CONTAINMENT_THRESH,
        "smoothness_thresh": SMOOTHNESS_THRESH,
        "score_thresholds": dict(PROMPT_SCORE_THRESHOLDS),
        "max_per_image_by_label": dict(MAX_PER_IMAGE_BY_LABEL),
        "max_per_image_by_prompt": dict(MAX_PER_IMAGE_BY_PROMPT),
        "note": (
            "category-based mode, no ABC; "
            "base label is preserved; "
            "only car/computer/trash can and independent rare labels are modified; "
            "animal/person/tree/building are not modified; "
            "truck/bus -> car; "
            "dustbin/garbage bin -> trash can; "
            "laptop/monitor/keyboard -> computer; "
            "window/door/fence/motorcycle kept as independent rare labels; "
            "pole/street light/utility pole -> pole_light; "
            "traffic light/sign/screen and other configured prompts dropped; "
            "rare masks are resized to base entry reference shape before subtract/dedup/union; "
            "pole/street light/utility pole are not filtered by smoothness because they are thin objects; top-k limits are used instead; "
            "window/door subtraction uses compact connected-component filtering and does not split one mask into many full-size candidates to reduce memory usage; "
            "unknown prompts dropped by whitelist"
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {output_path}")


def build_summary_csv(summary: Dict[str, Any], output_dir: Path) -> Path:
    """从 summary dict 生成 per-prompt 统计 CSV."""
    path = output_dir / "per_prompt_stats.csv"
    # 提取 per-prompt 的前缀统计
    prompt_prefixes: Dict[str, Dict[str, int]] = defaultdict(dict)

    # 收集所有已知 prompt
    all_known_prompts = (list(DROP_PROMPTS) + list(MERGE_TO_CLASS.keys())
                         + KEEP_AS_RARE)
    for src_list in CROSS_PROMPT_MERGE_GROUPS.values():
        all_known_prompts.extend(src_list)

    for key, val in summary.items():
        if isinstance(val, (int, float)):
            for prompt in all_known_prompts:
                if key.startswith(prompt) or prompt in key:
                    stat_name = key.replace(f"{prompt}_", "").replace(f"_{prompt}", "")
                    prompt_prefixes[prompt][stat_name] = int(val)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "stat", "value"])
        for prompt, stats in sorted(prompt_prefixes.items()):
            for stat, val in sorted(stats.items()):
                writer.writerow([prompt, stat, val])
    print(f"[OUTPUT] {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# ── 主流程 ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def run(args: argparse.Namespace) -> None:
    global MIN_AREA, IOU_THRESH, CONTAINMENT_THRESH, SMOOTHNESS_THRESH, VERBOSE

    # 命令行覆盖配置
    if hasattr(args, "min_area") and args.min_area is not None:
        MIN_AREA = args.min_area
    if hasattr(args, "iou_thresh") and args.iou_thresh is not None:
        IOU_THRESH = args.iou_thresh
    if hasattr(args, "containment_thresh") and args.containment_thresh is not None:
        CONTAINMENT_THRESH = args.containment_thresh
    if hasattr(args, "smoothness_thresh") and args.smoothness_thresh is not None:
        SMOOTHNESS_THRESH = args.smoothness_thresh

    backend = "cpu"
    if HAS_CUPY:
        backend = "cupy"
    elif HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda":
        backend = "torch"
    print(f"[SYS] GPU: {backend}  pycocotools: {'✓' if HAS_PYCOCOTOOLS else '✗'}  "
          f"cv2: {'✓' if HAS_CV2 else '✗'}  tqdm: {'✓' if HAS_TQDM else '✗'}")

    t0 = time.time()

    # 1. 加载数据
    base_entries = load_label_json(args.base_label)
    rare_entries = load_label_json(args.rare_label)
    if base_entries is None or rare_entries is None:
        print("[ERROR] 加载数据失败", file=sys.stderr)
        sys.exit(1)

    base_index = build_image_index(base_entries)

    # 2. 提取 rare candidates
    candidates = extract_rare_candidates(rare_entries)

    # 3. 基础过滤（DROP_PROMPTS / score / area / decode）
    summary: Dict[str, Any] = {}
    candidates, dropped_stats = basic_filter(candidates, summary)
    for k, v in dropped_stats.items():
        summary[k] = summary.get(k, 0) + v

    after_filter_counts: Dict[str, int] = defaultdict(int)
    for c in candidates:
        after_filter_counts[c["source_prompt"]] += 1
    summary["candidates_by_prompt_after_filter"] = dict(after_filter_counts)

    # 3.5 对齐 rare mask 到 base 同图参考尺寸，避免 1024x1280 vs 512x640 报错
    candidates = align_candidates_to_base_shape(candidates, base_index, summary)

    # 4. window/door 减去 car（差集后拆分连通域）
    candidates = subtract_car_from_candidates(candidates, base_index, summary)

    # 5. fence 特殊处理（形态学 + 边界质量）
    candidates = process_fence_candidates(candidates, summary)

    # 6. 通用边界过滤
    candidates = process_general_boundary_filter(candidates, summary)

    # 7. 跨 prompt 合并（pole + street_light → pole_light）
    for group_name, src_prompts in CROSS_PROMPT_MERGE_GROUPS.items():
        candidates = merge_cross_prompt_group(candidates, group_name, src_prompts, summary)

    # 8. MERGE_TO_CLASS 类按 target class 去重（含 base）
    merge_class_cands = [c for c in candidates if c["source_prompt"] in MERGE_TO_CLASS]
    other_cands = [c for c in candidates if c["source_prompt"] not in MERGE_TO_CLASS]

    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in merge_class_cands:
        tc = get_train_label(c)
        by_target[tc].append(c)

    all_merge_kept: List[Dict[str, Any]] = []
    for tc, items in by_target.items():
        kept, _ = dedup_same_class(items, base_index, tc, summary, is_merge_to_class=True)
        all_merge_kept.extend(kept)

    # 9. 独立类内部去重（KEEP_AS_RARE + pole_light 等）
    rare_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in other_cands:
        tl = get_train_label(c)
        rare_by_class[tl].append(c)

    all_rare_kept: List[Dict[str, Any]] = []
    for tl, items in rare_by_class.items():
        kept, _ = dedup_same_class(items, base_index, tl, summary, is_merge_to_class=False)
        all_rare_kept.extend(kept)

    # 10. 每图每类上限截断
    all_kept = all_merge_kept + all_rare_kept
    all_kept = apply_max_per_image_limits(all_kept, summary)

    summary["final_kept_count"] = len(all_kept)

    # 11. 合并写回 base
    merged_entries = merge_into_base(base_entries, all_kept, summary)

    # 12. 写输出
    output_label = args.output_label
    output_label.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label, "w", encoding="utf-8") as f:
        json.dump(merged_entries, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {output_label}  ({len(merged_entries)} entries)")

    summary_output = args.output_summary
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_summary(summary, summary_output)

    # CSV 仅当显式传参时输出
    if getattr(args, "write_csv_summary", False):
        build_summary_csv(summary, summary_output.parent)

    print(f"\n[DONE] elapsed={time.time() - t0:.1f}s")
    print(f"  base entries:      {len(base_entries)}")
    print(f"  rare candidates:   {summary.get('candidates_raw', 0)}")
    print(f"  after filter:      {summary.get('candidates_after_basic_filter', 0)}")
    print(f"  final kept:        {summary.get('final_kept_count', 0)}")
    print(f"  output label:      {output_label}")
    print(f"  output summary:    {summary_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗 rare prompt 伪标签并合并到 base label",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="所有可配参数默认值见脚本顶部「用户配置区」。",
    )
    parser.add_argument(
        "--base-label", type=Path, default=BASE_LABEL,
        help=f"原主伪标签 JSON (默认 {BASE_LABEL})",
    )
    parser.add_argument(
        "--rare-label", type=Path, default=RARE_LABEL,
        help=f"rare prompt 伪标签 JSON (默认 {RARE_LABEL})",
    )
    parser.add_argument(
        "--output-label", type=Path, default=OUTPUT_LABEL,
        help=f"合并后的新 label JSON (默认 {OUTPUT_LABEL})",
    )
    parser.add_argument(
        "--output-summary", type=Path, default=OUTPUT_SUMMARY,
        help=f"统计文件 JSON (默认 {OUTPUT_SUMMARY})",
    )
    parser.add_argument(
        "--min-area", type=int, default=MIN_AREA,
        help=f"最小 mask 面积 (默认 {MIN_AREA})",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=IOU_THRESH,
        help=f"IoU 去重阈值 (默认 {IOU_THRESH})",
    )
    parser.add_argument(
        "--containment-thresh", type=float, default=CONTAINMENT_THRESH,
        help=f"containment 去重阈值 (默认 {CONTAINMENT_THRESH})",
    )
    parser.add_argument(
        "--smoothness-thresh", type=float, default=SMOOTHNESS_THRESH,
        help=f"边界平滑度阈值 (默认 {SMOOTHNESS_THRESH})",
    )
    parser.add_argument(
        "--write-csv-summary", action="store_true", default=False,
        help="额外输出 per_prompt_stats.csv（默认不输出）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
