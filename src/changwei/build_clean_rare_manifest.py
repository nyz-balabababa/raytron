#!/usr/bin/env python3
"""
基于多版 SAM3 稀有类伪标签 pred_*.json，按策略筛选、映射到 7 类、去重，
输出 clean_rare_manifest，用于 CLIPSeg 微调。

用法示例：

  # 填好顶部「用户配置区」后直接运行
  python src/changwei/build_clean_rare_manifest.py

  # 也可以用命令行覆盖配置区的值
  python src/changwei/build_clean_rare_manifest.py \
      --rare-pred-jsons new=data/test/pred_val_animal.json \
      --dry-run

注意：
- 本脚本只生成 manifest，不改训练主逻辑，不改官方 SAM3 源码。
- 默认 validation 伪标签不进入训练 manifest，除非显式 INCLUDE_VAL=True。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
"""项目根目录，所有相对路径都基于这里"""

# 路径
IMAGE_ROOT = ROOT
#图片根目录，用于拼接 image_path 得到真实文件路径

OUTPUT_DIR = ROOT / "test" / "clean_rare" / "animal-val"
#输出目录，manifest / 报告 / meta 都放在这里

#ABC文件
RARE_PRED_JSONS = [
    r"A = D:\nyz\raytron_project\test\label_analysis\label_analysis_ABC(changwei-val)\pseudo_A.json",
    r"B = D:\nyz\raytron_project\test\label_analysis\label_analysis_ABC(changwei-val)\pseudo_B.json",
    r"C = D:\nyz\raytron_project\test\label_analysis\label_analysis_ABC(changwei-val)\pseudo_C.json"
    
]


BASE_PSEUDO_JSON = ROOT / "test" / "sam3_label_old" / "val_tasks1" / "pred_val_tasks1.json"
#最全主伪标签路径 (Path 或 None)，传入则与 clean rare 做并集去重。
#示例: BASE_PSEUDO_JSON = ROOT / "data" / "main_pseudo" / "pred_val_all.json"

POLICY_JSON = None
#自定义 prompt 策略 JSON 路径 (str 或 None)，不传则使用下方 PROMPT_POLICY

# 筛选开关
INCLUDE_VAL = True
#是否允许 validation 伪标签进入训练 manifest

SPLIT_FILTER = None
#只处理 image_path 中包含这些关键词的图片。None=全量, 示例: ["data5", "data7"]

MERGE_ANIMAL_INTO_BASE = True
#True=默认使用新模式，只把 rare animal 合进 base prompts["animal"]
#False=使用旧模式/只输出 clean rare manifest

DRY_RUN = False
#True=只输出报告不写 manifest

# 面积过滤
MIN_AREA = 16
#最小 mask 面积（像素），低于此值丢弃

MAX_AREA_RATIO = 0.40
#默认最大 mask 面积占图比，超过则丢弃

RELAXED_MAX_AREA_RATIO = 0.50
#animal/car 的放宽面积上限

# 去重参数
IOU_THRESHOLD = 0.6
#IoU 去重阈值

CONTAINMENT_THRESHOLD = 0.8
#containment 去重阈值

# 目标类别
FINAL_CLASSES = [
    "person",
    "tree",
    "building",
    "animal",
    "car",
    "computer",
    "trash can",
]
"""最终 7 类目标标签"""

FRAGILE_PROMPTS = {"duck", "pig", "rabbit", "screen", "bin"}
"""旧版无 top-k 的脆弱 prompt，直接丢弃"""

MAX_PER_TRAIN_LABEL = {
    "computer": 4,
    "trash can": 4,
    "animal": 5,
    "car": 5,
}
"""同图同 label 最多保留数量"""

DEFAULT_MAX_PER_LABEL = 5
"""默认同图同 label 上限"""

# ── ABC 合并策略 ───────────────────────────────────────────────────────
ABC_GRADE_POLICY = {
    "A": {
        "use": True,
        "weight_mul": 1.0,
        "prompt_mode": "animal_plus_source",  # prompt_pool = ["animal", source_prompt]
    },
    "B": {
        "use": True,
        "weight_mul": 0.4,
        "prompt_mode": "animal_only",  # prompt_pool = ["animal"]
    },
    "C": {
        "use": True,
        "weight_mul": 0.15,
        "prompt_mode": "animal_only",
        "max_per_image": 2,  # 每张图最多保留 2 个 C 类 animal
    },
}
"""ABC 三版伪标签的合并权重策略。grade 由 source_tag 首字母大写决定。"""

MAX_C_ANIMAL_PER_IMAGE = 2
"""每张图 grade==C 且 train_label==animal 的最大保留数"""

GRADE_PRIORITY = {
    "A": 0,
    "B": 1,
    "C": 2,
    "rare_clean": 3,
}
"""同图同 label 去重时的保留优先级（数值越小越优先）。
A > B > C 只在同一个目标重复时生效；
不重复的 B/C 仍然可以补进 animal。"""

ANIMAL_SOURCE_PROMPTS = {
    "rabbit", "horse", "pig", "zebra", "duck",
    "lion", "tiger", "monkey", "giraffe",
    "bear", "wolf", "fox", "dog",
}
"""属于 animal 大类、需要应用 ABC 策略的 source prompt 集合"""

# ── prompt 策略（默认内置策略，可用 --policy-json 覆盖） ────────────────
PROMPT_POLICY: Dict[str, Dict[str, Any]] = {
    # ---- computer 系 ----
    "computer": {
        "use_as": "positive",
        "train_label": "computer",
        "threshold": 0.50,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["computer"],
        "allowed_datasets": ["data5", "data7"],
    },
    "laptop": {
        "use_as": "positive",
        "train_label": "computer",
        "threshold": 0.50,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["computer", "laptop", "monitor", "keyboard"],
        "allowed_datasets": ["data5", "data7"],
    },
    "monitor": {
        "use_as": "positive",
        "train_label": "computer",
        "threshold": 0.52,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["computer", "laptop", "monitor", "keyboard"],
        "allowed_datasets": ["data5", "data7"],
    },
    "keyboard": {
        "use_as": "positive",
        "train_label": "computer",
        "threshold": 0.50,
        "sample_weight": 2.5,
        "max_per_image": 2,
        "prompt_pool": ["computer", "keyboard"],
        "allowed_datasets": ["data5", "data7"],
    },
    # ---- trash can 系 ----
    "trash can": {
        "use_as": "positive",
        "train_label": "trash can",
        "threshold": 0.48,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["trash can"],
        "allowed_datasets": ["data5", "data7"],
    },
    "dustbin": {
        "use_as": "positive",
        "train_label": "trash can",
        "threshold": 0.48,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["trash can", "dustbin", "garbage bin"],
        "allowed_datasets": ["data5", "data7"],
    },
    "garbage bin": {
        "use_as": "positive",
        "train_label": "trash can",
        "threshold": 0.50,
        "sample_weight": 3.0,
        "max_per_image": 3,
        "prompt_pool": ["trash can", "dustbin", "garbage bin"],
        "allowed_datasets": ["data5", "data7"],
    },
    # ---- animal 系 ----
    "fox": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.56,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "fox"],
        "allowed_datasets": ["data6"],
    },
    "wolf": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.56,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "wolf"],
        "allowed_datasets": ["data6"],
    },
    "lion": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.55,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "lion"],
        "allowed_datasets": ["data6"],
    },
    "dog": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.56,
        "sample_weight": 1.2,
        "max_per_image": 3,
        "prompt_pool": ["animal", "dog"],
        "allowed_datasets": ["data6"],
    },
    "horse": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.58,
        "sample_weight": 1.2,
        "max_per_image": 3,
        "prompt_pool": ["animal", "horse"],
        "allowed_datasets": ["data6"],
    },
    "tiger": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.58,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "tiger"],
        "allowed_datasets": ["data6"],
    },
    "bear": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.58,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "bear"],
        "allowed_datasets": ["data6"],
    },
    "zebra": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.55,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "zebra"],
        "allowed_datasets": ["data6"],
    },
    "giraffe": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.55,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "giraffe"],
        "allowed_datasets": ["data6"],
    },
    "monkey": {
        "use_as": "positive",
        "train_label": "animal",
        "threshold": 0.55,
        "sample_weight": 1.3,
        "max_per_image": 3,
        "prompt_pool": ["animal", "monkey"],
        "allowed_datasets": ["data6"],
    },
    # ---- vehicle 系 ----
    "truck": {
        "use_as": "positive",
        "train_label": "car",
        "threshold": 0.65,
        "sample_weight": 1.2,
        "max_per_image": 3,
        "prompt_pool": ["car", "truck"],
        "allowed_datasets": [],
    },
    "bus": {
        "use_as": "positive",
        "train_label": "car",
        "threshold": 0.65,
        "sample_weight": 1.2,
        "max_per_image": 3,
        "prompt_pool": ["car", "bus"],
        "allowed_datasets": [],
    },
    # ---- pending / drop ----
    "screen": {"use_as": "pending"},
    "bin": {"use_as": "pending"},
    "duck": {"use_as": "pending"},
    "pig": {"use_as": "pending"},
    "rabbit": {"use_as": "pending"},
    "traffic light": {"use_as": "pending"},
    "pole": {"use_as": "pending"},
    "window": {"use_as": "pending"},
    "sign": {"use_as": "pending"},
    "motorcycle": {"use_as": "pending"},
    "bicycle": {"use_as": "pending"},
}
# ── 用户配置区结束 ──────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 依赖导入（不要改这里）
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# ---- 可选依赖 ----
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    mask_utils = None
    HAS_PYCOCOTOOLS = False

# --- 进度条 ---
try:
    from tqdm import tqdm as _tqdm

    HAS_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore[assignment]
    HAS_TQDM = False


def _progress(iterable, desc: str = "", unit: str = "it", **kwargs):
    """tqdm 的安全包装，缺失时退化为普通迭代。"""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, unit=unit, **kwargs)


# --- GPU / 深度学习框架 ---
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
    """在 GPU 上计算均值（列表较小时直接用 numpy）。"""
    if len(arr) == 0:
        return 0.0
    if len(arr) < 1000:
        return float(np.mean(arr))
    xp_mod = _xp()
    gpu_arr = xp_mod.asarray(arr, dtype=xp_mod.float32)
    return float(xp_mod.mean(gpu_arr))

SUPPORTED_BASE_FORMATS = {
    "simple_list": "list of {image_path, label, rle, score}",
    "labeled_prompts": "list of {image_path, prompts: {label: {hit, score, rle}}}",
}


# ═══════════════════════════════════════════════════════════════════════════════
# RLE 工具
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

    if HAS_PYCOCOTOOLS:
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
            pos = 0
            val = 0
            for run_len in runs:
                if run_len > 0:
                    if val == 1:
                        mask[pos : pos + run_len] = 1
                    pos += run_len
                val = 1 - val
            return mask.reshape((h, w), order="F")
        except Exception:
            pass

    return None


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """编码二值 mask -> RLE dict (pycocotools 格式)."""
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError("encode_rle requires pycocotools")
    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_f)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def rle_area(rle: Dict[str, Any]) -> int:
    """RLE 面积."""
    mask = decode_rle(rle)
    if mask is not None:
        return int(mask.sum())
    return 0


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_mask_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """containment = intersection / min(area_a, area_b)."""
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    if min(area_a, area_b) == 0:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / min(area_a, area_b))


def get_target_hw_from_base_entry(entry: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """从 base prompt-style entry 中推断目标 mask 尺寸 (H, W)。"""
    prompts = entry.get("prompts", {})
    if not isinstance(prompts, dict):
        return None
    # 优先 animal
    animal_info = prompts.get("animal", {})
    if isinstance(animal_info, dict):
        rle = animal_info.get("rle")
        if isinstance(rle, dict) and rle.get("size"):
            h, w = rle["size"]
            return int(h), int(w)
    # 再找其他 prompt
    for _, info in prompts.items():
        if not isinstance(info, dict):
            continue
        rle = info.get("rle")
        if isinstance(rle, dict) and rle.get("size"):
            h, w = rle["size"]
            return int(h), int(w)
    return None


def resize_mask_to_hw(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """将 mask 最近邻 resize 到目标 (H, W)。"""
    if not HAS_CV2:
        return mask
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape[:2] == (target_h, target_w):
        return mask.astype(np.uint8)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_tag_path(raw: str) -> Tuple[str, Path]:
    """解析 'tag=path' 或 'path'."""
    if "=" in raw:
        tag, path_str = raw.split("=", 1)
        return tag.strip(), Path(path_str.strip())
    else:
        path = Path(raw.strip())
        tag = path.stem.replace("pred_", "")
        return tag, path


def load_policy(policy_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """加载 prompt 策略。不传则使用默认 PROMPT_POLICY."""
    if not policy_json:
        return dict(PROMPT_POLICY)
    with open(policy_json, "r", encoding="utf-8-sig") as f:
        custom = json.load(f)
    # 深合并：自定义覆盖默认
    merged = dict(PROMPT_POLICY)
    for k, v in custom.items():
        if k in merged:
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def load_pred_jsons(raw_specs: Optional[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """加载多个 pred_*.json."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    if not raw_specs:
        return result
    for raw in raw_specs:
        tag, path = _resolve_tag_path(raw)
        if not path.exists():
            print(f"[WARN] pred json 不存在: {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            result[tag] = data
        elif isinstance(data, dict) and "entries" in data:
            result[tag] = data["entries"]
        else:
            print(f"[WARN] 无法识别 pred json 格式: {path}", file=sys.stderr)
            continue
        print(f"[INFO] 加载 pred json: tag={tag}, entries={len(result[tag])}, path={path}")
    return result


def load_base_pseudo_json(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    """加载主伪标签 JSON。尝试自动识别格式。"""
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        if "entries" in data:
            entries = data["entries"]
        elif "annotations" in data:
            entries = data["annotations"]
        else:
            print(f"[WARN] 无法识别 base pseudo json 格式，keys={list(data.keys())[:5]}", file=sys.stderr)
            print(f"[INFO] 预期格式之一: {json.dumps(SUPPORTED_BASE_FORMATS, indent=2)}", file=sys.stderr)
            return None
    else:
        return None

    print(f"[INFO] 加载 base pseudo json: entries={len(entries)}, path={path}")
    return entries


def parse_base_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    尝试从 base pseudo entry 中提取标准字段。
    返回 {"image_path": str, "label": str, "rle": dict, "score": float} 或 None.
    """
    image_path = entry.get("image_path", "").replace("\\", "/")
    label = entry.get("label") or entry.get("train_label") or entry.get("category", "")
    rle = entry.get("rle") or entry.get("segmentation")
    score = float(entry.get("score", 0.5))

    if not image_path or not label:
        # 尝试 prompts 格式
        prompts = entry.get("prompts", {})
        if isinstance(prompts, dict):
            results = []
            for pname, pdata in prompts.items():
                if isinstance(pdata, dict) and pdata.get("hit"):
                    results.append(
                        {
                            "image_path": image_path or entry.get("image_path", ""),
                            "label": pname,
                            "rle": pdata.get("rle"),
                            "score": float(pdata.get("score", 0.5)),
                        }
                    )
            if results:
                return results[0]  # 只返回第一个
        return None
    return {"image_path": image_path, "label": label, "rle": rle, "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# 策略匹配
# ═══════════════════════════════════════════════════════════════════════════════


def get_policy_for_prompt(
    prompt: str, policy: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """查找 prompt 对应的策略。找不到则返回 None."""
    return policy.get(prompt)


def is_validation_image(image_path: str) -> bool:
    """判断是否为验证集图片."""
    lowered = image_path.lower()
    val_markers = ["/val_", "\\val_", "/val/", "\\val\\", "/validation", "\\validation",
                    "val_changwei", "val_animal"]
    for marker in val_markers:
        if marker in lowered:
            return True
    basename = os.path.basename(image_path).lower()
    if basename.startswith("val_"):
        return True
    return False


def check_split_filter(image_path: str, split_filters: Optional[List[str]]) -> bool:
    """检查 image_path 是否匹配 split_filter."""
    if not split_filters:
        return True
    lowered = image_path.lower()
    return any(f.lower() in lowered for f in split_filters)


def check_allowed_datasets(
    image_path: str, allowed_datasets: List[str]
) -> bool:
    """检查 image_path 是否属于 allowed_datasets.
    空列表表示允许所有数据集."""
    if not allowed_datasets:
        return True
    lowered = image_path.lower()
    return any(ds.lower() in lowered for ds in allowed_datasets)


# ═══════════════════════════════════════════════════════════════════════════════
# 主处理流水线
# ═══════════════════════════════════════════════════════════════════════════════


def process_candidates(
    all_preds: Dict[str, List[Dict[str, Any]]],
    policy: Dict[str, Dict[str, Any]],
    image_root: Path,
    include_val: bool,
    split_filters: Optional[List[str]],
    min_area: int = MIN_AREA,
    default_max_area_ratio: float = MAX_AREA_RATIO,
    relaxed_max_area_ratio: float = RELAXED_MAX_AREA_RATIO,
) -> Tuple[
    List[Dict[str, Any]],  # kept
    List[Dict[str, Any]],  # hard_negative
    List[Dict[str, Any]],  # rejected
    Dict[str, Dict[str, int]],  # merge_stats
]:
    """主流水线：遍历所有 pred entries，筛选 -> 面积过滤 -> 返回候选."""
    kept: List[Dict[str, Any]] = []
    hard_negative: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    merge_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # source_prompt -> {stat_key: count}

    for source_tag, entries in _progress(
        list(all_preds.items()), desc="筛选候选", unit="src"
    ):
        # 根据 source_tag 推导 grade（A/B/C），用于 ABC 合并策略
        grade = str(source_tag).upper()
        if grade not in ("A", "B", "C"):
            grade = "rare_clean"

        for entry in _progress(entries, desc=f"  {source_tag}", unit="img", leave=False):
            image_path = entry.get("image_path", "").replace("\\", "/")

            # validation 过滤
            if not include_val and is_validation_image(image_path):
                continue

            # split 过滤
            if not check_split_filter(image_path, split_filters):
                continue

            for prompt, pdata in entry.get("prompts", {}).items():
                if not isinstance(pdata, dict):
                    continue
                if not pdata.get("hit"):
                    continue

                score = float(pdata.get("score", 0))
                instances = int(pdata.get("instances", 0))
                rle = pdata.get("rle")
                has_selected_scores = "selected_scores" in pdata

                # 查找策略
                pol = get_policy_for_prompt(prompt, policy)
                if pol is None:
                    merge_stats[prompt]["total_hits"] += 1
                    merge_stats[prompt]["rejected_pending_or_drop"] += 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": "no_policy_default_pending",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "rle": rle,
                        }
                    )
                    continue

                use_as = pol.get("use_as", "pending")
                train_label = pol.get("train_label", prompt)
                threshold = float(pol.get("threshold", 0.5))
                sample_weight = float(pol.get("sample_weight", 1.0))
                allowed_datasets = pol.get("allowed_datasets", [])
                prompt_pool = pol.get("prompt_pool", [prompt])

                merge_stats[prompt]["total_hits"] += 1

                # use_as 过滤
                if use_as in ("pending", "drop"):
                    merge_stats[prompt]["rejected_pending_or_drop"] += 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": f"use_as={use_as}",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "rle": rle,
                        }
                    )
                    continue

                # allowed_datasets 过滤
                if not check_allowed_datasets(image_path, allowed_datasets):
                    merge_stats[prompt]["rejected_bad_dataset"] += 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": "bad_dataset",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "rle": rle,
                        }
                    )
                    continue

                # score 过滤
                if score < threshold:
                    merge_stats[prompt]["rejected_low_score"] += 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": "low_score",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "rle": rle,
                        }
                    )
                    continue

                # 旧版无 top-k 的 fragile prompt 丢弃
                if not has_selected_scores and prompt in FRAGILE_PROMPTS:
                    merge_stats[prompt]["rejected_too_many_instances_old_source"] += 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": "fragile_old_source_no_topk",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "rle": rle,
                        }
                    )
                    continue

                # 旧版无 top-k 且 instances 很高：降低权重
                if not has_selected_scores and instances > 5:
                    sample_weight *= 0.5
                    merge_stats[prompt]["rejected_too_many_instances_old_source"] += 1

                # 面积过滤
                area = rle_area(rle) if rle else 0
                if area < min_area:
                    merge_stats[prompt]["rejected_too_small"] = merge_stats[prompt].get(
                        "rejected_too_small", 0
                    ) + 1
                    rejected.append(
                        {
                            "image_path": image_path,
                            "source_prompt": prompt,
                            "reason": "too_small",
                            "source_tag": source_tag,
                            "score": score,
                            "instances": instances,
                            "area": area,
                            "rle": rle,
                        }
                    )
                    continue

                # 面积上限：animal/car 放宽
                if train_label in ("animal", "car"):
                    max_ratio = relaxed_max_area_ratio
                else:
                    max_ratio = default_max_area_ratio

                if rle and rle.get("size"):
                    img_h, img_w = rle["size"]
                    img_area = img_h * img_w
                    if img_area > 0 and area / img_area > max_ratio:
                        merge_stats[prompt]["rejected_too_large"] = merge_stats[prompt].get(
                            "rejected_too_large", 0
                        ) + 1
                        rejected.append(
                            {
                                "image_path": image_path,
                                "source_prompt": prompt,
                                "reason": "too_large",
                                "source_tag": source_tag,
                                "score": score,
                                "instances": instances,
                                "area": area,
                                "area_ratio": round(area / img_area, 4),
                                "rle": rle,
                            }
                        )
                        continue

                # positive_low_weight
                if use_as == "positive_low_weight":
                    sample_weight *= 0.5
                    threshold = max(threshold + 0.05, threshold)
                    if score < threshold:
                        merge_stats[prompt]["rejected_low_score"] += 1
                        rejected.append(
                            {
                                "image_path": image_path,
                                "source_prompt": prompt,
                                "reason": "low_score_after_low_weight_adjust",
                                "source_tag": source_tag,
                                "score": score,
                                "instances": instances,
                                "rle": rle,
                            }
                        )
                        continue

                # ── ABC 合并策略：仅对 animal 类 prompt 生效 ──────────
                if (
                    train_label == "animal"
                    and prompt in ANIMAL_SOURCE_PROMPTS
                    and grade in ABC_GRADE_POLICY
                ):
                    abc_cfg = ABC_GRADE_POLICY[grade]
                    if not abc_cfg.get("use", True):
                        merge_stats[prompt]["rejected_pending_or_drop"] += 1
                        rejected.append(
                            {
                                "image_path": image_path,
                                "source_prompt": prompt,
                                "reason": f"abc_grade_{grade}_disabled",
                                "source_tag": source_tag,
                                "score": score,
                                "instances": instances,
                                "rle": rle,
                            }
                        )
                        continue

                    sample_weight = sample_weight * float(abc_cfg.get("weight_mul", 1.0))

                    prompt_mode = abc_cfg.get("prompt_mode", "animal_only")
                    if prompt_mode == "animal_plus_source":
                        prompt_pool = ["animal", prompt]
                    else:
                        prompt_pool = ["animal"]

                candidate = {
                    "image_path": image_path,
                    "train_label": train_label,
                    "source_prompt": prompt,
                    "source_prompts": [prompt],
                    "prompt_pool": prompt_pool,
                    "use_as": use_as,
                    "score": score,
                    "instances": instances,
                    "rle": rle,
                    "area": area,
                    "sample_weight": sample_weight,
                    "source_tag": source_tag,
                    "source_file": source_tag,
                    "grade": grade,
                    "has_selected_scores": has_selected_scores,
                }

                if use_as == "hard_negative":
                    hard_negative.append(candidate)
                    merge_stats[prompt]["kept_as_hard_negative"] = merge_stats[prompt].get(
                        "kept_as_hard_negative", 0
                    ) + 1
                else:
                    kept.append(candidate)
                    merge_stats[prompt]["kept_after_area"] = merge_stats[prompt].get(
                        "kept_after_area", 0
                    ) + 1

    return kept, hard_negative, rejected, dict(merge_stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 去重
# ═══════════════════════════════════════════════════════════════════════════════


def dedup_candidates(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    同图同 train_label 去重。
    排序优先级：grade(A>B>C) > positive > 新版 > score高 > area合理.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        key = f"{c['image_path']}|||{c['train_label']}"
        groups[key].append(c)

    kept = []
    deduped = []

    for group_key, items in _progress(
        list(groups.items()), desc="同图同类去重", unit="grp"
    ):
        items.sort(
            key=lambda x: (
                GRADE_PRIORITY.get(x.get("grade", "rare_clean"), 3),
                0 if x.get("use_as", "positive") == "positive" else 1,
                0 if x.get("has_selected_scores", False) else 1,
                -x.get("score", 0),
                -x.get("area", 0),
            )
        )

        kept_items = []
        for item in items:
            mask_a = decode_rle(item.get("rle"))
            if mask_a is None:
                kept_items.append(item)
                continue

            is_dup = False
            for existing in kept_items:
                mask_b = decode_rle(existing.get("rle"))
                if mask_b is None:
                    continue
                iou = compute_mask_iou(mask_a, mask_b)
                containment = compute_mask_containment(mask_a, mask_b)
                if iou > iou_threshold or containment > containment_threshold:
                    is_dup = True
                    deduped.append(
                        {
                            **item,
                            "dedup_reason": f"iou={iou:.3f}_containment={containment:.3f}",
                            "dedup_against": existing.get("image_path", ""),
                        }
                    )
                    break

            if not is_dup:
                kept_items.append(item)

        # 同图同 label 最多保留数量
        train_label = items[0].get("train_label", "other")
        max_keep = MAX_PER_TRAIN_LABEL.get(train_label, DEFAULT_MAX_PER_LABEL)
        if len(kept_items) > max_keep:
            overflow = kept_items[max_keep:]
            kept_items = kept_items[:max_keep]
            for item in overflow:
                deduped.append(
                    {
                        **item,
                        "dedup_reason": f"exceed_max_per_label({max_keep})",
                    }
                )

        kept.extend(kept_items)

    return kept, deduped


def limit_c_animal_per_image(
    candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """对每张图，限制 grade==C 且 train_label==animal 的候选不超过 MAX_C_ANIMAL_PER_IMAGE."""
    # 分离 C 类 animal 和其他
    c_animal_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    other_items: List[Dict[str, Any]] = []

    for c in candidates:
        if c.get("grade") == "C" and c.get("train_label") == "animal":
            c_animal_items[c["image_path"]].append(c)
        else:
            other_items.append(c)

    kept = list(other_items)
    deduped: List[Dict[str, Any]] = []

    for image_path, items in c_animal_items.items():
        # 按 score 降序
        items.sort(key=lambda x: x.get("score", 0), reverse=True)
        max_keep = MAX_C_ANIMAL_PER_IMAGE
        if len(items) > max_keep:
            kept.extend(items[:max_keep])
            for item in items[max_keep:]:
                deduped.append(
                    {
                        **item,
                        "dedup_reason": f"exceed_max_c_animal_per_image({max_keep})",
                    }
                )
        else:
            kept.extend(items)

    return kept, deduped


# ═══════════════════════════════════════════════════════════════════════════════
# 与 base pseudo 合并
# ═══════════════════════════════════════════════════════════════════════════════


def merge_with_base(
    rare_candidates: List[Dict[str, Any]],
    base_entries: List[Dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """将 clean rare 与 base pseudo 合并去重.

    注意：prompt-style base pseudo（prompts: {label: {hit, score, rle}} 格式）
    当前不建议用本脚本直接合并，因为 parse_base_entry 只取第一个 hit。
    建议先跑 analyze_rare_pseudo.py 人工确认 prompt 质量，再决定合并策略。
    """
    merged = []
    merge_counts = {"rare_only": 0, "base_only": 0, "both_kept_rare": 0, "both_kept_base": 0}

    base_candidates = []
    for entry in base_entries:
        parsed = parse_base_entry(entry)
        if parsed:
            parsed["source_tag"] = "base"
            parsed["grade"] = "base_pseudo"
            parsed["sample_weight"] = 1.0
            parsed["prompt_pool"] = [parsed.get("label", "")]
            parsed["source_prompt"] = parsed.get("label", "")
            parsed["source_prompts"] = [parsed.get("label", "")]
            parsed["instances"] = 1
            parsed["area"] = rle_area(parsed.get("rle"))
            parsed["has_selected_scores"] = False
            base_candidates.append(parsed)

    base_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in base_candidates:
        key = f"{c['image_path']}|||{c.get('label', c.get('train_label', ''))}"
        base_groups[key].append(c)

    rare_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in rare_candidates:
        key = f"{c['image_path']}|||{c.get('train_label', '')}"
        rare_groups[key].append(c)

    all_keys = set(base_groups.keys()) | set(rare_groups.keys())

    for key in all_keys:
        rares = rare_groups.get(key, [])
        bases = base_groups.get(key, [])

        if rares and not bases:
            merged.extend(rares)
            merge_counts["rare_only"] += len(rares)
        elif bases and not rares:
            merged.extend(bases)
            merge_counts["base_only"] += len(bases)
        else:
            kept_rare = []
            kept_base = list(bases)
            for rare in rares:
                mask_r = decode_rle(rare.get("rle"))
                if mask_r is None:
                    kept_rare.append(rare)
                    continue
                is_dup = False
                for base in kept_base:
                    mask_b = decode_rle(base.get("rle"))
                    if mask_b is None:
                        continue
                    iou = compute_mask_iou(mask_r, mask_b)
                    containment = compute_mask_containment(mask_r, mask_b)
                    if iou > iou_threshold or containment > containment_threshold:
                        train_label = rare.get("train_label", "")
                        if train_label in ("computer", "trash can") and rare.get("score", 0) > 0.7:
                            kept_rare.append(rare)
                            kept_base = [
                                b for b in kept_base
                                if b.get("image_path") != base.get("image_path")
                                or b.get("rle") != base.get("rle")
                            ]
                            merge_counts["both_kept_rare"] += 1
                        else:
                            merge_counts["both_kept_base"] += 1
                        is_dup = True
                        break
                if not is_dup:
                    kept_rare.append(rare)

            merged.extend(kept_rare)
            merged.extend(kept_base)

    return merged, merge_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 新合并模式：只把 clean rare animal 合进 base pseudo 的 prompts["animal"]
# ═══════════════════════════════════════════════════════════════════════════════


def merge_animal_into_base_prompt_style(
    base_entries: List[Dict[str, Any]],
    rare_candidates: List[Dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> Tuple[
    List[Dict[str, Any]],  # merged_entries
    Dict[str, Any],  # summary_stats
    Dict[str, Dict[str, int]],  # grade_stats
    Dict[str, Dict[str, int]],  # prompt_stats
]:
    """只把 clean rare animal 合进 base pseudo 的 prompts["animal"]，
    其他类（person/tree/building/car/computer）原样保留。

    返回:
        merged_entries, summary_stats, grade_stats, prompt_stats
    """
    import copy

    # ── 统计初始化 ──────────────────────────────────────────────────────
    summary_stats: Dict[str, Any] = {
        "base_images": len(base_entries),
        "rare_candidates_total": len(rare_candidates),
        "rare_after_filter": 0,
        "rare_kept_merged": 0,
        "rare_dup_with_base": 0,
        "rare_dup_with_rare": 0,
        "rare_decode_fail": 0,
        "rare_low_score": 0,
        "rare_too_small": 0,
        "rare_too_large": 0,
        "rare_not_in_base": 0,
        "base_animal_hit_images": 0,
        "images_with_animal_updated": 0,
        "images_newly_got_animal": 0,
        "total_added_area": 0,
        "rare_resized_to_base": 0,
        "rare_shape_mismatch_after_resize": 0,
    }

    grade_stats: Dict[str, Dict[str, int]] = {}
    for g in ["A", "B", "C"]:
        grade_stats[g] = {"raw": 0, "after_filter": 0, "kept_merged": 0,
                          "dup_with_base": 0, "dup_with_rare": 0, "rejected": 0}
    grade_stats["other"] = {"raw": 0, "after_filter": 0, "kept_merged": 0,
                            "dup_with_base": 0, "dup_with_rare": 0, "rejected": 0}

    prompt_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"raw": 0, "after_filter": 0, "kept_merged": 0,
                 "dup_with_base": 0, "dup_with_rare": 0, "rejected": 0}
    )

    # ── 预处理 rare candidates 统计 ─────────────────────────────────────
    rare_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in rare_candidates:
        source_prompt = str(c.get("source_prompt", ""))
        grade = str(c.get("grade", "other"))
        prompt_stats[source_prompt]["raw"] += 1
        gk = grade if grade in grade_stats else "other"
        grade_stats[gk]["raw"] += 1

        if c.get("train_label") != "animal":
            prompt_stats[source_prompt]["rejected"] += 1
            grade_stats[gk]["rejected"] += 1
            continue
        if c.get("rle") is None:
            prompt_stats[source_prompt]["rejected"] += 1
            grade_stats[gk]["rejected"] += 1
            continue

        prompt_stats[source_prompt]["after_filter"] += 1
        grade_stats[gk]["after_filter"] += 1
        summary_stats["rare_after_filter"] += 1

        image_path = str(c.get("image_path", "")).replace("\\", "/")
        rare_by_image[image_path].append(c)

    # ── 1. 深拷贝 base，不改原对象 ──────────────────────────────────────
    merged_entries = copy.deepcopy(base_entries)

    # ── 2. 建 base 索引 ─────────────────────────────────────────────────
    base_index: Dict[str, Dict[str, Any]] = {}
    for entry in merged_entries:
        image_path = entry.get("image_path", "").replace("\\", "/")
        base_index[image_path] = entry

    # ── 3. 逐图合并 ─────────────────────────────────────────────────────
    for image_path, rare_list in _progress(
        sorted(rare_by_image.items()), desc="合并 rare animal → base", unit="img"
    ):
        entry = base_index.get(image_path)
        if entry is None:
            summary_stats["rare_not_in_base"] += len(rare_list)
            for c in rare_list:
                sp = str(c.get("source_prompt", ""))
                gk = str(c.get("grade", "other"))
                gk = gk if gk in grade_stats else "other"
                prompt_stats[sp]["rejected"] += 1
                grade_stats[gk]["rejected"] += 1
            continue

        # 确定目标尺寸（H, W）
        target_hw = get_target_hw_from_base_entry(entry)

        prompts = entry.setdefault("prompts", {})
        animal_info = prompts.get("animal", {}) if isinstance(prompts, dict) else {}

        base_animal_hit = animal_info.get("hit", False) if isinstance(animal_info, dict) else False
        base_animal_rle = animal_info.get("rle") if isinstance(animal_info, dict) else None
        base_animal_score = float(animal_info.get("score", 0)) if isinstance(animal_info, dict) else 0.0

        base_mask = None
        base_animal_area = 0
        if base_animal_hit and base_animal_rle is not None:
            base_mask = decode_rle(base_animal_rle)
            if base_mask is not None:
                base_animal_area = int(base_mask.sum())
                summary_stats["base_animal_hit_images"] += 1
                target_hw = base_mask.shape  # 强制以 base animal mask 尺寸为准

        existing_masks: List[np.ndarray] = []
        if base_mask is not None:
            existing_masks.append(base_mask)

        kept_rare_masks: List[np.ndarray] = []
        kept_rare_sources: List[str] = []
        kept_prompt_pools: List[str] = []
        max_rare_score = base_animal_score
        img_rare_kept = 0

        # A > B > C 优先，相同 grade 再比 score/area
        rare_list.sort(
            key=lambda x: (
                GRADE_PRIORITY.get(x.get("grade", "rare_clean"), 3),
                -float(x.get("score", 0)),
                -int(x.get("area", 0)),
            )
        )

        for c in rare_list:
            sp = str(c.get("source_prompt", ""))
            gk = str(c.get("grade", "other"))
            gk = gk if gk in grade_stats else "other"

            rare_mask = decode_rle(c.get("rle"))
            if rare_mask is None:
                summary_stats["rare_decode_fail"] += 1
                prompt_stats[sp]["rejected"] += 1
                grade_stats[gk]["rejected"] += 1
                continue

            # shape 不一致时 resize 到 base 目标尺寸
            if target_hw is not None and rare_mask.shape != target_hw:
                rare_mask = resize_mask_to_hw(rare_mask, target_hw)
                summary_stats["rare_resized_to_base"] += 1

            # resize 后再检查
            if target_hw is not None and rare_mask.shape != target_hw:
                summary_stats["rare_shape_mismatch_after_resize"] += 1
                prompt_stats[sp]["rejected"] += 1
                grade_stats[gk]["rejected"] += 1
                continue

            rare_score = float(c.get("score", 0))

            is_dup = False
            for existing_mask in existing_masks:
                # 尺寸不一致则跳过该 existing mask
                if rare_mask.shape != existing_mask.shape:
                    continue
                iou = compute_mask_iou(rare_mask, existing_mask)
                containment = compute_mask_containment(rare_mask, existing_mask)
                if iou > iou_threshold or containment > containment_threshold:
                    is_dup = True
                    if existing_mask is base_mask:
                        summary_stats["rare_dup_with_base"] += 1
                        prompt_stats[sp]["dup_with_base"] += 1
                        grade_stats[gk]["dup_with_base"] += 1
                    else:
                        summary_stats["rare_dup_with_rare"] += 1
                        prompt_stats[sp]["dup_with_rare"] += 1
                        grade_stats[gk]["dup_with_rare"] += 1
                    break

            if not is_dup:
                kept_rare_masks.append(rare_mask)
                existing_masks.append(rare_mask)
                kept_rare_sources.append(sp)
                # 收集 prompt_pool
                pp = c.get("prompt_pool", [])
                if isinstance(pp, list):
                    kept_prompt_pools.extend(pp)
                max_rare_score = max(max_rare_score, rare_score)
                img_rare_kept += 1
                summary_stats["rare_kept_merged"] += 1
                prompt_stats[sp]["kept_merged"] += 1
                grade_stats[gk]["kept_merged"] += 1

        # 更新 prompts["animal"]（仅当有新增 rare 时才重写）
        if kept_rare_masks:
            if base_mask is not None:
                all_masks = [base_mask] + kept_rare_masks
            else:
                all_masks = kept_rare_masks
            final_union = np.zeros_like(all_masks[0])
            for m in all_masks:
                final_union = np.logical_or(final_union, m)
            final_union = final_union.astype(np.uint8)
            final_animal_area = int(final_union.sum())
            added_area = final_animal_area - base_animal_area
            summary_stats["total_added_area"] += max(added_area, 0)

            if base_animal_hit:
                summary_stats["images_with_animal_updated"] += 1
            else:
                summary_stats["images_newly_got_animal"] += 1

            # 保留 base animal 原有字段，再叠加新增信息
            animal_out = dict(animal_info) if isinstance(animal_info, dict) else {}
            animal_out.update({
                "hit": True,
                "score": round(max_rare_score, 4),
                "rle": encode_rle(final_union),
                "merged_from": "base_plus_rare_animal" if base_animal_hit else "rare_animal_only",
                "rare_added": img_rare_kept,
                "rare_sources": sorted(set(kept_rare_sources)),
                "prompt_pool": sorted(set(kept_prompt_pools)),
            })
            prompts["animal"] = animal_out
        # 如果没有新增 rare，保持 base animal 原样不动

    return merged_entries, summary_stats, grade_stats, prompt_stats


def write_summary_md(
    summary_stats: Dict[str, Any],
    grade_stats: Dict[str, Dict[str, int]],
    prompt_stats: Dict[str, Dict[str, int]],
    output_path: Path,
) -> Path:
    """输出 merge_summary.md."""
    lines: List[str] = []

    lines.append("# Animal Rare Merge Summary")
    lines.append("")

    lines.append("## 1. 去重取并规则")
    lines.append("")
    lines.append("本次只将细分动物伪标签映射到主类 animal，并合并进 base label 的 prompts[\"animal\"]。")
    lines.append("base label 中的 person / tree / building / car / computer 完全原样保留。")
    lines.append("")
    lines.append("映射规则：")
    lines.append(f"- 动物类 source prompts: {', '.join(sorted(ANIMAL_SOURCE_PROMPTS))} -> animal")
    lines.append("- A 类：映射为 animal，可保留 source_prompt 信息，权重 x1.0")
    lines.append("- B 类：映射为 animal，低权重候选，权重 x0.4")
    lines.append("- C 类：映射为 animal，极低权重候选，权重 x0.15，每图最多 2 个")
    lines.append("")
    lines.append("去重规则：")
    lines.append("- rare animal 先和 base animal 比较")
    lines.append("- 再和已接收的 rare animal 比较")
    lines.append(f"- 如果 IoU > {IOU_THRESHOLD}，认为重复，丢弃 rare，保留 base/已接收候选")
    lines.append(f"- 如果 containment > {CONTAINMENT_THRESHOLD}，认为重复，丢弃 rare，保留 base/已接收候选")
    lines.append("- containment = intersection / min(area_a, area_b)")
    lines.append("- 不重复的 rare animal 与 base animal 做像素级 OR，写回 prompts[\"animal\"]")
    lines.append("")
    lines.append("ABC 优先级说明：")
    lines.append("- A > B > C 只用于同一个目标重复时的保留优先级")
    lines.append("- A 不是召回全集，A 漏检的目标仍允许由 B/C 补充")
    lines.append("- 若 B/C 与 base animal 或 A 候选重复，则丢弃 B/C")
    lines.append("- 若 B/C 不重复，则作为 animal 补充合入 prompts[\"animal\"]")
    lines.append("")
    lines.append("面积过滤：")
    lines.append(f"- min_area = {MIN_AREA}")
    lines.append(f"- max_area_ratio = {RELAXED_MAX_AREA_RATIO}")
    lines.append("")

    lines.append("## 2. 总体统计")
    lines.append("")
    lines.append("| item | count |")
    lines.append("|---|---:|")
    lines.append(f"| base images | {summary_stats['base_images']} |")
    lines.append(f"| rare candidates total | {summary_stats['rare_candidates_total']} |")
    lines.append(f"| rare candidates after policy/filter | {summary_stats['rare_after_filter']} |")
    lines.append(f"| rare kept and merged | {summary_stats['rare_kept_merged']} |")
    lines.append(f"| rare dropped by duplicate with base | {summary_stats['rare_dup_with_base']} |")
    lines.append(f"| rare dropped by duplicate with rare | {summary_stats['rare_dup_with_rare']} |")
    lines.append(f"| rare dropped by decode fail | {summary_stats['rare_decode_fail']} |")
    lines.append(f"| rare dropped (not in base) | {summary_stats['rare_not_in_base']} |")
    lines.append(f"| rare resized to base size | {summary_stats.get('rare_resized_to_base', 0)} |")
    lines.append(f"| rare shape mismatch after resize | {summary_stats.get('rare_shape_mismatch_after_resize', 0)} |")
    lines.append(f"| images with animal updated | {summary_stats['images_with_animal_updated']} |")
    lines.append(f"| images newly got animal | {summary_stats['images_newly_got_animal']} |")
    lines.append("")

    lines.append("## 3. ABC 统计")
    lines.append("")
    lines.append("| grade | raw | after filter | kept merged | dup with base | dup with rare | rejected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for g in ["A", "B", "C"]:
        s = grade_stats.get(g, {})
        lines.append(f"| {g} | {s.get('raw', 0)} | {s.get('after_filter', 0)} | "
                     f"{s.get('kept_merged', 0)} | {s.get('dup_with_base', 0)} | "
                     f"{s.get('dup_with_rare', 0)} | {s.get('rejected', 0)} |")
    lines.append("")

    lines.append("## 4. Prompt 统计")
    lines.append("")
    lines.append("| source_prompt | raw | after filter | kept merged | dup with base | dup with rare | rejected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sp in sorted(prompt_stats.keys()):
        s = prompt_stats[sp]
        lines.append(f"| {sp} | {s.get('raw', 0)} | {s.get('after_filter', 0)} | "
                     f"{s.get('kept_merged', 0)} | {s.get('dup_with_base', 0)} | "
                     f"{s.get('dup_with_rare', 0)} | {s.get('rejected', 0)} |")
    lines.append("")

    lines.append("## 5. 更新影响")
    lines.append("")
    base_animal_hit_images = summary_stats.get("base_animal_hit_images", 0)
    lines.append("| item | count |")
    lines.append("|---|---:|")
    lines.append(f"| base animal hit images | {base_animal_hit_images} |")
    lines.append(f"| rare added to existing animal images | {summary_stats['images_with_animal_updated']} |")
    lines.append(f"| rare created new animal images | {summary_stats['images_newly_got_animal']} |")
    lines.append(f"| total added animal area | {summary_stats['total_added_area']} |")
    updated_count = summary_stats['images_with_animal_updated'] + summary_stats['images_newly_got_animal']
    avg_added = (summary_stats['total_added_area'] / max(updated_count, 1))
    lines.append(f"| avg added area per updated image | {avg_added:.1f} |")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OUTPUT] {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════════════════════


def write_merge_report(
    merge_stats: Dict[str, Dict[str, int]],
    output_dir: Path,
    policy: Dict[str, Dict[str, Any]],
) -> Path:
    """输出 merge_report.csv."""
    path = output_dir / "merge_report.csv"
    all_prompts = sorted(merge_stats.keys())
    if not all_prompts:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("no_data\n")
        return path

    columns = [
        "source_prompt",
        "train_label",
        "use_as",
        "total_hits",
        "kept_after_area",
        "kept_after_dedup",
        "rejected_low_score",
        "rejected_bad_dataset",
        "rejected_too_large",
        "rejected_too_many_instances_old_source",
        "rejected_pending_or_drop",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for prompt in all_prompts:
            s = merge_stats[prompt]
            pol = policy.get(prompt, {})
            row = [
                prompt,
                pol.get("train_label", prompt),
                pol.get("use_as", "pending"),
                s.get("total_hits", 0),
                s.get("kept_after_area", 0),
                s.get("kept_after_dedup", 0),
                s.get("rejected_low_score", 0),
                s.get("rejected_bad_dataset", 0),
                s.get("rejected_too_large", 0),
                s.get("rejected_too_many_instances_old_source", 0),
                s.get("rejected_pending_or_drop", 0),
            ]
            writer.writerow(row)

    print(f"[OUTPUT] {path}")
    return path


def write_policy_report(policy: Dict[str, Dict[str, Any]], output_dir: Path) -> Path:
    """输出 prompt_policy_report.csv."""
    path = output_dir / "prompt_policy_report.csv"
    columns = [
        "prompt",
        "use_as",
        "train_label",
        "threshold",
        "sample_weight",
        "max_per_image",
        "allowed_datasets",
        "prompt_pool",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for prompt in sorted(policy.keys()):
            p = policy[prompt]
            writer.writerow(
                [
                    prompt,
                    p.get("use_as", ""),
                    p.get("train_label", prompt),
                    p.get("threshold", ""),
                    p.get("sample_weight", ""),
                    p.get("max_per_image", ""),
                    ", ".join(p.get("allowed_datasets", [])),
                    ", ".join(p.get("prompt_pool", [prompt])),
                ]
            )

    print(f"[OUTPUT] {path}")
    return path


def write_dedup_report(
    deduped: List[Dict[str, Any]],
    after_count: int,
    output_dir: Path,
) -> Path:
    """输出 dedup_report.csv."""
    path = output_dir / "dedup_report.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "source_prompt",
                "train_label",
                "score",
                "dedup_reason",
                "source_tag",
            ]
        )
        for item in deduped[:1000]:  # truncate
            writer.writerow(
                [
                    item.get("image_path", ""),
                    item.get("source_prompt", ""),
                    item.get("train_label", ""),
                    item.get("score", 0),
                    item.get("dedup_reason", ""),
                    item.get("source_tag", ""),
                ]
            )

    print(f"[OUTPUT] {path}  (deduped={after_count})")
    return path


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    """写 JSONL 文件."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"[OUTPUT] {path}  ({len(items)} lines)")


def write_manifest_json(path: Path, items: List[Dict[str, Any]]) -> None:
    """写完整 manifest JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {path}  ({len(items)} entries)")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def run(args: argparse.Namespace) -> None:
    """主流程：根据命令行参数更新配置并运行全部分析."""
    global IMAGE_ROOT, OUTPUT_DIR, INCLUDE_VAL, SPLIT_FILTER, DRY_RUN
    global MIN_AREA, MAX_AREA_RATIO, RELAXED_MAX_AREA_RATIO
    global IOU_THRESHOLD, CONTAINMENT_THRESHOLD

    # 用命令行参数覆盖用户配置区默认值
    if hasattr(args, "image_root") and args.image_root is not None:
        IMAGE_ROOT = args.image_root
    if hasattr(args, "output_dir") and args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if hasattr(args, "include_val"):
        INCLUDE_VAL = args.include_val
    if hasattr(args, "split_filter"):
        SPLIT_FILTER = args.split_filter
    if hasattr(args, "dry_run"):
        DRY_RUN = args.dry_run
    if hasattr(args, "min_area"):
        MIN_AREA = args.min_area
    if hasattr(args, "max_area_ratio"):
        MAX_AREA_RATIO = args.max_area_ratio
    if hasattr(args, "relaxed_max_area_ratio"):
        RELAXED_MAX_AREA_RATIO = args.relaxed_max_area_ratio
    if hasattr(args, "iou_threshold"):
        IOU_THRESHOLD = args.iou_threshold
    if hasattr(args, "containment_threshold"):
        CONTAINMENT_THRESHOLD = args.containment_threshold

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU 状态
    backend = "cpu"
    if HAS_CUPY:
        backend = "cupy"
    elif HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda":
        backend = "torch"
    print(f"[SYS] GPU: {backend}  tqdm: {'✓' if HAS_TQDM else '✗'}  pycocotools: {'✓' if HAS_PYCOCOTOOLS else '✗'}")

    # 1. 加载 policy
    policy_json = getattr(args, "policy_json", None)
    if policy_json is None:
        policy_json = POLICY_JSON
    policy = load_policy(policy_json)
    print(f"[INFO] 策略已加载: {len(policy)} prompts")

    # 2. 加载 rare pred jsons
    rare_pred_jsons = getattr(args, "rare_pred_jsons", None)
    if rare_pred_jsons is None:
        rare_pred_jsons = RARE_PRED_JSONS if RARE_PRED_JSONS else None
    all_preds = load_pred_jsons(rare_pred_jsons)
    if not all_preds:
        print("[ERROR] 至少需要传入一个 rare pred json", file=sys.stderr)
        sys.exit(1)

    # 3. 流水线筛选
    print(f"\n[INFO] 开始筛选候选...")
    t0 = time.time()
    kept, hard_negative, rejected, merge_stats = process_candidates(
        all_preds=all_preds,
        policy=policy,
        image_root=IMAGE_ROOT,
        include_val=INCLUDE_VAL,
        split_filters=SPLIT_FILTER,
        min_area=MIN_AREA,
        default_max_area_ratio=MAX_AREA_RATIO,
        relaxed_max_area_ratio=RELAXED_MAX_AREA_RATIO,
    )
    print(f"[INFO] 初步筛选: kept={len(kept)}, hard_negative={len(hard_negative)}, "
          f"rejected={len(rejected)}, elapsed={time.time() - t0:.1f}s")

    # 4. 去重
    print(f"\n[INFO] 同图同类去重...")
    t0 = time.time()
    kept_dedup, deduped = dedup_candidates(
        kept,
        iou_threshold=IOU_THRESHOLD,
        containment_threshold=CONTAINMENT_THRESHOLD,
    )
    print(f"[INFO] 去重后: kept={len(kept_dedup)}, deduped={len(deduped)}, "
          f"elapsed={time.time() - t0:.1f}s")

    # 更新 merge_stats 中的 kept_after_dedup
    dedup_counts: Dict[str, int] = defaultdict(int)
    for item in kept_dedup:
        dedup_counts[item["source_prompt"]] += 1
    for prompt in merge_stats:
        merge_stats[prompt]["kept_after_dedup"] = dedup_counts.get(prompt, 0)

    # 4.5 C 类 animal 每图上限
    kept_dedup, c_deduped = limit_c_animal_per_image(kept_dedup)
    deduped.extend(c_deduped)
    if c_deduped:
        print(f"[INFO] C 类 animal 每图限制后: deduped_c={len(c_deduped)}")

    # 5. 合并 base pseudo
    merge_counts = None
    debug_output = getattr(args, "debug_output", False)
    # 命令行 None → 用配置区默认；--no-merge 强制关闭
    merge_animal_into_base = (
        False if getattr(args, "no_merge_animal_into_base", False)
        else (
            args.merge_animal_into_base
            if getattr(args, "merge_animal_into_base", None) is not None
            else MERGE_ANIMAL_INTO_BASE
        )
    )
    base_pseudo_json = getattr(args, "base_pseudo_json", None) or BASE_PSEUDO_JSON

    summary_stats: Dict[str, Any] = {}
    grade_stats: Dict[str, Dict[str, int]] = {}
    prompt_stats: Dict[str, Dict[str, int]] = {}

    if merge_animal_into_base:
        # ── 新合并模式：只更新 base 的 animal ──
        if base_pseudo_json is None:
            print("[ERROR] --merge-animal-into-base 必须配合 --base-pseudo-json", file=sys.stderr)
            sys.exit(1)
        print(f"\n[INFO] 新模式：只把 clean rare animal 合进 base pseudo 的 prompts['animal']...")
        t0 = time.time()
        base_entries = load_base_pseudo_json(base_pseudo_json)
        if base_entries is not None:
            # 统计涉及多少张图
            rare_animal_images = len(set(
                c["image_path"] for c in kept_dedup
                if c.get("train_label") == "animal" and c.get("rle") is not None
            ))
            print(f"[INFO] base: {len(base_entries)} 张, rare animal 涉及: {rare_animal_images} 张, "
                  f"rare candidates: {len(kept_dedup)}")
            merged_prompt_entries, summary_stats, grade_stats, prompt_stats = \
                merge_animal_into_base_prompt_style(
                    base_entries=base_entries,
                    rare_candidates=kept_dedup,
                    iou_threshold=IOU_THRESHOLD,
                    containment_threshold=CONTAINMENT_THRESHOLD,
                )
            print(f"[INFO] animal 合并完成: base_entries={len(base_entries)}, "
                  f"rare_candidates={len(kept_dedup)}, "
                  f"elapsed={time.time() - t0:.1f}s")

            # 输出合并后的完整 prompt-style JSON（dry-run 时不写）
            merged_output = getattr(args, "merged_pseudo_output", None)
            if merged_output is None:
                merged_output = output_dir / "merged_base_plus_rare_animal.json"
            if not DRY_RUN:
                with open(merged_output, "w", encoding="utf-8") as f:
                    json.dump(merged_prompt_entries, f, ensure_ascii=False, indent=2)
                print(f"[OUTPUT] {merged_output}  ({len(merged_prompt_entries)} entries)")
            else:
                print(f"[DRY-RUN] 不写 merged label: {merged_output}")

            # 输出 summary md
            summary_output = getattr(args, "summary_output", None)
            if summary_output is None:
                summary_output = output_dir / "merge_summary.md"
            write_summary_md(summary_stats, grade_stats, prompt_stats, summary_output)

            final_manifest = kept_dedup
        else:
            print("[WARN] base pseudo 加载失败，只输出 clean rare")
            final_manifest = kept_dedup

    elif base_pseudo_json:
        # ── 旧合并模式：merge_with_base ──
        # 注意：merge_with_base 不适合 prompt-style base pseudo 的完整合并；
        # 如需只更新 animal 请使用 --merge-animal-into-base
        print(f"\n[INFO] 与 base pseudo 合并（旧模式）...")
        t0 = time.time()
        base_entries = load_base_pseudo_json(base_pseudo_json)
        if base_entries is not None:
            merged, merge_counts = merge_with_base(
                kept_dedup,
                base_entries,
                iou_threshold=IOU_THRESHOLD,
                containment_threshold=CONTAINMENT_THRESHOLD,
            )
            print(f"[INFO] 合并后: total={len(merged)}, rare_only={merge_counts.get('rare_only', 0)}, "
                  f"base_only={merge_counts.get('base_only', 0)}, "
                  f"overlap_kept_rare={merge_counts.get('both_kept_rare', 0)}, "
                  f"overlap_kept_base={merge_counts.get('both_kept_base', 0)}, "
                  f"elapsed={time.time() - t0:.1f}s")
            final_manifest = merged
        else:
            print("[WARN] base pseudo 加载失败，只输出 clean rare")
            final_manifest = kept_dedup
    else:
        final_manifest = kept_dedup

    # 6. dry-run 模式
    if DRY_RUN:
        print("\n" + "=" * 60)
        print("DRY RUN — 不输出最终 manifest，仅输出报告")
        print("=" * 60)

    # 7. 输出文件（默认只输出核心文件；debug 模式才输出中间文件）
    if debug_output:
        if not DRY_RUN and not merge_animal_into_base:
            write_manifest_json(output_dir / "clean_rare_manifest.json", final_manifest)
            write_jsonl(output_dir / "clean_rare_manifest.jsonl", final_manifest)
        write_jsonl(output_dir / "hard_negative_manifest.jsonl", hard_negative)
        write_jsonl(output_dir / "rejected_candidates.jsonl", rejected)
        write_merge_report(merge_stats, output_dir, policy)
        write_policy_report(policy, output_dir)
        write_dedup_report(deduped, len(deduped), output_dir)
    else:
        # 非 debug 模式：只输出精简信息
        if not DRY_RUN and not merge_animal_into_base:
            write_manifest_json(output_dir / "clean_rare_manifest.json", final_manifest)
        elif DRY_RUN:
            print(f"[DRY-RUN] 最终 manifest 候选数: {len(final_manifest)}")
            label_counts = defaultdict(int)
            for item in final_manifest:
                label_counts[item.get("train_label", "unknown")] += 1
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")
        if debug_output:
            write_jsonl(output_dir / "hard_negative_manifest.jsonl", hard_negative)
            write_jsonl(output_dir / "rejected_candidates.jsonl", rejected)
            write_merge_report(merge_stats, output_dir, policy)
            write_policy_report(policy, output_dir)
            write_dedup_report(deduped, len(deduped), output_dir)

    # 9. meta（仅 debug 模式写文件）
    if debug_output:
        meta = {
            "rare_pred_jsons": [str(x) for x in (rare_pred_jsons or [])],
            "base_pseudo_json": str(base_pseudo_json) if base_pseudo_json else None,
            "image_root": str(IMAGE_ROOT),
            "include_val": INCLUDE_VAL,
            "split_filter": SPLIT_FILTER,
            "dry_run": DRY_RUN,
            "min_area": MIN_AREA,
            "max_area_ratio": MAX_AREA_RATIO,
            "relaxed_max_area_ratio": RELAXED_MAX_AREA_RATIO,
            "iou_threshold": IOU_THRESHOLD,
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "policy_prompts": len(policy),
            "total_hits": sum(s.get("total_hits", 0) for s in merge_stats.values()),
            "kept_after_area": sum(s.get("kept_after_area", 0) for s in merge_stats.values()),
            "kept_after_dedup": len(kept_dedup),
            "final_manifest_count": len(final_manifest),
            "hard_negative_count": len(hard_negative),
            "rejected_count": len(rejected),
            "deduped_count": len(deduped),
            "merge_counts": merge_counts,
        }
        meta_path = output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] {meta_path}")

    # 10. 打印总结
    print("\n" + "=" * 60)
    print("BUILD CLEAN RARE MANIFEST — 完成")
    print("=" * 60)
    if merge_animal_into_base and summary_stats:
        print(f"  base images:              {summary_stats.get('base_images', 0)}")
        print(f"  rare candidates total:    {summary_stats.get('rare_candidates_total', 0)}")
        print(f"  rare after filter:        {summary_stats.get('rare_after_filter', 0)}")
        print(f"  rare kept merged:         {summary_stats.get('rare_kept_merged', 0)}")
        print(f"  rare dup with base:       {summary_stats.get('rare_dup_with_base', 0)}")
        print(f"  rare dup with rare:       {summary_stats.get('rare_dup_with_rare', 0)}")
        print(f"  images animal updated:    {summary_stats.get('images_with_animal_updated', 0)}")
        print(f"  images newly got animal:  {summary_stats.get('images_newly_got_animal', 0)}")
    else:
        total_hits = sum(s.get("total_hits", 0) for s in merge_stats.values())
        kept_area = sum(s.get("kept_after_area", 0) for s in merge_stats.values())
        print(f"  total hits:       {total_hits}")
        print(f"  kept after area:  {kept_area}")
        print(f"  kept after dedup: {len(kept_dedup)}")
        print(f"  final manifest:   {len(final_manifest)}")
        print(f"  hard negative:    {len(hard_negative)}")
        print(f"  rejected:         {len(rejected)}")
        print(f"  deduped:          {len(deduped)}")
        if merge_counts:
            print(f"  merge: rare_only={merge_counts.get('rare_only', 0)}, "
                  f"base_only={merge_counts.get('base_only', 0)}, "
                  f"overlap_kept_rare={merge_counts.get('both_kept_rare', 0)}, "
                  f"overlap_kept_base={merge_counts.get('both_kept_base', 0)}")
    print(f"\n输出目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 Clean Rare 伪标签 Manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="所有参数的默认值见脚本顶部「用户配置区」。",
    )
    parser.add_argument(
        "--rare-pred-jsons",
        type=str,
        nargs="+",
        default=None,
        help="多个 rare pred json，支持 tag=path 格式",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=IMAGE_ROOT,
        help=f"图片根目录 (默认 {IMAGE_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"输出目录 (默认 {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="自定义 prompt 策略 JSON 文件路径；不传则使用内置默认策略",
    )
    parser.add_argument(
        "--base-pseudo-json",
        type=Path,
        default=None,
        help="最全主伪标签路径，如果传入则与 clean rare 做并集去重",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        default=INCLUDE_VAL,
        help=f"允许 validation 伪标签进入训练 manifest (默认 {INCLUDE_VAL})",
    )
    parser.add_argument(
        "--split-filter",
        type=str,
        nargs="+",
        default=SPLIT_FILTER,
        help="只处理 image_path 中包含这些关键词的图片，如 data5 data7",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help=f"只输出报告，不写最终 manifest (默认 {DRY_RUN})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_AREA,
        help=f"最小 mask 面积（像素），默认 {MIN_AREA}",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=MAX_AREA_RATIO,
        help=f"最大 mask 面积占图比（默认），超过则丢弃 (默认 {MAX_AREA_RATIO})",
    )
    parser.add_argument(
        "--relaxed-max-area-ratio",
        type=float,
        default=RELAXED_MAX_AREA_RATIO,
        help=f"animal/car 的放宽面积上限 (默认 {RELAXED_MAX_AREA_RATIO})",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IoU 去重阈值 (默认 {IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=CONTAINMENT_THRESHOLD,
        help=f"containment 去重阈值 (默认 {CONTAINMENT_THRESHOLD})",
    )
    parser.add_argument(
        "--merge-animal-into-base",
        action="store_true",
        default=None,
        help="使用新模式：只把 clean rare animal 合进 base prompts['animal']（默认使用配置区 MERGE_ANIMAL_INTO_BASE）",
    )
    parser.add_argument(
        "--no-merge-animal-into-base",
        action="store_true",
        default=False,
        help="关闭新模式，只输出 clean rare manifest 或走旧逻辑",
    )
    parser.add_argument(
        "--merged-pseudo-output",
        type=Path,
        default=None,
        help="完整 prompt-style 合并伪标签输出路径",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="ABC 映射、去重、取并统计总结 md 输出路径",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        default=False,
        help="是否输出调试中间文件，默认不输出",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
