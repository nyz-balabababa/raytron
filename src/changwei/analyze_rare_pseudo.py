#!/usr/bin/env python3
"""
分析一个或多个 SAM3 伪标签 pred_*.json 的 prompt 质量。
输出统计表 + 可视化 gallery，用于人工判断 prompt 是否可用。

用法示例：

  # 填好顶部「用户配置区」的 PRED_JSONS 后直接运行
  python src/changwei/analyze_rare_pseudo.py

  # 也可以用命令行覆盖配置区的值
  python src/changwei/analyze_rare_pseudo.py \
      --pred-jsons new=data/test/pred_val_animal.json \
      --top-k 0 --random-k 0

注意：
- 本脚本只用于人工分析 prompt 质量，不参与训练。
- 不要改 sam3/ 官方源码，不要改训练主逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 用户配置区：平时直接改这里 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
"""项目根目录，所有相对路径都基于这里"""

# 路径
IMAGE_ROOT = ROOT


OUTPUT_DIR = ROOT / "test" / "label_analysis" / "rare_label" / "train_animal"
#summary CSV / case JSON / 可视化 gallery 都放在这里

PRED_JSONS = [
    
    "A=test/label_analysis/label_analysis_ABC(animal)/pseudo_A.json"
]

STATS_CSVS = [
]
#stats_*.csv 列表，支持 'tag=path' 格式。可选，不填则跳过

# 可视化
TOP_K = 50
"""每个 prompt 可视化分数最高的 top-k 张图 (<=0 则跳过)"""

RANDOM_K = 30
"""每个 prompt 随机可视化图片数 (<=0 则跳过)"""

MASK_OVERLAY_COLOR = (0, 255, 0)
"""mask 叠图颜色 (BGR)"""

MASK_OVERLAY_ALPHA = 0.4
"""mask 叠图透明度"""

# 统计阈值
SUSPICIOUS_INSTANCE_THRESHOLD = 8
"""单张图上某个 prompt 的 instance 数超过此值视为可疑（可能碎片乱检）"""

ACTIVATION_RATE_LOW = 0.05
"""激活率低于此值视为过低，会触发警告"""

AVG_INSTANCES_HIGH = 5.0
"""平均实例数高于此值视为偏高"""

MAX_INSTANCES_HIGH = 10
"""单图最大实例数高于此值视为偏高"""

# GPU
USE_GPU = True
"""是否尝试使用 GPU 加速 (cupy > torch CUDA > CPU)"""

GPU_ID = 0
"""使用的 GPU 设备 ID"""

# 随机种子
RANDOM_SEED = 42

# 输出控制
VERBOSE = True
# ── 用户配置区结束 ──────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 依赖导入（不要改这里）
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

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
        print(f"[进度] {desc} ...", file=sys.stderr)
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


def _get_gpu_backend() -> str:
    """返回当前可用的 GPU 后端名称: 'cupy', 'torch', 'cpu'."""
    if HAS_CUPY:
        return "cupy"
    if HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda":
        return "torch"
    return "cpu"


# --- 图像处理 ---
try:
    import cv2

    HAS_CV2 = True
    try:
        HAS_CV2_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0  # type: ignore[attr-defined]
    except Exception:
        HAS_CV2_CUDA = False
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False
    HAS_CV2_CUDA = False

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    HAS_PIL = False

# --- RLE 解码 ---
try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    mask_utils = None  # type: ignore[assignment]
    HAS_PYCOCOTOOLS = False


# ═══════════════════════════════════════════════════════════════════════════════
# GPU 工具
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_gpu() -> None:
    """根据用户配置区参数初始化 GPU 环境。"""
    global USE_GPU, GPU_ID

    if not USE_GPU:
        if VERBOSE:
            print("[GPU] 用户禁用 GPU，使用 CPU 模式", file=sys.stderr)
        return

    backend = _get_gpu_backend()
    if backend == "cupy":
        try:
            cp.cuda.Device(GPU_ID).use()  # type: ignore[union-attr]
            if VERBOSE:
                print(
                    f"[GPU] CuPy 已激活, 设备 {GPU_ID}: "
                    f"{cp.cuda.runtime.getDeviceProperties(GPU_ID)['name'].decode()}",
                    file=sys.stderr,
                )
        except Exception as e:
            if VERBOSE:
                print(f"[GPU] CuPy 初始化失败: {e}", file=sys.stderr)
    elif backend == "torch":
        torch.cuda.set_device(GPU_ID)  # type: ignore[union-attr]
        if VERBOSE:
            print(
                f"[GPU] PyTorch CUDA 已激活, 设备 {GPU_ID}: "
                f"{torch.cuda.get_device_name(GPU_ID)}",  # type: ignore[union-attr]
                file=sys.stderr,
            )
    else:
        if VERBOSE:
            print("[GPU] 无可用的 GPU 后端，使用 CPU 模式", file=sys.stderr)

    if HAS_CV2_CUDA and VERBOSE:
        print(
            f"[GPU] OpenCV CUDA 可用, 设备数: {cv2.cuda.getCudaEnabledDeviceCount()}",  # type: ignore[attr-defined]
            file=sys.stderr,
        )


def _xp():
    """返回当前最优的数组模块 (cupy > numpy)。"""
    if HAS_CUPY and USE_GPU:
        return cp
    return np


def _as_gpu_tensor(arr: np.ndarray):
    """将 numpy 数组转换为 GPU 张量（torch 可用时）或保持原样。"""
    if HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda" and USE_GPU:
        return torch.as_tensor(arr, device=TORCH_DEVICE)  # type: ignore[union-attr]
    return arr


def _as_numpy(tensor_or_arr) -> np.ndarray:
    """将 torch tensor / cupy array 转回 numpy。"""
    if HAS_TORCH and isinstance(tensor_or_arr, torch.Tensor):
        return tensor_or_arr.cpu().numpy()
    if HAS_CUPY and isinstance(tensor_or_arr, cp.ndarray):
        return cp.asnumpy(tensor_or_arr)  # type: ignore[union-attr]
    return np.asarray(tensor_or_arr)


def _gpu_argsort(arr: np.ndarray, ascending: bool = False) -> np.ndarray:
    """在 GPU（可用时）上做 argsort，返回 numpy 数组。"""
    xp_mod = _xp()
    gpu_arr = xp_mod.asarray(arr)
    indices = xp_mod.argsort(gpu_arr)
    if not ascending:
        indices = indices[::-1]
    return _as_numpy(indices)


def _gpu_random_choice(n: int, size: int) -> np.ndarray:
    """在 GPU（可用时）上做无放回随机采样，返回 numpy 索引数组。"""
    if size >= n:
        return np.arange(n)
    xp_mod = _xp()
    rng = xp_mod.random.RandomState(RANDOM_SEED)
    indices = rng.choice(n, size=size, replace=False)
    return _as_numpy(indices)


# ═══════════════════════════════════════════════════════════════════════════════
# RLE 工具
# ═══════════════════════════════════════════════════════════════════════════════


def decode_rle(rle: Dict[str, object]) -> Optional[np.ndarray]:
    """解码 RLE -> 二值 mask (H, W) uint8. 支持 pycocotools 和 simple_rle 两种格式."""
    if rle is None:
        return None

    size = rle.get("size")
    counts = rle.get("counts")
    if size is None or counts is None:
        return None

    h, w = int(size[0]), int(size[1])  # type: ignore[arg-type]

    # 尝试 pycocotools
    if HAS_PYCOCOTOOLS and mask_utils is not None:
        try:
            if isinstance(counts, bytes):
                coco_rle = {"size": [h, w], "counts": counts}
            elif isinstance(counts, str):
                coco_rle = {"size": [h, w], "counts": counts.encode("utf-8")}
            else:
                coco_rle = {"size": [h, w], "counts": counts}
            mask = mask_utils.decode(coco_rle)
            return mask.astype(np.uint8)
        except Exception:
            pass

    # fallback: simple_rle comma-separated counts
    if isinstance(counts, str) and "," in counts:
        try:
            runs = [int(x) for x in counts.split(",")]
            total_pixels = h * w
            if sum(runs) == total_pixels or sum(runs) <= total_pixels:
                mask = np.zeros(total_pixels, dtype=np.uint8)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_tag_path(raw: str) -> Tuple[str, Path]:
    """解析 'tag=path' 或 'path' 格式."""
    if "=" in raw:
        tag, path_str = raw.split("=", 1)
        return tag.strip(), Path(path_str.strip())
    else:
        path = Path(raw.strip())
        tag = path.stem.replace("pred_", "").replace("stats_", "")
        return tag, path


def load_pred_jsons(
    raw_specs: Optional[List[str]],
) -> Dict[str, List[Dict[str, object]]]:
    """加载一个或多个 pred_*.json，返回 {tag: [entries]}."""
    result: Dict[str, List[Dict[str, object]]] = {}
    if not raw_specs:
        return result
    for raw in _progress(raw_specs, desc="加载 pred json", unit="file"):
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
            print(f"[WARN] 无法识别的 pred json 格式: {path}", file=sys.stderr)
            continue
        if VERBOSE:
            print(f"[INFO] 加载 pred json: tag={tag}, entries={len(result[tag])}, path={path}")
    return result


def load_stats_csvs(
    raw_specs: Optional[List[str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """加载 stats_*.csv，返回 {tag: {prompt: {col: value}}}."""
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not raw_specs:
        return result
    for raw in _progress(raw_specs, desc="加载 stats csv", unit="file"):
        tag, path = _resolve_tag_path(raw)
        if not path.exists():
            print(f"[WARN] stats csv 不存在: {path}", file=sys.stderr)
            continue
        prompt_map: Dict[str, Dict[str, float]] = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                numeric: Dict[str, object] = {}
                for k, v in row.items():
                    if k == "prompt":
                        continue
                    try:
                        numeric[k] = float(v)
                    except (ValueError, TypeError):
                        numeric[k] = v
                prompt_map[prompt] = numeric  # type: ignore[assignment]
        result[tag] = prompt_map
        if VERBOSE:
            print(f"[INFO] 加载 stats csv: tag={tag}, prompts={len(prompt_map)}, path={path}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 统计计算
# ═══════════════════════════════════════════════════════════════════════════════


def _gpu_mean(arr: List[float]) -> float:
    """在 GPU 上计算均值（列表较小时直接用 numpy 更高效）。"""
    if len(arr) == 0:
        return 0.0
    if len(arr) < 1000:
        return float(np.mean(arr))
    xp_mod = _xp()
    gpu_arr = xp_mod.asarray(arr, dtype=xp_mod.float32)
    return float(xp_mod.mean(gpu_arr))


def _gpu_median(arr: List[float]) -> float:
    """在 GPU 上计算中位数。"""
    if len(arr) == 0:
        return 0.0
    if len(arr) < 1000:
        return float(np.median(arr))
    xp_mod = _xp()
    gpu_arr = xp_mod.asarray(arr, dtype=xp_mod.float32)
    return float(xp_mod.median(gpu_arr))


def compute_prompt_stats(
    all_preds: Dict[str, List[Dict[str, object]]],
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, object]]:
    """
    计算每个 (tag, prompt) 的统计数据。
    返回 {prompt_key: {stat_fields}}，prompt_key = "tag::prompt".
    """
    # 先收集所有 prompt
    all_prompts: Dict[str, set] = defaultdict(set)  # tag -> {prompts}
    for tag, entries in all_preds.items():
        for entry in entries:
            for prompt in entry.get("prompts", {}):
                all_prompts[tag].add(prompt)

    result: Dict[str, Dict[str, object]] = {}

    # 统计每个 tag
    tag_items = list(all_preds.items())
    for tag, entries in _progress(tag_items, desc="计算 prompt 统计", unit="tag"):
        total_images = len(entries)
        prompt_hits: Dict[str, int] = defaultdict(int)
        prompt_scores: Dict[str, List[float]] = defaultdict(list)
        prompt_instances: Dict[str, List[int]] = defaultdict(list)
        prompt_suspicious: Dict[str, int] = defaultdict(int)
        prompt_has_selected_scores: Dict[str, int] = defaultdict(int)

        for entry in _progress(
            entries, desc=f"  处理 {tag}", unit="img", leave=False
        ):
            for prompt, pdata in entry.get("prompts", {}).items():
                if pdata.get("hit"):
                    prompt_hits[prompt] += 1
                    score = float(pdata.get("score", 0))
                    instances = int(pdata.get("instances", 0))
                    prompt_scores[prompt].append(score)
                    prompt_instances[prompt].append(instances)
                    if instances >= SUSPICIOUS_INSTANCE_THRESHOLD:
                        prompt_suspicious[prompt] += 1
                    if "selected_scores" in pdata:
                        prompt_has_selected_scores[prompt] += 1

        for prompt in all_prompts.get(tag, set()):
            key = f"{tag}::{prompt}"
            scores = prompt_scores.get(prompt, [])
            instances = prompt_instances.get(prompt, [])
            hits = prompt_hits.get(prompt, 0)

            row: Dict[str, object] = {
                "tag": tag,
                "prompt": prompt,
                "total_images": total_images,
                "hit_images": hits,
                "activation_rate": round(hits / max(total_images, 1), 4),
                "avg_score": round(_gpu_mean(scores), 4) if scores else 0,
                "median_score": round(_gpu_median(scores), 4) if scores else 0,
                "avg_instances": round(_gpu_mean([float(i) for i in instances]), 2) if instances else 0,
                "max_instances": int(max(instances)) if instances else 0,
                "total_instances": int(sum(instances)),
                "suspicious_many_instance_count": prompt_suspicious.get(prompt, 0),
                "has_selected_scores_count": prompt_has_selected_scores.get(prompt, 0),
            }

            # 合并 stats csv 信息
            if tag in all_stats and prompt in all_stats[tag]:
                s = all_stats[tag][prompt]
                row["stats_raw_hits"] = s.get("raw_hits", "")
                row["stats_raw_avg_instances"] = s.get("raw_avg_instances", "")
                row["stats_raw_activation_rate"] = s.get("raw_activation_rate", "")
                row["stats_filtered_hits"] = s.get("filtered_hits", "")
                row["stats_filtered_avg_instances"] = s.get("filtered_avg_instances", "")
                row["stats_filtered_activation_rate"] = s.get("filtered_activation_rate", "")

            result[key] = row

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════════


def _find_best_entries_for_prompt(
    pred_entries: List[Dict[str, object]],
    prompt: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """找出某个 prompt 下分数最高的 TOP_K 和随机 RANDOM_K 个 hit 条目."""
    hits: List[Dict[str, object]] = []
    for entry in pred_entries:
        pdata = entry.get("prompts", {}).get(prompt)
        if pdata and pdata.get("hit"):
            hits.append(
                {
                    "image_path": entry.get("image_path", ""),
                    "score": float(pdata.get("score", 0)),
                    "instances": int(pdata.get("instances", 0)),
                    "rle": pdata.get("rle"),
                    "selected_scores": pdata.get("selected_scores", []),
                }
            )

    # 用 GPU 加速排序（hits 较多时）
    if len(hits) > 1000 and (
        HAS_CUPY or (HAS_TORCH and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda")
    ):
        scores_arr = np.array([h["score"] for h in hits], dtype=np.float32)
        order = _gpu_argsort(scores_arr, ascending=False)
        hits = [hits[i] for i in order]
    else:
        hits.sort(key=lambda x: float(x["score"]), reverse=True)

    top = hits[:max(TOP_K, 0)]

    # 随机采样
    if RANDOM_K > 0 and len(hits) > 0:
        indices = _gpu_random_choice(len(hits), min(RANDOM_K, len(hits)))
        random_picks = [hits[i] for i in indices]
    else:
        random_picks = []

    return top, random_picks


def _overlay_mask_on_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """将二值 mask 半透明叠加到 BGR 图像上."""
    if not HAS_CV2:
        return image_bgr

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]))

    color = MASK_OVERLAY_COLOR
    alpha = MASK_OVERLAY_ALPHA

    # 尝试 OpenCV CUDA 加速
    if HAS_CV2_CUDA and USE_GPU:
        try:
            gpu_img = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
            gpu_img.upload(image_bgr)
            gpu_mask = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
            gpu_mask.upload(mask)

            color_img = np.zeros_like(image_bgr)
            color_img[mask > 0] = color
            gpu_color = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
            gpu_color.upload(color_img)
            cv2.cuda.addWeighted(  # type: ignore[attr-defined]
                gpu_img, 1 - alpha, gpu_color, alpha, 0, gpu_img
            )
            result = gpu_img.download()
            # 轮廓仍在 CPU 上绘制（cv2.cuda 没有 findContours）
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
            return result
        except Exception:
            pass  # CUDA 失败则回退到 CPU

    # CPU 路径
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _save_visualization(
    output_dir: Path,
    gallery_type: str,
    prompt: str,
    items: List[Dict[str, object]],
    image_root: Path,
    source_tag: str,
) -> int:
    """保存一批可视化图片，返回成功保存的数量."""
    gallery_dir = output_dir / "galleries" / prompt / gallery_type
    gallery_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx, item in enumerate(
        _progress(items, desc=f"  可视化 {prompt[:40]} [{gallery_type}]", unit="img", leave=False)
    ):
        image_path = str(item.get("image_path", ""))
        rel_path = image_path.replace("\\", "/")
        abs_path = image_root / rel_path

        safe_basename = os.path.basename(rel_path).rsplit(".", 1)[0]
        safe_basename = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in safe_basename
        )
        out_name = (
            f"{idx:04d}_s{float(item['score']):.3f}_i{int(item['instances'])}"
            f"_{source_tag}_{safe_basename}.jpg"
        )

        out_path = gallery_dir / out_name

        # 尝试读取原图
        if not abs_path.exists():
            with open(gallery_dir / "_missing_files.txt", "a", encoding="utf-8") as f:
                f.write(f"{rel_path}\n")
            continue

        try:
            if HAS_CV2:
                img = cv2.imread(str(abs_path))
                if img is None:
                    raise ValueError("cv2 无法解码")
            else:
                pil_img = Image.open(str(abs_path)).convert("RGB")  # type: ignore[union-attr]
                img = np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR
        except Exception:
            import shutil

            fail_name = (
                f"{idx:04d}_s{float(item['score']):.3f}_i{int(item['instances'])}"
                f"_{source_tag}_decode_failed_{safe_basename}.jpg"
            )
            try:
                shutil.copy2(str(abs_path), str(gallery_dir / fail_name))
            except Exception:
                pass
            continue

        # 解码 RLE 并叠加 mask
        rle = item.get("rle")
        if rle is not None:
            mask = decode_rle(rle)  # type: ignore[arg-type]
            if mask is not None and HAS_CV2:
                img = _overlay_mask_on_image(img, mask)

        if HAS_CV2:
            cv2.imwrite(str(out_path), img)
        elif HAS_PIL:
            Image.fromarray(img[:, :, ::-1]).save(str(out_path))  # type: ignore[union-attr]
        else:
            continue

        saved += 1

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════════════════════


def _sorted_prompt_keys(stats: Dict[str, Dict[str, object]]) -> List[str]:
    """按 activation_rate 降序排列 prompt key."""
    return sorted(
        stats.keys(), key=lambda k: float(stats[k].get("activation_rate", 0)), reverse=True
    )


def write_summary_csv(stats: Dict[str, Dict[str, object]], output_dir: Path) -> Path:
    """输出 summary_by_prompt.csv."""
    path = output_dir / "summary_by_prompt.csv"
    keys = _sorted_prompt_keys(stats)
    if not keys:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("no_data\n")
        return path

    fieldnames = list(stats[keys[0]].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for k in keys:
            writer.writerow(stats[k])
    print(f"[OUTPUT] {path}")
    return path


def write_suspicious_csv(stats: Dict[str, Dict[str, object]], output_dir: Path) -> Path:
    """输出可疑 prompt 列表."""
    path = output_dir / "suspicious_cases.csv"
    suspicious = []
    for k, row in stats.items():
        if int(row.get("suspicious_many_instance_count", 0)) > 0:
            suspicious.append(row)
    suspicious.sort(
        key=lambda r: int(r.get("suspicious_many_instance_count", 0)), reverse=True
    )

    if not suspicious:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("no_suspicious_cases\n")
        return path

    fieldnames = list(suspicious[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in suspicious:
            writer.writerow(row)
    print(f"[OUTPUT] {path}")
    return path


def write_case_json(
    stats: Dict[str, Dict[str, object]],
    all_preds: Dict[str, List[Dict[str, object]]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """输出 prompt_top_cases.json 和 prompt_random_cases.json."""
    top_path = output_dir / "prompt_top_cases.json"
    random_path = output_dir / "prompt_random_cases.json"

    top_cases: Dict[str, List[Dict[str, object]]] = {}
    random_cases: Dict[str, List[Dict[str, object]]] = {}

    for key, row in _progress(
        stats.items(), desc="导出 case JSON", unit="prompt"
    ):
        tag = str(row["tag"])
        prompt = str(row["prompt"])
        entries = all_preds.get(tag, [])
        top_items, random_items = _find_best_entries_for_prompt(entries, prompt)
        # 精简输出（不存完整 RLE，太大）
        top_cases[f"{tag}::{prompt}"] = [
            {
                "image_path": it["image_path"],
                "score": it["score"],
                "instances": it["instances"],
                "selected_scores": it.get("selected_scores", []),
            }
            for it in top_items
        ]
        random_cases[f"{tag}::{prompt}"] = [
            {
                "image_path": it["image_path"],
                "score": it["score"],
                "instances": it["instances"],
                "selected_scores": it.get("selected_scores", []),
            }
            for it in random_items
        ]

    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_cases, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {top_path}")

    with open(random_path, "w", encoding="utf-8") as f:
        json.dump(random_cases, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] {random_path}")

    return top_path, random_path


def generate_galleries(
    stats: Dict[str, Dict[str, object]],
    all_preds: Dict[str, List[Dict[str, object]]],
    output_dir: Path,
    image_root: Path,
) -> None:
    """为每个 prompt 生成 top/random 可视化图库."""
    if TOP_K <= 0 and RANDOM_K <= 0:
        print("[INFO] TOP_K=0 and RANDOM_K=0, 跳过可视化")
        return
    if not HAS_CV2:
        print("[WARN] cv2 未安装，可视化质量会降低")

    total_saved_top = 0
    total_saved_random = 0

    for key, row in _progress(
        stats.items(), desc="生成可视化图库", unit="prompt"
    ):
        tag = str(row["tag"])
        prompt = str(row["prompt"])
        entries = all_preds.get(tag, [])
        top_items, random_items = _find_best_entries_for_prompt(entries, prompt)

        if TOP_K > 0:
            saved = _save_visualization(
                output_dir, "top", prompt, top_items, image_root, tag
            )
            print(f"[VIZ] {tag}::{prompt} top: saved {saved}/{len(top_items)}")
            total_saved_top += saved

        if RANDOM_K > 0:
            saved = _save_visualization(
                output_dir, "random", prompt, random_items, image_root, tag
            )
            print(f"[VIZ] {tag}::{prompt} random: saved {saved}/{len(random_items)}")
            total_saved_random += saved

    print(f"[VIZ] 总计: top={total_saved_top}, random={total_saved_random}")


def print_recommendations(stats: Dict[str, Dict[str, object]]) -> None:
    """打印每个 prompt 的人工判断建议."""

    print("\n" + "=" * 80)
    print("PROMPT 质量建议")
    print("=" * 80)

    keys = _sorted_prompt_keys(stats)
    for k in keys:
        row = stats[k]
        tag = row["tag"]
        prompt = row["prompt"]
        avg_inst = float(row["avg_instances"])
        max_inst = int(row["max_instances"])
        act_rate = float(row["activation_rate"])
        avg_score = float(row["avg_score"])
        suspicious = int(row.get("suspicious_many_instance_count", 0))

        issues = []
        if avg_inst > AVG_INSTANCES_HIGH or max_inst > MAX_INSTANCES_HIGH:
            issues.append(
                f"⚠ 实例数偏高 (avg={avg_inst}, max={max_inst})，可能碎片乱检"
            )
        if act_rate < ACTIVATION_RATE_LOW:
            issues.append(f"⚠ 激活率过低 ({act_rate:.2%})，样本不足")
        if suspicious > 0:
            issues.append(f"⚠ {suspicious} 张图实例数超过阈值({SUSPICIOUS_INSTANCE_THRESHOLD})")
        if avg_score > 0.6 and avg_inst < 4 and act_rate > 0.1:
            issues.append(
                f"✅ 质量较好 (avg_score={avg_score:.3f})，值得人工看 top 样本"
            )
        if not issues:
            issues.append("— 指标中性，建议人工抽查")

        print(f"\n[{tag}] {prompt}")
        for issue in issues:
            print(f"  {issue}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def _print_gpu_status() -> None:
    """打印当前 GPU 状态摘要."""
    backend = _get_gpu_backend()
    print(f"[SYS] GPU 后端: {backend}")
    print(f"[SYS] PyTorch: {'✓' if HAS_TORCH else '✗'}  "
          f"CuPy: {'✓' if HAS_CUPY else '✗'}  "
          f"OpenCV CUDA: {'✓' if HAS_CV2_CUDA else '✗'}")
    print(f"[SYS] tqdm: {'✓' if HAS_TQDM else '✗ (无进度条)'}  "
          f"cv2: {'✓' if HAS_CV2 else '✗'}  "
          f"PIL: {'✓' if HAS_PIL else '✗'}  "
          f"pycocotools: {'✓' if HAS_PYCOCOTOOLS else '✗'}")


def run(args: argparse.Namespace) -> None:
    """主流程：根据命令行参数更新配置并运行全部分析."""
    global IMAGE_ROOT, OUTPUT_DIR, TOP_K, RANDOM_K
    global SUSPICIOUS_INSTANCE_THRESHOLD, USE_GPU, GPU_ID, VERBOSE

    # 用命令行参数覆盖用户配置区默认值
    if hasattr(args, "image_root") and args.image_root is not None:
        IMAGE_ROOT = args.image_root
    if hasattr(args, "output_dir") and args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if hasattr(args, "top_k"):
        TOP_K = args.top_k
    if hasattr(args, "random_k"):
        RANDOM_K = args.random_k
    if hasattr(args, "suspicious_instance_threshold"):
        SUSPICIOUS_INSTANCE_THRESHOLD = args.suspicious_instance_threshold
    if hasattr(args, "use_gpu"):
        USE_GPU = args.use_gpu
    if hasattr(args, "gpu_id"):
        GPU_ID = args.gpu_id

    # 初始化 GPU
    _setup_gpu()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        _print_gpu_status()

    # 1. 加载数据
    pred_jsons = getattr(args, "pred_jsons", None)
    stats_csvs = getattr(args, "stats_csvs", None)
    # 命令行未传则用配置区的值
    if pred_jsons is None:
        pred_jsons = PRED_JSONS if PRED_JSONS else None
    if stats_csvs is None:
        stats_csvs = STATS_CSVS if STATS_CSVS else None
    all_preds = load_pred_jsons(pred_jsons)
    all_stats = load_stats_csvs(stats_csvs)

    if not all_preds:
        print("[ERROR] 至少需要传入一个 pred json", file=sys.stderr)
        sys.exit(1)

    # 2. 统计
    print(f"\n[INFO] 计算 prompt 统计...")
    stats = compute_prompt_stats(all_preds, all_stats)
    print(f"[INFO] 共 {len(stats)} 个 (tag, prompt) 组合")

    # 3. 输出 CSV
    print(f"\n[INFO] 输出 CSV...")
    write_summary_csv(stats, output_dir)
    write_suspicious_csv(stats, output_dir)

    # 4. 输出 case JSON
    print(f"\n[INFO] 导出 case JSON...")
    write_case_json(stats, all_preds, output_dir)

    # 5. 可视化
    print(f"\n[INFO] 生成可视化图库...")
    generate_galleries(stats, all_preds, output_dir, IMAGE_ROOT)

    # 6. 打印建议
    print_recommendations(stats)

    print(f"\n[DONE] 分析完成，输出目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 SAM3 伪标签 prompt 质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="所有参数的默认值见脚本顶部「用户配置区」。",
    )
    parser.add_argument(
        "--pred-jsons",
        type=str,
        nargs="+",
        default=None,
        help="多个 pred json，支持 tag=path 格式，如 old=old_pred.json new=new_pred.json",
    )
    parser.add_argument(
        "--stats-csvs",
        type=str,
        nargs="+",
        default=None,
        help="多个 stats csv，支持 tag=path 格式",
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
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"每个 prompt 可视化 top-k 张 (默认 {TOP_K}, <=0 则跳过)",
    )
    parser.add_argument(
        "--random-k",
        type=int,
        default=RANDOM_K,
        help=f"每个 prompt 随机可视化数量 (默认 {RANDOM_K}, <=0 则跳过)",
    )
    parser.add_argument(
        "--suspicious-instance-threshold",
        type=int,
        default=SUSPICIOUS_INSTANCE_THRESHOLD,
        help=f"实例数超过此值视为可疑 (默认 {SUSPICIOUS_INSTANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--use-gpu",
        type=lambda s: s.lower() not in ("0", "false", "no", "off"),
        default=USE_GPU,
        help=f"是否使用 GPU 加速 (默认 {USE_GPU}, 传 0/false/no/off 禁用)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=GPU_ID,
        help=f"使用的 GPU 设备 ID (默认 {GPU_ID})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
