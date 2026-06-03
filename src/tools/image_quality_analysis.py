#!/usr/bin/env python3
"""
图像质量分析脚本（重写）
1. 读取 filename_analysis.csv 获取各行命名规则
2. 仅分析 test/ 下仍存在的图片
3. 按 文件夹 → 数字归一化模式 分组计算质量指标
4. 组内同质性判定：一致的按规则汇总，差异大的逐张标出
"""
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace, gaussian_filter

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "test"
PATTERN_CSV = PROJECT_ROOT / "filename_analysis.csv"
OUTPUT_GROUP = PROJECT_ROOT / "image_quality_analysis.csv"
OUTPUT_DETAIL = PROJECT_ROOT / "image_quality_outliers.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 同质性阈值: CV = std/mean，超过阈值视为组内不一致
CV_THRESHOLD = 0.25


# ── 模式函数 ──
def normalize_digits(name: str) -> str:
    return re.sub(r"\d+", "#", name)


# ── 读取 filename_analysis.csv 获取已知的 digit_pattern ──
def load_known_patterns(csv_path: Path) -> dict:
    """解析 CSV，返回 {folder: set(digit_patterns)}"""
    if not csv_path.exists():
        print(f"警告: {csv_path} 不存在，将仅基于现存文件分析")
        return {}

    patterns = defaultdict(set)
    current_folder = None
    in_section = False

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0]:
                continue
            first = row[0]
            # 匹配 "=== data1 —— 数字归一化模式 ==="
            m = re.match(r"=== (\S+) —— 数字归一化模式", first)
            if m:
                current_folder = m.group(1)
                in_section = True
                continue
            # 匹配下一个 section 标题就退出
            if first.startswith("=== ") and "数字归一化模式" not in first:
                in_section = False
                continue
            # 匹配 "=== ... —— 前缀分布" 这类
            if first.startswith("=== ") and "前缀分布" in first:
                in_section = False
                continue
            if first.startswith("=== ") and "语义模式" in first:
                in_section = False
                continue
            if first.startswith("=== ") and "总体概览" in first:
                in_section = False
                continue

            if in_section and current_folder:
                # row[0] = 模式, row[1] = 数量, row[2] = 占比
                pattern = row[0].strip()
                if pattern and pattern != "模式 (数字→#)" and not pattern.startswith("==="):
                    patterns[current_folder].add(pattern)

    return dict(patterns)


# ── 质量指标 ──
def compute_quality_metrics(img_path: Path) -> dict:
    img = Image.open(img_path)
    w, h = img.size
    gray = np.array(img.convert("L"), dtype=np.float64)

    brightness = float(np.mean(gray))
    dynamic_range = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    contrast_rms = float(np.std(gray) / brightness) if brightness > 0 else 0.0

    # 全部在原图分辨率上计算
    lap = laplace(gray)
    sharpness = float(np.var(lap))

    blurred = gaussian_filter(gray, sigma=2.0)
    residual = gray - blurred
    noise_std = float(np.std(residual))

    snr = float(20 * np.log10(brightness / noise_std)) if noise_std > 0 else 0.0

    return {
        "file": img_path.name,
        "width": w, "height": h,
        "brightness": brightness,
        "contrast_rms": contrast_rms,
        "sharpness_laplacian": sharpness,
        "noise_std": noise_std,
        "snr_db": snr,
        "dynamic_range_p99p01": dynamic_range,
    }


# ── 同质性判定 ──
def check_homogeneity(metrics_list: list) -> dict:
    """返回各指标的 CV，以及整体是否同质。"""
    metric_keys = [
        "brightness", "contrast_rms", "sharpness_laplacian",
        "noise_std", "snr_db", "dynamic_range_p99p01",
    ]
    cvs = {}
    homogeneous = True
    for key in metric_keys:
        vals = np.array([m[key] for m in metrics_list])
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        cv = std_v / mean_v if mean_v > 0 else 0.0
        cvs[f"{key}_cv"] = round(cv, 3)
        if cv > CV_THRESHOLD:
            homogeneous = False
    return {"homogeneous": homogeneous, "cvs": cvs}


def aggregate_metrics(metrics_list: list) -> dict:
    numeric_keys = [
        "width", "height", "brightness", "contrast_rms",
        "sharpness_laplacian", "noise_std", "snr_db", "dynamic_range_p99p01",
    ]
    arr = {k: np.array([m[k] for m in metrics_list]) for k in numeric_keys}
    agg = {"count": len(metrics_list)}
    for k, vals in arr.items():
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 2)
        agg[f"{k}_median"] = round(float(np.median(vals)), 2)
        agg[f"{k}_std"] = round(float(np.std(vals)), 2)
        agg[f"{k}_min"] = round(float(np.min(vals)), 2)
        agg[f"{k}_max"] = round(float(np.max(vals)), 2)
    return agg


# ── 主流程 ──
def main():
    t0 = time.time()

    # 1. 读取已知命名规则
    known = load_known_patterns(PATTERN_CSV)
    if known:
        total_patterns = sum(len(v) for v in known.values())
        print(f"从 {PATTERN_CSV.name} 读取到 {total_patterns} 条命名规则 "
              f"(分布在 {len(known)} 个文件夹)")
    else:
        print("未读取到已知规则，将基于现存文件自动分组")

    # 2. 扫描现存图片，按 (folder, digit_pattern) 分组
    if not TEST_DIR.exists():
        print(f"错误: 找不到 test 目录 ({TEST_DIR})")
        sys.exit(1)

    folders = sorted(
        d for d in TEST_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    all_images = []
    for folder in folders:
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(f)

    total = len(all_images)
    print(f"现存图片: {total} 张\n")

    # 分组: (folder, digit_pattern) → [metrics]
    groups = defaultdict(list)

    for i, img_path in enumerate(all_images):
        folder = img_path.parent.name
        dp = normalize_digits(img_path.stem)

        # 如果已知规则存在，只保留命中规则的图片
        if known and folder in known:
            if dp not in known[folder]:
                if (i + 1) % 2000 == 0:
                    pass  # skip silently
                continue

        try:
            metrics = compute_quality_metrics(img_path)
        except Exception as e:
            print(f"  [跳过] {img_path.name}: {e}")
            continue

        groups[(folder, dp)].append(metrics)

        if (i + 1) % 500 == 0 or i == total - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(
                f"  [{i + 1:>5}/{total}] "
                f"{(i + 1) / total * 100:5.1f}%  "
                f"已耗时 {elapsed:.0f}s  ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\n计算完成，耗时 {elapsed:.0f}s")

    # 3. 逐组判定同质性 → 输出
    group_field_order = [
        "count",
        "width_mean", "width_median", "width_min", "width_max",
        "height_mean", "height_median", "height_min", "height_max",
        "brightness_mean", "brightness_median", "brightness_min", "brightness_max",
        "contrast_rms_mean", "contrast_rms_median", "contrast_rms_min", "contrast_rms_max",
        "sharpness_laplacian_mean", "sharpness_laplacian_median",
        "sharpness_laplacian_min", "sharpness_laplacian_max",
        "noise_std_mean", "noise_std_median", "noise_std_min", "noise_std_max",
        "snr_db_mean", "snr_db_median", "snr_db_min", "snr_db_max",
        "dynamic_range_p99p01_mean", "dynamic_range_p99p01_median",
        "dynamic_range_p99p01_min", "dynamic_range_p99p01_max",
    ]

    cv_keys = [
        "brightness_cv", "contrast_rms_cv", "sharpness_laplacian_cv",
        "noise_std_cv", "snr_db_cv", "dynamic_range_p99p01_cv",
    ]

    detail_field_order = [
        "file", "width", "height",
        "brightness", "contrast_rms", "sharpness_laplacian",
        "noise_std", "snr_db", "dynamic_range_p99p01",
    ]

    homo_count = 0
    hetero_count = 0

    with open(OUTPUT_GROUP, "w", newline="", encoding="utf-8-sig") as fg, \
         open(OUTPUT_DETAIL, "w", newline="", encoding="utf-8-sig") as fd:

        gw = csv.writer(fg)
        dw = csv.writer(fd)

        # 汇总表头
        gw.writerow(
            ["文件夹", "数字归一化模式", "一致"] + group_field_order + cv_keys
        )
        # 明细表头
        dw.writerow(
            ["文件夹", "数字归一化模式", "原因(超标CV)"] + detail_field_order
        )

        for (folder, dp) in sorted(groups, key=lambda x: (x[0], x[1])):
            metrics_list = groups[(folder, dp)]
            agg = aggregate_metrics(metrics_list)
            homo = check_homogeneity(metrics_list)

            if homo["homogeneous"]:
                homo_count += 1
                gw.writerow(
                    [folder, dp, "✓ 一致"]
                    + [agg.get(k, "") for k in group_field_order]
                    + [homo["cvs"].get(k, "") for k in cv_keys]
                )
            else:
                hetero_count += 1
                # 汇总行
                gw.writerow(
                    [folder, dp, "✗ 不一致"]
                    + [agg.get(k, "") for k in group_field_order]
                    + [homo["cvs"].get(k, "") for k in cv_keys]
                )
                # 超标原因
                over = [k for k in cv_keys if homo["cvs"].get(k, 0) > CV_THRESHOLD]
                reason = "; ".join(over)
                # 明细：逐张图片写出
                for m in metrics_list:
                    dw.writerow(
                        [folder, dp, reason]
                        + [m.get(k, "") for k in detail_field_order]
                    )

    print(f"\n{'='*60}")
    print(f"同质组（按规则汇总）: {homo_count}")
    print(f"异质组（逐张明细）:   {hetero_count}")
    print(f"{'='*60}")
    print(f"汇总表: {OUTPUT_GROUP}")
    print(f"明细表: {OUTPUT_DETAIL}")


if __name__ == "__main__":
    main()
