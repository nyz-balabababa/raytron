#!/usr/bin/env python3
"""
自适应预处理脚本
功能：尺寸统一 → 反色统一 → 质量评估 → 动态 CLAHE → 动态降噪 → 动态锐化
每张图按实时质量指标自适应触发各步骤，输出预处理后的图片和预处理日志 CSV。
"""
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VAL_LIST = ROOT / "test" / "val_list.txt"
OUTPUT_DIR = ROOT / "test" / "preprocessed"
LOG_CSV = OUTPUT_DIR / "preprocess_log.csv"

# ── 可调参数 ──
SAVE_IMAGES = False   # True=输出预处理图片, False=仅输出日志 CSV
TARGET_W, TARGET_H = 640, 512

# 对比度阈值
STD_LOW = 35      # < 35: 严重低对比度
STD_MID = 50      # 35~50: 中等低对比度

# 噪声阈值
NOISE_HIGH = 12   # > 12: 高噪声
NOISE_MED = 8     # 8~12: 中噪声

# 清晰度阈值
BLUR_LOW = 150    # < 150: 严重模糊
BLUR_MID = 300    # 150~300: 轻度模糊

# 反色关键词（文件名包含以下任一则触发反色）
INVERT_KEYWORDS = ["blackHot", "_pair"]


# ── 质量指标 ────────────────────────────────────────────────────────

def estimate_noise_sigma(gray: np.ndarray) -> float:
    """用拉普拉斯算子中值法快速估计噪声标准差（无需 scipy）。"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = np.median(np.abs(lap)) / 0.6745
    return float(sigma)


def compute_metrics(gray: np.ndarray) -> dict:
    return {
        "std": float(np.std(gray)),
        "noise_sigma": estimate_noise_sigma(gray),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean": float(np.mean(gray)),
    }


# ── 各处理步骤 ──────────────────────────────────────────────────────

def apply_resize(img: np.ndarray, target_w: int, target_h: int, img_path: str):
    """自适应缩放：极小图用 INTER_CUBIC 上采样，其余用 INTER_LINEAR。"""
    h, w = img.shape[:2]
    if (w, h) == (target_w, target_h):
        return img, "skip"

    if w <= target_w // 2 or h <= target_h // 2:
        interp = cv2.INTER_CUBIC
        note = "upsample_cubic"
    else:
        interp = cv2.INTER_LINEAR
        note = "linear"
    return cv2.resize(img, (target_w, target_h), interpolation=interp), note


def apply_invert(img: np.ndarray, img_name: str):
    """检测文件名中的反色关键词，若是则 255-img 反转。"""
    for kw in INVERT_KEYWORDS:
        if kw in img_name:
            return 255 - img, True
    # 补充：亮度极高且不是可见光场景 → 也可能是反色红外
    if img.ndim == 2 or img.shape[2] == 1:
        gray = img if img.ndim == 2 else img[:, :, 0]
        if np.mean(gray) > 200 and "vis" not in img_name.lower():
            return 255 - img, True
    return img, False


def apply_clahe(gray: np.ndarray, img_name: str) -> tuple[np.ndarray, str]:
    """自适应 CLAHE：严重低对比度用小格子大汗布，中对比度用大格子。"""
    std = float(np.std(gray))
    if std >= STD_MID:
        return gray, "skip"
    if std < STD_LOW:
        clip, tile = 2.5, (8, 8)
        label = "clahe_strong"
    else:
        clip, tile = 1.5, (16, 16)
        label = "clahe_mild"

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray), label


def apply_denoise(gray: np.ndarray, noise_sigma: float, img_name: str) -> tuple[np.ndarray, str]:
    """自适应降噪：高噪声用双边滤波，中噪声用中值滤波。"""
    if noise_sigma <= NOISE_MED:
        return gray, "skip"
    if noise_sigma > NOISE_HIGH:
        # 双边滤波 保边
        result = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
        return result, "bilateral"
    else:
        result = cv2.medianBlur(gray, 3)
        return result, "median3"


def apply_sharpen(gray: np.ndarray, blur_score: float, img_name: str) -> tuple[np.ndarray, str]:
    """自适应锐化：严重模糊用拉普拉斯锐化核，轻度用 USM。"""
    if blur_score >= BLUR_MID:
        return gray, "skip"
    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(gray, -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8), "laplacian_kernel"
    else:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        result = cv2.addWeighted(gray, 1.0 + 1.0, blurred, -1.0, 0)
        return np.clip(result, 0, 255).astype(np.uint8), "usm"


# ── 主处理管线 ──────────────────────────────────────────────────────

def preprocess_one(img_path: str, output_dir: Path) -> dict:
    """处理单张图片，返回日志记录。"""
    log = {"path": img_path}

    # 读取
    abs_path = ROOT / img_path
    if not abs_path.exists():
        log["status"] = "missing"
        return log

    img = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log["status"] = "read_error"
        return log

    img_name = Path(img_path).name

    # 1. 尺寸统一
    img, resize_note = apply_resize(img, TARGET_W, TARGET_H, img_path)
    log["resize"] = resize_note

    # 2. 反色统一
    img, inverted = apply_invert(img, img_name)
    log["invert"] = inverted

    # 3. 质量评估（在增强之前算，作为触发依据）
    metrics = compute_metrics(img)
    log.update({f"pre_{k}": v for k, v in metrics.items()})

    # 4. CLAHE
    img, clahe_note = apply_clahe(img, img_name)
    log["clahe"] = clahe_note

    # 5. 降噪
    img, denoise_note = apply_denoise(img, metrics["noise_sigma"], img_name)
    log["denoise"] = denoise_note

    # 6. 锐化
    img, sharpen_note = apply_sharpen(img, metrics["blur_score"], img_name)
    log["sharpen"] = sharpen_note

    # 后处理质量评估
    post_metrics = compute_metrics(img)
    log.update({f"post_{k}": v for k, v in post_metrics.items()})

    # 保存
    rel = Path(img_path)
    if SAVE_IMAGES:
        out_path = output_dir / rel.parent.name / rel.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        log["output"] = str(out_path.relative_to(ROOT))
    else:
        log["output"] = "(skipped)"

    log["status"] = "ok"
    return log


def main():
    # 读取路径列表
    if not VAL_LIST.exists():
        print(f"错误: 找不到 {VAL_LIST}")
        return

    with open(VAL_LIST, encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    total = len(paths)
    print(f"待处理: {total} 张")
    print(f"目标尺寸: {TARGET_W}×{TARGET_H}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FIELDS = [
        "path", "status", "output",
        "resize", "invert", "clahe", "denoise", "sharpen",
        "pre_std", "pre_noise_sigma", "pre_blur_score", "pre_mean",
        "post_std", "post_noise_sigma", "post_blur_score", "post_mean",
    ]

    # 统计
    stats = {
        "total": 0, "ok": 0, "missing": 0, "read_error": 0,
        "inverted": 0, "clahe_strong": 0, "clahe_mild": 0,
        "bilateral": 0, "median3": 0,
        "laplacian_kernel": 0, "usm": 0,
        "upsample": 0,
    }

    t0 = time.time()
    with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for i, img_path in enumerate(paths):
            log = preprocess_one(img_path, OUTPUT_DIR)
            writer.writerow({k: log.get(k, "") for k in LOG_FIELDS})

            # 统计
            stats["total"] += 1
            st = log.get("status", "?")
            stats[st] = stats.get(st, 0) + 1
            if log.get("invert"):
                stats["inverted"] += 1
            if log.get("resize") == "upsample_cubic":
                stats["upsample"] += 1
            for key in ["clahe", "denoise", "sharpen"]:
                val = log.get(key, "skip")
                if val != "skip":
                    stats[val] = stats.get(val, 0) + 1

            # 进度
            if (i + 1) % 500 == 0 or i == total - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                print(
                    f"  [{i+1:>5}/{total}] "
                    f"{(i+1)/total*100:5.1f}%  "
                    f"耗时 {elapsed:.0f}s  ETA {eta:.0f}s"
                )

    elapsed = time.time() - t0
    print(f"\n完成，总耗时 {elapsed:.0f}s ({elapsed/total*1000:.0f}ms/张)")

    print(f"\n{'='*50}")
    print("预处理统计")
    print(f"{'='*50}")
    print(f"  总计:       {stats['total']:>6}")
    print(f"  成功:       {stats['ok']:>6}")
    if stats["missing"]:     print(f"  缺失:       {stats['missing']:>6}")
    if stats["read_error"]:  print(f"  读取错误:   {stats['read_error']:>6}")
    print(f"  反色统一:   {stats['inverted']:>6}")
    print(f"  上采样:     {stats['upsample']:>6}")
    print(f"  CLAHE(强):  {stats['clahe_strong']:>6}")
    print(f"  CLAHE(弱):  {stats['clahe_mild']:>6}")
    print(f"  双边降噪:   {stats['bilateral']:>6}")
    print(f"  中值降噪:   {stats['median3']:>6}")
    print(f"  拉普拉斯锐化: {stats['laplacian_kernel']:>6}")
    print(f"  USM锐化:    {stats['usm']:>6}")
    print(f"\n日志: {LOG_CSV}")


if __name__ == "__main__":
    main()
