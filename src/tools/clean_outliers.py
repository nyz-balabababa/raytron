#!/usr/bin/env python3
"""
数据清洗脚本：检测并标记非红外图像（彩色图、截图、带准星图）。
输出 CSV 标记每张图片的异常类型，不移动任何文件。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = ROOT / "test"
OUTPUT = ROOT / "test" / "clean_outliers.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 阈值
SAT_THRESH = 30         # 均值饱和度 > 此值 → 彩色图
CROSSHAIR_MIN_LEN = 80  # 准星线最小长度
CROSSHAIR_CENTER = 0.15  # 准星线距图像中心容差（比例）
EDGE_DENSITY_THRESH = 0.08  # 边缘像素占比 → 截图/text
BORDER_EDGE_RATIO = 0.3  # 四边边缘占比 → 截图边框


def detect_color(img_bgr):
    """检测是否为彩色图像（饱和度均值高）"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = np.mean(hsv[:, :, 1])
    return sat > SAT_THRESH, float(sat)


def detect_crosshair(gray):
    """检测中心准星：找交叉在图像中心附近的直线"""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=CROSSHAIR_MIN_LEN, maxLineGap=5)
    if lines is None:
        return False, 0

    cx, cy = w // 2, h // 2
    margin = int(min(w, h) * CROSSHAIR_CENTER)
    cross_count = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        # 线段是否经过中心区域
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / \
               np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2 + 1e-6)
        if dist < margin:
            cross_count += 1
    return cross_count >= 2, cross_count


def detect_screenshot(gray):
    """检测截图：高边缘密度 + 四边有边框"""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (h * w)

    # 四边 5px 区域边缘密度
    border = np.concatenate([edges[:5, :].flatten(), edges[-5:, :].flatten(),
                             edges[:, :5].flatten(), edges[:, -5:].flatten()])
    border_ratio = np.sum(border > 0) / max(len(border), 1)

    return edge_ratio > EDGE_DENSITY_THRESH and border_ratio > BORDER_EDGE_RATIO, edge_ratio


def detect_text_overlay(gray):
    """检测是否有文字/UI叠加：高频区域集中在顶部/底部"""
    h, w = gray.shape
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # 顶部 15% 和底部 15% 的拉普拉斯能量 vs 中间
    top_h = int(h * 0.15)
    bot_start = int(h * 0.85)
    top_energy = np.mean(lap_abs[:top_h, :])
    bot_energy = np.mean(lap_abs[bot_start:, :])
    mid_energy = np.mean(lap_abs[top_h:bot_start, :])
    ratio = max(top_energy, bot_energy) / max(mid_energy, 1e-6)
    return ratio > 2.5, float(ratio)


def main():
    # 扫描 test/ 下所有 data* 子文件夹的图片
    paths = []
    for d in sorted(TEST_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("data"):
            for f in d.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    paths.append(str(f.relative_to(ROOT)).replace("\\", "/"))

    total = len(paths)
    print(f"分析 {total} 张图片...\n")

    stats = defaultdict(int)
    rows = []

    for i, rel in enumerate(paths):
        abs_path = ROOT / rel
        img = cv2.imread(str(abs_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        is_color, sat = detect_color(img)
        is_cross, cross_n = detect_crosshair(gray)
        is_shot, edge_r = detect_screenshot(gray)
        is_text, text_r = detect_text_overlay(gray)

        flags = []
        if is_color: flags.append("color")
        if is_text: flags.append("text_overlay")
        # crosshair / screenshot 暂不标记，仅保留检测值到 CSV

        for fg in flags:
            stats[fg] += 1

        row = {
            "path": rel,
            "w": w, "h": h,
            "sat": round(sat, 1),
            "is_color": is_color,
            "crosshair_lines": cross_n,
            "is_crosshair": is_cross,
            "edge_ratio": round(edge_r, 4),
            "is_screenshot": is_shot,
            "text_ratio": round(text_r, 2),
            "is_text_overlay": is_text,
            "flags": ";".join(flags),
        }
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>5}/{total}] {(i+1)/total*100:.0f}%  "
                  f"color={stats['color']} text={stats['text_overlay']}")

    # 输出 CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(1 for r in rows if r["flags"])
    flagged_paths = [r["path"] for r in rows if r["flags"]]

    # 输出标记图片的 txt（可直接用 vis_val_browser.py 浏览）
    flagged_txt = TEST_DIR / "clean_outliers_list.txt"
    with open(flagged_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(flagged_paths) + "\n")

    print(f"\n{'='*50}")
    print(f"总图片: {total}")
    print(f"异常图片: {flagged} ({flagged/total*100:.1f}%)")
    print(f"  - 彩色图:     {stats['color']}")
    print(f"  - 文字/UI:    {stats['text_overlay']}")
    print(f"\nCSV: {OUTPUT}")
    print(f"异常图片列表: {flagged_txt}  ← 可直接复制到 vis_val_browser.py 的 VAL_LIST 查看")


if __name__ == "__main__":
    main()
