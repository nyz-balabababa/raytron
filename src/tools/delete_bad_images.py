#!/usr/bin/env python3
"""
按质量指标筛选并剪贴图片:
  - 极小分辨率: 92×92, 384×288 (含 256×192)
  - 严重低 SNR: SNR < 10dB 的规则
从 test/dataN/ 剪贴到 data/dataN/，保留原路径不存在则新建同名文件夹。
"""
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "test"
DEST = ROOT / "data"
ANALYSIS = ROOT / "image_quality_analysis.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def normalize(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def print_tree(base: Path, prefix: str = ""):
    """打印目标目录树结构"""
    for p in sorted(base.iterdir()):
        if p.is_dir():
            count = len(list(p.iterdir()))
            print(f"  {prefix}{p.name}/ ({count} 张)")
        else:
            print(f"  {prefix}{p.name}")


def main():
    # 1. 读取 CSV 筛选目标规则
    with open(ANALYSIS, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    target_rules = set()  # (folder, digit_pattern)
    for r in rows:
        w = int(float(r["width_median"])) if r["width_median"] else 0
        snr = float(r["snr_db_mean"]) if r["snr_db_mean"] else 99
        folder = r["文件夹"]
        dp = r["数字归一化模式"]
        if w <= 384:
            target_rules.add((folder, dp))
        if snr < 10:
            target_rules.add((folder, dp))

    print(f"目标规则数: {len(target_rules)}")
    for f, dp in sorted(target_rules):
        w = "?"
        snr = "?"
        for r in rows:
            if r["文件夹"] == f and r["数字归一化模式"] == dp:
                w = int(float(r["width_median"])) if r["width_median"] else 0
                snr = float(r["snr_db_mean"]) if r["snr_db_mean"] else 99
                break
        reason = []
        if w <= 384:
            reason.append(f"{w}px")
        if snr < 10:
            reason.append(f"SNR={snr:.1f}")
        print(f"  {f}/{dp}  ({', '.join(reason)}), {r.get('count','?')} 张")

    # 2. 扫描现存图片
    existing = []
    for d in sorted(TEST.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                existing.append((d.name, normalize(f.stem), f))

    print(f"\n现存图片: {len(existing)} 张")

    # 3. 匹配目标规则
    matched = []
    for folder, dp, path in existing:
        if (folder, dp) in target_rules:
            matched.append((folder, dp, path))

    print(f"命中: {len(matched)} 张")

    # 按文件夹统计
    by_folder = defaultdict(list)
    for folder, dp, path in matched:
        by_folder[folder].append(path)

    # 4. 移动
    total_moved = 0
    for folder, files in sorted(by_folder.items()):
        dst_dir = DEST / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = dst_dir / src.name
            if dst.exists():
                print(f"  [跳过已存在] {dst.name}")
                continue
            shutil.move(str(src), str(dst))
            total_moved += 1
        print(f"  {folder}/: {len(files)} 张 → {dst_dir}")

    print(f"\n总共移动: {total_moved} 张")
    print(f"目标目录: {DEST}")
    print()
    print_tree(DEST)
    print("\n完成。")  # 注释掉末尾多余文本


if __name__ == "__main__":
    main()
