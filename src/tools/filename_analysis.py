#!/usr/bin/env python3
"""
文件名规则分析脚本
扫描 test/ 下所有子文件夹，自动发现图片文件，提取命名规则并统计数量。
输出 CSV 文件到当前目录。
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── 配置 ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_CSV = PROJECT_ROOT / "clean_filename_analysis.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── 模式提取 ───────────────────────────────────────────────────────

def normalize_digits(name: str) -> str:
    """将连续数字替换为 #，保留其他所有字符。"""
    return re.sub(r"\d+", "#", name)


def extract_semantic_pattern(name: str) -> str:
    """
    从文件名中提取语义模式，将已知结构替换为语义标签。
    返回便于人类阅读的模式字符串。
    """
    # 时间戳: 20230817162746, 20250613142016 等 (14或8位连续数字在一定上下文)
    name = re.sub(r"(?<![#\d])\d{14}(?!\d)", "[TS14]", name)  # 14位时间戳
    name = re.sub(r"(?<![#\d])\d{8}(?!\d)", "[TS8]", name)    # 8位日期

    # 长数字序列 (>=6位) → 可能是序号/ID
    name = re.sub(r"\d{7,}", "[SEQ]", name)
    name = re.sub(r"\d{4,6}", "[ID]", name)
    name = re.sub(r"\d{1,3}", "[N]", name)

    # 多余占位符合并
    name = re.sub(r"\[N\](\[N\])+", "[N+]", name)
    name = re.sub(r"\[N\+\]\[N\+\]", "[N+]", name)

    return name


# ── 统计分析 ───────────────────────────────────────────────────────

def analyze_folder(folder_path: Path) -> dict:
    """
    分析单个文件夹内的所有图片文件名。
    返回: {
        "folder": str,
        "total_files": int,
        "extensions": {ext: count},
        "digit_patterns": {pattern: count},       # 纯数字→# 归一化
        "semantic_patterns": {pattern: count},    # 语义标签归一化
        "prefixes": {prefix: count},              # 前缀统计 (第一个 _ 之前)
    }
    """
    files = []
    for f in folder_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f.stem)

    if not files:
        return None

    ext_counts = defaultdict(int)
    digit_patterns = defaultdict(int)
    semantic_patterns = defaultdict(int)
    prefix_counts = defaultdict(int)

    for stem in files:
        # 扩展名
        # (这里我们只知道 stem，实际扩展名需要从文件系统读)
        pass

    for f in folder_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            ext_counts[f.suffix.lower()] += 1

    for stem in files:
        # 数字归一化
        dp = normalize_digits(stem)
        digit_patterns[dp] += 1

        # 语义模式
        sp = extract_semantic_pattern(stem)
        semantic_patterns[sp] += 1

        # 前缀 (第一个 _ 或 - 之前的部分)
        prefix = re.split(r"[_\-]", stem)[0]
        prefix_counts[prefix] += 1

    return {
        "folder": folder_path.name,
        "total_files": len(files),
        "extensions": dict(sorted(ext_counts.items(), key=lambda x: -x[1])),
        "digit_patterns": dict(sorted(digit_patterns.items(), key=lambda x: -x[1])),
        "semantic_patterns": dict(sorted(semantic_patterns.items(), key=lambda x: -x[1])),
        "prefixes": dict(sorted(prefix_counts.items(), key=lambda x: -x[1])),
    }


# ── CSV 输出 ───────────────────────────────────────────────────────

def write_csv(all_results: list, output_path: Path):
    """将分析结果写入多 sheet 风格的 CSV 文件（用分区标题行分隔）。"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # ── Sheet 1: 概览 ──
        writer.writerow(["=== 总体概览 ==="])
        writer.writerow(["文件夹", "图片总数", "扩展名分布", "主要前缀"])
        for r in all_results:
            ext_str = "; ".join(f"{k}({v})" for k, v in r["extensions"].items())
            top_prefixes = "; ".join(
                f"{k}({v})" for k, v in list(r["prefixes"].items())[:5]
            )
            writer.writerow([r["folder"], r["total_files"], ext_str, top_prefixes])
        writer.writerow([])

        # ── Sheet 2: 各文件夹 - 数字归一化模式 ──
        for r in all_results:
            writer.writerow([f"=== {r['folder']} —— 数字归一化模式 (共 {r['total_files']} 张) ==="])
            writer.writerow(["模式 (数字→#)", "数量", "占比"])
            for pattern, count in r["digit_patterns"].items():
                pct = f"{count / r['total_files'] * 100:.1f}%"
                writer.writerow([pattern, count, pct])
            writer.writerow([])

        # ── Sheet 3: 各文件夹 - 语义模式 ──
        for r in all_results:
            writer.writerow([f"=== {r['folder']} —— 语义模式 (共 {r['total_files']} 张) ==="])
            writer.writerow(["语义模式", "数量", "占比"])
            for pattern, count in r["semantic_patterns"].items():
                pct = f"{count / r['total_files'] * 100:.1f}%"
                writer.writerow([pattern, count, pct])
            writer.writerow([])

        # ── Sheet 4: 各文件夹 - 前缀分布 ──
        for r in all_results:
            writer.writerow([f"=== {r['folder']} —— 前缀分布 (共 {r['total_files']} 张) ==="])
            writer.writerow(["前缀", "数量", "占比"])
            for prefix, count in r["prefixes"].items():
                pct = f"{count / r['total_files'] * 100:.1f}%"
                writer.writerow([prefix, count, pct])
            writer.writerow([])

    print(f"CSV 已输出: {output_path}")


def print_summary(all_results: list):
    """终端友好摘要。"""
    total_all = sum(r["total_files"] for r in all_results)
    print(f"\n{'='*70}")
    print(f"总览: {len(all_results)} 个文件夹, 共 {total_all} 张图片")
    print(f"{'='*70}")
    for r in all_results:
        unique_patterns = len(r["digit_patterns"])
        top3 = list(r["digit_patterns"].items())[:3]
        print(f"\n  [{r['folder']}] {r['total_files']} 张, {unique_patterns} 种命名模式")
        for pat, cnt in top3:
            pct = cnt / r["total_files"] * 100
            bar = "█" * int(pct / 5)
            print(f"    {pct:5.1f}% {bar} {pat}")
        if unique_patterns > 3:
            print(f"    ... 还有 {unique_patterns - 3} 种模式")


# ── 主入口 ─────────────────────────────────────────────────────────

def main():
    if not TEST_DIR.exists():
        print(f"错误: 找不到 test 目录 ({TEST_DIR})")
        sys.exit(1)

    folders = sorted(
        d for d in TEST_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not folders:
        print(f"错误: {TEST_DIR} 下没有子文件夹")
        sys.exit(1)

    print(f"扫描目录: {TEST_DIR}")
    print(f"发现 {len(folders)} 个子文件夹: {[f.name for f in folders]}")

    all_results = []
    for folder in folders:
        result = analyze_folder(folder)
        if result is None:
            print(f"  [跳过] {folder.name} (无图片文件)")
            continue
        all_results.append(result)
        print(f"  ✓ {folder.name}: {result['total_files']} 张图片")

    print_summary(all_results)
    write_csv(all_results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
