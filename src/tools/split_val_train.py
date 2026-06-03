#!/usr/bin/env python3
"""
按命名规则抽验证集 -> val_list.txt / train_list.txt
- data3, data4: 全部入验证集
- data1, data5, data6, data7: 按规则内图片数抽
    ≥ 100 张 → 抽 7% (5%~10% 中间值)
    20~99 张 → 抽 20 张 或 20%(取大)
    < 20 张 → 全部入验证集
- 不移动图片，仅写路径列表
"""
import csv
import re
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "test"
PATTERN_CSV = ROOT / "clean_filename_analysis.csv"
VAL_LIST = ROOT / "val_list.txt"
TRAIN_LIST = ROOT / "train_list.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# data3, data4 全部入验证集
ALL_VAL_FOLDERS = {"data3", "data4"}
# 按规则抽样的文件夹
SAMPLE_FOLDERS = {"data1", "data5", "data6", "data7"}

SEED = 42


def normalize(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def parse_csv(path: Path) -> dict:
    """解析 clean_filename_analysis.csv，返回 {folder: {digit_pattern: count}}"""
    result = defaultdict(dict)
    current_folder = None
    in_digit_section = False

    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0]:
                continue
            first = row[0]
            m = re.match(r"=== (\S+) —— 数字归一化模式", first)
            if m:
                current_folder = m.group(1)
                in_digit_section = True
                continue
            if first.startswith("=== ") and "数字归一化模式" not in first:
                in_digit_section = False
                continue

            if in_digit_section and current_folder and len(row) >= 2:
                pattern = row[0].strip()
                if pattern and pattern != "模式 (数字→#)" and not pattern.startswith("==="):
                    try:
                        result[current_folder][pattern] = int(row[1])
                    except ValueError:
                        pass
    return dict(result)


def sample_indices(total: int) -> set:
    """根据规则返回验证集索引集合"""
    if total >= 100:
        k = max(1, int(total * 0.07))
    elif total >= 20:
        k = max(10, min(20, int(total * 0.2)))
    else:
        k = total
    chosen = random.sample(range(total), min(k, total))
    return set(chosen)


def main():
    random.seed(SEED)

    # 1. 解析规则
    rule_counts = parse_csv(PATTERN_CSV)
    if not rule_counts:
        print("错误: 无法解析 CSV")
        return

    for f in SAMPLE_FOLDERS:
        if f not in rule_counts:
            print(f"警告: {f} 在 CSV 中无规则记录")

    # 2. 扫描 test/ 下所有图片，按 (folder, digit_pattern) 分组
    groups = defaultdict(list)  # (folder, dp) → [relative_path]
    all_val_folder_images = []  # data3, data4 的图片

    for d in sorted(TEST.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in d.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            dp = normalize(f.stem)

            if d.name in ALL_VAL_FOLDERS:
                all_val_folder_images.append(rel)
            elif d.name in SAMPLE_FOLDERS:
                groups[(d.name, dp)].append(rel)

    # 3. 抽样
    val_set = set(all_val_folder_images)
    val_details = defaultdict(list)
    train_set = set()

    for (folder, dp), images in sorted(groups.items()):
        # 找规则中的数量
        known_count = rule_counts.get(folder, {}).get(dp, len(images))
        idxs = sample_indices(len(images))

        for i, img in enumerate(images):
            if i in idxs:
                val_set.add(img)
                val_details[folder].append(img)
            else:
                train_set.add(img)

        if len(images) >= 20:
            count_label = f"≥100({len(images)})" if len(images) >= 100 else f"20~99({len(images)})"
            print(f"  {folder}/{dp}: {count_label} → val={len(idxs)}, train={len(images)-len(idxs)}")

    # 对 data3/data4 只统计
    for f in ALL_VAL_FOLDERS:
        imgs = [i for i in all_val_folder_images if i.startswith(f"test/{f}/")]
        print(f"  {f}/: ALL → val={len(imgs)}, train=0")

    # 4. 写入
    val_list = sorted(val_set)
    train_list = sorted(train_set)

    for path, lst in [(VAL_LIST, val_list), (TRAIN_LIST, train_list)]:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lst) + "\n")
        print(f"\n{path}: {len(lst)} 张")

    print(f"\n总计: val={len(val_set)}, train={len(train_set)}, "
          f"val/total={len(val_set)/(len(val_set)+len(train_set))*100:.1f}%")


if __name__ == "__main__":
    main()
