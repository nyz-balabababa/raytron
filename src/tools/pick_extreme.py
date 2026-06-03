#!/usr/bin/env python3
"""从质量分析报告中提取极端规则，在 val_list.txt 中每种规则抽取5张"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "质量分析报告.md"
VAL_LIST = ROOT / "test" / "val_list.txt"
OUTPUT = ROOT / "test" / "extreme_val_list.txt"

# 从报告表格中提取极端规则（文件夹, 数字归一化模式）
EXTREME_RULES = [
    # 最暗
    ("data5", "ir_p_bw_seg#_part#_tomx_pic#_rename_#.avi_#"),
    ("data7", "#_CH#_#_M#S#_out_I_#_#"),
    ("data7", "#_SF#_#_M#S_grass_shake_A_#_#"),
    ("data7", "#_SF#_#_M#S_grass_still_A_#_#"),
    ("data6", "bw_seg#-#_Mike Roberts e#_#"),
    # 最亮
    ("data5", "ir_p_bw_seg#lgq_AKM-CLOS-#_#"),
    ("data5", "ir_p_bw_seg#-#_#_#_A_#_pair"),
    ("data5", "ir_p_bw_oriseg_#_pair"),
    ("data5", "ir_p_bw_seg#_#_pair"),
    ("data5", "ir_p_bw_oriseg_#_#_#_#_pair"),
    # 对比度最低
    ("data6", "bw_seg#-#_TL#_#"),
    # SNR 最低 (原报告中的 D/C/G 在 val_list 中不存在，用同类型最近似规则替换)
    ("data7", "#_SF#_#_M#S_grass_#m_shake_J_#_#"),
    ("data7", "#_SF#_#_M#S_grass_#m_shake_I_#_#"),
    ("data7", "#_SF#_#_M#S_grass_#m_shake_F_#_#"),
]


def normalize(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def main():
    with open(VAL_LIST, encoding="utf-8") as f:
        images = [line.strip() for line in f if line.strip()]

    # 按 (folder, dp) 分组
    groups = {}
    for img in images:
        parts = img.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        folder = parts[1]  # test/data1/xxx.jpg → data1
        stem = Path(img).stem
        dp = normalize(stem)
        key = (folder, dp)
        groups.setdefault(key, []).append(img)

    # 去重（同一规则可能出现在多个极端类别）
    unique_rules = list(dict.fromkeys(EXTREME_RULES))

    # 每种极端规则取前5张
    picked = []
    for folder, dp in unique_rules:
        imgs = groups.get((folder, dp), [])
        if not imgs:
            print(f"  [!] 未找到: {folder}/{dp}")
            continue
        sample = imgs[:5]
        picked.extend(sample)
        print(f"  ✓ {folder}/{dp}: 取 {len(sample)} 张 (val_list 中共 {len(imgs)} 张)")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(picked) + "\n")

    print(f"\n输出: {OUTPUT}")
    print(f"极端规则: {len(EXTREME_RULES)} 条 (去重后 {len(unique_rules)} 种)")
    print(f"总图片: {len(picked)} 张")


if __name__ == "__main__":
    main()
