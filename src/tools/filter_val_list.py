#!/usr/bin/env python3
"""从 val_list.txt 中筛选指定文件夹的路径，输出到新 txt"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_LIST = ROOT / "test" / "train_list.txt"

# 要筛选的文件夹，按需修改
TARGET_FOLDERS = ["data6"]
OUTPUT = ROOT / "test" / "txt" / "train_list_d6.txt"


def main():
    paths = []
    with open(VAL_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            folder = line.split("/")[1]  # test/data5/xxx.jpg → data5
            if folder in TARGET_FOLDERS:
                paths.append(line)

    OUTPUT.write_text("\n".join(paths) + "\n", encoding="utf-8")

    from collections import Counter
    cnt = Counter(p.split("/")[1] for p in paths)
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    print(f"总计: {len(paths)} 张")
    print(f"输出: {OUTPUT}")


if __name__ == "__main__":
    main()
