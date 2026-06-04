#!/usr/bin/env python3
"""从 val_list.txt 生成 val_tasks.json，每张图 × 每个 prompt = 一条任务"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_LIST = ROOT / "test" / "train_list_d1.txt"
OUTPUT = ROOT / "test" / "json" / "train(person).json"

PROMPTS = [ "person" ]


def main():
    with open(VAL_LIST, encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    tasks = []
    ann_id = 1
    for img in image_paths:
        for prompt in PROMPTS:
            tasks.append({
                "ann_id": ann_id,
                "image_path": img,
                "text_prompt": prompt,
            })
            ann_id += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"图片数: {len(image_paths)}")
    print(f"prompt 数: {len(PROMPTS)}")
    print(f"任务总数: {len(tasks)}")
    print(f"输出: {OUTPUT}")


if __name__ == "__main__":
    main()
