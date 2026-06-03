#!/usr/bin/env python3
"""
YOLOv11-seg 训练脚本
- 按 train_list.txt / val_list.txt 划分数据集
- 从 SAM3 伪标签（RLE 格式）转为 YOLO polygon 格式
- 支持置信度过滤 + rare class 过采样（仅训练集）+ 分阶段训练
- 所有参数从 config.py 读取
"""
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    MODEL, CLASSES, PRED_JSON, VAL_PRED_JSON,
    TRAIN_LIST, VAL_LIST, IMAGE_ROOT, YOLO_DIR,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR,
    TASK, DEVICE, WORKERS, SEED,
    OPTIMIZER, LRF, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_MOMENTUM, WARMUP_BIAS_LR,
    COS_LR, LABEL_SMOOTHING, DROPOUT,
    CLS_LOSS_GAIN, BOX_LOSS_GAIN,
    MOSAIC, COPY_PASTE, SCALE, DEGREES, FLIPUD, FLIPLR,
    HSV_H, HSV_S, HSV_V, CLOSE_MOSAIC,
    PROJECT, RUN_NAME, EXIST_OK, PATIENCE,
    CONF_FILTER, RARE_OVERSAMPLE,
    STAGE1_EPOCHS, STAGE1_IMG_SIZE, STAGE1_BATCH, STAGE1_LR0,
    STAGE2_EPOCHS, STAGE2_IMG_SIZE, STAGE2_BATCH, STAGE2_LR0,
)

# ── 日志 ──────────────────────────────────────────────────────────────
LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# RLE → Mask 转换
# ══════════════════════════════════════════════════════════════════════

def rle_to_mask(rle):
    """将 COCO RLE 转为二值 numpy mask。"""
    try:
        from pycocotools import mask as maskUtils
        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return maskUtils.decode(rle_cp).astype(np.uint8)
    except ImportError:
        pass
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def mask_to_yolo_polygons(mask):
    """二值 mask → YOLO polygon 列表（归一化坐标）。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = cnt[:, 0, :].astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        polygons.append(pts)
    return polygons


def make_label_lines(by_prompt_per_image):
    """
    一张图的 prompt→pred 映射 → YOLO label 文本行。
    返回 (lines, max_rare_multiplier, n_prompts_kept, n_prompts_filtered)
    """
    lines = []
    rare_classes_seen = set()
    n_kept = 0
    n_filtered = 0

    for prompt in sorted(by_prompt_per_image.keys(), key=lambda x: CLASSES.index(x)):
        class_id = CLASSES.index(prompt)
        if isinstance(CONF_FILTER, dict):
            thresh = CONF_FILTER.get(prompt, 0.7)
        else:
            thresh = CONF_FILTER

        for v in by_prompt_per_image[prompt]:
            rle = v.get("rle")
            score = v.get("score", 1.0)
            if rle is None:
                continue
            if score < thresh:
                n_filtered += 1
                continue
            mask = rle_to_mask(rle)
            polygons = mask_to_yolo_polygons(mask)
            if not polygons:
                n_filtered += 1
                continue
            n_kept += 1
            for poly in polygons:
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                lines.append(f"{class_id} {flat}")

        if any(l.startswith(f"{class_id} ") for l in lines):
            if prompt in RARE_OVERSAMPLE:
                rare_classes_seen.add(prompt)

    max_multiplier = 1
    for rc in rare_classes_seen:
        max_multiplier = max(max_multiplier, RARE_OVERSAMPLE[rc])

    return lines, max_multiplier, n_kept, n_filtered


# ══════════════════════════════════════════════════════════════════════
# 数据集转换
# ══════════════════════════════════════════════════════════════════════

def load_image_list(list_path):
    """读取 train_list.txt / val_list.txt，返回归一化路径集合。"""
    with open(list_path, encoding="utf-8") as f:
        paths = [l.strip().replace("\\", "/") for l in f if l.strip()]
    return set(paths)


def convert_split(split, pred_json, image_list_path, images_dir, labels_dir,
                  do_oversample=True):
    """
    将一个 split 的 SAM3 伪标签转为 YOLO polygon 格式。

    Args:
        split: "train" | "val"
        pred_json: 伪标签 JSON 路径
        image_list_path: train_list.txt 或 val_list.txt
        images_dir: 输出图片目录
        labels_dir: 输出标签目录
        do_oversample: 是否对 rare class 过采样（仅训练集）
    """
    logger.info("=" * 60)
    logger.info(f"转换 {split} 集: {pred_json.name}")
    logger.info("=" * 60)

    # 读取允许的图片列表
    allowed = load_image_list(image_list_path)
    logger.info(f"  允许图片数: {len(allowed)} (来自 {image_list_path.name})")

    # 清理并创建目录
    for d in [images_dir, labels_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 读取伪标签
    with open(pred_json, encoding="utf-8") as f:
        preds = json.load(f)

    # 按图片分组，只保留在列表中的图片
    by_image = defaultdict(lambda: defaultdict(list))
    matched = 0
    unmatched = 0
    for p in preds:
        img_path = p["image_path"].replace("\\", "/")
        # 路径匹配：pred 中可能是 "test/data1/xxx.jpg"，列表中是 "test/data1/xxx.jpg"
        if img_path not in allowed:
            # 尝试去掉 "test/" 前缀再匹配
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allowed:
                unmatched += 1
                continue
            img_path = alt
        matched += 1
        for prompt, v in p["prompts"].items():
            if not v.get("hit") or prompt not in CLASSES:
                continue
            by_image[img_path][prompt].append(v)

    logger.info(f"  伪标签条目: {len(preds)}, 匹配: {matched}, 未匹配: {unmatched}")

    total_unique = len(by_image)
    prompts_kept = 0
    prompts_filtered = 0
    total_polygons = 0
    empty_skipped = 0
    rare_count = 0
    copies_created = 0
    rare_records = []

    # 阶段 A：写入有标签的图片
    img_idx = 0
    pbar = tqdm(by_image.items(), desc=f"  转换{split}", unit="img",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}]")
    for img_path, prompt_masks in pbar:
        label_lines, multiplier, n_kept, n_filtered = make_label_lines(prompt_masks)

        if not label_lines:
            empty_skipped += 1
            continue

        src = IMAGE_ROOT / img_path
        dst = images_dir / f"{img_idx:06d}.jpg"
        if not dst.exists() and src.exists():
            try:
                os.link(src, dst)  # 硬链接，不占额外空间
            except OSError:
                shutil.copy2(src, dst)  # 跨盘回退

        prompts_kept += n_kept
        prompts_filtered += n_filtered
        total_polygons += len(label_lines)

        label_path = labels_dir / f"{img_idx:06d}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if do_oversample and multiplier > 1:
            rare_count += 1
            rare_records.append((img_idx, img_path, multiplier, label_lines))

        img_idx += 1

    total_valid = img_idx

    # 阶段 B：过采样（仅训练集）
    oversample_log = defaultdict(int)
    if do_oversample and rare_records:
        copy_idx = total_valid
        for _, img_path, multiplier, label_lines in rare_records:
            src = IMAGE_ROOT / img_path
            for _ in range(multiplier - 1):
                dst_img = images_dir / f"{copy_idx:06d}.jpg"
                dst_label = labels_dir / f"{copy_idx:06d}.txt"
                if src.exists():
                    try:
                        os.link(src, dst_img)
                    except OSError:
                        shutil.copy2(src, dst_img)
                dst_label.write_text("\n".join(label_lines), encoding="utf-8")
                copy_idx += 1
                copies_created += 1

            rare_in_img = set()
            for l in label_lines:
                if not l.strip():
                    continue
                cls_name = CLASSES[int(l.split()[0])]
                if cls_name in RARE_OVERSAMPLE:
                    rare_in_img.add(cls_name)
            for rc in rare_in_img:
                oversample_log[rc] += multiplier - 1

        total_valid = copy_idx

    # 日志
    logger.info(f"  原始图片: {total_unique}, 空标签跳过: {empty_skipped}")
    total_prompts = prompts_kept + prompts_filtered
    logger.info(f"  prompt: 保留 {prompts_kept}, 过滤 {prompts_filtered} "
                f"({prompts_filtered / max(total_prompts, 1) * 100:.1f}%), "
                f"polygon 总数: {total_polygons}")
    if do_oversample and copies_created:
        logger.info(f"  过采样: +{copies_created} 张 → 最终 {total_valid} 张")
        for cls, n in oversample_log.items():
            logger.info(f"    {cls}: +{n}")
    else:
        logger.info(f"  最终: {total_valid} 张（无过采样）")

    return total_valid


def create_dataset_yaml():
    """生成 dataset.yaml。"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    content = f"""# YOLOv11-seg 数据集配置
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path.write_text(content, encoding="utf-8")
    logger.info(f"dataset.yaml: {yaml_path}")


# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

def train_stage(stage_name, epochs, imgsz, batch, lr0, copy_paste, resume_ckpt=None,
                auto_resume=False):
    """执行单个训练阶段。auto_resume=True 时自动找 last.pt 续训。"""
    run_dir = PROJECT / f"{RUN_NAME}_{stage_name}"
    last_pt = run_dir / "weights" / "last.pt"

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("请先安装 ultralytics: pip install ultralytics")
        return None

    # 断点续训：last.pt 存在则自动继续
    if auto_resume and last_pt.exists():
        logger.info(f"检测到未完成的训练，从 {last_pt} 续训")
        model = YOLO(str(last_pt), task=TASK)
    elif resume_ckpt is not None:
        logger.info(f"从 checkpoint 加载: {resume_ckpt}")
        model = YOLO(str(resume_ckpt), task=TASK)
    else:
        model = YOLO(MODEL, task=TASK)

    logger.info(f"{'─' * 50}")
    logger.info(f"阶段: {stage_name} | epochs={epochs} | imgsz={imgsz} | "
                f"batch={batch} | lr0={lr0} | copy_paste={copy_paste}")
    if auto_resume and last_pt.exists():
        logger.info(f"续训模式: 从上次断点继续")
    logger.info(f"{'─' * 50}")

    t_start = time.time()
    results = model.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        optimizer=OPTIMIZER,
        lr0=lr0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        cos_lr=COS_LR,
        label_smoothing=LABEL_SMOOTHING,
        dropout=DROPOUT,
        cls=CLS_LOSS_GAIN,
        box=BOX_LOSS_GAIN,
        mosaic=MOSAIC,
        copy_paste=copy_paste,
        scale=SCALE,
        degrees=DEGREES,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        close_mosaic=CLOSE_MOSAIC,
        project=str(PROJECT),
        name=f"{RUN_NAME}_{stage_name}",
        exist_ok=EXIST_OK,
        patience=PATIENCE,
        save=True,
        plots=True,
    )
    elapsed = time.time() - t_start
    logger.info(f"阶段耗时: {elapsed / 60:.1f} 分钟")

    run_dir = PROJECT / f"{RUN_NAME}_{stage_name}"
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        logger.info(f"best.pt: {best_pt}")
    else:
        logger.warning(f"未找到 {best_pt}")

    return best_pt


def train():
    """分阶段训练，支持断点续训。"""
    stage1_dir = PROJECT / f"{RUN_NAME}_stage1"
    stage2_dir = PROJECT / f"{RUN_NAME}_stage2"
    stage1_best = stage1_dir / "weights" / "best.pt"
    stage2_best = stage2_dir / "weights" / "best.pt"

    # 阶段二已完成 → 跳过
    if stage2_best.exists():
        logger.info("=" * 60)
        logger.info("训练已完成！")
        logger.info(f"  最终模型: {stage2_best}")
        logger.info(f"  训练日志: {LOG_FILE}")
        return stage2_best

    # 阶段一已完成，阶段二未开始
    if stage1_best.exists():
        logger.info("阶段一已完成，直接进入阶段二")
    else:
        logger.info("=" * 60)
        logger.info("阶段一：主训练")
        logger.info("=" * 60)
        result = train_stage(
            stage_name="stage1",
            epochs=STAGE1_EPOCHS,
            imgsz=STAGE1_IMG_SIZE,
            batch=STAGE1_BATCH,
            lr0=STAGE1_LR0,
            copy_paste=COPY_PASTE,
            resume_ckpt=None,
            auto_resume=True,           # 中断后续训
        )
        if result is None or not stage1_best.exists():
            logger.error("阶段一失败，终止")
            return None

    # 阶段二
    logger.info("=" * 60)
    logger.info("阶段二：精细收敛")
    logger.info("=" * 60)
    stage2_best_result = train_stage(
        stage_name="stage2",
        epochs=STAGE2_EPOCHS,
        imgsz=STAGE2_IMG_SIZE,
        batch=STAGE2_BATCH,
        lr0=STAGE2_LR0,
        copy_paste=0.0,
        resume_ckpt=stage1_best,
        auto_resume=True,
    )

    logger.info("=" * 60)
    logger.info("训练完成！")
    final = stage2_best_result if (stage2_best_result and stage2_best_result.exists()) else stage1_best
    logger.info(f"最终模型: {final}")
    logger.info(f"训练日志: {LOG_FILE}")
    return final


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    # 强制 GPU
    import torch
    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA，请安装 CUDA 版 PyTorch：")
        logger.error("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("=" * 60)
    logger.info("YOLOv11-seg 训练流程 v4")
    logger.info(f"  模型: {MODEL}")
    logger.info(f"  类别 ({len(CLASSES)}): {CLASSES}")
    logger.info(f"  过滤阈值: {CONF_FILTER}")
    logger.info(f"  过采样: {RARE_OVERSAMPLE}")
    logger.info(f"  图片尺寸: {STAGE1_IMG_SIZE}")
    logger.info(f"  阶段一: {STAGE1_EPOCHS}ep, lr0={STAGE1_LR0}")
    logger.info(f"  阶段二: {STAGE2_EPOCHS}ep, lr0={STAGE2_LR0}")
    logger.info(f"  随机种子: {SEED}")
    logger.info("=" * 60)

    # 检查必要文件
    for fpath in [PRED_JSON, TRAIN_LIST]:
        if not fpath.exists():
            logger.error(f"缺少文件: {fpath}")
            return

    # ── 1. 转换数据集（如已存在则跳过） ──
    dataset_yaml = YOLO_DIR / "dataset.yaml"
    if dataset_yaml.exists():
        logger.info("数据集已存在，跳过转换")
        # 统计已有图片数
        n_train = len(list(TRAIN_IMAGES_DIR.glob("*.jpg"))) if TRAIN_IMAGES_DIR.exists() else 0
        n_val = len(list(VAL_IMAGES_DIR.glob("*.jpg"))) if VAL_IMAGES_DIR.exists() else 0
        logger.info(f"  训练集: {n_train} 张, 验证集: {n_val} 张")
    else:
        n_train = convert_split(
            split="train",
            pred_json=PRED_JSON,
            image_list_path=TRAIN_LIST,
            images_dir=TRAIN_IMAGES_DIR,
            labels_dir=TRAIN_LABELS_DIR,
            do_oversample=True,
        )

        if VAL_PRED_JSON.exists():
            n_val = convert_split(
                split="val",
                pred_json=VAL_PRED_JSON,
                image_list_path=VAL_LIST,
                images_dir=VAL_IMAGES_DIR,
                labels_dir=VAL_LABELS_DIR,
                do_oversample=False,
            )
        else:
            logger.warning(f"验证集伪标签不存在: {VAL_PRED_JSON}")
            logger.warning("将使用训练集作为验证集（无法检测过拟合）")
            n_val = 0

        create_dataset_yaml()

    # ── 4. 训练 ──
    logger.info(f"\n训练集: {n_train} 张, 验证集: {n_val} 张")
    train()

    # ── 5. 输出清单 ──
    logger.info("\n" + "=" * 60)
    logger.info("输出文件:")
    logger.info("=" * 60)
    for stage in ["stage1", "stage2"]:
        run_dir = PROJECT / f"{RUN_NAME}_{stage}"
        if run_dir.exists():
            logger.info(f"\n  {run_dir}/")
            for fname in ["weights/best.pt", "results.csv", "results.png",
                          "confusion_matrix.png", "val_batch0_pred.jpg",
                          "val_batch0_labels.jpg", "labels.jpg", "args.yaml"]:
                fpath = run_dir / fname
                status = "✓" if fpath.exists() else "✗"
                logger.info(f"    {status} {fname}")
    logger.info(f"\n  训练日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
