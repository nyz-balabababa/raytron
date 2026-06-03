#!/usr/bin/env python3
"""
YOLO-World + FastSAM 两阶段训练与评估
阶段一：YOLO-World v2 检测（bbox）—— RLE 伪标签 → YOLO bbox 格式
阶段二：FastSAM 零样本分割 —— 量化两阶段 mask mIoU
"""
import os
os.environ["ULTRALYTICS_DOWNLOADS"] = "false"  # 禁止自动下载不需要的默认权重

import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config_world import (
    YOLO_MODEL, Fastsam_MODEL, CLASSES,
    PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST, IMAGE_ROOT, YOLO_DIR,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR,
    YOLO_TASK, IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    OPTIMIZER, LR0, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, COS_LR, PATIENCE,
    MOSAIC, MIXUP, COPY_PASTE, SCALE, DEGREES, FLIPUD, FLIPLR,
    HSV_H, HSV_S, HSV_V, CLOSE_MOSAIC,
    Fastsam_IMG_SIZE, Fastsam_CONF, Fastsam_IOU,
    PROJECT, RUN_NAME, EXIST_OK,
    CONF_FILTER, RARE_OVERSAMPLE, PROMPT_MAP,
)

LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# RLE 解码 + 反色统一
# ══════════════════════════════════════════════════════════════════════

def rle_to_mask(rle):
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
    if isinstance(counts, bytes): counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = val = 0
    for run_len in counts:
        if val == 1: mask[pos:pos + run_len] = 1
        pos += run_len; val = 1 - val
    return mask.reshape((h, w), order="F")


def unify_polarity(gray, fname):
    if "blackHot" in fname:
        return 255 - gray
    mean = float(np.mean(gray)); std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < -0.3: return 255 - gray
    if mean > 200 and "vis" not in fname.lower():
        return 255 - gray
    return gray


# ══════════════════════════════════════════════════════════════════════
# Mask → bbox（连通域拆分，每个实例一个框）
# ══════════════════════════════════════════════════════════════════════

def mask_to_bboxes(mask):
    """二值 mask → 归一化 bbox 列表。YOLO 检测格式: cx cy w h。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    bboxes = []
    for cnt in contours:
        if len(cnt) < 3: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        bboxes.append((cx, cy, nw, nh))
    return bboxes


# ══════════════════════════════════════════════════════════════════════
# 数据集转换（bbox 格式）
# ══════════════════════════════════════════════════════════════════════

def load_file_list(path):
    with open(path, encoding="utf-8") as f:
        return {l.strip().replace("\\", "/") for l in f if l.strip()}


def convert_split(split, pred_json, allow_set, images_dir, labels_dir, do_oversample=True):
    """RLE → bbox 标签（YOLO 检测格式: class_id cx cy w h）。"""
    logger.info(f"{'─' * 50}\n转换 {split} 集: {pred_json.name}")
    for d in [images_dir, labels_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    with open(pred_json, encoding="utf-8") as f:
        preds = json.load(f)

    by_image = defaultdict(lambda: defaultdict(list))
    for p in preds:
        img_path = p["image_path"].replace("\\", "/")
        if img_path not in allow_set:
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allow_set: continue
            img_path = alt
        for prompt, v in p["prompts"].items():
            if not v.get("hit") or prompt not in CLASSES: continue
            if v.get("rle") is None: continue
            thresh = CONF_FILTER.get(prompt, 0.7)
            if v.get("score", 1.0) < thresh: continue
            by_image[img_path][prompt].append(v)

    total_unique = len(by_image)
    prompts_kept = prompts_filtered = empty_skipped = 0
    rare_records = []
    img_idx = 0

    pbar = tqdm(by_image.items(), desc=f"  转换{split}", unit="img")
    for img_path, prompt_dict in pbar:
        label_lines = []
        has_rare = False

        for prompt in sorted(prompt_dict.keys(), key=lambda x: CLASSES.index(x)):
            class_id = CLASSES.index(prompt)
            for v in prompt_dict[prompt]:
                mask = rle_to_mask(v["rle"])
                bboxes = mask_to_bboxes(mask)
                if not bboxes:
                    prompts_filtered += 1; continue
                prompts_kept += 1
                if prompt in RARE_OVERSAMPLE:
                    has_rare = True
                for cx, cy, bw, bh in bboxes:
                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            empty_skipped += 1; continue

        src = IMAGE_ROOT / img_path
        dst = images_dir / f"{img_idx:06d}.jpg"
        if not dst.exists() and src.exists():
            img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img_gray is not None:
                img_gray = unify_polarity(img_gray, os.path.basename(img_path))
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(str(dst), img_rgb)

        (labels_dir / f"{img_idx:06d}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        if do_oversample and has_rare:
            multiplier = max(RARE_OVERSAMPLE.get(p, 1) for p in prompt_dict)
            rare_records.append((img_idx, img_path, multiplier, label_lines))
        img_idx += 1

    total_valid = img_idx
    oversample_log = defaultdict(int)
    if do_oversample and rare_records:
        copy_idx = total_valid
        for _, img_path, multiplier, label_lines in rare_records:
            src = IMAGE_ROOT / img_path
            for _ in range(multiplier - 1):
                dst_img = images_dir / f"{copy_idx:06d}.jpg"
                dst_label = labels_dir / f"{copy_idx:06d}.txt"
                if src.exists():
                    img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                    if img_gray is not None:
                        img_gray = unify_polarity(img_gray, os.path.basename(img_path))
                        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(str(dst_img), img_rgb)
                dst_label.write_text("\n".join(label_lines), encoding="utf-8")
                copy_idx += 1
            for l in label_lines:
                cls = CLASSES[int(l.split()[0])]
                if cls in RARE_OVERSAMPLE:
                    oversample_log[cls] += multiplier - 1
        total_valid = copy_idx

    logger.info(f"  图片: {total_unique} → {total_valid} 张, 跳过空: {empty_skipped}")
    logger.info(f"  实例保留: {prompts_kept}, 过滤: {prompts_filtered}")
    if oversample_log:
        logger.info(f"  过采样: {dict(oversample_log)}")
    return total_valid


def create_dataset_yaml():
    yaml_path = YOLO_DIR / "dataset.yaml"
    content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path.write_text(content, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# 两阶段推理：YOLO-World 检测 → FastSAM 分割 → 合并掩码
# ══════════════════════════════════════════════════════════════════════

def rle_encode(mask):
    """二值 mask → COCO RLE dict。"""
    try:
        from pycocotools import mask as maskUtils
        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except ImportError:
        pass
    # 简单 RLE 编码回退
    h, w = mask.shape
    flat = mask.flatten(order="F")
    counts = []
    prev = 0; run = 0
    for v in flat:
        if v == prev: run += 1; continue
        counts.append(str(run)); prev = v; run = 1
    counts.append(str(run))
    return {"size": [h, w], "counts": ",".join(counts)}


def compute_iou(pred_mask, gt_mask):
    """两个二值 mask 的 IoU。"""
    pred = pred_mask.astype(bool); gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-7)


def two_stage_eval(yolo_pt, pred_json, image_list_path, max_samples=None):
    """
    两阶段评估（同时加载 YOLO + FastSAM，请确保显存 ≥ 12GB）：
    1. YOLO-World 检测 → bbox
    2. FastSAM 逐框分割 → 合并 mask
    3. 与 SAM3 伪标签比对 mIoU + 同义词泛化测试
    """
    from ultralytics import YOLO, FastSAM

    # ── 加载模型 ──
    yolo = YOLO(str(yolo_pt), task="detect")
    yolo.to(f"cuda:{DEVICE}")  # 确保 CLIP 文本编码器在 GPU 上
    if not os.path.exists(Fastsam_MODEL):
        logger.info(f"FastSAM 权重不存在，尝试自动下载: {Fastsam_MODEL}")
    fastsam = FastSAM(Fastsam_MODEL)
    logger.info(f"YOLO-World + FastSAM 已加载，显存峰值约 12GB")

    # ── 加载验证集伪标签 ──
    with open(pred_json, encoding="utf-8") as f: preds = json.load(f)
    with open(image_list_path, encoding="utf-8") as f:
        allowed = {l.strip().replace("\\", "/") for l in f if l.strip()}

    by_image = defaultdict(dict)
    for p in preds:
        img_path = p["image_path"].replace("\\", "/")
        if img_path not in allowed:
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allowed: continue
            img_path = alt
        if max_samples and len(by_image) >= max_samples: break
        for prompt, v in p["prompts"].items():
            if not v.get("hit") or prompt not in CLASSES: continue
            if v.get("rle") is None: continue
            thresh = CONF_FILTER.get(prompt, 0.7)
            if v.get("score", 1.0) < thresh: continue
            by_image[img_path][prompt] = v

    logger.info(f"两阶段评估: {len(by_image)} 张图")
    per_class_iou = defaultdict(list)
    t0 = time.time()

    for img_path, prompts_dict in tqdm(by_image.items(), desc="  两阶段评估"):
        abs_path = str(IMAGE_ROOT / img_path)
        if not os.path.exists(abs_path): continue

        # 图像预处理：与 CLIPSeg 一致 — BGR → 灰度 → 反色统一 → RGB
        img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = unify_polarity(img_gray, os.path.basename(img_path))
        h, w = img_gray.shape
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        for prompt, v in prompts_dict.items():
            gt_mask = rle_to_mask(v["rle"])
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 阶段一：YOLO-World 检测
            yolo.set_classes([prompt])
            det_results = yolo.predict(img_rgb, imgsz=IMG_SIZE, conf=0.25,
                                       device=DEVICE, verbose=False)

            # 阶段二：FastSAM 分割 + 合并
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            if det_results and len(det_results[0].boxes) > 0:
                for box in det_results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1: continue
                    crop = img_rgb[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    fs_results = fastsam.predict(source=crop, imgsz=Fastsam_IMG_SIZE,
                                                 conf=Fastsam_CONF, iou=Fastsam_IOU,
                                                 device=DEVICE, verbose=False, retina_masks=True)
                    if fs_results and fs_results[0].masks is not None:
                        for cm_tensor in fs_results[0].masks.data:
                            cm = cm_tensor.cpu().numpy().astype(np.uint8)
                            cm = cv2.resize(cm, (x2 - x1, y2 - y1))
                            pred_mask[y1:y2, x1:x2] |= cm

            per_class_iou[prompt].append(compute_iou(pred_mask, gt_mask))

    elapsed = time.time() - t0
    summary = {}
    logger.info(f"\n  两阶段 mask mIoU（{len(by_image)} 张, {elapsed:.0f}s）:")
    for cls_name in CLASSES:
        if per_class_iou[cls_name]:
            miou = np.mean(per_class_iou[cls_name])
            summary[f"iou/{cls_name}"] = miou
            logger.info(f"    {cls_name:10s}: {miou:.4f}")
    overall = np.mean([v for vals in per_class_iou.values() for v in vals])
    summary["iou/overall"] = overall
    logger.info(f"    {'OVERALL':10s}: {overall:.4f}")

    # ── 同义词泛化测试（yolo + fastsam 均在作用域内）──
    logger.info(f"\n  同义词泛化测试（推理时 prompt 映射）:")
    class_synonyms = defaultdict(list)
    for syn, orig in PROMPT_MAP.items():
        if orig in CLASSES:
            class_synonyms[orig].append(syn)

    for cls_name, synonyms in class_synonyms.items():
        if len(by_image) == 0: break
        for syn in synonyms[:2]:
            syn_ious = []
            for img_path, prompts_dict in list(by_image.items())[:50]:
                if cls_name not in prompts_dict: continue
                abs_path = str(IMAGE_ROOT / img_path)
                if not os.path.exists(abs_path): continue
                img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
                if img_bgr is None: continue
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_gray = unify_polarity(img_gray, os.path.basename(img_path))
                h, w = img_gray.shape
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                gt_mask = rle_to_mask(prompts_dict[cls_name]["rle"])
                if gt_mask.shape != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                yolo.set_classes([cls_name])
                det_results = yolo.predict(img_rgb, imgsz=IMG_SIZE, conf=0.25,
                                           device=DEVICE, verbose=False)
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                if det_results and len(det_results[0].boxes) > 0:
                    for box in det_results[0].boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 <= x1 or y2 <= y1: continue
                        crop = img_rgb[y1:y2, x1:x2]
                        fs_results = fastsam.predict(source=crop, imgsz=Fastsam_IMG_SIZE,
                                                     conf=Fastsam_CONF, iou=Fastsam_IOU,
                                                     device=DEVICE, verbose=False, retina_masks=True)
                        if fs_results and fs_results[0].masks is not None:
                            for cm_tensor in fs_results[0].masks.data:
                                cm = cm_tensor.cpu().numpy().astype(np.uint8)
                                cm = cv2.resize(cm, (x2 - x1, y2 - y1))
                                pred_mask[y1:y2, x1:x2] |= cm
                syn_ious.append(compute_iou(pred_mask, gt_mask))
            if syn_ious:
                logger.info(f"    {syn:15s} → {cls_name:10s}: {np.mean(syn_ious):.4f}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

def train():
    run_dir = PROJECT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL, task=YOLO_TASK)
    model.set_classes(CLASSES)   # 设置开放词汇类别，让 CLIP text encoder 编码
    logger.info(f"模型: {YOLO_MODEL} task={YOLO_TASK} classes={CLASSES}")

    t0 = time.time()
    model.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        device=DEVICE, workers=WORKERS, seed=SEED,
        optimizer=OPTIMIZER, lr0=LR0, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR, patience=PATIENCE, amp=True,
        mosaic=MOSAIC, mixup=MIXUP, copy_paste=COPY_PASTE,
        scale=SCALE, degrees=DEGREES, flipud=FLIPUD, fliplr=FLIPLR,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V, close_mosaic=CLOSE_MOSAIC,
        project=str(PROJECT), name=RUN_NAME, exist_ok=EXIST_OK,
        save=True, plots=True, verbose=False,
    )
    elapsed = time.time() - t0
    best_pt = run_dir / "weights" / "best.pt"
    logger.info(f"检测训练完成，耗时 {elapsed/60:.1f} min, best.pt={'OK' if best_pt.exists() else 'MISSING'}")
    return best_pt


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    import torch
    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA，请安装 CUDA 版 PyTorch"); sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("=" * 60)
    logger.info("YOLO-World + FastSAM 两阶段训练与评估")
    logger.info(f"  阶段一: {YOLO_MODEL} (检测, epochs={EPOCHS})")
    logger.info(f"  阶段二: {Fastsam_MODEL} (零样本分割)")
    logger.info(f"  类别: {CLASSES}")
    logger.info("=" * 60)

    # ── 数据集转换 ──
    train_set = load_file_list(TRAIN_LIST)
    val_set = load_file_list(VAL_LIST)
    dataset_yaml = YOLO_DIR / "dataset.yaml"

    if dataset_yaml.exists():
        n_train = len(list(TRAIN_IMAGES_DIR.glob("*.jpg"))); n_val = len(list(VAL_IMAGES_DIR.glob("*.jpg")))
        logger.info(f"数据集已存在: train={n_train}, val={n_val}")
    else:
        n_train = convert_split("train", PRED_JSON, train_set,
                                TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, do_oversample=True)
        n_val = convert_split("val", VAL_PRED_JSON, val_set,
                              VAL_IMAGES_DIR, VAL_LABELS_DIR, do_oversample=False) if VAL_PRED_JSON.exists() else 0
        create_dataset_yaml()
    logger.info(f"训练集: {n_train}, 验证集: {n_val}")

    # ── 阶段一：检测训练 ──
    best_pt = train()

    # ── 阶段二：FastSAM 两阶段 mask 评估 ──
    if best_pt and best_pt.exists() and VAL_PRED_JSON.exists():
        logger.info(f"\n{'─' * 50}\n两阶段 mask mIoU 评估")
        two_stage_eval(best_pt, VAL_PRED_JSON, VAL_LIST)

    logger.info(f"\n日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
