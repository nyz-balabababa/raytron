#!/usr/bin/env python3
"""
YOLO-World + FastSAM v5
=======================

7 类全训练集“数据处理验证版”：
  - 基于 v3 主线结构
  - 原六类 + trash can
  - 使用已合并好的 new_train / new_val 伪标签
  - 目标是先验证 trash can 并回主线后的标签链路和训练效果
"""

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["ULTRALYTICS_DOWNLOADS"] = "false"

_LOCAL_CLIP_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model" / "clip"
_DEFAULT_CLIP_CACHE = Path.home() / ".cache" / "clip"


def _ensure_local_clip():
    _DEFAULT_CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    target = _DEFAULT_CLIP_CACHE / "ViT-B-32.pt"
    if target.exists():
        return

    local_pt = _LOCAL_CLIP_DIR / "ViT-B-32.pt"
    if local_pt.exists():
        shutil.copy2(str(local_pt), str(target))
        return

    import clip as _clip

    _LOCAL_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    _clip.load("ViT-B/32", download_root=str(_LOCAL_CLIP_DIR))
    local_pt = _LOCAL_CLIP_DIR / "ViT-B-32.pt"
    if local_pt.exists():
        shutil.copy2(str(local_pt), str(target))


_ensure_local_clip()

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config_world_v5 import (
    BASE_CLASSES, EXTRA_CLASSES, CLASSES, WATCH_CLASSES,
    TRAIN_PRED_SOURCES, VAL_PRED_SOURCES, TRAIN_LIST, VAL_LIST, IMAGE_ROOT,
    YOLO_MODEL, Fastsam_MODEL, YOLO_TASK, YOLO_DIR,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR,
    IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    OPTIMIZER, LR0, MOMENTUM, WEIGHT_DECAY, WARMUP_EPOCHS, COS_LR, PATIENCE,
    MOSAIC, MIXUP, COPY_PASTE, SCALE, DEGREES, FLIPUD, FLIPLR,
    HSV_H, HSV_S, HSV_V, CLOSE_MOSAIC,
    Fastsam_IMG_SIZE, Fastsam_CONF, Fastsam_IOU,
    PROJECT, RUN_NAME, EXIST_OK,
    CONF_FILTER, RARE_OVERSAMPLE, PROMPT_MAP,
    MULTI_SCALE_CLASSES, MULTI_SCALES, MS_NMS_IOU, MS_CONF,
)

LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def rle_to_mask(rle):
    try:
        from pycocotools import mask as mask_utils

        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return mask_utils.decode(rle_cp).astype(np.uint8)
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


def unify_polarity(gray, fname):
    if "blackHot" in fname:
        return 255 - gray
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < -0.3:
            return 255 - gray
    if mean > 200 and "vis" not in fname.lower():
        return 255 - gray
    return gray


def mask_to_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    bboxes = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        bboxes.append((cx, cy, nw, nh))
    return bboxes


def load_file_list(path):
    with open(path, encoding="utf-8") as f:
        return {line.strip().replace("\\", "/") for line in f if line.strip()}


def _normalize_allowed_path(img_path, allow_set):
    img_path = img_path.replace("\\", "/")
    if img_path in allow_set:
        return img_path
    alt = img_path[5:] if img_path.startswith("test/") else f"test/{img_path}"
    return alt if alt in allow_set else None


def _write_image(src, dst):
    if dst.exists() or not src.exists():
        return
    img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return
    img_gray = unify_polarity(img_gray, os.path.basename(str(src)))
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(dst), img_rgb)


def _copy_oversample(start_idx, img_path, label_lines, images_dir, labels_dir, copies):
    src = IMAGE_ROOT / img_path
    count = 0
    for _ in range(copies):
        dst_img = images_dir / f"{start_idx + count:06d}.jpg"
        dst_label = labels_dir / f"{start_idx + count:06d}.txt"
        _write_image(src, dst_img)
        dst_label.write_text("\n".join(label_lines), encoding="utf-8")
        count += 1
    return count


def build_prediction_index(source_defs, allow_set):
    by_image = defaultdict(dict)
    positive_by_class = defaultdict(set)
    source_stats = defaultdict(lambda: defaultdict(set))

    for source in source_defs:
        source_path = Path(source["path"])
        source_name = source["name"]
        allowed_classes = set(source["classes"])

        with open(source_path, encoding="utf-8-sig") as f:
            preds = json.load(f)

        for item in preds:
            img_path = _normalize_allowed_path(item["image_path"], allow_set)
            if img_path is None:
                continue

            for prompt, value in item["prompts"].items():
                if prompt not in allowed_classes:
                    continue
                if not value.get("hit") or value.get("rle") is None:
                    continue
                if value.get("score", 1.0) < CONF_FILTER.get(prompt, 0.7):
                    continue
                by_image[img_path][prompt] = value
                positive_by_class[prompt].add(img_path)
                source_stats[source_name][prompt].add(img_path)

    return by_image, positive_by_class, source_stats


def log_source_stats(split_name, source_defs, source_stats):
    logger.info(f"{split_name} 伪标签来源统计:")
    for source in source_defs:
        name = source["name"]
        cls_parts = []
        for cls_name in source["classes"]:
            cls_parts.append(f"{cls_name}={len(source_stats[name].get(cls_name, set()))}")
        logger.info(f"  {name:18s}: " + ", ".join(cls_parts))


def convert_split(split_name, by_image, images_dir, labels_dir, do_oversample=True):
    logger.info(f"{'─' * 50}\n转换 {split_name} 集")
    for folder in [images_dir, labels_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    total_unique = len(by_image)
    prompts_kept = 0
    prompts_filtered = 0
    empty_skipped = 0
    img_idx = 0
    oversample_records = []

    for img_path, prompt_dict in tqdm(by_image.items(), desc=f"  转换{split_name}", unit="img"):
        label_lines = []
        max_mult = 1

        for prompt in sorted(prompt_dict.keys(), key=lambda x: CLASSES.index(x)):
            class_id = CLASSES.index(prompt)
            value = prompt_dict[prompt]
            mask = rle_to_mask(value["rle"])
            bboxes = mask_to_bboxes(mask)
            if not bboxes:
                prompts_filtered += 1
                continue

            prompts_kept += 1
            max_mult = max(max_mult, RARE_OVERSAMPLE.get(prompt, 1))
            for cx, cy, bw, bh in bboxes:
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            empty_skipped += 1
            continue

        src = IMAGE_ROOT / img_path
        dst = images_dir / f"{img_idx:06d}.jpg"
        _write_image(src, dst)
        (labels_dir / f"{img_idx:06d}.txt").write_text("\n".join(label_lines), encoding="utf-8")

        if do_oversample and max_mult > 1:
            oversample_records.append((img_path, label_lines, max_mult))

        img_idx += 1

    total_base = img_idx
    oversample_added = 0
    if do_oversample and oversample_records:
        copy_idx = total_base
        for img_path, label_lines, mult in oversample_records:
            added = _copy_oversample(copy_idx, img_path, label_lines, images_dir, labels_dir, mult - 1)
            copy_idx += added
            oversample_added += added
        total_base = copy_idx

    logger.info(f"  图片: {total_unique} → {total_base} 张, 跳过空: {empty_skipped}")
    logger.info(f"  实例保留: {prompts_kept}, 过滤: {prompts_filtered}")
    logger.info(f"  过采样新增: {oversample_added}")
    return total_base


def create_dataset_yaml():
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )


def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-7)


def _multi_scale_detect(yolo, img_rgb, prompt, base_imgsz, scales, conf, nms_iou, device):
    all_boxes = []
    yolo.set_classes([prompt])

    for scale in scales:
        imgsz = int(base_imgsz * scale)
        results = yolo.predict(img_rgb, imgsz=imgsz, conf=conf, device=device, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes_xyxy, confs):
                all_boxes.append((*box.tolist(), float(c)))

    if not all_boxes:
        return []

    all_boxes.sort(key=lambda x: x[4], reverse=True)
    kept = []
    for box in all_boxes:
        x1, y1, x2, y2, _ = box
        if x2 <= x1 or y2 <= y1:
            continue
        overlap = False
        for kx1, ky1, kx2, ky2, _ in kept:
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix1 < ix2 and iy1 < iy2:
                iarea = (ix2 - ix1) * (iy2 - iy1)
                uarea = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - iarea
                if uarea > 0 and iarea / uarea > nms_iou:
                    overlap = True
                    break
        if not overlap:
            kept.append(box)

    return [[b[0], b[1], b[2], b[3]] for b in kept]


def _fastsam_segment_boxes(fastsam, img_rgb, boxes, h, w):
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            continue

        crop = img_rgb[y1:y2, x1:x2]
        max_side = max(crop_h, crop_w)
        dynamic_imgsz = max(256, min(Fastsam_IMG_SIZE, max_side))
        dynamic_imgsz = ((dynamic_imgsz + 31) // 32) * 32

        fs_results = fastsam.predict(
            source=crop,
            imgsz=dynamic_imgsz,
            conf=Fastsam_CONF,
            iou=Fastsam_IOU,
            device=DEVICE,
            verbose=False,
            retina_masks=True,
        )
        if fs_results and fs_results[0].masks is not None:
            for cm_tensor in fs_results[0].masks.data:
                cm = cm_tensor.cpu().numpy().astype(np.uint8)
                cm = cv2.resize(cm, (crop_w, crop_h))
                pred_mask[y1:y2, x1:x2] |= cm
    return pred_mask


def two_stage_eval(yolo_pt, by_image, max_samples=None):
    from ultralytics import YOLO, FastSAM

    yolo = YOLO(str(yolo_pt), task="detect")
    yolo.to(f"cuda:{DEVICE}")
    fastsam = FastSAM(Fastsam_MODEL)
    logger.info("YOLO-World + FastSAM 已加载")

    eval_items = list(by_image.items())
    if max_samples:
        eval_items = eval_items[:max_samples]

    logger.info(f"两阶段评估: {len(eval_items)} 张图")
    logger.info(f"  多尺度推理: {MULTI_SCALE_CLASSES} @ {MULTI_SCALES}")
    logger.info(f"  单尺度推理: {[c for c in CLASSES if c not in MULTI_SCALE_CLASSES]}")

    per_class_iou = defaultdict(list)
    t0 = time.time()

    for img_path, prompts_dict in tqdm(eval_items, desc="  两阶段评估"):
        abs_path = str(IMAGE_ROOT / img_path)
        if not os.path.exists(abs_path):
            continue

        img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = unify_polarity(img_gray, os.path.basename(img_path))
        h, w = img_gray.shape
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        for prompt, value in prompts_dict.items():
            gt_mask = rle_to_mask(value["rle"])
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if prompt in MULTI_SCALE_CLASSES:
                boxes = _multi_scale_detect(
                    yolo,
                    img_rgb,
                    prompt,
                    base_imgsz=IMG_SIZE,
                    scales=MULTI_SCALES,
                    conf=MS_CONF,
                    nms_iou=MS_NMS_IOU,
                    device=DEVICE,
                )
            else:
                yolo.set_classes([prompt])
                det_results = yolo.predict(img_rgb, imgsz=IMG_SIZE, conf=0.25, device=DEVICE, verbose=False)
                boxes = []
                if det_results and len(det_results[0].boxes) > 0:
                    boxes = det_results[0].boxes.xyxy.cpu().numpy().tolist()

            pred_mask = _fastsam_segment_boxes(fastsam, img_rgb, boxes, h, w)
            per_class_iou[prompt].append(compute_iou(pred_mask, gt_mask))

    elapsed = time.time() - t0
    summary = {}
    logger.info(f"\n  两阶段 mask mIoU（{len(eval_items)} 张, {elapsed:.0f}s）:")
    for cls_name in CLASSES:
        if per_class_iou[cls_name]:
            miou = float(np.mean(per_class_iou[cls_name]))
            summary[f"iou/{cls_name}"] = miou
            tag = " [MS]" if cls_name in MULTI_SCALE_CLASSES else ""
            logger.info(f"    {cls_name:16s}{tag}: {miou:.4f}  (n={len(per_class_iou[cls_name])})")

    overall_vals = [v for vals in per_class_iou.values() for v in vals]
    overall = float(np.mean(overall_vals)) if overall_vals else 0.0
    summary["iou/overall"] = overall
    logger.info(f"    {'OVERALL':16s}: {overall:.4f}")

    logger.info("\n  同义词泛化测试（推理时 prompt 映射）:")
    class_synonyms = defaultdict(list)
    for syn, orig in PROMPT_MAP.items():
        if orig in CLASSES:
            class_synonyms[orig].append(syn)

    sample_items = eval_items[:50]
    for cls_name, synonyms in class_synonyms.items():
        test_syns = synonyms[:8] if cls_name == "animal" else synonyms[:3]
        for syn in test_syns:
            syn_ious = []
            for img_path, prompts_dict in sample_items:
                if cls_name not in prompts_dict:
                    continue
                abs_path = str(IMAGE_ROOT / img_path)
                if not os.path.exists(abs_path):
                    continue

                img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_gray = unify_polarity(img_gray, os.path.basename(img_path))
                h, w = img_gray.shape
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                gt_mask = rle_to_mask(prompts_dict[cls_name]["rle"])
                if gt_mask.shape != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                yolo.set_classes([syn])
                det_results = yolo.predict(img_rgb, imgsz=IMG_SIZE, conf=0.25, device=DEVICE, verbose=False)
                syn_boxes = []
                if det_results and len(det_results[0].boxes) > 0:
                    syn_boxes = det_results[0].boxes.xyxy.cpu().numpy().tolist()

                pred_mask = _fastsam_segment_boxes(fastsam, img_rgb, syn_boxes, h, w)
                syn_ious.append(compute_iou(pred_mask, gt_mask))

            if syn_ious:
                logger.info(f"    {syn:20s} → {cls_name:16s}: {np.mean(syn_ious):.4f}")

    return summary


def log_gpu_info():
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU 训练（极慢）")
        logger.info(f"  PyTorch: {torch.__version__}")
        return

    props = torch.cuda.get_device_properties(0)
    total_vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  显存: {total_vram:.1f} GB | 已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB")
    logger.info(f"  CUDA: {torch.version.cuda if torch.version.cuda else 'N/A'} | PyTorch: {torch.__version__}")


def plot_training_curves(results_csv, run_dir):
    if not results_csv.exists():
        logger.warning(f"results.csv 不存在，跳过绘图: {results_csv}")
        return

    import csv

    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    col_map = {c.strip(): c for c in rows[0].keys()}

    def _get(col_suffix):
        for c_clean, c_orig in col_map.items():
            if col_suffix in c_clean:
                return [float(r[c_orig]) for r in rows]
        return []

    epochs = list(range(1, len(rows) + 1))
    train_box = _get("train/box_loss")
    train_cls = _get("train/cls_loss")
    train_dfl = _get("train/dfl_loss")
    val_box = _get("val/box_loss")
    val_cls = _get("val/cls_loss")
    val_dfl = _get("val/dfl_loss")
    map50 = _get("metrics/mAP50(B)")
    map50_95 = _get("metrics/mAP50-95(B)")
    precision = _get("metrics/precision(B)")
    recall = _get("metrics/recall(B)")
    lr_vals = _get("lr/pg0")

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 3, 1)
    for vals, label, color in [(train_box, "box", "#1f77b4"), (train_cls, "cls", "#ff7f0e"), (train_dfl, "dfl", "#2ca02c")]:
        if vals:
            ax1.plot(epochs, vals, "-", label=label, color=color, linewidth=1)
    ax1.set_title("1. Train Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(2, 3, 2)
    for vals, label, color in [(val_box, "box", "#1f77b4"), (val_cls, "cls", "#ff7f0e"), (val_dfl, "dfl", "#2ca02c")]:
        if vals:
            ax2.plot(epochs, vals, "-", label=label, color=color, linewidth=1)
    ax2.set_title("2. Val Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    if map50:
        ax3.plot(epochs, map50, "-", label="mAP50", color="#d62728", linewidth=1.5)
    if map50_95:
        ax3.plot(epochs, map50_95, "-", label="mAP50-95", color="#9467bd", linewidth=1.5)
    ax3.set_title("3. Detection mAP (BBox)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(2, 3, 4)
    if precision:
        ax4.plot(epochs, precision, "-", label="Precision", color="#8c564b", linewidth=1.5)
    if recall:
        ax4.plot(epochs, recall, "-", label="Recall", color="#e377c2", linewidth=1.5)
    ax4.set_title("4. Precision & Recall")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)
    ax4.set_ylim(0, 1.05)

    ax5 = fig.add_subplot(2, 3, 5)
    if lr_vals:
        ax5.plot(epochs, lr_vals, "-", color="red", linewidth=1)
    ax5.set_title("5. Learning Rate")
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = ["── Training Summary ──", ""]
    if map50:
        best_idx = int(np.argmax(map50))
        lines.append(f"Best mAP50:        {map50[best_idx]:.4f}  (epoch {epochs[best_idx]})")
    if map50_95:
        best_idx_95 = int(np.argmax(map50_95))
        lines.append(f"Best mAP50-95:     {map50_95[best_idx_95]:.4f}  (epoch {epochs[best_idx_95]})")
    if precision:
        lines.append(f"Final Precision:   {precision[-1]:.4f}")
    if recall:
        lines.append(f"Final Recall:      {recall[-1]:.4f}")
    lines.append(f"Total Epochs:      {len(epochs)}")
    lines.append(f"Early Stop:        {'Yes' if len(epochs) < EPOCHS else 'No (full)'}")
    for i, line in enumerate(lines):
        ax6.text(0.05, 0.95 - i * 0.06, line, transform=ax6.transAxes, fontsize=9, family="monospace", va="top")

    plt.tight_layout()
    out_path = run_dir / "training_curves_v5.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线图已保存: {out_path}")


def train():
    run_dir = PROJECT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL, task=YOLO_TASK)
    model.set_classes(CLASSES)
    logger.info(f"模型: {YOLO_MODEL} task={YOLO_TASK} classes={CLASSES}")

    t0 = time.time()
    model.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR,
        patience=PATIENCE,
        amp=True,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        scale=SCALE,
        degrees=DEGREES,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        close_mosaic=CLOSE_MOSAIC,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=EXIST_OK,
        save=True,
        plots=True,
        verbose=False,
    )
    elapsed = time.time() - t0

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    logger.info(f"\n{'─' * 50}")
    logger.info(f"阶段一训练完成 | 耗时: {elapsed / 60:.1f} min")
    logger.info(f"  best.pt:  {'✓' if best_pt.exists() else '✗ 缺失'}")
    logger.info(f"  last.pt:  {'✓' if last_pt.exists() else '✗ 缺失'}")

    if results_csv.exists():
        import csv

        with open(results_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            col_map = {c.strip(): c for c in last.keys()}

            def _v(suffix):
                for c_clean, c_orig in col_map.items():
                    if suffix in c_clean:
                        return last[c_orig]
                return "N/A"

            logger.info(f"  总 epochs:     {len(rows)}/{EPOCHS}")
            logger.info(f"  mAP50:         {_v('metrics/mAP50(B)')}")
            logger.info(f"  mAP50-95:      {_v('metrics/mAP50-95(B)')}")
            logger.info(f"  Precision:     {_v('metrics/precision(B)')}")
            logger.info(f"  Recall:        {_v('metrics/recall(B)')}")
            if len(rows) < EPOCHS:
                logger.info(f"  ⚠ 早停触发于 epoch {len(rows)} (patience={PATIENCE})")

        plot_training_curves(results_csv, run_dir)

    logger.info(f"{'─' * 50}")
    return best_pt


def main():
    import torch

    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA，请安装 CUDA 版 PyTorch")
        sys.exit(1)

    all_sources = TRAIN_PRED_SOURCES + VAL_PRED_SOURCES
    missing = [str(source["path"]) for source in all_sources if not Path(source["path"]).exists()]
    if missing:
        logger.error("以下伪标签文件不存在:")
        for path in missing:
            logger.error(f"  - {path}")
        sys.exit(1)

    log_gpu_info()
    logger.info("")
    logger.info("=" * 60)
    logger.info("YOLO-World + FastSAM v5 | 7 类全训练集数据处理验证版")
    logger.info(f"  原主线类别: {BASE_CLASSES}")
    logger.info(f"  新增类别:   {EXTRA_CLASSES}")
    logger.info(f"  重点观察:   {WATCH_CLASSES}")
    logger.info(f"  多尺度:     {MULTI_SCALE_CLASSES} @ {MULTI_SCALES}")
    logger.info(f"  阈值:       {CONF_FILTER}")
    logger.info(f"  过采样:     {RARE_OVERSAMPLE}")
    logger.info("  训练伪标签源:")
    for source in TRAIN_PRED_SOURCES:
        logger.info(f"    - {source['name']}: {source['path']}")
    logger.info("  验证伪标签源:")
    for source in VAL_PRED_SOURCES:
        logger.info(f"    - {source['name']}: {source['path']}")
    logger.info("=" * 60)

    train_set = load_file_list(TRAIN_LIST)
    val_set = load_file_list(VAL_LIST)
    train_index, train_pos_by_class, train_source_stats = build_prediction_index(TRAIN_PRED_SOURCES, train_set)
    val_index, val_pos_by_class, val_source_stats = build_prediction_index(VAL_PRED_SOURCES, val_set)

    logger.info("固定训练/验证集:")
    logger.info(f"  train_list: {TRAIN_LIST.name} -> {len(train_set)} 张")
    logger.info(f"  val_list:   {VAL_LIST.name} -> {len(val_set)} 张")
    for cls_name in CLASSES:
        logger.info(
            f"  {cls_name:16s} train_pos={len(train_pos_by_class.get(cls_name, set())):4d} | "
            f"val_pos={len(val_pos_by_class.get(cls_name, set())):4d}"
        )
    log_source_stats("train", TRAIN_PRED_SOURCES, train_source_stats)
    log_source_stats("val", VAL_PRED_SOURCES, val_source_stats)

    n_train = convert_split("train", train_index, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, do_oversample=True)
    n_val = convert_split("val", val_index, VAL_IMAGES_DIR, VAL_LABELS_DIR, do_oversample=False)
    create_dataset_yaml()
    logger.info(f"训练集: {n_train}, 验证集: {n_val}")

    best_pt = train()

    if best_pt and best_pt.exists() and val_index:
        logger.info(f"\n{'─' * 50}\n两阶段 mask mIoU 评估 (v5)")
        two_stage_eval(best_pt, val_index)

    logger.info(f"\n日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
