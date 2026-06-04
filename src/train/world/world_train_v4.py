#!/usr/bin/env python3
"""
YOLO-World + FastSAM v4
=======================

基于 v3 的 2 类尾类实验版本：
  - 只训练 computer / trash can
  - 训练/验证分别读取 d5&7 的固定列表与伪标签
  - 训练集支持按分层规则优先抽取空标签负样本
  - fire hydrant 暂不并入多类 head，后续单独做 binary few-shot
"""

import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

os.environ["ULTRALYTICS_DOWNLOADS"] = "false"

# ══════════════════════════════════════════════════════════════════════
# 离线 CLIP 缓存
# ══════════════════════════════════════════════════════════════════════

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

from config_world_v4 import (
    YOLO_MODEL, Fastsam_MODEL, CLASSES, PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST,
    IMAGE_ROOT, YOLO_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR,
    YOLO_TASK, IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    OPTIMIZER, LR0, MOMENTUM, WEIGHT_DECAY, WARMUP_EPOCHS, COS_LR, PATIENCE,
    FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA, CLASS_LOSS_WEIGHTS,
    MOSAIC, MIXUP, COPY_PASTE, SCALE, DEGREES, FLIPUD, FLIPLR,
    HSV_H, HSV_S, HSV_V, CLOSE_MOSAIC,
    Fastsam_IMG_SIZE, Fastsam_CONF, Fastsam_IOU,
    PROJECT, RUN_NAME, EXIST_OK,
    CONF_FILTER, RARE_OVERSAMPLE, NEGATIVE_TO_POSITIVE_RATIO, NEGATIVE_SAMPLE_SEED,
    PROMPT_MAP, MULTI_SCALE_CLASSES, MULTI_SCALES, MS_NMS_IOU, MS_CONF,
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
# 长尾分类损失：Focal Loss + 手动类别权重
# ══════════════════════════════════════════════════════════════════════

def _build_class_weight_tensor(class_names):
    import torch

    return torch.tensor(
        [float(CLASS_LOSS_WEIGHTS.get(cls_name, 1.0)) for cls_name in class_names],
        dtype=torch.float32,
    )


def _install_weighted_focal_loss(detect_model):
    """给 Ultralytics DetectionModel 安装自定义分类损失。

    只替换分类分支：
      - BCEWithLogits -> Focal Loss
      - 对 computer / trash can 施加更高类别权重
    bbox / dfl / assigner 全部保持原版实现。
    """
    import torch
    from ultralytics.utils.loss import E2ELoss, FocalLoss, make_anchors, v8DetectionLoss

    class_weights = _build_class_weight_tensor(CLASSES)
    detect_model.class_weights = class_weights

    class WeightedFocalDetectionLoss(v8DetectionLoss):
        def __init__(self, model, tal_topk=10, tal_topk2=None):
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            self.focal = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
            self.class_weights = class_weights.to(self.device).view(1, 1, -1)

        def get_assigned_targets_and_loss(self, preds, batch):
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            pred_distri, pred_scores = (
                preds["boxes"].permute(0, 2, 1).contiguous(),
                preds["scores"].permute(0, 2, 1).contiguous(),
            )
            anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            target_scores_sum = max(target_scores.sum(), 1)

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_scores,
                target_scores.to(dtype),
                reduction="none",
            )
            pred_prob = pred_scores.sigmoid()
            p_t = target_scores.to(dtype) * pred_prob + (1 - target_scores.to(dtype)) * (1 - pred_prob)
            modulating_factor = (1.0 - p_t) ** self.focal.gamma
            alpha = self.focal.alpha.to(device=pred_scores.device, dtype=pred_scores.dtype)
            alpha_factor = target_scores.to(dtype) * alpha + (1 - target_scores.to(dtype)) * (1 - alpha)
            cls_loss = cls_loss * modulating_factor * alpha_factor
            cls_loss = cls_loss * self.class_weights
            loss[1] = cls_loss.sum() / target_scores_sum

            if fg_mask.sum():
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                    imgsz,
                    stride_tensor,
                )

            loss[0] *= self.hyp.box
            loss[1] *= self.hyp.cls
            loss[2] *= self.hyp.dfl
            return (
                (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss,
                loss.detach(),
            )

    def _custom_init_criterion(self):
        def _factory(model_obj, tal_topk=10, tal_topk2=None):
            return WeightedFocalDetectionLoss(model_obj, tal_topk=tal_topk, tal_topk2=tal_topk2)

        return E2ELoss(self, _factory) if getattr(self, "end2end", False) else _factory(self)

    detect_model.init_criterion = MethodType(_custom_init_criterion, detect_model)
    return class_weights


# ══════════════════════════════════════════════════════════════════════
# RLE / 图像预处理
# ══════════════════════════════════════════════════════════════════════

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


def load_prediction_index(pred_json):
    with open(pred_json, encoding="utf-8-sig") as f:
        preds = json.load(f)

    all_images = []
    positives = defaultdict(dict)
    positive_by_class = defaultdict(set)

    for item in preds:
        img_path = item["image_path"].replace("\\", "/")
        all_images.append(img_path)
        for prompt, value in item["prompts"].items():
            if prompt not in CLASSES:
                continue
            if not value.get("hit") or value.get("rle") is None:
                continue
            if value.get("score", 1.0) < CONF_FILTER.get(prompt, 0.7):
                continue
            positives[img_path][prompt] = value
            positive_by_class[prompt].add(img_path)

    return preds, set(all_images), positives, positive_by_class


def _scene_meta(img_path):
    norm = img_path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    source = parts[1] if len(parts) >= 2 else "unknown"

    stem = Path(norm).stem
    family_stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    family_key = f"{source}::{family_stem}" if family_stem else None

    bucket_key = None
    match = re.search(r"(\d+)", family_stem)
    if match:
        number = int(match.group(1))
        bucket_key = f"{source}::{family_stem[:match.start()]}{number // 500:06d}"

    return {
        "source": source,
        "family": family_key,
        "bucket": bucket_key,
    }


def _sample_stratified_negatives(positive_images, negative_images, target_negatives, seed):
    if target_negatives <= 0 or not negative_images:
        return [], {"family": 0, "bucket": 0, "source_fallback": 0}

    rng = random.Random(seed)
    positive_meta = {img: _scene_meta(img) for img in positive_images}
    negative_meta = {img: _scene_meta(img) for img in negative_images}

    pos_source_counts = defaultdict(int)
    pos_family_keys = set()
    pos_bucket_keys = set()
    for meta in positive_meta.values():
        pos_source_counts[meta["source"]] += 1
        if meta["family"]:
            pos_family_keys.add(meta["family"])
        if meta["bucket"]:
            pos_bucket_keys.add(meta["bucket"])

    neg_by_source = defaultdict(list)
    for img, meta in negative_meta.items():
        neg_by_source[meta["source"]].append(img)

    selected = []
    selected_set = set()
    tier_counts = {"family": 0, "bucket": 0, "source_fallback": 0}

    def take_from_pool(pool, quota, tier_name):
        if quota <= 0:
            return 0
        candidates = [img for img in pool if img not in selected_set]
        if not candidates:
            return 0
        take_n = min(quota, len(candidates))
        if take_n == len(candidates):
            picked = candidates
            rng.shuffle(picked)
        else:
            picked = rng.sample(candidates, take_n)
        selected.extend(picked)
        selected_set.update(picked)
        tier_counts[tier_name] += len(picked)
        return len(picked)

    source_targets = {}
    allocated = 0
    sources = sorted(pos_source_counts.keys())
    total_pos = max(1, len(positive_images))
    for idx, source in enumerate(sources):
        if idx == len(sources) - 1:
            quota = target_negatives - allocated
        else:
            quota = int(round(target_negatives * (pos_source_counts[source] / total_pos)))
            quota = min(quota, target_negatives - allocated)
        quota = min(quota, len(neg_by_source.get(source, [])))
        source_targets[source] = quota
        allocated += quota

    remaining = target_negatives - allocated
    if remaining > 0:
        spare_sources = sorted(
            sources,
            key=lambda s: len(neg_by_source.get(s, [])) - source_targets.get(s, 0),
            reverse=True,
        )
        for source in spare_sources:
            spare = len(neg_by_source.get(source, [])) - source_targets.get(source, 0)
            if spare <= 0:
                continue
            add = min(spare, remaining)
            source_targets[source] += add
            remaining -= add
            if remaining == 0:
                break

    for source in sources:
        quota = source_targets.get(source, 0)
        if quota <= 0:
            continue
        source_imgs = neg_by_source.get(source, [])
        family_pool = [img for img in source_imgs if negative_meta[img]["family"] in pos_family_keys]
        bucket_pool = [
            img for img in source_imgs
            if img not in selected_set and negative_meta[img]["bucket"] in pos_bucket_keys
        ]
        source_pool = [img for img in source_imgs if img not in selected_set]

        quota -= take_from_pool(family_pool, quota, "family")
        quota -= take_from_pool(bucket_pool, quota, "bucket")
        quota -= take_from_pool(source_pool, quota, "source_fallback")

    if len(selected) < target_negatives:
        global_family_pool = [
            img for img in negative_images
            if img not in selected_set and negative_meta[img]["family"] in pos_family_keys
        ]
        global_bucket_pool = [
            img for img in negative_images
            if img not in selected_set and negative_meta[img]["bucket"] in pos_bucket_keys
        ]
        global_pool = [img for img in negative_images if img not in selected_set]
        remaining = target_negatives - len(selected)
        remaining -= take_from_pool(global_family_pool, remaining, "family")
        remaining -= take_from_pool(global_bucket_pool, remaining, "bucket")
        remaining -= take_from_pool(global_pool, remaining, "source_fallback")

    return sorted(selected), tier_counts


# ══════════════════════════════════════════════════════════════════════
# 数据集转换
# ══════════════════════════════════════════════════════════════════════

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


def convert_split(split_name, pred_json, allow_set, images_dir, labels_dir, do_oversample=True):
    logger.info(f"{'─' * 50}\n转换 {split_name} 集: {pred_json.name}")
    for folder in [images_dir, labels_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    with open(pred_json, encoding="utf-8-sig") as f:
        preds = json.load(f)

    by_image = defaultdict(dict)
    for item in preds:
        img_path = item["image_path"].replace("\\", "/")
        if img_path not in allow_set:
            continue
        for prompt, value in item["prompts"].items():
            if prompt not in CLASSES:
                continue
            if not value.get("hit") or value.get("rle") is None:
                continue
            if value.get("score", 1.0) < CONF_FILTER.get(prompt, 0.7):
                continue
            by_image[img_path][prompt] = value

    positive_images = set(by_image.keys())
    negative_images = sorted(allow_set - positive_images)
    selected_negative_images = []
    negative_tier_counts = {"family": 0, "bucket": 0, "source_fallback": 0}

    if split_name == "train":
        target_negatives = int(round(len(positive_images) * NEGATIVE_TO_POSITIVE_RATIO))
        target_negatives = min(len(negative_images), target_negatives)
        if target_negatives > 0:
            selected_negative_images, negative_tier_counts = _sample_stratified_negatives(
                positive_images=positive_images,
                negative_images=negative_images,
                target_negatives=target_negatives,
                seed=NEGATIVE_SAMPLE_SEED,
            )
        selected_images = sorted(positive_images | set(selected_negative_images))
    else:
        selected_images = sorted(allow_set)

    prompts_kept = 0
    prompts_filtered = 0
    empty_skipped = 0
    img_idx = 0
    oversample_records = []

    for img_path in tqdm(selected_images, desc=f"  转换{split_name}", unit="img"):
        prompt_dict = by_image.get(img_path, {})
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

        src = IMAGE_ROOT / img_path
        dst = images_dir / f"{img_idx:06d}.jpg"
        _write_image(src, dst)
        (labels_dir / f"{img_idx:06d}.txt").write_text("\n".join(label_lines), encoding="utf-8")

        if not label_lines:
            empty_skipped += 1
        elif do_oversample and max_mult > 1:
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

    logger.info(
        f"  正样本图: {len(positive_images)}, 负样本图候选: {len(negative_images)}, "
        f"选中负样本: {len(selected_negative_images) if split_name == 'train' else len(negative_images)}"
    )
    if split_name == "train":
        logger.info(
            "  负样本分层: "
            f"family={negative_tier_counts['family']}, "
            f"bucket={negative_tier_counts['bucket']}, "
            f"source_fallback={negative_tier_counts['source_fallback']}"
        )
    logger.info(f"  图片: {len(selected_images)} → {total_base} 张, 空标签图: {empty_skipped}")
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


# ══════════════════════════════════════════════════════════════════════
# 多尺度推理 / 两阶段评估
# ══════════════════════════════════════════════════════════════════════

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
            for box, box_conf in zip(boxes_xyxy, confs):
                all_boxes.append((*box.tolist(), float(box_conf)))

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


def two_stage_eval(yolo_pt, pred_json, allow_set, max_samples=None):
    from ultralytics import FastSAM, YOLO

    yolo = YOLO(str(yolo_pt), task="detect")
    yolo.to(f"cuda:{DEVICE}")
    fastsam = FastSAM(Fastsam_MODEL)
    logger.info("YOLO-World + FastSAM 已加载")

    with open(pred_json, encoding="utf-8") as f:
        preds = json.load(f)

    by_image = defaultdict(dict)
    for item in preds:
        img_path = item["image_path"].replace("\\", "/")
        if img_path not in allow_set:
            continue
        if max_samples and len(by_image) >= max_samples:
            break
        for prompt, value in item["prompts"].items():
            if prompt not in CLASSES:
                continue
            if not value.get("hit") or value.get("rle") is None:
                continue
            if value.get("score", 1.0) < CONF_FILTER.get(prompt, 0.7):
                continue
            by_image[img_path][prompt] = value

    per_class_iou = defaultdict(list)
    t0 = time.time()

    for img_path, prompts_dict in tqdm(by_image.items(), desc="  两阶段评估"):
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
                    yolo, img_rgb, prompt, IMG_SIZE, MULTI_SCALES, MS_CONF, MS_NMS_IOU, DEVICE
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
    logger.info(f"\n  两阶段 mask mIoU（{len(by_image)} 张, {elapsed:.0f}s）:")
    summary = {}
    for cls_name in CLASSES:
        if per_class_iou[cls_name]:
            miou = float(np.mean(per_class_iou[cls_name]))
            summary[f"iou/{cls_name}"] = miou
            tag = " [MS]" if cls_name in MULTI_SCALE_CLASSES else ""
            logger.info(f"    {cls_name:18s}{tag}: {miou:.4f}  (n={len(per_class_iou[cls_name])})")

    flat = [x for values in per_class_iou.values() for x in values]
    overall = float(np.mean(flat)) if flat else 0.0
    summary["iou/overall"] = overall
    logger.info(f"    {'OVERALL':18s} : {overall:.4f}")

    logger.info("\n  同义词泛化测试:")
    class_synonyms = defaultdict(list)
    for synonym, original in PROMPT_MAP.items():
        if original in CLASSES:
            class_synonyms[original].append(synonym)

    for cls_name, synonyms in class_synonyms.items():
        test_syns = synonyms[:3]
        sample_imgs = list(by_image.items())[:50]
        for syn in test_syns:
            syn_ious = []
            for img_path, prompts_dict in sample_imgs:
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


# ══════════════════════════════════════════════════════════════════════
# 训练 / 可视化
# ══════════════════════════════════════════════════════════════════════

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
    out_path = run_dir / "training_curves_v4.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线图已保存: {out_path}")


def train():
    run_dir = PROJECT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL, task=YOLO_TASK)
    model.set_classes(CLASSES)
    class_weight_tensor = _install_weighted_focal_loss(model.model)
    logger.info(f"模型: {YOLO_MODEL} task={YOLO_TASK} classes={CLASSES}")
    logger.info(
        "分类损失: Focal Loss "
        f"(gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA}) | "
        f"class_weights={class_weight_tensor.tolist()}"
    )

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
        cls_pw=0.0,
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

    log_gpu_info()
    logger.info("")
    logger.info("=" * 60)
    logger.info("YOLO-World + FastSAM v4 | d5&7 2 类尾类实验")
    logger.info(f"  类别: {CLASSES}")
    logger.info(f"  多尺度: {MULTI_SCALE_CLASSES} @ {MULTI_SCALES}")
    logger.info(f"  阈值: {CONF_FILTER}")
    logger.info(f"  过采样: {RARE_OVERSAMPLE}")
    logger.info(
        f"  分类损失: Focal Loss(gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA}) "
        f"+ class_weights={CLASS_LOSS_WEIGHTS}"
    )
    logger.info(f"  负样本比例: {NEGATIVE_TO_POSITIVE_RATIO}× (seed={NEGATIVE_SAMPLE_SEED})")
    logger.info(f"  train 伪标签: {PRED_JSON}")
    logger.info(f"  val   伪标签: {VAL_PRED_JSON}")
    logger.info("=" * 60)

    train_set = load_file_list(TRAIN_LIST)
    val_set = load_file_list(VAL_LIST)
    _, _, _, train_pos_by_class = load_prediction_index(PRED_JSON)
    _, _, _, val_pos_by_class = load_prediction_index(VAL_PRED_JSON)

    logger.info("固定训练/验证集:")
    logger.info(f"  train_list: {TRAIN_LIST.name} -> {len(train_set)} 张")
    logger.info(f"  val_list:   {VAL_LIST.name} -> {len(val_set)} 张")
    for cls_name in CLASSES:
        train_pos = len(train_pos_by_class.get(cls_name, {}))
        val_pos = len(val_pos_by_class.get(cls_name, {}))
        logger.info(f"  {cls_name:18s} train_pos={train_pos:4d} | val_pos={val_pos:4d}")

    n_train = convert_split("train", PRED_JSON, train_set, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, do_oversample=True)
    n_val = convert_split("val", VAL_PRED_JSON, val_set, VAL_IMAGES_DIR, VAL_LABELS_DIR, do_oversample=False)
    create_dataset_yaml()
    logger.info(f"训练集: {n_train}, 验证集: {n_val}")

    best_pt = train()

    if best_pt and best_pt.exists() and VAL_PRED_JSON.exists():
        logger.info(f"\n{'─' * 50}\n两阶段 mask mIoU 评估 (v4)")
        two_stage_eval(best_pt, VAL_PRED_JSON, val_set)

    logger.info(f"\n日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
