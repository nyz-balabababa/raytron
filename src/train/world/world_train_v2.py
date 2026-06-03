#!/usr/bin/env python3
"""
YOLO-World + FastSAM 两阶段训练与评估 v2
==========================================
相比 v1 的改动：
  1. 多尺度推理：person / tree 用 [0.8, 1.0, 1.25, 1.5]× 融合 → 小目标召回
  2. cls↑ box↑ label_smoothing：弥补 YOLO API 不支持 Focal Loss 的限制
  3. computer 阈值 0.55→0.4, 过采样 2×→5× → 长尾覆盖
  4. animal 同义词推理测试大幅扩充
  5. 独立数据集目录（yolo_world_dataset_v2），不与 v1 冲突

阶段一：YOLO-World v2 检测（bbox）—— RLE 伪标签 → YOLO bbox 格式
阶段二：FastSAM 零样本分割 —— 量化两阶段 mask mIoU
"""
import os
os.environ["ULTRALYTICS_DOWNLOADS"] = "false"

import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 离线 CLIP 模型 —— 禁止在线下载
# ══════════════════════════════════════════════════════════════════════
# YOLO-World 的 set_classes() 内部会调用 clip.load("ViT-B/32")，默认从网上
# 下载 340MB 的 CLIP 权重。这里确保始终使用本地缓存，零网络访问。

_LOCAL_CLIP_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model" / "clip"
_DEFAULT_CLIP_CACHE = Path.home() / ".cache" / "clip"

def _ensure_local_clip():
    """确保 CLIP ViT-B/32 模型在本地缓存中存在。首次运行从 model/clip/ 复制。"""
    _DEFAULT_CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    target = _DEFAULT_CLIP_CACHE / "ViT-B-32.pt"
    if target.exists():
        return  # 已缓存

    # 检查项目内的预下载副本
    local_pt = _LOCAL_CLIP_DIR / "ViT-B-32.pt"
    if local_pt.exists():
        shutil.copy2(str(local_pt), str(target))
        return

    # 本地也没有 → 一次性下载到 model/clip/，后续永久复用
    import clip as _clip
    _LOCAL_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    _clip.load("ViT-B/32", download_root=str(_LOCAL_CLIP_DIR))
    # 同时复制到默认缓存，让 ultralytics 内部能找到
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

from config_world_v2 import (
    YOLO_MODEL, Fastsam_MODEL, CLASSES,
    PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST, IMAGE_ROOT, YOLO_DIR,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR,
    YOLO_TASK, IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    OPTIMIZER, LR0, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, COS_LR, PATIENCE,
    MOSAIC, MIXUP, COPY_PASTE, SCALE, DEGREES, FLIPUD, FLIPLR,
    HSV_H, HSV_S, HSV_V, CLOSE_MOSAIC,
    CLS_LOSS_GAIN, BOX_LOSS_GAIN, LABEL_SMOOTHING,
    Fastsam_IMG_SIZE, Fastsam_CONF, Fastsam_IOU,
    PROJECT, RUN_NAME, EXIST_OK,
    CONF_FILTER, RARE_OVERSAMPLE, COMPUTER_EXTRA, PROMPT_MAP,
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


def _write_image(src, dst):
    """统一写图：灰度读取 → 反色 → RGB 存入。"""
    if dst.exists() or not src.exists():
        return
    img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
        img_gray = unify_polarity(img_gray, os.path.basename(str(src)))
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(dst), img_rgb)


def _copy_oversample(idx, img_path, label_lines, images_dir, labels_dir, copies):
    """将同一张图片+标签额外复制 copies 次。返回实际复制数。"""
    src = IMAGE_ROOT / img_path
    count = 0
    for _ in range(copies):
        dst_img = images_dir / f"{idx + count:06d}.jpg"
        dst_label = labels_dir / f"{idx + count:06d}.txt"
        _write_image(src, dst_img)
        dst_label.write_text("\n".join(label_lines), encoding="utf-8")
        count += 1
    return count


def convert_split(split, pred_json, allow_set, images_dir, labels_dir, do_oversample=True):
    """
    RLE → bbox 标签（YOLO 检测格式: class_id cx cy w h）。

    过采样分两个独立阶段：
      阶段 A — animal: 对含 animal 标签的图复制 (RARE_OVERSAMPLE[animal] - 1) 次
      阶段 B — computer: 对含 computer 标签的图额外复制 COMPUTER_EXTRA 次
      两阶段互不干扰，不会因为 max() 导致 animal 被意外多复制。
    """
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
    animal_records = []     # (img_idx, img_path, label_lines)
    computer_indices = []   # [img_idx, ...]  只记录索引，不存 label_lines 副本
    computer_labels = {}    # img_idx → label_lines（用于后续复制）
    img_idx = 0

    pbar = tqdm(by_image.items(), desc=f"  转换{split}", unit="img")
    for img_path, prompt_dict in pbar:
        label_lines = []
        has_animal = False
        has_computer = False

        for prompt in sorted(prompt_dict.keys(), key=lambda x: CLASSES.index(x)):
            class_id = CLASSES.index(prompt)
            for v in prompt_dict[prompt]:
                mask = rle_to_mask(v["rle"])
                bboxes = mask_to_bboxes(mask)
                if not bboxes:
                    prompts_filtered += 1; continue
                prompts_kept += 1
                if prompt == "animal":
                    has_animal = True
                elif prompt == "computer":
                    has_computer = True
                for cx, cy, bw, bh in bboxes:
                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            empty_skipped += 1; continue

        src = IMAGE_ROOT / img_path
        dst = images_dir / f"{img_idx:06d}.jpg"
        _write_image(src, dst)
        (labels_dir / f"{img_idx:06d}.txt").write_text("\n".join(label_lines), encoding="utf-8")

        if do_oversample and has_animal:
            animal_records.append((img_idx, img_path, label_lines))
        if do_oversample and has_computer:
            computer_indices.append(img_idx)
            computer_labels[img_idx] = label_lines

        img_idx += 1

    total_base = img_idx
    oversample_log = defaultdict(int)

    # ── 阶段 A：animal 过采样（RARE_OVERSAMPLE["animal"] - 1 次）──
    animal_mult = RARE_OVERSAMPLE.get("animal", 1)
    if do_oversample and animal_mult > 1 and animal_records:
        copy_idx = total_base
        for _, img_path, label_lines in animal_records:
            n = _copy_oversample(copy_idx, img_path, label_lines,
                                 images_dir, labels_dir, animal_mult - 1)
            copy_idx += n
            oversample_log["animal"] += n
        total_base = copy_idx

    # ── 阶段 B：computer 额外过采样（COMPUTER_EXTRA 次，与 animal 解耦）──
    if do_oversample and COMPUTER_EXTRA > 0 and computer_indices:
        copy_idx = total_base
        for cidx in computer_indices:
            # 找到原始图片路径（从第一个 animal_record 取不到，需要重建）
            # 直接从 dataset 记录中获取：用 computer_labels 持有的 label_lines
            label_lines = computer_labels[cidx]
            # 重建原始路径：通过文件名索引反查... 这里我们改用存储路径的方式
            src_img = images_dir / f"{cidx:06d}.jpg"
            n_copies = 0
            for _ in range(COMPUTER_EXTRA):
                dst_img = images_dir / f"{copy_idx + n_copies:06d}.jpg"
                dst_label = labels_dir / f"{copy_idx + n_copies:06d}.txt"
                # 硬链接 / 复制
                if src_img.exists():
                    try:
                        os.link(str(src_img), str(dst_img))
                    except OSError:
                        shutil.copy2(str(src_img), str(dst_img))
                dst_label.write_text("\n".join(label_lines), encoding="utf-8")
                n_copies += 1
            copy_idx += n_copies
            oversample_log["computer"] += n_copies
        total_base = copy_idx

    logger.info(f"  图片: {total_unique} → {total_base} 张, 跳过空: {empty_skipped}")
    logger.info(f"  实例保留: {prompts_kept}, 过滤: {prompts_filtered}")
    if oversample_log:
        logger.info(f"  过采样: animal={animal_mult}×(+{oversample_log['animal']}), "
                    f"computer=1+{COMPUTER_EXTRA}×(+{oversample_log['computer']})")
    return total_base


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
# v2: 多尺度推理与评估
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
    pred = pred_mask.astype(bool); gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-7)


def _multi_scale_detect(yolo, img_rgb, prompt, base_imgsz, scales, conf, nms_iou, device):
    """
    多尺度 YOLO-World 检测 → 跨尺度 NMS 合并。

    Args:
        yolo: YOLO-World 模型实例
        img_rgb: RGB 图像 (H, W, 3)
        prompt: 文本 prompt
        base_imgsz: 基准推理尺寸
        scales: 尺度因子列表，如 [0.8, 1.0, 1.25, 1.5]
        conf: 检测置信度阈值
        nms_iou: 跨尺度 NMS 的 IoU 阈值
        device: GPU 设备

    Returns:
        list of [x1, y1, x2, y2] 经过跨尺度 NMS 后的 bbox（原图坐标）
    """
    all_boxes = []  # [(x1, y1, x2, y2, conf), ...]

    # CLIP 文本编码只做一次，所有尺度复用
    yolo.set_classes([prompt])

    for scale in scales:
        imgsz = int(base_imgsz * scale)
        results = yolo.predict(
            img_rgb, imgsz=imgsz, conf=conf,
            device=device, verbose=False,
        )
        if results and len(results[0].boxes) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes_xyxy, confs):
                all_boxes.append((*box.tolist(), float(c)))

    if not all_boxes:
        return []

    # ── 跨尺度 NMS：按置信度降序，贪心剔除高重叠框 ──
    all_boxes.sort(key=lambda x: x[4], reverse=True)
    kept = []

    for box in all_boxes:
        x1, y1, x2, y2, box_conf = box
        # 跳过无效框
        if x2 <= x1 or y2 <= y1:
            continue
        overlap = False
        for kept_box in kept:
            kx1, ky1, kx2, ky2, _ = kept_box
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

    # 返回 bbox 坐标（去掉置信度）
    return [[b[0], b[1], b[2], b[3]] for b in kept]


def _fastsam_segment_boxes(fastsam, img_rgb, boxes, h, w):
    """
    用 FastSAM 对每个 bbox 区域做零样本分割 → 合并为并集 mask。

    imgsz 按 crop 实际尺寸动态选择，避免小 crop 被强行拉伸到 1024。
    """
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            continue
        crop = img_rgb[y1:y2, x1:x2]

        # 动态 imgsz: max(256, min(1024, max(crop_h, crop_w)))，向上取整到 32 的倍数
        max_side = max(crop_h, crop_w)
        dynamic_imgsz = max(256, min(Fastsam_IMG_SIZE, max_side))
        dynamic_imgsz = ((dynamic_imgsz + 31) // 32) * 32

        fs_results = fastsam.predict(
            source=crop, imgsz=dynamic_imgsz,
            conf=Fastsam_CONF, iou=Fastsam_IOU,
            device=DEVICE, verbose=False, retina_masks=True,
        )
        if fs_results and fs_results[0].masks is not None:
            for cm_tensor in fs_results[0].masks.data:
                cm = cm_tensor.cpu().numpy().astype(np.uint8)
                cm = cv2.resize(cm, (crop_w, crop_h))
                pred_mask[y1:y2, x1:x2] |= cm
    return pred_mask


def two_stage_eval(yolo_pt, pred_json, image_list_path, max_samples=None):
    """
    两阶段评估（v2：person/tree 走多尺度推理）。

    1. YOLO-World 检测 → bbox
       - person/tree: 多尺度 [0.8/1.0/1.25/1.5]× → 跨尺度 NMS
       - 其他类别: 单尺度
    2. FastSAM 逐框分割 → 合并 mask
    3. 与 SAM3 伪标签比对 mIoU + 同义词泛化测试
    """
    from ultralytics import YOLO, FastSAM

    # ── 加载模型 ──
    yolo = YOLO(str(yolo_pt), task="detect")
    yolo.to(f"cuda:{DEVICE}")
    if not os.path.exists(Fastsam_MODEL):
        logger.info(f"FastSAM 权重不存在，尝试自动下载: {Fastsam_MODEL}")
    fastsam = FastSAM(Fastsam_MODEL)
    logger.info("YOLO-World + FastSAM 已加载")

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
    logger.info(f"  多尺度推理: {MULTI_SCALE_CLASSES} @ {MULTI_SCALES}")
    logger.info(f"  单尺度推理: {[c for c in CLASSES if c not in MULTI_SCALE_CLASSES]}")

    per_class_iou = defaultdict(list)
    ms_stats = defaultdict(int)  # 多尺度统计
    t0 = time.time()

    for img_path, prompts_dict in tqdm(by_image.items(), desc="  两阶段评估"):
        abs_path = str(IMAGE_ROOT / img_path)
        if not os.path.exists(abs_path): continue

        # 图像预处理：BGR → 灰度 → 反色统一 → RGB
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

            # ── v2: 小目标类走多尺度推理 ──
            if prompt in MULTI_SCALE_CLASSES:
                boxes = _multi_scale_detect(
                    yolo, img_rgb, prompt,
                    base_imgsz=IMG_SIZE,
                    scales=MULTI_SCALES,
                    conf=MS_CONF,
                    nms_iou=MS_NMS_IOU,
                    device=DEVICE,
                )
                ms_stats[f"{prompt}_ms_calls"] += 1
                ms_stats[f"{prompt}_ms_boxes"] += len(boxes)
            else:
                yolo.set_classes([prompt])
                det_results = yolo.predict(
                    img_rgb, imgsz=IMG_SIZE, conf=0.25,
                    device=DEVICE, verbose=False,
                )
                boxes = []
                if det_results and len(det_results[0].boxes) > 0:
                    boxes = det_results[0].boxes.xyxy.cpu().numpy().tolist()

            # FastSAM 分割 + 合并
            pred_mask = _fastsam_segment_boxes(fastsam, img_rgb, boxes, h, w)
            per_class_iou[prompt].append(compute_iou(pred_mask, gt_mask))

    elapsed = time.time() - t0

    # ── 输出 ──
    summary = {}
    logger.info(f"\n  两阶段 mask mIoU（{len(by_image)} 张, {elapsed:.0f}s）:")
    for cls_name in CLASSES:
        if per_class_iou[cls_name]:
            miou = np.mean(per_class_iou[cls_name])
            summary[f"iou/{cls_name}"] = miou
            tag = " [MS]" if cls_name in MULTI_SCALE_CLASSES else ""
            logger.info(f"    {cls_name:10s}{tag}: {miou:.4f}  (n={len(per_class_iou[cls_name])})")

    overall = np.mean([v for vals in per_class_iou.values() for v in vals])
    summary["iou/overall"] = overall
    logger.info(f"    {'OVERALL':10s} : {overall:.4f}")

    # 多尺度统计
    if ms_stats:
        logger.info(f"\n  多尺度推理统计:")
        for cls_name in MULTI_SCALE_CLASSES:
            calls = ms_stats.get(f"{cls_name}_ms_calls", 0)
            boxes = ms_stats.get(f"{cls_name}_ms_boxes", 0)
            if calls > 0:
                logger.info(f"    {cls_name}: avg {boxes/calls:.1f} boxes/img ({calls} 张)")

    # ── v2: 增强同义词泛化测试 ──
    logger.info(f"\n  同义词泛化测试（推理时 prompt 映射）:")
    class_synonyms = defaultdict(list)
    for syn, orig in PROMPT_MAP.items():
        if orig in CLASSES:
            class_synonyms[orig].append(syn)

    for cls_name, synonyms in class_synonyms.items():
        if len(by_image) == 0: break
        # animal 测试更多同义词，其余类别测前 3 个
        test_syns = synonyms[:8] if cls_name == "animal" else synonyms[:3]
        for syn in test_syns:
            syn_ious = []
            sample_imgs = list(by_image.items())[:50]
            for img_path, prompts_dict in sample_imgs:
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

                # 同义词推理用单尺度（测试泛化，不引入多尺度变量）
                yolo.set_classes([syn])
                det_results = yolo.predict(
                    img_rgb, imgsz=IMG_SIZE, conf=0.25,
                    device=DEVICE, verbose=False,
                )
                syn_boxes = []
                if det_results and len(det_results[0].boxes) > 0:
                    syn_boxes = det_results[0].boxes.xyxy.cpu().numpy().tolist()

                pred_mask = _fastsam_segment_boxes(fastsam, img_rgb, syn_boxes, h, w)
                syn_ious.append(compute_iou(pred_mask, gt_mask))

            if syn_ious:
                logger.info(f"    {syn:20s} → {cls_name:10s}: {np.mean(syn_ious):.4f}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# GPU 信息
# ══════════════════════════════════════════════════════════════════════

def log_gpu_info():
    """记录 GPU 信息。兼容不同 PyTorch 版本和 CPU-only 环境。"""
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU 训练（极慢）")
        logger.info(f"  PyTorch: {torch.__version__}")
        return

    gpu_name = torch.cuda.get_device_name(0)

    # 不同 PyTorch 版本属性名不同
    props = torch.cuda.get_device_properties(0)
    if hasattr(props, "total_memory"):
        total_vram = props.total_memory / 1024**3
    elif hasattr(props, "total_mem"):
        total_vram = props.total_mem / 1024**3
    else:
        total_vram = 0

    try:
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
    except Exception:
        reserved = allocated = 0

    logger.info(f"GPU: {gpu_name}")
    logger.info(f"  显存: {total_vram:.1f} GB | 已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB")
    logger.info(f"  CUDA: {torch.version.cuda if torch.version.cuda else 'N/A'} | PyTorch: {torch.__version__}")
    logger.info(f"  GPU 数量: {torch.cuda.device_count()}")


# ══════════════════════════════════════════════════════════════════════
# 训练 —— v2 增强版：捕获指标 + 自定义折线图 + 详细总结
# ══════════════════════════════════════════════════════════════════════

def plot_training_curves(results_csv, run_dir):
    """
    从 Ultralytics results.csv 生成自定义多面板折线图。
    面板布局 3×2：
      1. train 三 loss（box/cls/dfl）
      2. val 三 loss（box/cls/dfl）
      3. mAP50 + mAP50-95
      4. Precision + Recall
      5. 学习率曲线
      6. 训练摘要文字
    """
    if not results_csv.exists():
        logger.warning(f"results.csv 不存在，跳过绘图: {results_csv}")
        return

    # 读取 CSV（Ultralytics 会在列名前加空格）
    import csv
    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    # 标准化列名（去前后空格）
    cols = list(rows[0].keys())
    col_map = {c.strip(): c for c in cols}

    def _get(col_suffix):
        """获取包含 suffix 的列数据，找不到返回空列表。"""
        for c_clean, c_orig in col_map.items():
            if col_suffix in c_clean:
                return [float(r[c_orig]) for r in rows]
        return []

    epochs = list(range(1, len(rows) + 1))

    train_box  = _get("train/box_loss")
    train_cls  = _get("train/cls_loss")
    train_dfl  = _get("train/dfl_loss")
    val_box    = _get("val/box_loss")
    val_cls    = _get("val/cls_loss")
    val_dfl    = _get("val/dfl_loss")
    map50      = _get("metrics/mAP50(B)")
    map50_95   = _get("metrics/mAP50-95(B)")
    precision  = _get("metrics/precision(B)")
    recall     = _get("metrics/recall(B)")
    lr_vals    = _get("lr/pg0")

    fig = plt.figure(figsize=(18, 12))

    # 1. Train Loss
    ax1 = fig.add_subplot(2, 3, 1)
    for vals, label, color in [(train_box, "box", "#1f77b4"),
                                 (train_cls, "cls", "#ff7f0e"),
                                 (train_dfl, "dfl", "#2ca02c")]:
        if vals: ax1.plot(epochs, vals, "-", label=label, color=color, linewidth=1, alpha=0.85)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("1. Train Loss")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # 2. Val Loss
    ax2 = fig.add_subplot(2, 3, 2)
    for vals, label, color in [(val_box, "box", "#1f77b4"),
                                 (val_cls, "cls", "#ff7f0e"),
                                 (val_dfl, "dfl", "#2ca02c")]:
        if vals: ax2.plot(epochs, vals, "-", label=label, color=color, linewidth=1, alpha=0.85)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("2. Val Loss")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # 3. mAP
    ax3 = fig.add_subplot(2, 3, 3)
    if map50: ax3.plot(epochs, map50, "-", label="mAP50", color="#d62728", linewidth=1.5)
    if map50_95: ax3.plot(epochs, map50_95, "-", label="mAP50-95", color="#9467bd", linewidth=1.5)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("mAP"); ax3.set_title("3. Detection mAP (BBox)")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # 4. Precision / Recall
    ax4 = fig.add_subplot(2, 3, 4)
    if precision: ax4.plot(epochs, precision, "-", label="Precision", color="#8c564b", linewidth=1.5)
    if recall: ax4.plot(epochs, recall, "-", label="Recall", color="#e377c2", linewidth=1.5)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Score"); ax4.set_title("4. Precision & Recall")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)

    # 5. Learning Rate
    ax5 = fig.add_subplot(2, 3, 5)
    if lr_vals: ax5.plot(epochs, lr_vals, "-", color="red", linewidth=1)
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("LR"); ax5.set_title("5. Learning Rate")
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # 6. Text Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = ["── Training Summary ──", ""]
    if map50:
        best_idx = np.argmax(map50)
        lines.append(f"Best mAP50:        {map50[best_idx]:.4f}  (epoch {epochs[best_idx]})")
    if map50_95:
        best_idx_95 = np.argmax(map50_95)
        lines.append(f"Best mAP50-95:     {map50_95[best_idx_95]:.4f}  (epoch {epochs[best_idx_95]})")
    if precision:
        lines.append(f"Final Precision:   {precision[-1]:.4f}")
    if recall:
        lines.append(f"Final Recall:      {recall[-1]:.4f}")
    lines.append(f"Total Epochs:      {len(epochs)}")
    lines.append(f"Early Stop:        {'Yes' if len(epochs) < EPOCHS else 'No (full)'}")
    # 收敛分析
    if map50 and len(map50) >= 10:
        last10_max = max(map50[-10:])
        lines.append(f"Last10 max mAP50:  {last10_max:.4f}")
    for i, line in enumerate(lines):
        ax6.text(0.05, 0.95 - i * 0.06, line, transform=ax6.transAxes,
                 fontsize=9, family="monospace", verticalalignment="top")

    plt.tight_layout()
    out_path = run_dir / "training_curves_v2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线图已保存: {out_path}")


def train():
    """阶段一：YOLO-World v2 检测训练（v2 增强：捕获指标 + 绘图 + 详细总结）。"""
    run_dir = PROJECT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL, task=YOLO_TASK)
    model.set_classes(CLASSES)
    logger.info(f"模型: {YOLO_MODEL} task={YOLO_TASK} classes={CLASSES}")

    t0 = time.time()
    results = model.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        device=DEVICE, workers=WORKERS, seed=SEED,
        optimizer=OPTIMIZER, lr0=LR0, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR, patience=PATIENCE, amp=True,
        cls=CLS_LOSS_GAIN,
        box=BOX_LOSS_GAIN,
        label_smoothing=LABEL_SMOOTHING,
        mosaic=MOSAIC, mixup=MIXUP, copy_paste=COPY_PASTE,
        scale=SCALE, degrees=DEGREES, flipud=FLIPUD, fliplr=FLIPLR,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V, close_mosaic=CLOSE_MOSAIC,
        project=str(PROJECT), name=RUN_NAME, exist_ok=EXIST_OK,
        save=True, plots=True, verbose=False,
    )
    elapsed = time.time() - t0

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    # ── 训练总结 ──
    logger.info(f"\n{'─' * 50}")
    logger.info(f"阶段一训练完成 | 耗时: {elapsed/60:.1f} min")
    logger.info(f"  best.pt:  {'✓' if best_pt.exists() else '✗ 缺失'}")
    logger.info(f"  last.pt:  {'✓' if last_pt.exists() else '✗ 缺失'}")

    # 从 CSV 提取关键指标
    if results_csv.exists():
        import csv
        with open(results_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            col_map = {c.strip(): c for c in last.keys()}
            def _v(suffix):
                for c_clean, c_orig in col_map.items():
                    if suffix in c_clean: return last[c_orig]
                return "N/A"

            logger.info(f"  总 epochs:     {len(rows)}/{EPOCHS}")
            logger.info(f"  mAP50:         {_v('metrics/mAP50(B)')}")
            logger.info(f"  mAP50-95:      {_v('metrics/mAP50-95(B)')}")
            logger.info(f"  Precision:     {_v('metrics/precision(B)')}")
            logger.info(f"  Recall:        {_v('metrics/recall(B)')}")
            train_box = _v("train/box_loss"); train_cls = _v("train/cls_loss")
            val_box = _v("val/box_loss"); val_cls = _v("val/cls_loss")
            logger.info(f"  train box/cls: {train_box} / {train_cls}")
            logger.info(f"  val box/cls:   {val_box} / {val_cls}")
            if len(rows) < EPOCHS:
                logger.info(f"  ⚠ 早停触发于 epoch {len(rows)} (patience={PATIENCE})")

        # 生成自定义曲线图
        plot_training_curves(results_csv, run_dir)

    logger.info(f"{'─' * 50}")
    return best_pt


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    import torch
    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA，请安装 CUDA 版 PyTorch"); sys.exit(1)

    log_gpu_info()
    logger.info("")  # 空行
    logger.info("=" * 60)
    logger.info("YOLO-World + FastSAM v2 两阶段训练与评估")
    logger.info(f"  v2 改动:")
    logger.info(f"    - 多尺度推理: {MULTI_SCALE_CLASSES} @ {MULTI_SCALES}")
    logger.info(f"    - cls={CLS_LOSS_GAIN} box={BOX_LOSS_GAIN} label_smoothing={LABEL_SMOOTHING}")
    logger.info(f"    - computer 阈值 {CONF_FILTER['computer']}, 过采样 1+{COMPUTER_EXTRA}×")
    logger.info(f"    - animal 过采样 {RARE_OVERSAMPLE['animal']}×")
    logger.info(f"  阶段一: {YOLO_MODEL} (检测, epochs={EPOCHS})")
    logger.info(f"  阶段二: {Fastsam_MODEL} (零样本分割)")
    logger.info(f"  类别: {CLASSES}")
    logger.info("=" * 60)

    # ── 数据集转换 ──
    train_set = load_file_list(TRAIN_LIST)
    val_set = load_file_list(VAL_LIST)
    dataset_yaml = YOLO_DIR / "dataset.yaml"

    if dataset_yaml.exists():
        n_train = len(list(TRAIN_IMAGES_DIR.glob("*.jpg")))
        n_val = len(list(VAL_IMAGES_DIR.glob("*.jpg")))
        logger.info(f"数据集已存在: train={n_train}, val={n_val}")
    else:
        n_train = convert_split("train", PRED_JSON, train_set,
                                TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, do_oversample=True)
        n_val = convert_split("val", VAL_PRED_JSON, val_set,
                              VAL_IMAGES_DIR, VAL_LABELS_DIR, do_oversample=False) \
                if VAL_PRED_JSON.exists() else 0
        create_dataset_yaml()
    logger.info(f"训练集: {n_train}, 验证集: {n_val}")

    # ── 阶段一：检测训练 ──
    best_pt = train()

    # ── 阶段二：FastSAM 两阶段 mask 评估（v2: 多尺度推理）──
    if best_pt and best_pt.exists() and VAL_PRED_JSON.exists():
        logger.info(f"\n{'─' * 50}\n两阶段 mask mIoU 评估 (v2 多尺度)")
        two_stage_eval(best_pt, VAL_PRED_JSON, VAL_LIST)

    logger.info(f"\n日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
