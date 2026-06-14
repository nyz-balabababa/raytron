#!/usr/bin/env python3
"""
SegFormer-B0 学生模型推理和评估脚本
- 加载训练好的模型
- 对test集进行多标签分割推理
- 输出RLE格式的预测结果
"""
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "train" / "clipseg-v4"))

from config_clipseg import (
    MODEL_DIR, CLASSES, NUM_CLASSES, IMG_SIZE,
    DEVICE, VAL_THRESHOLDS, IMAGENET_MEAN, IMAGENET_STD,
    PROJECT, RUN_NAME,
)
from clipseg_train import load_teacher_aligned_gray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import SegformerForSemanticSegmentation


def load_pretrained_model(model_path):
    """加载训练好的模型"""
    logger.info(f"从 {model_path} 加载模型...")
    model = SegformerForSemanticSegmentation.from_pretrained(str(MODEL_DIR))

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)
    model.eval()
    logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def preprocess_image(image_path, img_size=IMG_SIZE):
    """预处理图像为模型输入"""
    img = load_teacher_aligned_gray(image_path)
    h, w = img.shape

    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    ph, pw = img_size - nh, img_size - nw
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)

    # 灰度转RGB
    img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.float32) / 255.0

    # 标准化（ImageNet）
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, (h, w), (pt, pb, pl, pr)


def postprocess_mask(logits, original_size, pad_info, class_idx, threshold=0.5):
    """后处理预测掩码"""
    pt, pb, pl, pr = pad_info
    h_orig, w_orig = original_size

    # 移除padding
    logits = logits[:IMG_SIZE - pb, IMG_SIZE - pr] if pb > 0 and pr > 0 else logits
    logits = logits[:IMG_SIZE - pb, :] if pb > 0 else logits
    logits = logits[:, :IMG_SIZE - pr] if pr > 0 else logits

    # 调整到原始大小
    scale = max(h_orig, w_orig) / IMG_SIZE
    nh, nw = int(h_orig / scale), int(w_orig / scale)
    mask = cv2.resize(logits, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 二值化
    mask_bin = (mask > threshold).astype(np.uint8)
    return mask_bin


def mask_to_rle(mask):
    """将二值掩码转换为RLE格式（COCO标准）"""
    from pycocotools import mask as maskUtils
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.no_grad()
def infer(model, test_tasks_json, output_json):
    """对test集进行推理"""
    with open(test_tasks_json, encoding="utf-8") as f:
        tasks = json.load(f)

    predictions = {}
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    for task in tqdm(tasks, desc="推理"):
        ann_id = task["ann_id"]
        image_path = ROOT / task["image_path"]
        text_prompt = task.get("text_prompt", "")

        if not image_path.exists():
            logger.warning(f"图片不存在: {image_path}")
            predictions[ann_id] = {"prompt": text_prompt, "rle": {"size": [0, 0], "counts": ""}}
            continue

        # 预处理
        img_tensor, orig_size, pad_info = preprocess_image(image_path)

        # 推理
        outputs = model(pixel_values=img_tensor.to(DEVICE))
        logits = outputs.logits  # [1, NUM_CLASSES, H, W]

        # 映射prompt到类别索引
        if text_prompt in class_to_idx:
            cls_idx = class_to_idx[text_prompt]
            logit_map = torch.sigmoid(logits[0, cls_idx]).cpu().numpy()
            threshold = VAL_THRESHOLDS.get(text_prompt, 0.5) if isinstance(VAL_THRESHOLDS, dict) else 0.5
        else:
            # 未知prompt时，取所有类别的最大概率
            logit_map = torch.sigmoid(logits[0]).max(dim=0).values.cpu().numpy()
            threshold = 0.5

        # 后处理和二值化
        mask_bin = postprocess_mask(logit_map, orig_size, pad_info, None, threshold=threshold)

        # 转RLE
        rle = mask_to_rle(mask_bin)
        predictions[ann_id] = {"prompt": text_prompt, "rle": rle}

    # 保存预测结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"预测结果保存至: {output_json}")
    return predictions


def evaluate(predictions_json, gt_json=None):
    """评估预测结果（如果有gt）"""
    with open(predictions_json, encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"推理完成，总样本数: {len(predictions)}")

    if gt_json and Path(gt_json).exists():
        with open(gt_json, encoding="utf-8") as f:
            ground_truth = json.load(f)

        # 简单IoU计算
        from pycocotools import mask as maskUtils
        ious = []
        for ann_id, pred in predictions.items():
            if ann_id not in ground_truth:
                continue
            rle_pred = pred["rle"]
            rle_gt = ground_truth[ann_id]["rle"]

            mask_pred = maskUtils.decode(rle_pred)
            mask_gt = maskUtils.decode(rle_gt)

            intersection = (mask_pred & mask_gt).sum()
            union = (mask_pred | mask_gt).sum()
            iou = intersection / (union + 1e-7)
            ious.append(iou)

        mean_iou = np.mean(ious) if ious else 0.0
        logger.info(f"Mean IoU: {mean_iou:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="模型checkpoint路径 (default: best.pt)")
    parser.add_argument("--test_tasks", type=str, default="test/test_tasks.json")
    parser.add_argument("--output", type=str, default=None, help="输出JSON路径 (default: train_output/predictions.json)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else PROJECT / RUN_NAME / "best.pt"
    test_tasks_json = ROOT / args.test_tasks
    output_json = Path(args.output) if args.output else PROJECT / RUN_NAME / "predictions.json"

    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)

    if not test_tasks_json.exists():
        logger.error(f"测试任务文件不存在: {test_tasks_json}")
        sys.exit(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    model = load_pretrained_model(model_path)
    predictions = infer(model, test_tasks_json, output_json)
    evaluate(output_json)


if __name__ == "__main__":
    main()
