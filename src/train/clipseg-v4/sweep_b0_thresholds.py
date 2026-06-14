#!/usr/bin/env python3
"""
SegFormer-B0 11类阈值扫描脚本

用途：
1. 检查当前 checkpoint 在验证集上是否因为阈值过高导致 mIoU 接近 0
2. 检查模型是否已经塌成全背景（即使把阈值降得很低也几乎不出前景）
3. 输出统一阈值 sweep、配置阈值评估、每类最佳阈值诊断
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_clipseg import (  # noqa: E402
    APPLY_SCORE_FILTER,
    CLASSES,
    DEVICE,
    IMAGE_ROOT,
    IMG_SIZE,
    INCLUDE_NEGATIVE_SAMPLES,
    MODEL_DIR,
    MODEL_NAME,
    NEGATIVE_SAMPLE_WEIGHT,
    NUM_CLASSES,
    OLD5_CLASSES,
    RARE_CLASSES,
    RARE_OVERSAMPLE,
    SEED,
    VAL_INCLUDE_NEGATIVE_SAMPLES,
    VAL_LIST,
    VAL_PRED_JSON,
    VAL_THRESHOLDS,
    WORKERS,
)
import clipseg_train as train_mod  # noqa: E402
from clipseg_train import SegFormerMultiLabelDataset, collate_fn  # noqa: E402


LOGGER = logging.getLogger("sweep_b0_thresholds")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep SegFormer-B0 validation thresholds")
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评估 checkpoint 路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON 路径")
    parser.add_argument("--batch_size", type=int, default=4, help="验证 batch size")
    parser.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="统一阈值扫描列表，逗号分隔",
    )
    parser.add_argument("--device", type=str, default=None, help="覆盖 config DEVICE")
    parser.add_argument("--max_samples", type=int, default=0, help="只评估前 N 个样本，0 表示全量")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_override=None):
    if device_override is not None:
        requested = str(device_override)
    elif isinstance(DEVICE, int):
        requested = f"cuda:{DEVICE}"
    else:
        requested = str(DEVICE)

    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("请求 CUDA 但当前不可用，自动回退到 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def parse_thresholds(raw: str):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0 or value >= 1:
            raise ValueError(f"threshold 必须位于 (0, 1)，收到: {value}")
        values.append(round(value, 4))
    if not values:
        raise ValueError("threshold 列表为空")
    return sorted(set(values))


def load_model_for_eval(device: torch.device):
    local_model_ok = MODEL_DIR.exists() and (
        (MODEL_DIR / "pytorch_model.bin").exists()
        or (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "config.json").exists()
    )
    if local_model_ok:
        LOGGER.info("从本地加载 SegFormer backbone: %s", MODEL_DIR)
        model = SegformerForSemanticSegmentation.from_pretrained(str(MODEL_DIR), local_files_only=True)
    else:
        LOGGER.info("从 HuggingFace 加载 SegFormer backbone: %s", MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    in_channels = model.decode_head.classifier.in_channels
    model.decode_head.classifier = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
    model = model.to(device)
    model.eval()
    return model


def extract_model_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict"]:
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("checkpoint 中未找到可用的 model state_dict")


def load_checkpoint(model, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = extract_model_state(ckpt)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("missing_keys=%d unexpected_keys=%d", len(missing_keys), len(unexpected_keys))
    if missing_keys:
        LOGGER.info("missing_keys_preview=%s", missing_keys[:10])
    if unexpected_keys:
        LOGGER.info("unexpected_keys_preview=%s", unexpected_keys[:10])
    return ckpt, model


def build_val_loader(batch_size: int, workers: int, max_samples: int = 0):
    # 只在 sweep 脚本里关闭随机翻转，确保评估稳定；不修改训练脚本源码。
    train_mod.HFLIP_PROB = 0.0

    val_ds = SegFormerMultiLabelDataset(
        VAL_PRED_JSON,
        VAL_LIST,
        oversample=None,
        include_negatives=VAL_INCLUDE_NEGATIVE_SAMPLES,
        negative_ratio=1.0 if VAL_INCLUDE_NEGATIVE_SAMPLES else 0.0,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=APPLY_SCORE_FILTER,
        use_manifest=False,
        teacher_soft_cache=None,
    )
    if len(val_ds) <= 0:
        raise RuntimeError("验证集为空，请检查 VAL_PRED_JSON / VAL_LIST / prompt 过滤逻辑。")
    if max_samples > 0:
        val_ds.samples = val_ds.samples[:max_samples]
        LOGGER.info("限制验证样本数: %d", len(val_ds.samples))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return val_ds, val_loader


def empty_metric_summary():
    return {
        "count": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "pred_area_sum": 0.0,
        "gt_area_sum": 0.0,
        "pred_nonempty": 0,
        "gt_nonempty": 0,
        "both_empty": 0,
    }


def update_metric_summary(summary, prob_map, gt_map, threshold: float):
    pred_bin = prob_map > threshold
    gt_bin = gt_map > 0.5

    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    intersection = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())

    if pred_sum == 0 and gt_sum == 0:
        iou = dice = precision = recall = 1.0
        summary["both_empty"] += 1
    else:
        fp = float(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
        fn = float(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
        iou = intersection / (union + 1e-7)
        dice = 2 * intersection / (pred_sum + gt_sum + 1e-7)
        precision = intersection / (intersection + fp + 1e-7)
        recall = intersection / (intersection + fn + 1e-7)

    summary["count"] += 1
    summary["iou_sum"] += iou
    summary["dice_sum"] += dice
    summary["precision_sum"] += precision
    summary["recall_sum"] += recall
    summary["pred_area_sum"] += pred_sum
    summary["gt_area_sum"] += gt_sum
    summary["pred_nonempty"] += int(pred_sum > 0)
    summary["gt_nonempty"] += int(gt_sum > 0)


def finalize_metric_summary(summary):
    count = max(int(summary["count"]), 1)
    return {
        "count": int(summary["count"]),
        "iou": float(summary["iou_sum"] / count),
        "dice": float(summary["dice_sum"] / count),
        "precision": float(summary["precision_sum"] / count),
        "recall": float(summary["recall_sum"] / count),
        "pred_area_mean": float(summary["pred_area_sum"] / count),
        "gt_area_mean": float(summary["gt_area_sum"] / count),
        "pred_nonempty_ratio": float(summary["pred_nonempty"] / count),
        "gt_nonempty_ratio": float(summary["gt_nonempty"] / count),
        "both_empty_ratio": float(summary["both_empty"] / count),
    }


def quantile_or_none(values, q):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


@torch.no_grad()
def run_sweep(model, loader, threshold_grid, config_thresholds, device):
    global_summaries = {thr: defaultdict(empty_metric_summary) for thr in threshold_grid}
    config_summaries = defaultdict(empty_metric_summary)
    per_class_summaries = {
        cls_name: {thr: empty_metric_summary() for thr in threshold_grid}
        for cls_name in CLASSES
    }
    raw_prob_stats = {
        cls_name: {
            "sample_count": 0,
            "positive_sample_count": 0,
            "max_prob": [],
            "mean_prob": [],
            "fg_mean_prob": [],
            "bg_mean_prob": [],
        }
        for cls_name in CLASSES
    }

    for batch in tqdm(loader, desc="Sweep", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        imgs, class_indices, masks, _, class_names, _, _, _, _ = batch
        imgs = imgs.to(device)
        masks = masks.to(device)
        class_indices = class_indices.to(device)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)

        batch_size = imgs.size(0)
        for b in range(batch_size):
            cls_idx = class_indices[b].item()
            cls_name = class_names[b]
            prob_map = probs[b, cls_idx].detach().cpu().numpy().astype(np.float32, copy=False)
            gt_map = masks[b, 0].detach().cpu().numpy().astype(np.float32, copy=False)

            class_stats = raw_prob_stats[cls_name]
            class_stats["sample_count"] += 1
            class_stats["max_prob"].append(float(prob_map.max()))
            class_stats["mean_prob"].append(float(prob_map.mean()))

            gt_fg = gt_map > 0.5
            if bool(gt_fg.any()):
                class_stats["positive_sample_count"] += 1
                class_stats["fg_mean_prob"].append(float(prob_map[gt_fg].mean()))
                class_stats["bg_mean_prob"].append(float(prob_map[~gt_fg].mean()) if bool((~gt_fg).any()) else 0.0)

            config_thr = float(config_thresholds.get(cls_name, 0.5))
            update_metric_summary(config_summaries["overall"], prob_map, gt_map, config_thr)
            update_metric_summary(config_summaries[cls_name], prob_map, gt_map, config_thr)
            if cls_name in OLD5_CLASSES:
                update_metric_summary(config_summaries["old5"], prob_map, gt_map, config_thr)
            if cls_name in RARE_CLASSES:
                update_metric_summary(config_summaries["rare"], prob_map, gt_map, config_thr)

            for thr in threshold_grid:
                update_metric_summary(global_summaries[thr]["overall"], prob_map, gt_map, thr)
                update_metric_summary(global_summaries[thr][cls_name], prob_map, gt_map, thr)
                update_metric_summary(per_class_summaries[cls_name][thr], prob_map, gt_map, thr)
                if cls_name in OLD5_CLASSES:
                    update_metric_summary(global_summaries[thr]["old5"], prob_map, gt_map, thr)
                if cls_name in RARE_CLASSES:
                    update_metric_summary(global_summaries[thr]["rare"], prob_map, gt_map, thr)

    config_eval = {key: finalize_metric_summary(summary) for key, summary in config_summaries.items()}

    global_eval = {}
    for thr, metrics_dict in global_summaries.items():
        thr_key = f"{thr:.2f}"
        global_eval[thr_key] = {key: finalize_metric_summary(summary) for key, summary in metrics_dict.items()}

    per_class_best = {}
    for cls_name, threshold_dict in per_class_summaries.items():
        best_thr = None
        best_metrics = None
        best_iou = -1.0
        sweep_metrics = {}
        for thr, summary in threshold_dict.items():
            metrics = finalize_metric_summary(summary)
            sweep_metrics[f"{thr:.2f}"] = metrics
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_thr = thr
                best_metrics = metrics
        per_class_best[cls_name] = {
            "best_threshold": float(best_thr) if best_thr is not None else None,
            "best_metrics": best_metrics,
            "sweep": sweep_metrics,
        }

    raw_prob_summary = {}
    for cls_name, stats in raw_prob_stats.items():
        raw_prob_summary[cls_name] = {
            "sample_count": int(stats["sample_count"]),
            "positive_sample_count": int(stats["positive_sample_count"]),
            "max_prob_mean": float(np.mean(stats["max_prob"])) if stats["max_prob"] else None,
            "max_prob_p50": quantile_or_none(stats["max_prob"], 0.50),
            "max_prob_p90": quantile_or_none(stats["max_prob"], 0.90),
            "max_prob_p99": quantile_or_none(stats["max_prob"], 0.99),
            "mean_prob_mean": float(np.mean(stats["mean_prob"])) if stats["mean_prob"] else None,
            "fg_mean_prob_mean": float(np.mean(stats["fg_mean_prob"])) if stats["fg_mean_prob"] else None,
            "bg_mean_prob_mean": float(np.mean(stats["bg_mean_prob"])) if stats["bg_mean_prob"] else None,
        }

    return config_eval, global_eval, per_class_best, raw_prob_summary


def pick_best_uniform_threshold(global_eval):
    best_key = None
    best_metrics = None
    best_iou = -1.0
    for thr_key, metrics in global_eval.items():
        overall = metrics.get("overall", {})
        iou = float(overall.get("iou", -1.0))
        if iou > best_iou:
            best_iou = iou
            best_key = thr_key
            best_metrics = metrics
    return best_key, best_metrics


def build_diagnosis(config_eval, global_eval, raw_prob_summary):
    best_uniform_threshold, best_uniform_metrics = pick_best_uniform_threshold(global_eval)
    baseline = config_eval.get("overall", {})
    low_005 = global_eval.get("0.05", {}).get("overall", {})

    baseline_iou = float(baseline.get("iou", 0.0))
    baseline_nonempty = float(baseline.get("pred_nonempty_ratio", 0.0))
    best_iou = float(best_uniform_metrics.get("overall", {}).get("iou", 0.0)) if best_uniform_metrics else 0.0
    low_nonempty = float(low_005.get("pred_nonempty_ratio", 0.0))

    max_prob_means = [
        item["max_prob_mean"]
        for item in raw_prob_summary.values()
        if item["max_prob_mean"] is not None
    ]
    mean_max_prob = float(np.mean(max_prob_means)) if max_prob_means else 0.0

    diagnosis = {
        "baseline_iou": baseline_iou,
        "baseline_pred_nonempty_ratio": baseline_nonempty,
        "best_uniform_threshold": float(best_uniform_threshold) if best_uniform_threshold is not None else None,
        "best_uniform_iou": best_iou,
        "low_threshold_0.05_pred_nonempty_ratio": low_nonempty,
        "mean_max_prob_across_classes": mean_max_prob,
        "threshold_too_high_suspected": bool(baseline_iou < 0.01 and best_iou > max(0.05, baseline_iou + 0.03)),
        "collapsed_to_background_suspected": bool(best_iou < 0.01 and low_nonempty < 0.05 and mean_max_prob < 0.10),
    }
    return diagnosis


def main():
    configure_logging()
    args = parse_args()
    set_seed(SEED)

    threshold_grid = parse_thresholds(args.thresholds)
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model_for_eval(device)
    ckpt, model = load_checkpoint(model, args.checkpoint, device)

    config_thresholds = ckpt.get("val_thresholds", VAL_THRESHOLDS)
    if not isinstance(config_thresholds, dict):
        config_thresholds = dict(VAL_THRESHOLDS)

    LOGGER.info("验证集：%s + %s", VAL_PRED_JSON, VAL_LIST)
    LOGGER.info("验证集不加入负样本: %s", not VAL_INCLUDE_NEGATIVE_SAMPLES)
    LOGGER.info("checkpoint 保存阈值: %s", config_thresholds)
    LOGGER.info("统一阈值扫描: %s", threshold_grid)

    _, loader = build_val_loader(args.batch_size, args.workers, max_samples=args.max_samples)

    config_eval, global_eval, per_class_best, raw_prob_summary = run_sweep(
        model=model,
        loader=loader,
        threshold_grid=threshold_grid,
        config_thresholds=config_thresholds,
        device=device,
    )
    diagnosis = build_diagnosis(config_eval, global_eval, raw_prob_summary)

    best_uniform_threshold, best_uniform_metrics = pick_best_uniform_threshold(global_eval)
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "img_size": int(ckpt.get("img_size", IMG_SIZE)),
        "classes": ckpt.get("classes", CLASSES),
        "num_classes": int(ckpt.get("num_classes", NUM_CLASSES)),
        "config_thresholds": config_thresholds,
        "config_eval": config_eval,
        "best_uniform_threshold": float(best_uniform_threshold) if best_uniform_threshold is not None else None,
        "best_uniform_metrics": best_uniform_metrics,
        "uniform_threshold_sweep": global_eval,
        "per_class_best_thresholds": per_class_best,
        "raw_probability_summary": raw_prob_summary,
        "diagnosis": diagnosis,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    LOGGER.info("baseline config mIoU_all11=%.4f", config_eval.get("overall", {}).get("iou", 0.0))
    if best_uniform_threshold is not None:
        LOGGER.info(
            "best uniform threshold=%s mIoU_all11=%.4f",
            best_uniform_threshold,
            best_uniform_metrics.get("overall", {}).get("iou", 0.0),
        )
    LOGGER.info("threshold_too_high_suspected=%s", diagnosis["threshold_too_high_suspected"])
    LOGGER.info("collapsed_to_background_suspected=%s", diagnosis["collapsed_to_background_suspected"])
    LOGGER.info("结果已保存: %s", args.output)


if __name__ == "__main__":
    main()
