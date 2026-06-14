#!/usr/bin/env python3
"""
CLIPSeg threshold sweep 入口。

默认使用 streaming 评估，不缓存全量 prob/gt map。
`--max_total_samples` 仅用于 debug 截断总样本。
`--max_samples_per_class` 用于 streaming sweep 的每类限样。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_opt import (
    ClipSegDataset,
    collate_fn,
    compute_binary_metrics,
    load_checkpoint_into_model,
    load_clipseg_model,
    load_config,
    postprocess_probability_map,
    predict_probability_batch,
    resolve_device,
    set_seed,
)


# =========================
# 顶部配置区
# 直接在 IDE 里运行时，优先修改这里
# =========================
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_opt.py"
DEFAULT_CHECKPOINT = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_BATCH_SIZE = None
DEFAULT_WORKERS = None
DEFAULT_PROMPT_FUSION = None
DEFAULT_TTA = None
DEFAULT_POSTPROCESS_OVERRIDE = None   # None / True / False
DEFAULT_DEVICE = None
DEFAULT_MAX_TOTAL_SAMPLES = 0
DEFAULT_MAX_SAMPLES_PER_CLASS = 0

LOGGER = logging.getLogger("clipseg_v5_sweep_opt")


def configure_logging():
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep per-class thresholds for CLIPSeg opt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prompt_fusion", type=str, choices=["none", "weighted_mean", "max"], default=DEFAULT_PROMPT_FUSION)
    parser.add_argument("--tta", type=str, choices=["none", "hflip"], default=DEFAULT_TTA)
    parser.add_argument("--postprocess", action="store_true", default=(DEFAULT_POSTPROCESS_OVERRIDE is True))
    parser.add_argument("--no_postprocess", action="store_true", default=(DEFAULT_POSTPROCESS_OVERRIDE is False))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--max_total_samples", type=int, default=DEFAULT_MAX_TOTAL_SAMPLES)
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def resolve_sample_limits(args):
    max_total_samples = int(args.max_total_samples or 0)
    max_samples_per_class = int(args.max_samples_per_class or 0)
    if int(args.max_samples or 0) > 0:
        LOGGER.warning("--max_samples 已废弃，当前按 --max_total_samples=%d 处理。", int(args.max_samples))
        if max_total_samples <= 0:
            max_total_samples = int(args.max_samples)
    return max_total_samples, max_samples_per_class


def build_val_loader(cfg, batch_size: int, workers: int, max_total_samples: int):
    val_ds = ClipSegDataset(
        cfg=cfg,
        pred_json=cfg.VAL_PRED_JSON,
        image_list_path=cfg.VAL_LIST,
        is_train=False,
        include_negatives=False,
        negative_ratio=0.0,
        negative_weight=cfg.NEGATIVE_SAMPLE_WEIGHT,
        oversample=None,
        apply_score_filter=cfg.APPLY_SCORE_FILTER,
        logger=LOGGER,
    )
    if max_total_samples > 0:
        val_ds.samples = val_ds.samples[:max_total_samples]
        LOGGER.info("debug 模式：限制总样本数为 %d", len(val_ds.samples))
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)


def empty_bucket():
    return {
        "count": 0,
        "iou_sum": 0.0,
        "pos_iou_sum": 0.0,
        "pos_count": 0,
        "dice_sum": 0.0,
        "pred_count": 0,
        "empty_count": 0,
    }


def update_bucket(bucket, metrics, gt_positive: bool):
    bucket["count"] += 1
    bucket["iou_sum"] += float(metrics["iou"])
    bucket["dice_sum"] += float(metrics["dice"])
    bucket["pred_count"] += int(metrics["pred_positive"])
    bucket["empty_count"] += int(not metrics["pred_positive"])
    if gt_positive:
        bucket["pos_iou_sum"] += float(metrics["iou"])
        bucket["pos_count"] += 1


def finalize_bucket(bucket):
    count = max(bucket["count"], 1)
    pos_count = max(bucket["pos_count"], 1)
    mean_iou = float(bucket["iou_sum"] / count)
    return {
        "iou": mean_iou,
        "pos_iou": float(bucket["pos_iou_sum"] / pos_count) if bucket["pos_count"] > 0 else mean_iou,
        "dice": float(bucket["dice_sum"] / count),
        "prediction_count": int(bucket["pred_count"]),
        "empty_prediction_ratio": float(bucket["empty_count"] / count),
        "count": int(bucket["count"]),
    }


@torch.no_grad()
def run_streaming_sweep(model, processor, loader, cfg, prompt_fusion: str, tta: str, enable_postprocess: bool, max_samples_per_class: int = 0):
    per_class_grid = {
        class_name: {f"{thr:.2f}": empty_bucket() for thr in cfg.THRESHOLD_GRID}
        for class_name in cfg.CLASSES
    }
    class_kept_counts = defaultdict(int)

    pbar = tqdm(loader, desc="Collect+Sweep", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        images = batch["images"].to(next(model.parameters()).device)
        probs = predict_probability_batch(
            model=model,
            processor=processor,
            images=images,
            prompt_keys=batch["class_names"],
            cfg=cfg,
            prompt_fusion=prompt_fusion,
            tta=tta,
        ).cpu().numpy()
        masks = batch["masks"].numpy()

        for idx, class_name in enumerate(batch["class_names"]):
            if max_samples_per_class > 0 and class_kept_counts[class_name] >= max_samples_per_class:
                continue
            class_kept_counts[class_name] += 1
            gt = (masks[idx] > 0.5).astype(np.uint8)
            gt_positive = bool(gt.sum() > 0)
            for threshold in cfg.THRESHOLD_GRID:
                pred_bin = postprocess_probability_map(
                    probs[idx],
                    class_name,
                    threshold,
                    cfg,
                    enable_postprocess=enable_postprocess,
                )
                metrics = compute_binary_metrics(pred_bin, gt)
                update_bucket(per_class_grid[class_name][f"{threshold:.2f}"], metrics, gt_positive)

        if max_samples_per_class > 0:
            if all(class_kept_counts[class_name] >= max_samples_per_class for class_name in cfg.CLASSES):
                LOGGER.info("所有类别已收满 max_samples_per_class=%d，提前结束 streaming sweep。", max_samples_per_class)
                break

    per_class_logs = {}
    best_overall_thresholds = {}
    best_pos_thresholds = {}
    for class_name in cfg.CLASSES:
        sweep_log = {thr_key: finalize_bucket(bucket) for thr_key, bucket in per_class_grid[class_name].items()}
        best_overall = {"threshold": 0.5, "iou": -1.0}
        best_positive = {"threshold": 0.5, "pos_iou": -1.0}
        for thr_key, stats in sweep_log.items():
            threshold = float(thr_key)
            if stats["iou"] > best_overall["iou"]:
                best_overall = {"threshold": threshold, **stats}
            if stats["pos_iou"] > best_positive["pos_iou"]:
                best_positive = {"threshold": threshold, **stats}
        per_class_logs[class_name] = {
            "config_threshold": float(cfg.VAL_THRESHOLDS.get(class_name, cfg.DEFAULT_THRESHOLDS.get(class_name, 0.5))),
            "sweep": sweep_log,
            "best_overall": best_overall,
            "best_positive": best_positive,
        }
        best_overall_thresholds[class_name] = float(best_overall["threshold"])
        best_pos_thresholds[class_name] = float(best_positive["threshold"])

    def aggregate_from_threshold_map(threshold_map):
        per_class = {}
        miou_values = []
        pos_miou_values = []
        for class_name in cfg.CLASSES:
            stats = per_class_logs[class_name]["sweep"][f"{float(threshold_map[class_name]):.2f}"]
            per_class[class_name] = {"threshold": float(threshold_map[class_name]), **stats}
            miou_values.append(stats["iou"])
            pos_miou_values.append(stats["pos_iou"])
        return {
            "mIoU": float(np.mean(miou_values)) if miou_values else 0.0,
            "pos_mIoU": float(np.mean(pos_miou_values)) if pos_miou_values else 0.0,
            "per_class": per_class,
        }

    config_threshold_map = {class_name: float(cfg.VAL_THRESHOLDS.get(class_name, cfg.DEFAULT_THRESHOLDS.get(class_name, 0.5))) for class_name in cfg.CLASSES}
    return {
        "config_threshold_eval": aggregate_from_threshold_map(config_threshold_map),
        "best_overall_miou_thresholds": {
            "thresholds": best_overall_thresholds,
            "metrics": aggregate_from_threshold_map(best_overall_thresholds),
        },
        "best_pos_miou_thresholds": {
            "thresholds": best_pos_thresholds,
            "metrics": aggregate_from_threshold_map(best_pos_thresholds),
        },
        "per_class_logs": per_class_logs,
        "max_samples_per_class": int(max_samples_per_class) if max_samples_per_class > 0 else None,
    }


def main():
    args = parse_args()
    configure_logging()
    cfg = load_config(args.config)
    set_seed(cfg.SEED)

    checkpoint = Path(args.checkpoint) if args.checkpoint else (cfg.TRAIN_OUTPUT_ROOT / cfg.RUN_NAME / "best.pt")
    output_dir = Path(args.output_dir) if args.output_dir else (cfg.TRAIN_OUTPUT_ROOT / cfg.RUN_NAME / cfg.SWEEP_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE)
    workers = int(args.workers if args.workers is not None else cfg.WORKERS)
    max_total_samples, max_samples_per_class = resolve_sample_limits(args)
    prompt_fusion = args.prompt_fusion or cfg.SWEEP_PROMPT_FUSION
    tta = args.tta or cfg.SWEEP_TTA
    enable_postprocess = bool(getattr(cfg, "SWEEP_ENABLE_POSTPROCESS", getattr(cfg, "ENABLE_POSTPROCESS", True)))
    if args.postprocess:
        enable_postprocess = True
    if args.no_postprocess:
        enable_postprocess = False

    device = resolve_device(args.device, default_device=cfg.DEVICE, logger=LOGGER)
    model, processor = load_clipseg_model(cfg, device, logger=LOGGER)
    load_checkpoint_into_model(model, checkpoint, logger=LOGGER, map_location="cpu")
    model.eval()

    LOGGER.info("device: %s", device)
    LOGGER.info("config path: %s", cfg.CONFIG_PATH)
    LOGGER.info("checkpoint path: %s", checkpoint)
    LOGGER.info("output directory: %s", output_dir)
    LOGGER.info("prompt_fusion=%s tta=%s postprocess=%s", prompt_fusion, tta, enable_postprocess)
    LOGGER.info("sweep mode: %s", "debug(max_total_samples)" if max_total_samples > 0 else "streaming(full)")

    loader = build_val_loader(cfg, batch_size, workers, max_total_samples=max_total_samples)
    result = run_streaming_sweep(
        model=model,
        processor=processor,
        loader=loader,
        cfg=cfg,
        prompt_fusion=prompt_fusion,
        tta=tta,
        enable_postprocess=enable_postprocess,
        max_samples_per_class=max_samples_per_class,
    )
    result.update(
        {
            "checkpoint_path": str(checkpoint),
            "config_path": str(cfg.CONFIG_PATH),
            "output_directory": str(output_dir),
            "prompt_fusion": prompt_fusion,
            "tta": tta,
            "postprocess": bool(enable_postprocess),
            "threshold_grid": list(cfg.THRESHOLD_GRID),
            "classes": list(cfg.CLASSES),
            "sweep_mode": "debug" if max_total_samples > 0 else "streaming",
            "max_total_samples": int(max_total_samples) if max_total_samples > 0 else None,
            "max_samples_per_class": int(max_samples_per_class) if max_samples_per_class > 0 else None,
        }
    )

    output_json = output_dir / "threshold_sweep.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    LOGGER.info("config mIoU=%.4f pos_mIoU=%.4f", result["config_threshold_eval"]["mIoU"], result["config_threshold_eval"]["pos_mIoU"])
    LOGGER.info("best overall mIoU=%.4f", result["best_overall_miou_thresholds"]["metrics"]["mIoU"])
    LOGGER.info("best pos mIoU=%.4f", result["best_pos_miou_thresholds"]["metrics"]["pos_mIoU"])
    LOGGER.info("json: %s", output_json)


if __name__ == "__main__":
    main()
