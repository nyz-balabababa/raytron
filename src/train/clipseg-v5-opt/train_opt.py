#!/usr/bin/env python3
"""
CLIPSeg decoder-only 训练入口

默认支持：
1. 直接运行，无需命令行参数
2. 可通过 --config 切换配置文件
3. 可通过 --resume 指定旧权重或 baseline checkpoint
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_opt import (
    BinaryFocalLoss,
    ClipSegDataset,
    DiceLoss,
    collate_fn,
    compute_binary_metrics,
    freeze_for_decoder_only,
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
DEFAULT_RUN_NAME = "clipseg_11_opt_decoder_only"
DEFAULT_RESUME = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_EPOCHS = None
DEFAULT_BATCH_SIZE = None
DEFAULT_WORKERS = None
DEFAULT_DEVICE = None
DEFAULT_NEGATIVE_SAMPLE_RATIO = None
DEFAULT_ENABLE_RARE_OVERSAMPLE = True
DEFAULT_RESUME_WEIGHTS_ONLY = False
DEFAULT_AUTO_RESUME = True
DEFAULT_RESET_HISTORY = False


LOGGER = logging.getLogger("clipseg_v5_train_opt")


def configure_logging(log_file: Path):
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def build_args():
    parser = argparse.ArgumentParser(description="Train CLIPSeg decoder-only opt line")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--negative_sample_ratio", type=float, default=DEFAULT_NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--no_rare_oversample", action="store_true", default=not DEFAULT_ENABLE_RARE_OVERSAMPLE)
    parser.add_argument("--resume_weights_only", action="store_true", default=DEFAULT_RESUME_WEIGHTS_ONLY)
    parser.add_argument("--no_auto_resume", action="store_true", default=not DEFAULT_AUTO_RESUME)
    parser.add_argument("--reset_history", action="store_true", default=DEFAULT_RESET_HISTORY)
    return parser.parse_args()


def build_scheduler(optimizer, train_steps_per_epoch: int, cfg, total_epochs: int):
    warmup_steps = cfg.WARMUP_EPOCHS * train_steps_per_epoch
    total_steps = total_epochs * train_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return cfg.WARMUP_START_FACTOR + (1 - cfg.WARMUP_START_FACTOR) * (step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return cfg.LRF + (1 - cfg.LRF) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, cfg):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("没有可训练参数，请检查 decoder-only 冻结策略")
    return torch.optim.AdamW(params, lr=cfg.DECODER_LR, weight_decay=cfg.WEIGHT_DECAY)


def _zero_tensor(device):
    return torch.zeros((), device=device, dtype=torch.float32)


def train_epoch(model, processor, loader, optimizer, scheduler, dice_loss, focal_loss, cfg, epoch: int, total_epochs: int):
    model.train()
    if cfg.FREEZE_TEXT_ENCODER:
        model.clip.text_model.eval()
    if cfg.FREEZE_IMAGE_ENCODER:
        model.clip.vision_model.eval()
    metrics = defaultdict(float)
    pbar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        images = batch["images"].to(next(model.parameters()).device)
        masks = batch["masks"].to(images.device)
        valid_masks = batch["valid_masks"].to(images.device)
        scores = batch["scores"].to(images.device)
        is_positive = batch["is_positive"].to(images.device)

        tokenized = processor.tokenizer(batch["prompt_texts"], return_tensors="pt", padding=True, truncation=True)
        logits = model(
            pixel_values=images,
            input_ids=tokenized["input_ids"].to(images.device),
            attention_mask=tokenized["attention_mask"].to(images.device),
        ).logits
        if logits.ndim == 4:
            logits = logits[:, 0]
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits.unsqueeze(1), size=masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

        pos_idx = torch.where(is_positive)[0]
        neg_idx = torch.where(~is_positive)[0]
        pos_weights = scores[pos_idx].clamp(min=cfg.LOSS_WEIGHT_FLOOR) if pos_idx.numel() else None

        pos_bce = _zero_tensor(images.device)
        pos_dice = _zero_tensor(images.device)
        pos_focal = _zero_tensor(images.device)
        neg_loss = _zero_tensor(images.device)

        if pos_idx.numel():
            pos_logits = logits.index_select(0, pos_idx)
            pos_masks = masks.index_select(0, pos_idx)
            pos_valid = valid_masks.index_select(0, pos_idx)
            bce_map = F.binary_cross_entropy_with_logits(pos_logits, pos_masks, reduction="none")
            valid_area = pos_valid.sum(dim=(1, 2)).clamp(min=1.0)
            bce_per_sample = (bce_map * pos_valid).sum(dim=(1, 2)) / valid_area
            dice_per_sample = dice_loss.per_sample(pos_logits, pos_masks, valid_mask=pos_valid)
            focal_per_sample = focal_loss.per_sample(pos_logits, pos_masks, valid_mask=pos_valid)
            pos_bce = (bce_per_sample * pos_weights).mean()
            pos_dice = (dice_per_sample * pos_weights).mean()
            pos_focal = (focal_per_sample * pos_weights).mean()

        if neg_idx.numel():
            neg_logits = logits.index_select(0, neg_idx)
            neg_masks = masks.index_select(0, neg_idx)
            bce_map = F.binary_cross_entropy_with_logits(neg_logits, neg_masks, reduction="none")
            neg_loss = cfg.NEGATIVE_SAMPLE_WEIGHT * bce_map.mean()

        total_loss = cfg.BCE_WEIGHT * pos_bce + cfg.DICE_WEIGHT * pos_dice + cfg.FOCAL_WEIGHT * pos_focal + neg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        metrics["total"] += float(total_loss.item())
        metrics["bce"] += float(pos_bce.item())
        metrics["dice"] += float(pos_dice.item())
        metrics["focal"] += float(pos_focal.item())
        metrics["neg"] += float(neg_loss.item())

        pbar.set_postfix(
            total=f"{total_loss.item():.4f}",
            bce=f"{pos_bce.item():.4f}",
            dice=f"{pos_dice.item():.4f}",
            focal=f"{pos_focal.item():.4f}",
            neg=f"{neg_loss.item():.4f}",
        )

    denom = max(len(loader), 1)
    return {k: v / denom for k, v in metrics.items()}


@torch.no_grad()
def validate(model, processor, loader, cfg):
    model.eval()
    enable_postprocess = bool(getattr(cfg, "VAL_ENABLE_POSTPROCESS", getattr(cfg, "ENABLE_POSTPROCESS", True)))
    per_class_iou = defaultdict(list)
    per_class_pos_iou = defaultdict(list)
    overall_dice = []
    pred_counts = defaultdict(int)
    empty_counts = defaultdict(int)
    vis_samples = []

    pbar = tqdm(loader, desc="  Val", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for batch in pbar:
        images = batch["images"].to(next(model.parameters()).device)
        masks = batch["masks"].cpu().numpy()
        probs = predict_probability_batch(
            model=model,
            processor=processor,
            images=images,
            prompt_keys=batch["class_names"],
            cfg=cfg,
            prompt_fusion=cfg.VAL_PROMPT_FUSION,
            tta=cfg.VAL_TTA,
        ).cpu().numpy()

        for idx, class_name in enumerate(batch["class_names"]):
            threshold = float(cfg.VAL_THRESHOLDS.get(class_name, 0.5))
            pred_bin = postprocess_probability_map(probs[idx], class_name, threshold, cfg, enable_postprocess=enable_postprocess)
            gt_bin = (masks[idx] > 0.5).astype(np.uint8)
            result = compute_binary_metrics(pred_bin, gt_bin)
            per_class_iou[class_name].append(result["iou"])
            if result["gt_positive"]:
                per_class_pos_iou[class_name].append(result["iou"])
            overall_dice.append(result["dice"])
            pred_counts[class_name] += int(result["pred_positive"])
            empty_counts[class_name] += int(not result["pred_positive"])
            if len(vis_samples) < 8:
                vis_samples.append(
                    {
                        "gray": images[idx].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0],
                        "gt": gt_bin.astype(np.float32),
                        "pred": pred_bin.astype(np.float32),
                        "class_name": class_name,
                    }
                )

    summary = {}
    miou_values = []
    pos_miou_values = []
    for class_name in cfg.CLASSES:
        class_iou = float(np.mean(per_class_iou[class_name])) if per_class_iou[class_name] else 0.0
        pos_iou = float(np.mean(per_class_pos_iou[class_name])) if per_class_pos_iou[class_name] else class_iou
        summary[f"iou/{class_name}"] = class_iou
        summary[f"pred_count/{class_name}"] = int(pred_counts[class_name])
        summary[f"empty_ratio/{class_name}"] = float(empty_counts[class_name] / max(len(per_class_iou[class_name]), 1))
        miou_values.append(class_iou)
        pos_miou_values.append(pos_iou)
    summary["mIoU"] = float(np.mean(miou_values)) if miou_values else 0.0
    summary["pos_mIoU"] = float(np.mean(pos_miou_values)) if pos_miou_values else 0.0
    summary["Dice"] = float(np.mean(overall_dice)) if overall_dice else 0.0
    return summary, vis_samples


def plot_results(history, save_path: Path, vis_samples):
    fig = plt.figure(figsize=(16, 10))
    epochs = range(1, len(history["train_total"]) + 1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs, history["train_total"], label="TrainLoss")
    ax1.plot(epochs, history["train_neg"], label="NegLoss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Loss")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, history["train_bce"], label="BCE")
    ax2.plot(epochs, history["train_dice"], label="DiceLoss")
    ax2.plot(epochs, history["train_focal"], label="FocalLoss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("Train Parts")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs, history["val_miou"], label="mIoU")
    ax3.plot(epochs, history["val_pos_miou"], label="pos_mIoU")
    ax3.plot(epochs, history["val_dice"], label="Dice")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_title("Val Metrics")

    ax4 = fig.add_subplot(2, 3, 4)
    for class_name, values in history["per_class_iou"].items():
        ax4.plot(epochs, values, linewidth=1, alpha=0.8, label=class_name)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=6)
    ax4.set_title("Per-class IoU")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(epochs, history["lr"], color="red")
    ax5.grid(True, alpha=0.3)
    ax5.set_title("LR")

    ax6 = fig.add_subplot(2, 3, 6)
    if vis_samples:
        sample = vis_samples[0]
        overlay = np.stack([sample["gray"]] * 3, axis=-1)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + sample["gt"] * 0.4, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + sample["pred"] * 0.5, 0, 1)
        ax6.imshow(overlay)
        ax6.set_title(sample["class_name"], fontsize=8)
        ax6.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def save_history_csv(history, csv_path: Path, cfg):
    keys = ["train_total", "train_bce", "train_dice", "train_focal", "train_neg", "val_miou", "val_pos_miou", "val_dice", "lr"]
    keys.extend([f"iou/{class_name}" for class_name in cfg.CLASSES])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for idx in range(len(history["train_total"])):
            row = [idx + 1]
            for key in keys:
                if key.startswith("iou/"):
                    class_name = key.split("/", 1)[1]
                    row.append(history["per_class_iou"][class_name][idx])
                else:
                    row.append(history[key][idx])
            writer.writerow(row)


def maybe_resume(run_dir: Path, args, model, optimizer, scheduler, cfg):
    history = {
        "train_total": [],
        "train_bce": [],
        "train_dice": [],
        "train_focal": [],
        "train_neg": [],
        "val_miou": [],
        "val_pos_miou": [],
        "val_dice": [],
        "lr": [],
        "per_class_iou": {class_name: [] for class_name in cfg.CLASSES},
    }
    best_miou = -1.0
    start_epoch = 0

    resume_path = None
    resumed_full_state = False
    if args.resume:
        resume_path = Path(args.resume)
    elif not args.no_auto_resume:
        last_pt = run_dir / "last.pt"
        if last_pt.exists():
            resume_path = last_pt

    if resume_path is None:
        return start_epoch, best_miou, history, None, resumed_full_state

    checkpoint, _, _, _ = load_checkpoint_into_model(model, resume_path, logger=LOGGER, map_location="cpu")
    auto_resumed_from_last = (not args.resume) and resume_path.name.lower() == "last.pt"
    manual_resume = bool(args.resume)

    if manual_resume:
        if not args.resume_weights_only:
            LOGGER.info("手动 --resume 默认按 weights-only 处理；如需恢复优化器/历史，请改用 run_dir/last.pt 自动续训。")
        return 0, -1.0, history, resume_path, resumed_full_state

    if args.resume_weights_only:
        LOGGER.info("resume_weights_only=True，仅恢复模型权重")
        return 0, -1.0, history, resume_path, resumed_full_state

    if not auto_resumed_from_last:
        LOGGER.info("当前仅允许 run_dir/last.pt 自动续训恢复 optimizer/scheduler/history；其余 resume 请使用 --resume_weights_only。")
        return 0, -1.0, history, resume_path, resumed_full_state

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_miou = float(checkpoint.get("best_miou", -1.0))
    if not args.reset_history:
        saved_history = checkpoint.get("history", {})
        for key in ["train_total", "train_bce", "train_dice", "train_focal", "train_neg", "val_miou", "val_pos_miou", "val_dice", "lr"]:
            history[key] = list(saved_history.get(key, []))
        saved_per_class = saved_history.get("per_class_iou", {})
        for class_name in cfg.CLASSES:
            history["per_class_iou"][class_name] = list(saved_per_class.get(class_name, []))
    resumed_full_state = True
    return start_epoch, best_miou, history, resume_path, resumed_full_state


def maybe_subsample_val_dataset_by_image(dataset, cfg, logger):
    max_images = int(getattr(cfg, "VAL_SAMPLE_IMAGE_LIMIT", 0) or 0)
    image_to_samples = defaultdict(list)
    for sample in dataset.samples:
        image_to_samples[sample[0]].append(sample)
    all_images = sorted(image_to_samples.keys())
    total_images = len(all_images)

    if max_images <= 0 or total_images <= max_images:
        logger.info("validation images: using full set %d / %d", total_images, total_images)
        return {"total_images": total_images, "used_images": total_images, "used_samples": len(dataset.samples)}

    rng = random.Random(int(getattr(cfg, "VAL_SAMPLE_SEED", cfg.SEED)))
    selected_images = set(rng.sample(all_images, max_images))
    dataset.samples = [sample for sample in dataset.samples if sample[0] in selected_images]
    logger.info(
        "validation images: sampled %d / %d images, kept %d samples",
        max_images,
        total_images,
        len(dataset.samples),
    )
    return {"total_images": total_images, "used_images": max_images, "used_samples": len(dataset.samples)}


def main():
    args = build_args()
    cfg = load_config(args.config)
    run_name = args.run_name or cfg.RUN_NAME
    output_root = Path(args.output_dir) if args.output_dir else cfg.TRAIN_OUTPUT_ROOT
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    configure_logging(log_file)

    set_seed(cfg.SEED)
    device = resolve_device(args.device, default_device=cfg.DEVICE, logger=LOGGER)
    total_epochs = int(args.epochs if args.epochs is not None else cfg.EPOCHS)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE)
    workers = int(args.workers if args.workers is not None else cfg.WORKERS)
    negative_sample_ratio = float(args.negative_sample_ratio if args.negative_sample_ratio is not None else cfg.NEGATIVE_SAMPLE_RATIO)
    rare_oversample_enabled = not args.no_rare_oversample

    model, processor = load_clipseg_model(cfg, device, logger=LOGGER)
    freeze_for_decoder_only(model, cfg, logger=LOGGER)
    optimizer = build_optimizer(model, cfg)

    train_ds = ClipSegDataset(
        cfg=cfg,
        pred_json=cfg.MANIFEST_JSON if cfg.USE_MANIFEST else cfg.PRED_JSON,
        image_list_path=cfg.TRAIN_LIST,
        is_train=True,
        include_negatives=(not cfg.USE_MANIFEST and cfg.INCLUDE_NEGATIVE_SAMPLES),
        negative_ratio=negative_sample_ratio,
        negative_weight=cfg.NEGATIVE_SAMPLE_WEIGHT,
        oversample=cfg.RARE_OVERSAMPLE if rare_oversample_enabled else None,
        apply_score_filter=cfg.APPLY_SCORE_FILTER,
        logger=LOGGER,
    )
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
    val_sampling_info = maybe_subsample_val_dataset_by_image(val_ds, cfg, LOGGER)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)

    scheduler = build_scheduler(optimizer, len(train_loader), cfg, total_epochs)
    dice_loss = DiceLoss()
    focal_loss = BinaryFocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)

    start_epoch, best_miou, history, resume_path, resumed_full_state = maybe_resume(run_dir, args, model, optimizer, scheduler, cfg)
    if resume_path is None and cfg.DEFAULT_RESUME and Path(cfg.DEFAULT_RESUME).exists():
        LOGGER.info("默认 warm start checkpoint: %s", cfg.DEFAULT_RESUME)
        load_checkpoint_into_model(model, Path(cfg.DEFAULT_RESUME), logger=LOGGER, map_location="cpu")

    LOGGER.info("=" * 60)
    LOGGER.info("CLIPSeg decoder-only opt training")
    LOGGER.info("device: %s", device)
    LOGGER.info("config path: %s", cfg.CONFIG_PATH)
    LOGGER.info("checkpoint path: %s", resume_path if resume_path else args.resume if args.resume else cfg.DEFAULT_RESUME)
    LOGGER.info("resume full state: %s", resumed_full_state)
    LOGGER.info("output directory: %s", run_dir)
    LOGGER.info("train json: %s", cfg.PRED_JSON)
    LOGGER.info("val json: %s", cfg.VAL_PRED_JSON)
    LOGGER.info("train list: %s", cfg.TRAIN_LIST)
    LOGGER.info("val list: %s", cfg.VAL_LIST)
    LOGGER.info(
        "validation sampling: %d / %d images, %d samples",
        val_sampling_info["used_images"],
        val_sampling_info["total_images"],
        val_sampling_info["used_samples"],
    )
    LOGGER.info("prompt pool enabled: %s", cfg.PROMPT_POOL_ENABLED)
    LOGGER.info("negative sample ratio: %.4f", negative_sample_ratio)
    LOGGER.info("negative sample weight: %.4f", cfg.NEGATIVE_SAMPLE_WEIGHT)
    LOGGER.info("rare oversample enabled: %s", rare_oversample_enabled)
    LOGGER.info("freeze text encoder: %s", cfg.FREEZE_TEXT_ENCODER)
    LOGGER.info("freeze image encoder: %s", cfg.FREEZE_IMAGE_ENCODER)
    LOGGER.info("train decoder only: %s", cfg.TRAIN_DECODER_ONLY)
    LOGGER.info("=" * 60)

    epochs_no_improve = 0
    best_pt = run_dir / "best.pt"
    last_pt = run_dir / "last.pt"
    results_png = run_dir / "results.png"

    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_start = time.time()
        LOGGER.info("%s\nEpoch %d/%d", "─" * 60, epoch, total_epochs)

        train_stats = train_epoch(model, processor, train_loader, optimizer, scheduler, dice_loss, focal_loss, cfg, epoch, total_epochs)
        val_metrics, vis_samples = validate(model, processor, val_loader, cfg)
        lr = optimizer.param_groups[0]["lr"]
        elapsed_min = (time.time() - epoch_start) / 60.0

        history["train_total"].append(train_stats["total"])
        history["train_bce"].append(train_stats["bce"])
        history["train_dice"].append(train_stats["dice"])
        history["train_focal"].append(train_stats["focal"])
        history["train_neg"].append(train_stats["neg"])
        history["val_miou"].append(val_metrics["mIoU"])
        history["val_pos_miou"].append(val_metrics["pos_mIoU"])
        history["val_dice"].append(val_metrics["Dice"])
        history["lr"].append(lr)
        for class_name in cfg.CLASSES:
            history["per_class_iou"][class_name].append(val_metrics.get(f"iou/{class_name}", 0.0))

        LOGGER.info(
            "TrainLoss=%.4f  BCE=%.4f  DiceLoss=%.4f  FocalLoss=%.4f  NegLoss=%.4f  mIoU=%.4f  pos_mIoU=%.4f  Dice=%.4f  LR=%.2e  %.1fmin",
            train_stats["total"],
            train_stats["bce"],
            train_stats["dice"],
            train_stats["focal"],
            train_stats["neg"],
            val_metrics["mIoU"],
            val_metrics["pos_mIoU"],
            val_metrics["Dice"],
            lr,
            elapsed_min,
        )
        LOGGER.info("per-class IoU: %s", "  ".join(f"{name}={val_metrics.get(f'iou/{name}', 0.0):.3f}" for name in cfg.CLASSES))
        LOGGER.info("pred_count: %s", "  ".join(f"{name}={val_metrics.get(f'pred_count/{name}', 0)}" for name in cfg.CLASSES))
        LOGGER.info("empty_ratio: %s", "  ".join(f"{name}={val_metrics.get(f'empty_ratio/{name}', 0.0):.3f}" for name in cfg.CLASSES))

        is_best = val_metrics["mIoU"] > best_miou
        if is_best:
            best_miou = val_metrics["mIoU"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "history": history,
            "config_path": str(cfg.CONFIG_PATH),
            "output_dir": str(run_dir),
            "run_name": run_name,
            "val_thresholds": dict(cfg.VAL_THRESHOLDS),
            "prompt_pool": dict(cfg.PROMPT_POOL),
        }
        torch.save(checkpoint, last_pt)

        if is_best:
            torch.save(checkpoint, best_pt)
            LOGGER.info("★ 新最佳模型: mIoU=%.4f", best_miou)
        else:
            LOGGER.info("mIoU 未提升 (%d/%d)", epochs_no_improve, cfg.PATIENCE)
            if epochs_no_improve >= cfg.PATIENCE:
                LOGGER.info("早停触发，best mIoU=%.4f", best_miou)
                break

        plot_results(history, results_png, vis_samples)

    save_history_csv(history, run_dir / "results.csv", cfg)
    LOGGER.info("训练完成: best.pt=%s", best_pt)
    LOGGER.info("日志文件: %s", log_file)


if __name__ == "__main__":
    main()
