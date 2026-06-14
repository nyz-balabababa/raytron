#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPSegProcessor

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

THIS_DIR = Path(__file__).resolve().parent
V4_DIR = ROOT / "src" / "train" / "clipseg-v4"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

from config_clipseg_tiny_refine import (  # noqa: E402
    AMP,
    APPLY_SCORE_FILTER,
    AUX_WEIGHT,
    BASE_CHECKPOINT,
    BASE_MODEL_DIR,
    BATCH_SIZE,
    BCE_WEIGHT,
    BOUNDARY_IGNORE_MIN_AREA,
    BOUNDARY_IGNORE_WIDTH,
    CLASS_WEIGHTS,
    CLASSES,
    CONF_FILTER,
    DEVICE,
    DICE_WEIGHT,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOCAL_WEIGHT,
    FREEZE_CLIPSEG_DECODER,
    FREEZE_TEXT_ENCODER,
    FREEZE_VISION_BACKBONE,
    IMAGE_ROOT,
    IMG_SIZE,
    INCLUDE_NEGATIVE_SAMPLES,
    LOSS_WEIGHT_FLOOR,
    LR_BACKBONE,
    LR_DECODER,
    LR_REFINE,
    LRF,
    MANIFEST_JSON,
    NEGATIVE_SAMPLE_RATIO,
    NEGATIVE_SAMPLE_WEIGHT,
    NUM_CLASSES,
    OLD5_CLASSES,
    PRED_JSON,
    PROJECT,
    RARE_CLASSES,
    RARE_OVERSAMPLE,
    RESIDUAL_SCALE,
    RUN_NAME,
    SEED,
    TRAIN_LIST,
    TRAIN_REFINE_ONLY,
    USE_EDGE_MAP,
    USE_MANIFEST,
    VAL_INCLUDE_NEGATIVE_SAMPLES,
    VAL_LIST,
    VAL_PRED_JSON,
    VAL_THRESHOLDS,
    WARMUP_EPOCHS,
    WARMUP_START_FACTOR,
    WEIGHT_DECAY,
    WORKERS,
    MODEL_TYPE,
)
from clipseg_train import (  # noqa: E402
    DiceLoss,
    RareClassFocalLoss,
    build_boundary_valid_mask,
    compute_metrics,
    load_teacher_aligned_gray,
    rle_to_mask,
    stable_hash,
)
from models.clipseg_tiny_refine import CLIPSegTinyRefine  # noqa: E402

LOGGER = logging.getLogger("clipseg_v5")
LOG_FILE = PROJECT / f"{RUN_NAME}.log"


def configure_logging(run_name: str):
    global LOG_FILE
    PROJECT.mkdir(parents=True, exist_ok=True)
    log_file = PROJECT / f"{run_name}.log"
    logger = logging.getLogger("clipseg_v5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    LOG_FILE = log_file
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_prompt_text(class_name: str) -> str:
    return class_name


def build_canvas(gray: np.ndarray, mask: np.ndarray | None, img_size: int):
    h, w = gray.shape
    scale = img_size / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    gray_rs = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_rs = None
    if mask is not None:
        mask_rs = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    ph, pw = img_size - nh, img_size - nw
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    gray_pad = cv2.copyMakeBorder(gray_rs, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    mask_pad = None
    if mask_rs is not None:
        mask_pad = cv2.copyMakeBorder(mask_rs, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    return gray_pad, mask_pad


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1).astype(np.uint8, copy=False)


def compute_edge_map(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 32, 96)
    return edges.astype(np.float32) / 255.0


def parse_records(
    pred_json: Path,
    image_list_path: Path,
    image_root: Path,
    classes: list[str],
    include_negatives: bool,
    negative_ratio: float,
    negative_weight: float,
    apply_score_filter: bool,
    use_manifest: bool,
    oversample: dict[str, int] | None,
):
    with open(pred_json, encoding="utf-8") as file_obj:
        raw_data = json.load(file_obj)
    with open(image_list_path, encoding="utf-8") as file_obj:
        allowed = {line.strip().replace("\\", "/") for line in file_obj if line.strip()}

    samples = []
    rng = random.Random(SEED)
    skipped_empty = 0
    skipped_computer = 0
    skipped_missing = 0
    skipped_low_conf = 0
    negative_kept = 0
    cache_dir = PROJECT / ".mask_cache_clipseg_tiny_refine" / pred_json.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    def normalize_img_path(img_path_raw: str):
        img_path = img_path_raw.replace("\\", "/")
        if img_path not in allowed:
            alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
            if alt not in allowed:
                return None
            img_path = alt
        return img_path

    if use_manifest:
        records = raw_data.get("records", []) if isinstance(raw_data, dict) else []
        iterator = tqdm(records, desc=f"  parse {pred_json.name}")
        for record in iterator:
            img_path = normalize_img_path(record["image_path"])
            if img_path is None:
                continue
            abs_img_path = image_root / img_path
            if not abs_img_path.exists():
                skipped_missing += 1
                continue
            prompt = record.get("prompt")
            if prompt == "computer":
                skipped_computer += 1
                continue
            if prompt not in classes:
                continue
            if not record.get("include_in_train") or not record.get("selected_hit"):
                continue
            rle = record.get("rle")
            if rle is None:
                continue
            ann_id = record.get("ann_id")
            sample_weight = float(record.get("sample_weight", 1.0))
            cache_name = stable_hash([img_path, prompt, "manifest"])
            cache_path = cache_dir / f"{cache_name}.png"
            if not cache_path.exists():
                mask = rle_to_mask(rle)
                if mask.sum() == 0:
                    skipped_empty += 1
                    continue
                cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
            samples.append((img_path, prompt, cache_path, sample_weight, {"ann_id": ann_id, "is_positive": True}))
    else:
        iterator = tqdm(raw_data, desc=f"  parse {pred_json.name}")
        for item in iterator:
            img_path = normalize_img_path(item["image_path"])
            if img_path is None:
                continue
            abs_img_path = image_root / img_path
            if not abs_img_path.exists():
                skipped_missing += 1
                continue
            ann_id = item.get("ann_id")
            for prompt, info in item.get("prompts", {}).items():
                if prompt == "computer":
                    skipped_computer += 1
                    continue
                if prompt not in classes:
                    continue
                if not info.get("hit"):
                    if include_negatives and negative_ratio > 0 and rng.random() < negative_ratio:
                        samples.append((img_path, prompt, None, float(negative_weight), {"ann_id": ann_id, "is_positive": False}))
                        negative_kept += 1
                    continue
                rle = info.get("rle")
                if rle is None:
                    continue
                score = float(info.get("score", 1.0))
                threshold = CONF_FILTER.get(prompt, 0.5) if isinstance(CONF_FILTER, dict) else float(CONF_FILTER)
                if apply_score_filter and score < threshold:
                    skipped_low_conf += 1
                    continue
                cache_name = stable_hash([img_path, prompt, "pred"])
                cache_path = cache_dir / f"{cache_name}.png"
                if not cache_path.exists():
                    mask = rle_to_mask(rle)
                    if mask.sum() == 0:
                        skipped_empty += 1
                        continue
                    cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                samples.append((img_path, prompt, cache_path, score, {"ann_id": ann_id, "is_positive": True}))

    if oversample:
        extras = []
        for sample in samples:
            if not sample[4]["is_positive"]:
                continue
            for _ in range(oversample.get(sample[1], 1) - 1):
                extras.append(sample)
        samples.extend(extras)

    LOGGER.info(
        "parsed %s -> %d samples, negative_kept=%d skipped_empty=%d skipped_low_conf=%d skipped_missing=%d skipped_computer=%d",
        pred_json.name,
        len(samples),
        negative_kept,
        skipped_empty,
        skipped_low_conf,
        skipped_missing,
        skipped_computer,
    )
    return samples


class ClipSegTinyRefineDataset(Dataset):
    def __init__(
        self,
        pred_json: Path,
        image_list_path: Path,
        image_root: Path,
        img_size: int,
        classes: list[str],
        use_manifest: bool = False,
        include_negatives: bool = False,
        negative_ratio: float = 0.0,
        negative_weight: float = 0.0,
        apply_score_filter: bool = False,
        oversample: dict[str, int] | None = None,
        use_edge_map: bool = False,
        is_train: bool = True,
    ):
        self.samples = parse_records(
            pred_json=pred_json,
            image_list_path=image_list_path,
            image_root=image_root,
            classes=classes,
            include_negatives=include_negatives,
            negative_ratio=negative_ratio,
            negative_weight=negative_weight,
            apply_score_filter=apply_score_filter,
            use_manifest=use_manifest,
            oversample=oversample,
        )
        self.image_root = image_root
        self.img_size = img_size
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.use_edge_map = use_edge_map
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name, cache_path, sample_weight, meta = self.samples[idx]
        gray = load_teacher_aligned_gray(self.image_root / img_path)
        if meta["is_positive"]:
            mask_img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"mask cache missing: {cache_path}")
            mask = mask_img.astype(np.float32) / 255.0
        else:
            mask = np.zeros_like(gray, dtype=np.float32)

        gray_canvas, mask_canvas = build_canvas(gray, mask, self.img_size)
        valid_mask = (
            build_boundary_valid_mask(mask_canvas, class_name)
            if meta["is_positive"] and BOUNDARY_IGNORE_WIDTH > 0
            else np.ones_like(mask_canvas, dtype=np.float32)
        )

        if self.is_train and random.random() < 0.5:
            gray_canvas = np.fliplr(gray_canvas)
            mask_canvas = np.fliplr(mask_canvas)
            valid_mask = np.fliplr(valid_mask)

        rgb_canvas = gray_to_rgb(gray_canvas)
        gray_tensor = torch.from_numpy((gray_canvas.astype(np.float32) / 255.0).copy()).unsqueeze(0)
        edge_tensor = torch.from_numpy(compute_edge_map(gray_canvas).copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_canvas.astype(np.float32).copy()).unsqueeze(0)
        valid_mask_tensor = torch.from_numpy(valid_mask.astype(np.float32).copy()).unsqueeze(0)

        return {
            "rgb_image": rgb_canvas,
            "gray_image": gray_tensor,
            "edge_map": edge_tensor,
            "mask": mask_tensor,
            "valid_mask": valid_mask_tensor,
            "class_idx": self.class_to_idx[class_name],
            "class_name": class_name,
            "img_path": img_path,
            "ann_id": meta.get("ann_id"),
            "weight": float(sample_weight),
            "is_positive": bool(meta["is_positive"]),
            "prompt_text": resolve_prompt_text(class_name),
        }


def collate_fn(batch):
    return {
        "rgb_images": [item["rgb_image"] for item in batch],
        "gray_images": torch.stack([item["gray_image"] for item in batch], dim=0),
        "edge_maps": torch.stack([item["edge_map"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "valid_masks": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "class_indices": torch.tensor([item["class_idx"] for item in batch], dtype=torch.long),
        "class_names": [item["class_name"] for item in batch],
        "img_paths": [item["img_path"] for item in batch],
        "ann_ids": [item["ann_id"] for item in batch],
        "weights": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
        "is_positives": [item["is_positive"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
    }


def move_inputs_to_device(inputs: dict, device: str):
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device, non_blocking=True)
    return inputs


def build_optimizer(model: CLIPSegTinyRefine, args):
    freeze_text = True if args.freeze_text_encoder or FREEZE_TEXT_ENCODER else True
    freeze_vision = False if args.unfreeze_vision_backbone else (args.freeze_vision_backbone or FREEZE_VISION_BACKBONE)
    freeze_decoder = args.train_refine_only or args.freeze_clipseg_decoder or FREEZE_CLIPSEG_DECODER
    if args.unfreeze_decoder:
        freeze_decoder = False

    for name, param in model.base.named_parameters():
        if "clip.text_model" in name:
            param.requires_grad = not freeze_text
        elif "clip.vision_model" in name:
            param.requires_grad = not freeze_vision
        elif name.startswith("decoder."):
            param.requires_grad = not freeze_decoder
        else:
            param.requires_grad = not args.train_refine_only

    for param in model.refine_head.parameters():
        param.requires_grad = True

    param_groups = []
    refine_params = [p for p in model.refine_head.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.base.named_parameters() if n.startswith("decoder.") and p.requires_grad]
    backbone_params = [p for n, p in model.base.named_parameters() if "clip.vision_model" in n and p.requires_grad]

    if refine_params:
        param_groups.append({"params": refine_params, "lr": args.lr_refine})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr_decoder})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    LOGGER.info(
        "optimizer groups: refine=%d decoder=%d backbone=%d freeze_text=%s freeze_vision=%s freeze_decoder=%s",
        len(refine_params),
        len(decoder_params),
        len(backbone_params),
        freeze_text,
        freeze_vision,
        freeze_decoder,
    )
    return optimizer


def weighted_bce_with_logits(logits, targets, valid_mask):
    loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    valid_area = valid_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (loss_map * valid_mask).sum(dim=(1, 2, 3)) / valid_area


def train_epoch(model, processor, loader, optimizer, scheduler, scaler, dice_loss, focal_loss, args, epoch):
    model.train()
    metrics = defaultdict(float)
    start = time.time()
    pbar = tqdm(loader, desc=f"  Train {epoch}/{args.epochs}")

    for batch in pbar:
        gray_images = batch["gray_images"].to(args.device, non_blocking=True)
        edge_maps = batch["edge_maps"].to(args.device, non_blocking=True)
        masks = batch["masks"].to(args.device, non_blocking=True)
        valid_masks = batch["valid_masks"].to(args.device, non_blocking=True)
        class_indices = batch["class_indices"].to(args.device, non_blocking=True)
        weights = batch["weights"].to(args.device, non_blocking=True)

        for i, is_positive in enumerate(batch["is_positives"]):
            if is_positive:
                weights[i] = max(weights[i].item(), LOSS_WEIGHT_FLOOR)
            else:
                weights[i] = NEGATIVE_SAMPLE_WEIGHT

        proc_inputs = processor(
            text=batch["prompt_texts"],
            images=batch["rgb_images"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        proc_inputs = move_inputs_to_device(proc_inputs, args.device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and args.device.startswith("cuda")):
            outputs = model(
                pixel_values=proc_inputs["pixel_values"],
                input_ids=proc_inputs["input_ids"],
                attention_mask=proc_inputs["attention_mask"],
                gray_image=gray_images,
                edge_map=edge_maps if args.use_edge_map else None,
            )
            coarse_logits = outputs["coarse_logits"]
            refined_logits = outputs["refined_logits"]

            assert coarse_logits.shape == masks.shape, f"coarse shape mismatch {coarse_logits.shape} vs {masks.shape}"
            assert refined_logits.shape == masks.shape, f"refined shape mismatch {refined_logits.shape} vs {masks.shape}"

            bce_per_sample = weighted_bce_with_logits(refined_logits, masks, valid_masks)
            dice_per_sample = dice_loss.per_sample(refined_logits, masks, valid_mask=valid_masks).mean(dim=0, keepdim=False)
            focal_per_sample = focal_loss(refined_logits, masks, class_indices, valid_mask=valid_masks)
            aux_per_sample = weighted_bce_with_logits(coarse_logits, masks, valid_masks)

            if dice_per_sample.ndim == 0:
                dice_per_sample = dice_per_sample.unsqueeze(0).repeat(len(class_indices))
            if focal_per_sample.ndim == 0:
                focal_per_sample = focal_per_sample.unsqueeze(0).repeat(len(class_indices))

            main_loss_per_sample = (
                BCE_WEIGHT * bce_per_sample
                + DICE_WEIGHT * dice_per_sample
                + FOCAL_WEIGHT * focal_per_sample
            )
            aux_loss_per_sample = AUX_WEIGHT * aux_per_sample
            loss = (main_loss_per_sample * weights).mean() + (aux_loss_per_sample * weights).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics["loss"] += loss.item()
        metrics["bce"] += bce_per_sample.mean().item()
        metrics["dice"] += dice_per_sample.mean().item()
        metrics["focal"] += focal_per_sample.mean().item()
        metrics["aux"] += aux_per_sample.mean().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    denom = max(len(loader), 1)
    metrics = {k: v / denom for k, v in metrics.items()}
    metrics["epoch_time"] = time.time() - start
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        metrics["max_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        metrics["max_mem_gb"] = 0.0
    return metrics


@torch.no_grad()
def validate(model, processor, loader, args):
    model.eval()
    all_metrics = defaultdict(list)
    vis_samples = []
    start = time.time()

    for batch in tqdm(loader, desc="  Val"):
        gray_images = batch["gray_images"].to(args.device, non_blocking=True)
        edge_maps = batch["edge_maps"].to(args.device, non_blocking=True)
        masks = batch["masks"].to(args.device, non_blocking=True)
        proc_inputs = processor(
            text=batch["prompt_texts"],
            images=batch["rgb_images"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        proc_inputs = move_inputs_to_device(proc_inputs, args.device)

        with autocast(enabled=args.amp and args.device.startswith("cuda")):
            outputs = model(
                pixel_values=proc_inputs["pixel_values"],
                input_ids=proc_inputs["input_ids"],
                attention_mask=proc_inputs["attention_mask"],
                gray_image=gray_images,
                edge_map=edge_maps if args.use_edge_map else None,
            )

        coarse_logits = outputs["coarse_logits"]
        refined_logits = outputs["refined_logits"]
        for b, class_name in enumerate(batch["class_names"]):
            threshold = VAL_THRESHOLDS.get(class_name, 0.5)
            refined_m = compute_metrics(refined_logits[b:b + 1], masks[b:b + 1], threshold=threshold)
            coarse_m = compute_metrics(coarse_logits[b:b + 1], masks[b:b + 1], threshold=threshold)

            pred_bin = (torch.sigmoid(refined_logits[b:b + 1]) > threshold).float()
            gt = masks[b:b + 1].float()
            all_metrics["pred_area"].append(float(pred_bin.sum().item()))
            all_metrics["gt_area"].append(float(gt.sum().item()))

            for key, value in refined_m.items():
                all_metrics[f"{key}/{class_name}"].append(value)
            all_metrics["iou/overall"].append(refined_m["iou"])
            all_metrics["dice/overall"].append(refined_m["dice"])
            all_metrics["precision/overall"].append(refined_m["precision"])
            all_metrics["recall/overall"].append(refined_m["recall"])
            all_metrics["refine_gain"].append(refined_m["iou"] - coarse_m["iou"])
            if class_name in OLD5_CLASSES:
                all_metrics["iou/old5"].append(refined_m["iou"])
            if class_name in RARE_CLASSES:
                all_metrics["iou/rare"].append(refined_m["iou"])
            if batch["is_positives"][b]:
                all_metrics["iou/positive"].append(refined_m["iou"])

            if len(vis_samples) < 6:
                vis_samples.append({
                    "gray": gray_images[b, 0].detach().cpu().numpy(),
                    "gt": gt[b, 0].detach().cpu().numpy(),
                    "pred": pred_bin[b, 0].detach().cpu().numpy(),
                    "class_name": class_name,
                })

    summary = {key: float(np.mean(values)) for key, values in all_metrics.items() if values}
    summary["val_time"] = time.time() - start
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        summary["max_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        summary["max_mem_gb"] = 0.0
    return summary, vis_samples


def plot_results(history, vis_samples, save_path):
    fig = plt.figure(figsize=(16, 8))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs, history["train_loss"], label="TrainLoss")
    ax1.plot(epochs, history["train_aux"], label="Aux", linestyle="--")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, history["val_miou_all11"], label="all11")
    ax2.plot(epochs, history["val_miou_old5"], label="old5")
    ax2.plot(epochs, history["val_miou_rare"], label="rare")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("mIoU")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs, history["val_dice"], label="Dice", color="green")
    ax3.plot(epochs, history["val_refine_gain"], label="RefineGain", color="orange")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Dice / Gain")

    ax4 = fig.add_subplot(2, 3, 4)
    for class_name in CLASSES:
        key = f"iou/{class_name}"
        if key in history:
            ax4.plot(epochs, history[key], linewidth=1, alpha=0.8, label=class_name)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Per-class IoU")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(epochs, history["lr"], color="red")
    ax5.grid(True, alpha=0.3)
    ax5.set_title("LR")

    ax6 = fig.add_subplot(2, 3, 6)
    if vis_samples:
        sample = vis_samples[0]
        canvas = np.stack([sample["gray"]] * 3, axis=-1)
        canvas[:, :, 0] = np.clip(canvas[:, :, 0] + sample["gt"] * 0.4, 0, 1)
        canvas[:, :, 2] = np.clip(canvas[:, :, 2] + sample["pred"] * 0.5, 0, 1)
        ax6.imshow(canvas)
        ax6.set_title(sample["class_name"], fontsize=8)
        ax6.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("resume checkpoint=%s", checkpoint_path)
    LOGGER.info("resume missing_keys=%s", missing_keys[:20])
    LOGGER.info("resume unexpected_keys=%s", unexpected_keys[:20])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        checkpoint.get("best_metrics", {"best_all11": -1, "best_old5": -1, "best_rare": -1, "best_refine_gain": -1}),
        defaultdict(list, checkpoint.get("history", {})),
    )


def build_args():
    parser = argparse.ArgumentParser(description="Train CLIPSeg + Tiny Refine Head")
    parser.add_argument("--base_model_dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--base_checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--train_json", type=Path, default=PRED_JSON)
    parser.add_argument("--val_json", type=Path, default=VAL_PRED_JSON)
    parser.add_argument("--manifest", action="store_true", default=USE_MANIFEST)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--train_list", type=Path, default=TRAIN_LIST)
    parser.add_argument("--val_list", type=Path, default=VAL_LIST)
    parser.add_argument("--output_dir", type=Path, default=PROJECT)
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--lr_refine", type=float, default=LR_REFINE)
    parser.add_argument("--lr_decoder", type=float, default=LR_DECODER)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--amp", action="store_true", default=AMP)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume_weights_only", action="store_true", default=False)
    parser.add_argument("--reset_history", action="store_true", default=False)
    parser.add_argument("--no_auto_resume", action="store_true", default=False)
    parser.add_argument("--freeze_text_encoder", action="store_true", default=FREEZE_TEXT_ENCODER)
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=FREEZE_VISION_BACKBONE)
    parser.add_argument("--freeze_clipseg_decoder", action="store_true", default=FREEZE_CLIPSEG_DECODER)
    parser.add_argument("--train_refine_only", action="store_true", default=TRAIN_REFINE_ONLY)
    parser.add_argument("--unfreeze_decoder", action="store_true", default=False)
    parser.add_argument("--unfreeze_vision_backbone", action="store_true", default=False)
    parser.add_argument("--use_edge_map", action="store_true", default=USE_EDGE_MAP)
    parser.add_argument("--boundary_ignore", type=int, default=BOUNDARY_IGNORE_WIDTH)
    parser.add_argument("--negative_sample_ratio", type=float, default=NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--rare_oversample", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


def main():
    args = build_args()
    args.output_dir = args.output_dir.resolve()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = configure_logging(args.run_name)
    set_seed(SEED)

    processor = CLIPSegProcessor.from_pretrained(str(args.base_model_dir))
    model = CLIPSegTinyRefine(
        base_model_dir=args.base_model_dir,
        base_checkpoint=args.base_checkpoint,
        img_size=args.img_size,
        residual_scale=RESIDUAL_SCALE,
        use_edge_map=args.use_edge_map,
    ).to(args.device)
    optimizer = build_optimizer(model, args)

    train_ds = ClipSegTinyRefineDataset(
        pred_json=args.train_json if not args.manifest else MANIFEST_JSON,
        image_list_path=args.train_list,
        image_root=args.image_root,
        img_size=args.img_size,
        classes=CLASSES,
        use_manifest=args.manifest,
        include_negatives=not args.manifest and INCLUDE_NEGATIVE_SAMPLES,
        negative_ratio=args.negative_sample_ratio,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=APPLY_SCORE_FILTER,
        oversample=RARE_OVERSAMPLE if args.rare_oversample else None,
        use_edge_map=args.use_edge_map,
        is_train=True,
    )
    val_ds = ClipSegTinyRefineDataset(
        pred_json=args.val_json,
        image_list_path=args.val_list,
        image_root=args.image_root,
        img_size=args.img_size,
        classes=CLASSES,
        use_manifest=False,
        include_negatives=VAL_INCLUDE_NEGATIVE_SAMPLES,
        negative_ratio=0.0,
        negative_weight=NEGATIVE_SAMPLE_WEIGHT,
        apply_score_filter=APPLY_SCORE_FILTER,
        oversample=None,
        use_edge_map=args.use_edge_map,
        is_train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    warmup_steps = max(1, WARMUP_EPOCHS * len(train_loader))
    total_steps = max(1, args.epochs * len(train_loader))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return WARMUP_START_FACTOR + (1 - WARMUP_START_FACTOR) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LRF + (1 - LRF) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=args.amp and args.device.startswith("cuda"))
    dice_loss = DiceLoss()
    focal_loss = RareClassFocalLoss(class_weights=CLASS_WEIGHTS, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA).to(args.device)

    start_epoch = 0
    history = defaultdict(list)
    best_metrics = {"best_all11": -1, "best_old5": -1, "best_rare": -1, "best_refine_gain": -1}

    resume_path = None
    if args.resume is not None:
        resume_path = args.resume
    elif not args.no_auto_resume:
        candidate = run_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path and resume_path.exists():
        start_epoch, best_metrics, history = load_checkpoint(
            resume_path,
            model,
            None if args.resume_weights_only else optimizer,
            None if args.resume_weights_only else scheduler,
        )
        if args.reset_history:
            start_epoch = 0
            history = defaultdict(list)
            best_metrics = {"best_all11": -1, "best_old5": -1, "best_rare": -1, "best_refine_gain": -1}
        elif args.resume_weights_only:
            start_epoch = 0
    elif args.resume is not None:
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")

    config_used = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    with open(run_dir / "config_used.json", "w", encoding="utf-8") as file_obj:
        json.dump(config_used, file_obj, ensure_ascii=False, indent=2)

    LOGGER.info("train samples=%d val samples=%d", len(train_ds), len(val_ds))
    LOGGER.info("base_model_dir=%s", args.base_model_dir)
    LOGGER.info("base_checkpoint=%s", args.base_checkpoint)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        LOGGER.info("%s Epoch %d/%d", "-" * 60, epoch, args.epochs)
        train_metrics = train_epoch(model, processor, train_loader, optimizer, scheduler, scaler, dice_loss, focal_loss, args, epoch)
        val_metrics, vis_samples = validate(model, processor, val_loader, args)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_bce"].append(train_metrics["bce"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_focal"].append(train_metrics["focal"])
        history["train_aux"].append(train_metrics["aux"])
        history["val_miou_all11"].append(val_metrics.get("iou/overall", 0.0))
        history["val_miou_old5"].append(val_metrics.get("iou/old5", 0.0))
        history["val_miou_rare"].append(val_metrics.get("iou/rare", 0.0))
        history["val_pos_miou"].append(val_metrics.get("iou/positive", 0.0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
        history["val_refine_gain"].append(val_metrics.get("refine_gain", 0.0))
        history["lr"].append(current_lr)
        for class_name in CLASSES:
            history[f"iou/{class_name}"].append(val_metrics.get(f"iou/{class_name}", 0.0))

        LOGGER.info(
            "TrainLoss=%.4f BCE=%.4f Dice=%.4f Focal=%.4f Aux=%.4f "
            "mIoU_all11=%.4f mIoU_old5=%.4f mIoU_rare=%.4f Dice=%.4f pos_mIoU=%.4f "
            "LR=%.2e epoch_time=%.1fs val_time=%.1fs max_mem=%.2fGB",
            train_metrics["loss"],
            train_metrics["bce"],
            train_metrics["dice"],
            train_metrics["focal"],
            train_metrics["aux"],
            val_metrics.get("iou/overall", 0.0),
            val_metrics.get("iou/old5", 0.0),
            val_metrics.get("iou/rare", 0.0),
            val_metrics.get("dice/overall", 0.0),
            val_metrics.get("iou/positive", 0.0),
            current_lr,
            train_metrics["epoch_time"],
            val_metrics.get("val_time", 0.0),
            max(train_metrics.get("max_mem_gb", 0.0), val_metrics.get("max_mem_gb", 0.0)),
        )
        per_class_iou = "  ".join(f"{name}={val_metrics.get(f'iou/{name}', 0.0):.3f}" for name in CLASSES)
        LOGGER.info("per-class IoU: %s", per_class_iou)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metrics": best_metrics,
            "history": dict(history),
            "classes": CLASSES,
            "num_classes": NUM_CLASSES,
            "img_size": args.img_size,
            "model_type": MODEL_TYPE,
            "base_model_dir": str(args.base_model_dir),
            "base_checkpoint": str(args.base_checkpoint),
            "val_thresholds": VAL_THRESHOLDS,
            "run_name": args.run_name,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        improved = []
        if val_metrics.get("iou/overall", -1) > best_metrics["best_all11"]:
            best_metrics["best_all11"] = val_metrics["iou/overall"]
            torch.save(checkpoint, run_dir / "best_all11.pt")
            improved.append("best_all11")
        if val_metrics.get("iou/old5", -1) > best_metrics["best_old5"]:
            best_metrics["best_old5"] = val_metrics["iou/old5"]
            torch.save(checkpoint, run_dir / "best_old5.pt")
            improved.append("best_old5")
        if val_metrics.get("iou/rare", -1) > best_metrics["best_rare"]:
            best_metrics["best_rare"] = val_metrics["iou/rare"]
            torch.save(checkpoint, run_dir / "best_rare.pt")
            improved.append("best_rare")
        if val_metrics.get("refine_gain", -1) > best_metrics["best_refine_gain"]:
            best_metrics["best_refine_gain"] = val_metrics["refine_gain"]
            torch.save(checkpoint, run_dir / "best_refine_gain.pt")
            improved.append("best_refine_gain")
        if improved:
            LOGGER.info("saved improved checkpoints: %s", ", ".join(improved))

        plot_results(history, vis_samples, run_dir / "results.png")

    csv_keys = [
        "train_loss", "train_bce", "train_dice", "train_focal", "train_aux",
        "val_miou_all11", "val_miou_old5", "val_miou_rare", "val_pos_miou", "val_dice",
        "val_refine_gain", "lr",
    ] + [f"iou/{name}" for name in CLASSES]
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch"] + csv_keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in csv_keys])

    LOGGER.info("finished run=%s log=%s", args.run_name, LOG_FILE)


if __name__ == "__main__":
    main()
