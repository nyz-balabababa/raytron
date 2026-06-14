#!/usr/bin/env python3
import argparse
import csv
import hashlib
import inspect
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = str(Path(__file__).resolve().parent)
ESAM_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ESAM_ROOT.parents[1]
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
insert_index = 1
for candidate in [str(ESAM_ROOT), str(PROJECT_ROOT)]:
    if candidate in sys.path:
        sys.path.remove(candidate)
    sys.path.insert(insert_index, candidate)
    insert_index += 1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from config_esam_cclip_11 import (
    ALL_JSON,
    AMP,
    BATCH_SIZE,
    BCE_WEIGHT,
    CLASS_TO_IDX,
    CLASS_WEIGHTS,
    CLASSES,
    CONF_FILTER,
    DECODER_LR,
    DEVICE,
    DICE_WEIGHT,
    EFFICIENT_SAM_CKPT,
    EPOCHS,
    ESAM_INPUT_SIZE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOCAL_WEIGHT,
    FREEZE_IMAGE_ENCODER,
    FREEZE_TEXT_ENCODER,
    GRAD_CLIP,
    HFLIP_PROB,
    IMAGE_CACHE_DIR,
    IMAGE_LR,
    IMAGE_ROOT,
    IMG_SIZE,
    INCLUDE_NEGATIVE_SAMPLES,
    LOSS_WEIGHT_FLOOR,
    MIN_LR_RATIO,
    MODEL_TYPE,
    NEGATIVE_SAMPLE_RATIO,
    NEGATIVE_SAMPLE_WEIGHT,
    OLD5_CLASSES,
    OLD_CLASS_SAMPLE_RATIO,
    OUTPUT_ROOT,
    POSTPROCESS_DEFAULT,
    PROMPT_PROTOTYPES,
    RARE_CLASSES,
    RARE_BALANCED_OLD_CLASSES,
    RARE_BALANCED_RARE_CLASSES,
    RARE_CLASS_KEEP_RATIO,
    RARE_OVERSAMPLE,
    REBUILD_IMAGE_CACHE,
    REBUILD_TEXT_CACHE,
    RUN_NAME,
    SEED,
    TEXT_CACHE_PATH,
    TOKENIZER_DIR,
    TRAIN_DECODER_ONLY,
    TRAIN_JSON,
    TRAIN_LIST,
    USE_IMAGE_CACHE,
    USE_PROMPT_PROTOTYPE,
    VAL_INCLUDE_NEGATIVE_SAMPLES,
    VAL_JSON,
    VAL_LIST,
    VAL_THRESHOLDS,
    WEIGHT_DECAY,
    WORKERS,
    WARMUP_EPOCHS,
)

from common_esam_cclip_11 import (
    DiceLoss,
    ESAMCCLIPModel,
    FocalLoss,
    count_parameters,
    compute_metrics,
    load_tokenizer,
    maybe_tqdm,
    resolve_device,
    save_json,
)
from dataset_esam_cclip_11 import ESAMCCLIP11Dataset
from model_esam_cclip_11 import load_checkpoint_flexible
from path_utils import ensure_dir, ensure_file, resolve_project_path
from prompt_prototypes import load_or_build_text_cache

# =========================
# Logging
# =========================
LOGGER = logging.getLogger("ESAM_CCLIP_11")

# =========================
# Quick Run Config
# 直接改这里，然后在 IDE 里运行本脚本即可
# =========================
DEFAULT_RESUME_BEST = OUTPUT_ROOT / "ESAM-CCLIP-11-fastfinetune" / "final_fullset.pt"
DEFAULT_RESUME = None
DEFAULT_NO_VAL = True
DEFAULT_NO_TRAIN_SPLIT_FILTER = True
DEFAULT_NO_AUTO_RESUME = True
DEFAULT_REBUILD_TEXT_CACHE = True
DEFAULT_TEXT_CACHE_PATH = TEXT_CACHE_PATH.with_name("text_emb_11_rare_balanced_lite.pt")
DEFAULT_RUN_NAME = RUN_NAME
DEFAULT_TRAIN_JSON = TRAIN_JSON
DEFAULT_VAL_JSON = VAL_JSON
DEFAULT_IMAGE_ROOT = IMAGE_ROOT
DEFAULT_TRAIN_LIST = TRAIN_LIST
DEFAULT_VAL_LIST = VAL_LIST


_DATASET_SIGNATURE_LOGGED = False


def make_dataset(**kwargs):
    global _DATASET_SIGNATURE_LOGGED
    signature = inspect.signature(ESAMCCLIP11Dataset.__init__)
    supported_keys = set(signature.parameters.keys()) - {"self"}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_keys}
    dropped_keys = sorted(key for key in kwargs.keys() if key not in supported_keys)
    if not _DATASET_SIGNATURE_LOGGED:
        LOGGER.info("dataset module file=%s", inspect.getfile(ESAMCCLIP11Dataset))
        LOGGER.info("dataset init signature=%s", signature)
        _DATASET_SIGNATURE_LOGGED = True
    if dropped_keys:
        LOGGER.warning("当前 dataset 不支持这些参数，已自动忽略: %s", dropped_keys)
    return ESAMCCLIP11Dataset(**filtered_kwargs)


def configure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argparser():
    parser = argparse.ArgumentParser(description="ESAM-CCLIP-11 fast finetune")
    parser.add_argument("--train_json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--val_json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--all_json", type=Path, default=ALL_JSON)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--train_list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val_list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--tokenizer_dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--efficient_sam_ckpt", type=Path, default=EFFICIENT_SAM_CKPT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--esam_input_size", type=int, default=ESAM_INPUT_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--no_val", action="store_true", default=DEFAULT_NO_VAL)
    parser.add_argument("--with_val", dest="no_val", action="store_false")
    parser.add_argument("--no_train_split_filter", action="store_true", default=DEFAULT_NO_TRAIN_SPLIT_FILTER)
    parser.add_argument("--use_train_split_filter", dest="no_train_split_filter", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--resume_best", type=Path, default=DEFAULT_RESUME_BEST)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--resume_weights_only", action="store_true", default=False)
    parser.add_argument("--reset_history", action="store_true", default=False)
    parser.add_argument("--no_auto_resume", action="store_true", default=DEFAULT_NO_AUTO_RESUME)
    parser.add_argument("--freeze_image_encoder", dest="freeze_image_encoder", action="store_true")
    parser.add_argument("--unfreeze_image_encoder", dest="freeze_image_encoder", action="store_false")
    parser.add_argument("--freeze_text_encoder", action="store_true", default=FREEZE_TEXT_ENCODER)
    parser.add_argument("--train_decoder_only", action="store_true", default=TRAIN_DECODER_ONLY)
    parser.add_argument("--use_prompt_prototype", dest="use_prompt_prototype", action="store_true")
    parser.add_argument("--no_prompt_prototype", dest="use_prompt_prototype", action="store_false")
    parser.add_argument("--text_cache_path", type=Path, default=DEFAULT_TEXT_CACHE_PATH)
    parser.add_argument("--rebuild_text_cache", action="store_true", default=DEFAULT_REBUILD_TEXT_CACHE)
    parser.add_argument("--use_image_cache", action="store_true", default=USE_IMAGE_CACHE)
    parser.add_argument("--image_cache_dir", type=Path, default=IMAGE_CACHE_DIR)
    parser.add_argument("--build_image_cache", action="store_true", default=False)
    parser.add_argument("--rebuild_image_cache", action="store_true", default=REBUILD_IMAGE_CACHE)
    parser.add_argument("--negative_sample_ratio", type=float, default=NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--negative_sample_weight", type=float, default=NEGATIVE_SAMPLE_WEIGHT)
    parser.add_argument("--old_class_sample_ratio", type=float, default=OLD_CLASS_SAMPLE_RATIO)
    parser.add_argument("--rare_class_keep_ratio", type=float, default=RARE_CLASS_KEEP_RATIO)
    parser.add_argument("--train_eval_after", action="store_true", default=False)
    parser.add_argument("--train_eval_max_samples", type=int, default=3000)
    parser.add_argument("--train_eval_batch_size", type=int, default=None)
    parser.add_argument("--train_eval_seed", type=int, default=42)
    parser.add_argument("--decoder_lr", type=float, default=DECODER_LR)
    parser.add_argument("--image_lr", type=float, default=IMAGE_LR)
    parser.add_argument("--text_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.set_defaults(
        amp=AMP,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
        use_prompt_prototype=USE_PROMPT_PROTOTYPE,
    )
    return parser


def resolve_runtime_paths(args):
    args.train_json = ensure_file(args.train_json, "train_json")
    args.all_json = ensure_file(args.all_json, "all_json")
    args.image_root = ensure_dir(args.image_root, "image_root")
    args.tokenizer_dir = ensure_dir(args.tokenizer_dir, "tokenizer_dir")
    args.efficient_sam_ckpt = ensure_file(args.efficient_sam_ckpt, "efficient_sam_ckpt")
    args.output_dir = resolve_project_path(args.output_dir)
    args.text_cache_path = resolve_project_path(args.text_cache_path)
    args.image_cache_dir = resolve_project_path(args.image_cache_dir)
    if args.train_list is not None:
        args.train_list = ensure_file(args.train_list, "train_list")
    if not args.no_val:
        args.val_json = ensure_file(args.val_json, "val_json")
        if args.val_list is not None:
            args.val_list = ensure_file(args.val_list, "val_list")
    else:
        args.val_json = resolve_project_path(args.val_json)
        args.val_list = resolve_project_path(args.val_list) if args.val_list is not None else None
    if args.resume is not None:
        args.resume = ensure_file(args.resume, "resume")
    if args.resume_best is not None:
        args.resume_best = ensure_file(args.resume_best, "resume_best")
    return args


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
        "class_names": [item["class_name"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "is_positive": [item["is_positive"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def build_datasets(args):
    effective_hflip_prob = 0.0 if args.use_image_cache else HFLIP_PROB
    LOGGER.info("开始构建训练集: %s", args.train_json)
    train_dataset = make_dataset(
        annotation_json=args.train_json,
        split_txt=None if args.no_train_split_filter else args.train_list,
        image_root=args.image_root,
        img_size=(args.img_size, args.img_size),
        classes=CLASSES,
        prompt_prototypes=PROMPT_PROTOTYPES,
        augment_prompt=False,
        hflip_prob=effective_hflip_prob,
        use_conf_filter=False,
        negative_sample_prob=args.negative_sample_ratio if INCLUDE_NEGATIVE_SAMPLES else 0.0,
        negative_sample_weight=args.negative_sample_weight,
        rare_oversample=RARE_OVERSAMPLE,
        old_class_sample_ratio=args.old_class_sample_ratio,
        rare_class_keep_ratio=args.rare_class_keep_ratio,
        old_classes=RARE_BALANCED_OLD_CLASSES,
        rare_classes=RARE_BALANCED_RARE_CLASSES,
        training=True,
        seed=args.seed,
    )
    if len(train_dataset) <= 0:
        raise RuntimeError("train_dataset 为空，请检查 train_json / split filter / prompt 过滤逻辑。")
    val_dataset = None
    if not args.no_val:
        LOGGER.info("开始构建验证集: %s", args.val_json)
        val_dataset = make_dataset(
            annotation_json=args.val_json,
            split_txt=args.val_list,
            image_root=args.image_root,
            img_size=(args.img_size, args.img_size),
            classes=CLASSES,
            prompt_prototypes=PROMPT_PROTOTYPES,
            augment_prompt=False,
            hflip_prob=0.0,
            use_conf_filter=False,
            negative_sample_prob=0.0 if not VAL_INCLUDE_NEGATIVE_SAMPLES else args.negative_sample_ratio,
            negative_sample_weight=args.negative_sample_weight,
            rare_oversample=None,
            old_class_sample_ratio=1.0,
            rare_class_keep_ratio=1.0,
            old_classes=RARE_BALANCED_OLD_CLASSES,
            rare_classes=RARE_BALANCED_RARE_CLASSES,
            training=False,
            seed=args.seed,
        )
        if len(val_dataset) <= 0:
            raise RuntimeError("val_dataset 为空，请检查 val_json / val_list / prompt 过滤逻辑。")
    LOGGER.info("train class positive stats: %s", train_dataset.stats["positive"])
    LOGGER.info("train class negative stats: %s", train_dataset.stats["negative"])
    if val_dataset is not None:
        LOGGER.info("val class positive stats: %s", val_dataset.stats["positive"])
    LOGGER.info("effective train hflip_prob=%s", effective_hflip_prob)
    return train_dataset, val_dataset


def build_optimizer(model, decoder_lr, image_lr, text_lr, weight_decay):
    image_params, text_params, decoder_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_encoder"):
            image_params.append(param)
        elif name.startswith("txt_encoder"):
            text_params.append(param)
        elif name.startswith("decoder"):
            decoder_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder"})
    if image_params and image_lr > 0:
        param_groups.append({"params": image_params, "lr": image_lr, "weight_decay": weight_decay, "name": "image"})
    if text_params and text_lr > 0:
        param_groups.append({"params": text_params, "lr": text_lr, "weight_decay": weight_decay, "name": "text"})
    if other_params:
        param_groups.append({"params": other_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "other"})
    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查冻结设置。")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"]) / 1e6
        LOGGER.info("[ParamGroup] %s lr=%s params=%.2fM", group["name"], group["lr"], n_params)
    return optim.AdamW(
        [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in param_groups],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def class_weight_tensor(class_names, device):
    weights = [CLASS_WEIGHTS.get(name, 1.0) for name in class_names]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_text_features(batch_class_names, text_cache_payload, device):
    features = [text_cache_payload["embeddings"][class_name] for class_name in batch_class_names]
    return torch.stack(features, dim=0).to(device)


def encode_images_with_optional_cache(model, images, image_paths, cache_dir, use_cache, build_cache, rebuild_cache):
    hit_count = 0
    features = []
    failed = []
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        cache_key = hashlib.md5(str(image_path).replace("\\", "/").encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.pt"
        feature = None
        if use_cache and cache_path.exists() and not rebuild_cache:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                feature = payload["image_embedding"]
                hit_count += 1
            except Exception as exc:
                failed.append({"image_path": image_path, "error": str(exc)})
        if feature is None:
            feature = model.encode_image(images[idx: idx + 1]).detach().cpu()
            if use_cache and build_cache:
                try:
                    torch.save(
                        {
                            "image_path": image_path,
                            "image_embedding": feature,
                            "input_size": list(images.shape[-2:]),
                        },
                        cache_path,
                    )
                except Exception as exc:
                    failed.append({"image_path": image_path, "error": str(exc)})
        features.append(feature[0])
    return torch.stack(features, dim=0).to(images.device), hit_count, failed


def train_one_epoch(model, loader, optimizer, scheduler, criterion_dice, criterion_focal, scaler, device, text_cache_payload, args):
    model.train()
    if args.freeze_image_encoder:
        model.img_encoder.eval()
    if args.freeze_text_encoder:
        model.txt_encoder.eval()
    running = defaultdict(float)
    cache_hits = 0
    cache_total = 0
    failed_cache_items = []
    start = time.time()
    iterator = maybe_tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in iterator:
        batch_start = time.time()
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        class_names = batch["class_names"]

        if args.use_image_cache:
            image_features, hit_count, failed = encode_images_with_optional_cache(
                model=model,
                images=images,
                image_paths=batch["image_paths"],
                cache_dir=args.image_cache_dir,
                use_cache=True,
                build_cache=args.build_image_cache,
                rebuild_cache=args.rebuild_image_cache,
            )
            cache_hits += hit_count
            cache_total += len(batch["image_paths"])
            failed_cache_items.extend(failed)
        else:
            image_features = model.encode_image(images)

        text_features = get_text_features(class_names, text_cache_payload, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model.decode(image_features=image_features, text_features=text_features, target_size=(args.img_size, args.img_size))
            bce_map = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=(1, 2, 3))
            dice_val = torch.stack([criterion_dice(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
            focal_val = torch.stack([criterion_focal(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
            cls_weights = class_weight_tensor(class_names, device)
            weighted = (BCE_WEIGHT * bce_map + DICE_WEIGHT * dice_val + FOCAL_WEIGHT * focal_val) * sample_weight * cls_weights
            loss = weighted.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running["loss"] += loss.item()
        running["bce"] += bce_map.mean().item()
        running["dice"] += dice_val.mean().item()
        running["focal"] += focal_val.mean().item()
        running["batch_time"] += (time.time() - batch_start)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    if failed_cache_items:
        save_json(args.output_dir / args.run_name / "failed_image_cache.json", failed_cache_items)
    denom = max(len(loader), 1)
    running["loss"] /= denom
    running["bce"] /= denom
    running["dice"] /= denom
    running["focal"] /= denom
    running["batch_time"] /= denom
    running["epoch_time"] = time.time() - start
    running["image_cache_hit_rate"] = float(cache_hits / max(cache_total, 1)) if cache_total > 0 else 0.0
    return running


@torch.no_grad()
def validate(model, loader, criterion_dice, criterion_focal, device, text_cache_payload, args):
    model.eval()
    metrics = defaultdict(list)
    start = time.time()
    iterator = maybe_tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        class_names = batch["class_names"]
        image_features = model.encode_image(images)
        text_features = get_text_features(class_names, text_cache_payload, device)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model.decode(image_features=image_features, text_features=text_features, target_size=(args.img_size, args.img_size))
        bce_map = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=(1, 2, 3))
        dice_val = torch.stack([criterion_dice(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
        focal_val = torch.stack([criterion_focal(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
        metrics["bce"].append(float(bce_map.mean().item()))
        metrics["dice_loss"].append(float(dice_val.mean().item()))
        metrics["focal"].append(float(focal_val.mean().item()))
        for idx, class_name in enumerate(class_names):
            score = compute_metrics(logits[idx: idx + 1], masks[idx: idx + 1], threshold=VAL_THRESHOLDS.get(class_name, 0.5))
            metrics["iou/overall"].append(score["iou"])
            metrics["dice/overall"].append(score["dice"])
            metrics["precision/overall"].append(score["precision"])
            metrics["recall/overall"].append(score["recall"])
            metrics["pred_area/overall"].append(score["pred_area"])
            metrics["gt_area/overall"].append(score["gt_area"])
            metrics[f"iou/{class_name}"].append(score["iou"])
            metrics[f"precision/{class_name}"].append(score["precision"])
            metrics[f"recall/{class_name}"].append(score["recall"])
            if class_name in OLD5_CLASSES:
                metrics["iou/old5"].append(score["iou"])
            if class_name in RARE_CLASSES:
                metrics["iou/rare"].append(score["iou"])
            if batch["is_positive"][idx]:
                metrics["iou/pos_only"].append(score["iou"])
    summary = {key: float(np.mean(values)) for key, values in metrics.items() if values}
    summary["val_time"] = time.time() - start
    return summary


def plot_results(history, save_path, no_val=False):
    fig = plt.figure(figsize=(16, 8))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="train_loss")
    if no_val:
        ax1.plot(epochs, history["train_bce"], label="train_bce")
    else:
        ax1.plot(epochs, history["val_miou_all11"], label="val_miou_all11")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    if no_val:
        ax2.plot(epochs, history["train_dice"], label="train_dice")
        ax2.plot(epochs, history["train_focal"], label="train_focal")
    else:
        ax2.plot(epochs, history["val_miou_old5"], label="old5")
        ax2.plot(epochs, history["val_miou_rare"], label="rare")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    if no_val:
        ax3.plot(epochs, history["image_cache_hit_rate"], label="image_cache_hit_rate")
    else:
        ax3.plot(epochs, history["val_dice"], label="dice")
        ax3.plot(epochs, history["val_pos_miou"], label="pos_miou")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(epochs, history["lr"], label="lr", color="red")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def evaluate_train_diagnostic(model, loader, device):
    model.eval()
    metrics = defaultdict(list)
    per_class = {
        class_name: {
            "iou": [],
            "precision": [],
            "recall": [],
            "pred_area": [],
            "gt_area": [],
        }
        for class_name in CLASSES
    }
    iterator = maybe_tqdm(loader, total=len(loader), desc="TrainDiag", leave=False)
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        class_names = batch["class_names"]
        image_features = model.encode_image(images)
        text_features = torch.stack(
            [loader.text_cache_payload["embeddings"][class_name] for class_name in class_names],
            dim=0,
        ).to(device)
        logits = model.decode(
            image_features=image_features,
            text_features=text_features,
            target_size=(loader.diag_img_size, loader.diag_img_size),
        )
        for idx, class_name in enumerate(class_names):
            threshold = VAL_THRESHOLDS.get(class_name, 0.5)
            score = compute_metrics(logits[idx: idx + 1], masks[idx: idx + 1], threshold=threshold)
            metrics["iou/overall"].append(score["iou"])
            metrics["precision/overall"].append(score["precision"])
            metrics["recall/overall"].append(score["recall"])
            if class_name in OLD5_CLASSES:
                metrics["iou/old5"].append(score["iou"])
            if class_name in RARE_CLASSES:
                metrics["iou/rare"].append(score["iou"])
            if batch["is_positive"][idx]:
                metrics["iou/pos_only"].append(score["iou"])
            per_class[class_name]["iou"].append(score["iou"])
            per_class[class_name]["precision"].append(score["precision"])
            per_class[class_name]["recall"].append(score["recall"])
            per_class[class_name]["pred_area"].append(score["pred_area"])
            per_class[class_name]["gt_area"].append(score["gt_area"])

    summary = {
        "train_mIoU_all11_on_pseudo": float(np.mean(metrics["iou/overall"])) if metrics["iou/overall"] else 0.0,
        "train_mIoU_old5_on_pseudo": float(np.mean(metrics["iou/old5"])) if metrics["iou/old5"] else 0.0,
        "train_mIoU_rare_on_pseudo": float(np.mean(metrics["iou/rare"])) if metrics["iou/rare"] else 0.0,
        "train_pos_mIoU_on_pseudo": float(np.mean(metrics["iou/pos_only"])) if metrics["iou/pos_only"] else 0.0,
        "per_class_iou": {},
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_pred_area": {},
        "per_class_gt_area": {},
    }
    per_class_rows = []
    for class_name in CLASSES:
        class_summary = {
            "iou": float(np.mean(per_class[class_name]["iou"])) if per_class[class_name]["iou"] else 0.0,
            "precision": float(np.mean(per_class[class_name]["precision"])) if per_class[class_name]["precision"] else 0.0,
            "recall": float(np.mean(per_class[class_name]["recall"])) if per_class[class_name]["recall"] else 0.0,
            "pred_area": float(np.mean(per_class[class_name]["pred_area"])) if per_class[class_name]["pred_area"] else 0.0,
            "gt_area": float(np.mean(per_class[class_name]["gt_area"])) if per_class[class_name]["gt_area"] else 0.0,
        }
        summary["per_class_iou"][class_name] = class_summary["iou"]
        summary["per_class_precision"][class_name] = class_summary["precision"]
        summary["per_class_recall"][class_name] = class_summary["recall"]
        summary["per_class_pred_area"][class_name] = class_summary["pred_area"]
        summary["per_class_gt_area"][class_name] = class_summary["gt_area"]
        per_class_rows.append(
            [
                class_name,
                class_summary["iou"],
                class_summary["precision"],
                class_summary["recall"],
                class_summary["pred_area"],
                class_summary["gt_area"],
            ]
        )
    return summary, per_class_rows


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": dict(history),
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {idx: cls for idx, cls in enumerate(CLASSES)},
        "model_type": MODEL_TYPE,
        "use_prompt_prototype": args.use_prompt_prototype,
        "prompt_prototypes": args.prompt_prototype_cfg,
        "val_thresholds": VAL_THRESHOLDS,
        "postprocess_cfg": POSTPROCESS_DEFAULT,
        "img_size": int(args.img_size),
        "esam_input_size": int(args.esam_input_size),
        "text_cache_path": str(args.text_cache_path),
        "run_name": args.run_name,
    }
    torch.save(payload, path)


def resolve_resume_checkpoint(args, run_dir: Path):
    if args.resume is not None:
        return Path(args.resume)
    auto_resume_path = run_dir / "last.pt"
    if not args.no_auto_resume and auto_resume_path.exists():
        return auto_resume_path
    return None


def resume_training_state(model, optimizer, scheduler, checkpoint_path, device, args):
    checkpoint_path = Path(checkpoint_path)
    LOGGER.info("开始恢复断点续训: %s", checkpoint_path)
    checkpoint, missing_keys, unexpected_keys = load_checkpoint_flexible(
        model,
        checkpoint_path,
        device,
        require_decoder_match=True,
    )
    load_info = getattr(model, "_last_checkpoint_load_info", {})
    LOGGER.info("resume checkpoint path=%s", load_info.get("checkpoint_path", checkpoint_path))
    LOGGER.info("resume decoder_matched_keys=%d", len(load_info.get("decoder_matched_keys", [])))
    LOGGER.info("resume missing_keys=%s", missing_keys[:30])
    LOGGER.info("resume unexpected_keys=%s", unexpected_keys[:30])

    if args.resume_weights_only:
        LOGGER.info("resume_weights_only=True，仅恢复模型权重，不恢复 optimizer/scheduler/epoch/history。")
        return 1, defaultdict(list), {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0}, checkpoint

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    if args.reset_history:
        history = defaultdict(list)
        best_metrics = {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0}
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("reset_history=True，已清空 history/best_metrics，从 epoch %d 继续训练。", start_epoch)
    else:
        history_payload = checkpoint.get("history", {})
        history = defaultdict(list)
        for key, values in history_payload.items():
            history[key] = list(values)
        best_metrics = checkpoint.get(
            "best_metrics",
            {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0},
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("已恢复 epoch=%d, 下一轮从 epoch %d 开始。", int(checkpoint.get("epoch", 0)), start_epoch)

    return start_epoch, history, best_metrics, checkpoint


def main():
    parser = build_argparser()
    args = resolve_runtime_paths(parser.parse_args())
    args.output_dir = args.output_dir.resolve()
    run_dir = args.output_dir / args.run_name
    configure_logging(run_dir / f"{args.run_name}.log")
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.use_image_cache and not args.freeze_image_encoder:
        raise AssertionError("use_image_cache=True 时必须 freeze_image_encoder=True。")
    if args.use_image_cache:
        LOGGER.info("use_image_cache=True，训练阶段已强制关闭随机 hflip。")

    LOGGER.info("loaded checkpoint path=%s", args.resume_best)
    LOGGER.info("train_json=%s", args.train_json)
    LOGGER.info("val_json=%s", args.val_json)
    LOGGER.info("train_list=%s", args.train_list)
    LOGGER.info("val_list=%s", args.val_list)
    LOGGER.info("image_root=%s", args.image_root)
    LOGGER.info("resume=%s", args.resume)
    LOGGER.info("no_val=%s", args.no_val)
    LOGGER.info("no_train_split_filter=%s", args.no_train_split_filter)
    LOGGER.info("no_auto_resume=%s", args.no_auto_resume)
    LOGGER.info("freeze_image_encoder=%s", args.freeze_image_encoder)
    LOGGER.info("freeze_text_encoder=%s", args.freeze_text_encoder)
    LOGGER.info("train_decoder_only=%s", args.train_decoder_only)
    LOGGER.info("use_prompt_prototype=%s", args.use_prompt_prototype)
    LOGGER.info("use_image_cache=%s", args.use_image_cache)
    LOGGER.info("decoder_lr=%s image_lr=%s text_lr=%s", args.decoder_lr, args.image_lr, args.text_lr)
    LOGGER.info("negative_sample_ratio=%s", args.negative_sample_ratio)
    LOGGER.info("old_class_sample_ratio=%s", args.old_class_sample_ratio)
    LOGGER.info("rare_class_keep_ratio=%s", args.rare_class_keep_ratio)
    LOGGER.info("rare_oversample=%s", RARE_OVERSAMPLE)
    LOGGER.info("class_weights=%s", CLASS_WEIGHTS)
    LOGGER.info("old_classes=%s", RARE_BALANCED_OLD_CLASSES)
    LOGGER.info("rare_classes=%s", RARE_BALANCED_RARE_CLASSES)
    LOGGER.info("current classes=%s", CLASSES)
    if args.no_val:
        LOGGER.info("当前使用全集训练直接提交模式，不使用验证集选择 best checkpoint，最终请使用 final_fullset.pt 或 last.pt。")

    LOGGER.info("开始初始化模型...")
    model = ESAMCCLIPModel(
        tokenizer_dir=args.tokenizer_dir,
        efficient_sam_ckpt=args.efficient_sam_ckpt,
        freeze_image=args.freeze_image_encoder,
        freeze_text=args.freeze_text_encoder,
    ).to(device)
    if args.train_decoder_only:
        for param in model.img_encoder.parameters():
            param.requires_grad = False
        for param in model.txt_encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True

    total_params, trainable_params = count_parameters(model)
    LOGGER.info("total parameter count=%.2fM", total_params)
    LOGGER.info("trainable parameter count=%.2fM", trainable_params)

    resume_checkpoint_path = resolve_resume_checkpoint(args, run_dir)
    if resume_checkpoint_path is not None:
        LOGGER.info("检测到断点续训 checkpoint: %s", resume_checkpoint_path)
    elif args.resume_best is not None:
        LOGGER.info("开始加载热启动权重: %s", args.resume_best)
        checkpoint, missing_keys, unexpected_keys = load_checkpoint_flexible(
            model,
            args.resume_best,
            device,
            require_decoder_match=True,
        )
        load_info = getattr(model, "_last_checkpoint_load_info", {})
        LOGGER.info("checkpoint loaded path=%s", load_info.get("checkpoint_path", args.resume_best))
        LOGGER.info("decoder_matched_keys=%d", len(load_info.get("decoder_matched_keys", [])))
        LOGGER.info("missing_keys=%s", missing_keys[:30])
        LOGGER.info("unexpected_keys=%s", unexpected_keys[:30])
        if checkpoint and not args.resume_weights_only:
            LOGGER.info("checkpoint loaded with optimizer/scheduler reuse disabled in fastfinetune mode")

    LOGGER.info("开始加载 tokenizer: %s", args.tokenizer_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    prototype_config = PROMPT_PROTOTYPES if args.use_prompt_prototype else {class_name: [class_name] for class_name in CLASSES}
    args.prompt_prototype_cfg = prototype_config
    LOGGER.info("开始%s text cache: %s", "重建" if args.rebuild_text_cache else "加载/构建", args.text_cache_path)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=args.text_cache_path,
        classes=CLASSES,
        prompt_prototypes=prototype_config,
        device=device,
        rebuild=args.rebuild_text_cache,
    )
    if set(text_cache_payload["classes"]) != set(CLASSES):
        raise RuntimeError("text cache 与当前 11 类不一致，请使用 --rebuild_text_cache")
    for class_name in CLASSES:
        if class_name not in text_cache_payload["embeddings"]:
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}，请使用 --rebuild_text_cache")

    train_dataset, val_dataset = build_datasets(args)
    LOGGER.info("开始构建 train DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        LOGGER.info("开始构建 val DataLoader...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
        )

    optimizer = build_optimizer(model, args.decoder_lr, args.image_lr, args.text_lr, args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = WARMUP_EPOCHS * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, MIN_LR_RATIO) if total_steps > 0 else None
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    criterion_dice = DiceLoss()
    criterion_focal = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    save_json(run_dir / "config_used.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    history = defaultdict(list)
    best_metrics = {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0}
    start_epoch = 1
    if resume_checkpoint_path is not None:
        start_epoch, history, best_metrics, _ = resume_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=resume_checkpoint_path,
            device=device,
            args=args,
        )
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"断点续训下一轮 epoch={start_epoch} 已大于设定总轮数 epochs={args.epochs}，请增大 --epochs 或关闭自动续训。"
            )
    last_epoch = 0

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        LOGGER.info("%s Epoch %d/%d", "─" * 50, epoch, args.epochs)
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion_dice, criterion_focal, scaler, device, text_cache_payload, args)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_stats["loss"])
        history["train_bce"].append(train_stats["bce"])
        history["train_dice"].append(train_stats["dice"])
        history["train_focal"].append(train_stats["focal"])
        history["lr"].append(current_lr)
        history["epoch_time"].append(train_stats["epoch_time"])
        history["batch_time"].append(train_stats["batch_time"])
        history["image_cache_hit_rate"].append(train_stats["image_cache_hit_rate"])

        if args.no_val:
            LOGGER.info(
                "TrainLoss=%.4f BCE=%.4f Dice=%.4f Focal=%.4f LR=%.2e epoch_time=%.1fs batch_time=%.2fs image_cache_hit_rate=%.2f%%",
                train_stats["loss"],
                train_stats["bce"],
                train_stats["dice"],
                train_stats["focal"],
                current_lr,
                train_stats["epoch_time"],
                train_stats["batch_time"],
                100.0 * train_stats["image_cache_hit_rate"],
            )
        else:
            val_metrics = validate(model, val_loader, criterion_dice, criterion_focal, device, text_cache_payload, args)
            history["val_miou_all11"].append(val_metrics.get("iou/overall", 0.0))
            history["val_miou_old5"].append(val_metrics.get("iou/old5", 0.0))
            history["val_miou_rare"].append(val_metrics.get("iou/rare", 0.0))
            history["val_pos_miou"].append(val_metrics.get("iou/pos_only", 0.0))
            history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
            for class_name in CLASSES:
                history[f"iou/{class_name}"].append(val_metrics.get(f"iou/{class_name}", 0.0))

            LOGGER.info(
                "TrainLoss=%.4f BCE=%.4f Dice=%.4f Focal=%.4f mIoU_all11=%.4f mIoU_old5=%.4f mIoU_rare=%.4f "
                "pos_mIoU=%.4f Dice=%.4f LR=%.2e epoch_time=%.1fs batch_time=%.2fs image_cache_hit_rate=%.2f%%",
                train_stats["loss"],
                train_stats["bce"],
                train_stats["dice"],
                train_stats["focal"],
                val_metrics.get("iou/overall", 0.0),
                val_metrics.get("iou/old5", 0.0),
                val_metrics.get("iou/rare", 0.0),
                val_metrics.get("iou/pos_only", 0.0),
                val_metrics.get("dice/overall", 0.0),
                current_lr,
                train_stats["epoch_time"],
                train_stats["batch_time"],
                100.0 * train_stats["image_cache_hit_rate"],
            )
            LOGGER.info("Per-class IoU: %s", "  ".join(f"{name}={val_metrics.get(f'iou/{name}', 0.0):.3f}" for name in CLASSES))
        if torch.cuda.is_available() and device.type == "cuda":
            LOGGER.info("GPU memory max allocated: %.2f GB", torch.cuda.max_memory_allocated() / (1024 ** 3))

        save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
        if not args.no_val:
            current_all11 = val_metrics.get("iou/overall", 0.0)
            current_old5 = val_metrics.get("iou/old5", 0.0)
            current_rare = val_metrics.get("iou/rare", 0.0)
            if current_all11 > best_metrics["best_all11"]:
                best_metrics["best_all11"] = current_all11
                save_checkpoint(run_dir / "best_all11.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if current_old5 > best_metrics["best_old5"]:
                best_metrics["best_old5"] = current_old5
                save_checkpoint(run_dir / "best_old5.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if current_rare > best_metrics["best_rare"]:
                best_metrics["best_rare"] = current_rare
                save_checkpoint(run_dir / "best_rare.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)

    if last_epoch > 0:
        save_checkpoint(run_dir / "final_fullset.pt", last_epoch, model, optimizer, scheduler, history, best_metrics, args)

    if args.train_eval_after:
        LOGGER.info("开始训练集诊断评估。注意：这个指标只是伪标签拟合诊断，不是真实验证分数。")
        LOGGER.info("train diagnostic 使用的是训练伪标签，只用于检查是否学崩，不代表真实榜单表现。")
        train_diag_dataset = make_dataset(
            annotation_json=args.train_json,
            split_txt=None if args.no_train_split_filter else args.train_list,
            image_root=args.image_root,
            img_size=(args.img_size, args.img_size),
            classes=CLASSES,
            prompt_prototypes=PROMPT_PROTOTYPES,
            augment_prompt=False,
            hflip_prob=0.0,
            use_conf_filter=False,
            negative_sample_prob=0.0,
            negative_sample_weight=args.negative_sample_weight,
            rare_oversample=None,
            old_class_sample_ratio=1.0,
            rare_class_keep_ratio=1.0,
            old_classes=RARE_BALANCED_OLD_CLASSES,
            rare_classes=RARE_BALANCED_RARE_CLASSES,
            training=False,
            seed=args.train_eval_seed,
        )
        if args.train_eval_max_samples > 0 and len(train_diag_dataset) > args.train_eval_max_samples:
            rng = random.Random(args.train_eval_seed)
            indices = list(range(len(train_diag_dataset)))
            rng.shuffle(indices)
            train_diag_dataset = Subset(train_diag_dataset, indices[: args.train_eval_max_samples])
        diag_batch_size = args.train_eval_batch_size or args.batch_size
        train_diag_loader = DataLoader(
            train_diag_dataset,
            batch_size=diag_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
        )
        train_diag_loader.text_cache_payload = text_cache_payload
        train_diag_loader.diag_img_size = args.img_size
        diag_summary, diag_rows = evaluate_train_diagnostic(model, train_diag_loader, device)
        save_json(run_dir / "train_diagnostic_metrics.json", diag_summary)
        with open(run_dir / "train_diagnostic_per_class.csv", "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["class_name", "iou", "precision", "recall", "pred_area", "gt_area"])
            writer.writerows(diag_rows)
        LOGGER.info(
            "train diagnostic: all11=%.4f old5=%.4f rare=%.4f pos=%.4f",
            diag_summary["train_mIoU_all11_on_pseudo"],
            diag_summary["train_mIoU_old5_on_pseudo"],
            diag_summary["train_mIoU_rare_on_pseudo"],
            diag_summary["train_pos_mIoU_on_pseudo"],
        )

    if args.no_val:
        csv_keys = [
            "train_loss",
            "train_bce",
            "train_dice",
            "train_focal",
            "lr",
            "epoch_time",
            "batch_time",
            "image_cache_hit_rate",
        ]
    else:
        csv_keys = [
            "train_loss",
            "train_bce",
            "train_dice",
            "train_focal",
            "val_miou_all11",
            "val_miou_old5",
            "val_miou_rare",
            "val_pos_miou",
            "val_dice",
            "lr",
            "epoch_time",
            "batch_time",
            "image_cache_hit_rate",
        ] + [f"iou/{c}" for c in CLASSES]
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch"] + csv_keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in csv_keys])
    plot_results(history, run_dir / "results.png", no_val=args.no_val)


if __name__ == "__main__":
    main()
