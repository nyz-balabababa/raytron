import csv
import logging
import math
import random
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated\. Please use `torch\.utils\._pytree\.register_pytree_node` instead\.",
    category=FutureWarning,
    module=r"transformers\.utils\.generic",
)
warnings.filterwarnings(
    "ignore",
    message=r"Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\. Use `HF_HOME` instead\.",
    category=FutureWarning,
    module=r"transformers\.utils\.hub",
)

from config_hanxue import (
    APPLY_SCORE_FILTER,
    BATCH,
    BCE_WEIGHT,
    CLASSES,
    CONF_FILTER,
    CLASS_FOCAL_WEIGHT,
    DECODER_LR,
    DEVICE,
    DICE_WEIGHT,
    EPOCHS,
    FOCAL_GAMMA,
    FOCAL_NEG_FACTOR,
    FOCAL_POS_FACTOR,
    FOCAL_WEIGHT,
    GRAD_CLIP,
    HFLIP_PROB,
    IMAGE_LR,
    IMAGE_ROOT,
    INCLUDE_NEGATIVE_SAMPLES,
    IMG_SIZE,
    LOSS_WEIGHT_FLOOR,
    MIN_LR_RATIO,
    NEGATIVE_SAMPLE_RATIO,
    NEGATIVE_SAMPLE_WEIGHT,
    PROMPT_AUG_PROB,
    PROJECT,
    PROMPT_THRESHOLDS,
    PROMPT_AUGMENTATIONS,
    RARE_OVERSAMPLE,
    RUN_NAME,
    SEED,
    TOKENIZER_DIR,
    TRAIN_LIST,
    TRAIN_PRED_JSON,
    VAL_LIST,
    VAL_PRED_JSON,
    VAL_THRESHOLDS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    WORKERS,
    validate_required_paths,
)
from data.dataset import InfraredPromptDataset
from models.custom_sam_model import CustomSAMWorldModel
from utils.losses import SemanticSegLoss


LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total_params, trainable_params


def resolve_device(device_name: str):
    requested = str(device_name).lower()

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("config DEVICE 请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU。")

        device = torch.device(device_name if ":" in requested else "cuda:0")
        device_index = device.index if device.index is not None else 0
        logger.info(
            f"CUDA 可用: count={torch.cuda.device_count()} "
            f"current={device_index} "
            f"name={torch.cuda.get_device_name(device_index)}"
        )
        return device

    device = torch.device(device_name)
    logger.info(f"使用非 CUDA 设备: {device}")
    return device


def maybe_tqdm(iterable, total, desc, leave=False):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        ncols=96,
        bar_format="{l_bar}{bar:18}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )


def build_optimizer(model, decoder_lr, image_lr, weight_decay):
    image_params = []
    decoder_params = []
    text_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("img_encoder") or "image_encoder" in name or "img_encoder" in name:
            image_params.append(param)
            logger.info(f"[ParamGroup:image_lr] {name}")
        elif name.startswith("decoder") or ".decoder" in name or "fusion_decoder" in name:
            decoder_params.append(param)
            logger.info(f"[ParamGroup:decoder_lr] {name}")
        elif name.startswith("txt_encoder") or "text_encoder" in name or "txt_encoder" in name:
            text_params.append(param)
            logger.warning(f"[ParamGroup:text_trainable] {name}，注意：文本编码器本应默认冻结")
        else:
            other_params.append(param)
            logger.warning(f"[ParamGroup:other_decoder_lr] {name}")

    param_groups = []
    if image_params and image_lr > 0:
        param_groups.append({"params": image_params, "lr": image_lr, "weight_decay": weight_decay, "name": "image"})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder"})
    if text_params:
        param_groups.append(
            {
                "params": text_params,
                "lr": min(image_lr, 1e-6),
                "weight_decay": weight_decay,
                "name": "text",
            }
        )
    if other_params:
        param_groups.append({"params": other_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "other"})

    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查模型冻结设置。")

    logger.info("[Optimizer] 参数组:")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"]) / 1e6
        logger.info(f"  {group['name']}: lr={group['lr']}, params={n_params:.2f}M")

    return optim.AdamW(
        [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in param_groups],
        weight_decay=weight_decay,
    )


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.01):
    if total_steps <= 0:
        return None

    warmup_steps = int(warmup_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def compute_metrics(logits, targets, threshold):
    pred = (torch.sigmoid(logits) > threshold).float()
    gt = (targets > 0.5).float()

    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()
    if pred_sum == 0 and gt_sum == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}

    inter = (pred * gt).sum().item()
    union = (pred + gt).clamp(0, 1).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    return {
        "iou": inter / (union + 1e-7),
        "dice": 2 * inter / (pred_sum + gt_sum + 1e-7),
        "precision": inter / (inter + fp + 1e-7),
        "recall": inter / (inter + fn + 1e-7),
    }


def class_focal_aux_loss(
    logits,
    targets,
    prompts,
    sample_weight,
    gamma,
    class_focal_weight,
    weight_floor,
    pos_factor=1.0,
    neg_factor=0.25,
):
    """
    BCE + Dice 主损失之外的按类 focal 辅助项。
    logits: [B, 1, H, W]
    targets: [B, 1, H, W]
    prompts: list[str]
    sample_weight: [B]
    """
    if logits.shape != targets.shape:
        raise RuntimeError(f"class focal logits/targets shape 不一致: {logits.shape} vs {targets.shape}")

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    focal_map = ((1.0 - pt).clamp(min=1e-6) ** gamma) * bce

    pos_mask = (targets > 0.5).float()
    neg_mask = 1.0 - pos_mask

    flat_focal = focal_map.flatten(1)
    flat_pos = pos_mask.flatten(1)
    flat_neg = neg_mask.flatten(1)

    pos_area = flat_pos.sum(dim=1)
    neg_area = flat_neg.sum(dim=1).clamp(min=1.0)

    pos_loss = (flat_focal * flat_pos).sum(dim=1) / pos_area.clamp(min=1.0)
    neg_loss = (flat_focal * flat_neg).sum(dim=1) / neg_area

    has_pos = pos_area > 0

    per_sample = torch.where(
        has_pos,
        pos_factor * pos_loss + neg_factor * neg_loss,
        neg_factor * neg_loss,
    )

    cls_w = torch.tensor(
        [float(class_focal_weight.get(str(p), 0.0)) for p in prompts],
        device=logits.device,
        dtype=per_sample.dtype,
    )

    sw = sample_weight.float()
    sw = torch.where(
        has_pos,
        sw.clamp(min=weight_floor),
        sw,
    )
    return (per_sample * cls_w * sw).mean()


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_focal_loss = 0.0
    valid_batches = 0

    progress = maybe_tqdm(loader, total=len(loader), desc=f"Train {epoch}/{EPOCHS}", leave=False)

    for batch_idx, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        class_prompts = batch["class_name"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, input_ids, attention_mask)
        if logits.ndim != 4:
            raise RuntimeError(f"logits 应为 [B,1,H,W]，当前: {logits.shape}")
        if logits.shape[1] != 1:
            raise RuntimeError(f"logits channel 应为 1，当前: {logits.shape}")
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss_main = criterion(logits, masks, sample_weight=sample_weight)
        loss_focal = class_focal_aux_loss(
            logits=logits,
            targets=masks,
            prompts=class_prompts,
            sample_weight=sample_weight,
            gamma=FOCAL_GAMMA,
            class_focal_weight=CLASS_FOCAL_WEIGHT,
            weight_floor=LOSS_WEIGHT_FLOOR,
            pos_factor=FOCAL_POS_FACTOR,
            neg_factor=FOCAL_NEG_FACTOR,
        )
        loss = loss_main + loss_focal

        if not torch.isfinite(loss):
            logger.warning(f"Epoch {epoch} Batch {batch_idx}: loss 非有限值，跳过。")
            continue

        loss.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            trainable_parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(trainable_parameters, GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_main_loss += float(loss_main.item())
        total_focal_loss += float(loss_focal.item())
        valid_batches += 1

    if valid_batches == 0:
        raise RuntimeError(f"Epoch {epoch} 没有任何有效 batch。")
    return {
        "loss": total_loss / valid_batches,
        "main": total_main_loss / valid_batches,
        "focal": total_focal_loss / valid_batches,
    }


@torch.no_grad()
def validate(model, loader, device, epoch=None):
    model.eval()
    all_metrics = defaultdict(list)

    desc = f"Val {epoch}/{EPOCHS}" if epoch is not None else "Val"
    progress = maybe_tqdm(loader, total=len(loader), desc=desc, leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompts = batch["prompt"]

        logits = model(images, input_ids, attention_mask)
        if logits.ndim != 4:
            raise RuntimeError(f"[Val] logits 应为 [B,1,H,W]，当前: {logits.shape}")
        if logits.shape[1] != 1:
            raise RuntimeError(f"[Val] logits channel 应为 1，当前: {logits.shape}")
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        for i in range(images.size(0)):
            threshold = VAL_THRESHOLDS.get(prompts[i], 0.5)
            metrics = compute_metrics(logits[i : i + 1], masks[i : i + 1], threshold=threshold)
            for key, value in metrics.items():
                all_metrics[f"{key}/{prompts[i]}"].append(value)
            all_metrics["iou/overall"].append(metrics["iou"])
            all_metrics["dice/overall"].append(metrics["dice"])
            is_pos = bool((masks[i] > 0.5).sum().item() > 0)
            if is_pos:
                all_metrics["iou/pos_only"].append(metrics["iou"])
                all_metrics["dice/pos_only"].append(metrics["dice"])

        if tqdm is not None and all_metrics["iou/overall"]:
            progress.set_postfix(iou=f"{np.mean(all_metrics['iou/overall']):.4f}")

    return {key: float(np.mean(values)) for key, values in all_metrics.items()}


def smoke_test_forward(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(images, input_ids, attention_mask)

    logger.info(f"[SmokeTest] images: {tuple(images.shape)}")
    logger.info(f"[SmokeTest] masks:  {tuple(masks.shape)}")
    logger.info(f"[SmokeTest] logits: {tuple(logits.shape)}")
    logger.info(
        f"[SmokeTest] image value min/max before model: "
        f"{images.min().item():.4f} / {images.max().item():.4f}"
    )
    logger.info(f"[SmokeTest] logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    assert logits.ndim == 4, f"logits 应为 4D [B,1,H,W]，当前 {logits.shape}"
    assert logits.shape[1] == 1, f"logits channel 应为 1，当前 {logits.shape}"
    assert logits.shape[-2:] == masks.shape[-2:], f"logits/masks 尺寸不一致: {logits.shape} vs {masks.shape}"


def build_datasets():
    dataset_kwargs = dict(
        images_root=IMAGE_ROOT,
        tokenizer_dir=TOKENIZER_DIR,
        img_size=(IMG_SIZE, IMG_SIZE),
        use_conf_filter=APPLY_SCORE_FILTER,
        class_names=CLASSES,
        prompt_thresholds=PROMPT_THRESHOLDS,
        loss_weight_floor=LOSS_WEIGHT_FLOOR,
        negative_sample_weight=NEGATIVE_SAMPLE_WEIGHT,
        rare_oversample=RARE_OVERSAMPLE,
        seed=SEED,
    )

    train_dataset = InfraredPromptDataset(
        annotation_json=TRAIN_PRED_JSON,
        split_txt=TRAIN_LIST,
        augment_prompt=True,
        hflip_prob=HFLIP_PROB,
        negative_sample_prob=NEGATIVE_SAMPLE_RATIO if INCLUDE_NEGATIVE_SAMPLES else 0.0,
        training=True,
        **dataset_kwargs,
    )
    val_dataset = InfraredPromptDataset(
        annotation_json=VAL_PRED_JSON,
        split_txt=VAL_LIST,
        hflip_prob=0.0,
        augment_prompt=False,
        negative_sample_prob=1.0 if INCLUDE_NEGATIVE_SAMPLES and NEGATIVE_SAMPLE_RATIO > 0 else 0.0,
        training=False,
        **dataset_kwargs,
    )
    return train_dataset, val_dataset


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_miou, best_pos_miou):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history": dict(history),
            "best_miou": best_miou,
            "best_pos_miou": best_pos_miou,
        },
        path,
    )


def train():
    set_seed(SEED)
    device = resolve_device(DEVICE)

    run_dir = PROJECT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    best_pt = run_dir / "best.pt"
    best_pos_pt = run_dir / "best_pos.pt"
    last_pt = run_dir / "last.pt"
    results_csv = run_dir / "results.csv"

    logger.info("========== Hanxue Train Config ==========")
    logger.info(f"device: {device}")
    logger.info(f"classes: {CLASSES}")
    logger.info(f"train_pred: {TRAIN_PRED_JSON}")
    logger.info(f"val_pred: {VAL_PRED_JSON}")
    logger.info(f"input_size: {IMG_SIZE}")
    logger.info(f"batch_size: {BATCH}")
    logger.info(f"epochs: {EPOCHS}")
    logger.info(f"decoder_lr: {DECODER_LR}")
    logger.info(f"image_lr: {IMAGE_LR}")
    logger.info(f"score_filter: {APPLY_SCORE_FILTER} thresholds={CONF_FILTER}")
    logger.info(f"prompt_aug_prob: {PROMPT_AUG_PROB}")
    logger.info(f"prompt_aug_keys: {list(PROMPT_AUGMENTATIONS.keys())}")
    logger.info(f"focal_weight(global): {FOCAL_WEIGHT}")
    logger.info(f"focal_gamma(class-aware): {FOCAL_GAMMA}")
    logger.info(f"focal_pos_factor: {FOCAL_POS_FACTOR}")
    logger.info(f"focal_neg_factor: {FOCAL_NEG_FACTOR}")
    logger.info(f"class_focal_weight: {CLASS_FOCAL_WEIGHT}")
    logger.info(
        f"negatives: {INCLUDE_NEGATIVE_SAMPLES} "
        f"(train ratio={NEGATIVE_SAMPLE_RATIO}, weight={NEGATIVE_SAMPLE_WEIGHT})"
    )
    logger.info("========================================")

    model = CustomSAMWorldModel().to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"总参数量: {total_params:.2f} M")
    logger.info(f"可训练参数量: {trainable_params:.2f} M")

    train_dataset, val_dataset = build_datasets()
    if len(train_dataset) == 0:
        raise RuntimeError("训练集样本数为 0。")
    if len(val_dataset) == 0:
        raise RuntimeError("验证集样本数为 0。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    smoke_test_forward(model, train_loader, device)

    optimizer = build_optimizer(model, DECODER_LR, IMAGE_LR, WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, MIN_LR_RATIO)
    criterion = SemanticSegLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT, focal_weight=FOCAL_WEIGHT)

    history = defaultdict(list)
    best_miou = -1.0
    best_pos_miou = -1.0
    start_epoch = 1

    resume_ckpt = last_pt if last_pt.exists() else (best_pt if best_pt.exists() else None)
    if resume_ckpt is not None:
        logger.info(f"恢复 checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        for key, value in ckpt.get("history", {}).items():
            history[key] = list(value)
        best_miou = ckpt.get("best_miou", -1.0)
        best_pos_miou = ckpt.get("best_pos_miou", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if start_epoch > EPOCHS:
            logger.info(f"已完成训练: start_epoch={start_epoch}, EPOCHS={EPOCHS}，跳过训练。")
            return best_pt if best_pt.exists() else last_pt

    for epoch in range(start_epoch, EPOCHS + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_metrics = validate(model, val_loader, device, epoch=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_stats["loss"])
        history["val_miou"].append(val_metrics.get("iou/overall", 0.0))
        history["val_pos_miou"].append(val_metrics.get("iou/pos_only", 0.0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
        history["val_pos_dice"].append(val_metrics.get("dice/pos_only", 0.0))
        history["lr"].append(current_lr)
        for cls_name in CLASSES:
            history[f"iou/{cls_name}"].append(val_metrics.get(f"iou/{cls_name}", 0.0))

        logger.info(
            f"TrainLoss={train_stats['loss']:.4f}  Main={train_stats['main']:.4f}  "
            f"Focal={train_stats['focal']:.4f}  mIoU={val_metrics.get('iou/overall', 0.0):.4f}  "
            f"pos_mIoU={val_metrics.get('iou/pos_only', 0.0):.4f}  "
            f"Dice={val_metrics.get('dice/overall', 0.0):.4f}  LR={current_lr:.2e}  {elapsed/60:.1f}min"
        )
        logger.info("Per-class IoU: " + "  ".join(f"{c}={val_metrics.get(f'iou/{c}', 0.0):.3f}" for c in CLASSES))

        current_miou = val_metrics.get("iou/overall", 0.0)
        current_pos_miou = val_metrics.get("iou/pos_only", 0.0)
        is_best = current_miou > best_miou
        is_best_pos = current_pos_miou > best_pos_miou
        if is_best:
            best_miou = current_miou
        if is_best_pos:
            best_pos_miou = current_pos_miou

        save_checkpoint(last_pt, epoch, model, optimizer, scheduler, history, best_miou, best_pos_miou)

        if is_best:
            save_checkpoint(best_pt, epoch, model, optimizer, scheduler, history, best_miou, best_pos_miou)
            logger.info(f"★ overall best 更新 (mIoU={best_miou:.4f})")
        if is_best_pos:
            save_checkpoint(best_pos_pt, epoch, model, optimizer, scheduler, history, best_miou, best_pos_miou)
            logger.info(f"★ pos_only best 更新 (pos_mIoU={best_pos_miou:.4f})")

    keys = [
        "train_loss",
        "val_miou",
        "val_pos_miou",
        "val_dice",
        "val_pos_dice",
        "lr",
    ] + [f"iou/{c}" for c in CLASSES]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in keys])

    logger.info(f"完成, best mIoU={best_miou:.4f}, best.pt={best_pt}")
    return best_pt


def main():
    validate_required_paths()
    ckpt = train()
    logger.info(f"最终模型: {ckpt}")
    logger.info(f"日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
