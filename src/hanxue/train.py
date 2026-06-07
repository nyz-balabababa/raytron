import csv
import logging
import math
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config_hanxue import (
    APPLY_SCORE_FILTER,
    BATCH,
    BCE_WEIGHT,
    CLASSES,
    CONF_FILTER,
    DECODER_LR,
    DEVICE,
    DICE_WEIGHT,
    EPOCHS,
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


def build_optimizer(model, decoder_lr, image_lr, weight_decay):
    image_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_encoder"):
            image_params.append(param)
        elif name.startswith("decoder"):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if image_params and image_lr > 0:
        param_groups.append({"params": image_params, "lr": image_lr, "weight_decay": weight_decay, "name": "img_encoder"})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder"})
    if other_params:
        param_groups.append({"params": other_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "other"})

    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查模型冻结设置。")

    logger.info("[Optimizer] 参数组:")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"]) / 1e6
        logger.info(f"  {group['name']}: lr={group['lr']}, params={n_params:.2f}M")

    return optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in param_groups], weight_decay=weight_decay)


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


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, masks, sample_weight=sample_weight)

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
        valid_batches += 1

        if batch_idx % 10 == 0:
            lr_str = ", ".join([f"{group['lr']:.2e}" for group in optimizer.param_groups])
            logger.info(
                f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(loader)}] "
                f"Loss={loss.item():.4f} LR={lr_str}"
            )

    if valid_batches == 0:
        raise RuntimeError(f"Epoch {epoch} 没有任何有效 batch。")
    return total_loss / valid_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_metrics = defaultdict(list)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompts = batch["prompt"]

        logits = model(images, input_ids, attention_mask)
        for i in range(images.size(0)):
            threshold = VAL_THRESHOLDS.get(prompts[i], 0.5)
            metrics = compute_metrics(logits[i : i + 1], masks[i : i + 1], threshold=threshold)
            for key, value in metrics.items():
                all_metrics[f"{key}/{prompts[i]}"].append(value)
            all_metrics["iou/overall"].append(metrics["iou"])
            all_metrics["dice/overall"].append(metrics["dice"])

    return {key: float(np.mean(values)) for key, values in all_metrics.items()}


def build_datasets():
    dataset_kwargs = dict(
        images_root=IMAGE_ROOT,
        tokenizer_dir=TOKENIZER_DIR,
        img_size=(IMG_SIZE, IMG_SIZE),
        augment_prompt=True,
        hflip_prob=HFLIP_PROB,
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


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_miou):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history": dict(history),
            "best_miou": best_miou,
        },
        path,
    )


def train():
    set_seed(SEED)
    device = torch.device(f"cuda:0" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    run_dir = PROJECT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    best_pt = run_dir / "best.pt"
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
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    optimizer = build_optimizer(model, DECODER_LR, IMAGE_LR, WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, MIN_LR_RATIO)
    criterion = SemanticSegLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT, focal_weight=FOCAL_WEIGHT)

    history = defaultdict(list)
    best_miou = -1.0
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
        start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, EPOCHS + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_metrics = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_miou"].append(val_metrics.get("iou/overall", 0.0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
        history["lr"].append(current_lr)
        for cls_name in CLASSES:
            history[f"iou/{cls_name}"].append(val_metrics.get(f"iou/{cls_name}", 0.0))

        logger.info(
            f"Loss={train_loss:.4f}  mIoU={val_metrics.get('iou/overall', 0.0):.4f}  "
            f"Dice={val_metrics.get('dice/overall', 0.0):.4f}  LR={current_lr:.2e}  {elapsed/60:.1f}min"
        )
        logger.info("Per-class IoU: " + "  ".join(f"{c}={val_metrics.get(f'iou/{c}', 0.0):.3f}" for c in CLASSES))

        save_checkpoint(last_pt, epoch, model, optimizer, scheduler, history, best_miou)

        current_miou = val_metrics.get("iou/overall", 0.0)
        if current_miou > best_miou:
            best_miou = current_miou
            save_checkpoint(best_pt, epoch, model, optimizer, scheduler, history, best_miou)
            logger.info(f"★ 新最佳模型 (mIoU={best_miou:.4f})")

    keys = ["train_loss", "val_miou", "val_dice", "lr"] + [f"iou/{c}" for c in CLASSES]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in keys])

    logger.info(f"完成, best mIoU={best_miou:.4f}, best.pt={best_pt}")
    return best_pt


def main():
    ckpt = train()
    logger.info(f"最终模型: {ckpt}")
    logger.info(f"日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
