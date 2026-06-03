#!/usr/bin/env python3
"""
CLIPSeg 训练脚本 —— SAM3 蒸馏实验第一个模型
- 图像 + 文本 prompt → 单通道分割 mask
- 动态 prompt 增强 + 水平翻转 + warmup + 余弦退火
- 稀有类过采样 + 分级置信度过滤
- 断点续训 + 每 epoch 验证 + 实时 results.png（6 指标）
"""
import csv
import json
import logging
import math
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from config_clipseg import (
    MODEL_NAME, MODEL_DIR, CLASSES,
    PRED_JSON, VAL_PRED_JSON, TRAIN_LIST, VAL_LIST, IMAGE_ROOT,
    IMG_SIZE, BATCH, EPOCHS, DEVICE, WORKERS, SEED,
    DECODER_LR, BACKBONE_LR, WEIGHT_DECAY, LRF,
    WARMUP_EPOCHS, WARMUP_START_FACTOR,
    BCE_WEIGHT, DICE_WEIGHT, LOSS_WEIGHT_FLOOR,
    PROMPT_AUG_PROB, PROMPT_AUGMENTATIONS,
    HFLIP_PROB, GRAY2RGB,
    BLACKHOT_SKEW_THRESH, BLACKHOT_MEAN_THRESH,
    CLIP_MEAN, CLIP_STD, PROJECT, RUN_NAME,
    CONF_FILTER, RARE_OVERSAMPLE,
)

# ── 日志 ──
LOG_FILE = PROJECT / f"{RUN_NAME}.log"
PROJECT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 固定随机种子 ──
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════
# 1. RLE 解码
# ══════════════════════════════════════════════════════════════════════

def rle_to_mask(rle):
    try:
        from pycocotools import mask as maskUtils
        rle_cp = dict(rle)
        if isinstance(rle_cp["counts"], str):
            rle_cp["counts"] = rle_cp["counts"].encode("utf-8")
        return maskUtils.decode(rle_cp).astype(np.float32)
    except ImportError:
        pass
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes): counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.float32)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = val = 0
    for run_len in counts:
        if val == 1: mask[pos:pos + run_len] = 1
        pos += run_len; val = 1 - val
    return mask.reshape((h, w), order="F").astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════

class CLIPSegDataset(Dataset):

    def __init__(self, pred_json, image_list_path, augment_prompt=False, oversample=None):
        import hashlib
        with open(pred_json, encoding="utf-8") as f: preds = json.load(f)
        with open(image_list_path, encoding="utf-8") as f:
            allowed = {l.strip().replace("\\", "/") for l in f if l.strip()}

        self.samples = []
        self.augment_prompt = augment_prompt
        skipped_empty = skipped_low_conf = 0

        # 磁盘缓存目录：RLE 解码为 PNG 存盘，避免每 epoch 重复解码
        cache_dir = PROJECT / ".mask_cache" / pred_json.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  解码 RLE → PNG 缓存（{cache_dir}）...")
        for p in tqdm(preds, desc="  解码RLE"):
            img_path = p["image_path"].replace("\\", "/")
            if img_path not in allowed:
                alt = img_path[5:] if img_path.startswith("test/") else "test/" + img_path
                if alt not in allowed: continue
                img_path = alt
            for prompt, v in p["prompts"].items():
                if not v.get("hit") or prompt not in CLASSES: continue
                rle = v.get("rle")
                if rle is None: continue
                thresh = CONF_FILTER.get(prompt, 0.7) if isinstance(CONF_FILTER, dict) else CONF_FILTER
                if v.get("score", 1.0) < thresh: skipped_low_conf += 1; continue
                # 用 hash 生成唯一缓存文件名，避免中文路径问题
                cache_name = hashlib.md5(f"{img_path}|{prompt}".encode()).hexdigest()
                cache_path = cache_dir / f"{cache_name}.png"
                if not cache_path.exists():
                    mask = rle_to_mask(rle)
                    if mask.sum() == 0: skipped_empty += 1
                    cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                self.samples.append((img_path, prompt, cache_path, v.get("score", 1.0)))

        base = len(self.samples)
        oversampled = defaultdict(int)
        if oversample:
            extra = []
            for s in self.samples:
                mult = oversample.get(s[1], 1)
                for _ in range(mult - 1): extra.append(s); oversampled[s[1]] += 1
            self.samples.extend(extra)
        if oversampled:
            extra_str = ", ".join(f"{c}×{oversample[c]}" for c in oversampled)
            logger.info(f"  {pred_json.name}: {base} → {len(self.samples)} 样本 (过采样: {extra_str})")
        else:
            logger.info(f"  {pred_json.name}: {base} 样本 (无过采样)")
        logger.info(f"  跳过空掩码 {skipped_empty}, 低置信度 {skipped_low_conf}")

    def __len__(self): return len(self.samples)

    def _augment(self, prompt):
        opts = PROMPT_AUGMENTATIONS.get(prompt, [])
        return random.choice(opts) if opts else prompt

    @staticmethod
    def _unify_polarity(gray, fname):
        """黑热→白热统一：确保热辐射强的目标为亮区。"""
        # 文件名含 blackHot → 强制反色
        if "blackHot" in fname:
            return 255 - gray
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if std > 0:
            skew = float(np.mean(((gray - mean) / std) ** 3))
            if skew < BLACKHOT_SKEW_THRESH:
                return 255 - gray
        # 整体偏亮且非可见光图 → 兜底反色
        if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            return 255 - gray
        return gray

    def __getitem__(self, idx):
        img_path, orig_prompt, cache_path, score = self.samples[idx]
        prompt = self._augment(orig_prompt) if (self.augment_prompt and random.random() < PROMPT_AUG_PROB) else orig_prompt

        img = cv2.imread(str(IMAGE_ROOT / img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_bgr = cv2.imread(str(IMAGE_ROOT / img_path))
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr is not None else np.zeros((512, 640), dtype=np.uint8)

        # 反色统一：将黑热（热目标=暗）反转为白热（热目标=亮）
        img = self._unify_polarity(img, img_path)

        # 从 PNG 加载预解码的 mask（cv2.imread 是 C 实现，远快于 RLE 解码）
        mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        h, w = img.shape
        scale = IMG_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # pad 正方形
        ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)

        if GRAY2RGB: img = np.stack([img] * 3, axis=-1)
        else: img = img[:, :, None]

        # 水平翻转
        if random.random() < HFLIP_PROB:
            img = np.fliplr(img); mask = np.fliplr(mask)

        img = img.astype(np.float32) / 255.0

        # CLIP 标准化（与预训练时一致）
        mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(CLIP_STD,  dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std

        img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        return img, prompt, mask, orig_prompt, img_path, score


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    prompts = [b[1] for b in batch]
    masks = torch.stack([b[2] for b in batch], dim=0)
    orig_prompts = [b[3] for b in batch]
    img_paths = [b[4] for b in batch]
    weights = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return imgs, prompts, masks, orig_prompts, img_paths, weights


# ══════════════════════════════════════════════════════════════════════
# 3. Loss
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        return 1 - (2. * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)

    def per_sample(self, pred, target):
        """返回逐样本的 Dice loss，不取均值。"""
        return self.forward(pred, target)


# ══════════════════════════════════════════════════════════════════════
# 4. 指标
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    gt = gt_mask.float()
    intersection = (pred_bin * gt).sum().item()
    union = (pred_bin + gt).clamp(0, 1).sum().item()
    tp = intersection
    fp = (pred_bin * (1 - gt)).sum().item()
    fn = ((1 - pred_bin) * gt).sum().item()
    return {
        "iou": intersection / (union + 1e-7),
        "dice": 2 * intersection / (pred_bin.sum() + gt.sum() + 1e-7),
        "precision": tp / (tp + fp + 1e-7),
        "recall": tp / (tp + fn + 1e-7),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. 模型
# ══════════════════════════════════════════════════════════════════════

def load_model():
    if MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        logger.info(f"从本地加载: {MODEL_DIR}")
        processor = CLIPSegProcessor.from_pretrained(str(MODEL_DIR))
        model = CLIPSegForImageSegmentation.from_pretrained(str(MODEL_DIR))
    else:
        logger.info(f"从 HuggingFace 加载: {MODEL_NAME}")
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    decoder_params, vision_params = [], []
    for name, param in model.named_parameters():
        if "clip.visual" in name: vision_params.append(param)
        elif "clip.text" in name or "clip.logit_scale" in name: param.requires_grad = False
        else: decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": DECODER_LR},
        {"params": vision_params, "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")
    logger.info(f"  解码器 lr={DECODER_LR}, 视觉编码器 lr={BACKBONE_LR}, 文本编码器: 冻结")
    return model, processor, optimizer


# ══════════════════════════════════════════════════════════════════════
# 6. 训练一个 epoch
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, processor, loader, optimizer, dice_loss, scheduler, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for imgs, prompts, masks, _, _, weights in pbar:
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        weights = weights.to(DEVICE)
        text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)
        logits = model(pixel_values=imgs, input_ids=input_ids, attention_mask=attention_mask).logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        # 逐样本置信度加权（floor 保护稀有类不被过度降权）
        weights = weights.clamp(min=LOSS_WEIGHT_FLOOR)
        bce_per_sample = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=(1, 2, 3))
        loss_bce = (bce_per_sample * weights).mean()
        d_per_sample = dice_loss.per_sample(logits, masks)
        loss_dice = (d_per_sample * weights).mean()
        loss = BCE_WEIGHT * loss_bce + DICE_WEIGHT * loss_dice
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════
# 7. 验证
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, processor, loader):
    model.eval()
    all_metrics = defaultdict(list)
    vis_samples = []
    for imgs, prompts, masks, orig_prompts, _, _ in tqdm(loader, desc="  Val"):
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        logits = model(pixel_values=imgs, input_ids=text_inputs["input_ids"].to(DEVICE),
                       attention_mask=text_inputs["attention_mask"].to(DEVICE)).logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(len(imgs)):
            m = compute_metrics(logits[i:i+1], masks[i:i+1])
            for k, v in m.items(): all_metrics[f"{k}/{orig_prompts[i]}"].append(v)
            all_metrics["iou/overall"].append(m["iou"]); all_metrics["dice/overall"].append(m["dice"])
            if len(vis_samples) < 8:
                vis_samples.append({
                    "image": imgs[i].cpu(), "mask_gt": masks[i, 0].cpu(),
                    "mask_pred": (torch.sigmoid(logits[i, 0]) > 0.5).float().cpu(),
                    "prompt": prompts[i],
                })
    summary = {k: np.mean(v) for k, v in all_metrics.items()}
    return summary, vis_samples


# ══════════════════════════════════════════════════════════════════════
# 8. 绘图（6 指标）
# ══════════════════════════════════════════════════════════════════════

def plot_results(history, vis_samples, save_path, epoch):
    fig = plt.figure(figsize=(16, 10))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs_range, history["train_loss"], "b-", linewidth=1)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("1. Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs_range, history["val_miou"], "g-", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU"); ax2.set_title("2. Val mIoU")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(epochs_range, history["val_dice"], "m-", linewidth=1.5)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Dice"); ax3.set_title("3. Val Dice")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    for cls_name in CLASSES:
        k = f"iou/{cls_name}"
        if k in history and history[k]:
            ax4.plot(epochs_range, history[k], "-", label=cls_name, linewidth=1, alpha=0.8)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("IoU"); ax4.set_title("4. Per-Class IoU")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    if history.get("lr"):
        ax5.plot(epochs_range, history["lr"], "r-", linewidth=1)
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("LR"); ax5.set_title("5. Learning Rate")
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax6 = fig.add_subplot(2, 3, 6)
    if vis_samples:
        s = vis_samples[0]
        img_np = s["image"].permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 3: img_np = img_np[:, :, 0]
        h, w = img_np.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3): overlay[:, :, c] = np.clip(img_np, 0, 1)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + s["mask_gt"].numpy() * 0.4, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + s["mask_pred"].numpy() * 0.5, 0, 1)
        ax6.imshow(overlay)
        ax6.set_title(f"6. {s['prompt'][:40]}\nRed=GT Cyan=Pred", fontsize=8)
        ax6.axis("off")

    plt.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 9. 主训练
# ══════════════════════════════════════════════════════════════════════

def train():
    run_dir = PROJECT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    best_pt = run_dir / "best.pt"; last_pt = run_dir / "last.pt"
    results_png = run_dir / "results.png"

    model, processor, optimizer = load_model()

    # DataLoader 必须在 scheduler 之前创建（scheduler 依赖 len(train_loader)）
    train_ds = CLIPSegDataset(PRED_JSON, TRAIN_LIST, augment_prompt=True, oversample=RARE_OVERSAMPLE)
    val_ds = CLIPSegDataset(VAL_PRED_JSON, VAL_LIST, augment_prompt=False, oversample=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
                            collate_fn=collate_fn, pin_memory=True)
    logger.info(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    dice_loss = DiceLoss()

    # warmup + cosine scheduler（per-step）
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    total_steps = EPOCHS * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return WARMUP_START_FACTOR + (1 - WARMUP_START_FACTOR) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LRF + (1 - LRF) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 断点续训
    start_epoch = 0; history = defaultdict(list); best_miou = -1
    resume_ckpt = None
    if last_pt.exists(): resume_ckpt = last_pt
    elif best_pt.exists(): resume_ckpt = best_pt

    if resume_ckpt:
        logger.info(f"检测到 checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info("  已恢复 scheduler 状态（warmup/cosine 衔接正确）")
        else:
            # 旧 checkpoint 无 scheduler，手动推进
            start_epoch_ckpt = ckpt.get("epoch", 0)
            for _ in range(start_epoch_ckpt * len(train_loader)): scheduler.step()
            logger.info("  旧 checkpoint 无 scheduler，手动推进 step")
        start_epoch = ckpt.get("epoch", 0); best_miou = ckpt.get("best_miou", -1)
        for k, v in ckpt.get("history", {}).items(): history[k] = list(v)
        if start_epoch >= EPOCHS:
            logger.info(f"已完成 ({start_epoch}/{EPOCHS})，跳过"); return best_pt
        logger.info(f"从 epoch {start_epoch + 1}/{EPOCHS} 续训, best_mIoU={best_miou:.4f}")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        logger.info(f"{'─' * 50}\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_loss = train_epoch(model, processor, train_loader, optimizer,
                                 dice_loss, scheduler, epoch, EPOCHS)
        val_metrics, vis_samples = validate(model, processor, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_miou"].append(val_metrics.get("iou/overall", 0))
        history["val_dice"].append(val_metrics.get("dice/overall", 0))
        history["lr"].append(current_lr)
        for c in CLASSES: history[f"iou/{c}"].append(val_metrics.get(f"iou/{c}", 0))

        elapsed = time.time() - t0
        logger.info(f"  Loss={train_loss:.4f}  mIoU={val_metrics.get('iou/overall',0):.4f}  "
                    f"Dice={val_metrics.get('dice/overall',0):.4f}  LR={current_lr:.2e}  {elapsed/60:.1f}min")
        cls_str = "  ".join(f"{c}={val_metrics.get(f'iou/{c}', 0):.3f}" for c in CLASSES)
        logger.info(f"  Per-class IoU: {cls_str}")

        plot_results(history, vis_samples, results_png, epoch)

        save_dict = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "history": dict(history), "best_miou": best_miou}
        torch.save(save_dict, last_pt)

        current_miou = val_metrics.get("iou/overall", 0)
        if current_miou > best_miou:
            best_miou = current_miou; save_dict["best_miou"] = best_miou
            torch.save(save_dict, best_pt)
            logger.info(f"  ★ 新最佳模型 (mIoU={best_miou:.4f})")

    csv_path = run_dir / "results.csv"
    keys = ["train_loss", "val_miou", "val_dice", "lr"] + [f"iou/{c}" for c in CLASSES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch"] + keys)
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] if i < len(history[k]) else "" for k in keys])
    logger.info(f"完成, best mIoU={best_miou:.4f}, best.pt={best_pt}")
    return best_pt


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA"); sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)
    logger.info("CLIPSeg 训练 v2")
    logger.info(f"  模型: {MODEL_NAME}  分辨率: {IMG_SIZE}  Batch: {BATCH}  Epochs: {EPOCHS}")
    logger.info(f"  解码器 lr: {DECODER_LR}  视觉编码器 lr: {BACKBONE_LR}")
    logger.info(f"  Warmup: {WARMUP_EPOCHS}ep  HFlip: {HFLIP_PROB}  PromptAug: {PROMPT_AUG_PROB}")
    logger.info(f"  类别: {CLASSES}  过采样: {RARE_OVERSAMPLE}")
    logger.info("=" * 60)

    ckpt = train()
    logger.info(f"最终模型: {ckpt}")
    logger.info(f"日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
