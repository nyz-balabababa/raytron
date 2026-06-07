# 文件位置: PythonProject2/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def align_logits_and_targets(logits, targets):
    """
    统一 logits 和 targets 的形状为 [B, 1, H, W]。
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    if logits.ndim != 4:
        raise ValueError(f"logits 应为 [B,1,H,W] 或 [B,H,W]，当前形状: {logits.shape}")

    if targets.ndim != 4:
        raise ValueError(f"targets 应为 [B,1,H,W] 或 [B,H,W]，当前形状: {targets.shape}")

    if logits.shape != targets.shape:
        raise ValueError(f"logits 和 targets 形状不一致: {logits.shape} vs {targets.shape}")

    targets = targets.float()
    targets = (targets > 0.5).float()

    return logits, targets


def prepare_sample_weight(sample_weight, logits):
    """
    将 sample_weight 整理成可广播到 [B,1,H,W] 的形状。
    """
    if sample_weight is None:
        return None

    if not torch.is_tensor(sample_weight):
        sample_weight = torch.tensor(sample_weight, dtype=logits.dtype, device=logits.device)

    sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)

    if sample_weight.ndim == 0:
        sample_weight = sample_weight.view(1)

    if sample_weight.ndim == 1:
        sample_weight = sample_weight.view(-1, 1, 1, 1)

    if sample_weight.ndim == 2:
        sample_weight = sample_weight.view(sample_weight.size(0), 1, 1, 1)

    return sample_weight


class FocalLoss(nn.Module):
    """
    二分类 Focal Loss。
    输入 logits，不需要提前 sigmoid。
    支持逐样本 sample_weight。
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, sample_weight=None):
        logits, targets = align_logits_and_targets(logits, targets)
        sample_weight = prepare_sample_weight(sample_weight, logits)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        loss = focal_weight * bce_loss

        if sample_weight is not None:
            loss = loss * sample_weight

        return loss.mean()


class DiceLoss(nn.Module):
    """
    二分类 Dice Loss。
    输入 logits，不需要提前 sigmoid。
    按 batch 内每张图分别计算，再按 sample_weight 加权。
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sample_weight=None):
        logits, targets = align_logits_and_targets(logits, targets)

        probs = torch.sigmoid(logits)

        batch_size = probs.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        if sample_weight is not None:
            if not torch.is_tensor(sample_weight):
                sample_weight = torch.tensor(sample_weight, dtype=dice_loss.dtype, device=dice_loss.device)

            sample_weight = sample_weight.to(device=dice_loss.device, dtype=dice_loss.dtype)

            if sample_weight.ndim > 1:
                sample_weight = sample_weight.view(sample_weight.size(0), -1).mean(dim=1)

            dice_loss = dice_loss * sample_weight
            return dice_loss.sum() / sample_weight.sum().clamp_min(1e-6)

        return dice_loss.mean()


class SemanticSegLoss(nn.Module):
    """
    混合损失函数：BCEWithLogits + Dice + Focal。

    logits:
        模型原始输出，不要 sigmoid。
    targets:
        二值 mask，形状 [B,1,H,W] 或 [B,H,W]。
    sample_weight:
        可选，形状 [B]。用于 SAM3 伪标签置信度加权。
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_weight=None
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([float(pos_weight)])
        else:
            pos_weight_tensor = None

        self.register_buffer("pos_weight_tensor", pos_weight_tensor)

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

    def forward(self, logits, targets, sample_weight=None):
        logits, targets = align_logits_and_targets(logits, targets)

        pos_weight = self.pos_weight_tensor
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none"
        )

        weight = prepare_sample_weight(sample_weight, logits)

        if weight is not None:
            loss_bce = (bce_per_pixel * weight).sum() / weight.expand_as(bce_per_pixel).sum().clamp_min(1e-6)
        else:
            loss_bce = bce_per_pixel.mean()

        loss_dice = self.dice_loss(logits, targets, sample_weight=sample_weight)
        loss_focal = self.focal_loss(logits, targets, sample_weight=sample_weight)

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.focal_weight * loss_focal
        )

        return total_loss