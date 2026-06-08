# 文件位置: PythonProject2/models/fusion_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMFusionDecoder(nn.Module):
    """
    文本引导的 FiLM 融合解码器。

    输入:
        image_features: [B, C, H, W]
        text_global_features: [B, D]
        target_size: 输出 mask 目标尺寸，格式为 (H_out, W_out)

    输出:
        pred_mask: [B, 1, H_out, W_out]，未经过 sigmoid 的 logits
    """

    def __init__(self, image_dim=256, text_dim=768):
        super().__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim

        # 文本特征映射为 FiLM 的 gamma 和 beta
        self.film_projection = nn.Linear(text_dim, image_dim * 2)
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

        # 保留你原来的层名，避免已有 checkpoint 因层名变化而加载失败
        self.conv_after_fusion = nn.Sequential(
            nn.Conv2d(image_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    @staticmethod
    def _normalize_target_size(target_size):
        """
        将 target_size 统一整理成 (H, W)。
        """
        if target_size is None:
            return None

        if torch.is_tensor(target_size):
            target_size = target_size.detach().cpu().tolist()

        if isinstance(target_size, (list, tuple)):
            if len(target_size) == 2:
                return int(target_size[0]), int(target_size[1])

            # 兼容 [[H, W]] 这种情况
            if len(target_size) == 1 and isinstance(target_size[0], (list, tuple)):
                return int(target_size[0][0]), int(target_size[0][1])

        raise ValueError(f"target_size 格式错误，应为 (H, W)，当前为: {target_size}")

    def forward(self, image_features, text_global_features, target_size):
        if image_features.ndim != 4:
            raise ValueError(f"image_features 应为 [B, C, H, W]，当前形状: {image_features.shape}")

        if text_global_features.ndim == 3:
            # 兼容 [B, L, D]，默认取 CLS token
            text_global_features = text_global_features[:, 0, :]

        if text_global_features.ndim != 2:
            raise ValueError(
                f"text_global_features 应为 [B, D]，当前形状: {text_global_features.shape}"
            )

        if image_features.size(0) != text_global_features.size(0):
            raise ValueError(
                f"图像 batch 和文本 batch 不一致: "
                f"{image_features.size(0)} vs {text_global_features.size(0)}"
            )

        if image_features.size(1) != self.image_dim:
            raise ValueError(
                f"image_features 通道数应为 {self.image_dim}，"
                f"当前为 {image_features.size(1)}。请检查 image_encoder.py 的输出通道。"
            )

        target_size = self._normalize_target_size(target_size)
        if target_size is None:
            target_size = image_features.shape[-2:]

        film_params = self.film_projection(text_global_features)
        gamma, beta = torch.split(film_params, image_features.size(1), dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        fused_features = image_features * (1.0 + gamma) + beta

        x = self.conv_after_fusion(fused_features)
        x = self.mask_head(x)

        pred_mask = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        return pred_mask
