# 文件位置: PythonProject2/models/image_encoder.py
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# 确保可以导入项目根目录下的 efficient_sam 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


def resolve_weight_path(checkpoint_path: str) -> str:
    """
    兼容相对路径和绝对路径。
    默认权重位置: weights/efficient_sam/efficient_sam_vitt.pt
    """
    p = Path(checkpoint_path)

    if p.is_absolute():
        return str(p)

    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return str(cwd_path)

    root_path = PROJECT_ROOT / p
    return str(root_path)


def extract_state_dict(checkpoint):
    """
    兼容不同 EfficientSAM checkpoint 包装方式。
    """
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # checkpoint 本身就是 state_dict
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise RuntimeError("无法从 EfficientSAM checkpoint 中解析 state_dict。")


def strip_module_prefix(state_dict):
    """
    兼容 DataParallel 保存的 module.xxx。
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


class PureImageEncoder(nn.Module):
    """
    只保留 EfficientSAM 的 image_encoder。
    输出应为 [B, 256, H_feat, W_feat]。
    """

    def __init__(
        self,
        checkpoint_path="weights/efficient_sam/efficient_sam_vitt.pt",
        freeze=False
    ):
        super().__init__()

        checkpoint_path = resolve_weight_path(checkpoint_path)

        full_sam_model = build_efficient_sam_vitt()

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = extract_state_dict(checkpoint)
            state_dict = strip_module_prefix(state_dict)

            missing_keys, unexpected_keys = full_sam_model.load_state_dict(
                state_dict,
                strict=False
            )

            print(f"✅ 已加载 EfficientSAM 权重: {checkpoint_path}")

            if len(missing_keys) > 0:
                print(f"[EfficientSAM 警告] missing_keys 数量: {len(missing_keys)}")
                for k in missing_keys[:20]:
                    print("  MISSING:", k)

            if len(unexpected_keys) > 0:
                print(f"[EfficientSAM 警告] unexpected_keys 数量: {len(unexpected_keys)}")
                for k in unexpected_keys[:20]:
                    print("  UNEXPECTED:", k)
        else:
            print(f"⚠️ 未找到 EfficientSAM 权重: {checkpoint_path}")
            print("⚠️ 将使用随机初始化的 EfficientSAM image_encoder。")

        self.image_encoder = full_sam_model.image_encoder
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        del full_sam_model

        if freeze:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        参数:
            x: [B, 3, H, W]，float tensor，值域通常为 0~1

        返回:
            image_feats: [B, C, H_feat, W_feat]
        """
        if x.ndim != 4:
            raise ValueError(f"输入图像应为 [B, 3, H, W]，当前形状: {x.shape}")

        if x.size(1) != 3:
            raise ValueError(f"输入图像通道数应为 3，当前为: {x.size(1)}")

        x = (x - self.pixel_mean.to(device=x.device, dtype=x.dtype)) / self.pixel_std.to(
            device=x.device, dtype=x.dtype
        )

        image_feats = self.image_encoder(x)

        # 兼容某些 encoder 返回 tuple/list 的情况
        if isinstance(image_feats, (tuple, list)):
            image_feats = image_feats[0]

        if not torch.is_tensor(image_feats):
            raise TypeError(f"image_encoder 输出不是 tensor，当前类型: {type(image_feats)}")

        if image_feats.ndim != 4:
            raise ValueError(
                f"image_encoder 输出应为 [B, C, H, W]，当前形状: {image_feats.shape}"
            )

        return image_feats
