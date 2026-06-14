import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation

logger = logging.getLogger(__name__)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "model_state_dict", "state_dict", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            if any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
    raise RuntimeError("Unable to extract state_dict from checkpoint")


class TinyRefineHead(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, use_group_norm=True):
        super().__init__()
        norm1 = nn.GroupNorm(4, hidden_channels) if use_group_norm else nn.BatchNorm2d(hidden_channels)
        norm2 = nn.GroupNorm(4, hidden_channels) if use_group_norm else nn.BatchNorm2d(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            norm1,
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            norm2,
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class CLIPSegTinyRefine(nn.Module):
    def __init__(
        self,
        base_model_dir,
        base_checkpoint=None,
        img_size=768,
        residual_scale=1.0,
        use_edge_map=False,
    ):
        super().__init__()
        self.base_model_dir = str(base_model_dir)
        self.base_checkpoint = str(base_checkpoint) if base_checkpoint else None
        self.img_size = int(img_size)
        self.use_edge_map = bool(use_edge_map)
        self.base = CLIPSegForImageSegmentation.from_pretrained(str(base_model_dir))
        self.load_base_checkpoint(base_checkpoint)

        in_channels = 3 if self.use_edge_map else 2
        self.refine_head = TinyRefineHead(in_channels=in_channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)), requires_grad=False)

    def load_base_checkpoint(self, base_checkpoint):
        logger.info("CLIPSegTinyRefine base_model_dir=%s", self.base_model_dir)
        logger.info("CLIPSegTinyRefine base_checkpoint=%s", base_checkpoint)
        if not base_checkpoint:
            return

        checkpoint_path = Path(base_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"base checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        missing_keys, unexpected_keys = self.base.load_state_dict(state_dict, strict=False)
        logger.info("loaded checkpoint path=%s", checkpoint_path)
        logger.info("missing_keys=%s", missing_keys[:20])
        logger.info("unexpected_keys=%s", unexpected_keys[:20])

    def forward(self, pixel_values, input_ids, attention_mask, gray_image, edge_map=None):
        outputs = self.base(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        coarse_logits = outputs.logits
        if coarse_logits.ndim == 3:
            coarse_logits = coarse_logits.unsqueeze(1)
        coarse_logits = F.interpolate(
            coarse_logits,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        assert gray_image.ndim == 4 and gray_image.shape[1] == 1, f"gray_image shape invalid: {gray_image.shape}"
        assert coarse_logits.shape[-2:] == gray_image.shape[-2:], (
            f"shape mismatch coarse={coarse_logits.shape} gray={gray_image.shape}"
        )

        refine_inputs = [coarse_logits, gray_image]
        if self.use_edge_map:
            assert edge_map is not None, "edge_map is required when use_edge_map=True"
            refine_inputs.append(edge_map)
        residual_logits = self.refine_head(torch.cat(refine_inputs, dim=1))
        refined_logits = coarse_logits + self.residual_scale * residual_logits

        return {
            "coarse_logits": coarse_logits,
            "refined_logits": refined_logits,
            "residual_logits": residual_logits,
        }
