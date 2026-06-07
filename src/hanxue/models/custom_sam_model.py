# 文件位置: PythonProject2/models/custom_sam_model.py
import torch
import torch.nn as nn

from .text_encoder import FreezeTextEncoder
from .image_encoder import PureImageEncoder
from .fusion_decoder import FiLMFusionDecoder


class CustomSAMWorldModel(nn.Module):
    """
    轻量 SAM 图像编码器 + Chinese-CLIP 文本编码器 + FiLM 融合解码器。

    默认 forward 兼容你原来的 train.py 调用:
        model(images, input_ids, attention_mask)

    同时也兼容 inference.py 中传入原图尺寸:
        model(images, input_ids, attention_mask, target_size=(H, W))
    """

    def __init__(
        self,
        image_dim=256,
        text_dim=768,
        freeze_text=True,
        freeze_image=False
    ):
        super().__init__()

        self.img_encoder = PureImageEncoder()
        self.txt_encoder = FreezeTextEncoder()
        self.decoder = FiLMFusionDecoder(image_dim=image_dim, text_dim=text_dim)

        if freeze_text:
            self.freeze_text_encoder()

        if freeze_image:
            self.freeze_image_encoder()

    def freeze_text_encoder(self):
        for p in self.txt_encoder.parameters():
            p.requires_grad = False

    def freeze_image_encoder(self):
        for p in self.img_encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _get_text_global_feature(text_encoder_output):
        """
        兼容不同 text_encoder.py 的返回形式。

        支持:
            1. (last_hidden_state, pooler_output)
            2. {"last_hidden_state": ..., "pooler_output": ...}
            3. transformers 模型输出对象
            4. 直接返回 [B, D] tensor
        """
        if torch.is_tensor(text_encoder_output):
            return text_encoder_output

        if isinstance(text_encoder_output, (list, tuple)):
            if len(text_encoder_output) >= 2:
                return text_encoder_output[1]
            if len(text_encoder_output) == 1:
                return text_encoder_output[0]

        if isinstance(text_encoder_output, dict):
            if "pooler_output" in text_encoder_output and text_encoder_output["pooler_output"] is not None:
                return text_encoder_output["pooler_output"]
            if "last_hidden_state" in text_encoder_output:
                return text_encoder_output["last_hidden_state"][:, 0, :]

        if hasattr(text_encoder_output, "pooler_output") and text_encoder_output.pooler_output is not None:
            return text_encoder_output.pooler_output

        if hasattr(text_encoder_output, "last_hidden_state"):
            return text_encoder_output.last_hidden_state[:, 0, :]

        raise RuntimeError(
            "无法从 text_encoder 输出中解析文本全局特征，"
            "请检查 models/text_encoder.py 的 forward 返回值。"
        )

    @staticmethod
    def _normalize_target_size(target_size, images):
        """
        如果没有指定 target_size，则默认输出到输入图像张量尺寸。
        """
        if target_size is None:
            return int(images.size(2)), int(images.size(3))

        if torch.is_tensor(target_size):
            target_size = target_size.detach().cpu().tolist()

        if isinstance(target_size, (list, tuple)):
            if len(target_size) == 2:
                return int(target_size[0]), int(target_size[1])
            if len(target_size) == 1 and isinstance(target_size[0], (list, tuple)):
                return int(target_size[0][0]), int(target_size[0][1])

        raise ValueError(f"target_size 格式错误，应为 (H, W)，当前为: {target_size}")

    def forward(self, images, input_ids, attention_mask, target_size=None):
        """
        参数:
            images:
                [B, 3, H, W]
            input_ids:
                [B, L]
            attention_mask:
                [B, L]
            target_size:
                可选，最终 mask 输出尺寸 (H_out, W_out)

        返回:
            pred_mask:
                [B, 1, H_out, W_out]，未 sigmoid 的 logits
        """
        if images.ndim != 4:
            raise ValueError(f"images 应为 [B, 3, H, W]，当前形状: {images.shape}")

        target_size = self._normalize_target_size(target_size, images)

        image_features = self.img_encoder(images)

        text_output = self.txt_encoder(input_ids, attention_mask)
        text_global_feat = self._get_text_global_feature(text_output)

        pred_mask = self.decoder(
            image_features=image_features,
            text_global_features=text_global_feat,
            target_size=target_size
        )

        return pred_mask