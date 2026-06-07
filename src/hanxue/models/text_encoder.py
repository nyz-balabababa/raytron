# 文件位置: PythonProject2/models/text_encoder.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))

import torch
import torch.nn as nn
from transformers import ChineseCLIPTextModel



def resolve_model_dir(model_dir: str) -> str:
    """
    兼容相对路径和绝对路径。
    默认模型目录: weights/chinese_clip
    """
    p = Path(model_dir)

    if p.is_absolute():
        return str(p)

    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return str(cwd_path)

    root_path = PROJECT_ROOT / p
    return str(root_path)


def strip_prefix_from_text_state_dict(ckpt):
    """
    从完整 Chinese-CLIP checkpoint 中提取文本端权重。
    常见 key:
        text_model.embeddings.word_embeddings.weight
    需要转成:
        embeddings.word_embeddings.weight
    """
    text_state_dict = {}

    for key, value in ckpt.items():
        if key.startswith("text_model."):
            new_key = key[len("text_model."):]
            text_state_dict[new_key] = value

    return text_state_dict


class FreezeTextEncoder(nn.Module):
    """
    冻结的 Chinese-CLIP 文本编码器。

    返回:
        last_hidden_state: [B, L, D]
        pooler_output: [B, D]
    """

    def __init__(self, model_dir="weights/chinese_clip"):
        super().__init__()

        model_dir = resolve_model_dir(model_dir)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"找不到 Chinese-CLIP 模型目录: {model_dir}")

        print(f"[Chinese-CLIP] 正在加载文本编码器: {model_dir}")

        # 先构建文本模型骨架。
        # ignore_mismatched_sizes=True 用于兼容 vocab size 等不一致问题。
        self.text_model = ChineseCLIPTextModel.from_pretrained(
            model_dir,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

        bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(bin_path):
            ckpt = torch.load(bin_path, map_location="cpu")

            text_state_dict = strip_prefix_from_text_state_dict(ckpt)

            if len(text_state_dict) == 0:
                print("[Chinese-CLIP 警告] 没有找到 text_model. 前缀的权重，将使用 from_pretrained 自动加载结果。")
            else:
                # 根据真实词表大小修正 embedding，避免 21128 vs 30522 之类的问题。
                word_key = "embeddings.word_embeddings.weight"
                if word_key in text_state_dict:
                    real_vocab_size = text_state_dict[word_key].shape[0]
                    self.text_model.resize_token_embeddings(real_vocab_size)
                    print(f"[Chinese-CLIP] 已根据真实权重调整 vocab_size = {real_vocab_size}")

                missing_keys, unexpected_keys = self.text_model.load_state_dict(
                    text_state_dict,
                    strict=False
                )

                print("✅ 已手动加载 Chinese-CLIP 文本端真实权重。")

                if len(missing_keys) > 0:
                    print(f"[Chinese-CLIP 警告] missing_keys 数量: {len(missing_keys)}")
                    for k in missing_keys[:30]:
                        print("  MISSING:", k)

                if len(unexpected_keys) > 0:
                    print(f"[Chinese-CLIP 警告] unexpected_keys 数量: {len(unexpected_keys)}")
                    for k in unexpected_keys[:30]:
                        print("  UNEXPECTED:", k)
        else:
            print(f"⚠️ 未找到 {bin_path}，将只使用 from_pretrained 的加载结果。")

        # 冻结所有文本参数
        for param in self.text_model.parameters():
            param.requires_grad = False

        # 冻结模型必须保持 eval，否则 dropout 仍可能在训练阶段生效。
        self.text_model.eval()

    def train(self, mode: bool = True):
        """
        重写 train，确保外部调用 model.train() 时，文本编码器仍保持 eval。
        """
        super().train(False)
        self.text_model.eval()
        return self

    def forward(self, input_ids, attention_mask):
        self.text_model.eval()

        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            last_hidden_state = outputs.last_hidden_state
            pooler_output = outputs.pooler_output

            # 保险：如果 pooler_output 为空，则使用 CLS token。
            if pooler_output is None:
                pooler_output = last_hidden_state[:, 0, :]

        return last_hidden_state, pooler_output
