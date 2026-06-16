import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from common_esam_cclip_11_urf import (
    ESAMCCLIPModel as _BaseESAMCCLIPModel,
    count_parameters,
    load_state_dict_flexible,
)


LOGGER = logging.getLogger("ESAM_CCLIP_11_URF_MODEL")

KEY_PREFIX_REMAP = {
    "image_encoder.": "img_encoder.",
    "text_encoder.": "txt_encoder.",
    "fusion_decoder.": "decoder.",
}

PRESET_TRAINABLE_CONFIGS = {
    "decoder_only": {
        "train_decoder": True,
        "train_refine_head": False,
        "unfreeze_image_mode": "none",
    },
    "refine_only": {
        "train_decoder": False,
        "train_refine_head": True,
        "unfreeze_image_mode": "none",
    },
    "decoder_refine": {
        "train_decoder": True,
        "train_refine_head": True,
        "unfreeze_image_mode": "none",
    },
    "balanced_partial_unfreeze": {
        "train_decoder": True,
        "train_refine_head": False,
        "unfreeze_image_mode": "last_norm",
    },
    "balanced_partial_unfreeze_refine": {
        "train_decoder": True,
        "train_refine_head": True,
        "unfreeze_image_mode": "last_norm",
    },
    "balanced_partial_unfreeze_last1_refine": {
        "train_decoder": True,
        "train_refine_head": True,
        "unfreeze_image_mode": "last1",
    },
}


class ZeroInitResidualRefineHead(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 4)
        self.conv1 = torch.nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1)
        self.act1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = torch.nn.ReLU(inplace=True)
        self.out = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, coarse_logit_up: torch.Tensor, gray_image: torch.Tensor) -> torch.Tensor:
        if coarse_logit_up.dim() != 4 or gray_image.dim() != 4:
            raise RuntimeError(
                "refine head expects BCHW tensors, "
                f"got coarse={tuple(coarse_logit_up.shape)} gray={tuple(gray_image.shape)}"
            )
        refine_input = torch.cat([coarse_logit_up, gray_image], dim=1)
        hidden = self.act1(self.conv1(refine_input))
        hidden = self.act2(self.conv2(hidden))
        return self.out(hidden)


class ESAMCCLIPModel(_BaseESAMCCLIPModel):
    def __init__(
        self,
        tokenizer_dir,
        efficient_sam_ckpt,
        freeze_image: bool = True,
        freeze_text: bool = True,
        use_refine_head: bool = False,
        refine_head_hidden_dim: int = 16,
    ):
        super().__init__(
            tokenizer_dir=tokenizer_dir,
            efficient_sam_ckpt=efficient_sam_ckpt,
            freeze_image=freeze_image,
            freeze_text=freeze_text,
            use_refine_head=False,
            refine_head_hidden_dim=refine_head_hidden_dim,
        )
        self.refine_head = ZeroInitResidualRefineHead(hidden_dim=refine_head_hidden_dim)
        self.use_refine_head = bool(use_refine_head)
        self.refine_head_hidden_dim = int(refine_head_hidden_dim)
        self._last_decode_debug: Dict[str, Any] = {}

    def _prepare_refine_inputs(
        self,
        coarse_logits: torch.Tensor,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_logit_up = F.interpolate(
            coarse_logits,
            size=tuple(images.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        gray_image = images.mean(dim=1, keepdim=True)
        gray_image = F.interpolate(
            gray_image,
            size=tuple(coarse_logit_up.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        gray_min = gray_image.amin(dim=(-2, -1), keepdim=True)
        gray_max = gray_image.amax(dim=(-2, -1), keepdim=True)
        gray_image = (gray_image - gray_min) / (gray_max - gray_min).clamp_min(1e-6)
        return coarse_logit_up, gray_image

    def set_refine_head_enabled(self, enabled: bool) -> None:
        self.use_refine_head = bool(enabled)

    def decode(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        target_size,
        images: Optional[torch.Tensor] = None,
        use_refine_head: Optional[bool] = None,
        return_debug: bool = False,
    ):
        coarse_logits = self.decoder(
            image_features=image_features,
            text_global_features=text_features,
            target_size=target_size,
        )
        refine_enabled = self.use_refine_head if use_refine_head is None else bool(use_refine_head)
        if not refine_enabled:
            debug_payload = {
                "refine_enabled": False,
                "coarse_logit": coarse_logits.detach(),
                "residual_logit": None,
                "final_logit": coarse_logits.detach(),
            }
            self._last_decode_debug = debug_payload
            if return_debug:
                return coarse_logits, debug_payload
            return coarse_logits
        if images is None:
            raise RuntimeError("use_refine_head=True 时必须向 decode 传入原图 images。")

        coarse_logit_up, gray_image = self._prepare_refine_inputs(coarse_logits, images)
        residual_logits = self.refine_head(coarse_logit_up=coarse_logit_up, gray_image=gray_image)
        final_logits = coarse_logit_up + residual_logits
        debug_payload = {
            "refine_enabled": True,
            "coarse_logit": coarse_logit_up.detach(),
            "residual_logit": residual_logits.detach(),
            "final_logit": final_logits.detach(),
            "gray_image": gray_image.detach(),
        }
        self._last_decode_debug = debug_payload
        if return_debug:
            return final_logits, debug_payload
        return final_logits

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_size,
        return_debug: bool = False,
    ):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        return self.decode(
            image_features=image_features,
            text_features=text_features,
            target_size=target_size,
            images=images,
            return_debug=return_debug,
        )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model_from_config(cfg: Any) -> ESAMCCLIPModel:
    tokenizer_dir = _cfg_get(cfg, "tokenizer_dir")
    efficient_sam_ckpt = _cfg_get(cfg, "efficient_sam_ckpt")
    if tokenizer_dir is None or efficient_sam_ckpt is None:
        raise RuntimeError("build_model_from_config 需要 tokenizer_dir 和 efficient_sam_ckpt。")
    freeze_image = bool(_cfg_get(cfg, "freeze_image", _cfg_get(cfg, "freeze_image_encoder", True)))
    freeze_text = bool(_cfg_get(cfg, "freeze_text", _cfg_get(cfg, "freeze_text_encoder", True)))
    use_refine_head = bool(_cfg_get(cfg, "use_refine_head", False))
    refine_head_hidden_dim = int(_cfg_get(cfg, "refine_head_hidden_dim", 16))
    return ESAMCCLIPModel(
        tokenizer_dir=tokenizer_dir,
        efficient_sam_ckpt=efficient_sam_ckpt,
        freeze_image=freeze_image,
        freeze_text=freeze_text,
        use_refine_head=use_refine_head,
        refine_head_hidden_dim=refine_head_hidden_dim,
    )


def _strip_common_prefixes(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    return key


def _remap_key(key: str) -> str:
    key = _strip_common_prefixes(key)
    for old_prefix, new_prefix in KEY_PREFIX_REMAP.items():
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def _prepare_matched_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str], List[str], List[str]]:
    model_state = model.state_dict()
    matched_state: Dict[str, torch.Tensor] = {}
    matched_keys: List[str] = []
    decoder_matched_keys: List[str] = []
    refine_head_matched_keys: List[str] = []
    unexpected_keys: List[str] = []
    shape_mismatch_keys: List[str] = []

    for raw_key, value in state_dict.items():
        mapped_key = _remap_key(str(raw_key))
        if mapped_key not in model_state:
            unexpected_keys.append(mapped_key)
            continue
        if model_state[mapped_key].shape != value.shape:
            shape_mismatch_keys.append(
                f"{mapped_key}: ckpt={tuple(value.shape)} model={tuple(model_state[mapped_key].shape)}"
            )
            continue
        matched_state[mapped_key] = value
        matched_keys.append(mapped_key)
        if mapped_key.startswith("decoder."):
            decoder_matched_keys.append(mapped_key)
        elif mapped_key.startswith("refine_head."):
            refine_head_matched_keys.append(mapped_key)

    missing_keys = [key for key in model_state.keys() if key not in matched_state]
    unexpected_keys.extend(shape_mismatch_keys)
    return (
        matched_state,
        matched_keys,
        decoder_matched_keys,
        refine_head_matched_keys,
        missing_keys,
        unexpected_keys,
    )


def load_checkpoint_safely(
    model: torch.nn.Module,
    ckpt_path,
    device=None,
    strict: bool = False,
    require_decoder_match: bool = True,
):
    checkpoint_path = Path(ckpt_path)
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = load_state_dict_flexible(checkpoint)
    matched_state, matched_keys, decoder_matched_keys, refine_head_matched_keys, missing_keys, unexpected_keys = _prepare_matched_state_dict(
        model,
        state_dict,
    )
    shape_mismatch_keys = [item for item in unexpected_keys if ": ckpt=" in item]
    if require_decoder_match and len(decoder_matched_keys) == 0:
        raise RuntimeError("checkpoint 未命中任何 decoder 参数，拒绝热启动。")
    if strict:
        blocking_missing = [key for key in missing_keys if not key.startswith("refine_head.")]
        if blocking_missing or unexpected_keys:
            raise RuntimeError(
                "strict=True 加载失败: "
                f"missing_non_refine={blocking_missing[:20]} unexpected={unexpected_keys[:20]}"
            )
    model.load_state_dict(matched_state, strict=False)
    load_info = {
        "checkpoint_path": str(checkpoint_path),
        "matched_keys": matched_keys,
        "decoder_matched_keys": decoder_matched_keys,
        "refine_head_matched_keys": refine_head_matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
        "checkpoint_use_refine_head": bool(checkpoint.get("use_refine_head", False)),
    }
    model._last_checkpoint_load_info = load_info
    LOGGER.info("checkpoint loaded: %s", checkpoint_path)
    LOGGER.info(
        "matched=%d decoder=%d refine=%d missing=%d unexpected=%d strict=%s",
        len(matched_keys),
        len(decoder_matched_keys),
        len(refine_head_matched_keys),
        len(missing_keys),
        len(unexpected_keys),
        strict,
    )
    if missing_keys:
        LOGGER.info("missing_keys_preview=%s", missing_keys[:20])
    if unexpected_keys:
        LOGGER.info("unexpected_keys_preview=%s", unexpected_keys[:20])
    return checkpoint, missing_keys, unexpected_keys


def load_checkpoint_flexible(model, checkpoint_path, device, require_decoder_match=True):
    return load_checkpoint_safely(
        model=model,
        ckpt_path=checkpoint_path,
        device=device,
        strict=False,
        require_decoder_match=require_decoder_match,
    )


def _extract_last_block_prefixes(param_names: List[str], block_token: str) -> List[str]:
    block_indices = []
    for name in param_names:
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == block_token and idx + 1 < len(parts):
                try:
                    block_idx = int(parts[idx + 1])
                except ValueError:
                    continue
                block_indices.append((block_idx, ".".join(parts[: idx + 2])))
    unique = {}
    for block_idx, prefix in block_indices:
        unique[block_idx] = prefix
    return [unique[idx] for idx in sorted(unique.keys())]


def _summarize_trainable_params(model: torch.nn.Module) -> Dict[str, Any]:
    total_params, trainable_params = count_parameters(model)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    summary = {
        "total_params_m": float(total_params),
        "trainable_params_m": float(trainable_params),
        "image_trainable_params_m": sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("img_encoder")
        ) / 1e6,
        "decoder_trainable_params_m": sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("decoder")
        ) / 1e6,
        "refine_trainable_params_m": sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("refine_head")
        ) / 1e6,
        "text_trainable_params_m": sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("txt_encoder")
        ) / 1e6,
        "trainable_names": trainable_names,
    }
    LOGGER.info("trainable total params=%.2fM / %.2fM", summary["trainable_params_m"], summary["total_params_m"])
    LOGGER.info(
        "trainable decoder=%.2fM refine=%.2fM image=%.2fM text=%.2fM",
        summary["decoder_trainable_params_m"],
        summary["refine_trainable_params_m"],
        summary["image_trainable_params_m"],
        summary["text_trainable_params_m"],
    )
    LOGGER.info("trainable parameter names preview(first 80): %s", trainable_names[:80])
    return summary


def set_trainable_modules(model: torch.nn.Module, preset) -> Dict[str, Any]:
    if isinstance(preset, str):
        cfg = dict(PRESET_TRAINABLE_CONFIGS.get(preset, {}))
        if not cfg:
            raise RuntimeError(f"未知 trainable preset: {preset}")
    elif isinstance(preset, dict):
        cfg = dict(preset)
    else:
        cfg = {
            "train_decoder": bool(getattr(preset, "train_decoder_only", True)),
            "train_refine_head": bool(getattr(preset, "train_refine_head", False)),
            "unfreeze_image_mode": str(getattr(preset, "unfreeze_image_mode", "none")),
        }

    train_decoder = bool(cfg.get("train_decoder", True))
    train_refine_head = bool(cfg.get("train_refine_head", False))
    unfreeze_image_mode = str(cfg.get("unfreeze_image_mode", "none"))

    for param in model.txt_encoder.parameters():
        param.requires_grad = False
    for param in model.img_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = train_decoder
    for param in model.refine_head.parameters():
        param.requires_grad = train_refine_head

    warnings: List[str] = []
    matched_names: List[str] = []
    if unfreeze_image_mode != "none":
        named_params = list(model.img_encoder.named_parameters())
        all_names = [name for name, _ in named_params]

        def enable_by_predicate(predicate):
            local_matched = []
            for name, param in named_params:
                if predicate(name):
                    param.requires_grad = True
                    local_matched.append(name)
            return local_matched

        if unfreeze_image_mode == "last_norm":
            keywords = ("norm", "neck", "adapter", "output", "proj")
            keyword_matches = [name for name in all_names if any(keyword in name.lower() for keyword in keywords)]
            tail_names = set(keyword_matches[-32:]) if keyword_matches else set()
            matched_names = enable_by_predicate(lambda name: name in tail_names)
            if not matched_names:
                warnings.append("last_norm 未匹配到 image encoder 尾部 norm/neck/adapter/output/proj 参数。")
        else:
            prefixes = _extract_last_block_prefixes(all_names, "blocks")
            if not prefixes:
                prefixes = _extract_last_block_prefixes(all_names, "layers")
            if not prefixes:
                warnings.append(f"{unfreeze_image_mode} 未识别到 image encoder blocks/layers，fallback 到 last_norm。")
                return set_trainable_modules(
                    model,
                    {
                        "train_decoder": train_decoder,
                        "train_refine_head": train_refine_head,
                        "unfreeze_image_mode": "last_norm",
                    },
                )
            n_blocks = 1 if unfreeze_image_mode == "last1" else 2
            selected_prefixes = prefixes[-n_blocks:]
            matched_names = enable_by_predicate(lambda name: any(name.startswith(prefix + ".") for prefix in selected_prefixes))
            if not matched_names:
                fallback_mode = "last1" if unfreeze_image_mode == "last2" else "last_norm"
                warnings.append(f"{unfreeze_image_mode} 未成功打开 block 参数，fallback 到 {fallback_mode}。")
                return set_trainable_modules(
                    model,
                    {
                        "train_decoder": train_decoder,
                        "train_refine_head": train_refine_head,
                        "unfreeze_image_mode": fallback_mode,
                    },
                )

    summary = _summarize_trainable_params(model)
    if summary["text_trainable_params_m"] > 0:
        raise RuntimeError("text_encoder trainable params must stay 0.")
    if unfreeze_image_mode != "none" and summary["image_trainable_params_m"] <= 0:
        raise RuntimeError("partial unfreeze 已开启，但 image encoder trainable params = 0。")
    if summary["image_trainable_params_m"] > 8.0:
        raise RuntimeError(
            f"image encoder trainable params 过大({summary['image_trainable_params_m']:.2f}M)，疑似误解冻了 backbone。"
        )

    result = {
        "preset": preset,
        "train_decoder": train_decoder,
        "train_refine_head": train_refine_head,
        "unfreeze_image_mode": unfreeze_image_mode,
        "matched_image_param_names": matched_names,
        "warnings": warnings,
        "summary": summary,
    }
    LOGGER.info(
        "set_trainable_modules preset=%s train_decoder=%s train_refine_head=%s unfreeze_image_mode=%s image_matches=%d",
        preset,
        train_decoder,
        train_refine_head,
        unfreeze_image_mode,
        len(matched_names),
    )
    if warnings:
        LOGGER.warning("trainable warnings=%s", warnings)
    return result
