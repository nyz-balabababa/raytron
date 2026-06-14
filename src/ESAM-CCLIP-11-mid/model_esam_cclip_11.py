from pathlib import Path

import torch

from common_esam_cclip_11 import ESAMCCLIPModel, load_state_dict_flexible, maybe_tqdm


KEY_PREFIX_REMAP = {
    "image_encoder.": "img_encoder.",
    "text_encoder.": "txt_encoder.",
    "fusion_decoder.": "decoder.",
}


def _strip_common_prefixes(key):
    while key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("model."):
        key = key[len("model."):]
    return key


def _remap_key(key):
    key = _strip_common_prefixes(key)
    for old_prefix, new_prefix in KEY_PREFIX_REMAP.items():
        if key.startswith(old_prefix):
            return f"{new_prefix}{key[len(old_prefix):]}"
    return key


def _prepare_matched_state_dict(model, state_dict):
    model_state = model.state_dict()
    matched_state = {}
    matched_keys = []
    decoder_matched_keys = []
    unexpected_keys = []
    shape_mismatch_keys = []

    iterator = maybe_tqdm(
        state_dict.items(),
        total=len(state_dict),
        desc="Match ckpt keys",
        leave=False,
    )
    for raw_key, value in iterator:
        mapped_key = _remap_key(raw_key)
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

    missing_keys = [key for key in model_state.keys() if key not in matched_state]
    unexpected_keys.extend(shape_mismatch_keys)
    return matched_state, matched_keys, decoder_matched_keys, missing_keys, unexpected_keys
def load_checkpoint_flexible(model, checkpoint_path, device, require_decoder_match=True):
    checkpoint_path = Path(checkpoint_path)
    print(f"[Checkpoint] reading file: {checkpoint_path}")
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
        if checkpoint_path.stat().st_size <= 0:
            raise EOFError(f"checkpoint 文件为空: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except EOFError as exc:
        raise RuntimeError(
            f"checkpoint 读取失败，文件为空或已损坏: {checkpoint_path}。"
            "如果这是自动续训生成的 last.pt，删除它或使用 --no_auto_resume；"
            "如果这是热启动权重，请重新指定有效 pt 文件。"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"checkpoint 读取失败: {checkpoint_path} error={exc}") from exc
    print("[Checkpoint] extracting state_dict...")
    state_dict = load_state_dict_flexible(checkpoint)
    print(f"[Checkpoint] state_dict_keys={len(state_dict)}")
    print("[Checkpoint] matching keys...")
    matched_state, matched_keys, decoder_matched_keys, missing_keys, unexpected_keys = _prepare_matched_state_dict(
        model,
        state_dict,
    )
    shape_mismatch_keys = [item for item in unexpected_keys if ": ckpt=" in item]
    if require_decoder_match and len(decoder_matched_keys) == 0:
        raise RuntimeError(
            "checkpoint 未命中任何 decoder 参数，拒绝热启动。请检查旧权重 key 是否需要 remap。"
        )
    model.load_state_dict(matched_state, strict=False)
    print(f"[Checkpoint] loaded: {checkpoint_path}")
    print(f"[Checkpoint] matched_keys={len(matched_keys)}")
    print(f"[Checkpoint] decoder_matched_keys={len(decoder_matched_keys)}")
    print(f"[Checkpoint] missing_keys={len(missing_keys)}")
    print(f"[Checkpoint] unexpected_keys={len(unexpected_keys)}")
    print(f"[Checkpoint] shape_mismatch={len(shape_mismatch_keys)}")
    if missing_keys:
        print(f"[Checkpoint] missing_keys_preview={missing_keys[:20]}")
    if unexpected_keys:
        print(f"[Checkpoint] unexpected_keys_preview={unexpected_keys[:20]}")
    model._last_checkpoint_load_info = {
        "checkpoint_path": str(checkpoint_path),
        "matched_keys": matched_keys,
        "decoder_matched_keys": decoder_matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }
    return checkpoint, missing_keys, unexpected_keys
