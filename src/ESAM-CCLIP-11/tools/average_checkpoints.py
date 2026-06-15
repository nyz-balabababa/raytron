#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

INCLUDE_PATTERNS = [
    "decoder.",
    "fusion_decoder.",
    "film_projection",
    "conv_after_fusion",
    "mask_head",
    "adapter",
    "prompt_adapter",
]

EXCLUDE_PATTERNS = [
    "image_encoder",
    "img_encoder",
    "text_encoder",
    "txt_encoder",
    "clip",
    "chinese_clip",
    "efficient_sam",
    "backbone",
    "tokenizer",
]

STATE_DICT_CANDIDATE_KEYS = [
    "model_state_dict",
    "state_dict",
    "model",
    "net",
    "module",
]


def torch_load_compat(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_state_dict_like(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload) and all(torch.is_tensor(v) for v in payload.values())


def find_state_dict(checkpoint: Any, preferred_key: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if preferred_key:
        candidate = checkpoint.get(preferred_key) if isinstance(checkpoint, dict) else None
        if isinstance(candidate, dict):
            return preferred_key, candidate
        if is_state_dict_like(checkpoint):
            return None, checkpoint
        raise RuntimeError(f"preferred model key not found or not a dict: {preferred_key}")

    if isinstance(checkpoint, dict):
        for key in STATE_DICT_CANDIDATE_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return key, candidate
        if is_state_dict_like(checkpoint):
            return None, checkpoint
    raise RuntimeError("unable to find state_dict in checkpoint")


def validate_metadata(checkpoints: Sequence[Any], paths: Sequence[Path], skip_check: bool) -> None:
    fields = ["classes", "class_to_idx", "num_classes"]
    for field in fields:
        existing = [(path, ckpt[field]) for path, ckpt in zip(paths, checkpoints) if isinstance(ckpt, dict) and field in ckpt]
        if len(existing) <= 1:
            continue
        base_value = existing[0][1]
        mismatches = [(path, value) for path, value in existing[1:] if value != base_value]
        if not mismatches:
            continue
        message = (
            f"metadata mismatch for `{field}`: "
            + ", ".join([f"{path}={value!r}" for path, value in existing])
        )
        if skip_check:
            print(f"WARNING: {message}")
            continue
        raise RuntimeError(message)


def normalize_weights(raw_weights: Optional[Sequence[float]], count: int, do_normalize: bool) -> Tuple[List[float], List[float]]:
    if raw_weights is None:
        raw = [1.0] * count
    else:
        raw = [float(value) for value in raw_weights]
        if len(raw) != count:
            raise RuntimeError(f"weights count mismatch: got {len(raw)} for {count} checkpoints")
    if any(weight < 0 for weight in raw):
        raise RuntimeError("weights must be non-negative")
    if do_normalize:
        total = sum(raw)
        if total <= 0:
            raise RuntimeError("weights sum must be > 0 when normalize_weights=True")
        normalized = [weight / total for weight in raw]
    else:
        normalized = list(raw)
    return raw, normalized


def should_average_key(key: str, decoder_only: bool) -> bool:
    lower_key = key.lower()
    if any(pattern in lower_key for pattern in EXCLUDE_PATTERNS):
        return False
    if not decoder_only:
        return True
    return any(pattern in lower_key for pattern in INCLUDE_PATTERNS)


def clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def average_state_dicts(
    state_dicts: Sequence[Dict[str, Any]],
    weights: Sequence[float],
    base_index: int,
    strict_shapes: bool,
    decoder_only: bool,
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    base_state = state_dicts[base_index]
    output_state = {key: clone_state_value(value) for key, value in base_state.items()}
    averaged_keys: List[str] = []
    skipped_keys: List[str] = []
    non_avg_keys: List[str] = []

    for key, base_value in base_state.items():
        if not should_average_key(key, decoder_only):
            non_avg_keys.append(key)
            continue
        if not torch.is_tensor(base_value):
            if strict_shapes:
                raise RuntimeError(f"key `{key}` selected for averaging but base value is not tensor")
            skipped_keys.append(key)
            continue
        tensors: List[torch.Tensor] = []
        mismatch_reason: Optional[str] = None
        for idx, state_dict in enumerate(state_dicts):
            value = state_dict.get(key)
            if not torch.is_tensor(value):
                mismatch_reason = f"missing_or_non_tensor@{idx}"
                break
            if tuple(value.shape) != tuple(base_value.shape):
                mismatch_reason = f"shape_mismatch@{idx}:{tuple(base_value.shape)}!={tuple(value.shape)}"
                break
            if not (value.dtype.is_floating_point or value.dtype.is_complex):
                mismatch_reason = f"non_float_dtype@{idx}:{value.dtype}"
                break
            tensors.append(value.detach().cpu())
        if mismatch_reason is not None:
            if strict_shapes:
                raise RuntimeError(f"state_dict key `{key}` incompatible: {mismatch_reason}")
            skipped_keys.append(key)
            continue

        accum_dtype = torch.complex64 if base_value.dtype.is_complex else torch.float32
        weighted_sum = None
        for weight, tensor in zip(weights, tensors):
            part = tensor.to(accum_dtype) * float(weight)
            weighted_sum = part if weighted_sum is None else weighted_sum + part
        output_state[key] = weighted_sum.to(base_value.dtype)
        averaged_keys.append(key)

    if not averaged_keys:
        raise RuntimeError("averaged_key_count == 0, include patterns did not match any valid parameters")

    return output_state, averaged_keys, skipped_keys, non_avg_keys


def clear_training_state(output_checkpoint: Dict[str, Any]) -> None:
    for key in [
        "optimizer_state_dict",
        "scheduler_state_dict",
        "optimizer",
        "scheduler",
        "scaler",
        "amp_scaler",
    ]:
        if key in output_checkpoint:
            output_checkpoint[key] = None


def build_output_checkpoint(
    base_checkpoint: Any,
    base_model_key: Optional[str],
    output_state: Dict[str, Any],
    checkpoint_paths: Sequence[Path],
    raw_weights: Sequence[float],
    normalized_weights: Sequence[float],
    base_index: int,
    decoder_only: bool,
    drop_old_thresholds: bool,
    averaged_keys: Sequence[str],
    skipped_keys: Sequence[str],
    non_avg_keys: Sequence[str],
) -> Any:
    if base_model_key is None and is_state_dict_like(base_checkpoint):
        return output_state

    output_checkpoint = copy.deepcopy(base_checkpoint)
    output_checkpoint[base_model_key] = output_state
    clear_training_state(output_checkpoint)
    if drop_old_thresholds:
        for key in ["prompt_thresholds", "val_thresholds", "postprocess_cfg", "postprocess"]:
            output_checkpoint.pop(key, None)
    output_checkpoint["soup_metadata"] = {
        "type": "decoder_only_weighted_soup",
        "base_index": int(base_index),
        "base_checkpoint": str(checkpoint_paths[base_index]),
        "source_checkpoints": [str(path) for path in checkpoint_paths],
        "weights_raw": [float(v) for v in raw_weights],
        "weights_normalized": [float(v) for v in normalized_weights],
        "decoder_only": bool(decoder_only),
        "include_patterns": list(INCLUDE_PATTERNS),
        "exclude_patterns": list(EXCLUDE_PATTERNS),
        "averaged_key_count": len(averaged_keys),
        "skipped_key_count": len(skipped_keys),
        "non_avg_key_count": len(non_avg_keys),
        "model_key": base_model_key,
    }
    output_checkpoint["averaged_from"] = [str(path) for path in checkpoint_paths]
    output_checkpoint["average_num_checkpoints"] = len(checkpoint_paths)
    return output_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a safe ESAM decoder-only weighted checkpoint soup.",
        epilog=(
            "PowerShell example:\n"
            "  python src/ESAM-CCLIP-11/tools/average_checkpoints.py `\n"
            "    --checkpoints `\n"
            "      \"lite_best.pt\" `\n"
            "      \"stage2_mild_best_all11.pt\" `\n"
            "    --weights 0.85 0.15 `\n"
            "    --output \"soup_lite085_stage2mild015.pt\" `\n"
            "    --base_index 0 `\n"
            "    --no_strict_shapes"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Input checkpoint paths")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional soup weights")
    parser.add_argument("--base_index", type=int, default=0, help="Base checkpoint index")
    parser.add_argument("--model_key", default=None, help="Preferred state_dict key")
    parser.add_argument("--strict_shapes", dest="strict_shapes", action="store_true")
    parser.add_argument("--no_strict_shapes", dest="strict_shapes", action="store_false")
    parser.add_argument("--normalize_weights", dest="normalize_weights", action="store_true")
    parser.add_argument("--no_normalize_weights", dest="normalize_weights", action="store_false")
    parser.add_argument("--decoder_only", dest="decoder_only", action="store_true")
    parser.add_argument("--no_decoder_only", dest="decoder_only", action="store_false")
    parser.add_argument("--drop_old_thresholds", action="store_true", default=False)
    parser.add_argument("--skip_metadata_check", action="store_true", default=False)
    parser.add_argument("--print_keys", action="store_true", default=False)
    parser.set_defaults(strict_shapes=False, normalize_weights=True, decoder_only=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path).resolve() for path in args.checkpoints]
    output_path = Path(args.output).resolve()

    if len(checkpoint_paths) < 2:
        raise RuntimeError("need at least two checkpoints to average")
    if args.base_index < 0 or args.base_index >= len(checkpoint_paths):
        raise RuntimeError(f"base_index out of range: {args.base_index}")
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    raw_weights, normalized_weights = normalize_weights(args.weights, len(checkpoint_paths), bool(args.normalize_weights))
    checkpoints = [torch_load_compat(path) for path in checkpoint_paths]
    validate_metadata(checkpoints, checkpoint_paths, bool(args.skip_metadata_check))

    found = [find_state_dict(checkpoint, args.model_key) for checkpoint in checkpoints]
    model_keys = [item[0] for item in found]
    state_dicts = [item[1] for item in found]
    base_model_key = model_keys[args.base_index]

    output_state, averaged_keys, skipped_keys, non_avg_keys = average_state_dicts(
        state_dicts=state_dicts,
        weights=normalized_weights,
        base_index=args.base_index,
        strict_shapes=bool(args.strict_shapes),
        decoder_only=bool(args.decoder_only),
    )

    output_checkpoint = build_output_checkpoint(
        base_checkpoint=checkpoints[args.base_index],
        base_model_key=base_model_key,
        output_state=output_state,
        checkpoint_paths=checkpoint_paths,
        raw_weights=raw_weights,
        normalized_weights=normalized_weights,
        base_index=args.base_index,
        decoder_only=bool(args.decoder_only),
        drop_old_thresholds=bool(args.drop_old_thresholds),
        averaged_keys=averaged_keys,
        skipped_keys=skipped_keys,
        non_avg_keys=non_avg_keys,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)

    print("========== Decoder Soup Summary ==========")
    print("inputs:")
    for weight, path in zip(normalized_weights, checkpoint_paths):
        print(f"  weight={weight:.6f} path={path}")
    print(f"base_index: {args.base_index}")
    print(f"actual_model_key: {base_model_key}")
    print(f"decoder_only: {bool(args.decoder_only)}")
    print(f"strict_shapes: {bool(args.strict_shapes)}")
    print(f"normalize_weights: {bool(args.normalize_weights)}")
    print(f"drop_old_thresholds: {bool(args.drop_old_thresholds)}")
    print(f"averaged_key_count: {len(averaged_keys)}")
    print(f"skipped_key_count: {len(skipped_keys)}")
    print(f"non_avg_key_count: {len(non_avg_keys)}")
    print("averaged_keys_preview:")
    for key in averaged_keys[:20]:
        print(f"  {key}")
    print("skipped_keys_preview:")
    for key in skipped_keys[:20]:
        print(f"  {key}")
    if args.print_keys:
        print("non_avg_keys_preview:")
        for key in non_avg_keys[:50]:
            print(f"  {key}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
