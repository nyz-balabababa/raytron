#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def torch_load_compat(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not a dict: {path}")
    return payload


def validate_metadata(checkpoints: List[Dict[str, Any]], paths: List[Path]) -> None:
    fields = ["classes", "class_to_idx", "num_classes"]
    base = checkpoints[0]
    for field in fields:
        base_value = base.get(field, None)
        for path, checkpoint in zip(paths[1:], checkpoints[1:]):
            value = checkpoint.get(field, None)
            if value != base_value:
                raise RuntimeError(
                    f"metadata mismatch for `{field}`: base={paths[0]} value={base_value!r}, "
                    f"other={path} value={value!r}"
                )


def extract_state_dict(checkpoint: Dict[str, Any], model_key: str, path: Path) -> Dict[str, torch.Tensor]:
    state_dict = checkpoint.get(model_key)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"missing `{model_key}` in checkpoint: {path}")
    return state_dict


def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    strict_shapes: bool,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    base_state = state_dicts[0]
    averaged: Dict[str, torch.Tensor] = {}
    averaged_keys: List[str] = []
    skipped_keys: List[str] = []

    for key, base_tensor in base_state.items():
        if not torch.is_tensor(base_tensor):
            skipped_keys.append(key)
            continue
        tensors = [base_tensor]
        mismatch_reason = None
        for state_dict in state_dicts[1:]:
            tensor = state_dict.get(key)
            if not torch.is_tensor(tensor):
                mismatch_reason = "missing_or_non_tensor"
                break
            if tuple(tensor.shape) != tuple(base_tensor.shape):
                mismatch_reason = f"shape_mismatch:{tuple(base_tensor.shape)}!={tuple(tensor.shape)}"
                break
            tensors.append(tensor)
        if mismatch_reason is not None:
            if strict_shapes:
                raise RuntimeError(f"state_dict key `{key}` incompatible: {mismatch_reason}")
            skipped_keys.append(key)
            continue

        if base_tensor.dtype.is_floating_point or base_tensor.dtype.is_complex:
            stacked = torch.stack([tensor.detach().cpu().to(torch.float32) for tensor in tensors], dim=0)
            mean_tensor = stacked.mean(dim=0).to(base_tensor.dtype)
            averaged[key] = mean_tensor
            averaged_keys.append(key)
            continue

        if all(torch.equal(base_tensor, tensor) for tensor in tensors[1:]):
            averaged[key] = base_tensor.detach().cpu().clone()
            averaged_keys.append(key)
            continue

        if strict_shapes:
            raise RuntimeError(f"non-floating tensor key `{key}` differs across checkpoints")
        skipped_keys.append(key)

    return averaged, averaged_keys, skipped_keys


def build_output_checkpoint(
    base_checkpoint: Dict[str, Any],
    model_key: str,
    averaged_state_dict: Dict[str, torch.Tensor],
    input_paths: List[Path],
) -> Dict[str, Any]:
    output_checkpoint = dict(base_checkpoint)
    output_checkpoint[model_key] = averaged_state_dict
    for key in ["optimizer_state_dict", "scheduler_state_dict", "optimizer", "scheduler"]:
        if key in output_checkpoint:
            output_checkpoint[key] = None
    output_checkpoint["averaged_from"] = [str(path) for path in input_paths]
    output_checkpoint["average_num_checkpoints"] = len(input_paths)
    return output_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible ESAM checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Input checkpoint paths")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--base_index", type=int, default=0, help="Base checkpoint index for metadata")
    parser.add_argument("--model_key", default="model_state_dict", help="State-dict key to average")
    parser.add_argument("--strict_shapes", dest="strict_shapes", action="store_true")
    parser.add_argument("--no_strict_shapes", dest="strict_shapes", action="store_false")
    parser.set_defaults(strict_shapes=True)
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

    loaded = [torch_load_compat(path) for path in checkpoint_paths]
    validate_metadata(loaded, checkpoint_paths)
    state_dicts = [extract_state_dict(checkpoint, args.model_key, path) for checkpoint, path in zip(loaded, checkpoint_paths)]
    averaged_state_dict, averaged_keys, skipped_keys = average_state_dicts(
        state_dicts=state_dicts,
        strict_shapes=bool(args.strict_shapes),
    )

    base_checkpoint = loaded[args.base_index]
    output_checkpoint = build_output_checkpoint(
        base_checkpoint=base_checkpoint,
        model_key=args.model_key,
        averaged_state_dict=averaged_state_dict,
        input_paths=checkpoint_paths,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)

    print("========== Average Checkpoints Summary ==========")
    print("inputs:")
    for path in checkpoint_paths:
        print(f"  {path}")
    print(f"base_index: {args.base_index}")
    print(f"model_key: {args.model_key}")
    print(f"strict_shapes: {bool(args.strict_shapes)}")
    print(f"averaged_key_count: {len(averaged_keys)}")
    print(f"skipped_key_count: {len(skipped_keys)}")
    if skipped_keys:
        print("skipped_keys_preview:")
        for key in skipped_keys[:20]:
            print(f"  {key}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
