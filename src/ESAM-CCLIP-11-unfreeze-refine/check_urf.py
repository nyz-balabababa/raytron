#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_esam_cclip_11_urf import EFFICIENT_SAM_CKPT, TOKENIZER_DIR  # noqa: E402
from model_esam_cclip_11_urf import (  # noqa: E402
    build_model_from_config,
    load_checkpoint_safely,
    validate_refine_head_structure,
)


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def build_cfg(use_refine_head: bool) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_dir=TOKENIZER_DIR,
        efficient_sam_ckpt=EFFICIENT_SAM_CKPT,
        freeze_image=True,
        freeze_text=True,
        use_refine_head=use_refine_head,
        refine_head_hidden_dim=16,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight URF refine-head check")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--use_refine_head", type=parse_bool, default=True)
    parser.add_argument("--device", type=str, default=AUTO_DEVICE)
    args = parser.parse_args()

    model_plain = build_model_from_config(build_cfg(use_refine_head=False))
    model_refine = build_model_from_config(build_cfg(use_refine_head=True))
    plain_info = validate_refine_head_structure(model_plain)
    refine_info = validate_refine_head_structure(model_refine)

    require(model_plain.use_refine_head is False, "build use_refine_head=False failed")
    require(model_refine.use_refine_head is True, "build use_refine_head=True failed")
    require(plain_info["conv1_in_channels"] == 3, "refine_head.conv1.in_channels != 3")
    require(refine_info["conv1_in_channels"] == 3, "refine_head.conv1.in_channels != 3")
    require(int(torch.count_nonzero(model_refine.refine_head.out.weight.detach()).item()) == 0, "refine_head.out.weight not zero-init")
    require(int(torch.count_nonzero(model_refine.refine_head.out.bias.detach()).item()) == 0, "refine_head.out.bias not zero-init")

    if args.ckpt is not None:
        if not args.ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
        model = build_model_from_config(build_cfg(use_refine_head=bool(args.use_refine_head)))
        load_checkpoint_safely(model, args.ckpt, device=args.device, strict=False)
        load_info = getattr(model, "_last_checkpoint_load_info", {})
        checkpoint_claims_refine = bool(load_info.get("checkpoint_claims_refine", False))
        refine_matched_count = len(load_info.get("refine_head_matched_keys", []))
        refine_shape_mismatch = load_info.get("refine_shape_mismatch_keys", [])
        missing_refine_keys = load_info.get("missing_refine_keys", [])

        if checkpoint_claims_refine:
            require(refine_matched_count > 0, "refine checkpoint matched refine keys = 0")
            require(not refine_shape_mismatch, f"refine checkpoint has shape mismatch: {refine_shape_mismatch[:10]}")
        else:
            require(bool(args.use_refine_head), "old checkpoint no refine only checked under use_refine_head=True")
            require(bool(missing_refine_keys), "old checkpoint expected missing refine keys")
            require(int(torch.count_nonzero(model.refine_head.out.weight.detach()).item()) == 0, "old checkpoint should keep zero-init refine_head.out.weight")
            require(int(torch.count_nonzero(model.refine_head.out.bias.detach()).item()) == 0, "old checkpoint should keep zero-init refine_head.out.bias")

        print(
            f"CHECK_CKPT_OK ckpt={args.ckpt} "
            f"checkpoint_claims_refine={checkpoint_claims_refine} "
            f"refine_matched_count={refine_matched_count}"
        )

    print("URF_CHECK_OK")


if __name__ == "__main__":
    main()
