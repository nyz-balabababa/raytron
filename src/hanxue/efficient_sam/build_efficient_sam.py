# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from .efficient_sam import build_efficient_sam


ROOT = Path(__file__).resolve().parents[1]


def _resolve_checkpoint(rel_path: str) -> str:
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return str(cwd_path)
    return str(ROOT / p)

def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=_resolve_checkpoint("weights/efficient_sam/efficient_sam_vitt.pt"),
    ).eval()


def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=_resolve_checkpoint("weights/efficient_sam_vits.pt"),
    ).eval()
