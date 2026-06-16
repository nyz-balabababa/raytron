#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common_esam_cclip_11_urf import encode_mask_to_rle  # noqa: E402
from config_esam_cclip_11_urf import (  # noqa: E402
    DEFAULT_RESUME_BEST,
    EFFICIENT_SAM_CKPT,
    OLD_CLASSES,
    RARE_CLASSES,
    TOKENIZER_DIR,
)
from dataset_esam_cclip_11_urf import ESAMCCLIP11Dataset  # noqa: E402
from model_esam_cclip_11_urf import (  # noqa: E402
    build_model_from_config,
    load_checkpoint_safely,
    validate_refine_head_structure,
)


def build_cfg(use_refine_head: bool) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_dir=TOKENIZER_DIR,
        efficient_sam_ckpt=EFFICIENT_SAM_CKPT,
        freeze_image=True,
        freeze_text=True,
        use_refine_head=use_refine_head,
        refine_head_hidden_dim=16,
    )


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_models_check() -> Path:
    model_plain = build_model_from_config(build_cfg(use_refine_head=False))
    model_refine = build_model_from_config(build_cfg(use_refine_head=True))
    validate_refine_head_structure(model_plain)
    refine_info = validate_refine_head_structure(model_refine)
    assert_condition(model_plain.use_refine_head is False, "use_refine_head=False 构建失败")
    assert_condition(model_refine.use_refine_head is True, "use_refine_head=True 构建失败")
    assert_condition(refine_info["conv1_in_channels"] == 3, "refine_head.conv1.in_channels != 3")
    return model_refine


def checkpoint_loading_check(base_ckpt: Path | None) -> Path:
    model_refine = build_model_from_config(build_cfg(use_refine_head=True))
    valid_checkpoint = {
        "model_state_dict": model_refine.state_dict(),
        "use_refine_head": True,
        "refine_head_hidden_dim": 16,
    }

    valid_target = build_model_from_config(build_cfg(use_refine_head=True))
    load_checkpoint_safely(valid_target, valid_checkpoint, device="cpu", strict=False)
    load_info = getattr(valid_target, "_last_checkpoint_load_info", {})
    assert_condition(len(load_info.get("refine_head_matched_keys", [])) > 0, "新 refine checkpoint 未正确命中 refine_head")

    broken_state = model_refine.state_dict()
    broken_state["refine_head.conv1.weight"] = broken_state["refine_head.conv1.weight"][:, :2, :, :].clone()
    broken_checkpoint = {
        "model_state_dict": broken_state,
        "use_refine_head": True,
        "refine_head_hidden_dim": 16,
    }
    broken_target = build_model_from_config(build_cfg(use_refine_head=True))
    try:
        load_checkpoint_safely(broken_target, broken_checkpoint, device="cpu", strict=False)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("损坏 refine checkpoint 未触发 shape mismatch 保护")

    if base_ckpt is not None and base_ckpt.exists():
        old_target = build_model_from_config(build_cfg(use_refine_head=True))
        load_checkpoint_safely(old_target, base_ckpt, device="cpu", strict=False)
        old_info = getattr(old_target, "_last_checkpoint_load_info", {})
        assert_condition(
            not old_info.get("checkpoint_claims_refine", False),
            "旧 checkpoint 不应被识别为 refine checkpoint",
        )
    return base_ckpt if base_ckpt is not None else SCRIPT_DIR / "train_esam_cclip_11_urf.py"


def dataset_oversample_check() -> None:
    temp_dir = PROJECT_ROOT / ".tmp_urf_dataset_check"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / "test" / "images" / "demo.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16), 127, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    rle = encode_mask_to_rle(mask)

    items = []
    for idx in range(30):
        items.append(
            {
                "ann_id": idx,
                "image_path": "test/images/demo.png",
                "prompts": {
                    "fence": {
                        "selected_hit": True,
                        "selected_rle": rle,
                        "sample_weight": 1.0,
                        "include_in_train": True,
                    }
                },
            }
        )

    annotation_path = temp_dir / "annotations.json"
    with open(annotation_path, "w", encoding="utf-8") as file_obj:
        json.dump(items, file_obj, ensure_ascii=False)

    dataset = ESAMCCLIP11Dataset(
        annotation_json=annotation_path,
        image_root=temp_dir,
        img_size=(32, 32),
        classes=list(OLD_CLASSES) + list(RARE_CLASSES),
        tokenizer=None,
        prompt_prototypes=None,
        negative_sample_prob=0.0,
        negative_sample_weight=0.05,
        rare_oversample=1.2,
        old_class_sample_ratio=0.95,
        rare_class_keep_ratio=1.0,
        old_classes=OLD_CLASSES,
        rare_classes=RARE_CLASSES,
        rare_balance_enabled=True,
        training=True,
        seed=42,
    )
    extra_count = int(dataset.stats.get("rebalance", {}).get("rare_oversample_extra_count", 0))
    assert_condition(extra_count > 0, "rare_oversample=1.2 未产生额外样本")


def run_subprocess(command: list[str], must_contain: str | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = (result.stdout or "") + (result.stderr or "")
    if must_contain is not None and must_contain not in output:
        raise RuntimeError(f"命令输出缺少关键字 {must_contain!r}: {' '.join(command)}")


def cli_smoke_checks(python_exe: str, base_ckpt: Path) -> None:
    run_subprocess([python_exe, str(SCRIPT_DIR / "train_esam_cclip_11_urf.py"), "--help"], must_contain="usage:")
    run_subprocess([python_exe, str(SCRIPT_DIR / "sweep_thresholds_11_urf.py"), "--help"], must_contain="usage:")
    run_subprocess([python_exe, str(SCRIPT_DIR / "inference_urf.py"), "--help"], must_contain="usage:")
    run_subprocess(
        [
            python_exe,
            str(SCRIPT_DIR / "run_pipeline_urf.py"),
            "--pipeline",
            "unfreeze_then_refine",
            "--base_ckpt",
            str(base_ckpt),
            "--output_root",
            str(PROJECT_ROOT / "test" / "train_output"),
            "--dry_run",
        ],
        must_contain="[Stage 1]",
    )
    run_subprocess(
        [
            python_exe,
            str(SCRIPT_DIR / "run_pipeline_urf.py"),
            "--pipeline",
            "refine_then_unfreeze",
            "--base_ckpt",
            str(base_ckpt),
            "--output_root",
            str(PROJECT_ROOT / "test" / "train_output"),
            "--dry_run",
        ],
        must_contain="[Stage 1]",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight URF self-check")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--base_ckpt", type=Path, default=DEFAULT_RESUME_BEST)
    args = parser.parse_args()

    model_refine = build_models_check()
    assert_condition(model_refine._refine_head_structure_info["out_weight_all_zero"], "refine_head zero-init 自检失败")
    valid_ckpt_path = checkpoint_loading_check(args.base_ckpt if args.base_ckpt.exists() else None)
    dataset_oversample_check()
    cli_smoke_checks(args.python, args.base_ckpt if args.base_ckpt.exists() else valid_ckpt_path)
    print("URF_CHECK_OK")


if __name__ == "__main__":
    main()
