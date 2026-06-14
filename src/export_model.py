#!/usr/bin/env python3
"""
提交模型导出脚本。

支持两条提交路线：
1. `clipseg_v2`
2. `clipseg_xiyou_12cls_trainval_v1`
2. `efficientsam_easy_6cls_v1`

用法：
    python src/export_model.py --preset efficientsam_easy_6cls_v1
    python src/export_model.py --preset clipseg_v2
    python src/export_model.py --preset clipseg_xiyou_12cls_trainval_v1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = ROOT / "test" / ".hf_cache"

PRESETS: Dict[str, Dict[str, Any]] = {
    "clipseg_v2": {
        "model_family": "clipseg",
        "base_model_dir": ROOT / "model" / "clipseg-rd64-refined",
        "checkpoint_path": ROOT / "test" / "train_output" / "clipseg_v2" / "best.pt",
        "output_dir": ROOT / "model" / "submit-clipseg-v2",
        "required_files": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
    },
    "clipseg_xiyou_12cls_trainval_v1": {
        "model_family": "clipseg",
        "base_model_dir": ROOT / "model" / "clipseg-rd64-refined",
        "checkpoint_path": ROOT / "test" / "train_output" / "clipseg_xiyou_12cls_trainval_v1" / "best.pt",
        "output_dir": ROOT / "model" / "submit-clipseg-xiyou-12cls-trainval-v1",
        "required_files": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
    },
    "efficientsam_easy_6cls_v1": {
        "model_family": "esam_chineseclip",
        "base_model_dir": ROOT / "src" / "hanxue" / "weights" / "chinese_clip",
        "checkpoint_path": ROOT / "test" / "train_output" / "efficientsam_easy_6cls_v1" / "efficientsam_easy_6cls_v1" / "best.pt",
        "output_dir": ROOT / "model" / "submit-efficientsam-easy-6cls-v1",
        "required_files": [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
        "optional_files": [
            "preprocessor_config.json",
        ],
    },
}


def ensure_hf_cache() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_like = [torch.is_tensor(v) for v in checkpoint.values()]
            if tensor_like and any(tensor_like):
                return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict。")


def copy_files(src_dir: Path, output_dir: Path, required_files, optional_files=()) -> None:
    logger.info(f"从 {src_dir} 复制提交所需文件...")
    for fname in required_files:
        src = src_dir / fname
        dst = output_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"缺少必需文件: {src}")
        shutil.copy2(src, dst)
        logger.info(f"  ✅ {fname}")

    for fname in optional_files:
        src = src_dir / fname
        dst = output_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  ✅ {fname} (optional)")


def count_clipseg_params(base_model_dir: Path, checkpoint_path: Path) -> int:
    from transformers import CLIPSegForImageSegmentation

    logger.info(f"检查 CLIPSeg 参数量: {checkpoint_path}")
    model = CLIPSegForImageSegmentation.from_pretrained(str(base_model_dir))
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(ckpt)
        model.load_state_dict(state_dict, strict=False)
    total = sum(param.numel() for param in model.parameters())
    logger.info(f"  总参数量: {total:,}")
    logger.info(f"  300M 限制: {'✅ 通过' if total < 300_000_000 else '❌ 超标!'}")
    return int(total)


def count_state_dict_params(state_dict: Dict[str, torch.Tensor]) -> int:
    ignore_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    total = 0
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(ignore_suffixes):
            continue
        total += int(value.numel())
    logger.info(f"  近似参数量(按 state_dict 统计): {total:,}")
    logger.info(f"  300M 限制: {'✅ 通过' if total < 300_000_000 else '❌ 超标!'}")
    return total


def export(preset_name: str) -> None:
    if preset_name not in PRESETS:
        raise KeyError(f"未知 preset: {preset_name}，可选: {sorted(PRESETS)}")

    preset = PRESETS[preset_name]
    base_model_dir = Path(preset["base_model_dir"])
    checkpoint_path = Path(preset["checkpoint_path"])
    output_dir = Path(preset["output_dir"])
    required_files = list(preset.get("required_files", []))
    optional_files = list(preset.get("optional_files", []))

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base 模型目录不存在: {base_model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_files(base_model_dir, output_dir, required_files, optional_files)

    if checkpoint_path.exists():
        logger.info(f"导出训练权重: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(ckpt)
        sam3_path = output_dir / "sam3.pt"
        torch.save(state_dict, sam3_path)
        size_mb = sam3_path.stat().st_size / 1024**2
        logger.info(f"  已保存: {sam3_path} ({size_mb:.1f} MB)")
    else:
        raise FileNotFoundError(f"训练权重不存在: {checkpoint_path}")

    logger.info("")
    if preset["model_family"] == "clipseg":
        total_params = count_clipseg_params(base_model_dir, checkpoint_path)
    else:
        total_params = count_state_dict_params(state_dict)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"导出完成: {output_dir}")
    logger.info(f"preset: {preset_name}")
    for file_path in sorted(output_dir.iterdir()):
        size_mb = file_path.stat().st_size / 1024**2
        logger.info(f"  {file_path.name:30s} {size_mb:8.1f} MB")
    logger.info(f"{'─' * 50}")
    logger.info(f"统计参数量: {total_params:,}")
    logger.info("后续将以下内容同步到提交目录:")
    logger.info(f"  - 推理脚本: src/inference_esam_chineseclip_6cls.py 或 inference.py")
    logger.info(f"  - 模型目录: {output_dir}")
    logger.info(f"{'=' * 50}")


def parse_args():
    parser = argparse.ArgumentParser(description="导出提交所需的 sam3.pt + tokenizer/config 文件")
    parser.add_argument(
        "--preset",
        default="efficientsam_easy_6cls_v1",
        choices=sorted(PRESETS.keys()),
        help="导出预设",
    )
    return parser.parse_args()


def main() -> None:
    ensure_hf_cache()
    args = parse_args()
    try:
        export(args.preset)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
