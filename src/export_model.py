#!/usr/bin/env python3
"""
CLIPSeg 模型导出 —— 打包为 Docker 提交所需的 /raytron/code/model/ 目录

用法:
    python src/export_model.py

输入:
    test/train_output/clipseg_v1/best.pt    ← CLIPSeg 训练好的 checkpoint
    model/clipseg-rd64-refined/             ← HuggingFace base 模型 (config + tokenizer)

输出:
    model/submit/                           ← Docker 构建用，包含:
        ├── config.json                     ← HuggingFace 配置
        ├── preprocessor_config.json
        ├── tokenizer.json / vocab.json / merges.txt  ← CLIP tokenizer
        ├── special_tokens_map.json / tokenizer_config.json
        └── sam3.pt                         ← 训练好的 CLIPSeg state_dict (FP32)
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# ── 路径配置 ──
ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"   # HuggingFace base 模型
TRAINED_CKPT = ROOT / "test" / "train_output" / "clipseg_v2" / "best.pt"   # 训练好的权重
OUTPUT_DIR = ROOT / "model" / "submit-clipseg-v2"                      # 输出目录
HF_CACHE_DIR = ROOT / "test" / ".hf_cache"

# ══════════════════════════════════════════════════════════════════════
# HuggingFace 必须文件列表（tokenizer + config）
# ══════════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def ensure_hf_cache():
    """Use a repo-local HuggingFace cache to avoid user-profile permission issues."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))


def check_model_param_count(checkpoint_path: Path) -> int:
    """加载 checkpoint 并核实参数量 ≤ 300M。"""
    from transformers import CLIPSegForImageSegmentation

    logger.info(f"检查参数量: {checkpoint_path}")
    model = CLIPSegForImageSegmentation.from_pretrained(str(BASE_MODEL_DIR))

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  总参数量: {total:,}")
    logger.info(f"  300M 限制: {'✅ 通过' if total < 300_000_000 else '❌ 超标!'}")
    return total


def export(checkpoint_path: Path = TRAINED_CKPT, output_dir: Path = OUTPUT_DIR):
    """
    1. 复制 HuggingFace 配置文件到输出目录
    2. 从训练 checkpoint 提取 state_dict 保存为 sam3.pt
    3. 验证参数量 ≤ 300M
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 步骤 1: 复制 HuggingFace 配置文件 ──
    logger.info(f"从 {BASE_MODEL_DIR} 复制配置文件...")
    for fname in REQUIRED_FILES:
        src = BASE_MODEL_DIR / fname
        dst = output_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  ✅ {fname}")
        else:
            logger.warning(f"  ⚠️  {fname} 缺失")

    # ── 步骤 2: 处理训练权重 ──
    if checkpoint_path.exists():
        logger.info(f"导出训练权重: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 提取 state_dict（兼容多种保存格式）
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            logger.info("  格式: {'model_state_dict': ...}")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            logger.info("  格式: {'model': ...}")
        else:
            state_dict = ckpt
            logger.info("  格式: 裸 state_dict")

        # 保存为 sam3.pt (FP32)
        sam3_path = output_dir / "sam3.pt"
        torch.save(state_dict, sam3_path)
        size_mb = sam3_path.stat().st_size / 1024**2
        logger.info(f"  已保存: {sam3_path} ({size_mb:.0f} MB)")
    else:
        logger.warning(f"⚠️  训练权重不存在: {checkpoint_path}")
        logger.warning("  将使用 HuggingFace 原始 pytorch_model.bin 作为替代")
        src = BASE_MODEL_DIR / "pytorch_model.bin"
        if src.exists():
            dst = output_dir / "sam3.pt"
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1024**2
            logger.info(f"  已复制: {dst} ({size_mb:.0f} MB)")

    # ── 步骤 3: 验证参数量 ──
    logger.info("")
    total_params = check_model_param_count(checkpoint_path)

    # ── 输出清单 ──
    logger.info(f"\n{'=' * 50}")
    logger.info(f"导出完成: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / 1024**2
        logger.info(f"  {f.name:30s} {size:8.1f} MB")
    logger.info(f"{'─' * 50}")
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"下一步: docker build -t raytron-submit .")
    logger.info(f"{'=' * 50}")


def main():
    if not BASE_MODEL_DIR.exists():
        logger.error(f"Base 模型不存在: {BASE_MODEL_DIR}")
        sys.exit(1)

    ensure_hf_cache()
    export()


if __name__ == "__main__":
    main()
