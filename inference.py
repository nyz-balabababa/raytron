#!/usr/bin/env python3
"""
CLIPSeg 推理脚本（替代 SAM3）
============================
- 保留官方 inference.py 的全部接口合约（固定路径、RLE 编码、任务格式、输出格式）
- 将 SAM3 推理核心替换为 CLIPSeg（图像 + 文本 prompt → 二值掩码）
- 预处理管线与 CLIPSeg 训练一致：灰度读取 → 反色统一 → 等比缩放+pad → RGB → CLIP 标准化

使用方式:
    python inference.py

Docker 内路径约定 (不可修改):
    权重文件:  /raytron/code/model/sam3.pt           ← CLIPSeg state_dict
    模型配置:  /raytron/code/model/                   ← HuggingFace config + tokenizer
    任务文件:  /raytron/test/test_tasks.json
    图片根目录: /raytron/test/
    输出文件:  /raytron/test/predictions.json
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("警告: pycocotools 未安装，将使用备用 RLE 编码方法")
    maskUtils = None

# ══════════════════════════════════════════════════════════════════════
# 固定路径 — 不可修改
# ══════════════════════════════════════════════════════════════════════

DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test/"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"
DEFAULT_MODEL_DIR = "/raytron/code/model/"          # HuggingFace config + tokenizer
DEFAULT_CONF_THRESHOLD = 0.01

# ══════════════════════════════════════════════════════════════════════
# CLIPSeg 推理参数
# ══════════════════════════════════════════════════════════════════════

IMG_SIZE = 1024               # 等比缩放 + pad 正方形
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN = [0.48145466, 0.52048427, 0.45053169]
CLIP_STD  = [0.21028575, 0.23535925, 0.22184163]

# 反色统一参数（与训练一致）
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200


# ══════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════

def load_model(model_dir: str, checkpoint_path: Optional[str] = None):
    """
    加载 CLIPSeg 模型。

    Args:
        model_dir: HuggingFace 模型目录（含 config.json, tokenizer 等）
        checkpoint_path: 可选，训练好的 state_dict .pt 文件路径。
                         如果存在，会覆盖 base 模型权重。
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained(model_dir)
    model = CLIPSegForImageSegmentation.from_pretrained(model_dir)

    # 如果提供了训练好的权重，加载 state_dict
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        # 兼容两种保存格式：裸 state_dict 或 {'model_state_dict': ...}
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)

    model = model.to(DEVICE)
    model.eval()
    return model, processor


def count_model_params(model) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "device": DEVICE,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "param_limit_300M": int(total) < 300_000_000,
    }


# ══════════════════════════════════════════════════════════════════════
# RLE 编码（与原版 inference.py 完全一致）
# ══════════════════════════════════════════════════════════════════════

def simple_rle_encode(binary_mask: np.ndarray) -> Dict[str, Any]:
    flat = binary_mask.astype(np.uint8).flatten(order="F")
    runs: List[int] = []
    prev = 0
    count = 0
    for value in flat:
        value = int(value)
        if value == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = value
    runs.append(count)
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": ",".join(map(str, runs)),
    }


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = binary_mask.astype(np.uint8)
    if maskUtils is None:
        return simple_rle_encode(binary_mask)
    rle = maskUtils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle, list):
        rle = rle[0]
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("utf-8")
        if isinstance(rle["counts"], bytes)
        else rle["counts"],
    }


def build_empty_rle(height: int, width: int) -> Dict[str, Any]:
    return mask_to_rle(np.zeros((height, width), dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════════
# CLIPSeg 图像预处理（与训练完全一致）
# ══════════════════════════════════════════════════════════════════════

def _unify_polarity(gray: np.ndarray, fname: str) -> np.ndarray:
    """黑热→白热统一：确保热辐射强的目标是亮区。"""
    # 文件名含 blackHot → 强制反色
    if "blackHot" in fname:
        return 255 - gray
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < BLACKHOT_SKEW_THRESH:
            return 255 - gray
    # 整体偏亮且非可见光 → 兜底反色
    if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
        return 255 - gray
    return gray


def preprocess_image(image_path: str) -> Tuple[torch.Tensor, int, int]:
    """
    CLIPSeg 预处理管线。

    Returns:
        input_tensor: (1, 3, IMG_SIZE, IMG_SIZE) 已标准化的 tensor
        orig_h, orig_w: 原始图像尺寸（用于后续 resize mask 回原图）
    """
    import cv2

    # 1. 灰度读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 回退：BGR → 灰度
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 反色统一
    img = _unify_polarity(img, os.path.basename(image_path))

    # 3. 等比缩放 + pad 正方形
    orig_h, orig_w = img.shape
    scale = IMG_SIZE / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ph, pw = IMG_SIZE - new_h, IMG_SIZE - new_w
    pt = ph // 2
    pl = pw // 2
    img = cv2.copyMakeBorder(img, pt, ph - pt, pl, pw - pl, cv2.BORDER_CONSTANT, value=0)

    # 4. 灰度 → RGB（复制为 3 通道）
    img = np.stack([img] * 3, axis=-1).astype(np.float32) / 255.0

    # 5. CLIP 标准化
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(CLIP_STD,  dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    # 6. (H, W, 3) → (1, 3, H, W)
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return input_tensor, orig_h, orig_w


# ══════════════════════════════════════════════════════════════════════
# 推理核心
# ══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def do_inference(
    image_path: str,
    text_prompts: List[str],
    model,
    processor,
    conf_threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """
    对一张图的所有 prompt 做 CLIPSeg 推理。

    Returns:
        results_by_prompt: {prompt: {"prompt": ..., "score": ..., "instance_count": ..., "rle": ...}}
        width, height: 原始图像尺寸
    """
    # 预处理
    input_tensor, orig_h, orig_w = preprocess_image(image_path)

    results_by_prompt: Dict[str, Dict[str, Any]] = {}

    for prompt in text_prompts:
        # Tokenize 文本
        text_inputs = processor.tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        )
        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)

        # 前向推理
        outputs = model(pixel_values=input_tensor, input_ids=input_ids,
                        attention_mask=attention_mask)
        logits = outputs.logits  # (1, 1, H_pad, W_pad)

        # sigmoid → 二值化
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H_pad, W_pad)

        # Resize 回 pad 后的尺寸（logits 可能和 input 尺寸不完全一致）
        if probs.shape != (IMG_SIZE, IMG_SIZE):
            probs = cv2.resize(probs, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # 反 pad：裁剪回原始等比缩放后的尺寸，再 resize 回原始分辨率
        scale = IMG_SIZE / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        ph, pw = IMG_SIZE - new_h, IMG_SIZE - new_w
        pt = ph // 2
        pl = pw // 2
        probs_unpad = probs[pt:pt + new_h, pl:pl + new_w]
        probs_orig = cv2.resize(probs_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 阈值二值化
        binary_mask = (probs_orig > conf_threshold).astype(np.uint8)

        # score 取 mask 区域内 probs 的最大值
        if binary_mask.sum() > 0:
            max_score = float(probs_orig[binary_mask > 0].max())
            instance_count = 1  # CLIPSeg 输出单通道并集 mask
        else:
            max_score = 0.0
            instance_count = 0

        if instance_count > 0:
            results_by_prompt[prompt] = {
                "prompt": prompt,
                "score": max_score,
                "instance_count": instance_count,
                "rle": mask_to_rle(binary_mask),
            }

    return results_by_prompt, orig_w, orig_h


# ══════════════════════════════════════════════════════════════════════
# 任务处理（与原版完全一致）
# ══════════════════════════════════════════════════════════════════════

def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8") as file_obj:
        tasks = json.load(file_obj)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"第 {index} 条任务不是 JSON 对象")
        if "ann_id" not in task or "image_path" not in task or "text_prompt" not in task:
            raise ValueError(f"第 {index} 条任务缺少 ann_id/image_path/text_prompt 字段")
    return tasks


def process_tasks(
    tasks: List[Dict[str, Any]],
    image_root: str,
    model_dir: str,
    checkpoint_path: Optional[str],
    output_path: str,
    conf_threshold: float,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"[CLIPSeg] 加载模型...")
    model, processor = load_model(model_dir, checkpoint_path)
    model_info = count_model_params(model)
    if checkpoint_path:
        model_info["checkpoint_path"] = checkpoint_path
    print(f"  参数量: {model_info['total_params']:,} / 300M 限制 "
          f"({'✅' if model_info['param_limit_300M'] else '❌ 超标'})")
    print(f"  设备: {DEVICE}")

    # 按图片分组
    tasks_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        tasks_by_image[task["image_path"]].append(task)

    print(f"\n任务总数: {len(tasks)}")
    print(f"图片总数: {len(tasks_by_image)}")
    print(f"置信度阈值: {conf_threshold}")

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}

    for image_rel_path, image_tasks in tasks_by_image.items():
        image_abs_path = os.path.join(image_root, image_rel_path)
        print(f"\n处理图片: {image_rel_path}")

        if not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        unique_prompts = list(
            dict.fromkeys(
                task.get("text_prompt", "").strip()
                for task in image_tasks
                if task.get("text_prompt", "").strip()
            )
        )
        print(f"  prompts ({len(unique_prompts)}): {unique_prompts[:5]}{'...' if len(unique_prompts) > 5 else ''}")

        image_start = time.time()
        results_by_prompt, width, height = do_inference(
            image_path=image_abs_path,
            text_prompts=unique_prompts,
            model=model,
            processor=processor,
            conf_threshold=conf_threshold,
        )
        empty_rle = build_empty_rle(height, width)

        for task in image_tasks:
            ann_id = task["ann_id"]
            prompt = task.get("text_prompt", "").strip()
            prediction = results_by_prompt.get(prompt)
            task_to_rle[ann_id] = prediction["rle"] if prediction is not None else empty_rle

        elapsed = time.time() - image_start
        inference_total_time += elapsed
        processed_images += 1
        hit_count = len(results_by_prompt)
        print(f"  命中: {hit_count}/{len(unique_prompts)} prompts, 耗时 {elapsed:.2f}s")

    # 构建输出
    predictions_output = []
    for task in tasks:
        ann_id = task["ann_id"]
        if ann_id not in task_to_rle:
            raise RuntimeError(f"ann_id {ann_id} 未生成结果")
        predictions_output.append({"ann_id": ann_id, "rle": task_to_rle[ann_id]})

    avg_inference_time = (
        inference_total_time / processed_images if processed_images > 0 else 0.0
    )

    output_data = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_inference_time),
            "processed_images": processed_images,
            "total_tasks": len(tasks),
        },
        "predictions": predictions_output,
    }

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"推理完成!")
    print(f"  输出文件: {output_path_obj}")
    print(f"  纯推理总耗时: {inference_total_time:.2f}s")
    print(f"  平均每张图:   {avg_inference_time:.2f}s")
    print(f"  模型参数量:   {model_info['total_params']:,} (限制 300M)")
    print(f"  {'=' * 60}")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("CLIPSeg 推理配置")
    print("=" * 60)
    print(f"任务文件:     {DEFAULT_TASKS}")
    print(f"图片根目录:   {DEFAULT_IMAGE_ROOT}")
    print(f"输出路径:     {DEFAULT_OUTPUT_PATH}")
    print(f"模型目录:     {DEFAULT_MODEL_DIR}")
    print(f"训练权重:     {DEFAULT_CHECKPOINT_PATH}")
    print(f"置信度阈值:   {DEFAULT_CONF_THRESHOLD}")
    print(f"输入尺寸:     {IMG_SIZE}×{IMG_SIZE}")
    print(f"设备:         {DEVICE}")
    print("=" * 60)

    tasks = load_tasks(DEFAULT_TASKS)
    process_tasks(
        tasks=tasks,
        image_root=DEFAULT_IMAGE_ROOT,
        model_dir=DEFAULT_MODEL_DIR,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        conf_threshold=DEFAULT_CONF_THRESHOLD,
    )


if __name__ == "__main__":
    main()
