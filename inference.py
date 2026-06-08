#!/usr/bin/env python3
"""
CLIPSeg 提交推理脚本
基于 test_tasks.json 的文本提示进行推理，输出比赛提交格式 predictions.json。

使用方式：
1. 本文件顶部的 DEFAULT_* 常量不可修改
2. 运行 `python3 /raytron/code/inference.py`
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))

from transformers import CLIPSegConfig, CLIPSegForImageSegmentation, CLIPSegProcessor

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("警告: pycocotools 未安装，将使用备用 RLE 编码方法")
    maskUtils = None


DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test/"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_MODEL_DIR = "/raytron/code/model"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"
DEFAULT_MASK_THRESHOLD = 0.5
CLASS_MASK_THRESHOLDS = {
    "person": 0.65,
    "car": 0.65,
    "building": 0.68,
    "tree": 0.56,
    "animal": 0.55,
    "computer": 0.55,
}

DEFAULT_FALLBACK_IMG_SIZE = 352
PROMPT_BATCH_SIZE = 8
BLACKHOT_SKEW_THRESH = -0.3
BLACKHOT_MEAN_THRESH = 200.0
PSEUDO_COLOR_SAT_THRESH = 60.0
STD_LOW = 35.0
STD_MID = 50.0
NOISE_HIGH = 12.0
NOISE_MED = 8.0
BLUR_LOW = 150.0
BLUR_MID = 300.0
CLIP_MEAN = [0.48145466, 0.52048427, 0.45053169]
CLIP_STD = [0.21028575, 0.23535925, 0.22184163]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_hf_cache() -> None:
    """Use a writable local cache inside the container."""
    hf_home = os.environ.get("HF_HOME", "/tmp/huggingface")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)


def load_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT_PATH,
) -> Tuple[CLIPSegForImageSegmentation, CLIPSegProcessor, int]:
    ensure_hf_cache()

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    processor = CLIPSegProcessor.from_pretrained(str(model_dir_path))

    config_path = model_dir_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"缺少 CLIPSeg 配置文件: {config_path}")

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise FileNotFoundError(f"缺少模型权重文件: {checkpoint_path}")

    config = CLIPSegConfig.from_pretrained(str(model_dir_path))
    processor_size = getattr(processor.image_processor, "size", None)
    processor_img_size = None
    if isinstance(processor_size, dict):
        processor_img_size = int(processor_size.get("height") or processor_size.get("shortest_edge") or 0)
    elif isinstance(processor_size, int):
        processor_img_size = int(processor_size)
    model_img_size = int(getattr(config.vision_config, "image_size", 0) or 0)
    pos_embed = state_dict.get("clip.vision_model.embeddings.position_embedding.weight")
    patch_size = int(getattr(config.vision_config, "patch_size", 16) or 16)
    if pos_embed is not None and getattr(pos_embed, "ndim", 0) == 2:
        token_count = int(pos_embed.shape[0])
        grid_tokens = max(token_count - 1, 1)
        grid_size = int(round(grid_tokens ** 0.5))
        if grid_size * grid_size + 1 == token_count:
            model_img_size = grid_size * patch_size
            config.vision_config.image_size = model_img_size
    elif processor_img_size and processor_img_size != model_img_size:
        config.vision_config.image_size = processor_img_size
        model_img_size = processor_img_size
    model = CLIPSegForImageSegmentation(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"警告: checkpoint 缺少 {len(missing)} 个参数")
    if unexpected:
        print(f"警告: checkpoint 多出 {len(unexpected)} 个参数")

    model = model.to(DEVICE)
    model.eval()
    return model, processor, int(model_img_size or processor_img_size or DEFAULT_FALLBACK_IMG_SIZE)


def count_model_params(model) -> Dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "device": DEVICE,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


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


def resolve_image_path(image_root: str, image_rel_path: str) -> str:
    image_rel_path = image_rel_path.replace("\\", "/")
    direct = os.path.join(image_root, image_rel_path)
    if os.path.exists(direct):
        return direct

    if image_rel_path.startswith("test/"):
        stripped = os.path.join(image_root, image_rel_path[5:])
        if os.path.exists(stripped):
            return stripped

    prefixed = os.path.join(image_root, "test", image_rel_path)
    if os.path.exists(prefixed):
        return prefixed

    return direct


def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {image_path}")
    elif is_pseudo_color(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(image_path).replace("\\", "/")
    if "blackHot" in fname:
        img = 255 - img
    else:
        mean = float(np.mean(img))
        std = float(np.std(img))
        if std > 0:
            skew = float(np.mean(((img - mean) / std) ** 3))
            if skew < BLACKHOT_SKEW_THRESH:
                img = 255 - img
        if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            img = 255 - img

    std = float(np.std(img))
    noise_sigma = estimate_noise_sigma(img)
    blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())

    if std < STD_LOW:
        img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
    elif std < STD_MID:
        img = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(img)

    if noise_sigma > NOISE_HIGH:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > NOISE_MED:
        img = cv2.medianBlur(img, 3)

    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < BLUR_MID:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = np.clip(cv2.addWeighted(img, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)

    return img


def preprocess_image(image_path: str, input_size: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    gray = load_teacher_aligned_gray(image_path)

    orig_h, orig_w = gray.shape
    scale = input_size / max(orig_h, orig_w)
    resized_h = max(1, int(round(orig_h * scale)))
    resized_w = max(1, int(round(orig_w * scale)))

    gray = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h = input_size - resized_h
    pad_w = input_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    gray = cv2.copyMakeBorder(
        gray,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(CLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return tensor, meta


def logits_to_mask(
    logits: torch.Tensor,
    meta: Dict[str, int],
    threshold: float,
    input_size: int,
) -> Tuple[np.ndarray, float]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.ndim == 3:
        logits = logits.unsqueeze(1)

    logits = F.interpolate(logits, size=(input_size, input_size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

    top = meta["pad_top"]
    left = meta["pad_left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    prob = prob[top:top + resized_h, left:left + resized_w]

    prob = cv2.resize(prob, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)
    score = float(prob.max()) if prob.size > 0 else 0.0
    mask = (prob >= threshold).astype(np.uint8)
    return mask, score


def get_mask_threshold(prompt: str, default_threshold: float) -> float:
    return float(CLASS_MASK_THRESHOLDS.get(prompt, default_threshold))


@torch.inference_mode()
def do_inference(
    image_path: str,
    text_prompts: List[str],
    model,
    processor,
    model_input_size: int,
    default_mask_threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    image_tensor, meta = preprocess_image(image_path, model_input_size)
    results_by_prompt: Dict[str, Dict[str, Any]] = {}

    for start in range(0, len(text_prompts), PROMPT_BATCH_SIZE):
        prompt_batch = text_prompts[start:start + PROMPT_BATCH_SIZE]
        tokenized = processor.tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        pixel_values = image_tensor.unsqueeze(0).repeat(len(prompt_batch), 1, 1, 1).to(DEVICE)
        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)

        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        for prompt, prompt_logits in zip(prompt_batch, logits):
            threshold = get_mask_threshold(prompt, default_mask_threshold)
            mask, score = logits_to_mask(prompt_logits, meta, threshold, model_input_size)
            if mask.sum() == 0:
                continue
            results_by_prompt[prompt] = {
                "prompt": prompt,
                "score": score,
                "threshold": threshold,
                "rle": mask_to_rle(mask),
            }

    return results_by_prompt, meta["orig_w"], meta["orig_h"]


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    # Accept both plain UTF-8 and UTF-8 with BOM from Windows-generated task files.
    with open(tasks_path, "r", encoding="utf-8-sig") as file_obj:
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
    mask_threshold: float,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model, processor, model_input_size = load_model(model_dir=model_dir, checkpoint_path=checkpoint_path)
    model_info = count_model_params(model)
    model_info["model_type"] = "CLIPSegForImageSegmentation"
    model_info["model_dir"] = model_dir
    model_info["default_mask_threshold"] = float(mask_threshold)
    model_info["model_input_size"] = int(model_input_size)
    model_info["class_mask_thresholds"] = {
        key: float(value) for key, value in sorted(CLASS_MASK_THRESHOLDS.items())
    }
    if checkpoint_path:
        model_info["checkpoint_path"] = checkpoint_path

    tasks_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        tasks_by_image[task["image_path"]].append(task)

    print(f"任务总数: {len(tasks)}")
    print(f"图片总数: {len(tasks_by_image)}")

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}

    pbar = tqdm(
        tasks_by_image.items(),
        total=len(tasks_by_image),
        desc="推理进度",
        unit="img",
    )

    for image_rel_path, image_tasks in pbar:
        image_abs_path = resolve_image_path(image_root, image_rel_path)
        print(f"\n处理图片: {image_rel_path}")
        print(f"  绝对路径: {image_abs_path}")

        if not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        unique_prompts = list(
            dict.fromkeys(
                task.get("text_prompt", "").strip()
                for task in image_tasks
                if task.get("text_prompt", "").strip()
            )
        )
        print(f"  文本提示: {unique_prompts}")

        image_start = time.time()
        results_by_prompt, width, height = do_inference(
            image_path=image_abs_path,
            text_prompts=unique_prompts,
            model=model,
            processor=processor,
            model_input_size=model_input_size,
            default_mask_threshold=mask_threshold,
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
        pbar.set_postfix(
            prompts=f"{len(results_by_prompt)}/{len(unique_prompts)}",
            sec=f"{elapsed:.2f}",
        )
        print(
            f"  完成: {len(results_by_prompt)}/{len(unique_prompts)} 个 prompt 命中, 耗时 {elapsed:.2f}s"
        )

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

    print("\n推理完成!")
    print(f"输出文件: {output_path_obj}")
    print(f"纯推理总耗时: {inference_total_time:.2f}s")
    if processed_images > 0:
        print(f"平均每张图推理耗时: {avg_inference_time:.2f}s")


def main() -> None:
    print("=" * 60)
    print("CLIPSeg 推理配置")
    print("=" * 60)
    print(f"任务文件: {DEFAULT_TASKS}")
    print(f"图片根目录: {DEFAULT_IMAGE_ROOT}")
    print(f"输出路径: {DEFAULT_OUTPUT_PATH}")
    print(f"模型目录: {DEFAULT_MODEL_DIR}")
    print(f"模型检查点: {DEFAULT_CHECKPOINT_PATH}")
    print(f"默认掩码阈值: {DEFAULT_MASK_THRESHOLD}")
    print(f"按类阈值: {CLASS_MASK_THRESHOLDS}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    tasks = load_tasks(DEFAULT_TASKS)
    process_tasks(
        tasks=tasks,
        image_root=DEFAULT_IMAGE_ROOT,
        model_dir=DEFAULT_MODEL_DIR,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        mask_threshold=DEFAULT_MASK_THRESHOLD,
    )


if __name__ == "__main__":
    main()
