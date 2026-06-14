#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.cuda.amp import autocast
from transformers import CLIPSegProcessor

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

THIS_DIR = Path(__file__).resolve().parent
V4_DIR = ROOT / "src" / "train" / "clipseg-v4"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

from config_clipseg_tiny_refine import BASE_CHECKPOINT, BASE_MODEL_DIR, CLASSES, IMAGE_ROOT, IMG_SIZE, PROMPT_ALIASES, VAL_THRESHOLDS
from clipseg_train import load_teacher_aligned_gray
from models.clipseg_tiny_refine import CLIPSegTinyRefine
from train_clipseg_tiny_refine import build_canvas, compute_edge_map, gray_to_rgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("clipseg_v5_infer")


def load_refine_checkpoint(checkpoint_path: Path, base_model_dir: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CLIPSegTinyRefine(
        base_model_dir=base_model_dir,
        base_checkpoint=None,
        img_size=checkpoint.get("img_size", IMG_SIZE),
        residual_scale=1.0,
        use_edge_map=checkpoint.get("use_edge_map", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, img_size: int):
    gray = load_teacher_aligned_gray(image_path)
    gray_canvas, _ = build_canvas(gray, None, img_size)
    rgb_canvas = gray_to_rgb(gray_canvas)
    gray_tensor = torch.from_numpy((gray_canvas.astype(np.float32) / 255.0).copy()).unsqueeze(0).unsqueeze(0)
    edge_tensor = torch.from_numpy(compute_edge_map(gray_canvas).copy()).unsqueeze(0).unsqueeze(0)
    return rgb_canvas, gray_tensor, edge_tensor


def prompt_to_model_text(prompt: str) -> str:
    if prompt in PROMPT_ALIASES:
        return PROMPT_ALIASES[prompt][0]
    return prompt


def mask_to_rle(mask: np.ndarray):
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.no_grad()
def predict_refined_probs_for_prompts(model, processor, image_path: Path, prompts: list[str], device: str, img_size: int, use_amp: bool):
    rgb_canvas, gray_tensor, edge_tensor = preprocess_image(image_path, img_size)
    model_prompts = [prompt_to_model_text(prompt) for prompt in prompts]
    inputs = processor(
        text=model_prompts,
        images=[rgb_canvas] * len(model_prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    gray_batch = gray_tensor.repeat(len(model_prompts), 1, 1, 1).to(device)
    edge_batch = edge_tensor.repeat(len(model_prompts), 1, 1, 1).to(device)
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)

    with autocast(enabled=use_amp and device.startswith("cuda")):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            gray_image=gray_batch,
            edge_map=edge_batch if model.use_edge_map else None,
        )
    refined_probs = torch.sigmoid(outputs["refined_logits"][:, 0]).detach().cpu().numpy()
    return {prompt: refined_probs[idx] for idx, prompt in enumerate(prompts)}


@torch.no_grad()
def infer_tasks(model, processor, tasks, image_root: Path, device: str, img_size: int, use_amp: bool):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["image_path"]].append(task)

    predictions = {}
    total_prompt_time = 0.0
    total_prompt_count = 0
    for image_path_str, image_tasks in grouped.items():
        image_path = image_root / image_path_str
        prompts = [task.get("text_prompt", task.get("prompt", "")) for task in image_tasks]
        t0 = time.perf_counter()
        prompt_probs = predict_refined_probs_for_prompts(
            model=model,
            processor=processor,
            image_path=image_path,
            prompts=prompts,
            device=device,
            img_size=img_size,
            use_amp=use_amp,
        )
        elapsed = time.perf_counter() - t0
        total_prompt_time += elapsed
        total_prompt_count += len(prompts)

        for task in image_tasks:
            ann_id = str(task["ann_id"])
            prompt = task.get("text_prompt", task.get("prompt", ""))
            prob = prompt_probs[prompt]
            threshold = VAL_THRESHOLDS.get(prompt, 0.5)
            mask = (prob > threshold).astype(np.uint8)
            predictions[ann_id] = {
                "prompt": prompt,
                "rle": mask_to_rle(mask),
            }

    if total_prompt_count > 0:
        LOGGER.info(
            "infer speed: %.2f ms/prompt, %.2f ms/image",
            1000.0 * total_prompt_time / total_prompt_count,
            1000.0 * total_prompt_time / max(len(grouped), 1),
        )
    if torch.cuda.is_available() and device.startswith("cuda"):
        LOGGER.info("gpu max allocated: %.2f GB", torch.cuda.max_memory_allocated() / (1024 ** 3))
    return predictions


def run_benchmark(model, processor, image_path: Path, prompts: list[str], device: str, img_size: int, warmup: int, repeat: int, use_amp: bool):
    for _ in range(warmup):
        predict_refined_probs_for_prompts(model, processor, image_path, prompts, device, img_size, use_amp)
    t0 = time.perf_counter()
    for _ in range(repeat):
        predict_refined_probs_for_prompts(model, processor, image_path, prompts, device, img_size, use_amp)
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "benchmark image=%s prompts=%d repeat=%d avg=%.2f ms/run",
        image_path,
        len(prompts),
        repeat,
        1000.0 * elapsed / max(repeat, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Infer with CLIPSeg Tiny Refine")
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--base_model_dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--test_tasks", type=Path, default=ROOT / "test" / "test_tasks.json")
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "test" / "train_output" / "clipseg_tiny_refine_predictions.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.test_tasks, encoding="utf-8") as file_obj:
        tasks = json.load(file_obj)

    processor = CLIPSegProcessor.from_pretrained(str(args.base_model_dir))
    model = load_refine_checkpoint(args.checkpoint, args.base_model_dir, args.device)

    if args.benchmark and tasks:
        sample_image = args.image_root / tasks[0]["image_path"]
        sample_prompts = [tasks[0].get("text_prompt", tasks[0].get("prompt", "person"))]
        run_benchmark(model, processor, sample_image, sample_prompts, args.device, IMG_SIZE, args.warmup, args.repeat, args.amp)

    predictions = infer_tasks(model, processor, tasks, args.image_root, args.device, IMG_SIZE, args.amp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(predictions, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
