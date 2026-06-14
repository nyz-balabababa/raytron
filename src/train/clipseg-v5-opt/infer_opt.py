#!/usr/bin/env python3
"""
CLIPSeg infer-opt 推理入口。

支持：
1. baseline checkpoint
2. opt decoder-only checkpoint
3. prompt fusion
4. hflip TTA
5. postprocess
6. optional conflict suppression
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset_opt import (
    build_empty_rle,
    load_checkpoint_into_model,
    load_clipseg_model,
    load_config,
    mask_to_rle,
    postprocess_probability_map,
    predict_probability_batch,
    preprocess_for_infer,
    resolve_device,
    resolve_prompt_to_class,
    restore_to_original,
    set_seed,
    suppress_class_conflicts,
)


# =========================
# 顶部配置区
# 直接在 IDE 里运行时，优先修改这里
# =========================
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_opt.py"
DEFAULT_TASKS = Path(__file__).resolve().parents[3] / "test" / "json" / "val_tasks1.json"
DEFAULT_CHECKPOINT = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_IMAGE_ROOT = None
DEFAULT_PROMPT_FUSION = None
DEFAULT_TTA = None
DEFAULT_THRESHOLD_JSON = None
DEFAULT_THRESHOLD_MODE = "config"
DEFAULT_POSTPROCESS_OVERRIDE = None   # None / True / False
DEFAULT_CONFLICT_SUPPRESS = False
DEFAULT_DEVICE = None


LOGGER = logging.getLogger("clipseg_v5_infer_opt")


def configure_logging():
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIPSeg infer-opt on any CLIPSeg checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--prompt_fusion", type=str, choices=["none", "weighted_mean", "max"], default=DEFAULT_PROMPT_FUSION)
    parser.add_argument("--tta", type=str, choices=["none", "hflip"], default=DEFAULT_TTA)
    parser.add_argument("--threshold_json", type=Path, default=DEFAULT_THRESHOLD_JSON)
    parser.add_argument("--threshold_mode", type=str, choices=["config", "best_pos", "best_overall"], default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--postprocess", action="store_true", default=(DEFAULT_POSTPROCESS_OVERRIDE is True))
    parser.add_argument("--no_postprocess", action="store_true", default=(DEFAULT_POSTPROCESS_OVERRIDE is False))
    parser.add_argument("--conflict_suppress", action="store_true", default=DEFAULT_CONFLICT_SUPPRESS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser.parse_args()


def resolve_thresholds(cfg, threshold_json: Path | None, threshold_mode: str):
    if threshold_mode == "config" or threshold_json is None:
        return dict(cfg.VAL_THRESHOLDS), None
    if not Path(threshold_json).exists():
        raise FileNotFoundError(f"threshold_json 不存在: {threshold_json}")
    with open(threshold_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if threshold_mode == "best_pos":
        block = payload.get("best_pos_miou_thresholds", {})
    else:
        block = payload.get("best_overall_miou_thresholds", {})
    thresholds = block.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError(f"threshold_json 缺少 {threshold_mode} thresholds")
    resolved = {}
    for class_name in cfg.CLASSES:
        resolved[class_name] = float(thresholds.get(class_name, cfg.VAL_THRESHOLDS.get(class_name, 0.5)))
    return resolved, Path(threshold_json)


def load_tasks(tasks_path: Path):
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")
    return tasks


@torch.no_grad()
def infer_image_prompts(image_path: Path, prompts, model, processor, cfg, thresholds: dict, prompt_fusion: str, tta: str, enable_postprocess: bool, conflict_suppress: bool):
    image_tensor, orig_hw, resize_meta = preprocess_for_infer(image_path, cfg)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    resolved_prompt_map = {prompt: resolve_prompt_to_class(cfg, prompt) for prompt in prompts}
    known_prompts = [resolved_prompt_map[prompt] for prompt in prompts if resolved_prompt_map[prompt] in cfg.CLASSES]
    unknown_prompts = [resolved_prompt_map[prompt] for prompt in prompts if resolved_prompt_map[prompt] not in cfg.CLASSES]
    prompt_probs = {}

    if known_prompts:
        repeated = image_tensor.repeat(len(known_prompts), 1, 1, 1)
        probs = predict_probability_batch(
            model=model,
            processor=processor,
            images=repeated,
            prompt_keys=known_prompts,
            cfg=cfg,
            prompt_fusion=prompt_fusion,
            tta=tta,
        ).cpu().numpy()
        for idx, prompt_key in enumerate(known_prompts):
            prompt_probs[prompt_key] = probs[idx]
        if conflict_suppress:
            prompt_probs = suppress_class_conflicts(prompt_probs, known_prompts)

    if unknown_prompts:
        repeated = image_tensor.repeat(len(unknown_prompts), 1, 1, 1)
        probs = predict_probability_batch(
            model=model,
            processor=processor,
            images=repeated,
            prompt_keys=unknown_prompts,
            cfg=cfg,
            prompt_fusion="none",
            tta=tta,
        ).cpu().numpy()
        for idx, prompt in enumerate(unknown_prompts):
            prompt_probs[prompt] = probs[idx]

    results = {}
    for prompt in prompts:
        prompt_key = resolved_prompt_map[prompt]
        threshold = float(thresholds.get(prompt_key, getattr(cfg, "UNKNOWN_PROMPT_THRESHOLD", 0.5)))
        prob_map = prompt_probs[prompt_key]
        if prompt_key in cfg.CLASSES:
            pred_sq = postprocess_probability_map(prob_map, prompt_key, threshold, cfg, enable_postprocess=enable_postprocess)
        else:
            pred_sq = (prob_map > threshold).astype(np.uint8)
        pred_mask = restore_to_original(pred_sq, orig_hw, resize_meta)
        if pred_mask.sum() > 0:
            results[prompt] = {
                "hit": True,
                "score": round(float(prob_map.max()), 4),
                "threshold": threshold,
                "rle": mask_to_rle(pred_mask),
            }
        else:
            results[prompt] = {"hit": False, "threshold": threshold}
    return results, orig_hw


def main():
    args = parse_args()
    configure_logging()
    cfg = load_config(args.config)
    set_seed(cfg.SEED)

    checkpoint = Path(args.checkpoint) if args.checkpoint else (cfg.TRAIN_OUTPUT_ROOT / cfg.RUN_NAME / "best.pt")
    output_dir = Path(args.output_dir) if args.output_dir else (cfg.INFER_OUTPUT_ROOT / checkpoint.stem)
    image_root = Path(args.image_root) if args.image_root else cfg.IMAGE_ROOT
    prompt_fusion = args.prompt_fusion or cfg.INFER_PROMPT_FUSION
    tta = args.tta or cfg.INFER_TTA
    thresholds, threshold_json_path = resolve_thresholds(cfg, args.threshold_json, args.threshold_mode)
    enable_postprocess = bool(getattr(cfg, "INFER_ENABLE_POSTPROCESS", getattr(cfg, "ENABLE_POSTPROCESS", True)))
    if args.postprocess:
        enable_postprocess = True
    if args.no_postprocess:
        enable_postprocess = False
    conflict_suppress = True if args.conflict_suppress else cfg.ENABLE_CONFLICT_SUPPRESSION

    device = resolve_device(args.device, default_device=cfg.DEVICE, logger=LOGGER)
    model, processor = load_clipseg_model(cfg, device, logger=LOGGER)
    load_checkpoint_into_model(model, checkpoint, logger=LOGGER, map_location="cpu")
    model.eval()
    LOGGER.info("device: %s", device)
    LOGGER.info("threshold_mode=%s threshold_json=%s", args.threshold_mode, threshold_json_path)

    tasks = load_tasks(args.tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_by_image = defaultdict(list)
    for task in tasks:
        tasks_by_image[str(task["image_path"]).replace("\\", "/")].append(task)

    predictions_output = []
    prompt_style_output = []
    timing = {"processed_images": 0, "total_tasks": len(tasks), "inference_seconds": 0.0}

    for image_rel_path, image_tasks in tqdm(tasks_by_image.items(), total=len(tasks_by_image), desc="InferImages"):
        image_abs_path = image_root / image_rel_path
        if not image_abs_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")
        prompts = list(dict.fromkeys(str(task.get("text_prompt", "")).strip() for task in image_tasks if str(task.get("text_prompt", "")).strip()))

        t0 = time.time()
        results_by_prompt, (orig_h, orig_w) = infer_image_prompts(
            image_path=image_abs_path,
            prompts=prompts,
            model=model,
            processor=processor,
            cfg=cfg,
            thresholds=thresholds,
            prompt_fusion=prompt_fusion,
            tta=tta,
            enable_postprocess=enable_postprocess,
            conflict_suppress=conflict_suppress,
        )
        timing["processed_images"] += 1
        timing["inference_seconds"] += time.time() - t0

        empty_rle = build_empty_rle(orig_h, orig_w)
        prompt_style_output.append({"image_path": image_rel_path, "prompts": results_by_prompt})
        for task in image_tasks:
            ann_id = int(task["ann_id"])
            prompt = str(task.get("text_prompt", "")).strip()
            pred = results_by_prompt.get(prompt)
            rle = pred["rle"] if pred and pred.get("hit") else empty_rle
            predictions_output.append({"ann_id": ann_id, "rle": rle})

    timing["avg_inference_seconds_per_image"] = float(timing["inference_seconds"] / max(timing["processed_images"], 1))

    submit_path = output_dir / f"{args.tasks.stem}_submit.json"
    debug_path = output_dir / f"{args.tasks.stem}_debug.json"
    prompt_path = output_dir / f"pred_{args.tasks.stem}_clipseg_opt.json"
    meta_path = output_dir / f"{args.tasks.stem}_meta.json"

    with open(submit_path, "w", encoding="utf-8") as f:
        json.dump(predictions_output, f, ensure_ascii=False, indent=2)

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": {
                    "checkpoint": str(checkpoint),
                    "config_path": str(cfg.CONFIG_PATH),
                    "run_name": cfg.INFER_RUN_NAME,
                    "classes": cfg.CLASSES,
                    "val_thresholds": thresholds,
                    "threshold_mode": args.threshold_mode,
                    "threshold_json": str(threshold_json_path) if threshold_json_path else None,
                    "unknown_prompt_threshold": float(getattr(cfg, "UNKNOWN_PROMPT_THRESHOLD", 0.5)),
                    "prompt_fusion": prompt_fusion,
                    "tta": tta,
                    "postprocess": bool(enable_postprocess),
                    "conflict_suppress": bool(conflict_suppress),
                },
                "timing": timing,
                "predictions": predictions_output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_style_output, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": str(args.tasks),
                "image_root": str(image_root),
                "checkpoint": str(checkpoint),
                "config_path": str(cfg.CONFIG_PATH),
                "output_dir": str(output_dir),
                "threshold_mode": args.threshold_mode,
                "threshold_json": str(threshold_json_path) if threshold_json_path else None,
                "unknown_prompt_threshold": float(getattr(cfg, "UNKNOWN_PROMPT_THRESHOLD", 0.5)),
                "prompt_fusion": prompt_fusion,
                "tta": tta,
                "postprocess": bool(enable_postprocess),
                "conflict_suppress": bool(conflict_suppress),
                "processed_images": timing["processed_images"],
                "total_tasks": timing["total_tasks"],
                "avg_inference_seconds_per_image": timing["avg_inference_seconds_per_image"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info("完成: images=%d tasks=%d", timing["processed_images"], timing["total_tasks"])
    LOGGER.info("submit_json: %s", submit_path)
    LOGGER.info("debug_json: %s", debug_path)
    LOGGER.info("prompt_json: %s", prompt_path)
    LOGGER.info("meta: %s", meta_path)


if __name__ == "__main__":
    main()
