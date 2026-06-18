#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DINO_ROOT = PROJECT_ROOT / "src" / "DINO"

for candidate in [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), str(DINO_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("esam11_submit_base", SCRIPT_DIR / "inference.py")
DINO = load_module("dino_hybrid_patch_base", DINO_ROOT / "infer_DINO_ESAM_hybrid.py")
LOGGER = BASE.LOGGER
DINO.LOGGER = LOGGER

DEFAULT_DINO_ENABLED_CLASSES = ["pole_light", "motorcycle"]
DEFAULT_DINO_MIN_PIXELS = {
    "pole_light": 4,
    "motorcycle": 24,
}
DEFAULT_DINO_SCORE_MARGIN = 0.05
DEFAULT_MAX_DINO_CALLS_PER_IMAGE = 2
DEFAULT_DINO_BACKEND = "local"


def parse_key_value_ints(raw: str, default_map: Dict[str, int]) -> Dict[str, int]:
    values = dict(default_map)
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = int(value.strip())
    return values


def parse_key_value_floats(raw: str, default_map: Dict[str, float]) -> Dict[str, float]:
    values = dict(default_map)
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = float(value.strip())
    return values


def parse_enabled_classes(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def should_trigger_dino(
    class_name: Optional[str],
    area: int,
    score: float,
    threshold: float,
    dino_enabled_classes: set[str],
    min_pixels_cfg: Dict[str, int],
    score_margin: float,
) -> Optional[str]:
    if class_name is None or class_name not in dino_enabled_classes:
        return None
    min_pixels = int(min_pixels_cfg.get(class_name, 0))
    if area == 0:
        return "empty"
    if area < min_pixels and score < (threshold + score_margin):
        return "tiny_low_score"
    return None


@torch.inference_mode()
def do_inference_with_dino_patch(
    image_path: str,
    text_prompts: List[str],
    model,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    model_input_size: int,
    thresholds: Dict[str, float],
    postprocess_cfg: Dict[str, Dict[str, Any]],
    prompt_aliases: Dict[str, str],
    default_mask_threshold: float,
    prompt_feature_cache: Dict[str, Tuple[torch.Tensor, Optional[str]]],
    args,
    dino_enabled_classes: set[str],
    dino_min_pixels: Dict[str, int],
    dino_box_thresholds: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, Any]], int, int, List[Dict[str, Any]], int]:
    device = next(model.parameters()).device
    image_tensor, meta = BASE.preprocess_image(image_path, model_input_size)
    image_embedding = model.encode_image(image_tensor.unsqueeze(0).to(device, non_blocking=True))
    results_by_prompt: Dict[str, Dict[str, Any]] = {}
    suspicious_prompts: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    dino_calls_used = 0

    for start in range(0, len(text_prompts), BASE.PROMPT_BATCH_SIZE):
        prompt_batch = text_prompts[start:start + BASE.PROMPT_BATCH_SIZE]
        feature_list: List[torch.Tensor] = []
        mapped_classes: List[Optional[str]] = []
        for prompt_text in prompt_batch:
            feature, mapped_class = BASE.build_text_feature_for_prompt(
                model=model,
                tokenizer=tokenizer,
                text_cache_payload=text_cache_payload,
                prompt_text=prompt_text,
                prompt_aliases=prompt_aliases,
                device=device,
                prompt_feature_cache=prompt_feature_cache,
            )
            feature_list.append(feature)
            mapped_classes.append(mapped_class)

        if not feature_list:
            continue

        text_features = torch.stack(feature_list, dim=0).to(device, non_blocking=True)
        image_batch = image_embedding.expand(text_features.size(0), -1, -1, -1)
        logits = model.decode(
            image_features=image_batch,
            text_features=text_features,
            target_size=(model_input_size, model_input_size),
        )
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"logits 形状异常，期望 [B, 1, H, W]，当前: {tuple(logits.shape)}")

        for prompt_text, mapped_class, prompt_feature, prompt_logits in zip(
            prompt_batch,
            mapped_classes,
            feature_list,
            logits,
        ):
            threshold = BASE.get_prompt_threshold(prompt_text, mapped_class, thresholds, default_mask_threshold)
            post_cfg = BASE.get_prompt_postprocess(prompt_text, mapped_class, postprocess_cfg)
            mask, score = BASE.logits_to_mask(prompt_logits, meta, threshold, model_input_size)
            mask = BASE.apply_postprocess(
                mask,
                min_area=int(post_cfg.get("min_area", 0)),
                fill_holes=bool(post_cfg.get("fill_holes", False)),
            )
            area = int(mask.sum())
            result = {
                "prompt": prompt_text,
                "mapped_class": mapped_class,
                "score": float(score),
                "threshold": float(threshold),
                "area": area,
                "rle": BASE.encode_mask_to_rle(mask) if area > 0 else None,
                "source": "esam",
            }
            results_by_prompt[prompt_text] = result

            reason = should_trigger_dino(
                class_name=mapped_class,
                area=area,
                score=float(score),
                threshold=float(threshold),
                dino_enabled_classes=dino_enabled_classes,
                min_pixels_cfg=dino_min_pixels,
                score_margin=float(args.dino_score_margin),
            )
            if reason is not None:
                suspicious_prompts.append(
                    {
                        "prompt_text": prompt_text,
                        "mapped_class": mapped_class,
                        "prompt_feature": prompt_feature.detach(),
                        "threshold": float(threshold),
                        "post_cfg": post_cfg,
                        "orig_area": area,
                        "orig_score": float(score),
                        "reason": reason,
                    }
                )

    gray_full = None
    for item in suspicious_prompts:
        if dino_calls_used >= int(args.max_dino_calls_per_image):
            debug_records.append(
                {
                    "prompt": item["prompt_text"],
                    "mapped_class": item["mapped_class"],
                    "trigger_reason": item["reason"],
                    "accepted": False,
                    "skipped": "budget_exhausted",
                }
            )
            continue

        if gray_full is None:
            gray_full = DINO.load_teacher_aligned_gray(Path(image_path))

        dino_calls_used += 1
        dino_threshold = float(
            dino_box_thresholds.get(item["mapped_class"] or item["prompt_text"], args.dino_box_threshold)
        )
        try:
            patched_mask, dino_debug = DINO.segment_prompt_with_dino(
                gray_full=gray_full,
                image_path=Path(image_path),
                prompt_text=item["prompt_text"],
                mapped_class=item["mapped_class"],
                prompt_feature=item["prompt_feature"],
                model=model,
                model_input_size=model_input_size,
                threshold=item["threshold"],
                postprocess_cfg=item["post_cfg"],
                args=args,
                device=device,
                dino_box_threshold=dino_threshold,
            )
            patched_area = int(patched_mask.sum())
            accepted = patched_area > item["orig_area"]
            debug_record = {
                "prompt": item["prompt_text"],
                "mapped_class": item["mapped_class"],
                "trigger_reason": item["reason"],
                "orig_area": int(item["orig_area"]),
                "orig_score": float(item["orig_score"]),
                "dino_box_threshold": dino_threshold,
                "patched_area": patched_area,
                "accepted": accepted,
                "dino_raw_boxes": int(dino_debug.get("dino_raw_boxes", 0)),
                "dino_kept_boxes": int(dino_debug.get("dino_kept_boxes", 0)),
            }
            if accepted and patched_area > 0:
                results_by_prompt[item["prompt_text"]] = {
                    "prompt": item["prompt_text"],
                    "mapped_class": item["mapped_class"],
                    "score": float(max(item["orig_score"], max(dino_debug.get("scores", [0.0]) or [0.0]))),
                    "threshold": float(item["threshold"]),
                    "area": patched_area,
                    "rle": BASE.encode_mask_to_rle(patched_mask),
                    "source": "dino_patch",
                }
            debug_records.append(debug_record)
        except Exception as exc:
            debug_records.append(
                {
                    "prompt": item["prompt_text"],
                    "mapped_class": item["mapped_class"],
                    "trigger_reason": item["reason"],
                    "accepted": False,
                    "error": str(exc),
                }
            )

    return results_by_prompt, int(meta["orig_w"]), int(meta["orig_h"]), debug_records, dino_calls_used


def process_tasks_with_dino_patch(
    tasks: List[Dict[str, Any]],
    image_root: str,
    model_dir: str,
    checkpoint_path: Optional[str],
    output_path: str,
    mask_threshold: float,
    tokenizer_path: Optional[str],
    text_cache_path: Optional[str],
    threshold_json: Optional[str],
    postprocess_json: Optional[str],
    fail_safe: bool,
    save_debug_json: bool,
    args,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, text_cache_payload, model_input_size, checkpoint, classes, prompt_aliases = BASE.load_model(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        text_cache_path=text_cache_path,
    )
    device = next(model.parameters()).device
    checkpoint_thresholds = BASE.resolve_checkpoint_thresholds(checkpoint) if isinstance(checkpoint, dict) else BASE.DEFAULT_THRESHOLDS
    checkpoint_postprocess = BASE.resolve_checkpoint_postprocess(checkpoint) if isinstance(checkpoint, dict) else BASE.DEFAULT_POSTPROCESS
    thresholds = checkpoint_thresholds if threshold_json is None else json.loads(Path(threshold_json).read_text(encoding="utf-8"))
    postprocess_cfg = checkpoint_postprocess if postprocess_json is None else json.loads(Path(postprocess_json).read_text(encoding="utf-8"))
    prompt_feature_cache: Dict[str, Tuple[torch.Tensor, Optional[str]]] = {}
    dino_enabled_classes = set(parse_enabled_classes(args.dino_enabled_classes))
    dino_min_pixels = parse_key_value_ints(args.dino_min_pixels, DEFAULT_DINO_MIN_PIXELS)
    dino_box_thresholds = parse_key_value_floats(args.dino_box_threshold_overrides, DINO.DEFAULT_DINO_BOX_THRESHOLDS)

    model_info = BASE.count_model_params(model, device)
    model_info["model_type"] = "EfficientSAM+ChineseCLIP+DINO-Patch"
    model_info["model_dir"] = str(model_dir)
    model_info["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
    model_info["default_mask_threshold"] = float(mask_threshold)
    model_info["model_input_size"] = int(model_input_size)
    model_info["class_mask_thresholds"] = {key: float(value) for key, value in sorted(thresholds.items())}
    model_info["classes"] = list(classes)
    model_info["dino_patch"] = {
        "enabled_classes": sorted(dino_enabled_classes),
        "min_pixels": dino_min_pixels,
        "score_margin": float(args.dino_score_margin),
        "max_dino_calls_per_image": int(args.max_dino_calls_per_image),
        "dino_backend": str(args.dino_backend),
        "fallback_full_image": bool(args.fallback_full_image),
    }

    tasks_by_image = BASE.group_tasks_by_image(tasks)
    LOGGER.info("任务总数: %d", len(tasks))
    LOGGER.info("图片总数: %d", len(tasks_by_image))
    LOGGER.info("DINO patch classes: %s", sorted(dino_enabled_classes))

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}
    failed_items: List[Dict[str, Any]] = []
    empty_mask_count = 0
    class_pred_count: Dict[str, int] = defaultdict(int)
    class_empty_count: Dict[str, int] = defaultdict(int)
    patch_source_count: Dict[str, int] = defaultdict(int)
    dino_debug_records: List[Dict[str, Any]] = []
    total_dino_calls = 0

    for image_rel_path, image_tasks in BASE.maybe_tqdm(
        tasks_by_image.items(),
        total=len(tasks_by_image),
        desc="推理进度",
        leave=False,
    ):
        image_abs_path = BASE.resolve_image_path(image_root, image_rel_path)
        image_start = time.time()
        try:
            unique_prompts = list(dict.fromkeys(BASE.get_prompt_text(task) for task in image_tasks))
            results_by_prompt, width, height, image_debug_records, dino_calls_used = do_inference_with_dino_patch(
                image_path=image_abs_path,
                text_prompts=unique_prompts,
                model=model,
                tokenizer=tokenizer,
                text_cache_payload=text_cache_payload,
                model_input_size=model_input_size,
                thresholds=thresholds,
                postprocess_cfg=postprocess_cfg,
                prompt_aliases=prompt_aliases,
                default_mask_threshold=mask_threshold,
                prompt_feature_cache=prompt_feature_cache,
                args=args,
                dino_enabled_classes=dino_enabled_classes,
                dino_min_pixels=dino_min_pixels,
                dino_box_thresholds=dino_box_thresholds,
            )
            total_dino_calls += int(dino_calls_used)
            if image_debug_records:
                dino_debug_records.extend(
                    [{"image_path": image_rel_path, **record} for record in image_debug_records]
                )
            empty_rle = BASE.empty_mask_rle(height, width)
            for task in image_tasks:
                ann_id = task["ann_id"]
                prompt_text = BASE.get_prompt_text(task)
                prompt_class_key = BASE.map_prompt_to_known_class(prompt_text, prompt_aliases) or prompt_text
                prediction = results_by_prompt.get(prompt_text)
                if prediction is None or int(prediction.get("area", 0)) == 0 or prediction.get("rle") is None:
                    task_to_rle[ann_id] = empty_rle
                    empty_mask_count += 1
                    class_empty_count[prompt_class_key] += 1
                else:
                    task_to_rle[ann_id] = prediction["rle"]
                    pred_class_key = prediction.get("mapped_class") or prompt_class_key
                    class_pred_count[pred_class_key] += 1
                    patch_source_count[str(prediction.get("source", "esam"))] += 1
        except Exception as exc:
            if not fail_safe:
                raise
            width, height = 1, 1
            if os.path.exists(image_abs_path):
                try:
                    with Image.open(image_abs_path) as image_obj:
                        width, height = image_obj.size
                except Exception:
                    pass
            fallback_rle = BASE.empty_mask_rle(height, width)
            for task in image_tasks:
                task_to_rle[task["ann_id"]] = fallback_rle
                empty_mask_count += 1
                failed_items.append(
                    {
                        "ann_id": task["ann_id"],
                        "image_path": image_rel_path,
                        "error": str(exc),
                    }
                )
                prompt_text = BASE.get_prompt_text(task)
                prompt_class_key = BASE.map_prompt_to_known_class(prompt_text, prompt_aliases) or prompt_text
                class_empty_count[prompt_class_key] += 1
        inference_total_time += time.time() - image_start
        processed_images += 1

    predictions_output = [{"ann_id": task["ann_id"], "rle": task_to_rle[task["ann_id"]]} for task in tasks]
    predictions_payload = {"predictions": predictions_output}
    avg_time_per_image = inference_total_time / max(processed_images, 1)
    debug_payload = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_time_per_image),
            "avg_inference_seconds_per_task": float(inference_total_time / max(len(tasks), 1)),
            "processed_images": int(processed_images),
            "total_tasks": int(len(tasks)),
            "empty_mask_count": int(empty_mask_count),
            "failed_count": int(len(failed_items)),
        },
        "class_prediction_stats": {
            "pred_count": {key: int(value) for key, value in sorted(class_pred_count.items())},
            "empty_count": {key: int(value) for key, value in sorted(class_empty_count.items())},
        },
        "dino_patch_stats": {
            "total_dino_calls": int(total_dino_calls),
            "prediction_source_count": {key: int(value) for key, value in sorted(patch_source_count.items())},
        },
    }
    if dino_debug_records:
        debug_payload["dino_debug_records"] = dino_debug_records
    if failed_items:
        debug_payload["failed_items"] = failed_items

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(predictions_payload, file_obj, ensure_ascii=False, separators=(",", ":"))
    if save_debug_json:
        debug_path = output_path_obj.with_name(f"{output_path_obj.stem}_debug.json")
        BASE.save_json(debug_path, debug_payload)
        LOGGER.info("debug 信息已保存: %s", debug_path)

    LOGGER.info("推理完成，输出: %s", output_path_obj)
    LOGGER.info("纯推理总耗时: %.2fs", inference_total_time)
    LOGGER.info("平均每张图耗时: %.2fs", avg_time_per_image)
    LOGGER.info("平均每个 task 耗时: %.4fs", inference_total_time / max(len(tasks), 1))
    LOGGER.info("处理图片数: %d", processed_images)
    LOGGER.info("处理 task 数: %d", len(tasks))
    LOGGER.info("empty_mask_count: %d", empty_mask_count)
    LOGGER.info("failed_count: %d", len(failed_items))
    LOGGER.info("dino_patch_total_calls: %d", total_dino_calls)
    LOGGER.info("prediction_source_count: %s", dict(sorted(patch_source_count.items())))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESAM submit inference with weak-class GroundingDINO patch.")
    parser.add_argument("--tasks", type=str, default=BASE.DEFAULT_TASKS)
    parser.add_argument("--images_root", type=str, default=BASE.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default=BASE.DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_dir", type=str, default=BASE.DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=str, default=BASE.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--text_cache_path", type=str, default=None)
    parser.add_argument("--threshold_json", type=str, default=None)
    parser.add_argument("--postprocess_json", type=str, default=None)
    parser.add_argument("--mask_threshold", type=float, default=BASE.DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save_debug_json", action="store_true")

    parser.add_argument("--dino_enabled_classes", type=str, default=",".join(DEFAULT_DINO_ENABLED_CLASSES))
    parser.add_argument("--dino_min_pixels", type=str, default="pole_light=4,motorcycle=24")
    parser.add_argument("--dino_score_margin", type=float, default=DEFAULT_DINO_SCORE_MARGIN)
    parser.add_argument("--max_dino_calls_per_image", type=int, default=DEFAULT_MAX_DINO_CALLS_PER_IMAGE)

    parser.add_argument("--dino_backend", choices=["auto", "hf", "local"], default=DEFAULT_DINO_BACKEND)
    parser.add_argument("--hf_model_id", type=str, default=DINO.DEFAULT_HF_MODEL_ID)
    parser.add_argument("--groundingdino_config", type=str, default=str(DINO.DEFAULT_GD_CONFIG))
    parser.add_argument("--groundingdino_ckpt", type=str, default=str(DINO.DEFAULT_GD_CKPT))
    parser.add_argument("--hf_box_format", choices=["xyxy_abs", "cxcywh_abs", "cxcywh_norm"], default="xyxy_abs")
    parser.add_argument("--local_box_format", choices=["auto", "xyxy_abs", "cxcywh_abs", "cxcywh_norm"], default="auto")
    parser.add_argument("--dino_box_threshold", type=float, default=DINO.DEFAULT_DINO_BOX_THRESHOLD)
    parser.add_argument("--dino_text_threshold", type=float, default=DINO.DEFAULT_DINO_TEXT_THRESHOLD)
    parser.add_argument("--max_boxes_per_prompt", type=int, default=DINO.DEFAULT_MAX_BOXES_PER_PROMPT)
    parser.add_argument("--box_expand_ratio", type=float, default=DINO.DEFAULT_BOX_EXPAND_RATIO)
    parser.add_argument("--fallback_full_image", action="store_true", default=False)
    parser.add_argument(
        "--dino_box_threshold_overrides",
        type=str,
        default="pole_light=0.18,motorcycle=0.20",
    )
    return parser


def main() -> None:
    BASE.configure_logging()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = build_argparser()
    args = parser.parse_args()

    print("=" * 60)
    print("EfficientSAM + ChineseCLIP + DINO Patch 实验推理")
    print("=" * 60)
    print(f"任务文件: {args.tasks}")
    print(f"图片根目录: {args.images_root}")
    print(f"输出路径: {args.output}")
    print(f"模型目录: {args.model_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"DINO 类别: {args.dino_enabled_classes}")
    print(f"DINO backend: {args.dino_backend}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    tasks = BASE.load_tasks(BASE.resolve_path(args.tasks) or args.tasks)
    process_tasks_with_dino_patch(
        tasks=tasks,
        image_root=BASE.resolve_path(args.images_root) or args.images_root,
        model_dir=BASE.resolve_path(args.model_dir) or args.model_dir,
        checkpoint_path=BASE.resolve_path(args.checkpoint) or args.checkpoint,
        output_path=BASE.resolve_path(args.output) or args.output,
        mask_threshold=args.mask_threshold,
        tokenizer_path=args.tokenizer_path,
        text_cache_path=args.text_cache_path,
        threshold_json=BASE.resolve_path(args.threshold_json) if args.threshold_json else None,
        postprocess_json=BASE.resolve_path(args.postprocess_json) if args.postprocess_json else None,
        fail_safe=not args.strict,
        save_debug_json=args.save_debug_json,
        args=args,
    )


if __name__ == "__main__":
    main()
