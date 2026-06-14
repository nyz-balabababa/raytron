#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPSegProcessor

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config_clipseg_tiny_refine import BASE_MODEL_DIR, BASE_CHECKPOINT, CLASSES, IMAGE_ROOT, IMG_SIZE, PRED_JSON, TRAIN_LIST
from infer_clipseg_tiny_refine import load_refine_checkpoint, predict_refined_probs_for_prompts
from train_clipseg_tiny_refine import parse_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("clipseg_v5_cache")


def safe_class_name(class_name: str) -> str:
    return class_name.replace(" ", "_").replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description="Cache refined CLIPSeg teacher outputs")
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--base_model_dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--pred_json", type=Path, default=PRED_JSON)
    parser.add_argument("--train_list", type=Path, default=TRAIN_LIST)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "test" / "train_output" / "clipseg_refine_soft_cache_11")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = CLIPSegProcessor.from_pretrained(str(args.base_model_dir))
    model = load_refine_checkpoint(args.checkpoint, args.base_model_dir, args.device)
    records = parse_records(
        pred_json=args.pred_json,
        image_list_path=args.train_list,
        image_root=args.image_root,
        classes=CLASSES,
        include_negatives=False,
        negative_ratio=0.0,
        negative_weight=0.0,
        apply_score_filter=False,
        use_manifest=False,
        oversample=None,
    )

    num_total = 0
    num_success = 0
    num_failed = 0
    failed_items = []
    for img_path, class_name, _, _, meta in tqdm(records, desc="cache refined teacher"):
        if not meta["is_positive"]:
            continue
        num_total += 1
        image_path = args.image_root / img_path
        cache_key = str(meta.get("ann_id")) if meta.get("ann_id") is not None else Path(img_path).stem
        cache_path = args.output_dir / f"{cache_key}__{safe_class_name(class_name)}.npy"
        if cache_path.exists() and not args.overwrite:
            num_success += 1
            continue
        try:
            prompt_probs = predict_refined_probs_for_prompts(
                model=model,
                processor=processor,
                image_path=image_path,
                prompts=[class_name],
                device=args.device,
                img_size=IMG_SIZE,
                use_amp=args.device.startswith("cuda"),
            )
            refined_prob = prompt_probs[class_name]
            np.save(cache_path, refined_prob.astype(np.float16))
            num_success += 1
        except Exception as exc:
            num_failed += 1
            failed_items.append({"image_path": img_path, "class_name": class_name, "error": str(exc)})

    if failed_items:
        with open(args.output_dir / "failed_items.json", "w", encoding="utf-8") as file_obj:
            json.dump(failed_items, file_obj, ensure_ascii=False, indent=2)
    summary = {
        "num_total": num_total,
        "num_success": num_success,
        "num_failed": num_failed,
        "classes": CLASSES,
        "checkpoint": str(args.checkpoint),
        "img_size": IMG_SIZE,
        "cache_dtype": "float16",
    }
    with open(args.output_dir / "cache_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("cache done total=%d success=%d failed=%d", num_total, num_success, num_failed)


if __name__ == "__main__":
    main()
