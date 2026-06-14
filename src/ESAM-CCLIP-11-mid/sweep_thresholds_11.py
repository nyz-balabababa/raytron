#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config_esam_cclip_11 import (
    BATCH_SIZE,
    CLASSES,
    DEVICE,
    EFFICIENT_SAM_CKPT,
    IMAGE_ROOT,
    POSTPROCESS_DEFAULT,
    PROMPT_PROTOTYPES,
    REBUILD_TEXT_CACHE,
    ROOT,
    TOKENIZER_DIR,
    VAL_JSON,
    VAL_LIST,
)
from common_esam_cclip_11 import (
    ESAMCCLIPModel,
    apply_postprocess,
    compute_metrics,
    load_tokenizer,
    maybe_tqdm,
    resolve_device,
    save_json,
)
from dataset_esam_cclip_11 import ESAMCCLIP11Dataset
from model_esam_cclip_11 import load_checkpoint_flexible
from prompt_prototypes import load_or_build_text_cache

LOGGER = logging.getLogger("ESAM_CCLIP_11_SWEEP")

EXPECTED_CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
]

DEFAULT_CHECKPOINT = ROOT / "test" / "train_output" / "ESAM-CCLIP-11-rare-balanced-mid" / "final_fullset.pt"
DEFAULT_TEXT_CACHE_PATH = ROOT / "test" / "cache" / "text_emb_11_rare_balanced_mid.pt"
DEFAULT_SWEEP_MODE = "safe"
DEFAULT_DEVICE = DEVICE
DEFAULT_WRITE_BACK_CHECKPOINT = True
DEFAULT_REBUILD_TEXT_CACHE = True if REBUILD_TEXT_CACHE is False else REBUILD_TEXT_CACHE
DEFAULT_MAX_SAMPLES_PER_CLASS = 500

SAFE_THRESH_GRID = {
    "person": [0.60, 0.65, 0.70],
    "car": [0.60, 0.65, 0.70],
    "building": [0.60, 0.65, 0.70],
    "tree": [0.50, 0.55, 0.60],
    "animal": [0.45, 0.50, 0.55],
    "trash can": [0.30, 0.35, 0.40, 0.45],
    "window": [0.30, 0.35, 0.40, 0.45],
    "door": [0.30, 0.35, 0.40, 0.45],
    "fence": [0.20, 0.25, 0.30, 0.35, 0.40],
    "pole_light": [0.15, 0.20, 0.25, 0.30, 0.35],
    "motorcycle": [0.30, 0.35, 0.40, 0.45],
}

RECALL_THRESH_GRID = {
    "person": [0.55, 0.60, 0.65],
    "car": [0.55, 0.60, 0.65],
    "building": [0.60, 0.65, 0.70],
    "tree": [0.45, 0.50, 0.55],
    "animal": [0.40, 0.45, 0.50],
    "trash can": [0.25, 0.30, 0.35, 0.40],
    "window": [0.25, 0.30, 0.35, 0.40],
    "door": [0.25, 0.30, 0.35, 0.40],
    "fence": [0.15, 0.20, 0.25, 0.30, 0.35],
    "pole_light": [0.10, 0.15, 0.20, 0.25, 0.30],
    "motorcycle": [0.25, 0.30, 0.35, 0.40],
}

MIN_AREA_GRID = {
    "person": [16, 32, 64],
    "car": [16, 32, 64],
    "building": [64, 128, 256],
    "tree": [32, 64, 128],
    "animal": [8, 16, 32],
    "trash can": [2, 4, 8, 16],
    "window": [2, 4, 8, 16],
    "door": [4, 8, 16, 32],
    "fence": [1, 2, 4, 8],
    "pole_light": [1, 2, 4],
    "motorcycle": [2, 4, 8, 16],
}


def resolve_threshold_grid(sweep_mode: str) -> dict:
    if sweep_mode == "safe":
        return SAFE_THRESH_GRID
    if sweep_mode == "recall":
        return RECALL_THRESH_GRID
    raise ValueError(f"unsupported sweep_mode: {sweep_mode}")


def resolve_default_output_dir(sweep_mode: str) -> Path:
    if sweep_mode == "recall":
        return ROOT / "test" / "train_output" / "threshold_sweep_esam_11_mid_recall"
    return ROOT / "test" / "train_output" / "threshold_sweep_esam_11_mid_safe"


def resolve_default_write_back_path(sweep_mode: str) -> Path:
    if sweep_mode == "recall":
        return ROOT / "model" / "submit-rsam-mid-recall" / "sam3.pt"
    return ROOT / "model" / "submit-rsam-mid-safe" / "sam3.pt"


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        with open(tmp_path, "wb") as file_obj:
            torch.save(payload, file_obj)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"temporary checkpoint write failed or file is empty: {tmp_path}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() and tmp_path != path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def configure_logging():
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    LOGGER.addHandler(handler)


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "class_names": [item["class_name"] for item in batch],
    }


@torch.no_grad()
def collect_validation_logits(model, dataset, text_cache_payload, device, max_samples_per_class=500, seed=42):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    records = defaultdict(list)
    seen_counts = defaultdict(int)
    rng = np.random.default_rng(seed)
    for batch in maybe_tqdm(loader, total=len(loader), desc="Collect", leave=False):
        images = batch["images"].to(device)
        image_features = model.encode_image(images)
        text_features = torch.stack(
            [text_cache_payload["embeddings"][class_name] for class_name in batch["class_names"]],
            dim=0,
        ).to(device)
        logits = model.decode(image_features, text_features, target_size=(dataset.img_size[0], dataset.img_size[1]))
        for idx, class_name in enumerate(batch["class_names"]):
            seen_counts[class_name] += 1
            sample = {
                "logit": logits[idx, 0].detach().cpu().numpy(),
                "mask": batch["masks"][idx, 0].detach().cpu().numpy(),
            }
            if max_samples_per_class is None or max_samples_per_class <= 0:
                records[class_name].append(sample)
                continue
            if len(records[class_name]) < int(max_samples_per_class):
                records[class_name].append(sample)
                continue
            replace_index = int(rng.integers(0, seen_counts[class_name]))
            if replace_index < int(max_samples_per_class):
                records[class_name][replace_index] = sample
    return records, seen_counts


def default_cfg_for_class(class_name: str, threshold_grid: dict) -> dict:
    return {
        "threshold": float(threshold_grid[class_name][0]),
        "min_area": int(MIN_AREA_GRID[class_name][0]),
        "fill_holes": bool(POSTPROCESS_DEFAULT[class_name]["fill_holes"]),
    }


def log_sweep_summary(
    sweep_mode: str,
    checkpoint_path: Path,
    best_metric: float,
    best_thresholds: dict,
    best_postprocess: dict,
    write_back_path: Optional[Path],
    threshold_grid: dict,
):
    LOGGER.info("========== Sweep Summary ==========")
    LOGGER.info("sweep_mode: %s", sweep_mode)
    LOGGER.info("checkpoint: %s", checkpoint_path)
    LOGGER.info("best_metric: %.6f", best_metric)
    LOGGER.info("best_thresholds:")
    for class_name in CLASSES:
        LOGGER.info("  %s: %.2f", class_name, float(best_thresholds[class_name]))
    LOGGER.info("best_postprocess:")
    for class_name in CLASSES:
        post_cfg = best_postprocess[class_name]
        LOGGER.info(
            "  %s: min_area=%s, fill_holes=%s",
            class_name,
            post_cfg["min_area"],
            post_cfg["fill_holes"],
        )
    LOGGER.info("write_back_path: %s", write_back_path if write_back_path is not None else "disabled")

    if best_thresholds["pole_light"] == min(threshold_grid["pole_light"]) or best_thresholds["fence"] == min(threshold_grid["fence"]):
        LOGGER.info("如果 pole_light/fence 阈值被选到最低，说明 rare recall 仍偏弱；")
    if any(
        best_thresholds[class_name] == min(threshold_grid[class_name])
        for class_name in ("person", "car", "building", "tree", "animal")
    ):
        LOGGER.info("如果 old 类阈值被选到最低，说明模型偏保守；")
    if any(
        best_postprocess[class_name]["min_area"] == max(MIN_AREA_GRID[class_name])
        for class_name in CLASSES
    ):
        LOGGER.info("如果某类 min_area 被选到最大，说明该类噪声偏多；")
    if any(
        best_postprocess[class_name]["min_area"] == min(MIN_AREA_GRID[class_name])
        for class_name in CLASSES
    ):
        LOGGER.info("如果某类 min_area 被选到最小，说明小目标保留有收益。")


def main():
    if list(CLASSES) != EXPECTED_CLASSES:
        raise RuntimeError(f"CLASSES changed unexpectedly: {CLASSES}")

    configure_logging()
    parser = argparse.ArgumentParser(description="Sweep thresholds for ESAM-CCLIP-11 mid checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val_json", type=Path, default=VAL_JSON)
    parser.add_argument("--val_list", type=Path, default=VAL_LIST)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--tokenizer_dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--text_cache_path", type=Path, default=DEFAULT_TEXT_CACHE_PATH)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE if DEFAULT_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--sweep_mode", choices=["safe", "recall"], default=DEFAULT_SWEEP_MODE)
    parser.add_argument("--write_back_checkpoint", dest="write_back_checkpoint", action="store_true")
    parser.add_argument("--no_write_back_checkpoint", dest="write_back_checkpoint", action="store_false")
    parser.add_argument("--write_back_path", type=Path, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS)
    parser.add_argument("--rebuild_text_cache", dest="rebuild_text_cache", action="store_true")
    parser.add_argument("--no_rebuild_text_cache", dest="rebuild_text_cache", action="store_false")
    parser.set_defaults(
        write_back_checkpoint=DEFAULT_WRITE_BACK_CHECKPOINT,
        rebuild_text_cache=DEFAULT_REBUILD_TEXT_CACHE,
    )
    args = parser.parse_args()

    args.output_dir = args.output_dir or resolve_default_output_dir(args.sweep_mode)
    if args.write_back_checkpoint:
        args.write_back_path = args.write_back_path or resolve_default_write_back_path(args.sweep_mode)
    threshold_grid = resolve_threshold_grid(args.sweep_mode)

    device = resolve_device(args.device)
    LOGGER.info("sweep_mode=%s", args.sweep_mode)
    LOGGER.info("checkpoint=%s", args.checkpoint)
    LOGGER.info("text_cache_path=%s", args.text_cache_path)
    LOGGER.info("output_dir=%s", args.output_dir)
    LOGGER.info("write_back_path=%s", args.write_back_path if args.write_back_path is not None else "disabled")
    LOGGER.info("starting model and checkpoint load")

    model = ESAMCCLIPModel(
        tokenizer_dir=args.tokenizer_dir,
        efficient_sam_ckpt=EFFICIENT_SAM_CKPT,
        freeze_image=True,
        freeze_text=True,
    ).to(device)
    checkpoint, _, _ = load_checkpoint_flexible(model, args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=args.text_cache_path,
        classes=CLASSES,
        prompt_prototypes=PROMPT_PROTOTYPES,
        device=device,
        rebuild=args.rebuild_text_cache,
    )
    if set(text_cache_payload["classes"]) != set(CLASSES):
        raise RuntimeError("text cache does not match current 11 classes, please rebuild")
    for class_name in CLASSES:
        if class_name not in text_cache_payload["embeddings"]:
            raise RuntimeError(f"text cache missing embedding for class: {class_name}")

    img_size = int(checkpoint.get("img_size", 768)) if isinstance(checkpoint, dict) else 768
    dataset = ESAMCCLIP11Dataset(
        annotation_json=args.val_json,
        split_txt=args.val_list,
        image_root=args.image_root,
        img_size=(img_size, img_size),
        classes=CLASSES,
        prompt_prototypes=PROMPT_PROTOTYPES,
        augment_prompt=False,
        hflip_prob=0.0,
        use_conf_filter=False,
        negative_sample_prob=0.0,
        negative_sample_weight=0.0,
        rare_oversample=None,
        training=False,
        seed=42,
    )
    if args.max_samples_per_class is not None and int(args.max_samples_per_class) > 0:
        LOGGER.info("reservoir sampling enabled, max_samples_per_class=%d", int(args.max_samples_per_class))
    else:
        LOGGER.info("using all validation samples")

    LOGGER.info("collecting validation logits")
    records, seen_counts = collect_validation_logits(
        model,
        dataset,
        text_cache_payload,
        device,
        max_samples_per_class=args.max_samples_per_class,
        seed=42,
    )
    for class_name in CLASSES:
        LOGGER.info(
            "sampled %s: kept=%d seen=%d",
            class_name,
            len(records.get(class_name, [])),
            int(seen_counts.get(class_name, 0)),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_thresholds = {}
    best_postprocess = {}
    best_scores = {}
    summary_rows = []

    for class_name in maybe_tqdm(CLASSES, total=len(CLASSES), desc="Sweep classes", leave=False):
        best_iou = -1.0
        best_cfg = default_cfg_for_class(class_name, threshold_grid)
        samples = records.get(class_name, [])

        for threshold in threshold_grid[class_name]:
            for min_area in MIN_AREA_GRID[class_name]:
                fill_holes = bool(POSTPROCESS_DEFAULT[class_name]["fill_holes"])
                ious = []
                for sample in samples:
                    logit = np.clip(sample["logit"], -50, 50)
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    pred = (prob > threshold).astype(np.uint8)
                    pred = apply_postprocess(pred, min_area=min_area, fill_holes=fill_holes)
                    pred_t = torch.from_numpy(pred[None, None].astype(np.float32))
                    mask_t = torch.from_numpy(sample["mask"][None, None].astype(np.float32))
                    metrics = compute_metrics(pred_t, mask_t, threshold=0.5)
                    ious.append(metrics["iou"])

                score = float(np.mean(ious)) if ious else 0.0
                summary_rows.append([class_name, threshold, min_area, fill_holes, score])
                if score > best_iou:
                    best_iou = score
                    best_cfg = {
                        "threshold": float(threshold),
                        "min_area": int(min_area),
                        "fill_holes": fill_holes,
                    }

        best_thresholds[class_name] = best_cfg["threshold"]
        best_postprocess[class_name] = {
            "min_area": best_cfg["min_area"],
            "fill_holes": best_cfg["fill_holes"],
        }
        best_scores[class_name] = float(best_iou if best_iou >= 0.0 else 0.0)
        LOGGER.info(
            "best %s: threshold=%.2f min_area=%s score=%.4f",
            class_name,
            best_thresholds[class_name],
            best_postprocess[class_name]["min_area"],
            best_scores[class_name],
        )

    best_metric = float(np.mean([best_scores[class_name] for class_name in CLASSES])) if CLASSES else 0.0

    with open(args.output_dir / "best_thresholds.json", "w", encoding="utf-8") as file_obj:
        json.dump(best_thresholds, file_obj, ensure_ascii=False, indent=2)
    with open(args.output_dir / "best_postprocess.json", "w", encoding="utf-8") as file_obj:
        json.dump(best_postprocess, file_obj, ensure_ascii=False, indent=2)
    with open(args.output_dir / "threshold_sweep_summary.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["class_name", "threshold", "min_area", "fill_holes", "mean_iou"])
        writer.writerows(summary_rows)

    if args.write_back_checkpoint:
        if not isinstance(checkpoint, dict):
            checkpoint = {"model_state_dict": model.state_dict()}
        checkpoint["classes"] = list(CLASSES)
        checkpoint["prompt_thresholds"] = best_thresholds
        checkpoint["val_thresholds"] = best_thresholds
        checkpoint["postprocess"] = best_postprocess
        checkpoint["postprocess_cfg"] = best_postprocess
        checkpoint["sweep_mode"] = args.sweep_mode
        checkpoint["sweep_metric"] = best_metric
        checkpoint["img_size"] = img_size
        writeback_path = Path(args.write_back_path) if args.write_back_path is not None else args.checkpoint
        atomic_torch_save(checkpoint, writeback_path)
        LOGGER.info("checkpoint write-back saved: %s", writeback_path)
    else:
        writeback_path = None

    save_json(
        args.output_dir / "sweep_summary.json",
        {
            "sweep_mode": args.sweep_mode,
            "checkpoint": str(args.checkpoint),
            "text_cache_path": str(args.text_cache_path),
            "img_size": img_size,
            "write_back_checkpoint": bool(args.write_back_checkpoint),
            "write_back_path": str(writeback_path) if writeback_path is not None else None,
            "max_samples_per_class": args.max_samples_per_class,
            "best_metric": best_metric,
            "best_thresholds": best_thresholds,
            "best_postprocess": best_postprocess,
            "best_thresholds_path": str(args.output_dir / "best_thresholds.json"),
            "best_postprocess_path": str(args.output_dir / "best_postprocess.json"),
        },
    )

    log_sweep_summary(
        sweep_mode=args.sweep_mode,
        checkpoint_path=args.checkpoint,
        best_metric=best_metric,
        best_thresholds=best_thresholds,
        best_postprocess=best_postprocess,
        write_back_path=writeback_path,
        threshold_grid=threshold_grid,
    )


if __name__ == "__main__":
    main()
