#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = str(Path(__file__).resolve().parent)
ESAM_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ESAM_ROOT.parents[1]
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
insert_index = 1
for candidate in [str(ESAM_ROOT), str(PROJECT_ROOT)]:
    if candidate in sys.path:
        sys.path.remove(candidate)
    sys.path.insert(insert_index, candidate)
    insert_index += 1

import numpy as np
import torch

from config_esam_cclip_11 import (
    BATCH_SIZE,
    CLASSES,
    EFFICIENT_SAM_CKPT,
    IMAGE_ROOT,
    MIN_AREA_GRID,
    POSTPROCESS_DEFAULT,
    PROMPT_PROTOTYPES,
    RUN_NAME,
    TEXT_CACHE_PATH,
    THRESH_GRID,
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
from path_utils import ensure_dir, ensure_file, resolve_project_path
from prompt_prototypes import load_or_build_text_cache

LOGGER = logging.getLogger("ESAM_CCLIP_11_SWEEP")

# =========================
# Quick Run Config
# 直接改这里，然后在 IDE 里运行本脚本即可
# =========================
DEFAULT_CHECKPOINT = Path("test/train_output") / RUN_NAME / "final_fullset.pt"
DEFAULT_VAL_JSON = VAL_JSON
DEFAULT_VAL_LIST = VAL_LIST
DEFAULT_IMAGE_ROOT = IMAGE_ROOT
DEFAULT_TOKENIZER_DIR = TOKENIZER_DIR
DEFAULT_TEXT_CACHE_PATH = TEXT_CACHE_PATH.with_name("text_emb_11_rare_balanced.pt")
DEFAULT_OUTPUT_DIR = Path("test/train_output/threshold_sweep_esam_11_rare_balanced")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_WRITE_BACK_CHECKPOINT = True
DEFAULT_WRITE_BACK_PATH = Path("model/submit-rsam-rare-balanced/sam3.pt")
DEFAULT_REBUILD_TEXT_CACHE = True
DEFAULT_MAX_SAMPLES_PER_CLASS = 500

_DATASET_SIGNATURE_LOGGED = False


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


def make_dataset(**kwargs):
    global _DATASET_SIGNATURE_LOGGED
    signature = inspect.signature(ESAMCCLIP11Dataset.__init__)
    supported_keys = set(signature.parameters.keys()) - {"self"}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_keys}
    dropped_keys = sorted(key for key in kwargs.keys() if key not in supported_keys)
    if not _DATASET_SIGNATURE_LOGGED:
        LOGGER.info("dataset module file=%s", inspect.getfile(ESAMCCLIP11Dataset))
        LOGGER.info("dataset init signature=%s", signature)
        _DATASET_SIGNATURE_LOGGED = True
    if dropped_keys:
        LOGGER.warning("当前 dataset 不支持这些参数，已自动忽略: %s", dropped_keys)
    return ESAMCCLIP11Dataset(**filtered_kwargs)


@torch.no_grad()
def collect_validation_logits(model, dataset, tokenizer, text_cache_payload, device, max_samples_per_class=None):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    records = defaultdict(list)
    for batch in maybe_tqdm(loader, total=len(loader), desc="Collect", leave=False):
        images = batch["images"].to(device)
        image_features = model.encode_image(images)
        text_features = torch.stack(
            [text_cache_payload["embeddings"][class_name] for class_name in batch["class_names"]],
            dim=0,
        ).to(device)
        logits = model.decode(image_features, text_features, target_size=(dataset.img_size[0], dataset.img_size[1]))
        for idx, class_name in enumerate(batch["class_names"]):
            if max_samples_per_class is not None and len(records[class_name]) >= int(max_samples_per_class):
                continue
            records[class_name].append(
                {
                    "logit": logits[idx, 0].detach().cpu().numpy(),
                    "mask": batch["masks"][idx, 0].detach().cpu().numpy(),
                }
            )
        if max_samples_per_class is not None:
            if all(len(records[class_name]) >= int(max_samples_per_class) for class_name in CLASSES):
                LOGGER.info("所有类别已收满 max_samples_per_class=%d，提前结束 logits 收集。", int(max_samples_per_class))
                break
    return records


def build_argparser():
    parser = argparse.ArgumentParser(description="Sweep thresholds for ESAM-CCLIP-11 rare-balanced")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val_json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--val_list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--text_cache_path", type=Path, default=DEFAULT_TEXT_CACHE_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--write_back_checkpoint", action="store_true", default=DEFAULT_WRITE_BACK_CHECKPOINT)
    parser.add_argument("--no_write_back_checkpoint", dest="write_back_checkpoint", action="store_false")
    parser.add_argument("--write_back_path", type=Path, default=DEFAULT_WRITE_BACK_PATH)
    parser.add_argument("--rebuild_text_cache", action="store_true", default=DEFAULT_REBUILD_TEXT_CACHE)
    parser.add_argument("--reuse_text_cache", dest="rebuild_text_cache", action="store_false")
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS)
    return parser


def resolve_runtime_paths(args):
    args.checkpoint = ensure_file(args.checkpoint, "checkpoint")
    args.val_json = ensure_file(args.val_json, "val_json")
    if args.val_list is not None:
        args.val_list = ensure_file(args.val_list, "val_list")
    args.image_root = ensure_dir(args.image_root, "image_root")
    args.tokenizer_dir = ensure_dir(args.tokenizer_dir, "tokenizer_dir")
    args.text_cache_path = resolve_project_path(args.text_cache_path)
    args.output_dir = resolve_project_path(args.output_dir)
    args.write_back_path = resolve_project_path(args.write_back_path)
    return args


def main():
    configure_logging()
    parser = build_argparser()
    args = resolve_runtime_paths(parser.parse_args())

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    if args.max_samples_per_class is not None and int(args.max_samples_per_class) <= 0:
        args.max_samples_per_class = None

    device = resolve_device(args.device)
    LOGGER.info("开始加载模型与 checkpoint: %s", args.checkpoint)
    model = ESAMCCLIPModel(
        tokenizer_dir=args.tokenizer_dir,
        efficient_sam_ckpt=EFFICIENT_SAM_CKPT,
        freeze_image=True,
        freeze_text=True,
    ).to(device)
    checkpoint, _, _ = load_checkpoint_flexible(model, args.checkpoint, device, require_decoder_match=True)
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
        raise RuntimeError("text cache 与当前 11 类不一致，请重建 text cache")
    for class_name in CLASSES:
        if class_name not in text_cache_payload["embeddings"]:
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}")

    img_size = int(checkpoint.get("img_size", 768)) if isinstance(checkpoint, dict) else 768
    dataset = make_dataset(
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
        old_class_sample_ratio=1.0,
        rare_class_keep_ratio=1.0,
        training=False,
        seed=42,
    )
    if args.max_samples_per_class is None:
        LOGGER.info("本次 sweep 使用全量验证样本。")
    else:
        LOGGER.info("本次 sweep 每类最多采样 %d 个验证样本。", int(args.max_samples_per_class))
    LOGGER.info("开始收集验证 logits...")
    records = collect_validation_logits(
        model,
        dataset,
        tokenizer,
        text_cache_payload,
        device,
        max_samples_per_class=args.max_samples_per_class,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_thresholds = {}
    best_postprocess = {}
    summary_rows = []
    for class_name in CLASSES:
        best_iou = -1.0
        best_cfg = None
        samples = records.get(class_name, [])
        for threshold in THRESH_GRID:
            for min_area in MIN_AREA_GRID[class_name]:
                fill_holes = bool(POSTPROCESS_DEFAULT[class_name]["fill_holes"])
                ious = []
                for sample in samples:
                    logit = np.clip(sample["logit"], -50.0, 50.0)
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
                    best_cfg = {"threshold": threshold, "min_area": min_area, "fill_holes": fill_holes}
        best_thresholds[class_name] = best_cfg["threshold"] if best_cfg else 0.5
        best_postprocess[class_name] = {
            "min_area": best_cfg["min_area"] if best_cfg else 0,
            "fill_holes": best_cfg["fill_holes"] if best_cfg else False,
        }
        LOGGER.info(
            "best %s: threshold=%.2f min_area=%s score=%.4f",
            class_name,
            best_thresholds[class_name],
            best_postprocess[class_name]["min_area"],
            best_iou,
        )

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
        checkpoint["val_thresholds"] = best_thresholds
        checkpoint["postprocess_cfg"] = best_postprocess
        checkpoint["img_size"] = img_size
        args.write_back_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.write_back_path)
        LOGGER.info("checkpoint write-back saved: %s", args.write_back_path)

    save_json(
        args.output_dir / "sweep_summary.json",
        {
            "checkpoint": str(args.checkpoint),
            "text_cache_path": str(args.text_cache_path),
            "img_size": img_size,
            "write_back_checkpoint": bool(args.write_back_checkpoint),
            "max_samples_per_class": args.max_samples_per_class,
            "best_thresholds_path": str(args.output_dir / "best_thresholds.json"),
            "best_postprocess_path": str(args.output_dir / "best_postprocess.json"),
        },
    )


if __name__ == "__main__":
    main()
