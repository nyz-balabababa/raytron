#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ESAM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ESAM_ROOT.parents[1]
for candidate in [str(ESAM_ROOT), str(PROJECT_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from config_esam_cclip_11 import (
    ALL_JSON,
    AMP,
    apply_preset,
    BATCH_SIZE,
    BCE_WEIGHT,
    CLASS_TO_IDX,
    CLASS_WEIGHTS,
    CLASSES,
    CONF_FILTER,
    DECODER_LR,
    DEVICE,
    DICE_WEIGHT,
    EFFICIENT_SAM_CKPT,
    EPOCHS,
    ESAM_INPUT_SIZE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOCAL_WEIGHT,
    FREEZE_IMAGE_ENCODER,
    FREEZE_TEXT_ENCODER,
    GRAD_CLIP,
    HFLIP_PROB,
    IMAGE_CACHE_DIR,
    IMAGE_LR,
    IMAGE_ROOT,
    IMG_SIZE,
    INCLUDE_NEGATIVE_SAMPLES,
    LOSS_WEIGHT_FLOOR,
    MIN_LR_RATIO,
    MODEL_TYPE,
    NEGATIVE_SAMPLE_RATIO,
    NEGATIVE_SAMPLE_WEIGHT,
    OLD5_CLASSES,
    OLD_CLASS_SAMPLE_RATIO,
    OUTPUT_ROOT,
    PRESET_CHOICES,
    POSTPROCESS_DEFAULT,
    PROMPT_THRESHOLDS,
    PROMPT_PROTOTYPES,
    RARE_CLASSES,
    RARE_BALANCED_OLD_CLASSES,
    RARE_BALANCED_RARE_CLASSES,
    RARE_CLASS_KEEP_RATIO,
    RARE_OVERSAMPLE,
    REBUILD_IMAGE_CACHE,
    REBUILD_TEXT_CACHE,
    RUN_NAME,
    SEED,
    TEXT_CACHE_PATH,
    TOKENIZER_DIR,
    TRAIN_DECODER_ONLY,
    TRAIN_JSON,
    TRAIN_LIST,
    USE_IMAGE_CACHE,
    USE_PROMPT_PROTOTYPE,
    VAL_INCLUDE_NEGATIVE_SAMPLES,
    VAL_JSON,
    VAL_LIST,
    VAL_THRESHOLDS,
    WEAK3_ENABLED,
    WEAK3_SEED,
    WEAK_CLASS_KEEP_RATIO,
    WEAK_CLASS_LOSS_WEIGHT,
    WEIGHT_DECAY,
    WORKERS,
    WARMUP_EPOCHS,
)
from common_esam_cclip_11 import (
    DiceLoss,
    ESAMCCLIPModel,
    FocalLoss,
    count_parameters,
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

# =========================
# Logging
# =========================
LOGGER = logging.getLogger("ESAM_CCLIP_11")


def configure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argparser():
    parser = argparse.ArgumentParser(description="ESAM-CCLIP-11 fast finetune")
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="base")
    parser.add_argument("--train_json", type=Path, default=TRAIN_JSON)
    parser.add_argument("--val_json", type=Path, default=VAL_JSON)
    parser.add_argument("--all_json", type=Path, default=ALL_JSON)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--train_list", type=Path, default=TRAIN_LIST)
    parser.add_argument("--val_list", type=Path, default=VAL_LIST)
    parser.add_argument("--tokenizer_dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--efficient_sam_ckpt", type=Path, default=EFFICIENT_SAM_CKPT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--esam_input_size", type=int, default=ESAM_INPUT_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dry_run_dataset_stats", action="store_true", default=False)
    parser.add_argument("--no_val", action="store_true", default=False)
    parser.add_argument("--with_val", dest="no_val", action="store_false")
    parser.add_argument("--no_train_split_filter", action="store_true", default=False)
    parser.add_argument("--use_train_split_filter", dest="no_train_split_filter", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument(
        "--resume_best",
        type=Path,
        default=None,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume_weights_only", action="store_true", default=False)
    parser.add_argument("--reset_history", action="store_true", default=False)
    parser.add_argument("--no_auto_resume", action="store_true", default=False)
    parser.add_argument("--freeze_image_encoder", dest="freeze_image_encoder", action="store_true")
    parser.add_argument("--unfreeze_image_encoder", dest="freeze_image_encoder", action="store_false")
    parser.add_argument("--unfreeze_image_mode", choices=["none", "last_norm", "last1", "last2"], default="none")
    parser.add_argument("--freeze_text_encoder", action="store_true", default=FREEZE_TEXT_ENCODER)
    parser.add_argument("--train_decoder_only", action="store_true", default=TRAIN_DECODER_ONLY)
    parser.add_argument("--print_trainable_params", dest="print_trainable_params", action="store_true")
    parser.add_argument("--no_print_trainable_params", dest="print_trainable_params", action="store_false")
    parser.add_argument("--use_prompt_prototype", dest="use_prompt_prototype", action="store_true")
    parser.add_argument("--no_prompt_prototype", dest="use_prompt_prototype", action="store_false")
    parser.add_argument("--augment_prompt", dest="augment_prompt", action="store_true")
    parser.add_argument("--no_augment_prompt", dest="augment_prompt", action="store_false")
    parser.add_argument("--prompt_alias_train", dest="prompt_alias_train", action="store_true")
    parser.add_argument("--no_prompt_alias_train", dest="prompt_alias_train", action="store_false")
    parser.add_argument("--prompt_alias_prob", type=float, default=1.0)
    parser.add_argument("--val_augment_prompt", dest="val_augment_prompt", action="store_true")
    parser.add_argument("--no_val_augment_prompt", dest="val_augment_prompt", action="store_false")
    parser.add_argument("--text_cache_path", type=Path, default=TEXT_CACHE_PATH)
    parser.add_argument("--rebuild_text_cache", action="store_true", default=REBUILD_TEXT_CACHE)
    parser.add_argument("--use_image_cache", action="store_true", default=USE_IMAGE_CACHE)
    parser.add_argument("--image_cache_dir", type=Path, default=IMAGE_CACHE_DIR)
    parser.add_argument("--build_image_cache", action="store_true", default=False)
    parser.add_argument("--rebuild_image_cache", action="store_true", default=REBUILD_IMAGE_CACHE)
    parser.add_argument("--negative_sample_ratio", type=float, default=NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--negative_sample_weight", type=float, default=NEGATIVE_SAMPLE_WEIGHT)
    parser.add_argument("--disabled_train_classes", nargs="*", default=None)
    parser.add_argument("--class_keep_ratio_json", type=str, default=None)
    parser.add_argument("--class_loss_scale_json", type=str, default=None)
    parser.add_argument("--old_class_sample_ratio", type=float, default=OLD_CLASS_SAMPLE_RATIO)
    parser.add_argument("--rare_class_keep_ratio", type=float, default=RARE_CLASS_KEEP_RATIO)
    parser.add_argument("--rare_balance_enabled", dest="rare_balance_enabled", action="store_true")
    parser.add_argument("--no_rare_balance_enabled", dest="rare_balance_enabled", action="store_false")
    parser.add_argument("--train_eval_after", action="store_true", default=False)
    parser.add_argument("--train_eval_max_samples", type=int, default=3000)
    parser.add_argument("--train_eval_batch_size", type=int, default=None)
    parser.add_argument("--train_eval_seed", type=int, default=42)
    parser.add_argument("--decoder_lr", type=float, default=DECODER_LR)
    parser.add_argument("--image_lr", type=float, default=IMAGE_LR)
    parser.add_argument("--text_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--min_lr_ratio", type=float, default=MIN_LR_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.set_defaults(
        amp=AMP,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
        use_prompt_prototype=USE_PROMPT_PROTOTYPE,
        rare_balance_enabled=False,
        print_trainable_params=True,
        augment_prompt=False,
        prompt_alias_train=False,
        val_augment_prompt=False,
    )
    return parser


def collect_explicit_dests(parser, argv):
    explicit_dests = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        for action in parser._actions:
            if option in action.option_strings:
                explicit_dests.add(action.dest)
                break
    return explicit_dests


def parse_class_float_mapping(raw_text, arg_name):
    if raw_text is None:
        return None
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return {str(key): float(value) for key, value in payload.items()}


def normalize_weakwindow_args(args):
    if getattr(args, "class_keep_ratio_json", None) is not None:
        args.class_keep_ratio = parse_class_float_mapping(args.class_keep_ratio_json, "--class_keep_ratio_json")
    elif not hasattr(args, "class_keep_ratio") or args.class_keep_ratio is None:
        args.class_keep_ratio = {}
    else:
        args.class_keep_ratio = {str(key): float(value) for key, value in dict(args.class_keep_ratio).items()}

    if getattr(args, "class_loss_scale_json", None) is not None:
        args.class_loss_scale = parse_class_float_mapping(args.class_loss_scale_json, "--class_loss_scale_json")
    elif not hasattr(args, "class_loss_scale") or args.class_loss_scale is None:
        args.class_loss_scale = {}
    else:
        args.class_loss_scale = {str(key): float(value) for key, value in dict(args.class_loss_scale).items()}

    if not hasattr(args, "disabled_train_classes") or args.disabled_train_classes is None:
        args.disabled_train_classes = []
    else:
        args.disabled_train_classes = [str(name) for name in list(args.disabled_train_classes)]
    return args


def normalize_weak3_args(args):
    if not hasattr(args, "weak3_enabled"):
        args.weak3_enabled = bool(WEAK3_ENABLED)
    else:
        args.weak3_enabled = bool(args.weak3_enabled)

    if not hasattr(args, "weak3_seed") or args.weak3_seed is None:
        args.weak3_seed = int(WEAK3_SEED)
    else:
        args.weak3_seed = int(args.weak3_seed)

    if not hasattr(args, "weak_class_keep_ratio") or args.weak_class_keep_ratio is None:
        args.weak_class_keep_ratio = dict(WEAK_CLASS_KEEP_RATIO)
    else:
        args.weak_class_keep_ratio = {str(key): float(value) for key, value in dict(args.weak_class_keep_ratio).items()}

    if not hasattr(args, "weak_class_loss_weight") or args.weak_class_loss_weight is None:
        args.weak_class_loss_weight = dict(WEAK_CLASS_LOSS_WEIGHT)
    else:
        args.weak_class_loss_weight = {str(key): float(value) for key, value in dict(args.weak_class_loss_weight).items()}
    return args


def resolve_runtime_paths(args):
    args.train_json = ensure_file(args.train_json, "train_json")
    args.all_json = resolve_project_path(args.all_json)
    args.image_root = ensure_dir(args.image_root, "image_root")
    if args.dry_run_dataset_stats:
        args.tokenizer_dir = resolve_project_path(args.tokenizer_dir)
        args.efficient_sam_ckpt = resolve_project_path(args.efficient_sam_ckpt)
    else:
        args.tokenizer_dir = ensure_dir(args.tokenizer_dir, "tokenizer_dir")
        args.efficient_sam_ckpt = ensure_file(args.efficient_sam_ckpt, "efficient_sam_ckpt")
    args.output_dir = resolve_project_path(args.output_dir)
    args.text_cache_path = resolve_project_path(args.text_cache_path)
    args.image_cache_dir = resolve_project_path(args.image_cache_dir)
    if args.train_list is not None:
        args.train_list = ensure_file(args.train_list, "train_list")
    if not args.no_val:
        args.val_json = ensure_file(args.val_json, "val_json")
        if args.val_list is not None:
            args.val_list = ensure_file(args.val_list, "val_list")
    else:
        args.val_json = resolve_project_path(args.val_json)
        args.val_list = resolve_project_path(args.val_list) if args.val_list is not None else None
    if args.resume is not None:
        args.resume = ensure_file(args.resume, "resume")
    if args.resume_best is not None:
        args.resume_best = ensure_file(args.resume_best, "resume_best")
    return args


def normalize_unfreeze_args(args):
    if args.unfreeze_image_mode == "none":
        args.freeze_image_encoder = True
        args.image_lr = 0.0
    else:
        args.freeze_image_encoder = False
        if args.image_lr <= 0:
            args.image_lr = 1e-6
    return args


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
        "class_names": [item["class_name"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "is_positive": [item["is_positive"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def build_datasets(args):
    effective_hflip_prob = 0.0 if args.use_image_cache else HFLIP_PROB
    LOGGER.info("[Weak3] disabled_train_classes=%s", args.disabled_train_classes)
    LOGGER.info("[Weak3] class_keep_ratio=%s", args.class_keep_ratio)
    LOGGER.info("[Weak3] class_loss_scale=%s", args.class_loss_scale)
    LOGGER.info("[Weak3] val keeps all classes")
    LOGGER.info("开始构建训练集: %s", args.train_json)
    train_dataset = ESAMCCLIP11Dataset(
        annotation_json=args.train_json,
        split_txt=None if args.no_train_split_filter else args.train_list,
        image_root=args.image_root,
        img_size=(args.img_size, args.img_size),
        classes=CLASSES,
        prompt_prototypes=args.prompt_prototype_cfg,
        augment_prompt=args.augment_prompt,
        prompt_alias_prob=args.prompt_alias_prob,
        hflip_prob=effective_hflip_prob,
        use_conf_filter=False,
        negative_sample_prob=args.negative_sample_ratio if INCLUDE_NEGATIVE_SAMPLES else 0.0,
        negative_sample_weight=args.negative_sample_weight,
        rare_oversample=args.rare_oversample,
        weak3_enabled=args.weak3_enabled,
        weak3_seed=args.weak3_seed,
        weak_class_keep_ratio=args.weak_class_keep_ratio,
        weak_class_loss_weight=args.weak_class_loss_weight,
        disabled_train_classes=args.disabled_train_classes,
        class_keep_ratio=args.class_keep_ratio,
        class_loss_scale=args.class_loss_scale,
        old_class_sample_ratio=args.old_class_sample_ratio,
        rare_class_keep_ratio=args.rare_class_keep_ratio,
        old_classes=args.old_classes,
        rare_classes=args.rare_classes,
        rare_balance_enabled=args.rare_balance_enabled,
        training=True,
        seed=args.seed,
    )
    if len(train_dataset) <= 0:
        raise RuntimeError("train_dataset 为空，请检查 train_json / split filter / prompt 过滤逻辑。")
    val_dataset = None
    if not args.no_val:
        LOGGER.info("开始构建验证集: %s", args.val_json)
        val_dataset = ESAMCCLIP11Dataset(
            annotation_json=args.val_json,
            split_txt=args.val_list,
            image_root=args.image_root,
            img_size=(args.img_size, args.img_size),
            classes=CLASSES,
            prompt_prototypes=args.prompt_prototype_cfg,
            augment_prompt=args.val_augment_prompt,
            prompt_alias_prob=1.0,
            hflip_prob=0.0,
            use_conf_filter=False,
            negative_sample_prob=0.0,
            negative_sample_weight=0.0,
            rare_oversample=None,
            weak3_enabled=False,
            weak3_seed=args.weak3_seed,
            weak_class_keep_ratio={},
            weak_class_loss_weight={},
            old_class_sample_ratio=1.0,
            rare_class_keep_ratio=1.0,
            old_classes=args.old_classes,
            rare_classes=args.rare_classes,
            rare_balance_enabled=False,
            training=False,
            seed=args.seed,
        )
        if len(val_dataset) <= 0:
            raise RuntimeError("val_dataset 为空，请检查 val_json / val_list / prompt 过滤逻辑。")
    LOGGER.info("train class positive stats: %s", train_dataset.stats["positive"])
    LOGGER.info("train class negative stats: %s", train_dataset.stats["negative"])
    if val_dataset is not None:
        LOGGER.info("val class positive stats: %s", val_dataset.stats["positive"])
    LOGGER.info("effective train hflip_prob=%s", effective_hflip_prob)
    return train_dataset, val_dataset


def build_dry_run_summary(train_dataset, val_dataset):
    summary = {
        "train": train_dataset.stats,
        "val": val_dataset.stats if val_dataset is not None else None,
        "weak3": {},
    }
    for class_name in ("window", "door", "pole_light"):
        raw = int(train_dataset.stats.get("positive_raw", {}).get(class_name, 0))
        kept = int(train_dataset.stats.get("positive_kept", {}).get(class_name, 0))
        dropped = int(train_dataset.stats.get("dropped_by_weak3", {}).get(class_name, 0))
        keep_ratio_actual = (kept / raw) if raw > 0 else None
        summary["weak3"][class_name] = {
            "raw": raw,
            "kept": kept,
            "dropped": dropped,
            "keep_ratio_actual": keep_ratio_actual,
        }
    return summary


def build_optimizer(model, decoder_lr, image_lr, text_lr, weight_decay):
    image_params, text_params, decoder_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_encoder"):
            image_params.append(param)
        elif name.startswith("txt_encoder"):
            text_params.append(param)
        elif name.startswith("decoder"):
            decoder_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder"})
    if image_params and image_lr > 0:
        param_groups.append({"params": image_params, "lr": image_lr, "weight_decay": weight_decay, "name": "image"})
    if text_params and text_lr > 0:
        param_groups.append({"params": text_params, "lr": text_lr, "weight_decay": weight_decay, "name": "text"})
    if other_params:
        param_groups.append({"params": other_params, "lr": decoder_lr, "weight_decay": weight_decay, "name": "other"})
    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查冻结设置。")
    LOGGER.info("[ParamGroupSummary] decoder_groups=%d image_groups=%d text_groups=%d other_groups=%d",
                1 if decoder_params else 0,
                1 if image_params and image_lr > 0 else 0,
                1 if text_params and text_lr > 0 else 0,
                1 if other_params else 0)
    LOGGER.info("[ParamGroupSummary] decoder_params=%.2fM image_params=%.2fM text_params=%.2fM other_params=%.2fM",
                sum(p.numel() for p in decoder_params) / 1e6,
                sum(p.numel() for p in image_params) / 1e6,
                sum(p.numel() for p in text_params) / 1e6,
                sum(p.numel() for p in other_params) / 1e6)
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"]) / 1e6
        LOGGER.info("[ParamGroup] %s lr=%s params=%.2fM", group["name"], group["lr"], n_params)
    return optim.AdamW(
        [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in param_groups],
        weight_decay=weight_decay,
    )


def _extract_last_block_prefixes(param_names, block_token: str):
    block_indices = []
    for name in param_names:
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == block_token and idx + 1 < len(parts):
                try:
                    block_idx = int(parts[idx + 1])
                except ValueError:
                    continue
                block_indices.append((block_idx, ".".join(parts[: idx + 2])))
    unique = {}
    for block_idx, prefix in block_indices:
        unique[block_idx] = prefix
    return [unique[idx] for idx in sorted(unique.keys())]


def set_partial_image_encoder_trainable(model, mode: str):
    for param in model.txt_encoder.parameters():
        param.requires_grad = False
    for param in model.img_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    warnings = []
    matched_names = []
    if mode == "none":
        return warnings, matched_names

    named_params = list(model.img_encoder.named_parameters())
    all_names = [name for name, _ in named_params]

    def enable_by_predicate(predicate):
        local_matched = []
        for name, param in named_params:
            if predicate(name):
                param.requires_grad = True
                local_matched.append(name)
        return local_matched

    if mode == "last_norm":
        keywords = ("norm", "neck", "adapter", "output", "proj")
        keyword_matches = [name for name in all_names if any(keyword in name.lower() for keyword in keywords)]
        tail_names = set(keyword_matches[-32:]) if keyword_matches else set()
        matched_names = enable_by_predicate(lambda name: name in tail_names)
        if not matched_names:
            warning = "unfreeze_image_mode=last_norm 未匹配到 image encoder 的 norm/neck/adapter/output/proj 参数。"
            warnings.append(warning)
            LOGGER.warning(warning)
        return warnings, matched_names

    prefixes = _extract_last_block_prefixes(all_names, "blocks")
    if not prefixes:
        prefixes = _extract_last_block_prefixes(all_names, "layers")

    if not prefixes:
        warning = f"unfreeze_image_mode={mode} 未识别到 image encoder blocks/layers，fallback 到 last_norm。"
        warnings.append(warning)
        LOGGER.warning(warning)
        return set_partial_image_encoder_trainable(model, "last_norm")

    n_blocks = 1 if mode == "last1" else 2
    selected_prefixes = prefixes[-n_blocks:]
    matched_names = enable_by_predicate(lambda name: any(name.startswith(prefix + ".") for prefix in selected_prefixes))
    if not matched_names:
        fallback_mode = "last1" if mode == "last2" else "last_norm"
        warning = f"unfreeze_image_mode={mode} 未成功打开 block 参数，fallback 到 {fallback_mode}。"
        warnings.append(warning)
        LOGGER.warning(warning)
        return set_partial_image_encoder_trainable(model, fallback_mode)
    return warnings, matched_names


def log_trainable_parameter_summary(model, args):
    trainable_names = []
    image_trainable_names = []
    decoder_trainable_names = []
    text_trainable_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_names.append(name)
        if name.startswith("img_encoder"):
            image_trainable_names.append(name)
        elif name.startswith("decoder"):
            decoder_trainable_names.append(name)
        elif name.startswith("txt_encoder"):
            text_trainable_names.append(name)

    total_params, trainable_params = count_parameters(model)
    image_trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("img_encoder")) / 1e6
    decoder_trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("decoder")) / 1e6
    text_trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("txt_encoder")) / 1e6

    LOGGER.info("unfreeze_image_mode=%s", args.unfreeze_image_mode)
    LOGGER.info("image_lr=%s", args.image_lr)
    LOGGER.info("trainable total params=%.2fM", trainable_params)
    LOGGER.info("trainable decoder params=%.2fM", decoder_trainable_params)
    LOGGER.info("trainable image_encoder params=%.2fM", image_trainable_params)
    LOGGER.info("trainable text_encoder params=%.2fM", text_trainable_params)
    preview = trainable_names[:50]
    LOGGER.info("trainable parameter names preview(first %d): %s", len(preview), preview)

    total_param_count = max(total_params * 1e6, 1.0)
    image_ratio = (image_trainable_params * 1e6) / total_param_count
    if args.unfreeze_image_mode != "none" and image_trainable_params <= 0:
        raise RuntimeError("unfreeze_image_mode 已开启，但 image_encoder trainable params = 0，说明没有成功解冻。")
    if args.unfreeze_image_mode == "last_norm" and image_trainable_params > 5.0:
        LOGGER.warning(
            "last_norm opened %.2fM image params (>5M)，这对档位A偏多，请检查匹配规则。",
            image_trainable_params,
        )
    if image_ratio > 0.20:
        LOGGER.warning("image_encoder trainable params ratio is high: %.2f%% (>20%%)", image_ratio * 100.0)
    if args.unfreeze_image_mode != "none" and args.image_lr > args.decoder_lr:
        LOGGER.warning("image_lr (%.2e) > decoder_lr (%.2e)，请确认这不是误设。", args.image_lr, args.decoder_lr)
    if text_trainable_params > 0:
        raise RuntimeError("text_encoder trainable params must stay 0.")


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def class_weight_tensor(class_names, device, class_weights):
    weights = [class_weights.get(name, 1.0) for name in class_names]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_text_features(batch_text_keys, text_cache_payload, device):
    embeddings = text_cache_payload.get("embeddings", {})
    alias_embeddings = text_cache_payload.get("alias_embeddings") or text_cache_payload.get("prompt_embeddings") or {}
    alias_to_class = text_cache_payload.get("_alias_to_class")
    if alias_to_class is None:
        alias_to_class = {}
        for class_name, aliases in text_cache_payload.get("aliases", {}).items():
            for alias in aliases:
                alias_to_class[str(alias)] = str(class_name)
        text_cache_payload["_alias_to_class"] = alias_to_class
    features = []
    missing = []
    for text_key in batch_text_keys:
        text_key = str(text_key)
        if text_key in alias_embeddings:
            features.append(alias_embeddings[text_key])
        elif text_key in embeddings:
            features.append(embeddings[text_key])
        else:
            class_name = alias_to_class.get(text_key)
            if class_name is not None and class_name in embeddings:
                features.append(embeddings[class_name])
            else:
                missing.append(text_key)
    if missing:
        raise RuntimeError(f"text cache 缺少文本 embedding: {missing[:10]}，请使用 --rebuild_text_cache")
    return torch.stack(features, dim=0).to(device)


def encode_images_with_optional_cache(model, images, image_paths, cache_dir, use_cache, build_cache, rebuild_cache):
    hit_count = 0
    features = []
    failed = []
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        cache_key = hashlib.md5(str(image_path).replace("\\", "/").encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.pt"
        feature = None
        if use_cache and cache_path.exists() and not rebuild_cache:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                feature = payload["image_embedding"]
                hit_count += 1
            except Exception as exc:
                failed.append({"image_path": image_path, "error": str(exc)})
        if feature is None:
            feature = model.encode_image(images[idx: idx + 1]).detach().cpu()
            if use_cache and build_cache:
                try:
                    torch.save(
                        {
                            "image_path": image_path,
                            "image_embedding": feature,
                            "input_size": list(images.shape[-2:]),
                        },
                        cache_path,
                    )
                except Exception as exc:
                    failed.append({"image_path": image_path, "error": str(exc)})
        features.append(feature[0])
    return torch.stack(features, dim=0).to(images.device), hit_count, failed


def train_one_epoch(model, loader, optimizer, scheduler, criterion_dice, criterion_focal, scaler, device, text_cache_payload, args):
    model.train()
    if args.freeze_image_encoder:
        model.img_encoder.eval()
    if args.freeze_text_encoder:
        model.txt_encoder.eval()
    running = defaultdict(float)
    cache_hits = 0
    cache_total = 0
    failed_cache_items = []
    start = time.time()
    iterator = maybe_tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in iterator:
        batch_start = time.time()
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        class_names = batch["class_names"]
        if args.prompt_alias_train and "prompt_texts" in batch:
            text_keys = batch["prompt_texts"]
        else:
            text_keys = class_names

        if args.use_image_cache:
            image_features, hit_count, failed = encode_images_with_optional_cache(
                model=model,
                images=images,
                image_paths=batch["image_paths"],
                cache_dir=args.image_cache_dir,
                use_cache=True,
                build_cache=args.build_image_cache,
                rebuild_cache=args.rebuild_image_cache,
            )
            cache_hits += hit_count
            cache_total += len(batch["image_paths"])
            failed_cache_items.extend(failed)
        else:
            image_features = model.encode_image(images)

        text_features = get_text_features(text_keys, text_cache_payload, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model.decode(image_features=image_features, text_features=text_features, target_size=(args.img_size, args.img_size))
            bce_map = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=(1, 2, 3))
            dice_val = torch.stack([criterion_dice(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
            focal_val = torch.stack([criterion_focal(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
            cls_weights = class_weight_tensor(class_names, device, args.class_weights)
            weighted = (BCE_WEIGHT * bce_map + DICE_WEIGHT * dice_val + FOCAL_WEIGHT * focal_val) * sample_weight * cls_weights
            loss = weighted.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running["loss"] += loss.item()
        running["bce"] += bce_map.mean().item()
        running["dice"] += dice_val.mean().item()
        running["focal"] += focal_val.mean().item()
        running["batch_time"] += (time.time() - batch_start)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    if failed_cache_items:
        save_json(args.output_dir / args.run_name / "failed_image_cache.json", failed_cache_items)
    denom = max(len(loader), 1)
    running["loss"] /= denom
    running["bce"] /= denom
    running["dice"] /= denom
    running["focal"] /= denom
    running["batch_time"] /= denom
    running["epoch_time"] = time.time() - start
    running["image_cache_hit_rate"] = float(cache_hits / max(cache_total, 1)) if cache_total > 0 else 0.0
    return running


@torch.no_grad()
def validate(model, loader, criterion_dice, criterion_focal, device, text_cache_payload, args):
    model.eval()
    metrics = defaultdict(list)
    start = time.time()
    iterator = maybe_tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        class_names = batch["class_names"]
        image_features = model.encode_image(images)
        text_features = get_text_features(class_names, text_cache_payload, device)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model.decode(image_features=image_features, text_features=text_features, target_size=(args.img_size, args.img_size))
        bce_map = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=(1, 2, 3))
        dice_val = torch.stack([criterion_dice(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
        focal_val = torch.stack([criterion_focal(logits[i: i + 1], masks[i: i + 1]) for i in range(logits.size(0))], dim=0).view(-1)
        metrics["bce"].append(float(bce_map.mean().item()))
        metrics["dice_loss"].append(float(dice_val.mean().item()))
        metrics["focal"].append(float(focal_val.mean().item()))
        for idx, class_name in enumerate(class_names):
            score = compute_metrics(logits[idx: idx + 1], masks[idx: idx + 1], threshold=VAL_THRESHOLDS.get(class_name, 0.5))
            metrics["iou/overall"].append(score["iou"])
            metrics["dice/overall"].append(score["dice"])
            metrics["precision/overall"].append(score["precision"])
            metrics["recall/overall"].append(score["recall"])
            metrics["pred_area/overall"].append(score["pred_area"])
            metrics["gt_area/overall"].append(score["gt_area"])
            metrics[f"iou/{class_name}"].append(score["iou"])
            metrics[f"precision/{class_name}"].append(score["precision"])
            metrics[f"recall/{class_name}"].append(score["recall"])
            if class_name in OLD5_CLASSES:
                metrics["iou/old5"].append(score["iou"])
            if class_name in RARE_CLASSES:
                metrics["iou/rare"].append(score["iou"])
            if batch["is_positive"][idx]:
                metrics["iou/pos_only"].append(score["iou"])
    summary = {key: float(np.mean(values)) for key, values in metrics.items() if values}
    summary["val_time"] = time.time() - start
    return summary


def plot_results(history, save_path, no_val=False):
    fig = plt.figure(figsize=(16, 8))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="train_loss")
    if no_val:
        ax1.plot(epochs, history["train_bce"], label="train_bce")
    else:
        ax1.plot(epochs, history["val_miou_all11"], label="val_miou_all11")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    if no_val:
        ax2.plot(epochs, history["train_dice"], label="train_dice")
        ax2.plot(epochs, history["train_focal"], label="train_focal")
    else:
        ax2.plot(epochs, history["val_miou_old5"], label="old5")
        ax2.plot(epochs, history["val_miou_rare"], label="rare")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    if no_val:
        ax3.plot(epochs, history["image_cache_hit_rate"], label="image_cache_hit_rate")
    else:
        ax3.plot(epochs, history["val_dice"], label="dice")
        ax3.plot(epochs, history["val_pos_miou"], label="pos_miou")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(epochs, history["lr"], label="lr", color="red")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def evaluate_train_diagnostic(model, loader, device):
    model.eval()
    metrics = defaultdict(list)
    per_class = {
        class_name: {
            "iou": [],
            "precision": [],
            "recall": [],
            "pred_area": [],
            "gt_area": [],
        }
        for class_name in CLASSES
    }
    iterator = maybe_tqdm(loader, total=len(loader), desc="TrainDiag", leave=False)
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        class_names = batch["class_names"]
        image_features = model.encode_image(images)
        text_features = torch.stack(
            [loader.text_cache_payload["embeddings"][class_name] for class_name in class_names],
            dim=0,
        ).to(device)
        logits = model.decode(
            image_features=image_features,
            text_features=text_features,
            target_size=(loader.diag_img_size, loader.diag_img_size),
        )
        for idx, class_name in enumerate(class_names):
            threshold = VAL_THRESHOLDS.get(class_name, 0.5)
            score = compute_metrics(logits[idx: idx + 1], masks[idx: idx + 1], threshold=threshold)
            metrics["iou/overall"].append(score["iou"])
            metrics["precision/overall"].append(score["precision"])
            metrics["recall/overall"].append(score["recall"])
            if class_name in OLD5_CLASSES:
                metrics["iou/old5"].append(score["iou"])
            if class_name in RARE_CLASSES:
                metrics["iou/rare"].append(score["iou"])
            if batch["is_positive"][idx]:
                metrics["iou/pos_only"].append(score["iou"])
            per_class[class_name]["iou"].append(score["iou"])
            per_class[class_name]["precision"].append(score["precision"])
            per_class[class_name]["recall"].append(score["recall"])
            per_class[class_name]["pred_area"].append(score["pred_area"])
            per_class[class_name]["gt_area"].append(score["gt_area"])

    summary = {
        "train_mIoU_all11_on_pseudo": float(np.mean(metrics["iou/overall"])) if metrics["iou/overall"] else 0.0,
        "train_mIoU_old5_on_pseudo": float(np.mean(metrics["iou/old5"])) if metrics["iou/old5"] else 0.0,
        "train_mIoU_rare_on_pseudo": float(np.mean(metrics["iou/rare"])) if metrics["iou/rare"] else 0.0,
        "train_pos_mIoU_on_pseudo": float(np.mean(metrics["iou/pos_only"])) if metrics["iou/pos_only"] else 0.0,
        "per_class_iou": {},
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_pred_area": {},
        "per_class_gt_area": {},
    }
    per_class_rows = []
    for class_name in CLASSES:
        class_summary = {
            "iou": float(np.mean(per_class[class_name]["iou"])) if per_class[class_name]["iou"] else 0.0,
            "precision": float(np.mean(per_class[class_name]["precision"])) if per_class[class_name]["precision"] else 0.0,
            "recall": float(np.mean(per_class[class_name]["recall"])) if per_class[class_name]["recall"] else 0.0,
            "pred_area": float(np.mean(per_class[class_name]["pred_area"])) if per_class[class_name]["pred_area"] else 0.0,
            "gt_area": float(np.mean(per_class[class_name]["gt_area"])) if per_class[class_name]["gt_area"] else 0.0,
        }
        summary["per_class_iou"][class_name] = class_summary["iou"]
        summary["per_class_precision"][class_name] = class_summary["precision"]
        summary["per_class_recall"][class_name] = class_summary["recall"]
        summary["per_class_pred_area"][class_name] = class_summary["pred_area"]
        summary["per_class_gt_area"][class_name] = class_summary["gt_area"]
        per_class_rows.append(
            [
                class_name,
                class_summary["iou"],
                class_summary["precision"],
                class_summary["recall"],
                class_summary["pred_area"],
                class_summary["gt_area"],
            ]
        )
    return summary, per_class_rows


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": dict(history),
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {idx: cls for idx, cls in enumerate(CLASSES)},
        "model_type": MODEL_TYPE,
        "preset": args.preset,
        "use_prompt_prototype": args.use_prompt_prototype,
        "prompt_prototypes": args.prompt_prototype_cfg,
        "prompt_thresholds": PROMPT_THRESHOLDS,
        "val_thresholds": VAL_THRESHOLDS,
        "postprocess": POSTPROCESS_DEFAULT,
        "postprocess_cfg": POSTPROCESS_DEFAULT,
        "class_weights": args.class_weights,
        "rare_oversample": args.rare_oversample,
        "weak3_enabled": args.weak3_enabled,
        "weak3_seed": args.weak3_seed,
        "weak_class_keep_ratio": args.weak_class_keep_ratio,
        "weak_class_loss_weight": args.weak_class_loss_weight,
        "disabled_train_classes": args.disabled_train_classes,
        "class_keep_ratio": args.class_keep_ratio,
        "class_loss_scale": args.class_loss_scale,
        "negative_sample_ratio": args.negative_sample_ratio,
        "negative_sample_weight": args.negative_sample_weight,
        "old_class_sample_ratio": args.old_class_sample_ratio,
        "rare_class_keep_ratio": args.rare_class_keep_ratio,
        "rare_balance_enabled": args.rare_balance_enabled,
        "train_json": str(args.train_json),
        "val_json": str(args.val_json) if args.val_json is not None else None,
        "train_list": str(args.train_list) if args.train_list is not None else None,
        "val_list": str(args.val_list) if args.val_list is not None else None,
        "img_size": int(args.img_size),
        "esam_input_size": int(args.esam_input_size),
        "text_cache_path": str(args.text_cache_path),
        "run_name": args.run_name,
        "unfreeze_image_mode": args.unfreeze_image_mode,
        "image_lr": float(args.image_lr),
    }
    torch.save(payload, path)


def resolve_resume_checkpoint(args, run_dir: Path):
    if args.resume is not None:
        return Path(args.resume)
    if args.resume_best is not None:
        return None
    auto_resume_path = run_dir / "last.pt"
    if not args.no_auto_resume and auto_resume_path.exists():
        return auto_resume_path
    return None


def resume_training_state(model, optimizer, scheduler, checkpoint_path, device, args):
    checkpoint_path = Path(checkpoint_path)
    LOGGER.info("开始恢复断点续训: %s", checkpoint_path)
    checkpoint, missing_keys, unexpected_keys = load_checkpoint_flexible(
        model,
        checkpoint_path,
        device,
        require_decoder_match=True,
    )
    load_info = getattr(model, "_last_checkpoint_load_info", {})
    LOGGER.info("resume checkpoint path=%s", load_info.get("checkpoint_path", checkpoint_path))
    LOGGER.info("resume decoder_matched_keys=%d", len(load_info.get("decoder_matched_keys", [])))
    LOGGER.info("resume missing_keys=%s", missing_keys[:30])
    LOGGER.info("resume unexpected_keys=%s", unexpected_keys[:30])

    if args.resume_weights_only:
        LOGGER.info("resume_weights_only=True，仅恢复模型权重，不恢复 optimizer/scheduler/epoch/history。")
        return 1, defaultdict(list), {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0, "best_pos_only": -1.0}, checkpoint

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    if args.reset_history:
        history = defaultdict(list)
        best_metrics = {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0, "best_pos_only": -1.0}
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("reset_history=True，已清空 history/best_metrics，从 epoch %d 继续训练。", start_epoch)
    else:
        history_payload = checkpoint.get("history", {})
        history = defaultdict(list)
        for key, values in history_payload.items():
            history[key] = list(values)
        best_metrics = checkpoint.get(
            "best_metrics",
            {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0, "best_pos_only": -1.0, "best_rescue": -1.0},
        )
        best_metrics.setdefault("best_rescue", -1.0)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("已恢复 epoch=%d, 下一轮从 epoch %d 开始。", int(checkpoint.get("epoch", 0)), start_epoch)

    return start_epoch, history, best_metrics, checkpoint


def main():
    parser = build_argparser()
    explicit_dests = collect_explicit_dests(parser, sys.argv[1:])
    args = apply_preset(parser.parse_args(), explicit_dests)
    args = normalize_weakwindow_args(args)
    args = normalize_weak3_args(args)
    args = resolve_runtime_paths(args)
    args = normalize_unfreeze_args(args)
    args.output_dir = args.output_dir.resolve()
    run_dir = args.output_dir / args.run_name
    configure_logging(run_dir / f"{args.run_name}.log")
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.use_image_cache and not args.freeze_image_encoder:
        raise AssertionError("use_image_cache=True 时必须 freeze_image_encoder=True。")
    if args.use_image_cache:
        LOGGER.info("use_image_cache=True，训练阶段已强制关闭随机 hflip。")

    LOGGER.info("preset=%s", args.preset)
    LOGGER.info("run_name=%s", args.run_name)
    LOGGER.info("resume_best=%s", args.resume_best)
    LOGGER.info("train_json=%s", args.train_json)
    LOGGER.info("val_json=%s", args.val_json)
    LOGGER.info("train_list=%s", args.train_list)
    LOGGER.info("val_list=%s", args.val_list)
    LOGGER.info("image_root=%s", args.image_root)
    LOGGER.info("resume=%s", args.resume)
    LOGGER.info("no_val=%s", args.no_val)
    LOGGER.info("no_train_split_filter=%s", args.no_train_split_filter)
    LOGGER.info("no_auto_resume=%s", args.no_auto_resume)
    LOGGER.info("freeze_image_encoder=%s", args.freeze_image_encoder)
    LOGGER.info("freeze_text_encoder=%s", args.freeze_text_encoder)
    LOGGER.info("train_decoder_only=%s", args.train_decoder_only)
    LOGGER.info("unfreeze_image_mode=%s", args.unfreeze_image_mode)
    LOGGER.info("print_trainable_params=%s", args.print_trainable_params)
    LOGGER.info("use_prompt_prototype=%s", args.use_prompt_prototype)
    LOGGER.info("augment_prompt=%s", args.augment_prompt)
    LOGGER.info("prompt_alias_train=%s", args.prompt_alias_train)
    LOGGER.info("prompt_alias_prob=%s", args.prompt_alias_prob)
    LOGGER.info("val_augment_prompt=%s", args.val_augment_prompt)
    LOGGER.info("use_image_cache=%s", args.use_image_cache)
    LOGGER.info("text_cache_path=%s", args.text_cache_path)
    LOGGER.info("epochs=%s", args.epochs)
    LOGGER.info("decoder_lr=%s image_lr=%s text_lr=%s", args.decoder_lr, args.image_lr, args.text_lr)
    LOGGER.info("warmup_epochs=%s", args.warmup_epochs)
    LOGGER.info("min_lr_ratio=%s", args.min_lr_ratio)
    LOGGER.info("current classes=%s", CLASSES)
    LOGGER.info("negative_sample_ratio=%s", args.negative_sample_ratio)
    LOGGER.info("negative_sample_weight=%s", args.negative_sample_weight)
    LOGGER.info("weak3_enabled=%s", args.weak3_enabled)
    LOGGER.info("weak3_seed=%s", args.weak3_seed)
    LOGGER.info("weak_class_keep_ratio=%s", args.weak_class_keep_ratio)
    LOGGER.info("weak_class_loss_weight=%s", args.weak_class_loss_weight)
    LOGGER.info("sample_weight participates in loss = True")
    LOGGER.info("class_weights participates in loss = True")
    LOGGER.info("weak_class_loss_weight multiplied in dataset = False")
    LOGGER.info("effective weak class weight source = CLASS_WEIGHTS_ONLY")
    LOGGER.info("disabled_train_classes=%s", args.disabled_train_classes)
    LOGGER.info("class_keep_ratio=%s", args.class_keep_ratio)
    LOGGER.info("class_loss_scale=%s", args.class_loss_scale)
    LOGGER.info("old_class_sample_ratio=%s", args.old_class_sample_ratio)
    LOGGER.info("rare_class_keep_ratio=%s", args.rare_class_keep_ratio)
    LOGGER.info("rare_balance_enabled=%s", args.rare_balance_enabled)
    LOGGER.info("rare_oversample=%s", args.rare_oversample)
    LOGGER.info("class_weights=%s", args.class_weights)
    if args.preset in {"split_bridge_stage1", "split_bridge_stage2_fullset", "stage1_rare_rescue", "unfreeze_recalibrate", "text_realign_1ep", "gamble_no_window_from6", "gamble_weak_window_from6"} and args.resume is None and args.resume_best is None:
        LOGGER.warning("preset=%s 建议显式传入 --resume_best 或 --resume 作为热启动起点。", args.preset)
    if args.unfreeze_image_mode != "none" and not args.no_auto_resume:
        LOGGER.warning("partial unfreeze 建议使用 --resume_best 和 --no_auto_resume，避免加载旧 optimizer 状态。")
    if args.no_val:
        LOGGER.info("当前使用全集训练直接提交模式，不使用验证集选择 best checkpoint，最终请使用 final_fullset.pt 或 last.pt。")
        if args.no_train_split_filter:
            LOGGER.warning("当前是 fullset 直接提交模式。建议先用默认 train/val 对照训练确认收益，再决定是否用 fullset 收尾。")
    else:
        LOGGER.info("当前默认主线是 train/val 对照模式，会保存 best_all11.pt / best_old5.pt / best_rare.pt。")
    if args.weak3_enabled:
        LOGGER.info("========== WEAK3 TRAINING ENABLED ==========")
        LOGGER.info(
            "window keep_ratio=%.2f final_loss_weight=%.2f",
            float(args.weak_class_keep_ratio.get("window", 1.0)),
            float(args.weak_class_loss_weight.get("window", 1.0)),
        )
        LOGGER.info(
            "door keep_ratio=%.2f final_loss_weight=%.2f",
            float(args.weak_class_keep_ratio.get("door", 1.0)),
            float(args.weak_class_loss_weight.get("door", 1.0)),
        )
        LOGGER.info(
            "pole_light keep_ratio=%.2f final_loss_weight=%.2f",
            float(args.weak_class_keep_ratio.get("pole_light", 1.0)),
            float(args.weak_class_loss_weight.get("pole_light", 1.0)),
        )
        LOGGER.info("11-class inference format kept unchanged")
        LOGGER.info("avoid double weighting = True")
        LOGGER.info("============================================")

    preset_prompt_prototype_cfg = getattr(args, "prompt_prototype_cfg", None)
    if args.use_prompt_prototype:
        args.prompt_prototype_cfg = preset_prompt_prototype_cfg or PROMPT_PROTOTYPES
    else:
        args.prompt_prototype_cfg = {class_name: [class_name] for class_name in CLASSES}

    train_dataset, val_dataset = build_datasets(args)
    if args.dry_run_dataset_stats:
        dry_run_summary = build_dry_run_summary(train_dataset, val_dataset)
        for class_name in ("window", "door", "pole_light"):
            class_stats = dry_run_summary["weak3"][class_name]
            ratio = class_stats["keep_ratio_actual"]
            ratio_text = f"{ratio:.4f}" if ratio is not None else "n/a"
            LOGGER.info(
                "%s raw=%d kept=%d dropped=%d keep_ratio_actual=%s",
                class_name,
                class_stats["raw"],
                class_stats["kept"],
                class_stats["dropped"],
                ratio_text,
            )
        dry_run_path = run_dir / "dry_run_dataset_stats.json"
        save_json(dry_run_path, dry_run_summary)
        LOGGER.info("dry run dataset stats saved: %s", dry_run_path)
        LOGGER.info("dry_run_dataset_stats=True，已完成数据集统计，跳过 tokenizer / text cache / model / training。")
        return

    LOGGER.info("开始初始化模型...")
    model = ESAMCCLIPModel(
        tokenizer_dir=args.tokenizer_dir,
        efficient_sam_ckpt=args.efficient_sam_ckpt,
        freeze_image=args.freeze_image_encoder,
        freeze_text=args.freeze_text_encoder,
    ).to(device)
    if args.train_decoder_only:
        partial_warnings, partial_matches = set_partial_image_encoder_trainable(model, args.unfreeze_image_mode)
        if args.unfreeze_image_mode != "none":
            LOGGER.info("partial image encoder trainable params matched=%d", len(partial_matches))
            if partial_warnings:
                LOGGER.info("partial image encoder warnings=%s", partial_warnings)

    total_params, trainable_params = count_parameters(model)
    LOGGER.info("total parameter count=%.2fM", total_params)
    LOGGER.info("trainable parameter count=%.2fM", trainable_params)
    if args.print_trainable_params:
        log_trainable_parameter_summary(model, args)

    resume_checkpoint_path = resolve_resume_checkpoint(args, run_dir)
    if resume_checkpoint_path is not None:
        LOGGER.info("检测到断点续训 checkpoint: %s", resume_checkpoint_path)
    elif args.resume_best is not None:
        LOGGER.info("开始加载热启动权重: %s", args.resume_best)
        checkpoint, missing_keys, unexpected_keys = load_checkpoint_flexible(
            model,
            args.resume_best,
            device,
            require_decoder_match=True,
        )
        load_info = getattr(model, "_last_checkpoint_load_info", {})
        LOGGER.info("checkpoint loaded path=%s", load_info.get("checkpoint_path", args.resume_best))
        LOGGER.info("decoder_matched_keys=%d", len(load_info.get("decoder_matched_keys", [])))
        LOGGER.info("missing_keys=%s", missing_keys[:30])
        LOGGER.info("unexpected_keys=%s", unexpected_keys[:30])
        if checkpoint and not args.resume_weights_only:
            LOGGER.info("checkpoint loaded with optimizer/scheduler reuse disabled in fastfinetune mode")

    LOGGER.info("开始加载 tokenizer: %s", args.tokenizer_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    prototype_config = args.prompt_prototype_cfg
    LOGGER.info("prompt prototype classes=%d", len(args.prompt_prototype_cfg))
    for class_name in ("car", "window", "door", "pole_light"):
        if class_name in args.prompt_prototype_cfg:
            LOGGER.info("prompt aliases %s -> %s", class_name, args.prompt_prototype_cfg[class_name])
    LOGGER.info("开始%s text cache: %s", "重建" if args.rebuild_text_cache else "加载/构建", args.text_cache_path)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=args.text_cache_path,
        classes=CLASSES,
        prompt_prototypes=prototype_config,
        device=device,
        rebuild=args.rebuild_text_cache,
    )
    if set(text_cache_payload["classes"]) != set(CLASSES):
        raise RuntimeError("text cache 与当前 11 类不一致，请使用 --rebuild_text_cache")
    for class_name in CLASSES:
        if class_name not in text_cache_payload["embeddings"]:
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}，请使用 --rebuild_text_cache")
    LOGGER.info("开始构建 train DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        LOGGER.info("开始构建 val DataLoader...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
        )

    optimizer = build_optimizer(model, args.decoder_lr, args.image_lr, args.text_lr, args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = args.warmup_epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, args.min_lr_ratio) if total_steps > 0 else None
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    criterion_dice = DiceLoss()
    criterion_focal = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    save_json(run_dir / "config_used.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    history = defaultdict(list)
    best_metrics = {"best_all11": -1.0, "best_old5": -1.0, "best_rare": -1.0, "best_pos_only": -1.0, "best_rescue": -1.0}
    start_epoch = 1
    if resume_checkpoint_path is not None:
        start_epoch, history, best_metrics, _ = resume_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=resume_checkpoint_path,
            device=device,
            args=args,
        )
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"断点续训下一轮 epoch={start_epoch} 已大于设定总轮数 epochs={args.epochs}，请增大 --epochs 或关闭自动续训。"
            )
    last_epoch = 0

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        LOGGER.info("%s Epoch %d/%d", "─" * 50, epoch, args.epochs)
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion_dice, criterion_focal, scaler, device, text_cache_payload, args)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_stats["loss"])
        history["train_bce"].append(train_stats["bce"])
        history["train_dice"].append(train_stats["dice"])
        history["train_focal"].append(train_stats["focal"])
        history["lr"].append(current_lr)
        history["epoch_time"].append(train_stats["epoch_time"])
        history["batch_time"].append(train_stats["batch_time"])
        history["image_cache_hit_rate"].append(train_stats["image_cache_hit_rate"])

        if args.no_val:
            LOGGER.info(
                "TrainLoss=%.4f BCE=%.4f Dice=%.4f Focal=%.4f LR=%.2e epoch_time=%.1fs batch_time=%.2fs image_cache_hit_rate=%.2f%%",
                train_stats["loss"],
                train_stats["bce"],
                train_stats["dice"],
                train_stats["focal"],
                current_lr,
                train_stats["epoch_time"],
                train_stats["batch_time"],
                100.0 * train_stats["image_cache_hit_rate"],
            )
        else:
            val_metrics = validate(model, val_loader, criterion_dice, criterion_focal, device, text_cache_payload, args)
            history["val_miou_all11"].append(val_metrics.get("iou/overall", 0.0))
            history["val_miou_old5"].append(val_metrics.get("iou/old5", 0.0))
            history["val_miou_rare"].append(val_metrics.get("iou/rare", 0.0))
            history["val_pos_miou"].append(val_metrics.get("iou/pos_only", 0.0))
            history["val_dice"].append(val_metrics.get("dice/overall", 0.0))
            for class_name in CLASSES:
                history[f"iou/{class_name}"].append(val_metrics.get(f"iou/{class_name}", 0.0))

            LOGGER.info(
                "TrainLoss=%.4f BCE=%.4f Dice=%.4f Focal=%.4f mIoU_all11=%.4f mIoU_old5=%.4f mIoU_rare=%.4f "
                "pos_mIoU=%.4f Dice=%.4f LR=%.2e epoch_time=%.1fs batch_time=%.2fs image_cache_hit_rate=%.2f%%",
                train_stats["loss"],
                train_stats["bce"],
                train_stats["dice"],
                train_stats["focal"],
                val_metrics.get("iou/overall", 0.0),
                val_metrics.get("iou/old5", 0.0),
                val_metrics.get("iou/rare", 0.0),
                val_metrics.get("iou/pos_only", 0.0),
                val_metrics.get("dice/overall", 0.0),
                current_lr,
                train_stats["epoch_time"],
                train_stats["batch_time"],
                100.0 * train_stats["image_cache_hit_rate"],
            )
            LOGGER.info("Per-class IoU: %s", "  ".join(f"{name}={val_metrics.get(f'iou/{name}', 0.0):.3f}" for name in CLASSES))
        if torch.cuda.is_available() and device.type == "cuda":
            LOGGER.info("GPU memory max allocated: %.2f GB", torch.cuda.max_memory_allocated() / (1024 ** 3))

        save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
        if not args.no_val:
            current_all11 = val_metrics.get("iou/overall", 0.0)
            current_old5 = val_metrics.get("iou/old5", 0.0)
            current_rare = val_metrics.get("iou/rare", 0.0)
            current_pos_only = val_metrics.get("iou/pos_only", 0.0)
            rescue_score = current_all11 + 0.30 * current_rare - 0.50 * max(0.0, 0.655 - current_old5)
            history["rescue_score"].append(rescue_score)
            if current_all11 > best_metrics["best_all11"]:
                best_metrics["best_all11"] = current_all11
                save_checkpoint(run_dir / "best_all11.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if current_old5 > best_metrics["best_old5"]:
                best_metrics["best_old5"] = current_old5
                save_checkpoint(run_dir / "best_old5.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if current_rare > best_metrics["best_rare"]:
                best_metrics["best_rare"] = current_rare
                save_checkpoint(run_dir / "best_rare.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if current_pos_only > best_metrics["best_pos_only"]:
                best_metrics["best_pos_only"] = current_pos_only
                save_checkpoint(run_dir / "best_pos_only.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
            if rescue_score > best_metrics["best_rescue"]:
                best_metrics["best_rescue"] = rescue_score
                save_checkpoint(run_dir / "best_rescue.pt", epoch, model, optimizer, scheduler, history, best_metrics, args)
                LOGGER.info(
                    "★ best_rescue 更新 rescue_score=%.4f all11=%.4f old5=%.4f rare=%.4f",
                    rescue_score,
                    current_all11,
                    current_old5,
                    current_rare,
                )

    if last_epoch > 0:
        final_name = "final_fullset.pt" if args.no_val else "final_split.pt"
        final_path = run_dir / final_name
        save_checkpoint(final_path, last_epoch, model, optimizer, scheduler, history, best_metrics, args)
        LOGGER.info("训练结束，已保存最终 checkpoint: %s", final_path)

    if args.train_eval_after:
        LOGGER.info("开始训练集诊断评估。注意：这个指标只是伪标签拟合诊断，不是真实验证分数。")
        LOGGER.info("train diagnostic 使用的是训练伪标签，只用于检查是否学崩，不代表真实榜单表现。")
        train_diag_dataset = ESAMCCLIP11Dataset(
            annotation_json=args.train_json,
            split_txt=None if args.no_train_split_filter else args.train_list,
            image_root=args.image_root,
            img_size=(args.img_size, args.img_size),
            classes=CLASSES,
            prompt_prototypes=args.prompt_prototype_cfg,
            augment_prompt=False,
            hflip_prob=0.0,
            use_conf_filter=False,
            negative_sample_prob=0.0,
            negative_sample_weight=args.negative_sample_weight,
            rare_oversample=None,
            old_class_sample_ratio=1.0,
            rare_class_keep_ratio=1.0,
            old_classes=args.old_classes,
            rare_classes=args.rare_classes,
            rare_balance_enabled=False,
            training=False,
            seed=args.train_eval_seed,
        )
        if args.train_eval_max_samples > 0 and len(train_diag_dataset) > args.train_eval_max_samples:
            rng = random.Random(args.train_eval_seed)
            indices = list(range(len(train_diag_dataset)))
            rng.shuffle(indices)
            train_diag_dataset = Subset(train_diag_dataset, indices[: args.train_eval_max_samples])
        diag_batch_size = args.train_eval_batch_size or args.batch_size
        train_diag_loader = DataLoader(
            train_diag_dataset,
            batch_size=diag_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
        )
        train_diag_loader.text_cache_payload = text_cache_payload
        train_diag_loader.diag_img_size = args.img_size
        diag_summary, diag_rows = evaluate_train_diagnostic(model, train_diag_loader, device)
        save_json(run_dir / "train_diagnostic_metrics.json", diag_summary)
        with open(run_dir / "train_diagnostic_per_class.csv", "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["class_name", "iou", "precision", "recall", "pred_area", "gt_area"])
            writer.writerows(diag_rows)
        LOGGER.info(
            "train diagnostic: all11=%.4f old5=%.4f rare=%.4f pos=%.4f",
            diag_summary["train_mIoU_all11_on_pseudo"],
            diag_summary["train_mIoU_old5_on_pseudo"],
            diag_summary["train_mIoU_rare_on_pseudo"],
            diag_summary["train_pos_mIoU_on_pseudo"],
        )

    if args.no_val:
        csv_keys = [
            "train_loss",
            "train_bce",
            "train_dice",
            "train_focal",
            "lr",
            "epoch_time",
            "batch_time",
            "image_cache_hit_rate",
        ]
    else:
        csv_keys = [
            "train_loss",
            "train_bce",
            "train_dice",
            "train_focal",
            "val_miou_all11",
            "val_miou_old5",
            "val_miou_rare",
            "val_pos_miou",
            "val_dice",
            "lr",
            "epoch_time",
            "batch_time",
            "image_cache_hit_rate",
        ] + [f"iou/{c}" for c in CLASSES]
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch"] + csv_keys)
        for idx in range(len(history["train_loss"])):
            writer.writerow([idx + 1] + [history[key][idx] if idx < len(history[key]) else "" for key in csv_keys])
    plot_results(history, run_dir / "results.png", no_val=args.no_val)


if __name__ == "__main__":
    main()
