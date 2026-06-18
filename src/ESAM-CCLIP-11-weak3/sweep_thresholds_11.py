#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ESAM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ESAM_ROOT.parents[1]
for candidate in [str(ESAM_ROOT), str(PROJECT_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import cv2
import numpy as np
import torch

from config_esam_cclip_11 import (
    BATCH_SIZE,
    CLASSES,
    DEVICE,
    EFFICIENT_SAM_CKPT,
    IMAGE_ROOT,
    OUTPUT_ROOT,
    POSTPROCESS_DEFAULT,
    PROMPT_PROTOTYPES,
    REBUILD_TEXT_CACHE,
    ROOT,
    RUN_NAME,
    TEXT_CACHE_PATH,
    TOKENIZER_DIR,
    VAL_JSON,
    VAL_LIST,
)
from common_esam_cclip_11 import (
    ESAMCCLIPModel,
    compute_metrics,
    load_tokenizer,
    maybe_tqdm,
    normalize_text_feature,
    resolve_device,
    save_json,
)
from dataset_esam_cclip_11 import ESAMCCLIP11Dataset
from model_esam_cclip_11 import load_checkpoint_flexible
from path_utils import ensure_dir, ensure_file, resolve_project_path
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

OLD_CLASSES = {"person", "car", "building", "tree", "animal"}
RARE_CLASSES = {"trash can", "window", "door", "fence", "pole_light", "motorcycle"}
GROUP_CLASS_NAMES = {
    "all11": list(EXPECTED_CLASSES),
    "old5": ["person", "car", "building", "tree", "animal"],
    "rare6": ["trash can", "window", "door", "fence", "pole_light", "motorcycle"],
    "rare_without_window": ["trash can", "door", "fence", "pole_light", "motorcycle"],
    "all10_no_window": ["person", "car", "building", "tree", "animal", "trash can", "door", "fence", "pole_light", "motorcycle"],
    "rare_no_window": ["trash can", "door", "fence", "pole_light", "motorcycle"],
    "window": ["window"],
}
OBJECTIVE_WEIGHTS = {
    "all11": {"all11": 1.0},
    "balanced": {"old5": 0.60, "rare_no_window": 0.35, "window": 0.05},
    "weak_window_balanced": {"old5": 0.60, "rare_without_window": 0.35, "window": 0.05},
    "rare_without_window_safe": {"old5": 0.65, "rare_without_window": 0.35},
}

DEFAULT_CHECKPOINT = OUTPUT_ROOT / RUN_NAME / "best_all11.pt"
DEFAULT_TEXT_CACHE_PATH = TEXT_CACHE_PATH
DEFAULT_SWEEP_MODE = "best"
DEFAULT_DEVICE = DEVICE
DEFAULT_WRITE_BACK_CHECKPOINT = True
DEFAULT_REBUILD_TEXT_CACHE = True if REBUILD_TEXT_CACHE is False else REBUILD_TEXT_CACHE
DEFAULT_MAX_SAMPLES_PER_CLASS = 500
DEFAULT_SWEEP_CACHE_DIR = ROOT / "test" / "cache" / "sweep_thresholds_11"
DEFAULT_PROMPT_FUSION_MODE = "prototype"
DEFAULT_RAW_PROMPT_WEIGHT = 0.0
DEFAULT_PROMPT_MATCH_MODE = "exact"
WEAK3_SWEEP_ROOT = ROOT / "test" / "train_output" / "ESAM-CCLIP-11-weak3"
WEAKWINDOW_SWEEP_ROOT = WEAK3_SWEEP_ROOT
WINDOW_CONSERVATIVE_THRESH_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
WINDOW_CONSERVATIVE_MIN_AREA_GRID = [8, 16, 32]
WINDOW_CONSERVATIVE_TOPK_GRID = [None, 3]
WEAK3_THRESH_GRID = {
    "person": [0.42, 0.45, 0.50],
    "car": [0.50, 0.55, 0.60],
    "building": [0.42, 0.45, 0.50],
    "tree": [0.35, 0.40, 0.45],
    "animal": [0.28, 0.32, 0.36],
    "trash can": [0.28, 0.32, 0.36],
    "window": [0.50, 0.60, 0.70],
    "door": [0.40, 0.45, 0.50],
    "fence": [0.40, 0.45, 0.50],
    "pole_light": [0.45, 0.50, 0.55],
    "motorcycle": [0.35, 0.40, 0.45],
}
WEAK3_MIN_AREA_GRID = {
    "person": [32, 64],
    "car": [64, 96, 128],
    "building": [128, 192, 256],
    "tree": [128, 160, 224],
    "animal": [32, 64],
    "trash can": [16, 32],
    "window": [32, 64, 96],
    "door": [48, 64, 96],
    "fence": [8, 16, 32],
    "pole_light": [8, 16, 32],
    "motorcycle": [16, 32],
}

SUBMIT_CLEAN_PROMPT_PROTOTYPES = {
    "person": ["person", "people", "pedestrian", "human", "人", "行人"],
    "car": ["car", "vehicle", "automobile", "车辆", "汽车"],
    "building": ["building", "house", "architecture", "建筑", "楼"],
    "tree": ["tree", "vegetation", "树", "树木"],
    "animal": ["animal", "wild animal", "动物"],
    "trash can": ["trash can", "garbage bin", "trashbin", "rubbish bin", "垃圾桶"],
    "window": ["window", "窗户"],
    "door": ["door", "entrance", "门"],
    "fence": ["fence", "railing", "栏杆", "围栏"],
    "pole_light": ["pole_light", "street light", "lamp", "light pole", "路灯", "灯杆"],
    "motorcycle": ["motorcycle", "motorbike", "摩托车"],
}

# Stage2 submit-safe recommendation:
# python sweep_thresholds_11.py ^
#   --checkpoint D:\nyz\raytron_project\test\train_output\ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep\best_all11.pt ^
#   --sweep_mode stage2_submit_safe ^
#   --prompt_fusion_mode prototype ^
#   --raw_prompt_weight 0.0 ^
#   --prompt_match_mode exact ^
#   --max_samples_per_class 0 ^
#   --out_checkpoint D:\nyz\raytron_project\model\submit-stage2-mild-safe\sam3.pt
#
# python sweep_thresholds_11.py ^
#   --checkpoint D:\nyz\raytron_project\test\train_output\ESAM-CCLIP-11-text-realign-1ep-bs8\best_all11.pt ^
#   --sweep_mode stage2_submit_safe ^
#   --prompt_fusion_mode prototype ^
#   --raw_prompt_weight 0.0 ^
#   --prompt_match_mode exact ^
#   --max_samples_per_class 0 ^
#   --out_checkpoint D:\nyz\raytron_project\model\submit-stage2-text-safe\sam3.pt

BEST_THRESH_GRID = {
    "person": [0.38, 0.40, 0.42, 0.45],
    "car": [0.42, 0.45, 0.48, 0.50],
    "building": [0.42, 0.45, 0.48, 0.50],
    "tree": [0.32, 0.35, 0.38, 0.40],
    "animal": [0.28, 0.30, 0.33, 0.35],
    "trash can": [0.23, 0.25, 0.27, 0.30],
    "window": [0.32, 0.35, 0.38, 0.40],
    "door": [0.25, 0.27, 0.30, 0.32],
    "fence": [0.30, 0.35, 0.40, 0.45],
    "pole_light": [0.30, 0.35, 0.40, 0.45],
    "motorcycle": [0.35, 0.40, 0.45],
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

FINE_RECALL_THRESH_GRID = {
    "person": [0.53, 0.55, 0.57, 0.60, 0.62, 0.65],
    "car": [0.53, 0.55, 0.57, 0.60, 0.62, 0.65],
    "building": [0.58, 0.60, 0.62, 0.65, 0.67, 0.70],
    "tree": [0.43, 0.45, 0.47, 0.50, 0.52, 0.55],
    "animal": [0.38, 0.40, 0.42, 0.45, 0.47, 0.50],
    "trash can": [0.23, 0.25, 0.27, 0.30, 0.32, 0.35],
    "window": [0.23, 0.25, 0.27, 0.30, 0.32, 0.35],
    "door": [0.23, 0.25, 0.27, 0.30, 0.32, 0.35],
    "fence": [0.13, 0.15, 0.17, 0.20, 0.22, 0.25],
    "pole_light": [0.08, 0.10, 0.12, 0.15, 0.17, 0.20],
    "motorcycle": [0.23, 0.25, 0.27, 0.30, 0.32, 0.35],
}

BEST_MIN_AREA_GRID = {
    "person": [16, 24, 32],
    "car": [32, 64, 96],
    "building": [128, 256],
    "tree": [96, 128, 160],
    "animal": [16, 24, 32],
    "trash can": [8, 12, 16],
    "window": [8, 12, 16],
    "door": [16, 24, 32],
    "fence": [4, 8],
    "pole_light": [4, 8],
    "motorcycle": [12, 16],
}

FINE_RECALL_MIN_AREA_GRID = {
    "person": [8, 16, 32],
    "car": [8, 16, 32],
    "building": [64, 128, 256],
    "tree": [32, 64, 128],
    "animal": [4, 8, 16],
    "trash can": [1, 2, 4, 8],
    "window": [1, 2, 4, 8],
    "door": [2, 4, 8, 16],
    "fence": [1, 2, 4],
    "pole_light": [1, 2, 4],
    "motorcycle": [2, 4, 8],
}

TOPK_COMPONENTS_GRID = {
    "window": [None, 2, 3],
    "fence": [None, 3, 5],
    "pole_light": [None, 2, 3],
    "trash can": [None, 1],
    "door": [None, 1],
    "motorcycle": [None, 1],
}

HYBRID_SAFE_FIXED_OLD_CFG = {
    "person": {"threshold": 0.45, "min_area": 24, "topk_components": None},
    "car": {"threshold": 0.50, "min_area": 32, "topk_components": None},
    "building": {"threshold": 0.42, "min_area": 128, "topk_components": None},
    "tree": {"threshold": 0.40, "min_area": 160, "topk_components": None},
    "animal": {"threshold": 0.28, "min_area": 32, "topk_components": None},
}

HYBRID_SAFE_RARE_THRESH_GRID = {
    "trash can": [0.23, 0.25, 0.27, 0.30],
    "window": [0.30, 0.32, 0.35, 0.38],
    "door": [0.30, 0.32, 0.35],
    "fence": [0.25, 0.27, 0.30],
    "pole_light": [0.20, 0.22, 0.25, 0.28],
    "motorcycle": [0.35, 0.37, 0.40],
}

HYBRID_SAFE_RARE_FIXED_POST = {
    "trash can": {"min_area": 8, "topk_components": None},
    "window": {"min_area": 8, "topk_components": None},
    "door": {"min_area": 16, "topk_components": None},
    "fence": {"min_area": 4, "topk_components": 3},
    "pole_light": {"min_area": 4, "topk_components": None},
    "motorcycle": {"min_area": 8, "topk_components": None},
}

STAGE2_SUBMIT_OLD_THRESH_GRID = {
    "person": [0.43, 0.45, 0.47],
    "car": [0.48, 0.50, 0.52],
    "building": [0.42, 0.45, 0.48],
    "tree": [0.38, 0.40, 0.42],
    "animal": [0.28, 0.30, 0.32],
}

STAGE2_SUBMIT_RARE_THRESH_GRID = {
    "trash can": [0.27, 0.30, 0.32],
    "window": [0.35, 0.38, 0.40],
    "door": [0.30, 0.32, 0.35],
    "fence": [0.27, 0.30, 0.32],
    "pole_light": [0.22, 0.25, 0.28],
    "motorcycle": [0.37, 0.40, 0.42],
}

STAGE2_SUBMIT_OLD_MIN_AREA_GRID = {
    "person": [16, 24],
    "car": [32, 64],
    "building": [128],
    "tree": [128, 160],
    "animal": [16, 24, 32],
}

STAGE2_SUBMIT_RARE_POST = {
    "trash can": {"min_area": 8, "topk_components": None},
    "window": {"min_area": 8, "topk_components": None},
    "door": {"min_area": 16, "topk_components": None},
    "fence": {"min_area": 4, "topk_components": 3},
    "pole_light": {"min_area": 4, "topk_components": None},
    "motorcycle": {"min_area": 8, "topk_components": None},
}

STAGE2_REFINE_OLD_THRESH_GRID = {
    "person": [0.43, 0.44, 0.45, 0.46, 0.47],
    "car": [0.48, 0.49, 0.50, 0.51, 0.52],
    "building": [0.43, 0.44, 0.45, 0.46, 0.47],
    "tree": [0.38, 0.39, 0.40, 0.41, 0.42],
    "animal": [0.28, 0.29, 0.30, 0.31, 0.32],
}

STAGE2_REFINE_RARE_THRESH_GRID = {
    "trash can": [0.25, 0.27, 0.30, 0.32, 0.35],
    "window": [0.33, 0.35, 0.38, 0.40, 0.43],
    "door": [0.27, 0.30, 0.32, 0.35, 0.37],
    "fence": [0.25, 0.27, 0.30, 0.32, 0.35],
    "pole_light": [0.20, 0.22, 0.25, 0.28, 0.30],
    "motorcycle": [0.35, 0.37, 0.40, 0.42, 0.45],
}

STAGE2_REFINE_OLD_MIN_AREA_GRID = {
    "person": [16, 24, 32],
    "car": [32, 48, 64],
    "building": [96, 128, 160],
    "tree": [128, 160],
    "animal": [16, 24, 32],
}

STAGE2_REFINE_RARE_MIN_AREA_GRID = {
    "trash can": [4, 8],
    "window": [4, 8, 12],
    "door": [12, 16],
    "fence": [2, 4],
    "pole_light": [2, 4],
    "motorcycle": [8, 12],
}

STAGE2_REFINE_RARE_TOPK_GRID = {
    "window": [None, 2],
    "fence": [3, 5],
    "pole_light": [None, 2],
}


def canonicalize_sweep_mode(sweep_mode: str) -> str:
    if sweep_mode == "safe":
        return "best"
    if sweep_mode in {"stage2_submit_refine", "submit_stage2_refine"}:
        return "stage2_refine"
    if sweep_mode in {"weak3", "weak3safe", "weak3_safe", "weakwindow", "weakwindow_safe", "weakwindowsafe"}:
        return "weak3_safe"
    return sweep_mode


def resolve_prompt_prototypes_for_sweep(sweep_mode: str) -> Dict[str, List[str]]:
    sweep_mode = canonicalize_sweep_mode(sweep_mode)
    if sweep_mode in {"stage2_submit_safe", "stage2_refine"}:
        return SUBMIT_CLEAN_PROMPT_PROTOTYPES
    return PROMPT_PROTOTYPES


def resolve_threshold_grid(sweep_mode: str) -> dict:
    sweep_mode = canonicalize_sweep_mode(sweep_mode)
    if sweep_mode == "best":
        grid = dict(BEST_THRESH_GRID)
        grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return grid
    if sweep_mode == "weak3_safe":
        return dict(WEAK3_THRESH_GRID)
    if sweep_mode == "stage2_refine":
        grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                grid[class_name] = STAGE2_REFINE_OLD_THRESH_GRID[class_name]
            else:
                grid[class_name] = STAGE2_REFINE_RARE_THRESH_GRID[class_name]
        grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return grid
    if sweep_mode == "stage2_submit_safe":
        grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                grid[class_name] = STAGE2_SUBMIT_OLD_THRESH_GRID[class_name]
            else:
                grid[class_name] = STAGE2_SUBMIT_RARE_THRESH_GRID[class_name]
        grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return grid
    if sweep_mode == "hybrid_safe":
        hybrid_safe_grid = {}
        for class_name in CLASSES:
            if class_name in HYBRID_SAFE_FIXED_OLD_CFG:
                hybrid_safe_grid[class_name] = [HYBRID_SAFE_FIXED_OLD_CFG[class_name]["threshold"]]
            else:
                hybrid_safe_grid[class_name] = HYBRID_SAFE_RARE_THRESH_GRID[class_name]
        hybrid_safe_grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return hybrid_safe_grid
    if sweep_mode == "hybrid":
        hybrid_grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                hybrid_grid[class_name] = BEST_THRESH_GRID[class_name]
            else:
                hybrid_grid[class_name] = FINE_RECALL_THRESH_GRID[class_name]
        hybrid_grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return hybrid_grid
    if sweep_mode == "recall":
        grid = dict(RECALL_THRESH_GRID)
        grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return grid
    if sweep_mode == "fine_recall":
        grid = dict(FINE_RECALL_THRESH_GRID)
        grid["window"] = WINDOW_CONSERVATIVE_THRESH_GRID
        return grid
    raise ValueError(f"unsupported sweep_mode: {sweep_mode}")


def resolve_min_area_grid(sweep_mode: str) -> dict:
    sweep_mode = canonicalize_sweep_mode(sweep_mode)
    if sweep_mode == "best":
        grid = dict(BEST_MIN_AREA_GRID)
        grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return grid
    if sweep_mode == "weak3_safe":
        return dict(WEAK3_MIN_AREA_GRID)
    if sweep_mode == "stage2_refine":
        grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                grid[class_name] = STAGE2_REFINE_OLD_MIN_AREA_GRID[class_name]
            else:
                grid[class_name] = STAGE2_REFINE_RARE_MIN_AREA_GRID[class_name]
        grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return grid
    if sweep_mode == "stage2_submit_safe":
        grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                grid[class_name] = STAGE2_SUBMIT_OLD_MIN_AREA_GRID[class_name]
            else:
                grid[class_name] = [STAGE2_SUBMIT_RARE_POST[class_name]["min_area"]]
        grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return grid
    if sweep_mode == "hybrid_safe":
        hybrid_safe_grid = {}
        for class_name in CLASSES:
            if class_name in HYBRID_SAFE_FIXED_OLD_CFG:
                hybrid_safe_grid[class_name] = [HYBRID_SAFE_FIXED_OLD_CFG[class_name]["min_area"]]
            else:
                hybrid_safe_grid[class_name] = [HYBRID_SAFE_RARE_FIXED_POST[class_name]["min_area"]]
        hybrid_safe_grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return hybrid_safe_grid
    if sweep_mode == "hybrid":
        hybrid_grid = {}
        for class_name in CLASSES:
            if class_name in OLD_CLASSES:
                hybrid_grid[class_name] = BEST_MIN_AREA_GRID[class_name]
            else:
                hybrid_grid[class_name] = FINE_RECALL_MIN_AREA_GRID[class_name]
        hybrid_grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return hybrid_grid
    if sweep_mode == "fine_recall":
        grid = dict(FINE_RECALL_MIN_AREA_GRID)
        grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
        return grid
    grid = dict(BEST_MIN_AREA_GRID)
    grid["window"] = WINDOW_CONSERVATIVE_MIN_AREA_GRID
    return grid


def resolve_default_output_dir(sweep_mode: str) -> Path:
    sweep_mode = canonicalize_sweep_mode(sweep_mode)
    if sweep_mode == "stage2_refine":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_stage2_refine"
    if sweep_mode == "weak3_safe":
        return WEAK3_SWEEP_ROOT / "sweep_weak3_safe"
    if sweep_mode == "stage2_submit_safe":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_stage2_submit_safe"
    if sweep_mode == "fine_recall":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_fine_recall"
    if sweep_mode == "hybrid_safe":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_hybrid_safe"
    if sweep_mode == "hybrid":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_hybrid"
    if sweep_mode == "recall":
        return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_recall"
    return WEAKWINDOW_SWEEP_ROOT / "threshold_sweep_esam_11_best"


def resolve_default_write_back_path(sweep_mode: str) -> Path:
    sweep_mode = canonicalize_sweep_mode(sweep_mode)
    if sweep_mode == "stage2_refine":
        return ROOT / "model" / "submit-rsam-stage2-refine" / "sam3.pt"
    if sweep_mode == "weak3_safe":
        return ROOT / "model" / "submit-rsam-weak3" / "sam3.pt"
    if sweep_mode == "stage2_submit_safe":
        return ROOT / "model" / "submit-rsam-stage2-safe" / "sam3.pt"
    if sweep_mode == "fine_recall":
        return ROOT / "model" / "submit-rsam-fine-recall" / "sam3.pt"
    if sweep_mode == "hybrid_safe":
        return ROOT / "model" / "submit-rsam-hybrid-safe" / "sam3.pt"
    if sweep_mode == "hybrid":
        return ROOT / "model" / "submit-rsam-hybrid" / "sam3.pt"
    if sweep_mode == "recall":
        return ROOT / "model" / "submit-rsam-recall" / "sam3.pt"
    return ROOT / "model" / "submit-rsam-best" / "sam3.pt"


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


def attach_file_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    LOGGER.addHandler(file_handler)


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "class_names": [item["class_name"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
    }


def normalize_prompt_key(text: str) -> str:
    normalized = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


COMPLEX_PROMPT_ALIASES: Dict[str, List[str]] = {
    "car": [
        "车",
        "小车",
        "汽车",
        "车辆",
        "远处的车",
        "远处的小车",
        "草丛里的车",
        "被树遮挡的车",
        "被遮挡的车",
        "truck",
        "bus",
        "car in bushes",
        "car behind tree",
        "partially occluded car",
        "small distant car",
        "vehicle on road",
    ],
    "window": [
        "车窗",
        "汽车窗户",
        "建筑窗户",
        "楼上的窗户",
        "窗户",
        "窗",
        "glass window",
        "car window",
        "building window",
        "window on building",
        "window of car",
    ],
    "door": [
        "车门",
        "汽车门",
        "建筑门",
        "门",
        "building door",
        "car door",
        "door of car",
        "door on vehicle",
    ],
    "pole_light": [
        "路灯",
        "灯杆",
        "路边的路灯",
        "路边的灯杆",
        "远处的灯杆",
        "street light",
        "light pole",
        "pole light",
        "lamp post",
        "street light pole",
        "street light beside road",
    ],
    "motorcycle": [
        "摩托车",
        "远处的摩托车",
        "motorcycle",
        "motorbike",
        "scooter",
        "small motorcycle",
    ],
    "trash can": [
        "垃圾桶",
        "路边的垃圾桶",
        "trash can",
        "garbage bin",
        "dustbin",
        "waste bin",
    ],
}
ENGLISH_TARGET_RELATIONS = [" in ", " on ", " under ", " behind ", " near ", " beside ", " among ", " with ", " inside ", " at "]


def build_complex_prompt_alias_map(classes: List[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for class_name in classes:
        for alias in COMPLEX_PROMPT_ALIASES.get(class_name, []):
            alias_map[normalize_prompt_key(alias)] = class_name
    return alias_map


def build_prompt_aliases(prompt_prototypes: Dict[str, List[str]], classes: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for class_name in classes:
        for alias in prompt_prototypes.get(class_name, [class_name]) + [class_name]:
            aliases[normalize_prompt_key(alias)] = class_name
    aliases.update(build_complex_prompt_alias_map(classes))
    return aliases


def map_prompt_to_known_class(prompt_text: str, prompt_aliases: Dict[str, str]) -> Optional[str]:
    return prompt_aliases.get(normalize_prompt_key(prompt_text))


def is_short_english_alias(alias_norm: str) -> bool:
    if not alias_norm:
        return False
    if any(ord(ch) > 127 for ch in alias_norm):
        return False
    return len(alias_norm.replace(" ", "")) < 3


def contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in str(text))


def token_sequence_in_prompt(prompt_norm: str, alias_norm: str) -> bool:
    prompt_tokens = prompt_norm.split()
    alias_tokens = alias_norm.split()
    if not prompt_tokens or not alias_tokens or len(alias_tokens) > len(prompt_tokens):
        return False
    window = len(alias_tokens)
    for start in range(0, len(prompt_tokens) - window + 1):
        if prompt_tokens[start:start + window] == alias_tokens:
            return True
    return False


def extract_chinese_target_phrase(prompt_text: str) -> Optional[str]:
    prompt_norm = normalize_prompt_key(prompt_text)
    if "的" not in prompt_norm:
        return None
    target_phrase = prompt_norm.rsplit("的", 1)[-1].strip()
    return target_phrase or None


def extract_english_target_phrase(prompt_norm: str) -> Optional[str]:
    prompt_norm = normalize_prompt_key(prompt_norm)
    if not prompt_norm:
        return None
    if " of " in prompt_norm:
        target_phrase = prompt_norm.split(" of ", 1)[0].strip()
        if target_phrase:
            return target_phrase
    for relation_token in ENGLISH_TARGET_RELATIONS:
        if relation_token in prompt_norm:
            target_phrase = prompt_norm.split(relation_token, 1)[0].strip()
            if target_phrase:
                return target_phrase
    return None


def collect_soft_match_candidates(
    prompt_norm: str,
    alias_to_class: Dict[str, str],
    classes: List[str],
    complex_aliases: Optional[Dict[str, str]] = None,
) -> List[Tuple[int, int, int, int, str]]:
    rare_classes = set(RARE_CLASSES).intersection(set(classes))
    complex_aliases = complex_aliases or {}
    candidates: List[Tuple[int, int, int, int, str]] = []
    for alias_norm, class_name in alias_to_class.items():
        if not alias_norm or alias_norm == prompt_norm:
            continue
        if is_short_english_alias(alias_norm):
            continue
        if contains_non_ascii(alias_norm):
            matched = alias_norm in prompt_norm
        else:
            matched = token_sequence_in_prompt(prompt_norm, alias_norm)
        if not matched:
            continue
        is_complex = 1 if alias_norm in complex_aliases else 0
        token_count = len(alias_norm.split())
        rare_priority = 1 if class_name in rare_classes else 0
        candidates.append((is_complex, token_count, len(alias_norm), rare_priority, class_name))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return candidates


def match_target_phrase_to_class(
    target_phrase: str,
    prompt_aliases: Dict[str, str],
    complex_prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    target_norm = normalize_prompt_key(target_phrase)
    if not target_norm:
        return None
    exact_complex = complex_prompt_aliases.get(target_norm)
    if exact_complex is not None:
        return exact_complex
    exact = prompt_aliases.get(target_norm)
    if exact is not None:
        return exact
    candidates = collect_soft_match_candidates(
        prompt_norm=target_norm,
        alias_to_class=prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    return candidates[0][4] if candidates else None


def map_prompt_to_known_class_soft(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    exact = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if exact is not None:
        return exact
    prompt_norm = normalize_prompt_key(prompt_text)
    if not prompt_norm:
        return None
    complex_prompt_aliases = build_complex_prompt_alias_map(classes)
    candidates = collect_soft_match_candidates(
        prompt_norm=prompt_norm,
        alias_to_class=prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    return candidates[0][4] if candidates else None


def map_prompt_to_known_class_target_soft(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
) -> Optional[str]:
    exact = map_prompt_to_known_class(prompt_text, prompt_aliases)
    if exact is not None:
        return exact
    prompt_norm = normalize_prompt_key(prompt_text)
    if not prompt_norm:
        return None
    complex_prompt_aliases = build_complex_prompt_alias_map(classes)
    complex_candidates = collect_soft_match_candidates(
        prompt_norm=prompt_norm,
        alias_to_class=complex_prompt_aliases,
        classes=classes,
        complex_aliases=complex_prompt_aliases,
    )
    if complex_candidates:
        return complex_candidates[0][4]
    chinese_target = extract_chinese_target_phrase(prompt_text)
    if chinese_target:
        matched = match_target_phrase_to_class(chinese_target, prompt_aliases, complex_prompt_aliases, classes)
        if matched is not None:
            return matched
    english_target = extract_english_target_phrase(prompt_norm)
    if english_target:
        matched = match_target_phrase_to_class(english_target, prompt_aliases, complex_prompt_aliases, classes)
        if matched is not None:
            return matched
    return map_prompt_to_known_class_soft(prompt_text, prompt_aliases, classes)


def resolve_prompt_mapped_class(
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    classes: List[str],
    prompt_match_mode: str,
) -> Optional[str]:
    if prompt_match_mode == "target_soft":
        return map_prompt_to_known_class_target_soft(prompt_text, prompt_aliases, classes)
    if prompt_match_mode == "soft":
        return map_prompt_to_known_class_soft(prompt_text, prompt_aliases, classes)
    return map_prompt_to_known_class(prompt_text, prompt_aliases)


def build_text_feature_for_prompt(
    model,
    tokenizer,
    text_cache_payload: Dict[str, Any],
    prompt_text: str,
    prompt_aliases: Dict[str, str],
    device,
    prompt_feature_cache: Optional[Dict[str, Tuple[torch.Tensor, Optional[str]]]] = None,
    prompt_fusion_mode: str = DEFAULT_PROMPT_FUSION_MODE,
    raw_prompt_weight: float = DEFAULT_RAW_PROMPT_WEIGHT,
    prompt_match_mode: str = DEFAULT_PROMPT_MATCH_MODE,
) -> Tuple[torch.Tensor, Optional[str]]:
    prompt_text = str(prompt_text).strip()
    raw_prompt_weight = min(max(float(raw_prompt_weight), 0.0), 1.0)
    cache_key = f"{prompt_text}||{prompt_fusion_mode}||{raw_prompt_weight:.6f}||{prompt_match_mode}"
    if prompt_feature_cache is not None and cache_key in prompt_feature_cache:
        cached_feature, cached_mapped_class = prompt_feature_cache[cache_key]
        return cached_feature.to(device, non_blocking=True), cached_mapped_class

    mapped_class = resolve_prompt_mapped_class(
        prompt_text=prompt_text,
        prompt_aliases=prompt_aliases,
        classes=list(text_cache_payload.get("classes", CLASSES)),
        prompt_match_mode=prompt_match_mode,
    )
    device_embeddings = text_cache_payload.get("device_embeddings", {})

    def encode_raw_prompt_feature(text: str) -> torch.Tensor:
        raw_cache_key = f"__raw__::{text}"
        if prompt_feature_cache is not None and raw_cache_key in prompt_feature_cache:
            cached_feature, _ = prompt_feature_cache[raw_cache_key]
            return cached_feature.to(device, non_blocking=True)
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=15,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        feature = model.encode_text(input_ids, attention_mask).detach()
        feature = normalize_text_feature(feature)
        if feature.ndim == 2:
            feature = feature[0]
        feature = normalize_text_feature(feature.unsqueeze(0))[0]
        if prompt_feature_cache is not None:
            prompt_feature_cache[raw_cache_key] = (feature.detach().cpu(), None)
        return feature

    if mapped_class is not None and mapped_class in device_embeddings:
        prototype_feature = device_embeddings[mapped_class]
        if prototype_feature.ndim == 2:
            prototype_feature = prototype_feature[0]
        prototype_feature = normalize_text_feature(prototype_feature.unsqueeze(0))[0]
        if prompt_fusion_mode == "prototype":
            final_feature = prototype_feature
        elif prompt_fusion_mode == "raw":
            final_feature = encode_raw_prompt_feature(prompt_text)
        else:
            raw_prompt_feature = encode_raw_prompt_feature(prompt_text)
            blended = (1.0 - raw_prompt_weight) * prototype_feature + raw_prompt_weight * raw_prompt_feature
            final_feature = normalize_text_feature(blended.unsqueeze(0))[0]
        resolved = normalize_text_feature(final_feature.unsqueeze(0))[0], mapped_class
        if prompt_feature_cache is not None:
            prompt_feature_cache[cache_key] = (resolved[0].detach().cpu(), resolved[1])
        return resolved

    feature = encode_raw_prompt_feature(prompt_text)
    resolved = feature, None
    if prompt_feature_cache is not None:
        prompt_feature_cache[cache_key] = (resolved[0].detach().cpu(), resolved[1])
    return resolved


def resolve_runtime_paths(args):
    args.checkpoint = ensure_file(args.checkpoint, "checkpoint")
    args.val_json = ensure_file(args.val_json, "val_json")
    if args.val_list is not None:
        args.val_list = ensure_file(args.val_list, "val_list")
    args.image_root = ensure_dir(args.image_root, "image_root")
    args.tokenizer_dir = ensure_dir(args.tokenizer_dir, "tokenizer_dir")
    args.text_cache_path = resolve_project_path(args.text_cache_path)
    args.output_dir = resolve_project_path(args.output_dir)
    if args.write_back_path is not None:
        args.write_back_path = resolve_project_path(args.write_back_path)
    return args


@torch.no_grad()
def collect_validation_logits(
    model,
    dataset,
    text_cache_payload,
    device,
    tokenizer,
    prompt_aliases,
    prompt_fusion_mode=DEFAULT_PROMPT_FUSION_MODE,
    raw_prompt_weight=DEFAULT_RAW_PROMPT_WEIGHT,
    prompt_match_mode=DEFAULT_PROMPT_MATCH_MODE,
    max_samples_per_class=500,
    seed=42,
):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    records = defaultdict(list)
    seen_counts = defaultdict(int)
    rng = np.random.default_rng(seed)
    prompt_feature_cache: Dict[str, Tuple[torch.Tensor, Optional[str]]] = {}
    for batch in maybe_tqdm(loader, total=len(loader), desc="Collect", leave=False):
        images = batch["images"].to(device)
        image_features = model.encode_image(images)
        feature_list = []
        for prompt_text in batch["prompt_texts"]:
            feature, _ = build_text_feature_for_prompt(
                model=model,
                tokenizer=tokenizer,
                text_cache_payload=text_cache_payload,
                prompt_text=prompt_text,
                prompt_aliases=prompt_aliases,
                device=device,
                prompt_feature_cache=prompt_feature_cache,
                prompt_fusion_mode=prompt_fusion_mode,
                raw_prompt_weight=raw_prompt_weight,
                prompt_match_mode=prompt_match_mode,
            )
            feature_list.append(feature)
        text_features = torch.stack(feature_list, dim=0).to(device)
        logits = model.decode(image_features, text_features, target_size=(dataset.img_size[0], dataset.img_size[1]))
        for idx, class_name in enumerate(batch["class_names"]):
            seen_counts[class_name] += 1
            sample = {
                "logit": logits[idx, 0].detach().cpu().numpy().astype(np.float16),
                "mask": (batch["masks"][idx, 0].detach().cpu().numpy() > 0.5).astype(np.uint8),
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


def default_cfg_for_class(class_name: str, threshold_grid: dict, min_area_grid: dict, sweep_mode: str) -> dict:
    return {
        "threshold": float(threshold_grid[class_name][0]),
        "min_area": int(min_area_grid[class_name][0]),
        "fill_holes": bool(POSTPROCESS_DEFAULT[class_name]["fill_holes"]),
        "topk_components": resolve_topk_grid(class_name, sweep_mode)[0],
    }


def resolve_topk_grid(class_name: str, sweep_mode: Optional[str] = None):
    sweep_mode = canonicalize_sweep_mode(sweep_mode) if sweep_mode is not None else None
    if sweep_mode == "weak3_safe":
        if class_name == "window":
            return [None, 3]
        if class_name == "fence":
            return [3]
        return [None]
    if class_name == "window":
        return WINDOW_CONSERVATIVE_TOPK_GRID
    if sweep_mode == "stage2_refine":
        if class_name in OLD_CLASSES:
            return [None]
        if class_name in STAGE2_REFINE_RARE_TOPK_GRID:
            return STAGE2_REFINE_RARE_TOPK_GRID[class_name]
        return [STAGE2_SUBMIT_RARE_POST[class_name]["topk_components"]]
    if sweep_mode == "stage2_submit_safe":
        if class_name in OLD_CLASSES:
            return [None]
        return [STAGE2_SUBMIT_RARE_POST[class_name]["topk_components"]]
    if sweep_mode == "hybrid_safe":
        if class_name in HYBRID_SAFE_FIXED_OLD_CFG:
            return [HYBRID_SAFE_FIXED_OLD_CFG[class_name]["topk_components"]]
        return [HYBRID_SAFE_RARE_FIXED_POST[class_name]["topk_components"]]
    return TOPK_COMPONENTS_GRID.get(class_name, [None])


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= int(min_area):
            filtered[labels == label_idx] = 1
    return filtered


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    canvas = binary.copy()
    cv2.drawContours(canvas, contours, -1, 1, thickness=cv2.FILLED)
    return canvas.astype(np.uint8)


def keep_top_k_components(mask: np.ndarray, topk_components: Optional[int]) -> np.ndarray:
    if topk_components is None or int(topk_components) <= 0:
        return (mask > 0).astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, label_idx))
    if not components:
        return np.zeros_like(binary)
    components.sort(reverse=True)
    keep_labels = {label_idx for _, label_idx in components[: int(topk_components)]}
    filtered = np.zeros_like(binary)
    for label_idx in keep_labels:
        filtered[labels == label_idx] = 1
    return filtered


def apply_postprocess_with_topk(
    mask: np.ndarray,
    min_area: int = 0,
    fill_holes: bool = False,
    topk_components: Optional[int] = None,
) -> np.ndarray:
    processed = remove_small_components(mask, min_area=min_area)
    processed = keep_top_k_components(processed, topk_components=topk_components)
    if fill_holes:
        processed = fill_small_holes(processed)
    return processed.astype(np.uint8)


def compute_group_scores(per_class_scores: Dict[str, float]) -> Dict[str, float]:
    group_scores = {}
    for group_name, class_names in GROUP_CLASS_NAMES.items():
        values = [float(per_class_scores.get(class_name, 0.0)) for class_name in class_names]
        group_scores[group_name] = float(np.mean(values)) if values else 0.0
    return group_scores


def compute_objective_score(group_scores: Dict[str, float], objective: str) -> float:
    weights = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["all11"])
    return float(sum(float(group_scores.get(group_name, 0.0)) * float(weight) for group_name, weight in weights.items()))


def build_sweep_summary_lines(
    sweep_mode: str,
    objective: str,
    checkpoint_path: Path,
    max_samples_per_class: Optional[int],
    prompt_prototypes_source: str,
    prompt_fusion_mode: str,
    raw_prompt_weight: float,
    prompt_match_mode: str,
    sample_counts: dict,
    best_metric: float,
    best_thresholds: dict,
    best_postprocess: dict,
    best_scores: dict,
    best_metrics: dict,
    group_scores: dict,
    objective_score: float,
    write_back_path: Optional[Path],
    threshold_grid: dict,
    min_area_grid: dict,
):
    lines = [
        "========== Sweep Summary ==========",
        f"sweep_mode: {sweep_mode}",
        f"objective: {objective}",
        f"checkpoint: {checkpoint_path}",
        f"max_samples_per_class: {max_samples_per_class}",
        f"prompt_prototypes_source: {prompt_prototypes_source}",
        f"prompt_fusion_mode: {prompt_fusion_mode}",
        f"raw_prompt_weight: {raw_prompt_weight:.3f}",
        f"prompt_match_mode: {prompt_match_mode}",
        f"best_metric: {best_metric:.6f}",
        f"objective_score: {objective_score:.6f}",
        "sample_counts:",
    ]
    if sweep_mode == "stage2_submit_safe":
        lines.append("stage2 submit safe mode")
        lines.append("old classes use narrow adaptive grid")
        lines.append("rare classes use conservative grid")
        lines.append(f"stage2 old threshold grid: {json.dumps(STAGE2_SUBMIT_OLD_THRESH_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 rare threshold grid: {json.dumps(STAGE2_SUBMIT_RARE_THRESH_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 rare post cfg: {json.dumps(STAGE2_SUBMIT_RARE_POST, ensure_ascii=False)}")
    if sweep_mode == "stage2_refine":
        lines.append("stage2 refine mode")
        lines.append("old classes threshold local search within +/-0.02")
        lines.append("rare classes threshold local search within +/-0.03~0.05")
        lines.append(f"stage2 refine old threshold grid: {json.dumps(STAGE2_REFINE_OLD_THRESH_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 refine rare threshold grid: {json.dumps(STAGE2_REFINE_RARE_THRESH_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 refine old min_area grid: {json.dumps(STAGE2_REFINE_OLD_MIN_AREA_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 refine rare min_area grid: {json.dumps(STAGE2_REFINE_RARE_MIN_AREA_GRID, ensure_ascii=False)}")
        lines.append(f"stage2 refine rare topk grid: {json.dumps(STAGE2_REFINE_RARE_TOPK_GRID, ensure_ascii=False)}")
    if sweep_mode == "weak3_safe":
        lines.append("weak3 safe mode")
        lines.append(f"weak3 threshold grid: {json.dumps(WEAK3_THRESH_GRID, ensure_ascii=False)}")
        lines.append(f"weak3 min_area grid: {json.dumps(WEAK3_MIN_AREA_GRID, ensure_ascii=False)}")
    if sweep_mode == "hybrid_safe":
        lines.append("old classes are fixed")
        lines.append("rare classes threshold-only conservative sweep")
        lines.append(f"fixed old cfg: {json.dumps(HYBRID_SAFE_FIXED_OLD_CFG, ensure_ascii=False)}")
        lines.append(f"rare threshold grid: {json.dumps(HYBRID_SAFE_RARE_THRESH_GRID, ensure_ascii=False)}")
    for class_name in CLASSES:
        lines.append(f"  {class_name}: {int(sample_counts.get(class_name, 0))}")
    lines.append("group_scores:")
    for group_name in ["all11", "old5", "rare6", "rare_without_window", "all10_no_window", "rare_no_window", "window"]:
        lines.append(f"  {group_name}: {float(group_scores.get(group_name, 0.0)):.6f}")
    lines.append(f"all11_avg: {float(group_scores.get('all11', 0.0)):.6f}")
    lines.append(f"old5_avg: {float(group_scores.get('old5', 0.0)):.6f}")
    lines.append(f"rare_avg: {float(group_scores.get('rare6', 0.0)):.6f}")
    lines.append(f"all10_no_window_avg: {float(group_scores.get('all10_no_window', 0.0)):.6f}")
    lines.append(f"rare_no_window_avg: {float(group_scores.get('rare_no_window', 0.0)):.6f}")
    lines.append("best_per_class:")
    for class_name in CLASSES:
        post_cfg = best_postprocess[class_name]
        metric_cfg = best_metrics[class_name]
        lines.append(
            f"  {class_name}: threshold={float(best_thresholds[class_name]):.2f}, "
            f"min_area={post_cfg['min_area']}, topk_components={post_cfg.get('topk_components')}, "
            f"score={float(best_scores[class_name]):.4f}, "
            f"fill_holes={post_cfg['fill_holes']}, iou={float(metric_cfg['iou']):.4f}, "
            f"precision={float(metric_cfg['precision']):.4f}, recall={float(metric_cfg['recall']):.4f}, "
            f"pred_area={float(metric_cfg['pred_area']):.2f}, gt_area={float(metric_cfg['gt_area']):.2f}"
        )
    lines.append(f"write_back_path: {write_back_path if write_back_path is not None else 'disabled'}")

    if best_thresholds["pole_light"] == min(threshold_grid["pole_light"]) or best_thresholds["fence"] == min(threshold_grid["fence"]):
        lines.append("如果 pole_light/fence 阈值被选到最低，说明 rare recall 仍偏弱；")
    if any(
        best_thresholds[class_name] == min(threshold_grid[class_name])
        for class_name in ("person", "car", "building", "tree", "animal")
    ):
        lines.append("如果 old 类阈值被选到最低，说明模型偏保守；")
    if any(
        best_postprocess[class_name]["min_area"] == max(min_area_grid[class_name])
        for class_name in CLASSES
    ):
        lines.append("如果某类 min_area 被选到最大，说明该类噪声偏多；")
    if any(
        best_postprocess[class_name]["min_area"] == min(min_area_grid[class_name])
        for class_name in CLASSES
    ):
        lines.append("如果某类 min_area 被选到最小，说明小目标保留有收益。")
    return lines


def log_sweep_summary(summary_lines):
    for line in summary_lines:
        LOGGER.info(line)


def persist_sweep_summary_cache(
    output_dir: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    sweep_mode: str,
    summary_lines,
    summary_payload: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_stem = Path(checkpoint_path).stem
    output_summary_path = output_dir / "sweep_summary_terminal.txt"
    output_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    cache_text_latest = cache_dir / f"{checkpoint_stem}_{sweep_mode}_latest.txt"
    cache_json_latest = cache_dir / f"{checkpoint_stem}_{sweep_mode}_latest.json"
    cache_text_history = cache_dir / f"{checkpoint_stem}_{sweep_mode}_{timestamp}.txt"
    cache_json_history = cache_dir / f"{checkpoint_stem}_{sweep_mode}_{timestamp}.json"

    for path in [cache_text_latest, cache_text_history]:
        path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for path in [cache_json_latest, cache_json_history]:
        save_json(path, summary_payload)

    return {
        "output_summary_path": str(output_summary_path),
        "cache_text_latest": str(cache_text_latest),
        "cache_json_latest": str(cache_json_latest),
        "cache_text_history": str(cache_text_history),
        "cache_json_history": str(cache_json_history),
    }


def main():
    if list(CLASSES) != EXPECTED_CLASSES:
        raise RuntimeError(f"CLASSES changed unexpectedly: {CLASSES}")

    configure_logging()
    parser = argparse.ArgumentParser(description="Sweep thresholds for ESAM-CCLIP-11 checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val_json", type=Path, default=VAL_JSON)
    parser.add_argument("--val_list", type=Path, default=VAL_LIST)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--tokenizer_dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--text_cache_path", type=Path, default=DEFAULT_TEXT_CACHE_PATH)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE if DEFAULT_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--sweep_mode", choices=["best", "safe", "recall", "fine_recall", "hybrid", "hybrid_safe", "stage2_submit_safe", "stage2_refine", "stage2_submit_refine", "submit_stage2_refine", "weak3_safe", "weak3", "weak3safe", "weakwindow", "weakwindow_safe", "weakwindowsafe"], default=DEFAULT_SWEEP_MODE)
    parser.add_argument("--objective", choices=["all11", "balanced", "weak_window_balanced", "rare_without_window_safe"], default="all11")
    parser.add_argument("--write_back_checkpoint", dest="write_back_checkpoint", action="store_true")
    parser.add_argument("--no_write_back_checkpoint", dest="write_back_checkpoint", action="store_false")
    parser.add_argument("--write_back_path", type=Path, default=None)
    parser.add_argument("--out_checkpoint", type=Path, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS)
    parser.add_argument("--rebuild_text_cache", dest="rebuild_text_cache", action="store_true")
    parser.add_argument("--no_rebuild_text_cache", dest="rebuild_text_cache", action="store_false")
    parser.add_argument("--prompt_fusion_mode", choices=["prototype", "raw", "blend"], default=DEFAULT_PROMPT_FUSION_MODE)
    parser.add_argument("--raw_prompt_weight", type=float, default=DEFAULT_RAW_PROMPT_WEIGHT)
    parser.add_argument("--prompt_match_mode", choices=["exact", "soft", "target_soft"], default=DEFAULT_PROMPT_MATCH_MODE)
    parser.set_defaults(
        write_back_checkpoint=DEFAULT_WRITE_BACK_CHECKPOINT,
        rebuild_text_cache=DEFAULT_REBUILD_TEXT_CACHE,
    )
    args = parser.parse_args()

    requested_sweep_mode = args.sweep_mode
    args.sweep_mode = canonicalize_sweep_mode(args.sweep_mode)
    if requested_sweep_mode == "safe":
        LOGGER.warning("sweep_mode=safe is deprecated and now maps to sweep_mode=best")

    active_prompt_prototypes = resolve_prompt_prototypes_for_sweep(args.sweep_mode)
    prompt_prototypes_source = "submit_clean" if args.sweep_mode in {"stage2_submit_safe", "stage2_refine"} else "config"
    if args.sweep_mode in {"stage2_submit_safe", "stage2_refine"} and args.text_cache_path == DEFAULT_TEXT_CACHE_PATH:
        args.text_cache_path = Path(DEFAULT_TEXT_CACHE_PATH).with_name("text_emb_11_submit_clean.pt")

    args.output_dir = args.output_dir or resolve_default_output_dir(args.sweep_mode)
    if args.out_checkpoint is not None:
        if args.write_back_path is not None:
            LOGGER.warning("--out_checkpoint and --write_back_path were both provided; using --out_checkpoint")
        args.write_back_path = args.out_checkpoint
        args.write_back_checkpoint = True
    if args.write_back_checkpoint:
        args.write_back_path = args.write_back_path or resolve_default_write_back_path(args.sweep_mode)
    args = resolve_runtime_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = args.output_dir / "sweep_run.log"
    attach_file_logging(run_log_path)
    threshold_grid = resolve_threshold_grid(args.sweep_mode)
    min_area_grid = resolve_min_area_grid(args.sweep_mode)
    args.raw_prompt_weight = min(max(float(args.raw_prompt_weight), 0.0), 1.0)

    device = resolve_device(args.device)
    LOGGER.info("sweep_mode=%s", args.sweep_mode)
    LOGGER.info("checkpoint=%s", args.checkpoint)
    LOGGER.info("text_cache_path=%s", args.text_cache_path)
    LOGGER.info("prompt_fusion_mode=%s", args.prompt_fusion_mode)
    LOGGER.info("raw_prompt_weight=%.3f", args.raw_prompt_weight)
    LOGGER.info("prompt_match_mode=%s", args.prompt_match_mode)
    LOGGER.info("prompt_prototypes_source=%s", prompt_prototypes_source)
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
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    text_cache_payload = load_or_build_text_cache(
        model=model,
        tokenizer=tokenizer,
        cache_path=args.text_cache_path,
        classes=CLASSES,
        prompt_prototypes=active_prompt_prototypes,
        device=device,
        rebuild=args.rebuild_text_cache,
    )
    if set(text_cache_payload["classes"]) != set(CLASSES):
        raise RuntimeError("text cache does not match current 11 classes, please rebuild")
    for class_name in CLASSES:
        if class_name not in text_cache_payload["embeddings"]:
            raise RuntimeError(f"text cache missing embedding for class: {class_name}")
    text_cache_payload["device_embeddings"] = {
        class_name: embedding.to(device, non_blocking=True)
        for class_name, embedding in text_cache_payload["embeddings"].items()
    }
    prompt_aliases = build_prompt_aliases(active_prompt_prototypes, CLASSES)
    if args.sweep_mode in {"stage2_submit_safe", "stage2_refine"}:
        for prompt_text in ["草丛里的车", "被树遮挡的车", "远处的小车", "建筑上的窗户", "路边的垃圾桶", "路边的灯杆"]:
            LOGGER.info(
                "submit_clean alias selfcheck: %s -> %s",
                prompt_text,
                resolve_prompt_mapped_class(prompt_text, prompt_aliases, CLASSES, args.prompt_match_mode),
            )

    img_size = int(checkpoint.get("img_size", 768)) if isinstance(checkpoint, dict) else 768
    dataset = ESAMCCLIP11Dataset(
        annotation_json=args.val_json,
        split_txt=args.val_list,
        image_root=args.image_root,
        img_size=(img_size, img_size),
        classes=CLASSES,
        prompt_prototypes=active_prompt_prototypes,
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
        tokenizer,
        prompt_aliases,
        prompt_fusion_mode=args.prompt_fusion_mode,
        raw_prompt_weight=args.raw_prompt_weight,
        prompt_match_mode=args.prompt_match_mode,
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
    sample_counts = {class_name: len(records.get(class_name, [])) for class_name in CLASSES}

    best_thresholds = {}
    best_postprocess = {}
    best_scores = {}
    best_metrics = {}
    summary_rows = []

    for class_name in maybe_tqdm(CLASSES, total=len(CLASSES), desc="Sweep classes", leave=False):
        best_iou = -1.0
        best_cfg = default_cfg_for_class(class_name, threshold_grid, min_area_grid, args.sweep_mode)
        best_metric_cfg = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pred_area": 0.0,
            "gt_area": 0.0,
        }
        samples = records.get(class_name, [])

        for threshold in threshold_grid[class_name]:
            for min_area in min_area_grid[class_name]:
                fill_holes = bool(POSTPROCESS_DEFAULT[class_name]["fill_holes"])
                for topk_components in resolve_topk_grid(class_name, args.sweep_mode):
                    metric_lists = defaultdict(list)
                    for sample in samples:
                        logit = np.asarray(sample["logit"], dtype=np.float32)
                        logit = np.clip(logit, -50, 50)
                        prob = 1.0 / (1.0 + np.exp(-logit))
                        pred = (prob >= threshold).astype(np.uint8)
                        pred = apply_postprocess_with_topk(
                            pred,
                            min_area=min_area,
                            fill_holes=fill_holes,
                            topk_components=topk_components,
                        )
                        pred_t = torch.from_numpy(pred[None, None].astype(np.float32))
                        gt = sample["mask"].astype(np.uint8)
                        mask_t = torch.from_numpy(gt[None, None].astype(np.float32))
                        metrics = compute_metrics(pred_t, mask_t, threshold=0.5)
                        for metric_name in ["iou", "precision", "recall", "pred_area", "gt_area"]:
                            metric_lists[metric_name].append(float(metrics[metric_name]))

                    score = float(np.mean(metric_lists["iou"])) if metric_lists["iou"] else 0.0
                    metric_cfg = {
                        metric_name: float(np.mean(values)) if values else 0.0
                        for metric_name, values in metric_lists.items()
                    }
                    summary_rows.append([
                        class_name,
                        threshold,
                        min_area,
                        topk_components,
                        fill_holes,
                        metric_cfg.get("iou", 0.0),
                        metric_cfg.get("precision", 0.0),
                        metric_cfg.get("recall", 0.0),
                        metric_cfg.get("pred_area", 0.0),
                        metric_cfg.get("gt_area", 0.0),
                    ])
                    if score > best_iou:
                        best_iou = score
                        best_metric_cfg = metric_cfg
                        best_cfg = {
                            "threshold": float(threshold),
                            "min_area": int(min_area),
                            "fill_holes": fill_holes,
                            "topk_components": topk_components,
                        }

        best_thresholds[class_name] = best_cfg["threshold"]
        best_post_cfg = {
            "min_area": best_cfg["min_area"],
            "fill_holes": best_cfg["fill_holes"],
            "topk_components": int(best_cfg["topk_components"]) if best_cfg.get("topk_components") is not None else None,
        }
        best_postprocess[class_name] = best_post_cfg
        best_scores[class_name] = float(best_iou if best_iou >= 0.0 else 0.0)
        best_metrics[class_name] = best_metric_cfg
        LOGGER.info(
            "best %s: threshold=%.2f min_area=%s topk_components=%s score=%.4f precision=%.4f recall=%.4f pred_area=%.2f gt_area=%.2f",
            class_name,
            best_thresholds[class_name],
            best_postprocess[class_name]["min_area"],
            best_postprocess[class_name].get("topk_components"),
            best_scores[class_name],
            best_metric_cfg["precision"],
            best_metric_cfg["recall"],
            best_metric_cfg["pred_area"],
            best_metric_cfg["gt_area"],
        )

    best_metric = float(np.mean([best_scores[class_name] for class_name in CLASSES])) if CLASSES else 0.0
    group_scores = compute_group_scores(best_scores)
    objective_score = compute_objective_score(group_scores, args.objective)

    with open(args.output_dir / "best_thresholds.json", "w", encoding="utf-8") as file_obj:
        json.dump(best_thresholds, file_obj, ensure_ascii=False, indent=2)
    with open(args.output_dir / "best_postprocess.json", "w", encoding="utf-8") as file_obj:
        json.dump(best_postprocess, file_obj, ensure_ascii=False, indent=2)
    with open(args.output_dir / "threshold_sweep_summary.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["class_name", "threshold", "min_area", "topk_components", "fill_holes", "mean_iou", "precision", "recall", "pred_area", "gt_area"])
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
        checkpoint["prompt_fusion_mode"] = args.prompt_fusion_mode
        checkpoint["raw_prompt_weight"] = float(args.raw_prompt_weight)
        checkpoint["prompt_match_mode"] = args.prompt_match_mode
        checkpoint["img_size"] = img_size
        writeback_path = Path(args.write_back_path) if args.write_back_path is not None else args.checkpoint
        atomic_torch_save(checkpoint, writeback_path)
        LOGGER.info("checkpoint write-back saved: %s", writeback_path)
    else:
        writeback_path = None

    summary_payload = {
        "sweep_mode": args.sweep_mode,
        "checkpoint": str(args.checkpoint),
        "text_cache_path": str(args.text_cache_path),
        "img_size": img_size,
        "write_back_checkpoint": bool(args.write_back_checkpoint),
        "write_back_path": str(writeback_path) if writeback_path is not None else None,
        "max_samples_per_class": args.max_samples_per_class,
        "objective": args.objective,
        "prompt_prototypes_source": prompt_prototypes_source,
        "prompt_fusion_mode": args.prompt_fusion_mode,
        "raw_prompt_weight": float(args.raw_prompt_weight),
        "prompt_match_mode": args.prompt_match_mode,
        "sample_counts": sample_counts,
        "best_metric": best_metric,
        "objective_score": objective_score,
        "group_scores": group_scores,
        "all11_avg": float(group_scores.get("all11", 0.0)),
        "old5_avg": float(group_scores.get("old5", 0.0)),
        "rare_avg": float(group_scores.get("rare6", 0.0)),
        "all10_no_window_avg": float(group_scores.get("all10_no_window", 0.0)),
        "rare_no_window_avg": float(group_scores.get("rare_no_window", 0.0)),
        "best_thresholds": best_thresholds,
        "best_postprocess": best_postprocess,
        "best_scores": best_scores,
        "best_metrics": best_metrics,
        "best_per_class": {
            class_name: {
                "sample_count": int(sample_counts.get(class_name, 0)),
                "best_threshold": float(best_thresholds[class_name]),
                "best_min_area": int(best_postprocess[class_name]["min_area"]),
                "best_topk_components": best_postprocess[class_name].get("topk_components"),
                "best_fill_holes": bool(best_postprocess[class_name]["fill_holes"]),
                "best_iou": float(best_scores[class_name]),
                "best_score": float(best_scores[class_name]),
                "precision": float(best_metrics[class_name]["precision"]),
                "recall": float(best_metrics[class_name]["recall"]),
                "pred_area": float(best_metrics[class_name]["pred_area"]),
                "gt_area": float(best_metrics[class_name]["gt_area"]),
            }
            for class_name in CLASSES
        },
        "best_thresholds_path": str(args.output_dir / "best_thresholds.json"),
        "best_postprocess_path": str(args.output_dir / "best_postprocess.json"),
        "run_log_path": str(run_log_path),
    }
    save_json(
        args.output_dir / "sweep_summary.json",
        summary_payload,
    )

    summary_lines = build_sweep_summary_lines(
        sweep_mode=args.sweep_mode,
        objective=args.objective,
        checkpoint_path=args.checkpoint,
        max_samples_per_class=args.max_samples_per_class,
        prompt_prototypes_source=prompt_prototypes_source,
        prompt_fusion_mode=args.prompt_fusion_mode,
        raw_prompt_weight=args.raw_prompt_weight,
        prompt_match_mode=args.prompt_match_mode,
        sample_counts=sample_counts,
        best_metric=best_metric,
        best_thresholds=best_thresholds,
        best_postprocess=best_postprocess,
        best_scores=best_scores,
        best_metrics=best_metrics,
        group_scores=group_scores,
        objective_score=objective_score,
        write_back_path=writeback_path,
        threshold_grid=threshold_grid,
        min_area_grid=min_area_grid,
    )
    cache_artifacts = persist_sweep_summary_cache(
        output_dir=args.output_dir,
        cache_dir=DEFAULT_SWEEP_CACHE_DIR,
        checkpoint_path=args.checkpoint,
        sweep_mode=args.sweep_mode,
        summary_lines=summary_lines,
        summary_payload=summary_payload,
    )
    log_sweep_summary(summary_lines)
    LOGGER.info("sweep_run_log: %s", run_log_path)
    LOGGER.info("summary_text_path: %s", cache_artifacts["output_summary_path"])
    LOGGER.info("summary_cache_json: %s", cache_artifacts["cache_json_latest"])
    LOGGER.info("summary_cache_text: %s", cache_artifacts["cache_text_latest"])


if __name__ == "__main__":
    main()
