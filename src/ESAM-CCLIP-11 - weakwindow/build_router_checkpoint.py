from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from config_esam_cclip_11 import CLASSES as CONFIG_CLASSES
except Exception:
    CONFIG_CLASSES = None

DEFAULT_CLASSES = [
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
CLASSES = list(CONFIG_CLASSES) if CONFIG_CLASSES is not None else list(DEFAULT_CLASSES)
OLD5_CLASSES = ["person", "car", "building", "tree", "animal"]
RARE_CLASSES = ["trash can", "window", "door", "fence", "pole_light", "motorcycle"]

DEFAULT_ROUTE_MODE = "old_lite_rare_stage2"
DEFAULT_MAX_PARAMS = 300_000_000
DEFAULT_PROMPT_FUSION_MODE = "prototype"
DEFAULT_RAW_PROMPT_WEIGHT = 0.0
DEFAULT_PROMPT_MATCH_MODE = "exact"
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


def load_checkpoint_payload(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint 不是 dict: {checkpoint_path}")
    return checkpoint


def load_state_dict_flexible(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise RuntimeError("无法从 checkpoint 中提取 state_dict")


def strip_common_prefixes(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    return key


def remap_decoder_key(key: str) -> str:
    key = strip_common_prefixes(key)
    if key.startswith("fusion_decoder."):
        key = f"decoder.{key[len('fusion_decoder.') :]}"
    return key


def normalize_decoder_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = remap_decoder_key(str(raw_key))
        if key.startswith("decoder."):
            key = key[len("decoder.") :]
        normalized[key] = value.detach().cpu()
    return normalized


def extract_decoder_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = load_state_dict_flexible(checkpoint)
    decoder_state: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = remap_decoder_key(str(raw_key))
        if key.startswith("decoder."):
            decoder_state[key] = value
    if not decoder_state:
        raise RuntimeError("decoder_key_count=0，未能从 checkpoint 中提取 decoder 参数。")
    return normalize_decoder_state_dict_keys(decoder_state)


def resolve_checkpoint_classes(checkpoint: Dict[str, Any]) -> List[str]:
    classes = list(checkpoint.get("classes", []))
    if classes != CLASSES:
        raise RuntimeError(f"checkpoint classes 不匹配当前 11 类。ckpt={classes} expected={CLASSES}")
    return classes


def resolve_checkpoint_thresholds(checkpoint: Dict[str, Any]) -> Dict[str, float]:
    raw_thresholds = checkpoint.get("prompt_thresholds") or checkpoint.get("val_thresholds")
    if not isinstance(raw_thresholds, dict):
        raise RuntimeError("checkpoint 缺少 prompt_thresholds / val_thresholds")
    return {str(key): float(value) for key, value in raw_thresholds.items()}


def resolve_checkpoint_postprocess(checkpoint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_postprocess = checkpoint.get("postprocess_cfg") or checkpoint.get("postprocess")
    if not isinstance(raw_postprocess, dict):
        raise RuntimeError("checkpoint 缺少 postprocess_cfg / postprocess")
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, cfg in raw_postprocess.items():
        if not isinstance(cfg, dict):
            raise RuntimeError(f"postprocess 配置非法: {key} -> {cfg}")
        out_cfg = {
            "min_area": int(cfg.get("min_area", 0)),
            "fill_holes": bool(cfg.get("fill_holes", False)),
        }
        if cfg.get("topk_components") is not None:
            out_cfg["topk_components"] = int(cfg.get("topk_components", 0))
        normalized[str(key)] = out_cfg
    return normalized


def build_route_map(route_mode: str) -> Dict[str, str]:
    if route_mode != DEFAULT_ROUTE_MODE:
        raise RuntimeError(f"暂不支持的 route_mode: {route_mode}")
    route_map = {class_name: "lite" for class_name in OLD5_CLASSES}
    route_map.update({class_name: "stage2" for class_name in RARE_CLASSES})
    if set(route_map.keys()) != set(CLASSES):
        raise RuntimeError("route_map 未覆盖完整 11 类")
    return route_map


def merge_classwise_config(
    classes: List[str],
    route_map: Dict[str, str],
    lite_values: Dict[str, Any],
    stage2_values: Dict[str, Any],
    value_name: str,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    missing = []
    for class_name in classes:
        source_values = lite_values if route_map[class_name] == "lite" else stage2_values
        if class_name not in source_values:
            missing.append(class_name)
            continue
        merged[class_name] = copy.deepcopy(source_values[class_name])
    if missing:
        raise RuntimeError(f"{value_name} 缺少类别配置: {missing}")
    return merged


def count_tensor_params(state_dict: Dict[str, Any]) -> int:
    return int(
        sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
    )


def atomic_torch_save(payload: Dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    torch.save(payload, tmp_path)
    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        raise RuntimeError("temporary checkpoint write failed")
    os.replace(tmp_path, path_obj)


def main() -> None:
    if list(CLASSES) != DEFAULT_CLASSES:
        raise RuntimeError("当前 router 只支持固定 11 类，CLASSES 不一致")

    parser = argparse.ArgumentParser()
    parser.add_argument("--lite_checkpoint", required=True, type=str)
    parser.add_argument("--stage2_checkpoint", required=True, type=str)
    parser.add_argument("--out_checkpoint", required=True, type=str)
    parser.add_argument("--route_mode", default=DEFAULT_ROUTE_MODE, type=str)
    parser.add_argument("--max_params", default=DEFAULT_MAX_PARAMS, type=int)
    parser.add_argument("--store_lite_decoder_copy", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    lite_checkpoint = load_checkpoint_payload(args.lite_checkpoint)
    stage2_checkpoint = load_checkpoint_payload(args.stage2_checkpoint)

    classes = resolve_checkpoint_classes(lite_checkpoint)
    resolve_checkpoint_classes(stage2_checkpoint)

    route_map = build_route_map(args.route_mode)

    lite_decoder_state_dict = extract_decoder_state_dict(lite_checkpoint)
    stage2_decoder_state_dict = extract_decoder_state_dict(stage2_checkpoint)
    if len(lite_decoder_state_dict) == 0:
        raise RuntimeError("lite_decoder_key_count == 0")
    if len(stage2_decoder_state_dict) == 0:
        raise RuntimeError("stage2_decoder_key_count == 0")

    lite_thresholds = resolve_checkpoint_thresholds(lite_checkpoint)
    stage2_thresholds = resolve_checkpoint_thresholds(stage2_checkpoint)
    lite_postprocess = resolve_checkpoint_postprocess(lite_checkpoint)
    stage2_postprocess = resolve_checkpoint_postprocess(stage2_checkpoint)

    final_thresholds = merge_classwise_config(
        classes=classes,
        route_map=route_map,
        lite_values=lite_thresholds,
        stage2_values=stage2_thresholds,
        value_name="thresholds",
    )
    final_postprocess = merge_classwise_config(
        classes=classes,
        route_map=route_map,
        lite_values=lite_postprocess,
        stage2_values=stage2_postprocess,
        value_name="postprocess",
    )

    output_checkpoint = copy.deepcopy(lite_checkpoint)
    if args.store_lite_decoder_copy:
        output_checkpoint["decoder_lite_state_dict"] = lite_decoder_state_dict
    else:
        output_checkpoint.pop("decoder_lite_state_dict", None)
    output_checkpoint["decoder_stage2_state_dict"] = stage2_decoder_state_dict
    output_checkpoint["router_enabled"] = True
    output_checkpoint["router_type"] = "classwise_decoder"
    output_checkpoint["route_mode"] = args.route_mode
    output_checkpoint["route_map"] = route_map
    output_checkpoint["prompt_thresholds"] = final_thresholds
    output_checkpoint["val_thresholds"] = final_thresholds
    output_checkpoint["postprocess"] = final_postprocess
    output_checkpoint["postprocess_cfg"] = final_postprocess
    output_checkpoint["prompt_fusion_mode"] = DEFAULT_PROMPT_FUSION_MODE
    output_checkpoint["raw_prompt_weight"] = DEFAULT_RAW_PROMPT_WEIGHT
    output_checkpoint["prompt_match_mode"] = DEFAULT_PROMPT_MATCH_MODE
    output_checkpoint["prompt_prototypes"] = SUBMIT_CLEAN_PROMPT_PROTOTYPES
    output_checkpoint["prompt_prototypes_source"] = "submit_clean_router"
    output_checkpoint["use_prompt_prototype"] = True

    base_model_state_dict = load_state_dict_flexible(output_checkpoint)
    base_model_params = count_tensor_params(base_model_state_dict)
    stage2_decoder_params = count_tensor_params(stage2_decoder_state_dict)
    total_runtime_params = base_model_params + stage2_decoder_params
    if total_runtime_params > int(args.max_params):
        raise RuntimeError(
            f"router runtime params 超限: total={total_runtime_params} max={int(args.max_params)}"
        )

    output_checkpoint["base_model_params"] = int(base_model_params)
    output_checkpoint["stage2_decoder_params"] = int(stage2_decoder_params)
    output_checkpoint["total_runtime_params"] = int(total_runtime_params)

    out_path = Path(args.out_checkpoint)
    print("=" * 72)
    print("Router checkpoint build dry-run complete" if args.dry_run else "Router checkpoint build complete")
    print(f"lite checkpoint     : {args.lite_checkpoint}")
    print(f"stage2 checkpoint   : {args.stage2_checkpoint}")
    print(f"out checkpoint      : {out_path}")
    print(f"classes             : {classes}")
    print(f"route mode          : {args.route_mode}")
    print(f"route map           : {route_map}")
    print(f"final thresholds    : {final_thresholds}")
    print(f"final postprocess   : {final_postprocess}")
    print(f"lite decoder keys   : {len(lite_decoder_state_dict)}")
    print(f"stage2 decoder keys : {len(stage2_decoder_state_dict)}")
    print(f"prompt_fusion_mode  : {DEFAULT_PROMPT_FUSION_MODE}")
    print(f"raw_prompt_weight   : {DEFAULT_RAW_PROMPT_WEIGHT}")
    print(f"prompt_match_mode   : {DEFAULT_PROMPT_MATCH_MODE}")
    print("prompt_prototypes_source : submit_clean_router")
    print(f"base model params   : {base_model_params}")
    print(f"stage2 decoder params: {stage2_decoder_params}")
    print(f"total runtime params: {total_runtime_params}")
    print("=" * 72)
    if not args.dry_run:
        atomic_torch_save(output_checkpoint, out_path)


if __name__ == "__main__":
    main()
