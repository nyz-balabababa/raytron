#!/usr/bin/env python3
"""
基于 A/B/C 分档 + 新旧 teacher 伪标签，生成训练 manifest。

输入：
- pseudo_A.json / pseudo_B.json / pseudo_C.json
- 旧版伪标签 JSON
- 新版 refine_summary.json

输出：
- train_manifest.json

manifest 是逐条 (image_path, prompt) 的训练样本清单，包含：
- 是否进入训练
- 使用哪一版/哪种融合后的 mask
- 建议 sample_weight
- tiny 标记
- building 严格策略结果
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None

import pseudo_label_policy as policy
from pseudo_label_policy import (
    RLEDecodeError,
    build_policy_decision,
    set_policy_device,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ABC_DIR = ROOT / "test" / "label_analysis" / "label_analysis_ABC(train)"
DEFAULT_OLD_JSON = ROOT / "test" / "sam3_label_old" / "train_tasks" / "pred_train_tasks.json"
DEFAULT_REFINE_SUMMARY = DEFAULT_ABC_DIR / "refine_summary.json"
DEFAULT_OUTPUT = ROOT / "test" / "label_analysis" / "train_manifest.json"
DEFAULT_DENYLIST = ROOT / "noisy_data" / "unified_denylist.txt"
DEFAULT_DEVICE = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════
# 用户配置区
# ══════════════════════════════════════════════════════════════════════

MANIFEST_ABC_DIR = DEFAULT_ABC_DIR
MANIFEST_OLD_JSON = DEFAULT_OLD_JSON
MANIFEST_NEW_JSON = None
MANIFEST_REFINE_SUMMARY = DEFAULT_REFINE_SUMMARY
MANIFEST_DENYLIST = DEFAULT_DENYLIST
MANIFEST_OUTPUT = DEFAULT_OUTPUT
MANIFEST_GROUPED_OUTPUT = None
MANIFEST_LIMIT_RECORDS = 0
MANIFEST_DEVICE = DEFAULT_DEVICE


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_json(path: Path) -> Any:
    if orjson is not None:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        return orjson.loads(data)
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None:
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_path(path_str: str) -> str:
    return str(path_str).replace("\\", "/")


def load_prompt_json(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是 prompt-style JSON list")

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in data:
        image_path = normalize_path(item["image_path"])
        out[image_path] = item.get("prompts", {})
    return out


def load_denylist(path: Optional[Path]) -> Set[str]:
    if path is None or not path.exists():
        return set()
    denylist: Set[str] = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            denylist.add(normalize_path(line))
    return denylist


def load_grade_map(abc_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    files = {
        "A": abc_dir / "pseudo_A.json",
        "B": abc_dir / "pseudo_B.json",
        "C": abc_dir / "pseudo_C.json",
    }
    grade_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for grade, path in files.items():
        prompt_json = load_prompt_json(path)
        for image_path, prompts in prompt_json.items():
            for prompt, info in prompts.items():
                key = (image_path, prompt)
                if key in grade_map:
                    raise ValueError(f"重复分档记录: {key} 同时出现在多个 A/B/C 文件中")
                grade_map[key] = {
                    "grade": grade,
                    "new_info": info,
                }
    return grade_map


def load_refine_summary(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, float]]]:
    summary = load_json(path)
    if not isinstance(summary, dict):
        raise ValueError(f"{path} 必须是 refine_summary.json")

    row_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in summary.get("rows", []):
        image_path = normalize_path(row["image_path"])
        prompt = str(row["prompt"])
        row_map[(image_path, prompt)] = row
    prompt_stats = summary.get("prompt_stats", {})
    return row_map, prompt_stats


def build_grouped_preview(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not record["include_in_train"]:
            continue
        image_path = record["image_path"]
        if image_path not in grouped:
            grouped[image_path] = {"image_path": image_path, "prompts": {}}
        grouped[image_path]["prompts"][record["prompt"]] = {
            "hit": bool(record["selected_hit"]),
            "score": float(record["sample_weight"]),
            "instances": int(record["selected_instances"]),
            "rle": record["rle"],
            "sample_weight": float(record["sample_weight"]),
            "grade": record["grade"],
            "mask_source": record["mask_source"],
            "flags": record["flags"],
        }
    return list(grouped.values())


def summarize_manifest(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    grade_prompt_counts: Dict[str, Counter] = defaultdict(Counter)
    exclude_grade_prompt_counts: Dict[str, Counter] = defaultdict(Counter)
    source_counts = Counter()
    tiny_counts = Counter()
    tiny_include_by_class = Counter()
    tiny_exclude_by_class = Counter()
    disabled_reason_counts = Counter()
    drop_reason_counts = Counter()
    flags_counts = Counter()
    building_c_include_count = 0
    building_c_exclude_count = 0
    weight_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weight_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in records:
        grade = row["grade"]
        prompt = row["prompt"]
        for flag in row["flags"]:
            flags_counts[flag] += 1
        for reason in row["drop_reasons"]:
            drop_reason_counts[reason] += 1
        if row["disabled_reason"]:
            disabled_reason_counts[row["disabled_reason"]] += 1
        if prompt == "building" and grade == "C":
            if row["include_in_train"]:
                building_c_include_count += 1
            else:
                building_c_exclude_count += 1
        if row["include_in_train"]:
            counts["include_total"] += 1
            counts[f"include_grade_{grade}"] += 1
            counts[f"include_prompt_{prompt}"] += 1
            grade_prompt_counts[grade][prompt] += 1
            source_counts[row["mask_source"]] += 1
            if row["selected_hit"]:
                counts["positive_total"] += 1
                weight_sums[prompt][grade] += float(row["sample_weight"])
                weight_counts[prompt][grade] += 1
            else:
                counts["negative_total"] += 1
            if row["is_tiny"]:
                tiny_counts[prompt] += 1
                tiny_include_by_class[prompt] += 1
        else:
            counts["excluded_total"] += 1
            counts[f"excluded_grade_{grade}"] += 1
            exclude_grade_prompt_counts[grade][prompt] += 1
            if row["is_tiny"]:
                tiny_exclude_by_class[prompt] += 1

    mean_weight_by_class_grade: Dict[str, Dict[str, float]] = {}
    for prompt, grade_map in weight_sums.items():
        mean_weight_by_class_grade[prompt] = {}
        for grade, weight_sum in grade_map.items():
            denom = max(weight_counts[prompt][grade], 1)
            mean_weight_by_class_grade[prompt][grade] = float(weight_sum) / float(denom)

    return {
        "counts": dict(counts),
        "mask_source_counts": dict(source_counts),
        "tiny_counts": dict(tiny_counts),
        "include_by_grade_prompt": {
            grade: dict(counter)
            for grade, counter in sorted(grade_prompt_counts.items())
        },
        "exclude_by_grade_prompt": {
            grade: dict(counter)
            for grade, counter in sorted(exclude_grade_prompt_counts.items())
        },
        "disabled_reason_counts": dict(disabled_reason_counts),
        "drop_reason_counts": dict(drop_reason_counts),
        "flags_counts": dict(flags_counts),
        "mean_weight_by_class_grade": mean_weight_by_class_grade,
        "building_c_include_count": building_c_include_count,
        "building_c_exclude_count": building_c_exclude_count,
        "tiny_include_by_class": dict(tiny_include_by_class),
        "tiny_exclude_by_class": dict(tiny_exclude_by_class),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建伪标签训练 manifest")
    parser.add_argument("--abc-dir", type=Path, default=MANIFEST_ABC_DIR)
    parser.add_argument("--old-json", type=Path, default=MANIFEST_OLD_JSON)
    parser.add_argument(
        "--new-json",
        type=Path,
        default=MANIFEST_NEW_JSON,
        help="可选，仅用于完整性校验和统计 missing_in_new / extra_in_new；manifest 的 new_info 始终来自 pseudo_A/B/C",
    )
    parser.add_argument("--refine-summary", type=Path, default=MANIFEST_REFINE_SUMMARY)
    parser.add_argument(
        "--denylist",
        type=Path,
        default=MANIFEST_DENYLIST,
        help="可选，denylist 内的 image_path 会被标记为不进入训练；文件不存在时自动跳过",
    )
    parser.add_argument("--output", type=Path, default=MANIFEST_OUTPUT)
    parser.add_argument(
        "--grouped-output",
        type=Path,
        default=MANIFEST_GROUPED_OUTPUT,
        help="可选，额外导出 prompt-style grouped JSON，便于后续直接接现有训练脚本",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=MANIFEST_LIMIT_RECORDS,
        help="可选，仅构建前 N 条记录，便于快速调试；0 表示全量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=MANIFEST_DEVICE,
        help="运行设备。当前脚本的 GPU 加速主要用于像素级 IoU 计算；形态学组件分析仍依赖 CPU OpenCV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_policy_device(args.device)
    device_msg = args.device
    if args.device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        device_msg = "cpu (CUDA unavailable, fallback)"
        set_policy_device("cpu")

    print(f"device:            {device_msg}")
    if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_index = 0
        if ":" in args.device:
            try:
                gpu_index = int(args.device.split(":", 1)[1])
            except Exception:
                gpu_index = 0
        print(f"gpu_name:          {torch.cuda.get_device_name(gpu_index)}")
    print(f"json_backend:      {'orjson' if orjson is not None else 'json'}")
    print(f"pycocotools:       {'yes' if policy.HAS_PYCOCOTOOLS else 'no (slow Python RLE fallback)'}")

    t0 = time.perf_counter()
    grade_map = load_grade_map(args.abc_dir)
    t1 = time.perf_counter()
    old_data = load_prompt_json(args.old_json)
    t2 = time.perf_counter()
    new_data = load_prompt_json(args.new_json) if args.new_json is not None else None
    denylist = load_denylist(args.denylist)
    row_map, prompt_stats = load_refine_summary(args.refine_summary)
    t3 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    missing_in_new: Optional[int] = 0 if new_data is not None else None
    extra_in_new: Optional[int] = None
    missing_in_summary = 0
    new_keys: Optional[set[Tuple[str, str]]] = None

    if new_data is not None:
        grade_keys = set(grade_map.keys())
        new_keys = {
            (image_path, prompt)
            for image_path, prompts in new_data.items()
            for prompt in prompts.keys()
        }
        extra_in_new = len(new_keys - grade_keys)

    keys = sorted(grade_map.keys())
    if args.limit_records > 0:
        keys = keys[: args.limit_records]

    build_loop_start = time.perf_counter()
    for image_path, prompt in progress(
        keys,
        total=len(keys),
        desc="Build manifest",
        unit="record",
    ):
        grade_info = grade_map[(image_path, prompt)]
        grade = grade_info["grade"]
        new_info = grade_info["new_info"]
        old_info = old_data.get(image_path, {}).get(prompt)
        summary_row = row_map.get((image_path, prompt))

        if new_keys is not None and (image_path, prompt) not in new_keys:
            assert missing_in_new is not None
            missing_in_new += 1
        if summary_row is None:
            missing_in_summary += 1
            summary_row = {
                "image_path": image_path,
                "prompt": prompt,
                "label": grade,
                "score": float(new_info.get("score", 0.0)),
                "variant_hit_flags": [],
                "variant_mean_iou": None,
            }

        decision = build_policy_decision(
            image_path=image_path,
            prompt=prompt,
            grade=grade,
            new_info=new_info,
            old_info=old_info,
            summary_row=summary_row,
            prompt_stats=prompt_stats,
        )

        new_hit = bool(new_info.get("hit")) and new_info.get("rle") is not None
        old_hit = bool(old_info and old_info.get("hit") and old_info.get("rle") is not None)
        old_new_iou = decision.old_new_iou

        include_in_train = decision.include_in_train
        selected_hit = decision.selected_hit
        sample_weight = decision.sample_weight
        mask_source = decision.mask_source
        disabled_reason = decision.disabled_reason
        selected_score = decision.selected_score
        selected_instances = decision.selected_instances
        flags = list(decision.flags)
        drop_reasons = list(decision.drop_reasons)
        selected_rle = decision.selected_rle

        if image_path in denylist:
            include_in_train = False
            selected_hit = False
            sample_weight = 0.0
            mask_source = "denylist_excluded"
            disabled_reason = "denylist_excluded"
            selected_score = 0.0
            selected_instances = 0
            selected_rle = None
            flags.append("denylist_excluded")

        record = {
            "image_path": image_path,
            "prompt": prompt,
            "grade": grade,
            "include_in_train": include_in_train,
            "selected_hit": selected_hit,
            "sample_weight": sample_weight,
            "mask_source": mask_source,
            "disabled_reason": disabled_reason,
            "selected_score": selected_score,
            "selected_instances": selected_instances,
            "is_tiny": decision.is_tiny,
            "component_count": decision.component_count,
            "max_component_area": decision.max_component_area,
            "total_mask_area": decision.total_mask_area,
            "main_component_ratio": decision.main_component_ratio,
            "max_aspect_ratio": decision.max_aspect_ratio,
            "new_hit": new_hit,
            "old_hit": old_hit,
            "old_new_iou": old_new_iou,
            "new_score": float(new_info.get("score", 0.0)),
            "old_score": float(old_info.get("score", 0.0)) if old_info else 0.0,
            "score_q25": float(prompt_stats.get(prompt, {}).get("score_q25", 0.0)),
            "flags": sorted(set(flags)),
            "drop_reasons": sorted(set(drop_reasons)),
            "summary_row": {
                "area_ratio": summary_row.get("area_ratio"),
                "num_cc": summary_row.get("num_cc"),
                "max_comp_ratio": summary_row.get("max_comp_ratio"),
                "contrast_std": summary_row.get("contrast_std"),
                "laplacian": summary_row.get("laplacian"),
                "variant_hit_flags": summary_row.get("variant_hit_flags"),
                "variant_mean_iou": summary_row.get("variant_mean_iou"),
            },
            "rle": selected_rle,
        }
        records.append(record)
    build_loop_end = time.perf_counter()

    manifest = {
        "metadata": {
            "abc_dir": str(args.abc_dir),
            "old_json": str(args.old_json),
            "new_json": str(args.new_json) if args.new_json is not None else None,
            "refine_summary": str(args.refine_summary),
            "denylist": str(args.denylist) if args.denylist is not None and args.denylist.exists() else None,
            "denylist_size": len(denylist),
            "total_records": len(records),
            "missing_in_new": missing_in_new,
            "extra_in_new": extra_in_new,
            "missing_in_summary": missing_in_summary,
        },
        "summary": summarize_manifest(records),
        "records": records,
    }
    dump_start = time.perf_counter()
    dump_json(args.output, manifest)

    if args.grouped_output is not None:
        grouped = build_grouped_preview(records)
        dump_json(args.grouped_output, grouped)
    dump_end = time.perf_counter()

    print(f"records_total:     {len(records)}")
    print(f"missing_in_new:    {missing_in_new}")
    print(f"extra_in_new:      {extra_in_new}")
    print(f"missing_in_summary:{missing_in_summary}")
    print(f"load_abc_sec:      {t1 - t0:.2f}")
    print(f"load_old_sec:      {t2 - t1:.2f}")
    print(f"load_summary_sec:  {t3 - t2:.2f}")
    print(f"build_loop_sec:    {build_loop_end - build_loop_start:.2f}")
    print(f"dump_sec:          {dump_end - dump_start:.2f}")
    print(f"output:            {args.output}")
    if args.grouped_output is not None:
        print(f"grouped_output:    {args.grouped_output}")


if __name__ == "__main__":
    main()
