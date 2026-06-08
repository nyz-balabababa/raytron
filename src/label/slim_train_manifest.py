#!/usr/bin/env python3
"""
Slim an existing full train manifest into a lighter training manifest.

This script only:
- reads an existing manifest JSON
- drops non-essential large fields
- nulls RLE for non-training-positive records
- writes a smaller JSON and an adjacent stats JSON

It does NOT:
- rerun pseudo-label policy
- decode or re-encode RLE
- mutate the source manifest
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

ROOT = Path(__file__).resolve().parents[2]

# ══════════════════════════════════════════════════════════════════════
# 用户配置区：常改输入输出 / 压缩参数
# ══════════════════════════════════════════════════════════════════════

DEFAULT_INPUT = ROOT / "test" / "label_analysis" / "train_manifest.json"
DEFAULT_OUTPUT = ROOT / "test" / "label_analysis" / "train_manifest_light.json"
DEFAULT_KEEP_SUMMARY = True
DEFAULT_GZIP = False
DEFAULT_INDENT = None
DEFAULT_STATS_SUFFIX = "_stats.json"
DEFAULT_DEVICE = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"

KEEP_RECORD_FIELDS = [
    "image_path",
    "prompt",
    "include_in_train",
    "selected_hit",
    "sample_weight",
    "rle",
    "grade",
    "mask_source",
    "is_tiny",
    "flags",
    "disabled_reason",
    "component_count",
    "max_component_area",
    "total_mask_area",
]

KEEP_METADATA_FIELDS = [
    "abc_dir",
    "old_json",
    "refine_summary",
    "denylist",
    "denylist_size",
    "total_records",
    "missing_in_summary",
]

CORE_SUMMARY_FIELDS = [
    "counts",
    "mask_source_counts",
    "tiny_counts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slim full train manifest into a light manifest")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-summary",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_KEEP_SUMMARY,
        help="Whether to keep the original full summary. Default: true. Use --no-keep-summary to keep only core summary fields.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        default=DEFAULT_GZIP,
        help="Also write a .json.gz file next to output-json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=DEFAULT_INDENT,
        help="JSON indent. Default None for compact output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="运行设备。当前脚本主要是 JSON 裁剪与写出，GPU 仅做设备检测与日志展示。",
    )
    return parser.parse_args()


def resolve_user_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


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


def dump_json(path: Path, data: Any, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None and indent is None:
        path.write_bytes(orjson.dumps(data))
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


def dump_json_gz(path: Path, data: Any, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None and indent is None:
        payload = orjson.dumps(data)
    else:
        payload = json.dumps(
            data,
            ensure_ascii=False,
            indent=indent,
            separators=None if indent else (",", ":"),
        ).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)


def to_records_and_container(data: Any) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any] | None]:
    if isinstance(data, dict):
        records = data.get("records", [])
        if not isinstance(records, list):
            raise ValueError("manifest['records'] 必须是 list")
        return records, "dict", data
    if isinstance(data, list):
        return data, "list", None
    raise ValueError("输入 manifest 必须是 dict 或 list")


def slim_metadata(metadata: Dict[str, Any], input_json: Path) -> Dict[str, Any]:
    out = {key: metadata.get(key) for key in KEEP_METADATA_FIELDS if key in metadata}
    out.update(
        {
            "slimmed": True,
            "source_manifest": str(input_json),
            "slim_fields": list(KEEP_RECORD_FIELDS),
            "removed_large_fields": True,
        }
    )
    return out


def slim_summary(summary: Dict[str, Any], keep_summary: bool) -> Dict[str, Any]:
    if keep_summary:
        return summary
    return {key: summary.get(key) for key in CORE_SUMMARY_FIELDS if key in summary}


def slim_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
    slim = {key: record.get(key) for key in KEEP_RECORD_FIELDS if key in record}

    include_in_train = bool(record.get("include_in_train", False))
    selected_hit = bool(record.get("selected_hit", False))
    original_rle = record.get("rle")
    keep_rle = include_in_train and selected_hit and original_rle is not None

    slim["include_in_train"] = include_in_train
    slim["selected_hit"] = selected_hit
    slim["rle"] = original_rle if keep_rle else None

    return slim, original_rle is not None, keep_rle


def summarize_light(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_prompt = Counter()
    by_grade = Counter()
    by_mask_source = Counter()
    for row in records:
        by_prompt[str(row.get("prompt", ""))] += 1
        by_grade[str(row.get("grade", ""))] += 1
        by_mask_source[str(row.get("mask_source", ""))] += 1
    return {
        "by_prompt": dict(sorted(by_prompt.items())),
        "by_grade": dict(sorted(by_grade.items())),
        "by_mask_source": dict(sorted(by_mask_source.items())),
    }


def build_stats(
    input_json: Path,
    output_json: Path,
    gzip_path: Path | None,
    total_records: int,
    include_in_train: int,
    positive_records: int,
    negative_records: int,
    excluded_records: int,
    records_with_rle: int,
    records_rle_set_null: int,
    light_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    input_size_mb = input_json.stat().st_size / (1024 * 1024)
    output_size_mb = output_json.stat().st_size / (1024 * 1024)
    gzip_size_mb = (gzip_path.stat().st_size / (1024 * 1024)) if gzip_path is not None and gzip_path.exists() else 0.0
    summary = summarize_light(light_records)
    return {
        "input_json": str(input_json),
        "output_json": str(output_json),
        "input_size_mb": round(input_size_mb, 3),
        "output_size_mb": round(output_size_mb, 3),
        "gzip_size_mb": round(gzip_size_mb, 3),
        "total_records": total_records,
        "include_in_train": include_in_train,
        "positive_records": positive_records,
        "negative_records": negative_records,
        "excluded_records": excluded_records,
        "records_with_rle": records_with_rle,
        "records_rle_set_null": records_rle_set_null,
        **summary,
    }


def main() -> None:
    args = parse_args()
    args.input_json = resolve_user_path(args.input_json)
    args.output_json = resolve_user_path(args.output_json)

    device_msg = args.device
    if args.device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        device_msg = "cpu (CUDA unavailable, fallback)"
    print(f"device:               {device_msg}")
    if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_index = 0
        if ":" in args.device:
            try:
                gpu_index = int(args.device.split(":", 1)[1])
            except Exception:
                gpu_index = 0
        print(f"gpu_name:             {torch.cuda.get_device_name(gpu_index)}")
    print(f"json_backend:         {'orjson' if orjson is not None else 'json'}")
    print(f"tqdm:                 {'yes' if tqdm is not None else 'no'}")

    source = load_json(args.input_json)
    records, container_type, container = to_records_and_container(source)

    light_records: List[Dict[str, Any]] = []
    include_in_train = 0
    positive_records = 0
    negative_records = 0
    excluded_records = 0
    records_with_rle = 0
    records_rle_set_null = 0

    for record in progress(records, total=len(records), desc="Slim manifest", unit="record"):
        if not isinstance(record, dict):
            raise ValueError("records 中每条记录都必须是 dict")
        light, had_original_rle, kept_rle = slim_record(record)
        light_records.append(light)

        if bool(light.get("include_in_train", False)):
            include_in_train += 1
            if bool(light.get("selected_hit", False)):
                positive_records += 1
            else:
                negative_records += 1
        else:
            excluded_records += 1

        if kept_rle:
            records_with_rle += 1
        elif had_original_rle:
            records_rle_set_null += 1

    if container_type == "dict":
        metadata = slim_metadata(dict(container.get("metadata", {})), args.input_json)
        summary = slim_summary(dict(container.get("summary", {})), args.keep_summary)
        light_manifest: Any = {
            "metadata": metadata,
            "summary": summary,
            "records": light_records,
        }
    else:
        light_manifest = light_records

    dump_json(args.output_json, light_manifest, args.indent)

    gzip_path = None
    if args.gzip:
        gzip_path = args.output_json.with_suffix(args.output_json.suffix + ".gz")
        dump_json_gz(gzip_path, light_manifest, args.indent)

    stats = build_stats(
        input_json=args.input_json,
        output_json=args.output_json,
        gzip_path=gzip_path,
        total_records=len(light_records),
        include_in_train=include_in_train,
        positive_records=positive_records,
        negative_records=negative_records,
        excluded_records=excluded_records,
        records_with_rle=records_with_rle,
        records_rle_set_null=records_rle_set_null,
        light_records=light_records,
    )

    stats_path = args.output_json.with_name(f"{args.output_json.stem}{DEFAULT_STATS_SUFFIX}")
    dump_json(stats_path, stats, indent=2)

    print(f"input_json:           {args.input_json}")
    print(f"output_json:          {args.output_json}")
    print(f"stats_json:           {stats_path}")
    if gzip_path is not None:
        print(f"gzip_json:            {gzip_path}")
    print(f"input_size_mb:        {stats['input_size_mb']}")
    print(f"output_size_mb:       {stats['output_size_mb']}")
    if gzip_path is not None:
        print(f"gzip_size_mb:         {stats['gzip_size_mb']}")
    print(f"total_records:        {stats['total_records']}")
    print(f"include_in_train:     {stats['include_in_train']}")
    print(f"positive_records:     {stats['positive_records']}")
    print(f"negative_records:     {stats['negative_records']}")
    print(f"excluded_records:     {stats['excluded_records']}")
    print(f"records_with_rle:     {stats['records_with_rle']}")
    print(f"records_rle_set_null: {stats['records_rle_set_null']}")


if __name__ == "__main__":
    main()
