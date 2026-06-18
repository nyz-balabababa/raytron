#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_esam_cclip_11_urf.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parents[1] / "test" / "train_output"
AUTO_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

PIPELINE_DEFINITIONS = {
    "unfreeze_then_refine": [
        {
            "stage_name": "A1_unfreeze",
            "preset": "partial_unfreeze_balanced_safe",
            "extra_args": ["--no_use_refine_head"],
        },
        {
            "stage_name": "A2_recalibrate",
            "preset": "balanced_recalibrate",
            "extra_args": ["--no_use_refine_head"],
        },
        {
            "stage_name": "A3_refine",
            "preset": "zero_init_refine",
            "extra_args": ["--use_refine_head", "--train_refine_head"],
        },
    ],
    "refine_then_unfreeze": [
        {
            "stage_name": "B1_refine",
            "preset": "zero_init_refine",
            "extra_args": ["--use_refine_head", "--train_refine_head"],
        },
        {
            "stage_name": "B2_unfreeze",
            "preset": "partial_unfreeze_balanced_safe",
            "extra_args": [
                "--use_refine_head",
                "--train_refine_head",
                "--partial_unfreeze_image_encoder",
            ],
        },
        {
            "stage_name": "B3_recalibrate",
            "preset": "balanced_recalibrate_refine",
            "extra_args": ["--use_refine_head", "--train_refine_head"],
        },
    ],
}


def make_unique_pipeline_root(output_root: Path, pipeline_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for suffix_idx in range(100):
        suffix = f"_{timestamp}" if suffix_idx == 0 else f"_{timestamp}_{suffix_idx:02d}"
        candidate = output_root / f"urf_{pipeline_name}{suffix}"
        if candidate.exists():
            continue
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    raise RuntimeError(f"无法创建唯一 pipeline 输出目录: {output_root}")


def resolve_pipeline_root(
    output_root: Path,
    pipeline_name: str,
    explicit_pipeline_root: Path | None,
) -> Path:
    if explicit_pipeline_root is not None:
        pipeline_root = explicit_pipeline_root.resolve()
        pipeline_root.mkdir(parents=True, exist_ok=True)
        return pipeline_root
    output_root.mkdir(parents=True, exist_ok=True)
    return make_unique_pipeline_root(output_root.resolve(), pipeline_name)


def build_stage_command(
    python_exe: str,
    pipeline_label: str,
    stage_spec: dict,
    resume_path: Path,
    stage_output_dir: Path,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
    reference_old5: float,
) -> list[str]:
    stage_name = stage_spec["stage_name"]
    command = [
        python_exe,
        str(TRAIN_SCRIPT),
        "--preset",
        stage_spec["preset"],
        "--resume",
        str(resume_path),
        "--resume_weights_only",
        "--no_auto_resume",
        "--output_dir",
        str(stage_output_dir),
        "--run_name",
        stage_name,
        "--pipeline_name",
        pipeline_label,
        "--pipeline_stage",
        stage_name,
    ]
    if device:
        command.extend(["--device", device])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        command.extend(["--num_workers", str(num_workers)])
    if float(reference_old5) >= 0:
        command.extend(["--reference_old5", str(reference_old5)])
    command.extend(stage_spec.get("extra_args", []))
    return command


def resolve_stage_run_dir(stage_output_dir: Path, stage_name: str) -> Path:
    return stage_output_dir / stage_name


def find_resume_checkpoint(stage_run_dir: Path, allow_missing: bool = False) -> Path:
    best_path = stage_run_dir / "best.pt"
    last_path = stage_run_dir / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        print(f"[WARN] best.pt 不存在，fallback 到 last.pt: {last_path}")
        return last_path
    if allow_missing:
        print(f"[WARN] dry_run 未找到 checkpoint，使用占位 best.pt 路径: {best_path}")
        return best_path
    raise FileNotFoundError(f"stage 输出缺少 best.pt / last.pt: {stage_run_dir}")


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def stage_name_list(stage_specs: list[dict]) -> list[str]:
    return [spec["stage_name"] for spec in stage_specs]


def validate_stage_name(stage_names: list[str], stage_name: str | None, arg_name: str) -> str | None:
    if stage_name is None:
        return None
    if stage_name not in stage_names:
        raise ValueError(f"{arg_name}={stage_name} 不属于当前 pipeline，可选: {', '.join(stage_names)}")
    return stage_name


def resolve_stage_window(
    stage_specs: list[dict],
    start_stage: str | None,
    stop_after_stage: str | None,
) -> tuple[int, int]:
    stage_names = stage_name_list(stage_specs)
    start_stage = validate_stage_name(stage_names, start_stage, "--start_stage")
    stop_after_stage = validate_stage_name(stage_names, stop_after_stage, "--stop_after_stage")
    start_idx = stage_names.index(start_stage) if start_stage is not None else 0
    stop_idx = stage_names.index(stop_after_stage) if stop_after_stage is not None else len(stage_specs) - 1
    if stop_idx < start_idx:
        raise ValueError("--stop_after_stage 不能早于 --start_stage")
    return start_idx, stop_idx


def resolve_initial_resume(
    stage_specs: list[dict],
    pipeline_root: Path,
    base_ckpt: Path,
    start_idx: int,
    resume_override: Path | None,
    dry_run: bool,
) -> Path:
    if resume_override is not None:
        resolved = resume_override.resolve()
        if not dry_run and not resolved.exists():
            raise FileNotFoundError(f"resume_override not found: {resolved}")
        if dry_run and not resolved.exists():
            print(f"[WARN] dry_run resume_override 不存在，保留占位路径: {resolved}")
        return resolved
    if start_idx == 0:
        return base_ckpt
    previous_stage_name = stage_specs[start_idx - 1]["stage_name"]
    previous_stage_dir = pipeline_root / previous_stage_name / previous_stage_name
    return find_resume_checkpoint(previous_stage_dir, allow_missing=dry_run)


def write_pipeline_status(
    pipeline_root: Path,
    pipeline_name: str,
    executed_stages: list[str],
    skipped_stages: list[str],
    base_ckpt: Path,
    current_resume: Path | None,
    stage_output_dirs: dict[str, str],
    start_stage: str | None,
    stop_after_stage: str | None,
    reference_old5: float,
) -> Path:
    payload = {
        "pipeline": pipeline_name,
        "pipeline_root": str(pipeline_root),
        "executed_stages": executed_stages,
        "skipped_stages": skipped_stages,
        "base_ckpt": str(base_ckpt),
        "current_resume": str(current_resume) if current_resume is not None else None,
        "stage_output_dirs": stage_output_dirs,
        "start_stage": start_stage,
        "stop_after_stage": stop_after_stage,
        "reference_old5": float(reference_old5),
    }
    status_path = pipeline_root / "pipeline_status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status_path


def print_stage_completion(stage_name: str, stage_run_dir: Path) -> None:
    best_path = stage_run_dir / "best.pt"
    last_path = stage_run_dir / "last.pt"
    metrics_path = stage_run_dir / "metrics.json"
    print(f"  stage_name        : {stage_name}")
    print(f"  stage_run_dir     : {stage_run_dir}")
    print(f"  best.pt exists    : {best_path.exists()}")
    print(f"  last.pt exists    : {last_path.exists()}")
    print(f"  metrics.json path : {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run URF staged pipeline with one train script")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_DEFINITIONS.keys()),
        required=True,
    )
    parser.add_argument("--base_ckpt", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline_root", type=Path, default=None)
    parser.add_argument("--start_stage", type=str, default=None)
    parser.add_argument("--stop_after_stage", type=str, default=None)
    parser.add_argument("--list_stages", action="store_true")
    parser.add_argument("--resume_override", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default=AUTO_DEVICE)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--reference_old5", type=float, default=-1.0)
    args = parser.parse_args()

    base_ckpt = args.base_ckpt.resolve()
    output_root = args.output_root.resolve()
    if not base_ckpt.exists():
        raise FileNotFoundError(f"base_ckpt not found: {base_ckpt}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train script not found: {TRAIN_SCRIPT}")

    stage_specs = PIPELINE_DEFINITIONS[args.pipeline]
    stage_names = stage_name_list(stage_specs)
    if args.list_stages:
        print(f"[Pipeline] {args.pipeline}")
        for index, stage_name in enumerate(stage_names, start=1):
            print(f"{index}. {stage_name}")
        return

    pipeline_root = resolve_pipeline_root(output_root, args.pipeline, args.pipeline_root)
    start_idx, stop_idx = resolve_stage_window(stage_specs, args.start_stage, args.stop_after_stage)
    selected_stage_specs = stage_specs[start_idx : stop_idx + 1]
    executed_stages = stage_name_list(selected_stage_specs)
    skipped_stages = [name for name in stage_names if name not in executed_stages]
    initial_resume = resolve_initial_resume(
        stage_specs=stage_specs,
        pipeline_root=pipeline_root,
        base_ckpt=base_ckpt,
        start_idx=start_idx,
        resume_override=args.resume_override,
        dry_run=args.dry_run,
    )
    stage_output_dirs = {
        spec["stage_name"]: str(pipeline_root / spec["stage_name"])
        for spec in stage_specs
    }

    print(f"[Pipeline] {args.pipeline}")
    print(f"[BaseCkpt] {base_ckpt}")
    print(f"[PipelineRoot] {pipeline_root}")
    print(f"[ExecutedStages] {executed_stages}")
    print(f"[SkippedStages] {skipped_stages}")
    print(f"[InitialResume] {initial_resume}")

    current_resume = initial_resume
    write_pipeline_status(
        pipeline_root=pipeline_root,
        pipeline_name=args.pipeline,
        executed_stages=executed_stages,
        skipped_stages=skipped_stages,
        base_ckpt=base_ckpt,
        current_resume=current_resume,
        stage_output_dirs=stage_output_dirs,
        start_stage=args.start_stage,
        stop_after_stage=args.stop_after_stage,
        reference_old5=args.reference_old5,
    )

    pipeline_label = "A" if args.pipeline == "unfreeze_then_refine" else "B"
    for offset, stage_spec in enumerate(selected_stage_specs, start=1):
        stage_name = stage_spec["stage_name"]
        stage_output_dir = pipeline_root / stage_name
        stage_run_dir = resolve_stage_run_dir(stage_output_dir, stage_name)
        command = build_stage_command(
            python_exe=args.python,
            pipeline_label=pipeline_label,
            stage_spec=stage_spec,
            resume_path=current_resume,
            stage_output_dir=stage_output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            reference_old5=args.reference_old5,
        )

        print(f"[Stage {start_idx + offset}] {stage_name}")
        print(f"  preset      : {stage_spec['preset']}")
        print(f"  resume_from : {current_resume}")
        print(f"  output_dir  : {stage_output_dir}")
        print(f"  command     : {format_command(command)}")

        if args.dry_run:
            current_resume = stage_run_dir / "best.pt"
            print_stage_completion(stage_name, stage_run_dir)
            write_pipeline_status(
                pipeline_root=pipeline_root,
                pipeline_name=args.pipeline,
                executed_stages=executed_stages,
                skipped_stages=skipped_stages,
                base_ckpt=base_ckpt,
                current_resume=current_resume,
                stage_output_dirs=stage_output_dirs,
                start_stage=args.start_stage,
                stop_after_stage=args.stop_after_stage,
                reference_old5=args.reference_old5,
            )
            continue

        subprocess.run(command, check=True)
        current_resume = find_resume_checkpoint(stage_run_dir, allow_missing=False)
        print_stage_completion(stage_name, stage_run_dir)
        print(f"  next_resume      : {current_resume}")
        write_pipeline_status(
            pipeline_root=pipeline_root,
            pipeline_name=args.pipeline,
            executed_stages=executed_stages,
            skipped_stages=skipped_stages,
            base_ckpt=base_ckpt,
            current_resume=current_resume,
            stage_output_dirs=stage_output_dirs,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            reference_old5=args.reference_old5,
        )


if __name__ == "__main__":
    main()
