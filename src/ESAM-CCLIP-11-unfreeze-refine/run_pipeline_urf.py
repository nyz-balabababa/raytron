#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_esam_cclip_11_urf.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parents[1] / "test" / "train_output"

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


def make_unique_pipeline_root(output_root: Path, pipeline_name: str, dry_run: bool) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for suffix_idx in range(100):
        suffix = f"_{timestamp}" if suffix_idx == 0 else f"_{timestamp}_{suffix_idx:02d}"
        candidate = output_root / f"urf_{pipeline_name}{suffix}"
        if dry_run:
            return candidate
        if candidate.exists():
            continue
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    raise RuntimeError(f"无法创建唯一 pipeline 输出目录: {output_root}")


def build_stage_command(
    python_exe: str,
    stage_spec: dict,
    resume_path: Path,
    stage_output_dir: Path,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
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
    ]
    if device:
        command.extend(["--device", device])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        command.extend(["--num_workers", str(num_workers)])
    command.extend(stage_spec.get("extra_args", []))
    return command


def resolve_stage_run_dir(stage_output_dir: Path, stage_name: str) -> Path:
    return stage_output_dir / stage_name


def resolve_next_resume_checkpoint(stage_run_dir: Path) -> Path:
    best_path = stage_run_dir / "best.pt"
    last_path = stage_run_dir / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        print(f"[WARN] best.pt 不存在，fallback 到 last.pt: {last_path}")
        return last_path
    raise FileNotFoundError(f"stage 输出缺少 best.pt / last.pt: {stage_run_dir}")


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run URF staged pipeline with one train script")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_DEFINITIONS.keys()),
        required=True,
    )
    parser.add_argument("--base_ckpt", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    base_ckpt = args.base_ckpt.resolve()
    output_root = args.output_root.resolve()
    if not base_ckpt.exists():
        raise FileNotFoundError(f"base_ckpt not found: {base_ckpt}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train script not found: {TRAIN_SCRIPT}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    pipeline_root = make_unique_pipeline_root(output_root, args.pipeline, args.dry_run)
    stage_specs = PIPELINE_DEFINITIONS[args.pipeline]

    print(f"[Pipeline] {args.pipeline}")
    print(f"[BaseCkpt] {base_ckpt}")
    print(f"[PipelineRoot] {pipeline_root}")

    current_resume = base_ckpt
    for index, stage_spec in enumerate(stage_specs, start=1):
        stage_name = stage_spec["stage_name"]
        stage_output_dir = pipeline_root / stage_name
        stage_run_dir = resolve_stage_run_dir(stage_output_dir, stage_name)
        command = build_stage_command(
            python_exe=args.python,
            stage_spec=stage_spec,
            resume_path=current_resume,
            stage_output_dir=stage_output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        print(f"[Stage {index}] {stage_name}")
        print(f"  preset      : {stage_spec['preset']}")
        print(f"  resume_from : {current_resume}")
        print(f"  output_dir  : {stage_output_dir}")
        print(f"  command     : {format_command(command)}")

        if args.dry_run:
            current_resume = stage_run_dir / "best.pt"
            continue

        subprocess.run(command, check=True)
        current_resume = resolve_next_resume_checkpoint(stage_run_dir)
        print(f"  next_resume : {current_resume}")


if __name__ == "__main__":
    main()
