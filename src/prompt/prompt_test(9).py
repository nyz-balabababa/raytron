#!/usr/bin/env python3
"""
Prompt 测试与评估脚本（含自适应预处理）
- 自动发现 test/ 下所有任务 JSON，逐个推理
- 推理前先对图片执行自适应预处理（不做 resize，仅做反色/CLAHE/降噪/锐化）
- 输出：pred JSON + 统计 CSV + 可视化 + 预处理日志
"""
import csv
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))  # 让 src/ 下的脚本能 import 项目根目录的 inference

import inference as inference_module

inference_module.cv2 = cv2
load_model = inference_module.load_model
do_inference = inference_module.do_inference
DEVICE = inference_module.DEVICE

# ── 配置（按需修改） ─────────────────────────────────────────────────

IMAGE_ROOT = ROOT                 # 图片根目录，与 JSON 中 image_path 拼接
CHECKPOINT = ROOT / "model" / "sam3.pt"
MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"
OUTPUT_ROOT = ROOT / "test" / "prompt_test_output"
RAW_STATS_CONF_THRESHOLD = 0.01   # 过滤前统计用，尽量保留 teacher 原始响应
DEFAULT_OUTPUT_CONF_THRESHOLD = 0.60  # 未单独配置的 prompt 默认输出阈值
OUTPUT_CONF_THRESHOLDS = {
    "person": 0.70,
    "computer": 0.60,
    "trash can": 0.65,
    "cable connector": 0.60,
    "fire hydrant": 0.70,
    "electrical panel": 0.65,
    "chair": 0.65,
    "cable": 0.70,
    "pipe": 0.75,
    "sensor": 0.75,
}
MAX_VIS_IMAGES = 30  # 每个JSON最多可视化多少张图（按命中 prompt 数量排序）
REQUIRE_CUDA = True             # True=检测不到 CUDA 直接退出，避免误用 CPU 跑几十小时
TASK_JSON_WHITELIST = [
    "train(person).json",
    "val(person).json",
]  # 只跑当前 data1/person 伪标签任务
AUTO_SELECT_WHITELIST = True    # True=命中过滤名单后直接运行，不再交互选择

# 预处理开关
ENABLE_PREPROCESS = True          # True=推理前先自适应预处理
KEEP_PREPROCESSED = False         # True=保留预处理图片到 PREPROCESS_DIR
PREPROCESS_DIR = ROOT / "test" / "preprocessed_inference"

# 预处理参数
# 当前版本不做尺寸归一化，保留原始分辨率送入 SAM3 以保护小目标。
STD_LOW, STD_MID = 35, 50
NOISE_HIGH, NOISE_MED = 12, 8
BLUR_LOW, BLUR_MID = 150, 300
INVERT_KEYWORDS = ["blackHot"]  # 文件名辅助检测（优先级低于内容检测）
PSEUDO_COLOR_SAT_THRESH = 60  # HSV 饱和度均值 > 此值视为伪彩色图，转为灰度
BLACKHOT_SKEW_THRESH = -0.3   # 直方图偏度 < 此值判定为黑热（左偏 → 需要反色）


# ── inference.py 接口适配 ───────────────────────────────────────────

def load_prompt_model():
    """兼容官方 SAM3 与本地 CLIPSeg 两种 inference.py 接口。"""
    sig = inspect.signature(load_model)
    params = list(sig.parameters.keys())
    if params == ["checkpoint_path"]:
        return load_model(str(CHECKPOINT))
    if "checkpoint_path" in sig.parameters:
        return load_model(str(MODEL_DIR), checkpoint_path=str(CHECKPOINT))
    return load_model(str(CHECKPOINT))


def run_prompt_inference(infer_path: str, prompts: list, model, processor, conf_threshold: float):
    """兼容 do_inference 是否需要 model 参数的两种签名。"""
    sig = inspect.signature(do_inference)
    kwargs = {
        "image_path": infer_path,
        "text_prompts": prompts,
        "processor": processor,
        "conf_threshold": conf_threshold,
    }
    if "model" in sig.parameters:
        kwargs["model"] = model
    return do_inference(**kwargs)


def get_output_threshold(prompt: str) -> float:
    return OUTPUT_CONF_THRESHOLDS.get(prompt, DEFAULT_OUTPUT_CONF_THRESHOLD)


def group_prompts_by_threshold(prompts: list[str]) -> dict[float, list[str]]:
    grouped = defaultdict(list)
    for prompt in prompts:
        grouped[get_output_threshold(prompt)].append(prompt)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def log_runtime_device():
    """打印并校验实际运行时设备，防止导入错环境或 CPU 误跑。"""
    cuda_available = torch.cuda.is_available()
    print("  运行时检查:")
    print(f"    inference.py: {Path(inference_module.__file__).resolve()}")
    print(f"    torch:        {torch.__version__}")
    print(f"    torch cuda:   {cuda_available}")
    if cuda_available:
        print(f"    GPU:          {torch.cuda.get_device_name(0)}")
        print(f"    CUDA runtime: {torch.version.cuda}")
    print(f"    inference.DEVICE: {getattr(inference_module, 'DEVICE', 'N/A')}")

    if REQUIRE_CUDA and (not cuda_available or getattr(inference_module, "DEVICE", "cpu") != "cuda"):
        print()
        print("错误: 当前没有使用 CUDA。已停止，避免在 CPU 上长时间误跑。")
        print("请在 4090 机器上检查:")
        print('  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'CPU\')"')
        print('  python -c "import inference; print(inference.__file__); print(inference.DEVICE)"')
        sys.exit(1)


# ── 预处理函数 ──────────────────────────────────────────────────────

def is_pseudo_color(img_bgr: np.ndarray) -> bool:
    """检测是否为伪彩色图（HSV 饱和度均值超阈值）。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def convert_pseudo_color_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """将伪彩色图转为标准灰度图（加权亮度通道）。"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def should_invert(gray: np.ndarray, fname: str) -> bool:
    """
    判断是否需要反色。优先级：
    1. 文件名含 blackHot → 强制反色
    2. 直方图显著左偏（skew < -0.3）→ 内容判定为黑热，反色
    3. 均值 > 200 且非 vis → 兜底反色
    """
    if "blackHot" in fname:
        return True
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std > 0:
        skew = float(np.mean(((gray - mean) / std) ** 3))
        if skew < BLACKHOT_SKEW_THRESH:
            return True
    if mean > 200 and "vis" not in fname.lower():
        return True
    return False


def estimate_noise_sigma(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def preprocess_image(abs_path: str, preproc_dir: Optional[Path]) -> str:
    """
    对图片执行自适应预处理，返回用于推理的图片路径。
    若 KEEP_PREPROCESSED 且 preproc_dir 不为空，预处理图存到 preproc_dir，
    后续同一张图直接从缓存读取。
    """
    # 缓存路径
    if KEEP_PREPROCESSED and preproc_dir is not None:
        rel = os.path.relpath(abs_path, str(IMAGE_ROOT)).replace("\\", "/")
        cached = preproc_dir / rel
        if cached.exists():
            return str(cached)
    else:
        cached = None

    # 先读 BGR 判断是否为伪彩色图
    img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return abs_path  # 回退到原图
    elif is_pseudo_color(img_bgr):
        img = convert_pseudo_color_to_gray(img_bgr)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 反色统一（基于直方图偏度 + 文件名辅助）
    if should_invert(img, os.path.basename(abs_path)):
        img = 255 - img

    # 尺寸统一：已移除 resize，原始尺寸送入 SAM3 推理以保护小目标

    # 质量评估（增强前）
    std = float(np.std(img))
    noise_sigma = estimate_noise_sigma(img)
    blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())

    # CLAHE
    if std < STD_LOW:
        img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
    elif std < STD_MID:
        img = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(img)

    # 降噪
    if noise_sigma > NOISE_HIGH:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > NOISE_MED:
        img = cv2.medianBlur(img, 3)

    # 锐化
    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < BLUR_MID:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = np.clip(cv2.addWeighted(img, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)

    # 保存
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cached), img)
        return str(cached)

    # 临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        return tmp.name


# ── 可视化 ──────────────────────────────────────────────────────────

def rle_to_mask(rle: dict) -> np.ndarray:
    try:
        from pycocotools import mask as maskUtils
        if isinstance(rle["counts"], str):
            rle_cp = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        else:
            rle_cp = dict(rle)
        return maskUtils.decode(rle_cp).astype(np.uint8)
    except ImportError:
        pass
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = list(map(int, counts.split(",")))
    elif isinstance(counts, bytes):
        counts = list(map(int, counts.decode("utf-8").split(",")))
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def make_overlay(image: Image.Image, mask: np.ndarray, color: tuple) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mh, mw = mask.shape
    if (mw, mh) != image.size:
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            image.size, Image.NEAREST)) > 128
    for y, x in zip(*np.where(mask)):
        draw.point((x, y), fill=color)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def save_vis(results: list, vis_dir: Path, max_images: int):
    """对激活 prompt 最多的 top-N 图片生成叠加图。"""
    vis_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100),
        (255, 255, 0, 100), (255, 0, 255, 100), (0, 255, 255, 100),
    ]

    # 按命中 prompt 数量排序取 top
    scored = []
    for img_path, res in results:
        hit = sum(1 for r in res.values() if r.get("score", 0) > 0)
        scored.append((hit, img_path, res))
    scored.sort(key=lambda x: -x[0])

    for rank, (hit, img_path, res) in enumerate(scored[:max_images]):
        if hit == 0:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        combined = img.convert("RGBA")
        for idx, (prompt, pred) in enumerate(res.items()):
            rle = pred.get("rle")
            if rle is None:
                continue
            mask = rle_to_mask(rle)
            if mask.sum() == 0:
                continue
            color = colors[idx % len(colors)]
            combined = make_overlay(combined if idx > 0 else img, mask, color)
        name = Path(img_path).stem
        combined.save(vis_dir / f"{rank:03d}_{name}_hits{hit}.png")


# ── 按任务 JSON 推理 ────────────────────────────────────────────────

def load_task_json(json_path: Path) -> tuple[list, list]:
    """读取任务 JSON，返回 (image_paths, unique_prompts)。"""
    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    image_paths = list(dict.fromkeys(
        t["image_path"].replace("\\", "/") for t in tasks
    ))
    unique_prompts = list(dict.fromkeys(
        t["text_prompt"].strip() for t in tasks if t.get("text_prompt", "").strip()
    ))
    return image_paths, unique_prompts


@torch.inference_mode()
def run_json(json_path: Path, image_root: Path):
    """对单个任务 JSON 跑推理 + 统计 + 可视化。"""
    json_name = json_path.stem
    out_dir = OUTPUT_ROOT / json_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths, prompts = load_task_json(json_path)
    print(f"  任务 JSON:   {json_path.name}")
    print(f"  图片数:      {len(image_paths)}")
    print(f"  Prompt 数:   {len(prompts)}")
    print(f"  Prompts:     {prompts}")
    print(f"  原始统计阈值: {RAW_STATS_CONF_THRESHOLD:.2f}")
    print("  输出过滤阈值:")
    for prompt in prompts:
        print(f"    - {prompt}: {get_output_threshold(prompt):.2f}")
    if ENABLE_PREPROCESS:
        print("  预处理策略:  原始分辨率直推 SAM3（不做 resize）+ 反色/CLAHE/降噪/锐化")
    else:
        print("  预处理策略:  关闭，原图直接推理")
    print()

    model, processor = load_prompt_model()
    output_threshold_groups = group_prompts_by_threshold(prompts)

    all_results = []
    raw_prompt_hits = defaultdict(list)
    raw_prompt_misses = defaultdict(int)
    filtered_prompt_hits = defaultdict(list)
    filtered_prompt_misses = defaultdict(int)
    prompt_seen = defaultdict(int)

    t0 = time.time()
    n_processed = 0
    skipped = 0

    pbar = tqdm(image_paths, desc=f"  {json_path.name}", unit="img",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for rel_path in pbar:
        abs_path = str(image_root / rel_path)
        if not os.path.exists(abs_path):
            skipped += 1
            continue

        # 自适应预处理
        infer_path = abs_path
        if ENABLE_PREPROCESS:
            pp_dir = PREPROCESS_DIR / json_name if KEEP_PREPROCESSED else None
            infer_path = preprocess_image(abs_path, pp_dir)

        try:
            raw_results_by_prompt, w, h = run_prompt_inference(
                infer_path, prompts, model, processor, RAW_STATS_CONF_THRESHOLD
            )
            filtered_results_by_prompt = {}
            for conf_threshold, prompt_group in output_threshold_groups.items():
                group_results, _, _ = run_prompt_inference(
                    infer_path, prompt_group, model, processor, conf_threshold
                )
                filtered_results_by_prompt.update(group_results)
        except Exception as e:
            tqdm.write(f"  [!] {rel_path}: {e}")
            if ENABLE_PREPROCESS and not KEEP_PREPROCESSED and infer_path != abs_path:
                try: os.unlink(infer_path)
                except: pass
            skipped += 1
            continue

        if ENABLE_PREPROCESS and not KEEP_PREPROCESSED and infer_path != abs_path:
            try: os.unlink(infer_path)
            except: pass

        for prompt in prompts:
            prompt_seen[prompt] += 1
            raw_pred = raw_results_by_prompt.get(prompt)
            if raw_pred and raw_pred.get("score", 0) > 0:
                raw_prompt_hits[prompt].append({
                    "score": raw_pred["score"],
                    "instances": raw_pred.get("instance_count", 0),
                    "image": rel_path,
                })
            else:
                raw_prompt_misses[prompt] += 1

            filtered_pred = filtered_results_by_prompt.get(prompt)
            if filtered_pred and filtered_pred.get("score", 0) > 0:
                filtered_prompt_hits[prompt].append({
                    "score": filtered_pred["score"],
                    "instances": filtered_pred.get("instance_count", 0),
                    "image": rel_path,
                })
            else:
                filtered_prompt_misses[prompt] += 1

        all_results.append((abs_path, filtered_results_by_prompt))
        n_processed += 1

    elapsed = time.time() - t0
    print(f"\n  推理完成: {n_processed} 张, 跳过 {skipped}, {elapsed:.0f}s ({elapsed/max(n_processed,1)*1000:.0f}ms/张)")

    # ── pred JSON ──
    predictions = []
    for abs_path, results_by_prompt in all_results:
        entry = {
            "image_path": str(Path(abs_path).relative_to(image_root)).replace("\\", "/"),
            "prompts": {},
        }
        for prompt in prompts:
            pred = results_by_prompt.get(prompt)
            if pred and pred.get("score", 0) > 0:
                entry["prompts"][prompt] = {
                    "hit": True,
                    "score": round(pred["score"], 4),
                    "instances": pred.get("instance_count", 0),
                    "rle": pred.get("rle"),
                }
            else:
                entry["prompts"][prompt] = {"hit": False}
        predictions.append(entry)

    pred_path = out_dir / f"pred_{json_name}.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  pred JSON: {pred_path}")

    # ── 统计 CSV ──
    stats_path = out_dir / f"stats_{json_name}.csv"
    with open(stats_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt",
            "total_images",
            "raw_hits", "raw_misses", "raw_activation_rate", "raw_avg_score", "raw_avg_instances", "raw_total_instances",
            "filtered_hits", "filtered_misses", "filtered_activation_rate", "filtered_avg_score", "filtered_avg_instances", "filtered_total_instances",
        ])
        for prompt in prompts:
            total_img = prompt_seen[prompt]
            raw_hits = len(raw_prompt_hits.get(prompt, []))
            raw_misses = raw_prompt_misses.get(prompt, 0)
            raw_scores = [h["score"] for h in raw_prompt_hits.get(prompt, [])]
            raw_instances = [h["instances"] for h in raw_prompt_hits.get(prompt, [])]

            filtered_hits = len(filtered_prompt_hits.get(prompt, []))
            filtered_misses = filtered_prompt_misses.get(prompt, 0)
            filtered_scores = [h["score"] for h in filtered_prompt_hits.get(prompt, [])]
            filtered_instances = [h["instances"] for h in filtered_prompt_hits.get(prompt, [])]
            writer.writerow([
                prompt,
                total_img,
                raw_hits,
                raw_misses,
                round(raw_hits / max(total_img, 1), 4),
                round(float(np.mean(raw_scores)), 4) if raw_scores else 0,
                round(float(np.mean(raw_instances)), 2) if raw_instances else 0,
                sum(raw_instances),
                filtered_hits,
                filtered_misses,
                round(filtered_hits / max(total_img, 1), 4),
                round(float(np.mean(filtered_scores)), 4) if filtered_scores else 0,
                round(float(np.mean(filtered_instances)), 2) if filtered_instances else 0,
                sum(filtered_instances),
            ])
    print(f"  stats CSV:  {stats_path}")

    # ── 终端统计 ──
    raw_total_hits = sum(len(v) for v in raw_prompt_hits.values())
    filtered_total_hits = sum(len(v) for v in filtered_prompt_hits.values())
    print(f"\n  命中过滤汇总: raw={raw_total_hits} -> filtered={filtered_total_hits} (drop={raw_total_hits - filtered_total_hits})")
    print(f"\n  {'Prompt':<20} {'raw率':>8} {'flt率':>8} {'raw_score':>10} {'flt_score':>10} {'raw_inst':>9} {'flt_inst':>9}")
    print("  " + "-" * 82)
    for prompt in prompts:
        total_img = prompt_seen[prompt]
        raw_hits = len(raw_prompt_hits.get(prompt, []))
        raw_scores = [h["score"] for h in raw_prompt_hits.get(prompt, [])]
        raw_instances = [h["instances"] for h in raw_prompt_hits.get(prompt, [])]
        filtered_hits = len(filtered_prompt_hits.get(prompt, []))
        filtered_scores = [h["score"] for h in filtered_prompt_hits.get(prompt, [])]
        filtered_instances = [h["instances"] for h in filtered_prompt_hits.get(prompt, [])]

        raw_rate = raw_hits / max(total_img, 1)
        filtered_rate = filtered_hits / max(total_img, 1)
        raw_avg_s = float(np.mean(raw_scores)) if raw_scores else 0
        filtered_avg_s = float(np.mean(filtered_scores)) if filtered_scores else 0
        raw_avg_i = float(np.mean(raw_instances)) if raw_instances else 0
        filtered_avg_i = float(np.mean(filtered_instances)) if filtered_instances else 0

        print(
            f"  {prompt:<20} "
            f"{raw_rate:>7.1%} "
            f"{filtered_rate:>7.1%} "
            f"{raw_avg_s:>10.4f} "
            f"{filtered_avg_s:>10.4f} "
            f"{raw_avg_i:>9.2f} "
            f"{filtered_avg_i:>9.2f}"
        )

    # ── 可视化 ──
    vis_dir = out_dir / "visualizations"
    save_vis(all_results, vis_dir, MAX_VIS_IMAGES)
    print(f"\n  可视化: {vis_dir}")


# ── main ────────────────────────────────────────────────────────────

def main():
    log_runtime_device()

    if not CHECKPOINT.exists():
        print(f"错误: 找不到模型 {CHECKPOINT}")
        sys.exit(1)
    sig = inspect.signature(load_model)
    needs_model_dir = list(sig.parameters.keys()) != ["checkpoint_path"] and "checkpoint_path" in sig.parameters
    if needs_model_dir and not MODEL_DIR.exists():
        print(f"错误: 找不到模型目录 {MODEL_DIR}")
        sys.exit(1)

    # 自动扫描 test/json/ 下所有任务 JSON
    json_dir = ROOT / "test" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(json_dir.glob("*.json"))

    # 过滤掉非任务 JSON
    skip_keywords = ["predictions", "activation_stats", "confidence", "stats", "pred_"]
    all_jsons = [jf for jf in json_files
                 if not any(kw in jf.stem.lower() for kw in skip_keywords)]

    if TASK_JSON_WHITELIST:
        whitelist = set(TASK_JSON_WHITELIST)
        all_jsons = [jf for jf in all_jsons if jf.name in whitelist]

    if not all_jsons:
        print(f"错误: {json_dir} 下没有找到任务 JSON 文件")
        if TASK_JSON_WHITELIST:
            print(f"  当前白名单: {TASK_JSON_WHITELIST}")
        print(f"  需要格式: [{{\"ann_id\":1, \"image_path\":\"...\", \"text_prompt\":\"...\"}}, ...]")
        sys.exit(1)

    if AUTO_SELECT_WHITELIST and TASK_JSON_WHITELIST:
        task_jsons = all_jsons
    else:
        # 交互式选择 JSON
        print(f"\n{'─'*50}")
        print(f"{'#':<4} {'文件名':<40} {'大小':>8}")
        print(f"{'─'*50}")
        for i, jf in enumerate(all_jsons):
            size_kb = jf.stat().st_size / 1024
            print(f"{i:<4} {jf.name:<40} {size_kb:>7.0f} KB")
        print(f"{'─'*50}")
        print(f"a   全部运行")
        print(f"q   退出")
        print(f"{'─'*50}")

        choice = input("选择 (编号/a/q): ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        elif choice.lower() == "a":
            task_jsons = all_jsons
        else:
            idxs = [int(x.strip()) for x in choice.replace(",", " ").split() if x.strip().isdigit()]
            task_jsons = [all_jsons[i] for i in idxs if 0 <= i < len(all_jsons)]
            if not task_jsons:
                print("无效选择")
                sys.exit(1)

    print(f"\n图片根目录:   {IMAGE_ROOT}")
    print(f"预处理:       {'启用' if ENABLE_PREPROCESS else '关闭'}", end="")
    if ENABLE_PREPROCESS:
        print(" (原始分辨率, 不做 resize, CLAHE/降噪/锐化自适应)")
    else:
        print()
    print(f"已选择 {len(task_jsons)} 个 JSON:")
    for jf in task_jsons:
        print(f"  - {jf.name}")
    print()

    for jf in task_jsons:
        print(f"\n{'#' * 60}")
        print(f"处理: {jf.name}")
        print(f"{'#' * 60}")
        run_json(jf, IMAGE_ROOT)


if __name__ == "__main__":
    main()
