# 文件位置: PythonProject2/inference.py
import os
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(__file__).resolve().parent / ".hf_cache" / "hub"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BertTokenizer

from config_hanxue import (
    CLASSES,
    BLACKHOT_MEAN_THRESH,
    BLACKHOT_SKEW_THRESH,
    BLUR_LOW,
    BLUR_MID,
    IMAGE_ROOT as LOCAL_IMAGE_ROOT,
    NOISE_HIGH,
    NOISE_MED,
    PROJECT,
    PSEUDO_COLOR_SAT_THRESH,
    RUN_NAME,
    STD_LOW,
    STD_MID,
    TOKENIZER_DIR as LOCAL_TOKENIZER_DIR,
    VAL_THRESHOLDS,
)
from models.custom_sam_model import CustomSAMWorldModel
from utils.format_utils import encode_mask_to_rle, empty_mask_rle


# =========================
# 官方固定路径，提交时不要改
# =========================
DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"

DEFAULT_TOKENIZER_PATH = "weights/chinese_clip"

# EfficientSAM ViT-T 通常要求 1024 输入
DEFAULT_INPUT_SIZE = 1024
LOCAL_DEFAULT_TASKS = Path(__file__).resolve().parents[2] / "test" / "json" / "val_tasks1.json"
LOCAL_DEFAULT_EVAL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "test" / "inference_eval" / RUN_NAME


def resolve_path(path):
    if path is None:
        return None

    path = Path(path)

    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)

    file_dir_path = Path(__file__).resolve().parent / path
    return str(file_dir_path)


def load_tokenizer(tokenizer_path):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as e:
        print(f"[Tokenizer 警告] AutoTokenizer 加载失败，回退 BertTokenizer。错误: {repr(e)}")
        return BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def load_tasks(tasks_json):
    with open(tasks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["tasks", "annotations", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        f"任务文件格式不正确，应为 list，或包含 tasks/annotations/data 的 dict。当前类型: {type(data)}"
    )


def get_prompt_from_task(task):
    if "text_prompt" in task:
        return str(task["text_prompt"])

    if "prompt" in task:
        return str(task["prompt"])

    if "text" in task:
        return str(task["text"])

    raise KeyError(
        f"任务中找不到文本提示字段。需要 text_prompt，当前 keys: {list(task.keys())}"
    )


def get_image_path(images_root, image_rel_path):
    """
    兼容:
        image_path = data1/xxx.jpg
        image_path = test/data1/xxx.jpg
    """
    image_rel_path = str(image_rel_path).replace("\\", "/")

    if os.path.isabs(image_rel_path):
        return image_rel_path

    candidates = [
        os.path.join(images_root, image_rel_path),
    ]

    if image_rel_path.startswith("test/"):
        candidates.append(os.path.join(images_root, image_rel_path[len("test/"):]))

    if not image_rel_path.startswith("test/"):
        candidates.append(os.path.join(images_root, "test", image_rel_path))

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]


def compute_skewness(gray):
    gray_f = gray.astype(np.float32)
    mean = float(gray_f.mean())
    std = float(gray_f.std())

    if std < 1e-6:
        return 0.0

    return float(np.mean(((gray_f - mean) / std) ** 3))


def should_invert_ir(gray, filename):
    name = str(filename).lower()
    mean_val = float(gray.mean())
    skew = compute_skewness(gray)

    if "blackhot" in name or "black_hot" in name or "black-hot" in name:
        return True

    if skew < -0.3:
        return True

    if mean_val > 200 and "vis" not in name:
        return True

    return False


def apply_clahe_to_gray(gray):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)


def is_pseudo_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def letterbox_gray(gray, input_size=1024, fill_value=0):
    orig_h, orig_w = gray.shape[:2]

    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_size, input_size), fill_value, dtype=resized.dtype)

    pad_left = (input_size - new_w) // 2
    pad_top = (input_size - new_h) // 2

    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = {
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "input_size": int(input_size),
        "scale": float(scale),
    }

    return padded, meta


def preprocess_image(image_path, input_size=1024, use_clahe=False):
    """
    推理预处理必须和训练保持一致：
    灰度读取 → 极性统一 → 动态 CLAHE → 降噪 → 锐化 → letterbox padding → GRAY2RGB → 0~1。
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {image_path}")
    elif is_pseudo_color(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(image_path).replace("\\", "/")
    if "blackHot" in fname:
        gray = 255 - gray
    else:
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if std > 0:
            skew = float(np.mean(((gray - mean) / std) ** 3))
            if skew < BLACKHOT_SKEW_THRESH:
                gray = 255 - gray
        if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            gray = 255 - gray

    std = float(np.std(gray))
    noise_sigma = estimate_noise_sigma(gray)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if use_clahe:
        if std < STD_LOW:
            gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        elif std < STD_MID:
            gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(gray)

    if noise_sigma > NOISE_HIGH:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > NOISE_MED:
        gray = cv2.medianBlur(gray, 3)

    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = np.clip(cv2.filter2D(gray, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < BLUR_MID:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = np.clip(cv2.addWeighted(gray, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)

    gray_pad, meta = letterbox_gray(gray, input_size=input_size, fill_value=0)

    rgb = np.stack([gray_pad, gray_pad, gray_pad], axis=-1)
    img_np = rgb.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))

    img_tensor = torch.from_numpy(img_np).float()

    return img_tensor, meta


def unletterbox_logits_to_original(pred_logit, meta):
    """
    将 padded 预测结果去 padding，再 resize 回原图尺寸。
    """
    if pred_logit.ndim != 2:
        raise ValueError(f"pred_logit 应为二维张量，当前形状: {pred_logit.shape}")

    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    cropped = pred_logit[
        pad_top:pad_top + new_h,
        pad_left:pad_left + new_w
    ]

    cropped = cropped[None, None]

    restored = F.interpolate(
        cropped,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    return restored


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "module"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_like = [torch.is_tensor(v) for v in checkpoint.values()]
            if len(tensor_like) > 0 and any(tensor_like):
                return checkpoint

    raise RuntimeError(
        "无法从 checkpoint 中提取 state_dict。"
        "建议保存方式为 torch.save(model.state_dict(), '/raytron/code/model/sam3.pt')"
    )


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def load_model(checkpoint_path, device):
    model = CustomSAMWorldModel()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        print("[警告] 加载模型时存在 missing_keys:")
        for k in missing_keys[:30]:
            print("  MISSING:", k)
        if len(missing_keys) > 30:
            print(f"  ... 还有 {len(missing_keys) - 30} 个")

    if len(unexpected_keys) > 0:
        print("[警告] 加载模型时存在 unexpected_keys:")
        for k in unexpected_keys[:30]:
            print("  UNEXPECTED:", k)
        if len(unexpected_keys) > 30:
            print(f"  ... 还有 {len(unexpected_keys) - 30} 个")

    model.to(device)
    model.eval()

    return model


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total_params), int(trainable_params)


def predict_one_task(
    model,
    tokenizer,
    task,
    images_root,
    device,
    input_size=1024,
    threshold=0.0,
    use_clahe=False
):
    ann_id = int(task["ann_id"])
    image_rel_path = task["image_path"]
    prompt_text = get_prompt_from_task(task)

    image_path = get_image_path(images_root, image_rel_path)

    img_tensor, meta = preprocess_image(
        image_path=image_path,
        input_size=input_size,
        use_clahe=use_clahe
    )

    encoded_text = tokenizer(
        prompt_text,
        padding="max_length",
        truncation=True,
        max_length=15,
        return_tensors="pt"
    )

    input_ids = encoded_text["input_ids"].to(device)
    attention_mask = encoded_text["attention_mask"].to(device)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logit = model(
            img_tensor,
            input_ids,
            attention_mask,
            target_size=(input_size, input_size)
        )

        if pred_logit.ndim == 4:
            pred_logit = pred_logit[0, 0]
        elif pred_logit.ndim == 3:
            pred_logit = pred_logit[0]
        elif pred_logit.ndim != 2:
            raise ValueError(f"模型输出形状异常: {pred_logit.shape}")

        pred_logit = unletterbox_logits_to_original(pred_logit, meta)

        prob = torch.sigmoid(pred_logit).detach().cpu().numpy()
        binary_mask = (prob > threshold).astype(np.uint8)

    rle = encode_mask_to_rle(binary_mask)

    return {
        "ann_id": ann_id,
        "rle": rle
    }


@torch.inference_mode()
def infer_image_prompt(
    model,
    tokenizer,
    image_path,
    prompt_text,
    device,
    threshold,
    input_size,
    use_clahe,
):
    img_tensor, meta = preprocess_image(
        image_path=image_path,
        input_size=input_size,
        use_clahe=use_clahe,
    )
    encoded_text = tokenizer(
        prompt_text,
        padding="max_length",
        truncation=True,
        max_length=15,
        return_tensors="pt",
    )

    input_ids = encoded_text["input_ids"].to(device)
    attention_mask = encoded_text["attention_mask"].to(device)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    logits = model(img_tensor, input_ids, attention_mask, target_size=(input_size, input_size))
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]
    logits = unletterbox_logits_to_original(logits, meta)
    prob = torch.sigmoid(logits).detach().cpu()
    binary_mask = (prob.numpy() > threshold).astype("uint8")
    score = float(prob.max().item())
    return binary_mask, score


def run_inference(
    test_tasks_json=DEFAULT_TASKS,
    images_root=DEFAULT_IMAGE_ROOT,
    output_json_path=DEFAULT_OUTPUT_PATH,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    input_size=DEFAULT_INPUT_SIZE,
    threshold=0.0,
    use_clahe=False,
    fail_safe=True
):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tasks_json = resolve_path(test_tasks_json)
    images_root = resolve_path(images_root)
    output_json_path = resolve_path(output_json_path)
    checkpoint_path = resolve_path(checkpoint_path)
    tokenizer_path = resolve_path(tokenizer_path)

    print("========== Inference Config ==========")
    print("device:", device)
    print("tasks:", test_tasks_json)
    print("images_root:", images_root)
    print("output:", output_json_path)
    print("checkpoint:", checkpoint_path)
    print("tokenizer:", tokenizer_path)
    print("input_size:", input_size)
    print("threshold:", threshold)
    print("use_clahe:", use_clahe)
    print("======================================")

    if not os.path.exists(test_tasks_json):
        raise FileNotFoundError(f"找不到测试任务文件: {test_tasks_json}")

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"找不到图片根目录: {images_root}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"找不到 tokenizer 路径: {tokenizer_path}")

    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(checkpoint_path, device)

    total_params, trainable_params = count_params(model)

    tasks = load_tasks(test_tasks_json)
    predictions = []
    errors = []
    processed_images = set()

    for idx, task in enumerate(tasks):
        ann_id = task.get("ann_id", None)

        try:
            pred = predict_one_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                images_root=images_root,
                device=device,
                input_size=input_size,
                threshold=threshold,
                use_clahe=use_clahe
            )

            predictions.append(pred)

            if "image_path" in task:
                processed_images.add(str(task["image_path"]))

        except Exception as e:
            err_msg = f"任务 idx={idx}, ann_id={ann_id} 推理失败: {repr(e)}"
            print("[错误]", err_msg)
            errors.append(err_msg)

            if not fail_safe:
                raise

            try:
                image_path = get_image_path(images_root, task["image_path"])
                with Image.open(image_path) as img:
                    w, h = img.size

                predictions.append({
                    "ann_id": int(ann_id),
                    "rle": empty_mask_rle(h, w)
                })

                if "image_path" in task:
                    processed_images.add(str(task["image_path"]))

            except Exception as e2:
                raise RuntimeError(
                    f"任务失败后生成空 mask 也失败。原错误: {repr(e)}；空 mask 错误: {repr(e2)}"
                )

    elapsed = time.time() - start_time

    output = {
        "model_info": {
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "checkpoint_path": checkpoint_path
        },
        "timing": {
            "inference_seconds": float(elapsed),
            "avg_inference_seconds_per_task": float(elapsed / max(len(tasks), 1)),
            "processed_images": int(len(processed_images)),
            "total_tasks": int(len(tasks))
        },
        "predictions": predictions
    }

    if len(errors) > 0:
        output["errors"] = errors

    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"推理完成，共 {len(tasks)} 个任务，输出 {len(predictions)} 条预测。")
    print(f"结果已保存到: {output_json_path}")

    if len(predictions) != len(tasks):
        raise RuntimeError(
            f"输出数量和任务数量不一致: predictions={len(predictions)}, tasks={len(tasks)}"
        )

    return output


def run_eval(
    tasks_path=LOCAL_DEFAULT_TASKS,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    output_dir=LOCAL_DEFAULT_EVAL_OUTPUT_DIR,
    image_root=LOCAL_IMAGE_ROOT,
    tokenizer_path=LOCAL_TOKENIZER_DIR,
    input_size=DEFAULT_INPUT_SIZE,
    use_clahe=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_path = Path(resolve_path(tasks_path))
    checkpoint_path = Path(resolve_path(checkpoint_path))
    output_dir = Path(resolve_path(output_dir))
    image_root = Path(resolve_path(image_root))
    tokenizer_path = Path(resolve_path(tokenizer_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(str(tasks_path))
    tokenizer = load_tokenizer(str(tokenizer_path))
    model = load_model(str(checkpoint_path), device)

    from collections import defaultdict

    tasks_by_image = defaultdict(list)
    for task in tasks:
        tasks_by_image[str(task["image_path"]).replace("\\", "/")].append(task)

    predictions = []
    prompt_predictions = []
    t0 = time.time()

    for image_rel_path, image_tasks in tasks_by_image.items():
        image_abs_path = get_image_path(str(image_root), image_rel_path)
        prompt_map = {}
        prompt_entry = {"image_path": image_rel_path, "prompts": {}}
        height = width = None

        for task in image_tasks:
            prompt = get_prompt_from_task(task).strip()
            if prompt not in prompt_map:
                threshold = VAL_THRESHOLDS.get(prompt, 0.5)
                mask, score = infer_image_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=image_abs_path,
                    prompt_text=prompt,
                    device=device,
                    threshold=threshold,
                    input_size=input_size,
                    use_clahe=use_clahe,
                )
                if height is None or width is None:
                    height, width = mask.shape[:2]
                if mask.sum() > 0:
                    prompt_map[prompt] = {
                        "hit": True,
                        "score": round(score, 4),
                        "instances": 1,
                        "rle": encode_mask_to_rle(mask),
                    }
                else:
                    prompt_map[prompt] = {"hit": False}

            pred = prompt_map[prompt]
            if pred.get("hit"):
                rle = pred["rle"]
            else:
                if height is None or width is None:
                    with Image.open(image_abs_path) as img:
                        width, height = img.size
                rle = empty_mask_rle(height, width)
            predictions.append({"ann_id": int(task["ann_id"]), "rle": rle})
            prompt_entry["prompts"][prompt] = pred

        prompt_predictions.append(prompt_entry)

    elapsed = time.time() - t0
    predictions_path = output_dir / f"{tasks_path.stem}_predictions.json"
    prompt_path = output_dir / f"pred_{tasks_path.stem}_hanxue.json"
    meta_path = output_dir / f"{tasks_path.stem}_meta.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_info": {
                    "checkpoint": str(checkpoint_path),
                    "run_name": RUN_NAME,
                    "classes": CLASSES,
                    "prompt_thresholds": VAL_THRESHOLDS,
                },
                "timing": {
                    "inference_seconds": float(elapsed),
                    "avg_inference_seconds_per_image": float(elapsed / max(len(tasks_by_image), 1)),
                    "processed_images": len(tasks_by_image),
                    "total_tasks": len(tasks),
                },
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_predictions, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": str(tasks_path),
                "checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
                "processed_images": len(tasks_by_image),
                "total_tasks": len(tasks),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"完成: images={len(tasks_by_image)}, tasks={len(tasks)}")
    print(f"predictions: {predictions_path}")
    print(f"prompt_json:  {prompt_path}")
    print(f"meta:         {meta_path}")

    return {
        "predictions_path": str(predictions_path),
        "prompt_path": str(prompt_path),
        "meta_path": str(meta_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["submit", "eval"], default="submit")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-dir", default=str(LOCAL_DEFAULT_EVAL_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--threshold", type=float, default=0.0)

    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--no-fail-safe", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks_arg = args.tasks or (LOCAL_DEFAULT_TASKS if args.mode == "eval" else DEFAULT_TASKS)

    if args.mode == "eval":
        run_eval(
            tasks_path=tasks_arg,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            image_root=args.image_root,
            tokenizer_path=args.tokenizer,
            input_size=args.input_size,
            use_clahe=True,
        )
    else:
        run_inference(
            test_tasks_json=tasks_arg,
            images_root=args.image_root,
            output_json_path=args.output,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            input_size=args.input_size,
            threshold=args.threshold,
            use_clahe=args.use_clahe,
            fail_safe=not args.no_fail_safe
        )
