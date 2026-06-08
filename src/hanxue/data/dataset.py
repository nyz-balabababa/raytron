import hashlib
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

from config_hanxue import (
    BLACKHOT_MEAN_THRESH,
    BLACKHOT_SKEW_THRESH,
    BLUR_LOW,
    BLUR_MID,
    CONF_FILTER,
    HFLIP_PROB,
    IMAGE_ROOT,
    IMG_SIZE,
    LOSS_WEIGHT_FLOOR,
    NEGATIVE_SAMPLE_WEIGHT,
    NOISE_HIGH,
    NOISE_MED,
    PROMPT_AUGMENTATIONS,
    PROMPT_AUG_PROB,
    PSEUDO_COLOR_SAT_THRESH,
    RARE_OVERSAMPLE,
    STD_LOW,
    STD_MID,
    TOKENIZER_DIR,
)


def rle_to_mask(rle):
    try:
        from pycocotools import mask as mask_utils

        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return mask_utils.decode(rle_copy).astype(np.float32)
    except Exception:
        pass

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]

    if not counts:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos : pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F").astype(np.float32)


def resolve_existing_path(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    candidates = [
        Path.cwd() / p,
        Path(__file__).resolve().parents[1] / p,
        Path(__file__).resolve().parents[3] / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


def load_tokenizer(tokenizer_dir):
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    except Exception:
        return BertTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def is_pseudo_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) > PSEUDO_COLOR_SAT_THRESH


def estimate_noise_sigma(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def load_teacher_aligned_gray(abs_path):
    img_bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图片不存在或无法读取: {abs_path}")
    elif is_pseudo_color(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fname = str(abs_path).replace("\\", "/")
    if "blackHot" in fname:
        img = 255 - img
    else:
        mean = float(np.mean(img))
        std = float(np.std(img))
        if std > 0:
            skew = float(np.mean(((img - mean) / std) ** 3))
            if skew < BLACKHOT_SKEW_THRESH:
                img = 255 - img
        if mean > BLACKHOT_MEAN_THRESH and "vis" not in fname.lower():
            img = 255 - img

    std = float(np.std(img))
    noise_sigma = estimate_noise_sigma(img)
    blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())

    if std < STD_LOW:
        img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
    elif std < STD_MID:
        img = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(img)

    if noise_sigma > NOISE_HIGH:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    elif noise_sigma > NOISE_MED:
        img = cv2.medianBlur(img, 3)

    if blur_score < BLUR_LOW:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)
    elif blur_score < BLUR_MID:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = np.clip(cv2.addWeighted(img, 2.0, blurred, -1.0, 0), 0, 255).astype(np.uint8)

    return img


def letterbox_image(gray, target_hw, fill_value=0):
    target_h, target_w = target_hw
    orig_h, orig_w = gray.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w), fill_value, dtype=resized.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return padded, meta


def letterbox_mask(mask, target_hw, fill_value=0):
    target_h, target_w = target_hw
    orig_h, orig_w = mask.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.full((target_h, target_w), fill_value, dtype=resized.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return padded, meta


class InfraredPromptDataset(Dataset):
    def __init__(
        self,
        images_root=IMAGE_ROOT,
        annotation_json=None,
        split_txt=None,
        tokenizer_dir=TOKENIZER_DIR,
        img_size=(IMG_SIZE, IMG_SIZE),
        augment_prompt=False,
        hflip_prob=HFLIP_PROB,
        use_conf_filter=False,
        class_names=None,
        prompt_thresholds=None,
        loss_weight_floor=LOSS_WEIGHT_FLOOR,
        negative_sample_prob=0.0,
        negative_sample_weight=NEGATIVE_SAMPLE_WEIGHT,
        rare_oversample=None,
        training=True,
        seed=42,
    ):
        if annotation_json is None or split_txt is None:
            raise ValueError("annotation_json 和 split_txt 不能为空。")
        self.images_root = resolve_existing_path(images_root)
        self.annotation_json = resolve_existing_path(annotation_json)
        self.split_txt = resolve_existing_path(split_txt)
        self.tokenizer_dir = resolve_existing_path(tokenizer_dir)
        self.img_size = tuple(img_size)
        self.augment_prompt = augment_prompt
        self.hflip_prob = hflip_prob
        self.use_conf_filter = use_conf_filter
        self.class_names = set(class_names) if class_names else None
        self.prompt_thresholds = prompt_thresholds or {}
        self.loss_weight_floor = loss_weight_floor
        self.negative_sample_prob = negative_sample_prob
        self.negative_sample_weight = negative_sample_weight
        self.rare_oversample = rare_oversample or RARE_OVERSAMPLE
        self.training = training
        self.rng = random.Random(seed)

        self.tokenizer = load_tokenizer(self.tokenizer_dir)
        self.allowed = self._load_allowed_set()
        self.samples = self._build_samples()

    def _load_allowed_set(self):
        with open(self.split_txt, "r", encoding="utf-8") as f:
            return {line.strip().replace("\\", "/") for line in f if line.strip()}

    def _normalize_image_path(self, image_path):
        image_path = str(image_path).replace("\\", "/")
        if image_path in self.allowed:
            return image_path
        alt = image_path[5:] if image_path.startswith("test/") else f"test/{image_path}"
        if alt in self.allowed:
            return alt
        return None

    def _resolve_image_path(self, image_path):
        rel = image_path.replace("\\", "/")
        candidates = [
            self.images_root / rel,
            self.images_root / rel[5:] if rel.startswith("test/") else self.images_root / "test" / rel,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    def _score_threshold(self, prompt):
        prompt = str(prompt)
        thresholds = dict(CONF_FILTER)
        thresholds.update(self.prompt_thresholds)
        return thresholds.get(prompt, 0.60)

    def _build_samples(self):
        with open(self.annotation_json, "r", encoding="utf-8-sig") as f:
            preds = json.load(f)

        base_samples = []
        cache_dir = PROJECT_ROOT / "test" / "train_output" / ".mask_cache_hanxue" / self.annotation_json.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        for item in preds:
            image_path = self._normalize_image_path(item["image_path"])
            if image_path is None:
                continue
            abs_path = self._resolve_image_path(image_path)
            if not abs_path.exists():
                continue

            for prompt, prompt_info in item.get("prompts", {}).items():
                if self.class_names is not None and str(prompt) not in self.class_names:
                    continue
                hit = bool(prompt_info.get("hit"))
                score = float(prompt_info.get("score", 1.0))

                if hit:
                    if self.use_conf_filter and score < self._score_threshold(prompt):
                        continue
                    rle = prompt_info.get("rle")
                    if rle is None:
                        continue
                    cache_name = hashlib.md5(f"{image_path}|{prompt}".encode()).hexdigest()
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() <= 0:
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                    sample_weight = max(score, self.loss_weight_floor)
                    base_samples.append(
                        {
                            "image_path": image_path,
                            "prompt": str(prompt),
                            "mask_path": cache_path,
                            "sample_weight": sample_weight,
                            "is_positive": True,
                        }
                    )
                else:
                    if self.negative_sample_prob > 0 and self.rng.random() < self.negative_sample_prob:
                        base_samples.append(
                            {
                                "image_path": image_path,
                                "prompt": str(prompt),
                                "mask_path": None,
                                "sample_weight": self.negative_sample_weight,
                                "is_positive": False,
                            }
                        )

        if not self.rare_oversample:
            return base_samples

        extra = []
        for sample in base_samples:
            if not sample["is_positive"]:
                continue
            mult = int(self.rare_oversample.get(sample["prompt"], 1))
            for _ in range(max(mult - 1, 0)):
                extra.append(sample.copy())
        return base_samples + extra

    def __len__(self):
        return len(self.samples)

    def _augment(self, prompt):
        opts = PROMPT_AUGMENTATIONS.get(prompt, [])
        return self.rng.choice(opts) if opts else prompt

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._resolve_image_path(sample["image_path"])
        gray = load_teacher_aligned_gray(image_path)

        if sample["mask_path"] is None:
            mask = np.zeros_like(gray, dtype=np.float32)
        else:
            mask_img = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"掩码缓存不存在或无法读取: {sample['mask_path']}")
            mask = mask_img.astype(np.float32) / 255.0

        gray, meta = letterbox_image(gray, self.img_size, fill_value=0)
        mask, _ = letterbox_mask(mask, self.img_size, fill_value=0)

        if self.training and self.rng.random() < self.hflip_prob:
            gray = np.fliplr(gray)
            mask = np.fliplr(mask)

        prompt = sample["prompt"]
        if self.augment_prompt and self.training and self.rng.random() < PROMPT_AUG_PROB:
            prompt = self._augment(prompt)

        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        mask = (mask > 0.5).astype(np.float32)[None, ...]

        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=15,
            return_tensors="pt",
        )

        return {
            "image": torch.from_numpy(rgb).float(),
            "mask": torch.from_numpy(mask).float(),
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "sample_weight": torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
            "prompt": sample["prompt"],
            "class_name": sample["prompt"],
            "orig_prompt": sample["prompt"],
            "prompt_text": prompt,
            "image_path": sample["image_path"],
            "is_positive": sample["is_positive"],
            "meta": meta,
        }
