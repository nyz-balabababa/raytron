import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config_esam_cclip_11 import CONF_FILTER, PROMPT_AUG_PROB
from common_esam_cclip_11 import (
    letterbox_image,
    letterbox_mask,
    load_teacher_aligned_gray,
    maybe_tqdm,
    resolve_existing_path,
    rle_to_mask,
)

LOGGER = logging.getLogger("ESAM_CCLIP_11")


class ESAMCCLIP11Dataset(Dataset):
    def __init__(
        self,
        annotation_json,
        image_root,
        img_size,
        classes,
        split_txt=None,
        tokenizer=None,
        prompt_prototypes=None,
        augment_prompt=False,
        hflip_prob=0.5,
        use_conf_filter=False,
        negative_sample_prob=0.0,
        negative_sample_weight=0.30,
        rare_oversample=None,
        training=True,
        seed=42,
    ):
        self.annotation_json = resolve_existing_path(annotation_json)
        self.split_txt = resolve_existing_path(split_txt) if split_txt is not None else None
        self.image_root = resolve_existing_path(image_root)
        self.img_size = tuple(img_size)
        self.classes = list(classes)
        self.class_set = set(classes)
        self.tokenizer = tokenizer
        self.prompt_prototypes = prompt_prototypes or {}
        self.augment_prompt = augment_prompt
        self.hflip_prob = hflip_prob
        self.use_conf_filter = use_conf_filter
        self.negative_sample_prob = negative_sample_prob
        self.negative_sample_weight = negative_sample_weight
        self.rare_oversample = rare_oversample or {}
        self.training = training
        self.rng = random.Random(seed)
        self.allowed = self._load_allowed_set()
        self.samples, self.stats = self._build_samples()
        LOGGER.info("dataset=%s", Path(self.annotation_json).name)
        LOGGER.info("total samples=%d", self.stats["total"])
        LOGGER.info("positive stats=%s", self.stats["positive"])
        LOGGER.info("negative stats=%s", self.stats["negative"])
        LOGGER.info("filtered_computer=%d", self.stats["filtered_computer"])
        LOGGER.info("filtered_invalid_prompt=%d", self.stats["filtered_invalid_prompt"])
        LOGGER.info("empty_mask=%d", self.stats["empty_mask"])

    def _load_allowed_set(self):
        if self.split_txt is None:
            return None
        with open(self.split_txt, "r", encoding="utf-8") as file_obj:
            return {line.strip().replace("\\", "/") for line in file_obj if line.strip()}

    def _normalize_image_path(self, image_path):
        image_path = str(image_path).replace("\\", "/")
        if self.allowed is None:
            return image_path
        if image_path in self.allowed:
            return image_path
        alt = image_path[5:] if image_path.startswith("test/") else f"test/{image_path}"
        if alt in self.allowed:
            return alt
        return None

    def _resolve_image_path(self, image_path):
        rel = str(image_path).replace("\\", "/")
        candidates = [
            Path(self.image_root) / rel,
            Path(self.image_root) / rel[5:] if rel.startswith("test/") else Path(self.image_root) / "test" / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _score_threshold(self, prompt):
        return CONF_FILTER.get(str(prompt), 0.5)

    def _build_samples(self):
        with open(self.annotation_json, "r", encoding="utf-8-sig") as file_obj:
            preds = json.load(file_obj)

        base_samples = []
        class_pos_counter = Counter()
        class_neg_counter = Counter()
        filtered_computer = 0
        filtered_invalid_prompt = 0
        empty_mask_count = 0
        cache_dir = Path(self.image_root) / "test" / "train_output" / ".mask_cache_esam_cclip_11" / Path(self.annotation_json).stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        iterator = maybe_tqdm(
            preds,
            total=len(preds),
            desc=f"BuildDataset[{Path(self.annotation_json).stem}]",
            leave=False,
        )
        for item in iterator:
            image_path = self._normalize_image_path(item["image_path"])
            if image_path is None:
                continue
            abs_path = self._resolve_image_path(image_path)
            if not abs_path.exists():
                continue
            ann_id = item.get("ann_id")
            for prompt, prompt_info in item.get("prompts", {}).items():
                prompt = str(prompt)
                if prompt == "computer":
                    filtered_computer += 1
                    continue
                if prompt not in self.class_set:
                    filtered_invalid_prompt += 1
                    continue
                hit = bool(prompt_info.get("hit"))
                score = float(prompt_info.get("score", 1.0))
                if hit:
                    if self.use_conf_filter and score < self._score_threshold(prompt):
                        continue
                    rle = prompt_info.get("rle")
                    if rle is None:
                        continue
                    rle_counts = rle.get("counts", "")
                    if isinstance(rle_counts, bytes):
                        rle_counts = rle_counts.decode("utf-8", errors="ignore")
                    rle_hash = hashlib.md5(str(rle_counts).encode("utf-8")).hexdigest()[:12]
                    cache_name = hashlib.md5(f"{image_path}|{prompt}|{ann_id}|{rle_hash}".encode("utf-8")).hexdigest()
                    cache_path = cache_dir / f"{cache_name}.png"
                    if not cache_path.exists():
                        mask = rle_to_mask(rle)
                        if mask.sum() <= 0:
                            empty_mask_count += 1
                            continue
                        cv2.imwrite(str(cache_path), (mask * 255).astype(np.uint8))
                    base_samples.append(
                        {
                            "image_path": image_path,
                            "prompt": prompt,
                            "mask_path": cache_path,
                            "sample_weight": max(score, 0.7),
                            "is_positive": True,
                            "ann_id": ann_id,
                        }
                    )
                    class_pos_counter[prompt] += 1
                else:
                    if self.negative_sample_prob > 0 and self.rng.random() < self.negative_sample_prob:
                        base_samples.append(
                            {
                                "image_path": image_path,
                                "prompt": prompt,
                                "mask_path": None,
                                "sample_weight": self.negative_sample_weight,
                                "is_positive": False,
                                "ann_id": ann_id,
                            }
                        )
                        class_neg_counter[prompt] += 1

        if self.rare_oversample:
            extra = []
            for sample in base_samples:
                if not sample["is_positive"]:
                    continue
                mult = int(self.rare_oversample.get(sample["prompt"], 1))
                for _ in range(max(mult - 1, 0)):
                    extra.append(sample.copy())
            base_samples.extend(extra)

        stats = {
            "positive": dict(class_pos_counter),
            "negative": dict(class_neg_counter),
            "total": len(base_samples),
            "filtered_computer": filtered_computer,
            "filtered_invalid_prompt": filtered_invalid_prompt,
            "empty_mask": empty_mask_count,
        }
        return base_samples, stats

    def __len__(self):
        return len(self.samples)

    def _augment_prompt(self, prompt):
        aliases = self.prompt_prototypes.get(prompt, [prompt])
        return self.rng.choice(aliases) if aliases else prompt

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

        if self.training and self.hflip_prob > 0 and self.rng.random() < self.hflip_prob:
            gray = np.fliplr(gray)
            mask = np.fliplr(mask)

        prompt = sample["prompt"]
        prompt_text = prompt
        if self.augment_prompt and self.training and self.rng.random() < PROMPT_AUG_PROB:
            prompt_text = self._augment_prompt(prompt)

        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        mask = (mask > 0.5).astype(np.float32)[None, ...]

        return {
            "image": torch.from_numpy(rgb).float(),
            "mask": torch.from_numpy(mask).float(),
            "sample_weight": torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
            "prompt": prompt,
            "class_name": prompt,
            "prompt_text": prompt_text,
            "image_path": sample["image_path"],
            "is_positive": sample["is_positive"],
            "ann_id": sample.get("ann_id"),
            "meta": meta,
        }
