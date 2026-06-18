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

from config_esam_cclip_11 import (
    CONF_FILTER,
    LOSS_WEIGHT_FLOOR,
    PROMPT_AUG_PROB,
    RARE_BALANCED_OLD_CLASSES,
    RARE_BALANCED_RARE_CLASSES,
)
from common_esam_cclip_11 import (
    letterbox_image,
    letterbox_mask,
    load_teacher_aligned_gray,
    maybe_tqdm,
    resolve_existing_path,
    rle_to_mask,
)

LOGGER = logging.getLogger("ESAM_CCLIP_11")


def deterministic_keep(key: str, keep_ratio: float, seed: int) -> bool:
    if keep_ratio >= 1.0:
        return True
    if keep_ratio <= 0.0:
        return False
    raw = f"{seed}|{key}".encode("utf-8")
    value = int(hashlib.md5(raw).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < float(keep_ratio)


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
        prompt_alias_prob=PROMPT_AUG_PROB,
        hflip_prob=0.5,
        use_conf_filter=False,
        negative_sample_prob=0.0,
        negative_sample_weight=0.30,
        rare_oversample=None,
        weak3_enabled=False,
        weak3_seed=0,
        weak_class_keep_ratio=None,
        weak_class_loss_weight=None,
        disabled_train_classes=None,
        class_keep_ratio=None,
        class_loss_scale=None,
        old_class_sample_ratio=1.0,
        rare_class_keep_ratio=1.0,
        old_classes=None,
        rare_classes=None,
        rare_balance_enabled=False,
        training=True,
        seed=42,
        **extra_kwargs,
    ):
        if extra_kwargs:
            LOGGER.warning("忽略未使用的 dataset kwargs: %s", sorted(extra_kwargs.keys()))
        self.annotation_json = resolve_existing_path(annotation_json)
        self.split_txt = resolve_existing_path(split_txt) if split_txt is not None else None
        self.image_root = resolve_existing_path(image_root)
        self.img_size = tuple(img_size)
        self.classes = list(classes)
        self.class_set = set(classes)
        self.tokenizer = tokenizer
        self.prompt_prototypes = prompt_prototypes or {}
        self.augment_prompt = augment_prompt
        self.prompt_alias_prob = float(prompt_alias_prob)
        self.hflip_prob = hflip_prob
        self.use_conf_filter = use_conf_filter
        self.negative_sample_prob = negative_sample_prob
        self.negative_sample_weight = negative_sample_weight
        self.rare_oversample = rare_oversample or {}
        self.weak3_enabled = bool(training and weak3_enabled)
        self.weak3_seed = int(weak3_seed)
        self.weak_class_keep_ratio = {str(k): float(v) for k, v in (weak_class_keep_ratio or {}).items()}
        self.weak_class_loss_weight = {str(k): float(v) for k, v in (weak_class_loss_weight or {}).items()}
        self.disabled_train_classes = {str(name) for name in (disabled_train_classes or [])}
        self.class_keep_ratio = {str(k): float(v) for k, v in (class_keep_ratio or {}).items()}
        self.class_loss_scale = {str(k): float(v) for k, v in (class_loss_scale or {}).items()}
        self.old_class_sample_ratio = float(old_class_sample_ratio)
        self.rare_class_keep_ratio = float(rare_class_keep_ratio)
        self.old_classes = set(old_classes or RARE_BALANCED_OLD_CLASSES)
        self.rare_classes = set(rare_classes or RARE_BALANCED_RARE_CLASSES)
        self.rare_balance_enabled = bool(training and rare_balance_enabled)
        self.training = training
        self.rng = random.Random(seed)
        self.allowed = self._load_allowed_set()
        self.samples, self.stats = self._build_samples()
        LOGGER.info("dataset=%s", Path(self.annotation_json).name)
        LOGGER.info("total samples=%d", self.stats["total"])
        LOGGER.info("positive stats=%s", self.stats["positive"])
        LOGGER.info("negative stats=%s", self.stats["negative"])
        LOGGER.info("positive_raw=%s", self.stats.get("positive_raw", {}))
        LOGGER.info("positive_rebalanced=%s", self.stats.get("positive_rebalanced", {}))
        LOGGER.info("positive_kept=%s", self.stats.get("positive_kept", {}))
        LOGGER.info("dropped_by_disabled=%s", self.stats.get("dropped_by_disabled", {}))
        LOGGER.info("dropped_by_keep_ratio=%s", self.stats.get("dropped_by_keep_ratio", {}))
        LOGGER.info("dropped_by_weak3=%s", self.stats.get("dropped_by_weak3", {}))
        LOGGER.info("final_sample_count=%s", self.stats.get("final_sample_count", {}))
        LOGGER.info("avg_sample_weight=%s", self.stats.get("avg_sample_weight", {}))
        LOGGER.info("rebalance=%s", self.stats.get("rebalance", {}))
        LOGGER.info("filtered_computer=%d", self.stats["filtered_computer"])
        LOGGER.info("filtered_invalid_prompt=%d", self.stats["filtered_invalid_prompt"])
        LOGGER.info("filtered_disabled=%d", self.stats["filtered_disabled"])
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

    def _should_keep_positive(self, prompt: str) -> bool:
        if not self.rare_balance_enabled:
            return True
        if prompt in self.old_classes:
            return self.rng.random() < self.old_class_sample_ratio
        if prompt in self.rare_classes:
            return self.rng.random() < self.rare_class_keep_ratio
        return True

    def _build_samples(self):
        with open(self.annotation_json, "r", encoding="utf-8-sig") as file_obj:
            preds = json.load(file_obj)

        base_samples = []
        class_pos_counter_raw = Counter()
        class_pos_counter_kept = Counter()
        class_neg_counter = Counter()
        dropped_by_disabled = Counter()
        dropped_by_keep_ratio = Counter()
        dropped_by_weak3 = Counter()
        filtered_computer = 0
        filtered_invalid_prompt = 0
        filtered_disabled = 0
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
                include_in_train = bool(prompt_info.get("include_in_train", True))
                if prompt_info.get("disabled_reason"):
                    include_in_train = False
                if not include_in_train:
                    filtered_disabled += 1
                    continue
                hit = bool(prompt_info.get("selected_hit", prompt_info.get("hit")))
                score = float(prompt_info.get("sample_weight", prompt_info.get("selected_score", prompt_info.get("score", 1.0))))
                if hit:
                    if self.use_conf_filter and score < self._score_threshold(prompt):
                        continue
                    rle = prompt_info.get("selected_rle", prompt_info.get("rle"))
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
                    class_pos_counter_raw[prompt] += 1
                    if self.training and self.weak3_enabled and prompt in self.weak_class_keep_ratio:
                        weak_keep_ratio = float(self.weak_class_keep_ratio[prompt])
                        instance_key = f"{image_path}|{prompt}|{ann_id if ann_id is not None else rle_hash}"
                        if not deterministic_keep(instance_key, weak_keep_ratio, self.weak3_seed):
                            dropped_by_weak3[prompt] += 1
                            continue
                    if self.training and prompt in self.disabled_train_classes:
                        dropped_by_disabled[prompt] += 1
                        continue
                    keep_ratio = float(self.class_keep_ratio.get(prompt, 1.0))
                    if self.training and keep_ratio < 1.0:
                        if self.rng.random() > keep_ratio:
                            dropped_by_keep_ratio[prompt] += 1
                            continue
                    sample_weight = max(score, LOSS_WEIGHT_FLOOR)
                    # Weak3 final class loss weight is handled by CLASS_WEIGHTS in train loss.
                    # Do not multiply WEAK_CLASS_LOSS_WEIGHT here to avoid double weighting.
                    sample_weight *= float(self.class_loss_scale.get(prompt, 1.0))
                    base_samples.append(
                        {
                            "image_path": image_path,
                            "prompt": prompt,
                            "mask_path": cache_path,
                            "sample_weight": sample_weight,
                            "is_positive": True,
                            "ann_id": ann_id,
                        }
                    )
                    class_pos_counter_kept[prompt] += 1
                else:
                    if self.training and prompt in self.disabled_train_classes:
                        continue
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

        rebalance_stats = {
            "old_positive_kept": 0,
            "old_positive_dropped": 0,
            "rare_positive_kept": 0,
            "rare_positive_dropped": 0,
            "old_class_sample_ratio": self.old_class_sample_ratio,
            "rare_class_keep_ratio": self.rare_class_keep_ratio,
            "rare_balance_enabled": self.rare_balance_enabled,
        }
        positives = [sample for sample in base_samples if sample["is_positive"]]
        negatives = [sample for sample in base_samples if not sample["is_positive"]]
        kept_positives = []
        for sample in positives:
            prompt = sample["prompt"]
            keep = self._should_keep_positive(prompt)
            if prompt in self.old_classes:
                rebalance_stats["old_positive_kept" if keep else "old_positive_dropped"] += 1
            elif prompt in self.rare_classes:
                rebalance_stats["rare_positive_kept" if keep else "rare_positive_dropped"] += 1
            if keep:
                kept_positives.append(sample)
        base_samples = kept_positives + negatives

        if self.rare_oversample:
            extra = []
            for sample in kept_positives:
                mult = int(self.rare_oversample.get(sample["prompt"], 1))
                for _ in range(max(mult - 1, 0)):
                    extra.append(sample.copy())
            base_samples.extend(extra)

        class_pos_counter = Counter(sample["prompt"] for sample in base_samples if sample["is_positive"])
        final_sample_counter = Counter(sample["prompt"] for sample in base_samples)
        class_weight_sum = Counter()
        class_weight_count = Counter()
        for sample in base_samples:
            class_weight_sum[sample["prompt"]] += float(sample["sample_weight"])
            class_weight_count[sample["prompt"]] += 1
        avg_sample_weight = {
            class_name: round(class_weight_sum[class_name] / class_weight_count[class_name], 6)
            for class_name in sorted(class_weight_count.keys())
            if class_weight_count[class_name] > 0
        }
        stats = {
            "positive": dict(class_pos_counter),
            "positive_raw": dict(class_pos_counter_raw),
            "positive_kept": dict(class_pos_counter_kept),
            "positive_rebalanced": dict(class_pos_counter),
            "negative": dict(class_neg_counter),
            "dropped_by_disabled": dict(dropped_by_disabled),
            "dropped_by_keep_ratio": dict(dropped_by_keep_ratio),
            "dropped_by_weak3": dict(dropped_by_weak3),
            "final_sample_count": dict(final_sample_counter),
            "avg_sample_weight": avg_sample_weight,
            "total": len(base_samples),
            "filtered_computer": filtered_computer,
            "filtered_invalid_prompt": filtered_invalid_prompt,
            "filtered_disabled": filtered_disabled,
            "empty_mask": empty_mask_count,
            "rebalance": rebalance_stats,
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
        if self.augment_prompt and self.training and self.rng.random() < self.prompt_alias_prob:
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
