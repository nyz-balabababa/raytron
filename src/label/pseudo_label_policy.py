#!/usr/bin/env python3
"""
伪标签训练策略模块。

职责：
1. 定义 A/B/C 分档进入训练的默认权重规则
2. 定义 tiny target 判断与类别级保护策略
3. 定义 building 的严格清理与 C 档 rescue 策略
4. 定义新旧伪标签的融合策略
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    from pycocotools import mask as _mask_utils
except Exception:  # pragma: no cover
    _mask_utils = None


TINY_PROTECT_CLASSES = {"person", "car", "animal"}
NOISE_PRONE_CLASSES = {"building"}

TINY_AREA_THRESHOLDS: Dict[str, int] = {
    "person": 64,
    "car": 64,
    "animal": 64,
    "tree": 64,
    "building": 80,
}

POSITIVE_GRADE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "A": {"default": 1.00},
    "B": {
        "person": 0.75,
        "car": 0.75,
        "animal": 0.70,
        "tree": 0.60,
        "building": 0.65,
        "default": 0.60,
    },
    "C": {
        "person": 0.45,
        "car": 0.45,
        "animal": 0.40,
        "tree": 0.30,
        "building": 0.35,
        "default": 0.00,
    },
}

NEGATIVE_GRADE_WEIGHTS: Dict[str, float] = {
    "A": 0.35,
    "B": 0.20,
    "C": 0.00,
}

VERY_TINY_AREA = 5
EXTREME_TINY_AREA = 3
OLD_SUPPORT_DILATION = 3
NEARBY_SUPPORT_DILATION = 6
LOW_IOU_BUILDING_SUPPORT = 0.35
POLICY_DEVICE = "cpu"
HAS_PYCOCOTOOLS = _mask_utils is not None


class RLEDecodeError(ValueError):
    pass


def set_policy_device(device: str) -> None:
    global POLICY_DEVICE
    if torch is None:
        POLICY_DEVICE = "cpu"
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        POLICY_DEVICE = device
    else:
        POLICY_DEVICE = "cpu"


@dataclass
class ComponentInfo:
    mask: np.ndarray
    area: int
    bbox: Tuple[int, int, int, int]
    bbox_area: int
    aspect_ratio: float
    density: float
    touches_border: bool


@dataclass
class MaskStats:
    component_count: int
    total_mask_area: int
    max_component_area: int
    min_component_area: int
    main_component_ratio: float
    max_aspect_ratio: float
    is_tiny: bool


@dataclass
class PolicyDecision:
    include_in_train: bool
    selected_hit: bool
    sample_weight: float
    mask_source: str
    disabled_reason: Optional[str]
    selected_score: float
    selected_instances: int
    is_tiny: bool
    component_count: int
    max_component_area: int
    total_mask_area: int
    main_component_ratio: float
    max_aspect_ratio: float
    old_new_iou: Optional[float]
    flags: List[str]
    drop_reasons: List[str]
    selected_rle: Optional[Dict[str, Any]]


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    try:
        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        if _mask_utils is None:
            raise ImportError("pycocotools unavailable")
        return _mask_utils.decode(rle_copy).astype(np.uint8)
    except Exception as exc:
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        if isinstance(counts, str):
            compact = counts.strip()
            if compact and any(ch not in "0123456789, \t\r\n" for ch in compact):
                raise RLEDecodeError("pycocotools failed to decode compressed RLE string") from exc

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        compact = counts.strip()
        if compact and any(ch not in "0123456789, \t\r\n" for ch in compact):
            raise RLEDecodeError("unsupported non-numeric RLE string without pycocotools")
        counts = [int(x) for x in compact.split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos : pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    binary = (mask > 0).astype(np.uint8)
    try:
        if _mask_utils is None:
            raise ImportError("pycocotools unavailable")
        encoded = _mask_utils.encode(np.asfortranarray(binary))
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(encoded["size"]), "counts": counts}
    except Exception:
        flat = binary.reshape(-1, order="F")
        counts: List[int] = []
        last = 0
        run = 0
        for value in flat:
            if int(value) == last:
                run += 1
            else:
                counts.append(run)
                run = 1
                last = int(value)
        counts.append(run)
        return {
            "size": [int(binary.shape[0]), int(binary.shape[1])],
            "counts": ",".join(str(x) for x in counts),
        }


def ensure_same_shape(mask: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == ref_shape:
        return mask.astype(np.uint8)
    return cv2.resize(
        mask.astype(np.uint8),
        (ref_shape[1], ref_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        mask_b = ensure_same_shape(mask_b, mask_a.shape)
    if torch is not None and POLICY_DEVICE.startswith("cuda") and torch.cuda.is_available():
        a_t = torch.from_numpy((mask_a > 0).astype(np.uint8)).to(POLICY_DEVICE)
        b_t = torch.from_numpy((mask_b > 0).astype(np.uint8)).to(POLICY_DEVICE)
        inter = int(torch.logical_and(a_t > 0, b_t > 0).sum().item())
        union = int(torch.logical_or(a_t > 0, b_t > 0).sum().item())
    else:
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)
        inter = int((a & b).sum())
        union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or int(mask.sum()) == 0:
        return (mask > 0).astype(np.uint8)
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)


def extract_components(mask: np.ndarray) -> List[ComponentInfo]:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape[:2]
    components: List[ComponentInfo] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(bw * bh, 1)
        aspect_ratio = float(max(bw / max(bh, 1), bh / max(bw, 1)))
        density = float(area) / float(bbox_area)
        touches_border = bool(x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h)
        comp_mask = (labels == label).astype(np.uint8)
        components.append(
            ComponentInfo(
                mask=comp_mask,
                area=area,
                bbox=(x, y, bw, bh),
                bbox_area=bbox_area,
                aspect_ratio=aspect_ratio,
                density=density,
                touches_border=touches_border,
            )
        )
    components.sort(key=lambda item: item.area, reverse=True)
    return components


def compute_mask_stats(mask: np.ndarray, prompt: str) -> MaskStats:
    components = extract_components(mask)
    if not components:
        return MaskStats(
            component_count=0,
            total_mask_area=0,
            max_component_area=0,
            min_component_area=0,
            main_component_ratio=0.0,
            max_aspect_ratio=0.0,
            is_tiny=False,
        )

    areas = [comp.area for comp in components]
    total_area = int(sum(areas))
    max_area = int(max(areas))
    min_area = int(min(areas))
    max_aspect_ratio = max(comp.aspect_ratio for comp in components)
    main_component_ratio = float(max_area) / float(max(total_area, 1))
    tiny_thresh = TINY_AREA_THRESHOLDS.get(prompt, 64)
    is_tiny = bool(max_area < tiny_thresh or all(area < tiny_thresh for area in areas))
    return MaskStats(
        component_count=len(components),
        total_mask_area=total_area,
        max_component_area=max_area,
        min_component_area=min_area,
        main_component_ratio=main_component_ratio,
        max_aspect_ratio=max_aspect_ratio,
        is_tiny=is_tiny,
    )


def component_has_support(component: ComponentInfo, support_mask: Optional[np.ndarray], radius: int) -> bool:
    if support_mask is None or int(support_mask.sum()) == 0:
        return False
    support = dilate_mask(support_mask, radius)
    return bool((support & component.mask).any())


def is_component_abnormal(component: ComponentInfo) -> bool:
    if component.area <= 1:
        return True
    if component.aspect_ratio >= 10.0 and component.density <= 0.40:
        return True
    if component.density <= 0.12 and component.bbox_area >= max(component.area * 8, 80):
        return True
    if component.touches_border and component.area <= 8:
        return True
    return False


def get_positive_grade_weight(prompt: str, grade: str) -> float:
    grade_cfg = POSITIVE_GRADE_WEIGHTS.get(grade, {})
    if prompt in grade_cfg:
        return float(grade_cfg[prompt])
    return float(grade_cfg.get("default", 0.0))


def get_negative_grade_weight(grade: str) -> float:
    return float(NEGATIVE_GRADE_WEIGHTS.get(grade, 0.0))


def has_variant_reference(summary_row: Dict[str, Any]) -> bool:
    flags = summary_row.get("variant_hit_flags")
    if isinstance(flags, list) and len(flags) > 0:
        return True
    return summary_row.get("variant_mean_iou") is not None


def summarize_variant_support(summary_row: Dict[str, Any]) -> bool:
    flags = summary_row.get("variant_hit_flags") or []
    if any(bool(x) for x in flags):
        return True
    mean_iou = summary_row.get("variant_mean_iou")
    if mean_iou is None:
        return False
    return float(mean_iou) >= 0.40


def filter_protected_mask(
    prompt: str,
    grade: str,
    mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    score: float,
    score_q25: float,
    has_variant_reference_flag: bool,
    has_variant_support: bool,
) -> Tuple[np.ndarray, List[str]]:
    kept = np.zeros_like(mask, dtype=np.uint8)
    drop_reasons: List[str] = []
    tiny_thresh = TINY_AREA_THRESHOLDS.get(prompt, 64)

    for component in extract_components(mask):
        if component.area >= tiny_thresh:
            kept |= component.mask
            continue
        if component.area > EXTREME_TINY_AREA:
            kept |= component.mask
            continue

        near_old = component_has_support(component, support_mask, OLD_SUPPORT_DILATION)
        abnormal = is_component_abnormal(component)
        old_unsupported = support_mask is None or int(support_mask.sum()) == 0 or not near_old
        low_score = score_q25 > 0 and score < score_q25

        should_drop = component.area <= EXTREME_TINY_AREA and low_score and abnormal and old_unsupported
        if should_drop:
            reasons = [
                "extreme_tiny_component",
                "low_score",
                "abnormal_shape",
                "old_unsupported",
            ]
            if has_variant_reference_flag and not has_variant_support:
                reasons.append("no_variant_support")
            drop_reasons.extend(reasons)
            continue

        kept |= component.mask

    return kept, drop_reasons


def filter_tree_mask(
    grade: str,
    mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    score: float,
    score_q25: float,
) -> Tuple[np.ndarray, List[str]]:
    kept = np.zeros_like(mask, dtype=np.uint8)
    drop_reasons: List[str] = []

    for component in extract_components(mask):
        near_old = component_has_support(component, support_mask, OLD_SUPPORT_DILATION)
        abnormal = is_component_abnormal(component)
        if component.area <= EXTREME_TINY_AREA and not near_old:
            drop_reasons.append("tree_extreme_tiny_without_support")
            continue
        if grade == "C" and abnormal and not near_old:
            drop_reasons.append("tree_grade_c_abnormal_without_support")
            continue
        if score_q25 > 0 and score < score_q25 and abnormal and not near_old:
            drop_reasons.append("tree_low_score_abnormal")
            continue
        kept |= component.mask

    return kept, drop_reasons


def filter_building_mask(
    grade: str,
    mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    score: float,
    score_q25: float,
) -> Tuple[np.ndarray, List[str]]:
    kept = np.zeros_like(mask, dtype=np.uint8)
    drop_reasons: List[str] = []
    tiny_thresh = TINY_AREA_THRESHOLDS["building"]
    support_exists = support_mask is not None and int(support_mask.sum()) > 0

    for component in extract_components(mask):
        near_old = component_has_support(component, support_mask, OLD_SUPPORT_DILATION)
        abnormal = is_component_abnormal(component)
        low_supported = not support_exists or not near_old

        if component.area <= VERY_TINY_AREA:
            drop_reasons.append("building_very_tiny")
            continue
        if component.area < tiny_thresh and low_supported and score_q25 > 0 and score < score_q25:
            drop_reasons.append("building_tiny_low_score_without_old_support")
            continue
        if component.area < tiny_thresh and low_supported and abnormal:
            drop_reasons.append("building_tiny_abnormal_without_old_support")
            continue
        kept |= component.mask

    stats = compute_mask_stats(kept, "building")
    if stats.component_count >= 5 and stats.main_component_ratio < 0.40:
        drop_reasons.append("building_fragmented_low_main_ratio")

    return kept, drop_reasons


def get_building_c_quality_flags(mask: np.ndarray) -> List[str]:
    stats = compute_mask_stats(mask, "building")
    if stats.total_mask_area == 0:
        return ["building_c_empty_after_clean"]

    reasons: List[str] = []
    if stats.component_count >= 8 and stats.main_component_ratio < 0.25:
        reasons.append("building_c_fragmented_extreme")
    if stats.component_count >= 5 and stats.max_component_area < 24 and stats.main_component_ratio < 0.35:
        reasons.append("building_c_no_clear_main_component")
    if stats.max_aspect_ratio >= 20.0 and stats.max_component_area < max(80, stats.total_mask_area):
        reasons.append("building_c_line_like_extreme")
    return reasons


def keep_drop_reason(reason: str) -> bool:
    return "low_iou" not in reason and "empty" not in reason


def filter_mask_by_prompt(
    prompt: str,
    grade: str,
    mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    score: float,
    score_q25: float,
    has_variant_reference_flag: bool,
    has_variant_support: bool,
) -> Tuple[np.ndarray, List[str]]:
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.uint8), []
    if prompt in TINY_PROTECT_CLASSES:
        return filter_protected_mask(
            prompt=prompt,
            grade=grade,
            mask=mask,
            support_mask=support_mask,
            score=score,
            score_q25=score_q25,
            has_variant_reference_flag=has_variant_reference_flag,
            has_variant_support=has_variant_support,
        )
    if prompt == "building":
        return filter_building_mask(
            grade=grade,
            mask=mask,
            support_mask=support_mask,
            score=score,
            score_q25=score_q25,
        )
    return filter_tree_mask(
        grade=grade,
        mask=mask,
        support_mask=support_mask,
        score=score,
        score_q25=score_q25,
    )


def select_positive_mask(
    prompt: str,
    grade: str,
    new_mask: Optional[np.ndarray],
    old_mask: Optional[np.ndarray],
    new_score: float,
    old_score: float,
    score_q25: float,
    has_variant_reference_flag: bool,
    has_variant_support: bool,
) -> Tuple[np.ndarray, str, List[str]]:
    flags: List[str] = []
    shape = None
    if new_mask is not None:
        shape = new_mask.shape
    elif old_mask is not None:
        shape = old_mask.shape
    else:
        return np.zeros((1, 1), dtype=np.uint8), "none", flags

    if new_mask is not None and old_mask is not None and new_mask.shape != old_mask.shape:
        old_mask = ensure_same_shape(old_mask, new_mask.shape)

    if new_mask is not None:
        cleaned_new, new_drop_reasons = filter_mask_by_prompt(
            prompt=prompt,
            grade=grade,
            mask=new_mask,
            support_mask=old_mask,
            score=new_score,
            score_q25=score_q25,
            has_variant_reference_flag=has_variant_reference_flag,
            has_variant_support=has_variant_support,
        )
        if new_drop_reasons:
            flags.extend(sorted(set(new_drop_reasons)))
    else:
        cleaned_new = np.zeros(shape, dtype=np.uint8)

    if old_mask is not None:
        cleaned_old, old_drop_reasons = filter_mask_by_prompt(
            prompt=prompt,
            grade=grade if prompt != "building" else "B",
            mask=old_mask,
            support_mask=new_mask,
            score=old_score if old_score > 0 else new_score,
            score_q25=score_q25,
            has_variant_reference_flag=True,
            has_variant_support=True,
        )
        if old_drop_reasons:
            flags.extend(sorted(set(f"old:{item}" for item in old_drop_reasons)))
    else:
        cleaned_old = np.zeros(shape, dtype=np.uint8)

    if prompt == "building":
        if int(cleaned_new.sum()) > 0 and int(cleaned_old.sum()) > 0:
            overlap_iou = mask_iou(cleaned_new, cleaned_old)
            if overlap_iou < LOW_IOU_BUILDING_SUPPORT:
                flags.append("building_old_new_low_iou")
                if grade == "C":
                    flags.append("building_c_old_new_low_iou")
            source = "building_c_cleaned" if grade == "C" else "building_new_filtered"
            return cleaned_new.astype(np.uint8), source, flags
        if int(cleaned_new.sum()) > 0:
            source = "building_c_cleaned" if grade == "C" else "building_new_filtered"
            return cleaned_new.astype(np.uint8), source, flags
        if int(cleaned_old.sum()) > 0:
            return cleaned_old.astype(np.uint8), "building_old_fallback", flags
        return np.zeros(shape, dtype=np.uint8), "building_empty", flags

    if prompt in TINY_PROTECT_CLASSES:
        if int(cleaned_new.sum()) > 0 and int(cleaned_old.sum()) > 0:
            supported_old = cleaned_old & dilate_mask(cleaned_new, NEARBY_SUPPORT_DILATION)
            fused = cleaned_new | supported_old
            return fused.astype(np.uint8), "protected_new_plus_old_support", flags
        if int(cleaned_new.sum()) > 0:
            return cleaned_new.astype(np.uint8), "protected_new", flags
        if int(cleaned_old.sum()) > 0:
            return cleaned_old.astype(np.uint8), "protected_old_fallback", flags
        return np.zeros(shape, dtype=np.uint8), "protected_empty", flags

    if int(cleaned_new.sum()) > 0:
        return cleaned_new.astype(np.uint8), "tree_new", flags
    if int(cleaned_old.sum()) > 0:
        return cleaned_old.astype(np.uint8), "tree_old_fallback", flags
    return np.zeros(shape, dtype=np.uint8), "tree_empty", flags


def build_policy_decision(
    image_path: str,
    prompt: str,
    grade: str,
    new_info: Dict[str, Any],
    old_info: Optional[Dict[str, Any]],
    summary_row: Dict[str, Any],
    prompt_stats: Dict[str, Dict[str, float]],
) -> PolicyDecision:
    del image_path

    new_hit = bool(new_info.get("hit")) and new_info.get("rle") is not None
    old_hit = bool(old_info and old_info.get("hit") and old_info.get("rle") is not None)
    new_score = float(new_info.get("score", 0.0))
    old_score = float(old_info.get("score", 0.0)) if old_info else 0.0
    score_q25 = float(prompt_stats.get(prompt, {}).get("score_q25", 0.0))
    has_variant_reference_flag = has_variant_reference(summary_row)
    has_variant_support = summarize_variant_support(summary_row)
    old_new_iou: Optional[float] = None

    positive_weight = get_positive_grade_weight(prompt, grade)
    negative_weight = get_negative_grade_weight(grade)
    flags: List[str] = [f"grade:{grade}"]
    drop_reasons: List[str] = []

    try:
        new_mask = rle_to_mask(new_info["rle"]) if new_hit else None
        old_mask = rle_to_mask(old_info["rle"]) if old_hit and old_info else None
    except RLEDecodeError:
        flags.extend(["rle_decode_failed"])
        if new_hit:
            flags.append("new_rle_decode_failed")
        if old_hit:
            flags.append("old_rle_decode_failed")
        return PolicyDecision(
            include_in_train=False,
            selected_hit=False,
            sample_weight=0.0,
            mask_source="rle_decode_failed",
            disabled_reason="rle_decode_failed",
            selected_score=0.0,
            selected_instances=0,
            is_tiny=False,
            component_count=0,
            max_component_area=0,
            total_mask_area=0,
            main_component_ratio=0.0,
            max_aspect_ratio=0.0,
            old_new_iou=None,
            flags=sorted(set(flags)),
            drop_reasons=["rle_decode_failed"],
            selected_rle=None,
        )

    if new_mask is not None and old_mask is not None and new_mask.shape != old_mask.shape:
        old_mask = ensure_same_shape(old_mask, new_mask.shape)
    if new_mask is not None and old_mask is not None:
        old_new_iou = float(mask_iou(new_mask, old_mask))

    if new_hit or old_hit:
        selected_mask, mask_source, filter_flags = select_positive_mask(
            prompt=prompt,
            grade=grade,
            new_mask=new_mask,
            old_mask=old_mask,
            new_score=new_score,
            old_score=old_score,
            score_q25=score_q25,
            has_variant_reference_flag=has_variant_reference_flag,
            has_variant_support=has_variant_support,
        )
        flags.extend(filter_flags)
        stats = compute_mask_stats(selected_mask, prompt)
        selected_hit = stats.total_mask_area > 0

        if selected_hit and positive_weight > 0:
            quality_flags: List[str] = []
            if prompt == "building" and grade == "C":
                quality_flags = get_building_c_quality_flags(selected_mask)
            if quality_flags:
                selected_hit = False
                selected_score = 0.0
                selected_instances = 0
                selected_rle = None
                include_in_train = False
                sample_weight = 0.0
                mask_source = "positive_dropped_after_policy"
                disabled_reason = "building_c_noise_disabled"
                drop_reasons = sorted(
                    set(reason for reason in (filter_flags + quality_flags + ["building_c_noise_disabled"]) if keep_drop_reason(reason))
                )
                flags.extend(quality_flags)
                flags.append("building_c_noise_disabled")
                stats = compute_mask_stats(np.zeros_like(selected_mask, dtype=np.uint8), prompt)
            else:
                disabled_reason = None
                sample_weight = positive_weight
                if prompt == "building" and grade == "C":
                    if mask_source == "building_old_fallback":
                        flags.append("building_c_old_fallback")
                    else:
                        flags.append("building_c_cleaned_train")
                if prompt in TINY_PROTECT_CLASSES and stats.is_tiny:
                    flags.append("tiny_protected")
                if prompt in NOISE_PRONE_CLASSES:
                    flags.append("noise_prone_class")
                include_in_train = True
                selected_score = max(new_score, old_score)
                selected_instances = int(stats.component_count)
                selected_rle = mask_to_rle(selected_mask)
                drop_reasons = sorted(set(reason for reason in filter_flags if keep_drop_reason(reason)))
        else:
            selected_hit = False
            selected_score = 0.0
            selected_instances = 0
            selected_rle = None
            include_in_train = False
            sample_weight = 0.0
            mask_source = "positive_dropped_after_policy"
            if prompt == "building" and grade == "C":
                disabled_reason = "building_c_noise_disabled"
                flags.append("building_c_noise_disabled")
                drop_reasons = sorted(
                    set(reason for reason in (filter_flags + ["building_c_noise_disabled"]) if keep_drop_reason(reason))
                )
            else:
                disabled_reason = "positive_dropped_after_policy"
                drop_reasons = sorted(
                    set(reason for reason in (filter_flags + ["positive_dropped_after_policy"]) if keep_drop_reason(reason))
                )
            stats = compute_mask_stats(np.zeros_like(selected_mask, dtype=np.uint8), prompt)
    else:
        selected_hit = False
        selected_score = 0.0
        selected_instances = 0
        selected_rle = None
        stats = compute_mask_stats(np.zeros((1, 1), dtype=np.uint8), prompt)
        if negative_weight > 0:
            include_in_train = True
            sample_weight = negative_weight
            mask_source = "stable_negative"
            disabled_reason = None
        else:
            include_in_train = False
            sample_weight = 0.0
            mask_source = "negative_excluded"
            disabled_reason = "negative_excluded"

    if prompt in TINY_PROTECT_CLASSES and stats.is_tiny:
        flags.append("is_tiny")
    if prompt in NOISE_PRONE_CLASSES:
        flags.append("building_strict_policy")
    if new_hit and old_hit:
        flags.append("old_new_both_hit")
    elif new_hit:
        flags.append("new_hit_only")
    elif old_hit:
        flags.append("old_hit_only")
    else:
        flags.append("no_positive_hit")

    return PolicyDecision(
        include_in_train=include_in_train,
        selected_hit=selected_hit,
        sample_weight=float(sample_weight),
        mask_source=mask_source,
        disabled_reason=disabled_reason,
        selected_score=float(selected_score),
        selected_instances=int(selected_instances),
        is_tiny=bool(stats.is_tiny),
        component_count=int(stats.component_count),
        max_component_area=int(stats.max_component_area),
        total_mask_area=int(stats.total_mask_area),
        main_component_ratio=float(stats.main_component_ratio),
        max_aspect_ratio=float(stats.max_aspect_ratio),
        old_new_iou=old_new_iou,
        flags=sorted(set(flags)),
        drop_reasons=drop_reasons,
        selected_rle=selected_rle,
    )
