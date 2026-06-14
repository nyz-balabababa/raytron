import logging
from pathlib import Path

import torch

from config_esam_cclip_11 import CLASSES, PROMPT_PROTOTYPES
from common_esam_cclip_11 import maybe_tqdm, normalize_text_feature, save_json

LOGGER = logging.getLogger("ESAM_CCLIP_11")


def validate_text_cache(payload, classes):
    expected_classes = list(classes)
    cache_classes = list(payload.get("classes", []))
    if set(cache_classes) != set(expected_classes):
        raise RuntimeError("text cache 与当前 11 类不一致，请使用 --rebuild_text_cache")
    embeddings = payload.get("embeddings", {})
    for class_name in expected_classes:
        if class_name not in embeddings:
            raise RuntimeError(f"text cache 缺少类别 embedding: {class_name}，请使用 --rebuild_text_cache")
        embedding = embeddings[class_name]
        if not torch.is_tensor(embedding):
            raise RuntimeError(f"text cache embedding 不是 tensor: {class_name}")
        if embedding.ndim != 1:
            raise RuntimeError(f"text cache embedding 维度异常: {class_name} shape={tuple(embedding.shape)}")
        if int(embedding.numel()) <= 0:
            raise RuntimeError(f"text cache embedding 为空: {class_name}")


def build_text_cache_payload(model, tokenizer, classes, prompt_prototypes, device):
    embeddings = {}
    aliases_used = {}
    iterator = maybe_tqdm(
        classes,
        total=len(classes),
        desc="BuildTextCache",
        leave=False,
    )
    for class_name in iterator:
        aliases = prompt_prototypes.get(class_name, [class_name])
        aliases_used[class_name] = aliases
        emb_list = []
        for alias in aliases:
            encoded = tokenizer(
                alias,
                padding="max_length",
                truncation=True,
                max_length=15,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            text_feature = model.encode_text(input_ids, attention_mask)
            text_feature = normalize_text_feature(text_feature).detach().cpu()
            emb_list.append(text_feature[0])
        prototype = normalize_text_feature(torch.stack(emb_list, dim=0).mean(dim=0, keepdim=True))[0]
        embeddings[class_name] = prototype.cpu()
    return {
        "classes": list(classes),
        "use_prompt_prototype": True,
        "embeddings": embeddings,
        "aliases": aliases_used,
    }


def save_text_cache(cache_path: Path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    LOGGER.info("text cache saved: %s", cache_path)


def _try_remove_bad_cache(cache_path: Path):
    try:
        if cache_path.exists():
            cache_path.unlink()
            LOGGER.warning("已删除损坏的 text cache: %s", cache_path)
    except Exception as exc:
        LOGGER.warning("删除损坏 text cache 失败: %s error=%s", cache_path, exc)


def load_or_build_text_cache(
    model,
    tokenizer,
    cache_path: Path,
    classes=None,
    prompt_prototypes=None,
    device="cpu",
    rebuild=False,
):
    classes = classes or CLASSES
    prompt_prototypes = prompt_prototypes or PROMPT_PROTOTYPES
    if cache_path.exists() and not rebuild:
        try:
            if cache_path.stat().st_size <= 0:
                raise EOFError("text cache 文件为空")
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            validate_text_cache(payload, classes)
            LOGGER.info("loaded text cache: %s", cache_path)
            for class_name, aliases in payload.get("aliases", {}).items():
                LOGGER.info("prototype aliases %s -> %s", class_name, aliases)
            return payload
        except Exception as exc:
            LOGGER.warning("加载 text cache 失败，将自动重建: %s error=%s", cache_path, exc)
            _try_remove_bad_cache(cache_path)

    payload = build_text_cache_payload(
        model=model,
        tokenizer=tokenizer,
        classes=classes,
        prompt_prototypes=prompt_prototypes,
        device=device,
    )
    validate_text_cache(payload, classes)
    save_text_cache(cache_path, payload)
    save_json(cache_path.with_suffix(".json"), {"classes": classes, "aliases": payload["aliases"]})
    return payload
