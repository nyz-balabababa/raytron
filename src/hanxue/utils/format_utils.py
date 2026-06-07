# 文件位置: PythonProject2/utils/format_utils.py
import numpy as np
from pycocotools import mask as mask_utils


def encode_mask_to_rle(binary_mask):
    """
    将二值 mask 转换为 COCO RLE 格式。

    参数:
        binary_mask:
            numpy.ndarray，形状 [H, W]
            值可以是 bool / 0-1 / 0-255

    返回:
        dict:
            {
                "size": [H, W],
                "counts": "..."
            }
    """
    if binary_mask is None:
        raise ValueError("binary_mask 不能为 None")

    binary_mask = np.asarray(binary_mask)

    if binary_mask.ndim == 3:
        # 兼容 [1, H, W] 或 [H, W, 1]
        if binary_mask.shape[0] == 1:
            binary_mask = binary_mask[0]
        elif binary_mask.shape[-1] == 1:
            binary_mask = binary_mask[..., 0]
        else:
            raise ValueError(f"encode_mask_to_rle 只接受二维 mask，当前形状: {binary_mask.shape}")

    if binary_mask.ndim != 2:
        raise ValueError(f"encode_mask_to_rle 只接受二维 mask，当前形状: {binary_mask.shape}")

    # 统一为 0/1 的 uint8
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # pycocotools 要求 Fortran order
    mask_fortran = np.asfortranarray(binary_mask)

    rle = mask_utils.encode(mask_fortran)

    # counts 是 bytes，JSON 不能直接序列化
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    # 确保 size 是普通 Python int，避免 numpy 类型 JSON 序列化问题
    rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]

    return rle


def empty_mask_rle(height, width):
    """
    生成一张空 mask 的 RLE。
    用于模型没有找到目标时，保证每个 ann_id 都有输出。
    """
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    return encode_mask_to_rle(mask)