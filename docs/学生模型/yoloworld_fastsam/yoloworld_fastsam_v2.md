# YOLO-World + FastSAM v2 改进总结

> 基于 v1 的增量改进，对应文件: `src/train/world/world_train_v2.py` + `src/train/world/config_world_v2.py`

---

## 一、改进总览

| # | 改进项 | 类别 | 改动位置 |
|---|--------|------|----------|
| 1 | 多尺度推理 (person/tree/animal) | 小目标 | `_multi_scale_detect()` + config |
| 2 | cls↑ box↑ label_smoothing | 替代 Focal Loss | `train()` + config |
| 3 | computer 阈值 0.4 + 独立过采样 5× | 类别不平衡 | `convert_split()` + config |
| 4 | FastSAM 动态 imgsz | 小 crop 分割质量 | `_fastsam_segment_boxes()` |
| 5 | set_classes 提到循环外 | 推理效率 | `_multi_scale_detect()` |
| 6 | MS_NMS_IOU 0.35 | 跨尺度合并 | config |
| 7 | GPU 信息日志 (多版本兼容) | 可移植性 | `log_gpu_info()` |
| 8 | CLIP 权重离线缓存 | 免网络依赖 | `_ensure_local_clip()` |
| 9 | 自定义 6 面板训练曲线图 | 训练监控 | `plot_training_curves()` |
| 10 | 训练总结日志 (mAP/P/R/早停) | 训练监控 | `train()` 末尾 |
| 11 | `ULTRALYTICS_DOWNLOADS=false` | 免网络依赖 | 文件顶部 |
| 12 | 独立数据集目录 `yolo_world_dataset_v2` | 实验隔离 | config |

---

## 二、各改进详细说明

### 1. 多尺度推理

**问题**: person 类 74% 实例面积 <1%，tree 中小目标多，animal 在远距离监控中目标极小。单尺度 (800px) 对小目标特征提取不足。

**方案**: 推理时对 `["person", "tree", "animal"]` 三类的同一张图跑 4 个尺度 `[0.8, 1.0, 1.25, 1.5]×`，跨尺度 NMS (IoU=0.35) 合并后送给 FastSAM。

```python
# 关键参数
MULTI_SCALE_CLASSES = ["person", "tree", "animal"]
MULTI_SCALES = [0.8, 1.0, 1.25, 1.5]
MS_NMS_IOU = 0.35         # 低阈值容忍跨尺度框漂移
MS_CONF = 0.2             # 多尺度降低检测阈值避免漏检
```

**收益**: person/tree/animal 的小目标检测召回率提升，不增加训练成本（纯推理时改动）。

### 2. cls/box loss 调整

**问题**: YOLO-World 基于 Ultralytics API，分类 loss 内部是 `BCEWithLogitsLoss`，不提供 Focal Loss 开关。

**方案**: 用三个可配置参数近似 Focal Loss 效果：

| 参数 | v1 | v2 | 作用 |
|------|----|----|------|
| `cls` | 0.5 (默认) | 1.0 | 加大分类 loss → 稀有类误分类惩罚更重 |
| `box` | 7.5 (默认) | 10.0 | 加大框 loss → 小目标框更精准 → FastSAM ROI 质量更高 |
| `label_smoothing` | 0.0 | 0.05 | 让稀有类有更确定的分类边界 |

### 3. computer 过采样解耦

**问题**: v1 用 `max(RARE_OVERSAMPLE.values())` 作为同一张图的过采样倍数。当 animal=3× 和 computer=5× 共存时，animal 也被意外复制 5 次。

**方案**: 拆分两阶段，animal 和 computer 完全解耦：

```
阶段 A (主循环):
  RARE_OVERSAMPLE = {"animal": 3}
  → animal 图片复制 2 次 (总 3×)

阶段 B (追加):
  COMPUTER_EXTRA = 4
  → computer 图片额外复制 4 次 (总 5×)
```

**收益**: animal 严格 3×，computer 严格 5×，互不干扰。

### 4. FastSAM 动态 imgsz

**问题**: 小目标检测框可能只有 20×30 像素，固定 `imgsz=1024` 会严重拉伸导致分割质量极差。

**方案**:
```python
max_side = max(crop_h, crop_w)
dynamic_imgsz = max(256, min(1024, max_side))
dynamic_imgsz = ((dynamic_imgsz + 31) // 32) * 32   # 对齐 32 的倍数
```

| crop 尺寸 | 原 imgsz | 新 imgsz |
|-----------|----------|----------|
| 20×30 | 1024 | 256 |
| 200×300 | 1024 | 320 |
| 600×800 | 1024 | 800 |
| 1200×1600 | 1024 | 1024 |

### 5. set_classes 优化

**问题**: `_multi_scale_detect()` 中每个尺度都调用 `yolo.set_classes([prompt])`，CLIP 文本编码器对同一 prompt 重复编码 4 次。

**方案**: 提到 `for scale` 循环外，只编码一次。

### 6. MS_NMS_IOU = 0.35

**问题**: 不同尺度检测同一目标时，框位置有偏差。原 0.5 阈值在 IoU=0.45 时判定为不同目标（漏合并），导致 FastSAM 对同一目标重复分割。

**方案**: 降到 0.35，容忍跨尺度漂移。0.35 远低于不同目标间的正常 IoU（通常 <0.1），不会误合并。

### 7. GPU 信息日志

**问题**: `total_mem` vs `total_memory` 属性名在不同 PyTorch 版本中不一致，导致 `AttributeError`。

**方案**: 用 `hasattr` 兼容两种属性名，同时兼容 CPU-only 环境、处理 CUDA 版本缺失等情况。

### 8. CLIP 权重离线缓存

**问题**: `set_classes()` 内部调用 `clip.load("ViT-B/32")`，默认从网上下载 338MB 权重。换电脑或首次运行必触发下载。

**方案**: 启动时 `_ensure_local_clip()` 检查 `~/.cache/clip/ViT-B-32.pt`：
- 存在 → 跳过
- 不存在 + `model/clip/` 有 → 复制到默认缓存
- 都不存在 → 一次性下载到 `model/clip/`，再复制到缓存

### 9-10. 训练曲线图 + 总结日志

训练结束后自动输出：

```
阶段一训练完成 | 耗时: XX min
  best.pt:  ✓
  last.pt:  ✓
  总 epochs:     53/100
  mAP50:          0.686
  mAP50-95:       0.572
  Precision:      0.791
  Recall:         0.644
  train box/cls:  0.587 / 0.555
  val box/cls:    0.748 / 0.722
  ⚠ 早停触发于 epoch 53 (patience=10)
```

同时生成 `training_curves_v2.png` (6 面板)：
1. Train Loss (box/cls/dfl)
2. Val Loss (box/cls/dfl)
3. mAP50 + mAP50-95
4. Precision + Recall
5. Learning Rate
6. 文字摘要 (best epoch, 早停判断, 收敛分析)

---

## 三、文件对应关系

| 文件 | 作用 |
|------|------|
| `src/train/world/config_world_v2.py` | 全部可调参数 |
| `src/train/world/world_train_v2.py` | 训练+评估主脚本 |
| `model/world/yolov8s-worldv2.pt` | YOLO-World v2 检测权重 (25MB) |
| `model/world/FastSAM-x.pt` | FastSAM 分割权重 (138MB) |
| `model/clip/ViT-B-32.pt` | CLIP 文本编码器权重 (338MB) |
| `test/yolo_world_dataset_v2/` | v2 独立数据集目录 |
| `test/train_output/yoloworld_fastsam_v2/` | 训练输出 (weights + results + curves) |

## 四、运行命令

```bash
/d/anconda3/envs/rayton/python.exe src/train/world/world_train_v2.py
```

## 五、v1 → v2 参数对照

| 参数 | v1 | v2 |
|------|----|----|
| `IMG_SIZE` | 800 | 800 |
| `BATCH` | 16 | 16 |
| `EPOCHS` | 100 | 100 |
| `LR0` | 2e-4 | 2e-4 |
| `PATIENCE` | 10 | 10 |
| `cls` | (default 0.5) | 1.0 |
| `box` | (default 7.5) | 10.0 |
| `label_smoothing` | (default 0.0) | 0.05 |
| `computer` 阈值 | 0.55 | 0.40 |
| `computer` 过采样 | 2× | 5× (独立阶段) |
| `animal` 过采样 | 3× | 3× (解耦) |
| 多尺度类 | 无 | person/tree/animal |
| 多尺度因子 | — | [0.8, 1.0, 1.25, 1.5] |
| 数据集目录 | `yolo_world_dataset` | `yolo_world_dataset_v2` |
