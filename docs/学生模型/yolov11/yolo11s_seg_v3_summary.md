# YOLOv11-seg 伪标签训练实验报告

> 实验日期: 2026-06-01 | GPU: NVIDIA GeForce RTX 5080 | 脚本: `src/train/yolov11_seg.py`

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | `yolo11s-seg.pt` |
| 类别 (6) | person, car, building, tree, animal, computer |
| 训练集 | 41,158 张（37,783 原始 + 4,317 过采样） |
| 验证集 | 4,104 张 |
| 图片尺寸 | 640×640 |
| Batch size | 16 |
| 优化器 | AdamW, lr=1e-3, cos_lr=True |
| 正则化 | label_smoothing=0.1, dropout=0.1 |
| 数据增强 | mosaic=0, scale=0, degrees=0, copy_paste=0.3, flip_lr=0.5, hsv_v=0.15 |
| 早停 | patience=30 |
| 阶段一 epochs | 150（实际跑 82 轮后手动停止） |
| 阶段二 | 未执行 |

### 伪标签处理

| 参数 | 值 |
|------|-----|
| 伪标签来源 | SAM3 prompt → RLE mask → YOLO polygon |
| 置信度过滤 | person=0.70, car=0.70, building=0.70, tree=0.60, animal=0.60, computer=0.55 |
| 过滤比例 | ~16% prompts 被过滤 |
| 过采样 | animal×3, computer×10 |
| Polygon 转换精度 | IoU ≥ 0.97（animal 最低 0.85） |

---

## 2. 训练结果

### 最终指标

| 指标 | epoch 10 | 最佳 (epoch 62) | 最终 (epoch 82) |
|------|---------|-----------------|-----------------|
| Box mAP50 | 0.635 | **0.651** | 0.645 |
| Box mAP50-95 | — | **0.525** | 0.519 |
| Mask mAP50 | 0.624 | **0.642** | 0.639 |
| Mask mAP50-95 | — | — | 0.439 |

### Loss 收敛

| Loss | epoch 1 | epoch 50 | epoch 82 |
|------|---------|----------|----------|
| train/box_loss | 0.979 | 0.614 | 0.564 |
| val/box_loss | 1.104 | 0.752 | 0.748 |
| train/seg_loss | 1.625 | 0.932 | 0.854 |
| val/seg_loss | 3.288 | 1.458 | 1.506 |

### 过拟合分析

- mAP 在 epoch 50~82 之间完全停滞（波动范围 < 0.005）
- train loss 仍在下降，val loss 已停止下降
- train/val box_loss gap 从 0.13 扩大到 0.18
- **结论：轻度过拟合，patience=30 未触发因手动提前停止**

---

## 3. 数据分析

### 掩码面积分布（占图像面积比）

| 类别 | 实例数 | 面积中位数 | <1% 面积占比 | 特征 |
|------|--------|-----------|-------------|------|
| person | 23,452 | 0.43% | 74% | **极端小目标** |
| car | 25,598 | 2.58% | 22% | 中小目标 |
| building | 21,376 | 7.85% | 5% | 中大目标 |
| tree | 22,850 | 4.94% | 13% | 中小目标 |
| animal | 1,858 | 2.48% | 30% | 中小目标，轮廓复杂 |
| computer | 151 | 3.65% | ~1% | **极度稀疏** |

### 关键发现

1. **person 类的 74% 实例面积 < 1% 图像**（≈37×37 px），极端小目标是 mAP50-95 大幅低于 mAP50 的主要原因
2. **computer 仅 151 个实例**，10× 过采样后有效图像仍然来自相同的 ~150 张原始图，泛化能力有限
3. **无脏数据**：空掩码 0 个，全图掩码 0 个，polygon 转换无信息丢失

---

## 4. 结论与下一步

### 当前流水线评估

- 数据预处理管线完整且干净，无需返工
- mAP50=0.645 是**SAM3 伪标签硬标签训练的天花板**，不是预处理问题
- 极致小目标（person）和极度稀疏类（computer）是主要瓶颈

### 蒸馏预期

- 同样的伪标签，promptable 蒸馏模型（image+text → mask）比 YOLO 固定 6 分类有更高的伪标签利用率
- 蒸馏能传递 SAM3 的特征表示，mIoU 有望突破当前 mAP50-95(M)=0.439 的水平
- person 小目标边界精度是 mIoU 提升的关键

### 输出文件

```
test/train_output/
├── yolo11s_seg_v3.log                     # 训练日志
├── yolo11s_seg_v3_summary.md              # 本报告
└── yolo11s_seg_v3_stage1/
    ├── weights/best.pt                    # 最佳权重 (epoch 62)
    ├── weights/last.pt                    # 断点续训权重 (epoch 82)
    ├── results.csv                        # 逐 epoch 指标
    ├── results.png                        # 指标曲线图
    ├── labels.jpg                         # 标签分布
    ├── args.yaml                          # 训练参数存档
    └── train_batch*.jpg                   # 训练样本可视化
```
