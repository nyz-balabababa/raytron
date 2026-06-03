# YOLO-World + FastSAM 两阶段实验技术文档

## 一、实验定位

本实验是 SAM3 → 学生模型蒸馏路线的 **第三个对比实验**：用 YOLO-World v2（开放词汇检测）+ FastSAM（零样本分割）的两阶段管线，验证"检测 + 分割解耦"方案在红外域的效果。

与 CLIPSeg 的关键区别：

| | CLIPSeg | YOLO-World + FastSAM |
|---|---|---|
| 架构 | 单模型端到端 | 两阶段解耦 |
| 阶段一 | — | YOLO-World v2 检测（bbox） |
| 阶段二 | — | FastSAM 逐框分割 → 合并 mask |
| 文本输入 | CLIP text encoder 进 FiLM 解码器 | YOLO-World 内置 CLIP text encoder |
| 训练 | 解码器 + 视觉编码器微调 | 仅 YOLO-World 检测（FastSAM 零样本） |
| 参数 | ~150M（~30M 可训练） | YOLO ~80M + FastSAM ~22M = ~102M |
| 推理速度 | ~50ms | ~60-80ms（两阶段） |

### 验证目标

| 目标 | 方法 |
|------|------|
| **红外域检测迁移** | YOLO-World v2 在红外灰度图上微调，验证 bbox 检测精度 |
| **FastSAM 零样本红外分割** | 无需训练，直接用预训练 FastSAM 对 bbox 区域分割 |
| **两阶段 mask 质量** | 与 SAM3 伪标签比对 per-class IoU 和 overall mIoU |
| **同义词泛化** | 推理时用 "vehicle"→"car" 映射测试开放词汇泛化 |
| **与 CLIPSeg 对照** | 同等数据/训练预算下，两阶段 vs 端到端的 mask mIoU |

---

## 二、模型架构

```
推理:
  图像 + "car" → YOLO-World → [box1, box2, box3]  ← CLIP text encoder 语义匹配
                   │
         ┌────────┼────────┐
         ▼        ▼        ▼
       crop1   crop2    crop3
         │        │        │
     FastSAM  FastSAM  FastSAM  ← 零样本，不需要训练
         │        │        │
         ▼        ▼        ▼
      mask1   mask2   mask3  → 合并(OR) → 最终并集 mask

训练:
  只需训练 YOLO-World 检测
  伪标签 RLE → findContours → 每个实例的归一化 bbox
  FastSAM 不参与训练
```

- **YOLO-World v2**：基于 YOLOv8s，内置 CLIP ViT-B/32 文本编码器。`set_classes` 将文本 prompt 编码为语义向量，检测头用语义向量指导 bbox 定位
- **FastSAM**：YOLOv8-seg 架构在 SA-1B 上训练，用 bbox 作为 prompt 做零样本分割
- 两个模型各自独立，仅推理时串联

模型权重本地 `model/world/yolov8s-worldv2.pt` + `model/world/FastSAM-x.pt`。

---

## 三、数据管线

### 3.1 数据源（与 CLIPSeg 完全一致）

| 项目 | 值 |
|------|-----|
| 训练集 | 39,104 张 `train_list.txt` |
| 验证集 | 4,524 张 `val_list.txt` |
| 训练伪标签 | `pred_train_tasks.json` |
| 验证伪标签 | `pred_val_tasks1.json` |
| 类别 | 6 类 person/car/building/tree/animal/computer |

### 3.2 分级置信度过滤（与 CLIPSeg 一致）

| 类别 | 阈值 |
|------|------|
| person/car/building | 0.70 |
| tree/animal | 0.60 |
| computer | 0.55 |

### 3.3 图像预处理（与 CLIPSeg 一致）

```
1. cv2.imread BGR → cv2.cvtColor 灰度
2. 反色统一：黑热→白热（偏度检测 + blackHot 文件名 + 均值兜底）
3. cv2.cvtColor 灰度→RGB（YOLO-World 期望 3 通道）
```

### 3.4 标签格式（与 CLIPSeg 不同：bbox vs mask）

CLIPSeg 用 RLE mask 直接训练。本实验将 RLE 转为实例级 bbox：

```
RLE → mask → findContours → 逐连通域 boundingRect
→ 归一化 YOLO 检测格式: class_id cx cy w h
```

### 3.5 过采样（与 CLIPSeg 一致）

| 类别 | 倍数 |
|------|------|
| animal | 3× |
| computer | 2× |

---

## 四、训练配置

### 4.1 阶段一：YOLO-World v2 检测训练

| 参数 | 值 | 理由 |
|------|-----|------|
| `MODEL` | yolov8s-worldv2.pt | 检测 + CLIP 文本编码器 |
| `TASK` | detect | 训检测而非分割 |
| `IMG_SIZE` | 800 | 平衡小目标与显存 |
| `BATCH` | 16 | RTX 5060 16GB 可用 |
| `EPOCHS` | 100 | 微调适配红外域 |
| `LR0` | 2e-4 | 低 LR 保护 CLIP 编码器 |
| `OPTIMIZER` | AdamW | Transformer 友好 |
| `PATIENCE` | 10 | 收敛自动停 |
| `MOSAIC/MIXUP` | 0.0 | 小目标敏感，关闭 |
| `SCALE` | 0.3 | 保留多尺度 |
| `FLIPLR` | 0.5 | 水平翻转 |
| `FLIPUD/DEGREES` | 0.0 | 红外域不适用 |

### 4.2 阶段二：FastSAM 零样本分割

| 参数 | 值 | 理由 |
|------|-----|------|
| 模型 | FastSAM-x.pt | 22M，SA-1B 预训练 |
| 训练 | 不需要 | 零样本 |
| `IMG_SIZE` | 1024 | 分割精度需要 |
| `CONF` | 0.25 | 低阈值避免漏检 |
| `IOU` | 0.7 | box 内 NMS |
| prompt 方式 | bbox | YOLO-World 输出框作为位置提示 |

---

## 五、推理管线

### 5.1 主流程

```python
# 1. YOLO-World 检测（按 prompt 切换词汇）
yolo.set_classes(["car"])
boxes = yolo.predict(image)  # → [bbox1, bbox2, ...]

# 2. FastSAM 逐框分割
for box in boxes:
    crop = image[box.y1:box.y2, box.x1:box.x2]
    masks = fastsam.predict(crop)  # → [mask1, mask2, ...]
    pred_mask |= resize(mask, box_size)

# 3. 与伪标签比对 IoU
iou = compute_iou(pred_mask, gt_mask)
```

### 5.2 同义词泛化测试

推理时不一定用训练时的类别名。用同义词映射测试泛化：

```
"vehicle" → 映射到 "car" 的伪标签 mask → 算 IoU
"pedestrian" → 映射到 "person" 的伪标签 mask → 算 IoU
```

期望：同义词的 IoU 接近原始词的 IoU，说明 CLIP 编码器确实泛化了语义。

---

## 六、与 CLIPSeg 数据处理对比

| 处理步骤 | CLIPSeg | YOLO-World + FastSAM | 一致？ |
|---------|---------|---------------------|--------|
| 反色统一 | ✓ | ✓ | ✓ |
| 分级过滤 | ✓ | ✓ | ✓ |
| 过采样 | animal 3×, computer 2× | animal 3×, computer 2× | ✓ |
| 水平翻转 | 0.5 | 0.5 | ✓ |
| 旋转/垂直翻转 | 0.0 | 0.0 | ✓ |
| 图像分辨率 | 1024 | 800 | 模型适配 |
| Epoch | 200 | 100 | 模型适配 |
| 标签格式 | RLE→mask | RLE→bbox | 模型适配 |
| 置信度加权 loss | 逐样本 score | 不支持 | 模型差异 |
| Prompt 增强 | 50% 替换 | set_classes | 模型差异 |

---

## 七、评估指标

### 7.1 检测指标（阶段一，自动记录）

Ultralytics 自动记录到 `results.csv` 和 `results.png`：
- `box_loss` / `cls_loss` / `dfl_loss`
- `mAP50(B)` / `mAP50-95(B)`
- 混淆矩阵、PR 曲线

### 7.2 两阶段 mask 指标（阶段二，自定义评估）

`two_stage_eval()` 遍历验证集，逐图跑完整两阶段管线，与 SAM3 伪标签比对：
- Per-class mask IoU
- Overall mask mIoU
- 同义词泛化 IoU（prompt 映射后与原始类伪标签比对）

---

## 八、输出目录

```
test/train_output/yoloworld_fastsam_v1/
├── weights/best.pt / last.pt        ← YOLO-World 检测器权重
├── results.csv                      ← 每 epoch 检测指标
├── results.png                      ← loss/mAP/LR 曲线
├── confusion_matrix.png             ← 6 类混淆矩阵
└── yoloworld_fastsam_v1.log         ← 文本日志（含两阶段 mask mIoU）
```

---

## 九、后续改进方向

1. **FastSAM 红外微调**：用 SAM3 伪标签对 FastSAM 做少量微调，适配红外域
2. **多尺度推理**：测试时用 [512, 800, 1024] 多分辨率融合
3. **YOLO-World + SAM 三阶段**：用 MobileSAM 替代 FastSAM，精度更高
4. **模型合并**：把 FastSAM 的 mask head 嫁接到 YOLO-World 上做成单模型
5. **训练 FastSAM**：当前完全零样本，SAM3 伪标签可以用于 FastSAM 的后训练

---

## 十、运行方式

```bash
cd src/train/world && python world_train.py
```

依赖：
- CUDA 版 PyTorch + ultralytics + openai-clip
- 权重文件 `model/world/yolov8s-worldv2.pt` + `model/world/FastSAM-x.pt`
- 伪标签 `test/prompt_test_output/train_tasks/pred_train_tasks.json`
