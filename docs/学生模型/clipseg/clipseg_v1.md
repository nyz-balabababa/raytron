# CLIPSeg 蒸馏实验技术文档

## 一、实验定位

本实验是 SAM3 → 学生模型路线的 **第二个实验**：通过伪标签监督学习，用 CLIPSeg（开放词汇分割模型）验证"文本 prompt 进入模型后，轻量 backbone 能否学会 SAM3 的语义定位能力"。严格来说这不是知识蒸馏——只学了 SAM3 的 mask 输出（硬标签），但功能上实现了等价替代。

与 YOLO 实验的关键区别：

| | YOLOv11-seg | CLIPSeg |
|---|---|---|
| 输入 | 图像 | 图像 + 文本 prompt |
| 输出 | 6 通道 mask（固定类别） | 单通道 mask（任意文本） |
| 文本参与 | 不进模型，外部 class_id 映射 | 进模型，CLIP text encoder 编码 |
| 开放词汇 | 不支持 | 支持（CLIP 在 40 万+ 语义上预训练） |
| 蒸馏意义 | 验证伪标签可用性 | 验证 text→mask 对齐能力 |

### 验证目标

| 目标 | 方法 |
|------|------|
| **文本条件分割** | 图像 + "car" → mask，同图换 "vehicle" 应输出相似结果 |
| **prompt 泛化** | 训练时 50% 概率替换 prompt，测试原始 prompt 能否泛化 |
| **IR 域视觉编码** | CLIP 视觉编码器 lr=1e-5 微调 vs 冻结 |
| **空掩码** | 保留全零 mask 样本，模型需学会"找不到就输出全零" |
| **噪声鲁棒** | 逐样本置信度加权 + loss weight floor + warmup |
| **类别不平衡** | animal 3×、computer 2× 过采样 + LOSS_WEIGHT_FLOOR=0.7 |

---

## 二、模型架构

```
输入: 红外图像 (H×W×3) + 文本 prompt (如 "car")

  ┌──────────────────┐     ┌──────────────────┐
  │ CLIP Visual       │     │ CLIP Text         │
  │ Encoder (ViT)     │     │ Encoder           │
  │ 微调 lr=1e-5      │     │ 冻结 lr=0         │
  └───────┬──────────┘     └───────┬──────────┘
          │ visual features        │ text embedding
          └────────┬───────────────┘
                   ▼
          ┌──────────────────┐
          │ FiLM-Conditioned  │
          │ Decoder           │
          │ 训练 lr=1e-4      │
          └───────┬──────────┘
                  ▼
          单通道 mask (H×W×1)
          0 = 背景, 1 = 目标
```

- **CLIP 视觉编码器**：ViT-B/16，微调适应红外灰度域
- **CLIP 文本编码器**：冻结以保留 40 万+ 预训练语义空间
- **FiLM 解码器**：用文本 embedding 调节视觉特征，逐层上采样输出 mask

模型来源：[CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)，权重本地 `model/clipseg-rd64-refined/`。

---

## 三、数据管线

### 3.1 数据源

| 项目 | 值 | 路径 |
|------|-----|------|
| 训练集 | 39,104 张 | `train_list.txt` |
| 验证集 | 4,524 张 | `val_list.txt` |
| 训练伪标签 | SAM3 Json1 | `pred_train_tasks.json` |
| 验证伪标签 | SAM3 Json1 | `pred_val_tasks1.json` |
| 类别 | 6 类 | person/car/building/tree/animal/computer |

### 3.2 分级置信度过滤

| 类别 | 阈值 | 理由 |
|------|------|------|
| person | 0.70 | p25=0.705 |
| car | 0.70 | p25=0.848 |
| building | 0.70 | p25=0.731 |
| tree | 0.60 | p25=0.628，降低阈值保留样本 |
| animal | 0.60 | p25=0.653，稀有类降低门槛 |
| computer | 0.55 | 仅 151 样本，能用的都保留 |

### 3.3 图像预处理管线

`__getitem__` 中的处理顺序：

```
1. cv2.imread 灰度读取
2. 反色统一：黑热（热目标=暗）→ 白热（热目标=亮）
   - 文件名含 "blackHot" → 强制反色 (255 - gray)
   - 直方图偏度 < -0.3（左偏）→ 反色
   - 均值 > 200 且非 "vis" → 兜底反色
3. 等比例缩放：长边缩放到 1024
4. padding：补零到 1024×1024 正方形
5. GRAY → RGB：灰度复制为 3 通道
6. 水平翻转：50% 概率
7. [0,255] → [0,1] 归一化
8. CLIP 标准化：(x - mean) / std
```

### 3.4 RLE 掩码缓存

RLE 只在首次运行时解码一次，结果存为 PNG 到磁盘（`test/train_output/.mask_cache/`）。PNG 对二值 mask 压缩率极高（512×640 约 2-5KB），训练集 92K 张总约 300MB。

后续每次 `__getitem__` 用 `cv2.imread(PNG)` 加载，C 实现微秒级，比 RLE 字符串解析+逐像素重建快 50-100 倍。

### 3.5 样本构成

每张图每个命中 prompt 独立成一个样本：

```
img_001 + "car"      → mask_car      (1 条样本)
img_001 + "person"   → mask_person   (1 条样本)
img_001 + "tree"     → mask_tree     (1 条样本)
```

训练集约 95K 条样本，验证集约 10K 条。

### 3.6 类别过采样（仅训练集）

| 类别 | 倍数 | 原始样本 | 过采样后 |
|------|------|---------|---------|
| animal | 3× | ~1,500 | ~4,500 |
| computer | 2× | ~150 | ~300 |
| 其余 4 类 | 1× | ~73,000 | ~73,000 |

### 3.7 空掩码处理

SAM3 未找到目标时 RLE 解码为全零 mask。这些样本保留在数据集中——模型需要学习"找不到就输出全零"，这对 zero-shot 场景至关重要。

---

## 四、逐样本置信度加权 Loss

### 4.1 设计思路

不同类别的 SAM3 伪标签质量差异大：car avg_score=0.91，tree avg_score=0.72。对低质量类别，降低其 loss 贡献可抑制噪声，同时保留学习信号。

### 4.2 实现

```python
w = max(score, LOSS_WEIGHT_FLOOR)  # score ∈ [0.55, 1.0], floor = 0.7

loss = BCE_WEIGHT × mean(w × BCE_per_sample)
     + DICE_WEIGHT × mean(w × Dice_per_sample)
```

### 4.3 各类别实际权重

| 类别 | avg score | 有效权重 | 效果 |
|------|-----------|---------|------|
| car | 0.915 | ~0.92 | 几乎满权，监督信号最强 |
| building | 0.841 | ~0.84 | 正常 |
| person | 0.811 | ~0.81 | 正常 |
| animal | 0.812 | ~0.81 | 正常，配合 3× 过采样 |
| tree | 0.723 | ~0.72~0.78 | 低分自动降权，噪声抑制 |
| computer | 0.736 | ~0.74~0.79 | floor 保护，最小权重 ≥ 0.7 |

### 4.4 LOSS_WEIGHT_FLOOR

设为 0.7，确保 computer/tree 的弱信号至少保留 70% 的梯度贡献，不会因降权过度而学不动。

---

## 五、动态 Prompt 增强

训练时 50% 概率将原始 prompt 替换为同义描述，让解码器学会不同措辞对应同一视觉模式。

| 原始 | 候选 |
|------|------|
| person | "person, human, people" / "a person walking or standing" / "people in the scene" |
| car | "cars, vehicles, trucks, or any automobiles" / "a car or vehicle on the road" / "vehicles including cars and trucks" |
| building | "buildings, houses, or structures" / "any building or architectural structure" / "houses and buildings" |
| tree | "trees, plants, bushes, or any vegetation" / "trees and vegetation" / "plants, bushes, and trees" |
| animal | "animal, wildlife" / "any animal or wildlife creature" / "animals in the wild" |
| computer | **不增强**（样本太少） |

---

## 六、完整配置参数

```python
# ── 数据 ──
CLASSES = ["person", "car", "building", "tree", "animal", "computer"]
CONF_FILTER = {person:0.70, car:0.70, building:0.70, tree:0.60, animal:0.60, computer:0.55}
RARE_OVERSAMPLE = {animal:3, computer:2}

# ── 训练 ──
IMG_SIZE = 1024          # 等比例缩放 + pad
BATCH = 4
EPOCHS = 200             # 单阶段
SEED = 42

# ── 优化器 ──
DECODER_LR = 1e-4        # FiLM 解码器
BACKBONE_LR = 1e-5       # CLIP 视觉编码器（低 10×）
WEIGHT_DECAY = 0.03
LRF = 0.01               # 最终 lr = lr0 × LRF
WARMUP_EPOCHS = 2        # 前 2 epoch 线性 warmup
WARMUP_START_FACTOR = 0.01

# ── Loss ──
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
LOSS_WEIGHT_FLOOR = 0.7  # 置信度加权下限

# ── 数据增强 ──
HFLIP_PROB = 0.5
GRAY2RGB = True
PROMPT_AUG_PROB = 0.5
```

### 差分学习率策略

| 模块 | 学习率 | 策略 |
|------|--------|------|
| FiLM 解码器 | 1e-4 | 全量训练 |
| CLIP 视觉编码器 | 1e-5 | 微调适应红外域 |
| CLIP 文本编码器 | **0（冻结）** | 保留预训练语义空间 |

---

## 七、训练策略

### 7.1 单阶段 + Warmup + Cosine

```
200 epoch，单次训练:

Epoch 1-2:   线性 warmup，lr 从 1e-6 → 1e-4
Epoch 3-200: 余弦退火，lr 从 1e-4 → 1e-6

总 steps = 200 × len(train_loader)
warmup steps = 2 × len(train_loader)

         lr
    1e-4 ┤        ╱╲
         │       ╱    ╲
         │      ╱      ╲___
    1e-6 ┤─────╱            ╲___
         └──────────────────────→ epoch
         0   2                 200
```

不用两阶段——YOLO 的分阶段是为了开关 `copy_paste`，CLIPSeg 无此概念。warmup 前 2 epoch 稳定训练初期，余弦退火自动衰减。

### 7.2 断点续训

每 epoch 保存完整训练状态到 `last.pt`：

```python
checkpoint = {
    "epoch": N,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),  # LambdaLR 内部 step 计数器
    "history": {"train_loss": [...], "val_miou": [...], ...},
    "best_miou": 0.523,
}
```

再次运行自动检测并恢复全部状态，scheduler 从精确断点继续（warmup 不会重复执行，LR 曲线连续）。

---

## 八、验证与输出

### 8.1 每 epoch 输出

| 指标 | 计算方式 |
|------|---------|
| Training Loss | 逐样本置信度加权 BCE + Dice 均值 |
| Val mIoU | 验证集平均 IoU |
| Val Dice | 验证集平均 Dice |
| Per-class IoU | 6 类各自的 IoU |
| Learning Rate | 当前 lr |
| Sample Overlay | 验证图 + GT(红) + Pred(蓝) 叠加 |

### 8.2 实时 results.png（6 面板）

**1. Training Loss（蓝色）** — 判断收敛，下降太慢/震荡/一直不降各对应不同问题

**2. Val mIoU（绿色）** — 核心分割精度指标，越接近 1 越好。loss 降但 mIoU 降 → 过拟合噪声

**3. Val Dice（品红）** — 与 mIoU 相关但更鲁棒，对类别不平衡不敏感。两者背离说明空掩码占比异常

**4. Per-Class IoU（6 色）** — 定位弱类：全部低→容量不够；某类低→不平衡/噪声/增强问题

**5. Learning Rate（红色）** — 确认 warmup+cosine 是否符合预期，LR 接近 0 但 loss 仍降→可加 epoch

**6. Sample Overlay（GT红+Pred蓝）** — 红蓝重合好→准确；只有红→漏检；只有蓝→误检

### 8.3 输出目录

```
test/train_output/
├── clipseg_v1.log                          ← 文本日志（含转换统计、训练耗时、指标）
├── .mask_cache/                            ← RLE→PNG 缓存（~300MB，二次运行免解码）
└── clipseg_v1/
    ├── best.pt / last.pt                   ← 权重（含完整续训状态）
    ├── results.csv                         ← 每 epoch 全指标
    └── results.png                         ← 6 面板实时图表
```

---

## 九、后续改进方向

1. **LoRA 微调 CLIP**：用 LoRA 替代全量微调视觉编码器，减小过拟合风险、节省显存
2. **对比学习辅助**：加 text-image contrastive loss，拉近匹配图文对、推远不匹配的
3. **多尺度推理**：测试时用 [512, 768, 1024] 多分辨率融合
4. **更多 prompt 候选**：扩充增强词表的多样性（中文、不同粒度描述）
5. **混合精度训练**：AMP 可加速 ~30%，1024 下显存从 ~10GB 降到 ~7GB

---

## 十、与 YOLO 实验对比

| 维度 | 预计 CLIPSeg 优于 YOLO | 预计 CLIPSeg 劣于 YOLO |
|------|----------------------|----------------------|
| 开放词汇泛化 | prompt 增强后同义词泛化明显 | — |
| 小目标 | — | YOLO FPN 天然多尺度 |
| 推理速度 | — | 双编码器 vs 单次前向 |
| 训练稳定性 | — | Transformer 微调更敏感 |
| 部署复杂度 | 标准 HuggingFace | Ultralytics 一键导出 |

关键验证点：**CLIPSeg mIoU 在非增强 prompt 上是否 ≥ YOLO mIoU**。达到则文本条件蒸馏成立，开放词汇路线可行。

---

## 十一、运行方式

```bash
cd src/train && python clipseg_train.py
```

依赖：
- CUDA 版 PyTorch（手动装：`pip install torch --index-url https://download.pytorch.org/whl/cu128`）
- `transformers` ≥ 4.46、`matplotlib`
- 模型权重：`model/clipseg-rd64-refined/pytorch_model.bin`（已下载，603MB）
