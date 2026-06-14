## CLIPSeg + Tiny Refine Head 精度增强线

### 1. 路线定位

`CLIPSeg + Tiny Refine Head` 不是替代当前 `B0-hard / B0+D` 主线，而是作为一条低风险精度增强线，主要目标有三个：

```text
1. 提升 CLIPSeg-11 自身的边界、小目标、细长物体分割质量
2. 作为更强 teacher，重新生成 B0+D 的 teacher soft cache
3. 在最终推理中作为未知 prompt / 低置信度 prompt 的兜底模型
```

这条线的优势是：

```text
不换 CLIPSeg 主干
不破坏已有 CLIPSeg 训练流程
只增加一个很小的修补头
训练和工程风险低
对 fence / pole_light / window / door / trash can / motorcycle / animal 这些细类更有针对性
```

最终关系：

```text
CLIPSeg-11 baseline
    ↓
CLIPSeg + Tiny Refine Head
    ↓
生成更强 teacher soft cache
    ↓
SegFormer-B0 + D 蒸馏
```

所以它的核心价值不是单独冲速度，而是提高 teacher 质量和兜底模型质量。

------

### 2. 适用类别

当前仍然使用 v4 的 11 类：

```text
person
car
building
tree
animal
trash can
window
door
fence
pole_light
motorcycle
```

`computer` 继续过滤，不参与训练和评估。

重点关注的收益类别：

```text
window
door
fence
pole_light
trash can
motorcycle
animal
```

这些类别通常存在小目标、细长结构、边界不清、低对比度等问题，正好适合 tiny refine head 修边界和局部细节。

------

### 3. 模型结构

基础模型仍然是：

```python
CLIPSegForImageSegmentation
```

在 CLIPSeg 原始输出后增加一个轻量 refine head。

整体结构：

```text
image + text_prompt
    ↓
CLIPSeg backbone + decoder
    ↓
coarse_logit
    ↓
Tiny Refine Head
    ↓
refined_logit
```

推荐第一版不要改 CLIPSeg 内部结构，只做最小侵入式设计：

```text
输入：
    coarse_logit：CLIPSeg 原始预测 logit
    gray_image：红外灰度图
    edge_map：可选，Sobel / Laplacian 边缘图

输出：
    residual_logit：局部修补残差

最终：
    refined_logit = coarse_logit + residual_scale * residual_logit
```

第一版 refine head 建议：

```text
Conv 3x3, in_channels=2 or 3, out_channels=32
GroupNorm / BatchNorm
GELU / ReLU
Conv 3x3, 32 → 32
GroupNorm / BatchNorm
GELU / ReLU
Conv 1x1, 32 → 1
```

输入通道建议：

```text
V1：coarse_logit + gray_image
V2：coarse_logit + gray_image + edge_map
```

先做 V1，不要一开始接 CLIPSeg 多层 hidden states，也不要一开始改 FPN decoder。这样风险最低，最容易快速验证。

------

### 4. 冻结策略

保持 CLIPSeg 原有稳定训练思路：

```text
text encoder：始终冻结
vision backbone：默认冻结，后期可低学习率微调
CLIPSeg decoder：可训练
tiny refine head：重点训练
```

推荐训练阶段：

#### Stage A：只训练 tiny refine head

```text
epoch：3~5
text encoder：freeze
vision backbone：freeze
CLIPSeg decoder：freeze
tiny refine head：train
lr_refine_head：3e-4
```

目的：

```text
验证 refine head 是否能在不破坏原 CLIPSeg 的情况下修边界
```

#### Stage B：训练 decoder + refine head

```text
epoch：5~10
text encoder：freeze
vision backbone：freeze
CLIPSeg decoder：train
tiny refine head：train
lr_decoder：1e-4
lr_refine_head：3e-4
```

目的：

```text
让 decoder 和 refine head 共同适应 11 类红外伪标签
```

#### Stage C：可选低学习率微调 vision backbone

```text
epoch：2~5
text encoder：freeze
vision backbone：train with low lr
CLIPSeg decoder：train
tiny refine head：train
lr_backbone：1e-5
lr_decoder：1e-4
lr_refine_head：3e-4
```

只有在 Stage B 收益明确时才开 Stage C。否则不要动视觉 backbone，防止过拟合和训练不稳定。

------

### 5. 数据与预处理

沿用当前 v4 数据流程：

```text
输入来源：普通 pred_json 或 manifest
样本形式：image + class_name/text_prompt + binary_mask
类别过滤：过滤 computer，过滤不在 11 类内的 prompt
空 mask：默认过滤
负样本：可保留少量训练负样本，验证集不混入负样本
稀有类：继续支持 rare oversample
```

空间对齐继续使用统一画布：

```text
输入图像：768 x 768
hard mask：768 x 768
coarse_logit：上采样到 768 x 768
refined_logit：768 x 768
```

红外预处理继续沿用：

```text
灰度化
伪彩 / 黑热自动判断
低对比度 CLAHE
高噪声降噪
模糊图锐化
```

注意：refine head 的 `gray_image / edge_map` 必须和 hard mask 完全对齐，不能使用未经 resize/pad 的原图。

------

### 6. Loss 设计

主 loss 只对 `refined_logit` 计算：

```python
main_loss = 1.0 * BCEWithLogitsLoss(refined_logit, target)
          + 0.5 * DiceLoss(refined_logit, target)
          + 0.5 * RareClassFocalLoss(refined_logit, target)
```

可选辅助 loss 对 CLIPSeg 原始 `coarse_logit` 计算：

```python
aux_loss = 0.3 * BCEWithLogitsLoss(coarse_logit, target)
```

总 loss：

```python
loss = main_loss + aux_loss
```

如果使用边界弱监督：

```text
hard mask 边界带不参与 hard loss
边界区域由 refine head 自己学习 soft transition
减少伪标签边缘噪声对训练的伤害
```

稀有类 focal 继续重点照顾：

```text
animal
trash can
window
door
pole_light
motorcycle
```

但不要一开始把 rare class 权重拉太高，防止主类 recall 掉、mask 变碎。

------

### 7. 指标与保存

验证集输出以下指标：

```text
all11_mIoU
old5_mIoU
rare_mIoU
Dice
per-class IoU
per-class Dice
pos_mIoU
```

额外建议记录：

```text
small_object_mIoU
boundary_iou
avg_infer_ms_per_prompt
avg_infer_ms_per_image
```

checkpoint 保存：

```text
last.pt
best_all11.pt
best_old5.pt
best_rare.pt
best_refine_gain.pt
results.csv
results.png
```

checkpoint 元信息必须包含：

```python
{
    "classes": CLASSES,
    "num_classes": 11,
    "img_size": 768,
    "model_type": "clipseg_tiny_refine",
    "base_model_dir": base_model_dir,
    "base_checkpoint": base_checkpoint,
    "val_thresholds": val_thresholds,
    "run_name": run_name,
}
```

------

### 8. 与 B0+D 的衔接

当 `CLIPSeg + Tiny Refine Head` 在验证集上优于原始 CLIPSeg-11 后，用它重新生成 teacher soft cache。

cache 内容保持和 v4 一致：

```text
文件名：
    {ann_id}__{class_name}.npy
    或 {image_stem}__{class_name}.npy

内容：
    refined_prob map
    shape = 768 x 768
    dtype = float16
    value range = [0, 1]
```

生成方式：

```python
refined_prob = sigmoid(refined_logit)
np.save(cache_path, refined_prob.astype(np.float16))
```

然后启动新的蒸馏实验：

```text
B0+D 原 teacher cache
vs
B0+D refine-teacher cache
```

核心观察：

```text
all11 是否提升
old5 是否不掉
rare 是否明显提升
teacher cache hit_rate 是否正常
```

如果 refine teacher 对 rare 有提升，但 old5 掉分，可以降低 `DISTILL_LAMBDA_RARE` 或只对部分 rare class 使用 refine teacher。

------

### 9. 最终推理使用方式

最终推理不建议直接把 CLIPSeg-refine 当唯一主模型，除非它精度明显高于 B0+D。

推荐最终路由：

```python
if prompt in known_11_classes:
    mask = segformer_b0_output[class_id]

    if confidence_low or area_abnormal:
        mask = clipseg_refine_fallback(image, prompt)
else:
    mask = clipseg_refine_fallback(image, prompt)
```

CLIPSeg-refine fallback 必须做推理加速：

```text
按 image_path 分组
同一张图只读一次
同一张图多个 prompt batch 推理
11 类 prompt 文本 embedding 预缓存
fp16 autocast
每个 ann_id 都输出结果
空结果也输出空 RLE
COCO RLE counts 转字符串
```

------

### 10. 消融实验顺序

建议按以下顺序跑：

#### V0：原始 CLIPSeg-11 baseline

```text
不加 refine head
只记录当前验证指标和速度
```

#### V1：Refine-only

```text
冻结 CLIPSeg 全部参数
只训练 tiny refine head
epoch：3~5
```

判断 refine head 是否有基础收益。

#### V2：Decoder + Refine

```text
冻结 text encoder 和 vision backbone
训练 decoder + refine head
epoch：5~10
```

这是最推荐作为主结果的版本。

#### V3：Decoder + Refine + Edge Map

```text
在 refine head 输入中加入 edge_map
观察 fence / pole_light / window / door 是否提升
```

#### V4：Rare Prompt Ensemble

```text
只对 rare class 推理时启用 prompt ensemble
例如 pole_light 使用 street light / lamp / pole light
```

这个只作为推理 trick，不作为第一阶段主实验。

------

### 11. 成功标准

保留该路线的条件：

```text
CLIPSeg-refine 比 CLIPSeg-11：
    all11_mIoU 提升 >= 0.5
    或 rare_mIoU 提升 >= 1.5
    且 old5_mIoU 不明显下降

B0+D refine-teacher 比 B0+D 原 teacher：
    all11_mIoU 不下降
    rare_mIoU 有提升
    old5_mIoU 基本稳定
```

放弃该路线的条件：

```text
连续 3 个 epoch 验证集不如原始 CLIPSeg
old5 明显掉分
mask 明显变碎
rare recall 提升但误检大幅增加
训练速度/显存开销明显影响主线实验
```

------

### 12. 当前建议结论

这条线优先做最小版本：

```text
CLIPSeg 原模型
+ gray_image
+ coarse_logit
+ 3 层 tiny conv refine head
+ freeze text encoder
+ freeze vision backbone
+ train decoder + refine head
```

不要一开始做 FPN，不要一开始接 EfficientSAM，不要一开始全量多尺度推理。

最推荐的落地路径：

```text
先跑 V1 refine-only
再跑 V2 decoder + refine
如果 CLIPSeg-refine 本身有效
再用它生成 teacher cache
最后对比 B0+D 原 teacher 和 B0+D refine teacher
```