# ESAM-CCLIP-11cls 最后两天冲榜方案总结

当前最高分路线：

```text
ESAM-CCLIP-11cls-自创decoder
6类伪标签训练
→ 3轮11cls热启动
→ 3轮训练
→ old-balanced
→ 当前最高分 66.462
```

现在基本判断：

```text
继续同结构微调、rare-balanced、lite、old-balanced 这类小修小补，提升空间已经很小。
如果想冲 67/68，需要在不推翻最高分权重的前提下，做轻量结构/训练目标/推理策略变化。
```

之前两版 ESAM 优化方案里，核心本来就是围绕当前 best 热启动、冻结大模型、只训 decoder、prompt prototype、cache、阈值搜索、rare balance、小 refine head 这些低风险方向展开。

------

# 总体策略

最后两天不要再开大路线：

```text
不要从头训 11cls
不要解冻整个 EfficientSAM
不要重训 ChineseCLIP
不要大改 Mask2Former / FPN / UPerNet
不要大规模重新生成伪标签
不要 rare-heavy 猛训
```

应该围绕最高分 checkpoint 做：

```text
1. 保 old5
2. 小幅提 rare
3. 不破坏当前 66.462
4. 所有新模型都必须能回退到旧模型
5. 训练最多 1~2 epoch
6. 每个候选最后都重新 sweep
```

------

# 方案一：balanced partial unfreeze

## 目标

```text
old5 做锚点
rare 做增益
不让 old5 掉
但允许 rare 参与梯度
```

这条不是 old5-only，也不是 rare-heavy，而是：

```text
partial_unfreeze_balanced_safe
```

适合当前判断：

```text
榜单可能更偏 old5，但应该不是极端只看 old5。
所以训练不能只看 rare，也不能完全放弃 rare。
```

------

## 核心思路

从当前 66.462 checkpoint 出发：

```text
只小范围解冻 EfficientSAM image encoder 的最后部分
不动 ChineseCLIP text encoder
decoder 继续训练
image_lr 极低
训练 1 epoch
然后重新冻结 image encoder，再做 1 epoch recalibrate
```

流程：

```text
66.462 best
↓
partial_unfreeze_balanced_safe 1 epoch
↓
freeze image encoder
↓
balanced_recalibrate 1 epoch
↓
old5-safe + balanced sweep
↓
生成提交候选
```

------

## 推荐训练比例

```text
old5: 70% ~ 75%
rare: 20% ~ 25%
negative: 3% ~ 5%
```

不建议 rare 过采样太猛。

推荐配置：

```python
old_class_sample_ratio = 1.0
rare_class_keep_ratio = 1.0
rare_oversample = 1.2   # 最多 1.5，不要 2+
negative_sample_ratio = 0.005 ~ 0.01
negative_sample_weight = 0.05 ~ 0.10
```

如果支持 batch mix：

```python
batch_mix = {
    "old": 0.72,
    "rare": 0.23,
    "negative": 0.05,
}
```

------

## 推荐 loss 权重

rare 只比 old 略高一点，不能让 rare 接管训练方向。

```python
class_loss_weight = {
    "person": 1.00,
    "car": 1.05,
    "building": 1.05,
    "tree": 1.00,
    "animal": 0.95,

    "trash can": 1.15,
    "window": 1.10,
    "door": 1.10,
    "fence": 1.15,
    "pole_light": 1.20,
    "motorcycle": 1.15,
}
```

------

## 推荐学习率

partial unfreeze 最大风险不是训不动，而是把当前最高分特征毁掉。

```python
epochs = 1

decoder_lr = 5e-6 ~ 1e-5
image_lr = 1e-7 ~ 3e-7
text_lr = 0

weight_decay = 1e-4
grad_clip = 1.0
```

recalibrate 阶段：

```python
freeze_esam = True
freeze_chineseclip = True
train_decoder = True

epochs = 1
decoder_lr = 5e-6
rare_oversample = 1.0
negative_sample_ratio = 0.005
negative_sample_weight = 0.05
```

------

## 选择 checkpoint 的指标

不要只看 all11 mIoU。

推荐目标：

```text
score = 0.70 * old5_mIoU
      + 0.25 * rare_mIoU
      + 0.05 * all11_mIoU
      - old_drop_penalty
```

淘汰规则：

```text
old5 掉超过 0.5%：强惩罚
old5 掉超过 1.0%：基本淘汰
old5 掉超过 1.5%：直接不要
rare 涨但 old5 明显掉：不要
old5 稳住且 rare 小涨：保留
```

------

# 方案二：zero-init tiny refine head

## 目标

当前自创 decoder 太轻，表达能力可能到顶。
但最后两天不能大改模型，所以加一个**零初始化 residual refine head**。

核心要求：

```text
刚加载旧 checkpoint 时，输出必须几乎等于旧模型。
不能一上来毁掉 66.462。
```

------

## 结构

原 decoder 输出：

```text
coarse_logit
```

新增 refine head：

```text
coarse_logit
+ gray_image
+ maybe sigmoid(coarse_logit)
→ Conv 3x3
→ Conv 3x3
→ Conv 1x1
→ residual_logit
```

最终输出：

```text
final_logit = coarse_logit + residual_logit
```

关键：

```text
最后一层 Conv 1x1 权重和 bias 初始化为 0
所以初始 residual_logit ≈ 0
final_logit ≈ coarse_logit
```

这样可以安全加载最高分旧权重。

------

## 训练策略

```python
resume = "66.462 best checkpoint"

freeze_image_encoder = True
freeze_text_encoder = True

train_decoder = True
train_refine_head = True

epochs = 2
decoder_lr = 1e-5
refine_lr = 5e-5

negative_sample_ratio = 0.005 ~ 0.01
negative_sample_weight = 0.05 ~ 0.10

rare_balance_enabled = False
# 或者只开极轻 rare 权重，不要 rare-heavy
```

------

## 主要可能提升的类

```text
old5:
person 边界
car 粘连
building 边缘
tree 细碎结构
animal 小目标弱响应

rare:
window
door
fence
pole_light
trash can
motorcycle
```

这条不是靠 rare oversample，而是靠补 decoder/refine 的表达能力。

------

## 工程要求

必须兼容旧权重和旧推理：

```text
1. 保留原 FiLMFusionDecoder 字段名
2. 新增 refine_head 字段
3. 加载旧权重时 strict=False
4. refine_head 最后一层 zero init
5. checkpoint metadata 记录 use_refine_head=True
6. 推理脚本兼容：
   - 有 refine_head 权重 → 用 refine
   - 没有 refine_head 权重 → 走旧 decoder
```

------

## 风险

```text
优点：结构上限更高，有机会突破当前平台期
风险：需要改模型、训练脚本、推理脚本、权重加载
止损：如果改代码半天还没跑通，就先放弃，不要拖死主线
```

------

# 方案三：old5-safe sweep

## 目标

如果榜单相对更偏 old5，那么不能再用 all11 等权 sweep。
应该单独做 old5 加权 sweep。

当前 old5：

```text
person
car
building
tree
animal
```

目标：

```text
不训练
只调 threshold / min_area / postprocess
优先提高 old5
rare 只做 tie-break
```

------

## sweep 目标函数

不要用：

```text
score = mean(all11_iou)
```

改成：

```text
score = 0.30 * person
      + 0.25 * car
      + 0.25 * building
      + 0.15 * tree
      + 0.05 * animal
      - penalty_fp
```

或者简化：

```text
score = mean(person, car, building, tree, animal)
```

rare 类只作为并列时的参考，不主导选择。

------

## old5 threshold 搜索范围

```python
OLD5_THRESHOLDS = {
    "person":   [0.50, 0.55, 0.60, 0.65],
    "car":      [0.55, 0.60, 0.65, 0.70],
    "building": [0.55, 0.60, 0.65, 0.70],
    "tree":     [0.45, 0.50, 0.55, 0.60],
    "animal":   [0.40, 0.45, 0.50, 0.55],
}
```

------

## old5 min_area 搜索范围

```python
OLD5_MIN_AREA = {
    "person":   [16, 24, 32, 48],
    "car":      [32, 48, 64, 96],
    "building": [128, 192, 256, 384],
    "tree":     [32, 64, 96, 128],
    "animal":   [16, 24, 32, 48],
}
```

------

## rare 类处理

rare 不要 aggressive recall。

```python
RARE_THRESHOLDS_SAFE = {
    "trash can": 0.40,
    "window": 0.40,
    "door": 0.40,
    "fence": 0.35,
    "pole_light": 0.32,
    "motorcycle": 0.40,
}
RARE_MIN_AREA_SAFE = {
    "trash can": 12,
    "window": 8,
    "door": 12,
    "fence": 8,
    "pole_light": 6,
    "motorcycle": 16,
}
```

原则：

```text
rare 可以补
但不能抢 old5 区域
rare 和 old5 重叠时，old5 优先
```

------

# 方案四：old5-polish 短训

## 目标

从 66.462 checkpoint 出发，不追 rare，轻轻把 decoder 拉回 old5 最舒服的分布。

适合情况：

```text
11cls 后 old5 被 rare prompt 稍微干扰
old5 边界/置信度/粘连还有一点修正空间
```

------

## 训练配置

```python
resume = "66.462 best checkpoint"

freeze_esam = True
freeze_chineseclip = True
train_decoder = True

epochs = 1 ~ 2
lr = 5e-6 ~ 1e-5

negative_sample_ratio = 0.005 ~ 0.01
negative_sample_weight = 0.05

rare_oversample = False
rare_loss_weight = 0.0 ~ 0.3
old_loss_weight = 1.0
```

------

## 数据策略

```text
person / car / building / tree / animal：
    全保留或优先高质量样本

rare 类：
    少量保留
    不过采样
    不加大 loss

negative：
    极低比例
```

------

## 风险和止损

```text
只跑 1 epoch 先看。
如果 old5 有提升，再跑第 2 epoch。
不要一口气跑 3~5 epoch。
```

淘汰规则：

```text
old5 不涨：不要继续
old5 涨但 rare 崩太多：看榜单判断
old5 掉：直接放弃
```

------

# 方案五：old5 / 11cls 双模型 class-router

## 目标

不同 checkpoint 可能各有优势：

```text
old6 或 early checkpoint：
    old5 更稳

11cls checkpoint：
    rare 更强
```

不要强行一个模型全包，可以推理时融合。

------

## 推理策略

```text
person / car / building / tree / animal：
    用 old6-best 或 old-balanced-best 输出

rare 类：
    用 11cls-best 输出

最后做 overlap 处理：
    old5 优先级高于 rare
```

优先级示例：

```python
PRIORITY = [
    "building",
    "car",
    "person",
    "tree",
    "animal",
    "motorcycle",
    "fence",
    "door",
    "window",
    "trash can",
    "pole_light",
]
```

------

## overlap 规则

```python
if iou(rare_mask, old_mask) > 0.30:
    drop_or_clip_rare()

if overlap_area / rare_area > 0.40:
    drop_or_clip_rare()
```

原则：

```text
rare 只能补 old5 没覆盖的区域
不能抢 old5
```

------

## 优缺点

```text
优点：
    不需要训练
    可以利用已有 checkpoint 的不同优势
    符合榜单偏 old5 的猜测

缺点：
    推理更慢
    融合规则需要小心
    需要确认官方速度是否还能接受
```

------

# 方案六：rare-only DINO → ROI ESAM hybrid

## 目标

这条不是训练线，是推理范式变化线。

不建议全类启用 DINO，只对 rare 类启用：

```text
old5：
    仍用当前最高分 ESAM 全图推理

rare：
    DINO proposal
    → ROI crop
    → ESAM 在局部区域分割
    → 回贴原图
```

------

## 适合启用的 rare 类

```text
trash can
pole_light
motorcycle
door
window
fence
```

可选：

```text
animal
```

------

## 推荐参数

```python
dino_box_threshold = 0.12 ~ 0.18
dino_text_threshold = 0.20 ~ 0.25
box_expand_ratio = 0.15 ~ 0.25
max_boxes_per_prompt = 3 ~ 8
fallback_full_image = True
```

------

## 规则

```text
DINO 有框：
    用 ROI ESAM

DINO 没框：
    fallback 到全图 ESAM

rare 和 old5 重叠：
    old5 优先
```

------

## 风险

```text
优点：
    不占训练时间
    可能提升 rare 小目标 recall

风险：
    推理变慢
    DINO 框不准会引入噪声
    如果榜单偏 old5，这条收益未必明显
```

这条优先级低于：

```text
balanced partial unfreeze
zero-init refine
old5-safe sweep
old5-polish
```

------

# 方案七：checkpoint averaging

## 目标

如果两个 checkpoint 各有优势，可以做权重平均。

适合组合：

```text
A = 当前 66.462 best
B = balanced partial unfreeze + recalibrate
```

或者：

```text
A = lite_stage2_submit
B = old5-polish
```

------

## 平均方式

不要平均太激进。

```python
avg = 0.7 * A + 0.3 * B
```

或者：

```python
avg = 0.8 * A + 0.2 * B
```

不建议直接：

```python
avg = 0.5 * A + 0.5 * B
```

------

## 注意

```text
平均后必须重新 sweep。
不能直接拿 averaged checkpoint 提交。
```

------

# 提交选择建议

## 今天只有一次提交机会时

优先提交：

```text
lite_stage2_submit
```

原因：

```text
它已经是仓库里当前记录的 submit-safe 最强版本
不是 rare 激进版
比临时 old5-safe sweep 更有证据
```

------

## 明天提交候选顺序

优先级：

```text
1. lite_stage2_submit
2. balanced partial unfreeze + recalibrate + sweep
3. zero-init refine head + sweep
4. old5-polish + old5-safe sweep
5. class-router 双模型融合
6. DINO rare-only hybrid
```

------

# 最终优先级排序

## S 级：必须做

```text
1. lite_stage2_submit 提交/保底
2. old5-safe sweep
3. balanced partial unfreeze + recalibrate
```

## A 级：值得赌

```text
4. zero-init tiny refine head
5. old5-polish 1 epoch
6. old5 / 11cls class-router
```

## B 级：有时间再做

```text
7. checkpoint averaging
8. rare-only DINO → ROI ESAM hybrid
```

## 不建议做

```text
从头训练
全量解冻 EfficientSAM
解冻 ChineseCLIP
rare-heavy loss
rare oversample 3~4 倍
大改 decoder 成复杂结构
多尺度全类 TTA
大规模 self-distillation
```

------

# 最终执行路线

最后两天建议按这个顺序：

```text
Step 1：
提交 lite_stage2_submit 作为保底/验证榜单反馈

Step 2：
跑 old5-safe sweep
观察 old5_avg / rare_avg / all11

Step 3：
从 66.462 best 开 balanced partial unfreeze 1 epoch

Step 4：
冻结 image encoder，接 balanced recalibrate 1 epoch

Step 5：
对 unfreeze/recalibrate checkpoint 做 old5-safe + balanced sweep

Step 6：
如果 agent 改代码快，开 zero-init tiny refine head，跑 2 epoch

Step 7：
如果发现 old6 checkpoint 的 old5 更强，做 class-router 双模型融合

Step 8：
最后只保留 2~3 个提交候选：
    - lite_stage2_submit
    - balanced partial unfreeze + recalibrate + sweep
    - zero-init refine 或 old5-polish 二选一
```

------

# 一句话总结

现在冲 67/68 的核心不是继续 rare-heavy，也不是继续普通 decoder-only 微调，而是：

```text
用当前 66.462 作为底座
old5 不能掉
rare 轻微参与
小范围动 EfficientSAM 后段特征
必要时加 zero-init refine head
最后用 old5-safe sweep 和 old5 优先融合规则提交
```

最推荐主赌线：

```text
66.462 best
→ balanced partial unfreeze 1ep
→ freeze recalibrate 1ep
→ old5-safe + balanced sweep
→ 提交
```

最推荐结构赌线：

```text
66.462 best
→ zero-init tiny refine head
→ 2ep
→ sweep
→ 提交候选
```

最推荐保守线：

```text
66.462 best
→ old5-safe sweep
→ submit-safe
```

最推荐融合线：

```text
old5 用 old6/old-balanced 强 checkpoint
rare 用 11cls checkpoint
old5 overlap 永远优先
```