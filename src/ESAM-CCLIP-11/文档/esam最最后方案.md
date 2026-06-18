有，但我客观说：**真正有机会冲 67/68 的，不是再开一条全新大路线，而是把现在已有强点做“组合 + 全量定型”。**

现在离 67/68 只差一点点，**新架构从零试验的风险大于收益**。
你现在的有效信息是：

```text
B1 zero-init：
old5 raw = 0.6679
rare raw = 0.4631

B1 old5_safe sweep：
old5_avg = 0.6580
rare_avg = 0.4959
all11_avg = 0.5696
```

这说明 B1 不是废线，确实有 old5 提升，只是 rare 没完全保住。
A2 也是 old5 稳住了，但 rare 没自动回来：`old5=0.6651, rare=0.4569`。

所以你现在要冲 67/68，最现实的不是“再训练一个新东西”，而是下面几条。

------

# 结论先给

我现在只建议你考虑 **4 条**，按优先级：

```text
1. class-router：新模型管 old5，lite 管 rare
2. checkpoint soup：B1/A3/B2/lite 同架构权重平均
3. fullset final tune：选最强 split 模型，全量轻训 0.5~1 epoch
4. old5-primary submit：干脆押 old5，rare 不崩就行
```

不建议现在开：

```text
DINO 新线
SegFormer 新线
重新大规模造伪标签
大幅改 decoder
全量解冻 backbone
rare-heavy 继续训练
```

这些不是不能提分，是**来不及验证**。提交次数宝贵时，没验证的东西就是毒药。

------

# 路线 1：class-router，最现实的“组合提分”

这个是你刚才提到的。

核心：

```text
old5 用新 zero-init / A3 / B2 里面 old5 最强的模型
rare 用 lite_stage2_submit
```

也就是：

```text
person      → 新模型
car         → 新模型
building    → 新模型
tree        → 新模型
animal      → 新模型

trash can   → lite
window      → lite
door        → lite
fence       → lite
pole_light  → lite
motorcycle  → lite
```

## 为什么它有冲 67/68 的可能？

因为你现在的问题不是模型整体都差，而是：

```text
新模型：old5 更好，rare 没保住
lite：rare 更稳，old5 不一定最高
```

router 是直接把两个模型强项拼起来。

这条线**不需要再训练**，只改推理逻辑。
如果实现没 bug，它的风险比 rare-rescue 小。

## 风险

最大风险是 rare 抢 old 区域，所以必须：

```text
old mask 优先
rare mask 和 old mask 大重叠就删掉 rare
```

推荐规则：

```text
IoU(rare, old) > 0.30 → 删除 rare
overlap_area / rare_area > 0.40 → 删除 rare
```

这条路线我认为是**除 A/B 训练外最值得做的冲分路线**。

------

# 路线 2：checkpoint soup / 权重平均

这个很适合你现在这种情况。

你现在可能会有这些 checkpoint：

```text
lite_stage2_submit
B1_zero_init
A3_refine
B2_unfreeze_after_refine
```

如果结构兼容，可以做权重平均：

```text
soup = 0.5 * B1 + 0.5 * A3
soup = 0.6 * B1 + 0.4 * B2
soup = 0.7 * B1 + 0.3 * lite_common
```

## 它为什么有用？

不同阶段学到的偏差不一样：

```text
B1：old5 强，refine 稳
A3：先 unfreeze 再 refine，可能边界更好
B2：refine 后 unfreeze，可能特征更适配
lite：原始 rare 更稳
```

平均后可能：

```text
old5 不掉太多
rare 稍微回来
预测更平滑
泛化更稳
```

这类操作常见收益不大，但胜在低成本：

```text
可能 +0.1 ~ +0.4
少数情况 +0.5
```

冲 68 单靠它不现实，但冲 67 很可能有帮助。

## 注意

如果一个有 refine，一个没有 refine：

```text
共同参数做平均
refine_head 用 B1/A3/B2 中最好那个
```

不要乱平均 shape 不同的参数。

------

# 路线 3：fullset final tune，全量定型

这条是“从根源提分”里最合理的训练路线。

现在 split 上看到的结果，未必等于榜单。最后真正提交时，可以用：

```text
train + val 全部伪标签
从 split 最强 checkpoint 出发
freeze image/text
只训 decoder + refine
0.5~1 epoch
```

## 为什么它有机会提分？

因为你的 val 其实也是伪标签，不是真正干净 GT。
最后用全量数据定型，可能让模型对分布更稳，尤其 old5：

```text
person / car / building / tree / animal
```

这些样本多，全量训练更容易收益。

## 怎么训

不要再 unfreeze image encoder。
不要 rare-heavy。

建议：

```text
use_refine_head = True
train_decoder = True
train_refine_head = True
freeze_image_encoder = True
freeze_text_encoder = True
epochs = 1
decoder_lr = 3e-6 ~ 5e-6
refine_lr = 1e-5 ~ 2e-5
rare_oversample = 1.0~1.2
negative_ratio = 0.005
```

目标是：

```text
全量定型
不是继续学习新东西
```

如果你 A3/B2 跑完有一版 old5 强、rare 不崩，这条可以作为最后冲榜版本。

------

# 路线 4：old5-primary submit，直接押确定性

这个不花时间，但是策略上很关键。

你的判断是对的：**rare 是你猜的，old5 是确定的。**

所以如果最终结果是：

```text
新模型 old5 明显高
rare 比 lite 低一点，但 >= 0.49
```

那可以直接押 old5。

你现在 B1 old5_safe sweep 的 rare 是 `0.4959`，不是理想，但也没崩。
如果 A3/B2 也类似：

```text
old5 >= 0.658
rare >= 0.49
```

它就有提交价值。

------

# 真正不建议的“新根路线”

## 1. DINO / GroundingDINO 再开

不建议。

原因：

```text
训练慢
推理复杂
和当前最高分体系不一致
最后两天很难验证
```

除非你已经有现成强结果，否则现在开它是分散火力。

------

## 2. SegFormer / B0 蒸馏

也不建议现在作为冲 67/68 主线。

它可能长期有价值，但现在问题是：

```text
泛化弱
需要 teacher soft cache
需要多轮验证
```

时间不够。

------

## 3. 重新大规模造 rare 伪标签

不建议。

因为你自己也说了，rare 可能猜错。
现在再花时间强化 rare，收益不确定，还可能把 old5 搞坏。

------

## 4. 更大 decoder / 新 head 大改

不建议。

zero-init refine 已经是安全版结构增量。
再加复杂模块，短时间很容易：

```text
val 看着涨
榜单掉
推理慢
加载出 bug
```

------

# 我现在给你的冲 67/68 作战顺序

## 第一步：等 A3 / B2 跑完

不要接下一阶段。

拿到：

```text
A3: all11 / old5 / rare / per-class
B2: all11 / old5 / rare / per-class
```

------

## 第二步：同规则 sweep

对这些都跑：

```text
old5_safe
balanced
submit_safe
```

对象：

```text
B1
A3
B2
lite_stage2_submit
```

整理表：

```text
模型       objective      old5_avg    rare_avg    all11_avg
B1         old5_safe      ...
B1         balanced       ...
A3         old5_safe      ...
A3         balanced       ...
B2         old5_safe      ...
B2         balanced       ...
lite       submit_safe    ...
```

------

## 第三步：选两个候选方向

一个是：

```text
old5 最强单模型
```

一个是：

```text
router / soup 均衡模型
```

不要提交两个差不多的。

------

## 第四步：最后才考虑 fullset final tune

只有当 split 上已经有明确强候选时，才做 fullset。

比如：

```text
B1/A3/B2 某版：
old5 >= 0.658
rare >= 0.49
```

那可以全量定型一版。

------

# 我的客观判断

你现在要冲 67，**最有希望的是**：

```text
B1/A3/B2 中 old5 最强单模型
+ sweep
+ 可能 fullset final tune
```

你要冲 68，**更可能靠**：

```text
class-router
或
checkpoint soup + fullset tune
```

单纯继续 A/B 训练，很可能只是：

```text
old5 小涨
rare 小掉
总分不确定
```

所以现在不要再追“新训练阶段”。
真正的提分空间在：

```text
组合模型
全量定型
提交阈值
old/rare 路由
```

一句话：**有别的路线，但不是再开新大模型；现在最值得冲 67/68 的是 router、checkpoint soup、fullset final tune。**