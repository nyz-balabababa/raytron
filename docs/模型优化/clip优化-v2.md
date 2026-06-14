有。除了你现在这条 **“CLIPSeg 原架构 + prompt pool + 轻负样本 + loss + 阈值 + 后处理”**，纯 CLIP 还能走几条路线。你文件里原方案属于低风险主线优化，优先级确实最高。

但如果你还想给纯 CLIP 再开一个并行实验，我建议从下面选。

------

## 路线 A：不重训，只做推理增强版

这是最快的路线。
直接拿你现在正在跑的 **11cls+CLIPSeg 权重**，不改训练，只改验证/推理。

做这些：

```text
1. 多 prompt 融合
2. 每类 threshold sweep
3. TTA，多尺度或水平翻转
4. 按类连通域后处理
5. 小目标类保护
6. 类间冲突抑制
```

这个路线的优点是：

```text
不用重新训练
最快能出结果
不会破坏 CLIP 泛化
适合作为线1 checkpoint 的后处理增强版
```

建议融合方式：

```python
final_prob = 0.7 * prob_main + 0.3 * mean(prob_aliases)
```

TTA 可以先只做：

```text
原图
水平翻转
```

别一开始上太多尺度，不然推理太慢。

这个路线适合命名：

```text
CLIPSeg-11 infer-opt
```

如果你今天时间很紧，我甚至觉得这个比重新训练一版还值得。

------

## 路线 B：冻结 CLIP，只训 decoder / projection

这条和你现在的优化路线不一样。

你现在那条大概率是：

```text
加载 CLIPSeg
用 11 类继续整体 fine-tune
```

而这条是：

```text
CLIP image encoder 冻结
CLIP text encoder 冻结
只训练 mask decoder / visual projection / fusion 层
```

目的：

```text
保住 CLIP 的泛化能力
只让 decoder 适应 11 类伪标签和比赛域 mask 形态
```

适合配置：

```python
FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
TRAIN_DECODER_ONLY = True

LR_DECODER = 1e-4
LR_ENCODER = 0.0
```

Loss 可以用：

```python
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.2
FOCAL_WEIGHT = 0.05
```

负样本更轻：

```python
NEGATIVE_SAMPLE_RATIO = 0.04
NEGATIVE_SAMPLE_WEIGHT = 0.05
```

这条线的特点是：

```text
更稳
不容易把 CLIP 训坏
但上限可能比全量 fine-tune 低
```

如果你担心“11 类伪标签把 CLIP 泛化训没了”，这条路线很适合跑。

------

## 路线 C：prompt consistency 一致性训练

这条更像纯 CLIP 专属优化。

思路是：
同一个类别的不同 prompt，预测出来的 mask 不应该差太多。

比如：

```text
car
vehicle
automobile
```

它们应该指向同一个区域。

训练时做：

```text
主 prompt 正常监督 mask
alias prompt 不单独监督标签，而是和主 prompt 做一致性 loss
```

比如：

```python
loss = supervised_loss(main_prompt, gt_mask) \
     + 0.1 * consistency_loss(alias_prompt_prob, main_prompt_prob.detach())
```

可以用：

```python
CONSISTENCY_WEIGHT = 0.05 ~ 0.10
```

优点：

```text
增强 prompt 稳定性
更符合 CLIPSeg 的泛化逻辑
不会改结构
```

缺点：

```text
训练会变慢
一个样本要多跑 1-2 个 prompt
```

所以建议每个样本只额外采一个 alias，不要把所有 alias 都跑一遍。

这条路线适合叫：

```text
CLIPSeg-11 prompt-consistency
```

我觉得这条比 LoRA / adapter 更适合你现在的情况。

------

## 路线 D：负 prompt / absent class 拒识训练

你之前的问题是：

```text
recall 低
拒识能力差
但又不能太保守
```

普通负样本是整张图没有目标时训空 mask。
但 CLIPSeg 还可以做一种更精细的负样本：

```text
图里有 car
但没有 trash can
那就用 trash can prompt 监督空 mask
```

也就是 absent class negative。

训练时：

```text
正样本：image + present class prompt -> gt mask
负样本：same image + absent class prompt -> empty mask
```

配置建议非常轻：

```python
ABSENT_NEGATIVE_RATIO = 0.05
ABSENT_NEGATIVE_WEIGHT = 0.05
```

别太大，否则会压 recall。

优点：

```text
比普通负样本更能提升 prompt 拒识
更适合 CLIPSeg
```

风险：

```text
如果伪标签漏标严重，会把真实目标当负样本压掉
```

所以这条只适合小权重跑，不要激进。

------

## 路线 E：高置信样本精修版

这条不是改模型，而是改训练样本权重。

你的 11 类伪标签肯定质量不完全一致。可以让训练变成：

```text
高置信样本正常训练
低置信样本降权训练
稀有类不过度过滤
```

比如：

```python
SAMPLE_WEIGHT = {
    "high_conf": 1.0,
    "mid_conf": 0.6,
    "low_conf": 0.3,
}
```

如果没有置信度字段，也可以用一些近似规则：

```text
mask 太碎：降权
面积异常大：降权
边界极乱：降权
稀有类小目标：不轻易降权
```

这条路线适合伪标签噪声比较明显时用。

优点：

```text
不破坏 CLIP
能减少伪标签噪声影响
```

缺点：

```text
需要你现在的伪标签里有 score / source / quality 信息
没有的话就要额外写规则
```

------

## 我最推荐的额外纯 CLIP 路线

如果你已经有：

```text
线1：11cls + 原 CLIPSeg 正在跑
线2：11cls + CLIPSeg opt 准备跑
```

那我不建议再开一条重训练线。
最推荐额外开的是：

```text
线2.5：CLIPSeg infer-opt
```

也就是：

```text
不重训
只对线1和线2的 checkpoint 做：
多 prompt 融合 + threshold sweep + TTA + 后处理
```

因为它：

```text
成本最低
最不容易翻车
能同时服务线1和线2
如果有效，两个 checkpoint 都能涨
```

------

## 如果非要再训练一版，我推荐这条

不要 LoRA，不要 adapter，不要改 decoder。
直接跑：

```text
CLIPSeg-11 decoder-only fine-tune
```

配置：

```python
FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
TRAIN_DECODER_ONLY = True

NEGATIVE_SAMPLE_RATIO = 0.04
NEGATIVE_SAMPLE_WEIGHT = 0.05

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.2
FOCAL_WEIGHT = 0.05
```

这条就是为了验证：

```text
是不是全量 fine-tune 把 CLIP 泛化能力训坏了
```

如果 decoder-only 比全量 opt 更稳，那你最后就用 decoder-only 或者拿它做集成参考。

------

## 最终建议

现在纯 CLIP 最合理的组合是：

```text
线1：11cls + 原 CLIPSeg
作为基线，继续跑。

线2：11cls + CLIPSeg opt
prompt pool + 轻负样本 + Dice/Focal + rare oversample。

线2.5：CLIPSeg infer-opt
不重训，只做多 prompt 融合 + threshold sweep + TTA + 后处理。

备选线：decoder-only fine-tune
冻结 CLIP encoder，只训 decoder，验证是否能保泛化。
```

一句话：
**除了原来的低风险训练优化，最值得加的是“推理增强版”和“冻结 CLIP 的 decoder-only 微调版”。这两条都还是纯 CLIP，不会变成别的模型。**