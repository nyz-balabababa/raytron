你现在这版正在跑的 **热启动 3 epoch** 不要停，它本身就是 ESAM 线最稳的主方案：
**当前 best → 11 类 → 冻结 ESAM / ChineseCLIP → 只训 decoder → prompt prototype / cache / 阈值搜索**，这个方向和你发的方案一致，优先级也对。文件里也明确说了现在别开大路线，主要围绕缓存、热启动、prompt prototype、阈值、rare oversample 做小改。

但你还可以并行开 **2 条 ESAM 轻量分支**，别超过 2 条，不然会乱。

------

## 你现在并行跑这两条最划算

### 方案 A：ESAM-rare-balanced 分支

这个是我最推荐你并行开的。

**核心目的：让 11 类新增小类真的学到，不让 old5/old6 被冲坏。**

你现在 11 类以后，最大风险是：

```text
person / car / building / tree / animal 这几个大类样本太多
trash can / window / door / fence / pole_light / motorcycle 信号太弱
decoder 训练完还是偏向老类
```

所以另开一版：

```text
从同一个 ESAM best 热启动
冻结 ESAM
冻结 ChineseCLIP
只训 decoder
但改采样和 loss
```

配置建议：

```python
epochs = 3~5
negative_ratio = 0.05~0.10

old_class_sample_ratio = 0.35~0.50
rare_class_keep_ratio = 1.0
rare_oversample = 2~4

loss = bce + dice + focal
rare_class_loss_weight = 1.5~2.5
old_class_loss_weight = 1.0
```

11 类大概这样分：

```python
old_classes = [
    "person", "car", "building", "tree", "animal", "computer"
]

rare_classes = [
    "trash can", "window", "door", "fence", "pole_light", "motorcycle"
]
```

这个分支和你当前热启动主线区别是：

```text
当前主线：更稳，整体适配 11 类
方案 A：更偏 rare recall，赌新增类涨分
```

如果方案 A 跑完发现：

```text
old6 mIoU 小掉 <= 1~2%
rare mIoU / recall 明显涨
```

那它就有价值。
如果 old6 掉很多，说明 rare 权重太猛，降 oversample 或 loss_weight。

------

### 方案 B：ESAM-low-threshold + 后处理搜索分支

这个不用训练，必须并行做。

你发的方案里已经提到每类阈值搜索，这条很便宜，而且 11 类后统一阈值肯定不合理。

你现在应该单独开一个脚本：

```text
固定当前热启动 checkpoint
在 val 上扫每类 threshold
扫 min_area
扫后处理参数
```

初始阈值我建议这样：

```python
VAL_THRESHOLDS = {
    "person": 0.65,
    "car": 0.65,
    "building": 0.65,
    "tree": 0.55,
    "animal": 0.50,
    "computer": 0.55,

    "trash can": 0.40,
    "window": 0.35,
    "door": 0.40,
    "fence": 0.30,
    "pole_light": 0.28,
    "motorcycle": 0.40,
}
```

然后每类扫：

```text
0.25 / 0.30 / 0.35 / 0.40 / 0.45 / 0.50 / 0.55 / 0.60 / 0.65
```

尤其注意：

```text
fence / pole_light / window
```

这三个不要阈值太高。它们很可能不是“分割不好”，而是 **logit 分数偏低，直接被你阈值筛没了**。

后处理也要 class-aware：

```python
MIN_AREA = {
    "person": 32,
    "car": 48,
    "building": 128,
    "tree": 64,
    "animal": 24,
    "computer": 24,

    "trash can": 12,
    "window": 8,
    "door": 12,
    "fence": 8,
    "pole_light": 6,
    "motorcycle": 16,
}
```

这条分支的好处是：

```text
不占 GPU 训练时间
能直接提高榜单分
可以服务所有 ESAM checkpoint
```

------

## 另外一条可选：tiny refine，不是现在第一优先

你文件里提到 `decoder tiny refine head`，这个能做，但我不建议现在立刻开，除非你有 agent 很快改完。它主要收益在小目标和边界，尤其是 `fence / pole_light / window / door / trash can / motorcycle`。

结构可以很小：

```text
coarse_logit + gray_image
→ Conv 3x3
→ Conv 3x3
→ Conv 1x1
→ residual_logit
→ final_logit = coarse_logit + residual_logit
```

训练策略：

```python
freeze_esam = True
freeze_chineseclip = True
train_decoder = True
train_refine_head = True
epochs = 2~3
lr = 原 decoder lr 的 0.5~1.0
```

但它的问题是：

```text
要改模型结构
要改权重加载
要改推理脚本
```

所以我把它排在 **rare-balanced 和阈值搜索之后**。

------

## 现在三条线怎么排

你现在已经有：

```text
线1：11cls + CLIP 正在跑
线2：11cls + ESAM 热启动正在跑
线3：11cls + ESAM 优化
```

我建议 ESAM 优化不要再拆太多，就这样：

```text
ESAM 主线：
当前正在跑的热启动 3 epoch
看 old6 / new5 / all11 指标

ESAM 并行分支 A：
rare-balanced 采样 + rare loss weight
跑 3 epoch

ESAM 并行分支 B：
不训练，只做 threshold + min_area + 后处理搜索
```

暂时别做：

```text
解冻 ESAM
从零训 11 类
多尺度推理
self-distillation
大改 decoder
Mask2Former/FPN
SAM refine
```

这些你文件里也列为不推荐，现在时间不够，确实别碰。

------

## 最终我建议你现在立刻做的事

等当前热启动 3 epoch 结束后，先不要直接继续训，先看这几个指标：

```text
1. all11 mIoU
2. old6 mIoU
3. new5 / rare6 mIoU
4. per-class IoU
5. 每类预测数量
6. 每类 recall
```

如果结果是：

```text
old6 稳，但 new rare 低
```

就开 **rare-balanced 分支**。

如果结果是：

```text
rare 类几乎没有预测
```

先别急着继续训，优先做 **低阈值搜索**。

如果结果是：

```text
rare 类有预测但 mask 边界差 / 小目标碎
```

再考虑 **tiny refine head**。

一句话：
**现在 ESAM 线最该并行的是“rare-balanced 训练分支 + 阈值/后处理搜索分支”，不是再开新结构。**