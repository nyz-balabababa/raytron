不是，**优化 CLIPSeg 主线不等于只能调参数**。但是现在比赛时间紧，最值得做的是：

```text
优先做低风险优化：
数据 / prompt / loss / 负样本 / 阈值 / 后处理

谨慎做中风险结构优化：
只改 decoder / 加轻量 adapter / 多 prompt 融合

不建议做高风险大结构替换：
换 backbone / 换 text encoder / 改成新架构
```

也就是说：**可以改模型结构，但不要大改。**

------

## 1. 最稳的 CLIPSeg 优化：不改结构

这一类最适合你现在继续跑榜：

```text
新 11 类标签
prompt 扩展
负样本比例
类别权重
Dice / Focal / BCE 比例
每类阈值
小目标后处理
多 prompt 融合
```

这些不是“只调小参数”，它们会实质影响模型学到什么，而且风险最低。

你现在主问题是：

```text
recall 低
拒识能力差
但又不能过保守
```

那最该优先调的是：

```python
NEGATIVE_SAMPLE_RATIO
NEGATIVE_SAMPLE_WEIGHT
DICE_WEIGHT
FOCAL_WEIGHT
FOCAL_ALPHA
CLASS_WEIGHTS
VAL_THRESHOLDS / TEST_THRESHOLDS
```

这比换结构更快见效。

------

## 2. 可以做的轻结构优化：只动 decoder/head

如果你说“想在 CLIPSeg 上做结构优化”，我建议只做这种小改：

### 方案 A：增强 decoder head

保留：

```text
CLIP image encoder
CLIP text encoder
CLIPSeg 主体逻辑
```

只在 mask decoder 末端加一点：

```text
Conv-BN-ReLU
轻量 refinement head
边界细化分支
```

目标是让 mask 边界更细、召回小目标更好。

风险：中等。
缺点：需要重新训，而且如果你代码不熟，容易引入推理不兼容。

------

### 方案 B：LoRA / Adapter 微调

保留 CLIPSeg 主结构，只加小参数模块：

```text
CLIP image encoder adapter
decoder adapter
LoRA on attention / projection
```

优点是比全量改结构稳，能增强领域适应。
缺点是你现在时间紧，写起来和验证都要时间。

------

### 方案 C：多 prompt 融合

这个我反而最推荐，因为它不大改模型结构。

比如同一个类别不只用一个 prompt：

```text
car:
"car"
"vehicle"
"automobile"

trash can:
"trash can"
"garbage bin"
"bin"
```

推理时多个 prompt 的 mask 做融合：

```text
mean / max / weighted mean
```

这属于 **CLIPSeg 特有的结构优势利用**，不改变模型文件，但能提升 prompt 泛化。

------

## 3. 不建议现在做的大结构优化

这些别碰：

```text
CLIPSeg 换成 SegFormer
换 CLIP backbone
换 text encoder
重新设计 image-text fusion
引入 SAM decoder
把 CLIPSeg 改成多通道固定类别输出
```

原因很简单：

```text
时间不够
训练风险高
推理脚本要大改
很可能还不如原 CLIPSeg 稳
```

尤其你现在真正需要的是保留 CLIPSeg 的 prompt 泛化能力，那就不要把它改成固定类别模型。

------

## 你现在最合理的 CLIPSeg 主线优化路线

我建议按这个顺序：

```text
1. 新 11 类标签训练原 CLIPSeg
2. 加 prompt alias / prompt pool
3. 轻负样本训练，别压 recall
4. Dice 权重提高，Focal 小权重辅助
5. 按类 threshold sweep
6. 推理时多 prompt 融合
7. 按类面积过滤 / 连通域后处理
```

这已经不是简单调参了，而是完整的 **训练 + 推理策略优化**。

------

## 如果只能再跑一版，我建议别改结构

你现在应该选：

```text
原 CLIPSeg 架构 + 新 11 类标签 + prompt 扩展 + 轻负样本 + 重新扫阈值
```

不要再开新模型结构。

配置方向：

```python
NEGATIVE_SAMPLE_RATIO = 0.05 ~ 0.08
NEGATIVE_SAMPLE_WEIGHT = 0.08 ~ 0.10

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
FOCAL_WEIGHT = 0.10 ~ 0.15
FOCAL_ALPHA = 0.75

RARE_OVERSAMPLE = {
    "animal": 2,
    "trash can": 2,
    "window": 2,
    "door": 2,
    "pole_light": 2,
    "motorcycle": 2,
}
```

推理阈值先偏 recall：

```python
{
    "person": 0.55,
    "car": 0.55,
    "building": 0.60,
    "tree": 0.50,
    "animal": 0.45,
    "trash can": 0.40,
    "window": 0.40,
    "door": 0.40,
    "fence": 0.40,
    "pole_light": 0.35,
    "motorcycle": 0.40,
}
```

然后按验证集 sweep 定最终阈值。

------

一句话：**不是只能调参数，但现在最该做的是“不大改结构的 CLIPSeg 主线优化”。你可以利用 CLIPSeg 的 prompt 泛化优势，通过 prompt 扩展、多 prompt 融合、轻负样本、loss 和阈值后处理来提分；大结构改造现在不划算。**