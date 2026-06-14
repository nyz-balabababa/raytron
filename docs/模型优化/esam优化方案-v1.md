对，现在时间不多，就不要再开“看起来高级但训练很久”的大改路线了。你现在最好模型已经是：

```text
EfficientSAM + ChineseCLIP + 自写 decoder
```

那最高性价比一定是围绕它做**小改、热启动、缓存、阈值、推理加速**。我建议你只押下面这几条，别再分散。

------

## 我建议的优先级

| 优先级 | 方案                                                         | 是否需要长训练   | 主要收益           | 建议       |
| ------ | ------------------------------------------------------------ | ---------------- | ------------------ | ---------- |
| **1**  | **训练/推理缓存：image embedding + text embedding**          | 不需要或少量训练 | 大幅提速           | **必须做** |
| **2**  | **从当前 best 热启动训 11 类，冻结 ESAM 和 ChineseCLIP，只训 decoder** | 需要，但最省     | 快速适配 11 类     | **主方案** |
| **3**  | **11 类 prompt prototype**                                   | 几乎不需要       | 精度小涨、速度也涨 | **必须做** |
| **4**  | **每类阈值搜索 + 后处理微调**                                | 不训练           | 榜单性价比高       | **必须做** |
| **5**  | **decoder tiny refine head**                                 | 需要短训         | 小目标/边界提升    | 时间够再做 |
| **6**  | self-distillation / 解冻 ESAM                                | 训练重           | 可能涨             | 暂时不建议 |

------

# 方案一：先做 embedding cache，最高性价比

你之前 6 类一轮 2 小时，根本原因大概率是：

```text
每个 image + prompt 样本都重复跑 EfficientSAM image encoder
```

但同一张图会对应多个类别 prompt。正确做法应该是：

```text
一张图只跑一次 EfficientSAM image encoder
多个 prompt 共用 image embedding
```

训练阶段可以做两种缓存。

### A. text embedding 缓存

11 类固定，ChineseCLIP 文本编码根本不用每次训练都跑。

```text
person → text_emb
car → text_emb
...
motorcycle → text_emb
```

直接提前缓存成 `.pt`。

这个改动小，收益稳定。

### B. image embedding 缓存

如果 EfficientSAM encoder 是冻结的，那就直接缓存每张图的 image feature：

```text
image_path → efficient_sam_image_embedding.pt
```

训练 decoder 时直接读 embedding，不再跑图像 encoder。

这条可能把你一轮 2 小时明显压下来。尤其你现在 11 类后，同图多 prompt 更多，重复编码浪费更大。

**结论：这条必须优先做。**
它既能加速训练，也能帮助推理脚本做同图分组。

------

# 方案二：从当前 best 热启动，短训 11 类 decoder

这是当前最稳的精度路线。

不要从头训 11 类，也不要一上来解冻 EfficientSAM。

直接：

```text
加载当前 6类/旧 best 的 EfficientSAM+ChineseCLIP+decoder 权重
冻结 EfficientSAM
冻结 ChineseCLIP
只训练 decoder
使用新 11 类伪标签
```

训练策略：

```text
epoch：3～5 轮先看趋势
输入尺寸：沿用当前最好模型，不要乱改
负样本比例：先低一点，0.10～0.15
rare oversample：开启
learning rate：比原训练略低
```

原因是你的模型已经学会了：

```text
红外图像特征
text embedding 和 mask decoder 的对应关系
基本分割边界
```

新 11 类只是让 decoder 适配更多 prompt，不值得从头大训。

我建议第一版配置：

```text
freeze_esam = True
freeze_chineseclip = True
train_decoder = True
train_refine = False
epochs = 3～5
negative_ratio = 0.10
rare_oversample = True
```

如果 3 轮后 `old5` 不掉、`rare` 有提升，再继续跑。

------

# 方案三：prompt prototype，几乎白嫖

不要每类只用一个 prompt。11 类建议给每类做别名，然后平均成一个 prototype embedding。

例如：

```text
person: person, people, pedestrian, 人
car: car, vehicle, automobile, 车辆
trash can: trash can, garbage bin, trashbin, 垃圾桶
pole_light: pole light, street light, lamp, light pole, 路灯
motorcycle: motorcycle, motorbike, 摩托车
```

最后：

```python
class_emb = mean(encode_text(alias_list))
```

训练和推理都用这个 `class_emb`。

这条的好处：

```text
不明显增加推理时间
不需要长训练
对新增 rare 类特别有用
能减少单个英文类别名不准的问题
```

尤其你这几个类：

```text
trash can
pole_light
motorcycle
window
door
fence
animal
```

很适合做 prompt prototype。

------

# 方案四：每类阈值搜索，便宜但很可能涨分

现在从 6 类变 11 类，统一阈值肯定不够。

建议你在验证集上直接扫每类阈值：

```text
0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
```

大概会出现这种倾向：

```text
person/car/building：阈值偏高
tree/animal：中等
trash can/window/door/fence/pole_light/motorcycle：偏低
```

初始可以试：

```python
VAL_THRESHOLDS = {
    "person": 0.65,
    "car": 0.65,
    "building": 0.65,
    "tree": 0.55,
    "animal": 0.50,
    "trash can": 0.45,
    "window": 0.45,
    "door": 0.45,
    "fence": 0.40,
    "pole_light": 0.35,
    "motorcycle": 0.45,
}
```

注意：`pole_light / fence / window` 这种细目标阈值太高会直接没了。

这条不需要训练，性价比很高。

------

# 方案五：训练集别全量硬跑，先做 balanced quick train

你现在 11 类如果全量跑，一轮可能更久。别直接全量 10 轮。

建议先构造一个快训集：

```text
old5：每类抽一部分高质量样本
new6：尽量保留全部正样本
负样本：少量，0.10 左右
```

也就是：

```text
person/car/building/tree/animal：抽样
trash can/window/door/fence/pole_light/motorcycle：全保留或过采样
```

目标不是训练最终模型，而是快速判断：

```text
新 11 类能不能学
old5 会不会崩
rare 类有没有信号
```

等趋势对了，再切回全量训 2～3 轮。

这个比直接全量跑更省时间。

------

# 方案六：decoder tiny refine head，时间够再做

这条不是第一优先级，但如果你当前 best 已经很强，可以加一个很小的 refine head。

结构：

```text
decoder coarse_logit
+ gray_image
→ 3层小卷积
→ residual_logit
→ refined_logit = coarse_logit + residual_logit
```

只训练：

```text
decoder + refine head
```

不要动 EfficientSAM。

收益主要在：

```text
fence
pole_light
window
door
trash can
motorcycle
```

但它毕竟要改模型和重新训练，所以放在 cache、热启动、阈值之后。

------

# 我现在不建议你做的

时间不多，这些先别碰：

```text
1. 从零训练 11 类 ESAM+ChineseCLIP+decoder
2. 解冻整个 EfficientSAM
3. 在线 teacher self-distillation
4. 多尺度推理
5. EfficientSAM 后面再接 SAM refine
6. 大改 decoder 成 Mask2Former/FPN
7. CLIPSeg-refine 主线
```

这些都有可能有用，但现在性价比不高。

------

# 我给你的最终推荐组合

现在最该落地的是这个：

```text
ESAM-CCLIP-v2-fast11
```

组合如下：

```text
当前 best 权重热启动
+ 11 类 prompt prototype
+ 冻结 EfficientSAM image encoder
+ 冻结 ChineseCLIP text encoder
+ 只训 decoder
+ text embedding 缓存
+ image embedding 缓存
+ rare oversample
+ 低负样本比例
+ 每类阈值搜索
+ 推理按 image_path 分组，一张图只编码一次
```

实验顺序：

```text
Step 1：先改推理/训练缓存，不改模型
Step 2：加载当前 best，短训 11 类 decoder 3 轮
Step 3：验证集扫每类阈值
Step 4：如果效果好，继续全量 2～3 轮
Step 5：时间够再加 tiny refine head
```

------

## 一句话决策

**现在别开新大路线。**

你当前最好模型已经换成 ESAM+ChineseCLIP+decoder，那最划算就是：

```text
用当前 best 热启动训 11 类 decoder
用 embedding cache 把训练/推理速度降下来
用 prompt prototype 和每类阈值把精度榨出来
```

这几项是现在最可能在短时间内同时提高**精度 + 速度**的方案。