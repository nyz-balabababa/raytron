看完这个评测说明后，我觉得 **BCD 不应该分开押，而是组合成一个“轻量学生模型 + CLIPSeg 兜底”的方案**。

核心原因：官方任务是按 `text_prompt` 做分割，不是固定类别语义分割；同一张图可能对应多个 prompt，说明里还明确建议**按 image_path 分组，同一张图尽量只做一次图像编码**；并且一个 prompt 要输出该语义的**并集 mask**，不是实例拆分。
所以，纯 SegFormer / PIDNet / BiSeNet 固定 11 类虽然快，但会牺牲开放词；纯 CLIPSeg 有开放词，但慢。最划算的是：

# 最推荐：**B + D 主线，C 做速度备胎**

也就是：

```text
SegFormer-B0 多标签学生模型
+ CLIPSeg teacher 蒸馏
+ rare class focal loss
+ CLIPSeg 未知 prompt 兜底
```

这比单独跑 B/C/D 都更适合你现在的情况。

------

## 先说结论：三个方案怎么排

| 方案                        | 我现在建议             | 原因                                          |
| --------------------------- | ---------------------- | --------------------------------------------- |
| **B：SegFormer-B0/B1**      | **主力学生模型**       | 精度更稳，改造难度适中，适合伪标签训练        |
| **C：PIDNet-S / BiSeNetV2** | 速度备胎               | 很快，但精度和训练稳定性不一定比 SegFormer 好 |
| **D：CLIPSeg 蒸馏**         | **必须加到 B 或 C 上** | 保留 CLIPSeg 的泛化知识，提升学生模型稳健性   |

最终不是“选 B 还是 C 还是 D”，而是：

```text
主线：SegFormer-B0 + focal + dice + CLIPSeg distillation
备胎：PIDNet-S + focal + dice
兜底：未知 text_prompt 仍走 CLIPSeg
```

------

# 为什么不是纯固定 11 类模型？

因为官方输入里有 `text_prompt`，不是只给图片。每条任务都有 `ann_id / image_path / text_prompt`，结果也要按 `ann_id` 回写。
这说明模型必须能响应文本提示。

所以如果你直接做：

```text
image → 11类语义分割
```

会有一个问题：官方如果给了没见过的 prompt，比如 `vehicle`、`street light`、`pedestrian`、`entrance`，甚至你 11 类之外的词，固定模型就不好处理。

但可以改成混合路由：

```text
如果 text_prompt 能映射到你的11类：
    用学生模型一次前向输出对应 mask，速度快

如果 text_prompt 不在11类/无法映射：
    回退到 CLIPSeg，根据原始文本 prompt 做开放词分割
```

CLIPSeg 本身就是为任意文本/图像 prompt 生成二值分割图设计的，适合作这个开放词兜底。([arXiv](https://arxiv.org/abs/2112.10003?utm_source=chatgpt.com))

------

# 最划算落地方案：SegFormer-B0 多标签学生模型

不要做 softmax 12 类语义分割，我建议做：

```text
image → 11个 sigmoid mask
```

也就是 **multi-label segmentation**，不是互斥分类。

原因是你的任务是 prompt 分割，一个 prompt 输出一个并集 mask。`building/window/door/fence` 这些类可能有空间包含关系，强行 softmax 会让它们互相竞争，不适合。

模型输出：

```python
logits = model(image)  # [B, 11, H, W]
```

训练时仍然可以沿用你现在的 CLIPSeg 数据形式：

```text
image + class_name + binary_mask
```

只不过训练 loss 时只取当前类别通道：

```python
pred = logits[:, class_idx]
loss = loss_fn(pred, target_mask)
```

这样改动比重写整套数据管线小很多。

SegFormer 的优点是结构本身偏轻，encoder 有多尺度特征，decoder 是轻量 MLP，论文也就是主打 simple/efficient semantic segmentation。([arXiv](https://arxiv.org/abs/2105.15203?utm_source=chatgpt.com))
对你这种红外图像 + 伪标签 + 小目标/细类任务，SegFormer-B0 比 BiSeNet/PIDNet 更稳一点。

------

## Loss 怎么配：把你之前有效的 focal 用上

建议用这个组合：

```text
Loss = BCEWithLogits + Dice + RareClassFocal + DistillLoss
```

第一版先别太复杂：

```python
loss = 1.0 * bce_loss \
     + 0.5 * dice_loss \
     + 0.5 * focal_loss
```

如果加蒸馏：

```python
loss = 1.0 * hard_loss \
     + 0.2 * distill_loss
```

稀有类 focal 权重大概这样：

```python
class_weights = {
    "person": 1.0,
    "car": 1.0,
    "building": 1.0,
    "tree": 1.2,
    "animal": 2.0,
    "trash can": 2.0,
    "window": 1.8,
    "door": 1.5,
    "fence": 1.5,
    "pole_light": 2.5,
    "motorcycle": 2.0,
}
```

`gamma=2.0` 先不动。
不要一开始把 rare focal 拉太猛，否则会出现主类 recall 掉、mask 变碎的问题。

------

# D：CLIPSeg 蒸馏怎么加最划算？

D 不应该单独成为一个模型，它应该作为 B/C 的训练增强。

具体做法：

```text
CLIPSeg teacher：输出 soft probability map
SegFormer/PIDNet student：学习 hard pseudo mask + teacher soft map
```

也就是：

```text
hard label：你的伪标签 mask
soft label：CLIPSeg 对同一 prompt 的概率图
```

这样学生模型既学你的扩充标签，又学 CLIPSeg 的开放词/泛化边界。

最省时间的做法：

```text
1. 先不蒸馏，训 SegFormer-B0 hard label 版，确认能跑通
2. 再用当前最好的 CLIPSeg-11 模型离线缓存 soft mask
3. 从 hard label 版继续 finetune，加 distill loss
```

不要一上来就在线跑 teacher，那会太慢。

缓存格式可以简单点：

```text
cache_soft/
  image_id_class.npy 或 npz
```

保存 `float16` 概率图即可。

蒸馏 loss 建议：

```python
distill_loss = BCEWithLogitsLoss(student_logit, teacher_prob)
```

权重先用：

```python
distill_lambda = 0.2
```

稀有类可以稍微高一点：

```python
distill_lambda_rare = 0.3
```

但 teacher 对 `window/pole_light` 如果本身不稳，就不要过度相信 teacher，hard label 仍然是主导。

------

# C：PIDNet / BiSeNetV2 怎么跑才划算？

C 我不建议作为第一主线，但可以开一个速度版。

BiSeNetV2 本来就是实时语义分割结构，把细节分支和语义分支分开，目标就是速度和精度平衡。([arXiv](https://arxiv.org/abs/2004.02147?utm_source=chatgpt.com))
PIDNet 是 detail/context/boundary 三分支结构，边界分支对 `fence / pole_light / window / door` 这类细边界目标理论上有帮助。([arXiv](https://arxiv.org/abs/2206.02066?utm_source=chatgpt.com))

但问题是：它们都是固定类别模型，开放词能力更弱，所以也要做同样的混合路由：

```text
已知11类 prompt → PIDNet/BiSeNet 输出
未知 prompt → CLIPSeg 兜底
```

如果你有空电脑，可以这样跑：

```text
C1：PIDNet-S，输入 640 或 768，11 sigmoid channels
C2：loss 同 SegFormer：BCE + Dice + focal
C3：先不加蒸馏，快速看速度和 mIoU
```

如果 PIDNet-S 本地验证比 SegFormer-B0 只低一点，但推理快很多，它就适合最终冲速度分。
如果低太多，就别继续浪费时间。

------

# 评测文件对推理脚本的直接影响

你这个方案落地时，推理脚本必须按官方要求写，尤其这几条：

1. **必须从固定路径读任务和模型**，`test_tasks.json`、图片根目录、`predictions.json`、`/raytron/code/model/sam3.pt` 这些路径不要改；模型文件也必须放在 `/raytron/code/model/sam3.pt`。
2. **按 image_path 分组**，同一张图多个 prompt 时，学生模型只前向一次，直接取不同通道输出。这个正好是学生模型最大的速度优势。
3. **每个 prompt 输出并集 mask**，不要做实例拆分输出多个结果。比如 prompt 是 car，图里多辆车就合成一个总 mask。
4. **每个 ann_id 都必须有结果**，没检测到也要输出空 mask 的 RLE，不能跳过。
5. **RLE 用 COCO 标准格式**，`mask` 要是 `uint8` 二值，`counts` 如果是 bytes 要转字符串。
6. 后台还会对 `/raytron/code/model/sam3.pt` 做参数量统计，所以最终模型别无脑塞一堆不用的权重。

------

# 我建议你现在的实验顺序

## 第 1 条线：继续 CLIPSeg-11 保底

这条不用说，已经最稳。
目标是拿一个能提交、开放词最强的版本。

------

## 第 2 条线：马上开 SegFormer-B0 学生模型

这是我最推荐开的新实验。

配置：

```text
模型：SegFormer-B0
输出：11 sigmoid channels
输入：768
loss：BCE + Dice + rare focal
训练数据：删 computer 后的 11 类伪标签
保存：best_mIoU_11.pt / best_old5.pt / best_rare.pt
```

先训 5～8 轮看趋势。
如果 5 轮后 11 类 mIoU 接近或超过 CLIPSeg-11，而且速度明显快，就继续。

------

## 第 3 条线：SegFormer-B0 加 CLIPSeg 蒸馏

等第 2 条线跑通后再加。

```text
teacher：当前最好的 CLIPSeg-11
student：SegFormer-B0
distill：teacher soft mask
loss：hard loss + 0.2 * soft loss
```

这条最可能成为最终主力。

------

## 第 4 条线：PIDNet-S 速度版

只有在你有多余机器、代码能很快跑通时开。
别为了 C 方案耽误 B+D 主线。

配置：

```text
模型：PIDNet-S
输出：11 sigmoid channels
输入：640 或 768
loss：BCE + Dice + rare focal
用途：测速度上限
```

如果 PIDNet-S 精度低很多，就直接放弃。
如果只低 1～2 分但速度快很多，可以作为最终速度版提交。

------

# 最终我推荐的落地方案

**最适配你现在情况的是：**

```text
CLIPSeg-11 保底
+
SegFormer-B0 11通道 multi-label student
+
rare class focal loss
+
CLIPSeg soft mask distillation
+
未知 prompt 回退 CLIPSeg
```

也就是：

```text
B + D 为主，C 只做速度备胎
```

这样你同时满足：

```text
精度：SegFormer-B0 + focal + 蒸馏
速度：已知11类一次前向出所有 mask
开放词：未知 prompt 仍然走 CLIPSeg
提交规范：每个 ann_id 输出一个并集 RLE
```

这个方案性价比最高，也最符合现在的评测接口。