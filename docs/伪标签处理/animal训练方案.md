# Animal Rare Label / Prompt 扩展 / 训练推理规则

## 1. 总体目标

当前 rare animal 方案的目标不是新增细分类，而是：

```txt
用 fox / wolf / dog / lion / tiger / horse / monkey 等细分动物伪标签
补充旧 base label 中的 animal 类
提升 animal 主类的召回和掩码完整度
```

最终训练类别仍保持原来的主类：

```txt
person
tree
building
animal
car
computer
```

不要新增：

```txt
fox
dog
wolf
lion
tiger
horse
monkey
...
```

这些细分类只作为 animal 的来源 prompt 或 prompt 扩展使用。

------

## 2. 新 label JSON 输入规则

新生成的 label 仍然保持 prompt-style 格式：

```json
{
  "image_path": "...",
  "prompts": {
    "person": {...},
    "tree": {...},
    "building": {...},
    "animal": {...},
    "car": {...},
    "computer": {...}
  }
}
```

其中：

```txt
person / tree / building / car / computer
完全沿用 base label，不做修改。
```

只允许修改：

```txt
prompts["animal"]
```

细分动物映射后不要写成新的顶层 prompt：

```json
"fox": {...}
"dog": {...}
"lion": {...}
```

正确做法是把它们合并进：

```json
"animal": {
  "hit": true,
  "score": ...,
  "rle": ...
}
```

如果需要保留来源信息，可以写在 animal 的附加字段里：

```json
"animal": {
  "hit": true,
  "score": 0.82,
  "rle": {...},
  "rare_sources": ["fox", "wolf"],
  "prompt_pool": ["animal", "fox"]
}
```

其中：

```txt
rare_sources：只用于记录来源，方便分析
prompt_pool：可用于训练脚本的 prompt 扩展
真正的训练类别仍然是 animal
```

------

## 3. 细分动物映射规则

细分动物 prompt 统一映射到 animal：

```txt
fox / wolf / dog / lion / tiger / horse / monkey / zebra / giraffe / bear / duck / pig / rabbit
→ animal
```

映射时：

```txt
mask 不变
rle 不变
image_path 不变
训练类别改为 animal
```

也就是说：

```txt
source_prompt = fox
train_label = animal
```

不是训练 fox 类。

------

## 4. Val 阶段 ABC 使用规则

Val 阶段主要用于验证合并策略，可以保留 A/B/C 三档。

### A 类

A 类质量相对最高，可以用于细分 prompt 扩展：

```txt
train_label = animal
prompt_pool = ["animal", source_prompt]
sample_weight = 较高
```

例如：

```json
{
  "source_prompt": "fox",
  "train_label": "animal",
  "grade": "A",
  "prompt_pool": ["animal", "fox"]
}
```

### B 类

B 类只作为 animal 补充，不使用 source_prompt 训练：

```txt
train_label = animal
prompt_pool = ["animal"]
sample_weight = 低
```

原因：

```txt
B 类细分类不够可靠，可能 source_prompt=fox，但实际是 dog/wolf/其他动物。
如果用 fox 参与训练，会污染 fox 这个文本 prompt。
```

### C 类

C 类更弱，只能作为极低权重 animal 候选：

```txt
train_label = animal
prompt_pool = ["animal"]
sample_weight = 极低
每图限量
```

C 类不能用 source_prompt 训练。

------

## 5. Train 阶段 ABC 使用规则

Train 阶段要比 val 更保守。

当前建议：

```txt
只使用严格筛选后的 A 类 rare animal
B/C 默认 drop
```

原因：

```txt
train 的 B/C 质量明显比 val 差
存在类别错、框不准、噪声多的问题
直接加入训练可能污染 animal 类
```

Train A 类也需要筛选：

```txt
hit = true
rle 不为空
score 达到阈值
面积正常
不是大片背景
不是明显噪声
source_prompt 在白名单内
```

推荐 train 使用的 A 类 prompt：

```txt
fox
wolf
dog
lion
tiger
horse
monkey
```

谨慎或暂时不用：

```txt
duck
pig
rabbit
zebra
giraffe
bear
```

Train A 类可以保留 prompt 扩展：

```txt
prompt_pool = ["animal", source_prompt]
```

Train B/C 默认：

```txt
drop
```

------

## 6. 去重取并规则

rare animal 合并进 base label 前，必须先去重。

### 去重对象

```txt
rare animal vs base prompts["animal"]
rare animal vs 已接收 rare animal
```

### 重复判断

满足任一条件即认为重复：

```txt
IoU > 0.6
containment > 0.8
```

其中：

```txt
containment = intersection / min(area_a, area_b)
```

### 重复时保留规则

```txt
rare 和 base animal 重复：
保留 base animal，丢弃 rare

rare 和 rare 重复：
按 A > B > C 优先级保留
同等级再按 score 高优先
```

注意：

```txt
A > B > C 只表示重复目标时的保留优先级。
如果 A 漏检，而 B/C 检出了新的不重复 animal，B/C 可以补进 animal。
```

### 不重复时

不重复的 rare animal 与 base animal 做 OR：

```txt
final_animal_mask = base_animal_mask OR kept_rare_animal_masks
```

然后写回：

```txt
prompts["animal"]
```

------

## 7. 训练脚本输入规则

训练脚本输入应使用新的完整 label JSON：

```txt
new_train_label_6class_plus_rare_animal.json
```

训练脚本读取时：

```txt
JSON key 决定主类别
prompts["animal"] 决定 animal mask
不要把 rare_sources 当作新类别
不要把 fox/dog/lion 当作 class_id
```

类别列表仍然保持：

```python
CLASSES = [
    "person",
    "tree",
    "building",
    "animal",
    "car",
    "computer",
]
```

训练时如果看到：

```json
"animal": {
  "hit": true,
  "rle": {...},
  "prompt_pool": ["animal", "fox"]
}
```

它的含义是：

```txt
这个 mask 的主类别是 animal
训练时可以用 animal prompt
如果 prompt 扩展开启，也可以额外用 fox prompt 监督同一个 animal mask
```

不是新增 fox 类。

------

## 8. Prompt 扩展训练规则

Prompt 扩展的作用是：

```txt
同一个 mask 可以用多个 prompt 训练
提升模型对同义词/细分类 prompt 的响应能力
```

例如：

```txt
animal → animal mask
fox → 同一个 animal mask
dog → 同一个 animal mask
```

但 prompt 扩展必须分级使用。

### 普通 base animal

普通 base animal 建议：

```txt
prompt_pool = ["animal"]
```

可以不扩展，避免过度引入噪声。

### Rare A animal

A 类细分类质量相对高，可以扩展：

```txt
prompt_pool = ["animal", source_prompt]
```

例如：

```txt
source_prompt = fox
prompt_pool = ["animal", "fox"]
```

训练时可以生成两个训练样本：

```txt
animal → mask
fox → same mask
```

### Rare B/C animal

B/C 不使用 source_prompt：

```txt
prompt_pool = ["animal"]
```

原因：

```txt
B/C 的 source_prompt 不可靠，容易把 dog 当 fox、把噪声当 animal。
如果继续用 source_prompt 训练，会污染细分 prompt 的语义。
```

------

## 9. 训练脚本应实现的逻辑

训练脚本读取每个 prompt 时：

```txt
1. 遍历 prompts
2. 只处理主类 key：person/tree/building/animal/car/computer
3. 如果 hit=true 且 rle 有效，则作为正样本
4. 默认 prompt_text = 当前 key
5. 如果当前 key 是 animal 且存在 prompt_pool，则按 prompt_pool 扩展训练 prompt
6. 但 class_id 仍然是 animal
```

伪代码：

```python
for label_name, info in entry["prompts"].items():
    if label_name not in CLASSES:
        continue
    if not info.get("hit"):
        continue

    mask = decode_rle(info["rle"])
    class_id = class_to_id[label_name]

    prompt_pool = info.get("prompt_pool", [label_name])

    for prompt_text in prompt_pool:
        add_train_sample(
            image=image,
            prompt=prompt_text,
            mask=mask,
            class_id=class_id,
        )
```

关键点：

```txt
prompt_text 可以是 fox
但 class_id 必须还是 animal
```

------

## 10. 推理脚本规则

推理阶段建议做 prompt 映射。

如果官方输入的是主类：

```txt
animal
```

正常用 animal 推理。

如果官方输入的是细分类：

```txt
fox / dog / wolf / lion / tiger / horse / monkey ...
```

内部统一映射成：

```txt
animal
```

推荐映射表：

```python
PROMPT_ALIAS = {
    "fox": "animal",
    "wolf": "animal",
    "dog": "animal",
    "lion": "animal",
    "tiger": "animal",
    "horse": "animal",
    "monkey": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    "bear": "animal",
    "duck": "animal",
    "pig": "animal",
    "rabbit": "animal",
}
```

推理伪代码：

```python
raw_prompt = task_prompt
model_prompt = PROMPT_ALIAS.get(raw_prompt, raw_prompt)

mask = model.predict(image, model_prompt)
```

也就是说：

```txt
官方给 fox
内部用 animal 推理
输出仍按任务要求返回
```

这样最稳，因为训练主体学的是 animal 主类。

------

## 11. 为什么训练扩展 + 推理映射可以同时保留

两者作用不同，不冲突。

```txt
训练 prompt 扩展：
让模型见过 fox/dog/lion 这些文本和 animal mask 的对应关系

推理 prompt 映射：
保证最终预测时稳定走 animal 主类能力
```

当前推荐优先级：

```txt
第一优先：新 label 补 animal mask recall
第二优先：推理时细分 prompt → animal
第三优先：只对 A 类 rare animal 做 source_prompt 扩展
```

如果时间紧，至少要做：

```txt
新 label 补 animal
推理 alias 映射
```

A 类 source_prompt 扩展是锦上添花。

------

## 12. 最终推荐方案

### 训练输入

```txt
使用 new_train_label_6class_plus_rare_animal.json
```

### JSON 结构

```txt
只保留 6 个主类 prompt key
不新增 fox/dog/lion 顶层 key
细分来源写入 animal 的 prompt_pool / rare_sources
```

### 训练策略

```txt
base animal：用 animal 训练
rare A animal：用 animal + source_prompt 训练
rare B/C animal：只用 animal，train 阶段默认可 drop
```

### 推理策略

```txt
如果输入是细分动物 prompt，先映射成 animal 再推理
```

### 核心原则

```txt
细分 prompt 可以作为文本增强
但不能作为新类别
最终类别仍然是 animal
```