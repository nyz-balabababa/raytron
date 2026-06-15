# 伪标签与 Prompt 实验总结

## 1. 文档目的

这份文档用于系统整理本项目在伪标签阶段做过的两类实验：

1. 早期 **6 类 prompt 选择与对照实验**
2. 后续从 **6 类 / 5 类主线** 逐步过渡到 **11 类 clean rare 标签体系** 的过程

目标不是复述单个脚本，而是把：

- 用过哪些 prompt
- 为什么保留 / 为什么丢弃
- 有哪些定量依据
- 最后为什么会走到当前 11 类

串成一条完整证据链，方便后续写论文和方法章节。

---

## 2. 证据来源

本总结主要基于四部分材料：

### 2.1 早期伪标签实验输出

- `test/sam3_label_old/`

这里保留了大量早期 prompt 扫描结果，包括：

- `val_tasks1/`：全验证集 6 类主 prompt
- `val_tasks2/`：全验证集同义词 / 近义词对照
- `data6_tasks/`：细粒度动物 prompt
- `train_d5&7_tasks/`、`val_d5&7_tasks/`：工控 / 尾类专项 prompt
- `d57_new/`：更多候选 rare prompt 的筛查结果

### 2.2 前期工作与 prompt 决策文档

- `docs/前期准备/训练前工作流程总结.md`
- `docs/前期准备/验证集分配.md`

### 2.3 伪标签处理与扩类方案文档

- `docs/伪标签处理/长尾分布.md`
- `docs/伪标签处理/训练方案-v1.md`
- `docs/伪标签处理/训练方案-v2.md`
- `docs/伪标签处理/animal训练方案.md`

### 2.4 当前代码

- `src/prompt/`
- `src/label/`
- `src/ESAM-CCLIP-11/config_esam_cclip_11.py`

其中当前 11 类最终定义来自：

- `src/ESAM-CCLIP-11/config_esam_cclip_11.py`

---

## 3. 当前视角下的整体演进

如果从最终结果往回看，项目的伪标签路线并不是“直接从 6 类扩成 11 类”，而是经历了三步：

1. **6 类主线 prompt 验证**
   - 先确认哪些基础类在 SAM3 teacher 上最稳定
2. **主线收缩**
   - 发现 `computer` 太弱，先从“主线类”里降级为尾类专项
   - 主线实际收敛成 `person / car / building / tree / animal`
3. **clean rare 扩展**
   - 再通过 rare prompt、同义词、人工可视化筛查补进新类
   - 最后形成 11 类：
     - `person`
     - `car`
     - `building`
     - `tree`
     - `animal`
     - `trash can`
     - `window`
     - `door`
     - `fence`
     - `pole_light`
     - `motorcycle`

所以更准确地说，这条线是：

> **先验证 6 类，后收缩到 5 类主干，再扩展为 11 类 clean rare。**

这也是为什么后续很多文档里会同时出现“6 类主线”“5 类主线”“11 类 clean rare”三种说法。

---

## 4. 6 类 prompt 的早期验证

## 4.1 实验设置

最早的主线 prompt 组来自：

- `test/sam3_label_old/val_tasks1/stats_val_tasks1.csv`

对应 prompt：

```text
person
car
building
tree
computer
animal
```

验证集规模：

- `4524` 张图

这是最核心的一组基线，因为它决定了项目最初的主类别集合。

## 4.2 结果

| prompt | total_images | hits | activation_rate | avg_score | avg_instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| person | 4524 | 2800 | 0.6189 | 0.8273 | 4.19 |
| car | 4524 | 2163 | 0.4781 | 0.8688 | 15.95 |
| building | 4524 | 2125 | 0.4697 | 0.7909 | 5.27 |
| tree | 4524 | 2114 | 0.4673 | 0.7189 | 8.61 |
| animal | 4524 | 395 | 0.0873 | 0.7880 | 2.42 |
| computer | 4524 | 23 | 0.0051 | 0.6439 | 3.43 |

## 4.3 结论

这组实验直接给出了早期 6 类主线的第一层证据：

- `person / car / building / tree` 都是高激活率类，属于主干类
- `animal` 虽然激活率只有 `8.73%`，但仍明显高于 `computer`
- `computer` 极弱，只有 `0.51%`

这意味着：

1. `computer` 从一开始就不是一个“稳定主类”
2. `animal` 虽然少，但还有作为主线稀有类保留的价值
3. 如果后续要继续扩类，`computer` 更像一个“尾类专项候选”，而不是必须和主干类一起训练的核心类

---

## 5. 同义词 / 近义词对照：为什么最后选精确类别词

## 5.1 实验设置

同义词对照来自：

- `test/sam3_label_old/val_tasks2/stats_val_tasks2.csv`

对应 prompt：

```text
human
vehicle
house
plant
pc
wildlife
```

它和早期 6 类 prompt 的关系大致是：

| 主类 prompt | 对照 prompt |
| --- | --- |
| person | human |
| car | vehicle |
| building | house |
| tree | plant |
| computer | pc |
| animal | wildlife |

## 5.2 结果

| prompt | hits | activation_rate | avg_score | avg_instances |
| --- | ---: | ---: | ---: | ---: |
| human | 1595 | 0.3526 | 0.7949 | 2.25 |
| vehicle | 2317 | 0.5122 | 0.8714 | 16.88 |
| house | 409 | 0.0904 | 0.6840 | 2.13 |
| plant | 1064 | 0.2352 | 0.6753 | 6.61 |
| pc | 6 | 0.0013 | 0.5459 | 1.33 |
| wildlife | 69 | 0.0153 | 0.7078 | 1.96 |

## 5.3 与主类 prompt 的直接对比

| 语义 | 精确类名 | activation_rate | 同义词/近义词 | activation_rate | 结论 |
| --- | --- | ---: | --- | ---: | --- |
| person | person | 0.6189 | human | 0.3526 | `person` 明显更强 |
| car | car | 0.4781 | vehicle | 0.5122 | 两者接近，`vehicle` 略高 |
| building | building | 0.4697 | house | 0.0904 | `building` 远强于 `house` |
| tree | tree | 0.4673 | plant | 0.2352 | `tree` 明显更强 |
| computer | computer | 0.0051 | pc | 0.0013 | 两者都弱，`pc` 更差 |
| animal | animal | 0.0873 | wildlife | 0.0153 | `animal` 明显更强 |

## 5.4 结论

这组实验是后面 prompt 设计中最重要的一条规律：

> **在当前红外 teacher 条件下，SAM3 对“精确类别词”的响应，通常明显强于宽泛同义词。**

尤其体现在：

- `person > human`
- `building >> house`
- `tree > plant`
- `animal >> wildlife`

只有 `car / vehicle` 这组接近，说明车辆类语义范围更宽时也能稳定命中。

因此，后续项目没有走“全面替换成自然语言同义词”的路线，而是采取：

- 主训练标签仍保留精确类名
- 同义词更多用于：
  - prompt prototype
  - rare 类补扫
  - alias 映射

而不是直接取代主类词本身。

---

## 6. 为什么 `computer` 最终退出主线

`computer` 的问题不是后来才出现，而是早期 prompt 实验中就已经埋下了证据。

## 6.1 全验证集上非常弱

来自 `val_tasks1`：

- activation_rate：`0.0051`

## 6.2 全训练集上也很弱

来自 `test/sam3_label_old/train_tasks/stats_train_tasks.csv`：

| prompt | total_images | hits | activation_rate | avg_score | avg_instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| computer | 39094 | 151 | 0.0039 | 0.7074 | 5.82 |

也就是说，训练集里 `computer` 的覆盖率也只有 `0.39%`。

## 6.3 在 d5&7 工控场景里稍好，但仍然属于尾类

来自：

- `test/sam3_label_old/val_d5&7_tasks/stats_val_d5&7_tasks.csv`
- `test/sam3_label_old/train_d5&7_tasks/stats_train_d5&7_tasks.csv`

### val d5&7

| prompt | total_images | hits | activation_rate | avg_score |
| --- | ---: | ---: | ---: | ---: |
| computer | 1698 | 12 | 0.0071 | 0.6686 |
| trash can | 1698 | 104 | 0.0612 | 0.6873 |
| cable connector | 1698 | 49 | 0.0289 | 0.5731 |
| fire hydrant | 1698 | 9 | 0.0053 | 0.5717 |

### train d5&7

| prompt | total_images | hits | activation_rate | avg_score |
| --- | ---: | ---: | ---: | ---: |
| computer | 8127 | 92 | 0.0113 | 0.7054 |
| trash can | 8127 | 305 | 0.0375 | 0.6268 |
| cable connector | 8127 | 99 | 0.0122 | 0.6052 |
| fire hydrant | 8127 | 19 | 0.0023 | 0.7173 |
| circuit board | 8127 | 0 | 0.0000 | 0.0000 |

## 6.4 解释

这些结果说明：

1. `computer` 确实存在，但覆盖率始终很低
2. 在全局主线里，它远弱于 `person/car/building/tree`
3. 在工控场景里，它也没有强到足以自然成为主类
4. `trash can`、`cable connector` 等类在专项场景中的可见性，反而比 `computer` 更值得继续挖

因此项目后面逐步形成了这个认识：

> **`computer` 可以做尾类专项，但不适合作为稳定主线类长期绑在主模型里。**

这也是为什么后续路线里经常出现：

- 先降级 `computer`
- 再补 `trash can / window / door / fence / pole_light / motorcycle`

---

## 7. 细粒度动物实验：为什么最后仍保留 `animal` 而不是拆细类

## 7.1 实验设置

细粒度动物实验来自：

- `test/sam3_label_old/data6_tasks/stats_data6_tasks.csv`

对应 11 个 prompt：

```text
lion
bear
wild boar
duck
swan
bird
deer
fox
wolf
zebra
monkey
```

数据子集：

- `900` 张 `data6` 图像

## 7.2 结果

按 activation_rate 排序：

| prompt | hits | activation_rate | avg_score | avg_instances |
| --- | ---: | ---: | ---: | ---: |
| deer | 75 | 0.0833 | 0.8723 | 1.68 |
| fox | 65 | 0.0722 | 0.7860 | 1.66 |
| bird | 48 | 0.0533 | 0.7898 | 4.56 |
| wolf | 22 | 0.0244 | 0.7590 | 1.55 |
| lion | 19 | 0.0211 | 0.7699 | 1.32 |
| duck | 14 | 0.0156 | 0.7978 | 11.50 |
| monkey | 13 | 0.0144 | 0.7541 | 1.38 |
| swan | 9 | 0.0100 | 0.7852 | 12.89 |
| zebra | 8 | 0.0089 | 0.6520 | 1.38 |
| wild boar | 4 | 0.0044 | 0.7384 | 5.75 |
| bear | 2 | 0.0022 | 0.7003 | 1.00 |

## 7.3 结论

这组实验说明了两件事：

### 第一，细粒度动物可以作为“补召回 prompt”

像：

- `deer`
- `fox`
- `bird`
- `wolf`

这些 prompt 的命中率并不算差，说明用它们挖补充性伪标签是有价值的。

### 第二，细粒度动物不适合直接变成最终训练类别

原因是：

1. 各细类分布极不均衡
2. 很多类只在少数图里激活
3. 一些鸟类 prompt 的 `avg_instances` 很高，容易带来碎片或噪声
4. 训练目标是补 `animal` 类，而不是把 `animal` 再拆成 10 多个小类

所以后来 `docs/伪标签处理/训练方案-v2.md` 采取的是：

- 用细粒度动物 prompt 生成伪标签
- 再按 A/B/C 做质量分档
- 最终统一映射回 `animal`

也就是：

> **细粒度 prompt 用来挖标签，不直接变成最终类别。**

---

## 8. 更多 rare prompt 候选：哪些试过，哪些被放弃

## 8.1 d57_new 的额外候选

`test/sam3_label_old/d57_new/stats_val(wire).csv` 里保留了一组很有代表性的“高召回但高风险”候选：

| prompt | raw_activation_rate | filtered_activation_rate | filtered_avg_score |
| --- | ---: | ---: | ---: |
| wire | 0.9611 | 0.1196 | 0.7497 |
| electronics | 0.9517 | 0.2049 | 0.7259 |
| chair | 0.9176 | 0.2108 | 0.7668 |
| electronic component | 0.8722 | 0.1413 | 0.7189 |

## 8.2 这组结果说明什么

这组数据非常典型，说明：

1. 一些 prompt 在原始 proposal 层面会“到处都能触发”
2. 经过阈值 / 后处理过滤后，命中率虽然下降，但仍可能过宽
3. 它们适合做候选探索，不适合直接当主类

尤其是：

- `wire`
- `electronics`
- `electronic component`

这类词语义太泛，极易把亮线、局部结构、边缘噪声都吸进去。

`chair` 也是类似问题：

- filtered 命中率看上去不低
- 但语义外延太宽，很容易和背景结构、桌椅、建筑轮廓混淆

因此在后续策略里，这类 prompt 大多被归入三种处理方式之一：

1. 直接丢弃
2. 仅保留 top 样本做人审
3. 作为 hard negative 来源，而不是正样本来源

---

## 9. 伪标签处理流程是如何逐步升级的

早期 prompt 试验只是第一步，后面真正让伪标签体系成熟的，是 `src/prompt` 和 `src/label` 这套工具链。

## 9.1 任务生成：`src/prompt/gen_val_tasks.py`

作用：

- 从 `train_list.txt` / `val_list.txt` 生成标准任务 JSON
- 将图片列表和 prompt 列表展开成：
  - `image_path`
  - `text_prompt`
  - `ann_id`

这让 prompt 试验从“手工改脚本”变成了标准化流水线。

## 9.2 teacher 生成：`src/prompt/generate_sam3_labels.py`

作用：

- 基于官方 SAM3 推理链路生成 prompt-style 伪标签
- 保留之前在 `prompt_test(9).py` 里验证有效的红外预处理经验

关键点：

- 伪彩色转灰度
- 黑热 / 白热统一
- 动态 CLAHE
- 动态降噪
- 动态锐化
- 按类阈值覆盖
- 每个 prompt 的 top-k 实例限制

这意味着后期的伪标签已经不是“直接把图丢进官方模型”，而是带有一整套针对红外域的预处理策略。

## 9.3 坏图管理：`src/prompt/build_unified_denylist.py`

作用：

- 合并自动异常图
- 人工坏图
- 后续分析得到的整图坏样本

这一步减少了明显坏图对 teacher 的污染。

## 9.4 新旧 teacher 对比：`src/label/compare_pseudo_labels.py`

作用：

- 比较旧版 teacher 与新版 teacher 的差异
- 判断哪些 prompt 在变保守、哪些变激进
- 找分歧样本

这一步标志着项目已经从“只看激活率”，升级为“看新旧伪标签一致性与差异”。

## 9.5 质量分档：`src/label/refine_pseudo_labels.py`

作用：

- 对当前主版本 teacher 做 A/B/C 分档
- 输出：
  - `pseudo_A.json`
  - `pseudo_B.json`
  - `pseudo_C.json`

这意味着后期伪标签已经不是简单二元的“用/不用”，而是变成了：

- 高质量强正样本
- 可疑但可低权重使用样本
- 高风险样本

## 9.6 策略层：`src/label/pseudo_label_policy.py`

作用：

- 封装不同类别的选择逻辑
- 例如：
  - tiny target 保护
  - building 更严格清理
  - 正负样本权重

这一步是从“经验判断”过渡到“规则化准入策略”的关键。

## 9.7 manifest 构建：`src/label/build_train_manifest.py`

作用：

- 把旧 teacher、新 teacher、A/B/C 分档、denylist 统一融合成训练 manifest

也就是说，后期标签已经不是一份单纯的 prompt-style json，而是：

> **一份带有质量、来源、权重、是否纳入训练等元信息的训练样本清单。**

---

## 10. 为什么后来要从 6 类走向扩类

从早期 6 类实验来看，`person/car/building/tree` 很稳，`animal` 有一定稀有价值，`computer` 很弱。

如果只停留在这一步，问题有两个：

1. 主线类别覆盖面不足
2. 数据里确实存在更多结构化目标，但没有被训练目标吸收

后续扩类的动机主要来自：

### 10.1 数据中真实存在更多目标

在 `d5&7`、`data6`、工业/室内场景里，除了原 6 类外，还能观察到：

- `trash can`
- `window`
- `door`
- `fence`
- `pole_light`
- `motorcycle`

这些类并不是纯幻想，而是通过：

- rare prompt 扫描
- 可视化人工检查
- clean rare 方案

逐步沉淀出来的。

### 10.2 不是所有新 prompt 都要直接变训练类

`docs/伪标签处理/长尾分布.md` 里已经明确提出过一个很重要的原则：

> 先做 prompt bank 扫描，再决定哪些类真的存在、哪些类 teacher 稳定、哪些类值得留下。

候选里其实试过很多：

```text
computer, laptop, monitor, screen, keyboard
trash can, dustbin, garbage bin, bin
chair, table, bench
bicycle, motorcycle, bus, truck
traffic light, street light, pole, sign, billboard
fence, window, door
```

最后不是全部保留，而是经过“质量 + 可学性 + 训练反馈”再收缩。

---

## 11. 从“6 类 / 5 类主线”到“12 类候选”，再固定 11 类

这一步可以概括成：

## 11.1 第一阶段：6 类基线

早期类集合：

```text
person
car
building
tree
computer
animal
```

## 11.2 第二阶段：主线收缩为 5 类 + rare 专项

由于 `computer` 太弱，项目逐渐把主线理解为：

```text
person
car
building
tree
animal
```

而把：

```text
computer
trash can
...
```

转为尾类专项。

## 11.3 第三阶段：rare prompt 扩展，形成 12 类或近似 12 类候选体系

来自 `docs/文档编写指引.md` 的明确记录：

- 曾形成包含 `computer` 的 12 类或近似 12 类扩展标签体系
- 目的在于提升类别覆盖范围

新增类主要包括：

```text
trash can
window
door
fence
pole_light
motorcycle
```

同时还尝试过同义词 / 归并：

```text
truck / bus → car
dustbin / garbage bin → trash can
laptop / monitor / keyboard → computer
pole / street light / utility pole → pole_light
```

## 11.4 第四阶段：删掉 computer，固定 11 类

最终 11 类来自当前配置文件：

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

删掉 `computer` 的原因，在文档和早期实验里是一致的：

1. 样本少
2. 激活率低
3. teacher 伪标签不稳定
4. 训练时容易被主类或背景压制

换句话说，项目不是“突然决定删掉 computer”，而是：

> **从早期 prompt 统计，到中期 rare 专项，再到后期训练反馈，三轮证据都指向 `computer` 不适合作为最终主类保留。**

---

## 12. 11 类不是“所有候选都成功”，而是“筛选后的稳定子集”

这一点非常重要，后续论文里最好写清楚。

最终 11 类并不是“所有试过的 rare prompt”：

- `wire`
- `electronics`
- `electronic component`
- `chair`
- `fire hydrant`
- `circuit board`
- `screen`
- `bin`
- `traffic light`
- `sign`

这些都没有直接进入最终主类集合。

原因主要是三类：

### 12.1 语义太泛，容易误触

例如：

- `wire`
- `electronics`
- `electronic component`

### 12.2 虽然命中，但训练价值不稳定

例如：

- `chair`
- `screen`
- `bin`

### 12.3 可以当同义词 / alias，但不适合成为独立标签

例如：

- `dustbin / garbage bin` 最终归到 `trash can`
- `truck / bus` 更适合补 `car`
- `pole / street light / utility pole` 更适合归到 `pole_light`

因此，最终 11 类其实代表的是：

> **在可见性、teacher 稳定性、类别边界和训练可学性之间综合筛过一轮后的结果。**

---

## 13. 可以直接写进论文的结论

## 13.1 关于早期 prompt 选择

可以写成：

- 对 4524 张验证图像进行了基于 SAM3 teacher 的 prompt 激活统计
- 结果表明 `person/car/building/tree` 为高稳定基础类
- `animal` 虽稀有但具有保留价值
- `computer` 激活率极低，不适合作为长期主线类

## 13.2 关于同义词实验

可以写成：

- 对精确类别词与宽泛同义词进行了对照实验
- `person > human`、`building >> house`、`tree > plant`、`animal >> wildlife`
- 说明红外 teacher 对精确类名响应更稳定
- 因而最终采用“主类词保守、同义词用于补扫和 alias”的策略

## 13.3 关于细粒度动物

可以写成：

- 细粒度动物 prompt 对 `animal` 类有补标签价值
- 但不适合作为最终独立训练类别
- 因此采用“细粒度 prompt 挖标签，最终统一映射回 animal”的方案

## 13.4 关于 rare prompt 扩类

可以写成：

- 项目并非直接把所有新 prompt 并入训练
- 而是先构建 rare prompt bank，再结合可视化和训练反馈筛选
- 最终从候选 12 类 / 近似 12 类体系中收缩为稳定的 11 类 clean rare 标签

## 13.5 关于 `computer`

可以写成：

- `computer` 在全训练 / 全验证上的激活率持续偏低
- 在专项场景中也未表现出足够稳定的 teacher 行为
- 因此从最终主训练类别中删除，仅保留为历史专项探索类

---

## 14. 最终总结

本项目在伪标签阶段的核心决策链可以概括为：

1. **用 6 类 prompt 先验证基础主类可学性**
2. **用同义词对照证明“精确类名优于宽泛词”**
3. **把 `computer` 从主线中降级为尾类专项**
4. **用细粒度动物和 rare prompt bank 补充新类别候选**
5. **通过可视化、规则筛选和训练反馈，从候选扩类中收缩到稳定 11 类**

最终，项目并不是简单“扩类越多越好”，而是形成了这样一套更稳的原则：

> **主类靠稳定 prompt，rare 类靠可视化筛选与 clean rare 补充；  
> 同义词优先用于挖标签和增强鲁棒性，而不是无条件变成训练类别；  
> 只有同时满足“存在、稳定、可学”的类别，才会进入最终主类集合。**

这也是当前 11 类 clean rare 体系形成的根本原因。
