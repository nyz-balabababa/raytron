# clipseg-v5 技术文档

> 2026-06-13 更新：
> 当前 v5 推荐使用的新主线脚本是：
> `config_opt.py`、`dataset_opt.py`、`train_opt.py`、`infer_opt.py`、`sweep_opt.py`
>
> 这套脚本对应的路线是：
> `CLIPSeg-11 Opt Decoder-Only + Infer-Opt`
>
> 旧的 `config_clipseg_11_opt.py / train_clipseg_11_opt.py / clipseg_infer_eval_11_opt.py / sweep_thresholds_clipseg_11_opt.py`
> 仍保留在目录中，但现在应视为早期草稿版本，不再是推荐入口。

## 1. 概述

`clipseg-v5` 是当前面向 11 类红外分割任务的 `CLIPSeg` 优化版训练线。

这条线的定位是：

1. 保留 `CLIPSeg` 的原始工作方式，即 `image + text prompt -> binary mask`
2. 不改 `backbone`，不改 `text encoder`，不改 `CLIPSeg` 主体结构
3. 在当前 11 类伪标签和数据目录不变的前提下，做一条低风险、可直接和原版 11 类 `CLIPSeg` 对比的优化线

当前 `v5` 对应的路线是：

`CLIPSeg 原架构 + prompt pool + 轻负样本 + BCE/Dice/Focal + rare oversample + 每类阈值 + 多 prompt 融合 + 按类后处理`

当前 11 类类别顺序不在 `v5` 内手写，而是直接继承当前 11 类主配置：

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

`computer` 不在当前 11 类集合内，因此不会在 `v5` 中被训练或评估。

## 2. 代码结构

`v5` 当前新增的主文件如下：

- [config_clipseg_11_opt.py](/d:/nyz/raytron_project/src/train/clipseg-v5/config_clipseg_11_opt.py)
- [train_clipseg_11_opt.py](/d:/nyz/raytron_project/src/train/clipseg-v5/train_clipseg_11_opt.py)
- [clipseg_infer_eval_11_opt.py](/d:/nyz/raytron_project/src/train/clipseg-v5/clipseg_infer_eval_11_opt.py)
- [sweep_thresholds_clipseg_11_opt.py](/d:/nyz/raytron_project/src/train/clipseg-v5/sweep_thresholds_clipseg_11_opt.py)

职责划分：

- `config_clipseg_11_opt.py`
  - 继承当前 11 类配置中的类别顺序、数据路径和过滤阈值
  - 定义 `prompt pool`、训练损失权重、负样本策略、过采样、默认阈值、后处理参数
- `train_clipseg_11_opt.py`
  - 实现训练数据集、损失、训练循环、验证循环、续训和旧权重热启动
- `clipseg_infer_eval_11_opt.py`
  - 实现任务集推理、可选多 prompt 融合、按类阈值与按类后处理
- `sweep_thresholds_clipseg_11_opt.py`
  - 实现每类独立 threshold sweep，输出最优阈值和完整评估日志

## 3. 配置原则

### 3.1 类别与数据路径

`v5` 不重新定义 11 类类别顺序，而是从 `src/train/clipseg-v4/config_clipseg.py` 动态读取当前 `CLASSES`、`PRED_JSON`、`VAL_PRED_JSON`、`TRAIN_LIST`、`VAL_LIST` 和 `IMAGE_ROOT`。

这样做的目的有两个：

1. 保证 `v5` 和当前 11 类主线严格使用同一份数据切分
2. 避免后续主配置调整后，`v5` 出现类别顺序漂移

### 3.2 训练损失

当前优化版的 hard loss 为：

```python
total_loss = 1.0 * BCE + 1.1 * Dice + 0.10 * Focal
```

对应参数：

```python
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.1
FOCAL_WEIGHT = 0.10
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
```

其中 `Focal` 是二值分割版 focal loss，不改 `CLIPSeg` 输出头结构，只在 loss 端叠加。

### 3.3 负样本策略

当前 `v5` 采用轻负样本：

```python
NEGATIVE_SAMPLE_RATIO = 0.06
NEGATIVE_SAMPLE_WEIGHT = 0.08
```

含义：

- 训练时只保留一小部分 `hit=false` 样本
- 这些负样本仍参与损失
- 但其样本权重显著低于正样本，避免过强拒识把 recall 压低

验证集固定不引入负样本。

### 3.4 Prompt Pool

`v5` 支持每类多个 prompt alias。

设计原则：

1. 每个类别至少有一个主 prompt
2. 若存在 alias，则训练时随机抽样一个 prompt
3. 验证和推理时默认仍以主 prompt 为主
4. 可选开启多 prompt 融合

当前融合策略为：

```python
final_prob = 0.7 * prob_main + 0.3 * mean(prob_aliases)
```

`max` 融合保留为可选模式，但默认不开。

### 3.5 稀有类过采样

当前 `v5` 在数据集层面对正样本做复制式 oversample。

当前重点照顾的类别有：

- `animal`
- `trash can`
- `window`
- `door`
- `fence`
- `pole_light`

是否生效取决于该类是否存在于当前 `CLASSES` 中。

### 3.6 默认阈值

`v5` 使用一份独立的按类默认阈值：

- `person`: `0.55`
- `car`: `0.55`
- `building`: `0.60`
- `tree`: `0.50`
- `animal`: `0.45`
- `trash can`: `0.40`
- `window`: `0.40`
- `door`: `0.40`
- `fence`: `0.40`
- `pole_light`: `0.35`
- `motorcycle`: 若未单独配置则回退到基础阈值或默认值

阈值 sweep 默认网格为：

```python
[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
```

### 3.7 后处理

`v5` 在验证与推理侧增加了按类连通域过滤。

策略要点：

1. 不是对所有类别使用同一个面积阈值
2. `building`、`tree` 更偏向去噪
3. `trash can`、`animal`、`window`、`door`、`pole_light` 更偏向保护小目标

当前后处理使用：

- 二值化阈值
- 连通域分析
- 按类 `min_area` 过滤

## 4. 训练数据处理流程

训练数据按如下流程构建：

1. 读取 `trainval_label.json` 或 manifest
2. 读取 `trainval_list.txt`
3. 只保留在列表中的图片
4. 按 `prompts` 拆成单类样本
5. 过滤不在当前 `CLASSES` 中的类别
6. 过滤空 mask
7. 可选做 score 过滤
8. 对正样本缓存 RLE 解码后的 PNG
9. 按 `RARE_OVERSAMPLE` 对稀有类复制样本
10. 按 `NEGATIVE_SAMPLE_RATIO` 抽取少量负样本

最终每个训练样本的语义仍然是：

`一张图 + 一个文本 prompt + 一个单通道 binary mask`

这点是 `v5` 与固定多通道语义分割模型的根本区别。

## 5. 图像预处理流程

训练、验证和推理都复用同一套红外预处理逻辑：

1. 读图后统一转灰度
2. 自动判断伪彩或黑热
3. 必要时做反色统一
4. 低对比场景做 `CLAHE`
5. 高噪声场景做降噪
6. 模糊场景做锐化
7. 等比例缩放到长边 `768`
8. 短边补零 pad 到 `768 x 768`
9. 灰度复制成 3 通道
10. 使用 `CLIP` 的均值和方差做归一化

这样可以保证：

- 训练输入
- 验证输入
- 推理输入

三者的图像域保持一致。

## 6. 训练时每个 batch 的处理流程

训练循环中，每个 batch 的处理步骤如下：

1. `Dataset` 返回 `image, prompt, mask, valid_mask, class_name`
2. 对 `prompt` 做 tokenizer 编码
3. `CLIPSeg` 前向输出单通道 logits
4. 若输出分辨率不一致，则插值到目标 mask 尺寸
5. 计算 `BCE loss`
6. 计算 `Dice loss`
7. 计算 `Binary Focal loss`
8. 对正样本应用 `LOSS_WEIGHT_FLOOR`
9. 对负样本保留较低原始权重
10. 加权汇总总损失并反传

边界弱监督的作用位置在于：

- `valid_mask=1` 的区域参与监督
- `valid_mask=0` 的边界环带不参与 loss

这样可以减弱伪标签边缘不准对训练的伤害。

## 7. 验证流程

验证阶段与训练不同的点有：

1. 不做随机 prompt 采样
2. 不引入负样本
3. 默认使用主 prompt
4. 可选开启多 prompt 融合
5. 二值化后走按类后处理

验证指标包括：

- `mIoU`
- `pos_mIoU`
- `Dice`
- `precision`
- `recall`
- `per-class IoU`
- `per-class prediction count`
- `per-class empty prediction ratio`

其中：

- `mIoU` 是对所有样本的类别 IoU 均值
- `pos_mIoU` 更关注有正目标的样本

## 8. 推理流程

推理脚本 `clipseg_infer_eval_11_opt.py` 的处理流程如下：

1. 读取任务 json
2. 按图片聚合多个 prompt
3. 对每张图做图像预处理
4. 对每个 prompt 或类别执行 `CLIPSeg` 前向
5. 若启用多 prompt 融合，则做加权融合
6. 按类使用 `VAL_THRESHOLDS`
7. 做按类连通域后处理
8. 将方形画布上的预测映射回原图尺寸
9. 编码成 `RLE`
10. 输出提交格式和 prompt 风格 json

当前推理版已经支持：

- 单 prompt
- 多 prompt 融合
- 按类阈值
- 按类后处理

当前尚未加入：

- TTA
- 类间冲突抑制

## 9. 阈值扫描流程

`sweep_thresholds_clipseg_11_opt.py` 用于对验证集做每类独立阈值搜索。

流程如下：

1. 加载指定 checkpoint
2. 跑完整个验证集，收集每个样本的 probability map
3. 按类别分桶保存 `prob + gt`
4. 针对每个类别枚举阈值网格
5. 分别计算：
   - overall IoU
   - positive-only IoU
   - prediction count
   - empty prediction ratio
6. 汇总出两套阈值：
   - `best overall mIoU thresholds`
   - `best pos_mIoU thresholds`

脚本输出的 JSON 中会保留：

- 配置阈值评估结果
- 每类 sweep 完整日志
- 每类最佳 overall 阈值
- 每类最佳 positive 阈值

## 10. 权重加载与续训

`v5` 支持两类权重加载方式。

### 10.1 旧权重 warm start

用于把旧 `CLIPSeg` 权重作为初始化：

- 默认路径为 `test/train_output/clipseg_v1/best.pt`
- 若结构兼容，则按 `strict=False` 加载
- 会打印：
  - `Loaded pretrained CLIPSeg weights from ...`
  - `Missing keys: ...`
  - `Unexpected keys: ...`

### 10.2 正常续训

优先级为：

1. `--resume`
2. 当前 `run_name` 下的 `last.pt`
3. 当前 `run_name` 下的 `best.pt`

支持：

- 恢复模型
- 恢复 optimizer
- 恢复 scheduler
- 恢复 history

若使用 `--resume_weights_only`，则只恢复模型权重，不恢复训练状态。

## 11. 输出目录与结果文件

训练输出目录：

`test/train_output/<run_name>/`

默认 `run_name`：

`clipseg_11cls_opt_v1`

主要输出包括：

- `best.pt`
- `last.pt`
- `results.csv`
- `results.png`

checkpoint 内还会保存：

- `run_name`
- `val_thresholds`
- `prompt_pool`
- `use_prompt_fusion`
- `fusion_mode`

## 12. 当前路线边界

当前 `v5` 已实现的是一条单主线 `CLIPSeg opt`，不是所有备选路线的合集。

当前已经实现：

- `prompt pool`
- `轻负样本`
- `BCE + Dice + Focal`
- `rare oversample`
- `多 prompt 融合`
- `threshold sweep`
- `按类后处理`
- `旧权重 warm start`

当前尚未实现：

- `decoder-only fine-tune`
- `prompt consistency loss`
- `absent class negative`
- `TTA`
- `类间冲突抑制`

也就是说，`v5` 当前是一条低风险主线优化版，而不是一个包含所有试验分支的统一框架。

## 13. 推荐使用方式

如果当前时间紧，建议把 `v5` 当作单主线使用：

1. 先训练 `clipseg_11cls_opt_v1`
2. 用同一权重跑 threshold sweep
3. 把 sweep 后的阈值用于最终验证或推理
4. 如有必要，再在此基础上合并 `infer-opt` 的附加增强项

如果后续要做路线合并，建议优先考虑：

1. 把 `infer-opt` 中的 `TTA` 合进当前推理脚本
2. 再决定是否增加一个可切换的 `decoder-only` 模式

这份文档描述的是当前实现状态；后续如果 `v5` 做路线合并，应同步更新本文件。
