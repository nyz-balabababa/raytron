# clipseg-v4 技术文档

## 1. 概述

`clipseg-v4` 是一套面向 11 类红外语义分割的训练方案。它延续 `docs/模型优化/clip优化.md` 的主线思路，但把学生模型切换为 `SegFormer-B0`，并保留后续接入 `CLIPSeg` teacher soft mask 蒸馏的能力。

当前目标有两个：

1. 先得到可稳定训练、可直接对比的 `B0-hard` 基线。
2. 在不改变既有数据处理主流程的前提下，做 `B0+D` 蒸馏强化。

当前 11 类为：

`person`, `car`, `building`, `tree`, `animal`, `trash can`, `window`, `door`, `fence`, `pole_light`, `motorcycle`

`computer` 类在 v4 中被明确剔除。

`全集训练.md` 的内容已经并入本文件；后续以本文件作为 `clipseg-v4` 的方案与流程主文档。

## 2. 训练模式

### 2.1 B0-hard

默认模式，只使用 hard label 训练。

特点：

- 不依赖 teacher cache
- 用于建立 SegFormer-B0 11 类基线
- 训练逻辑简单，便于先验证数据和指标是否正常

总损失：

`BCE_WEIGHT * BCE + DICE_WEIGHT * Dice + FOCAL_WEIGHT * Focal`

### 2.2 B0+D

在 hard loss 基础上叠加 CLIPSeg teacher 蒸馏损失。

基本流程：

1. 先训练或拿到一个可用的 `B0-hard` checkpoint
2. 生成训练集对应的 teacher soft cache
3. 从 `B0-hard` 权重继续 finetune
4. 用 cache 命中样本计算 distill loss

蒸馏训练日志会额外输出：

- `Teacher cache hit`
- `Teacher cache missing`
- `hit_rate`

这是判断蒸馏是否真正生效的直接指标。

## 3. 数据与样本组织

### 3.1 输入来源

v4 兼容两种数据格式：

1. 普通 `pred_json`
2. `manifest` 风格记录

默认训练走普通 `pred_json`，每张图按 `prompts` 拆成多个单类样本。

### 3.2 样本过滤规则

- `computer` 直接过滤
- 不在 `CLASSES` 内的 prompt 过滤
- 图片不在 `trainval_list.txt / val_list.txt` 中的过滤
- 空 mask 过滤
- 可选按 `score` 和阈值做过滤

### 3.3 负样本策略

训练集：

- 可保留部分负样本
- 由 `INCLUDE_NEGATIVE_SAMPLES` 和 `NEGATIVE_SAMPLE_RATIO` 控制
- 负样本单独使用 `NEGATIVE_SAMPLE_WEIGHT`

验证集：

- 固定 `VAL_INCLUDE_NEGATIVE_SAMPLES=False`
- 避免负样本把 mIoU 虚高

### 3.4 稀有类过采样

`RARE_OVERSAMPLE` 会复制正样本，重点照顾：

- `animal`
- `trash can`
- `window`
- `door`
- `pole_light`
- `motorcycle`

## 4. 图像预处理与空间对齐

### 4.1 红外预处理

训练和 cache 生成沿用同一套思路：

- 读图后统一转灰度
- 自动判断伪彩 / 黑热
- 低对比度场景做 CLAHE
- 高噪声场景做降噪
- 模糊场景做锐化

### 4.2 统一画布

学生输入、hard label、teacher soft mask 都对齐到同一训练画布：

- 按长边等比例缩放
- 短边补零 pad
- 最终尺寸为 `768 x 768`

这是 v4 的核心约束，用来保证：

- 图像输入
- 二值 mask
- teacher soft prob

三者空间严格一致。

## 5. 损失设计

### 5.1 Hard loss

当前 hard loss 组合为：

```python
hard_loss = 1.0 * BCE_loss + 0.5 * Dice_loss + 0.5 * RareClassFocal_loss
```

### 5.2 RareClassFocalLoss

特点：

- 按类别使用不同 `CLASS_WEIGHTS`
- 使用 `FOCAL_GAMMA`
- `FOCAL_ALPHA` 已真正生效

### 5.3 Distill loss

蒸馏时使用 teacher soft probability 对学生 logit 做监督。

逻辑上等价于：

```python
distill_loss = BCEWithLogits(student_logit, teacher_soft_prob)
```

稀有类可以使用更高蒸馏权重：

```python
if class_name in RARE_CLASSES:
    lambda_cls = DISTILL_LAMBDA_RARE
else:
    lambda_cls = DISTILL_LAMBDA

total_loss = hard_loss + lambda_cls * distill_loss
```

### 5.4 边界弱监督

`build_boundary_valid_mask()` 会在 hard supervision 中忽略边界带，减少伪标签边缘误差对训练的影响。

## 6. 续训与热启动

v4 已修复 `--run_name / --epochs / --resume` 不生效的问题。

checkpoint 选择优先级：

1. `--resume`
2. 当前 `run_name` 下的 `last.pt`
3. 当前 `run_name` 下的 `best_all11.pt`

支持三种常用场景。

### 正常续训

- 不加 `--resume_weights_only`
- 恢复模型、optimizer、scheduler、history、epoch

### 从旧权重启动新实验

- 使用 `--resume xxx.pt --resume_weights_only --reset_history`
- 仅加载模型参数
- 清空历史和 best 指标

### 禁用自动续训

- 使用 `--no_auto_resume`

## 7. Teacher cache 机制

### 7.1 teacher 加载方式

cache 脚本分两步加载 teacher：

1. `from_pretrained(base_model_dir)`
2. `load_state_dict(teacher_checkpoint)`

这样能确保真正用到指定的微调 teacher，而不是只用原始 base model。

### 7.2 cache 命名

统一格式：

`{ann_id_or_image_stem}__{class_name}.npy`

训练脚本读取顺序为：

1. `{ann_id}__{class_name}.npy`
2. `{ann_id}.npy`
3. `{image_stem}__{class_name}.npy`

### 7.3 cache 内容

每个 `.npy` 保存：

- 单类别 soft probability map
- 尺寸 `768 x 768`
- `float16`
- 值域 `[0, 1]`

目录还会输出：

- `cache_summary.json`
- `failed_items.json`

## 8. 输出文件

训练目录：

`test/train_output/<run_name>/`

关键输出：

- `last.pt`
- `best_all11.pt`
- `best_old5.pt`
- `best_rare.pt`
- `results.csv`
- `results.png`

checkpoint 中还额外保存：

- `classes`
- `num_classes`
- `img_size`
- `val_thresholds`
- `model_type`
- `run_name`

这些信息是后续提交推理脚本对齐类别顺序和阈值时的重要元数据。

## 9. 推荐工作流程

1. 先跑 `B0-hard`，验证数据、loss 和指标是否正常。
2. 从 hard 结果里选择合适 checkpoint，作为蒸馏初始化权重。
3. 生成 `CLIPSeg` teacher soft cache，并检查命中情况与失败样本。
4. 启动 `B0+D` 蒸馏，重点观察 `all11 / old5 / rare` 三组指标变化。
5. 根据 `best_all11`、`best_old5`、`best_rare` 三类 best checkpoint 决定后续对比和提交权重。

## 10. 当前版本已解决的关键问题

- `--run_name / --epochs / --resume` 真正生效
- 验证集不再混入负样本
- `Focal alpha` 真正参与计算
- teacher cache 命中率会在蒸馏日志中打印
- cache 脚本不再依赖外部 `clipseg_train_base`
- cache 命名与训练读取规则统一
- teacher soft mask 与学生训练画布对齐
- teacher checkpoint 加载方式已修正

## 11. 后续实验建议

1. 先比较 `B0-hard vs CLIPSeg-11`
2. 再比较 `B0-hard vs B0+D`
3. 分别观察 `all11 / old5 / rare` 三组指标
4. 若稀有类收益不明显，再调 `DISTILL_LAMBDA_RARE`
