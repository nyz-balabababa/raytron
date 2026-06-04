# YOLO-World + FastSAM v3 技术文档总结

> 对应代码: `src/train/world/world_train_v3.py` + `src/train/world/config_world_v3.py`

## 一、版本定位

`v3` 不是在 `v2` 基础上继续叠加技巧，而是一次**基于实验结论的回退与收敛**。

核心判断是：

1. `v2` 里并不是所有改动都有效
2. 检测指标提升不等于开放词汇泛化更好
3. `computer` 和 `tree` 的问题主要不在“再加力度”，而在伪标签质量和错误增强方向

因此 `v3` 的目标很明确：

- 保留 `v2` 里被证明有效的部分
- 回退会破坏同义词泛化或引入负收益的部分
- 把两阶段路线收敛到一版更稳、更可解释的 6 类基线

类别仍为：

- `person`
- `car`
- `building`
- `tree`
- `animal`
- `computer`

---

## 二、v2 → v3 关键改动

### 2.1 回退项

| 项目 | v2 | v3 | 原因 |
|---|---|---|---|
| `cls / box / label_smoothing` | 手动增大 | 回退 Ultralytics 默认 | `v2` 证明会破坏 CLIP 同义词泛化 |
| `computer` 阈值 | `0.40` | `0.50` | `0.40 + 5x` 没有稳定收益 |
| `computer` 过采样 | `5x` | `2x` | 高倍率复制收益不明显，且更容易放大噪声 |
| `tree` 多尺度 | 开启 | 移除 | `v2` 里 `tree` 多尺度反而掉点 |

### 2.2 保留项

| 项目 | 保留原因 |
|---|---|
| `person / animal` 多尺度推理 | `v2` 已验证有效 |
| `animal` 独立过采样 | 稀有类确实受益 |
| FastSAM 动态 `imgsz` | 小 crop 分割更稳 |
| CLIP 离线缓存 | 保证完全离线可运行 |
| GPU 日志 / 自定义训练曲线图 | 便于复盘和迁移 |
| 跨尺度 NMS `IoU=0.35`、`MS_CONF=0.2` | 与 `v2` 配套、已验证可用 |

一句话概括：  
`v3` 是“删掉无效技巧，只保留被验证过的有效部分”的稳定版。

---

## 三、整体流程

`world_train_v3.py` 仍然是两阶段管线：

1. `SAM3` 伪标签 `RLE` 转 YOLO 检测框
2. 微调 `YOLO-World`
3. 用 `YOLO-World` 出框
4. 用 `FastSAM` 对每个框做零样本分割
5. 与伪标签做 `mask mIoU` 对比

流程图：

```text
pred_train_tasks.json / pred_val_tasks1.json
        ↓
RLE -> mask -> contours -> bbox
        ↓
YOLO bbox 数据集
        ↓
YOLO-World v2 检测训练
        ↓
验证时按 prompt 推理出 bbox
        ↓
FastSAM 对每个 bbox crop 分割
        ↓
并集 mask
        ↓
与 SAM3 伪标签比较 mIoU
```

---

## 四、配置摘要

`config_world_v3.py` 的核心配置如下。

### 4.1 数据与类别

| 项目 | 配置 |
|---|---|
| 训练伪标签 | `test/prompt_test_output/train_tasks/pred_train_tasks.json` |
| 验证伪标签 | `test/prompt_test_output/val_tasks1/pred_val_tasks1.json` |
| 训练列表 | `test/train_list.txt` |
| 验证列表 | `test/val_list.txt` |
| 类别 | `["person", "car", "building", "tree", "animal", "computer"]` |

### 4.2 伪标签过滤阈值

| 类别 | 阈值 |
|---|---:|
| `person` | `0.70` |
| `car` | `0.70` |
| `building` | `0.70` |
| `tree` | `0.60` |
| `animal` | `0.60` |
| `computer` | `0.50` |

### 4.3 稀有类补偿

| 类别 | 方式 |
|---|---|
| `animal` | `3x` 过采样 |
| `computer` | `2x` 总倍数，靠 `COMPUTER_EXTRA = 1` 实现 |

### 4.4 训练超参

| 参数 | 值 |
|---|---:|
| `IMG_SIZE` | `800` |
| `BATCH` | `16` |
| `EPOCHS` | `100` |
| `OPTIMIZER` | `AdamW` |
| `LR0` | `2e-4` |
| `PATIENCE` | `10` |
| `WORKERS` | `8` |

### 4.5 数据增强

`v3` 继续采用偏保守增强：

- `mosaic = 0.0`
- `mixup = 0.0`
- `copy_paste = 0.0`
- `scale = 0.3`
- `degrees = 0.0`
- `flipud = 0.0`
- `fliplr = 0.5`
- `hsv_v = 0.15`

这套配置的思路是：  
对红外小目标和伪标签监督来说，先避免重增强破坏几何一致性。

---

## 五、代码模块说明

## 5.1 离线 CLIP 缓存

`_ensure_local_clip()` 会确保 `ViT-B-32.pt` 已经在本地缓存中。

作用：

- 避免 `YOLO-World set_classes()` 首次运行时在线下载
- 保证脚本在离线环境中可运行
- 将权重统一放到 `~/.cache/clip/` 和项目内 `model/clip/`

这是工程稳定性改动，不影响指标，但非常重要。

## 5.2 反色统一

`unify_polarity()` 的逻辑与项目其他训练脚本一致，主要处理红外图的黑热/白热不统一问题。

判定依据：

- 文件名包含 `blackHot`
- 灰度偏度 `skew`
- 图像均值兜底

这一步是整个项目数据预处理的一致基础。

## 5.3 RLE → bbox

`convert_split()` 会把伪标签里的 `RLE` 掩码转换为 YOLO 检测格式。

关键步骤：

1. 读取 `pred_*.json`
2. 按 `image_path` 聚合 prompt
3. 应用 `CONF_FILTER`
4. `RLE -> mask`
5. `mask -> contours`
6. `contours -> boundingRect`
7. 写成 `class_id cx cy w h`

这里的核心取舍是：

- `YOLO-World` 只训练检测，不训练掩码
- 伪标签掩码的信息在训练阶段被压缩成框
- 真正的分割质量留给 `FastSAM` 在推理阶段补回来

## 5.4 解耦过采样

`convert_split()` 里把 `animal` 和 `computer` 的过采样拆成两个阶段，避免 `v2` 中“一个类的倍率误伤另一个类”。

阶段 A：

- 对 `animal` 图做 `RARE_OVERSAMPLE["animal"] - 1` 次复制

阶段 B：

- 对 `computer` 图再额外复制 `COMPUTER_EXTRA` 次

这样能保证：

- `animal` 固定是 `3x`
- `computer` 固定是 `2x`
- 二者互不牵连

## 5.5 多尺度检测

`_multi_scale_detect()` 是 `v3` 里最关键的推理增强模块之一。

只对这两类启用：

- `person`
- `animal`

尺度：

- `[0.8, 1.0, 1.25, 1.5]`

策略：

1. 同一个 prompt 先 `set_classes([prompt])`
2. 同一张图跑多个 `imgsz`
3. 收集所有尺度的框
4. 用跨尺度 NMS 合并

目的：

- 提升小目标召回
- 不改训练，只改验证/推理

而 `tree` 被从多尺度里移除，正是 `v3` 相对 `v2` 最重要的收缩之一。

## 5.6 FastSAM 动态分割

`_fastsam_segment_boxes()` 不再固定所有 crop 都用同一个 `imgsz`，而是按 crop 实际大小动态选择：

```text
imgsz = max(256, min(1024, max(crop_h, crop_w)))
再向上对齐到 32 的倍数
```

这样做的原因很直接：

- 小框目标如果强行放到 `1024`，会被过度拉伸
- 大框目标如果压太小，细节又会丢

这一步本质是在减少两阶段解耦带来的裁剪副作用。

## 5.7 两阶段评估

`two_stage_eval()` 是 `v3` 的核心评估函数。

它做三件事：

1. 按验证伪标签逐图、逐 prompt 推理
2. 比较 `pred_mask` 和伪标签 `gt_mask` 的 `IoU`
3. 额外做同义词泛化测试

输出包括：

- per-class `mask mIoU`
- overall `mIoU`
- 多尺度统计
- 同义词映射结果

这比只看 `YOLO-World` 的 `mAP50` 更接近你项目真正关心的目标。

## 5.8 训练过程可视化

`plot_training_curves()` 会从 `results.csv` 生成 `training_curves_v3.png`。

图里包含 6 个面板：

1. train loss
2. val loss
3. `mAP50 / mAP50-95`
4. `Precision / Recall`
5. 学习率
6. 文字摘要

这部分是工程监控增强，不改变训练行为，但很适合做版本对比复盘。

---

## 六、v3 的技术取舍

`v3` 有几个非常明确的设计取舍。

### 6.1 不追求检测指标最大化

`v3` 的判断标准不是“把 `mAP50` 尽量做高”，而是：

- 两阶段最终 `mask mIoU` 是否更稳
- 开放词汇同义词泛化是否保住

这也是为什么 `cls / box / label_smoothing` 会被回退。

### 6.2 优先保护语义泛化

因为 `YOLO-World` 的文本能力来自 CLIP 语义空间，所以一旦训练策略把语义空间拉坏：

- 检测分数可能看起来还行
- 但开放词汇能力会掉

`v3` 明确把“保护 CLIP 语义能力”放在首位。

### 6.3 只在被证明有效的类上用多尺度

`v3` 没有把多尺度做成全类别默认，而是只留给：

- `person`
- `animal`

这说明你们在这一版里已经开始按类做策略，而不是统一一把梭。

### 6.4 保守处理 `computer`

`computer` 在 `v2` 中被重点加强，但收益不稳定，所以 `v3` 做了回退：

- 阈值从 `0.40` 回到 `0.50`
- 过采样从 `5x` 回到 `2x`

这实际是在承认：  
`computer` 当前瓶颈不只是样本少，更可能是伪标签本身不够稳。

---

## 七、输出与产物

`v3` 会生成以下内容：

| 路径 | 内容 |
|---|---|
| `test/yolo_world_dataset_v3/` | 转换后的 YOLO 检测数据集 |
| `test/train_output/yoloworld_fastsam_v3/weights/` | `best.pt` / `last.pt` |
| `test/train_output/yoloworld_fastsam_v3/results.csv` | 逐 epoch 检测指标 |
| `test/train_output/yoloworld_fastsam_v3/training_curves_v3.png` | 自定义训练曲线图 |
| `test/train_output/yoloworld_fastsam_v3.log` | 全流程日志，含两阶段 `mIoU` |

---

## 八、适用结论

从代码设计层面看，`v3` 的价值主要有三点：

1. 它把 `YOLO-World + FastSAM` 路线从“不断叠技巧”收敛成了“只保留有效项”的稳定版
2. 它明确区分了“检测训练指标”和“最终 mask 质量”这两个目标
3. 它为后续 `v4` 的尾类专项实验提供了一个更干净的 6 类基线

换句话说：

- `v1` 是路线验证版
- `v2` 是技巧扩展版
- `v3` 是实验回收后的稳定基线版

---

## 九、一句话总结

`world_train_v3.py` 的核心意义，不是新增了更多技巧，而是把 `YOLO-World + FastSAM` 两阶段路线收敛成一版更稳、更离线可复现、且更重视开放词汇泛化能力的 6 类基线实现。
