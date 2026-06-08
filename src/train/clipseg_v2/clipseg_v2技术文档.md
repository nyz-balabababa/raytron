# CLIPSeg V2 技术文档

本文档对应当前目录下的：

- `config_clipseg.py`
- `clipseg_train.py`

目标是说明当前 `clipseg_v2` 这一版训练线的输入、配置、训练逻辑和实验定位。

---

## 1. 当前定位

`clipseg_v2` 不是最初那版直接吃旧 `pred_train_tasks.json` 的训练脚本，而是基于：

- `A/B/C` 质量分档
- 新旧伪标签融合
- full/light manifest

做的五类 `CLIPSeg` 短程微调版本。

当前主线类别是：

- `person`
- `car`
- `building`
- `tree`
- `animal`

当前不包含：

- `computer`
- `trash can`

这些尾类如果后续要做，应另开专项实验线。

---

## 2. 当前训练目标

当前 `clipseg_v2` 的目标不是从头训练，而是：

1. 继承旧 `clipseg_v1` 的已有分割能力
2. 切换到 manifest 驱动的数据入口
3. 使用新的正负样本定义和 sample weight
4. 做一轮低学习率、短周期微调

对应策略是：

- `INIT_CHECKPOINT` 指向旧 `clipseg_v1/best.pt`
- 首次跑新实验时，从旧 best 初始化参数
- 后续同一 `RUN_NAME` 下若存在 `last.pt/best.pt`，则直接断点续训

---

## 3. 当前配置概览

当前关键配置来自 [config_clipseg.py](/abs/path/d:/nyz/raytron_project/src/train/clipseg_v2/config_clipseg.py)。

### 3.1 数据入口

- `USE_MANIFEST = True`
- `MANIFEST_JSON = ROOT / "test" / "label_analysis" / "train_manifest_light.json"`
- `PRED_JSON = ROOT / "test" / "sam3_label_old" / "train_tasks" / "pred_train_tasks.json"`
- `VAL_PRED_JSON = ROOT / "test" / "sam3_label_old" / "val_tasks1" / "pred_val_tasks1.json"`
- `TRAIN_LIST = ROOT / "test" / "train_list.txt"`
- `VAL_LIST = ROOT / "test" / "val_list.txt"`

解释：

- 训练集当前实际读取 `MANIFEST_JSON`
- `PRED_JSON` 只作为旧模式备用入口
- 验证集仍固定读取 `VAL_PRED_JSON`

### 3.2 模型初始化

- `MODEL_NAME = "CIDAS/clipseg-rd64-refined"`
- `MODEL_DIR = ROOT / "model" / "clipseg-rd64-refined"`
- `INIT_CHECKPOINT = ROOT / "test" / "train_output" / "clipseg_v1" / "best.pt"`

### 3.3 训练超参数

- `IMG_SIZE = 768`
- `BATCH = 6`
- `EPOCHS = 5`
- `DECODER_LR = 5e-5`
- `BACKBONE_LR = 5e-6`
- `WEIGHT_DECAY = 0.03`
- `WARMUP_EPOCHS = 2`
- `LRF = 0.01`

### 3.4 Manifest 相关配置

- `MANIFEST_LOSS_WEIGHT_FLOOR = 0.1`
- `MANIFEST_NEGATIVE_RATIO = 0.25`
- `MANIFEST_CACHE_TAG = "train_manifest_v1"`
- `FORCE_REBUILD_MASK_CACHE = False`

### 3.5 验证口径

- `VAL_APPLY_SCORE_FILTER = True`
- `VAL_THRESHOLDS = PROMPT_THRESHOLDS`

含义是：

- 训练集不再按旧 teacher score 阈值二次过滤
- 验证集继续沿用旧伪标签阈值口径，方便和历史实验对比

---

## 4. Manifest 数据流

当前训练集输入不是直接来自旧 `pred_train_tasks.json`，而是来自 `train_manifest_light.json`。

每条 record 当前至少保留：

- `image_path`
- `prompt`
- `include_in_train`
- `selected_hit`
- `sample_weight`
- `rle`
- `grade`
- `mask_source`
- `is_tiny`

训练脚本的处理方式如下。

### 4.1 正样本

条件：

- `include_in_train = true`
- `selected_hit = true`
- `rle 非空`

处理：

- 解码 `rle`
- 写入 mask cache
- `is_positive = True`

### 4.2 负样本

条件：

- `include_in_train = true`
- 且 `selected_hit = false`
  或 `rle is None`

处理：

- 不解码 `rle`
- 不写 cache
- 训练时使用全 0 mask
- `is_positive = False`

### 4.3 excluded 样本

条件：

- `include_in_train = false`

处理：

- 直接跳过，不进入训练样本池

---

## 5. Manifest 负样本采样

当前 manifest 里的负样本不会全量使用。

训练脚本会先拆成：

- `positives`
- `negatives`

然后按：

- `MANIFEST_NEGATIVE_RATIO = 0.25`

做固定随机种子采样。

即：

- 最多保留为正样本数量 `25%` 的负样本

当前目的：

- 保留拒识能力
- 避免负样本占比过大导致训练时间膨胀
- 避免模型变得过保守

---

## 6. Mask Cache 设计

当前脚本会把 manifest 正样本 mask 提前缓存成 PNG。

cache 根目录：

- `PROJECT / ".mask_cache" / MANIFEST_CACHE_TAG`

当前为：

- `test/train_output/.mask_cache/train_manifest_v1`

这样做的目的：

- 避免每个 epoch 重复做 RLE 解码
- 让 manifest 模式与旧 `pred_json` cache 彻底隔离

### 当前 cache hash 组成

manifest 正样本缓存键至少包含：

- `image_path`
- `prompt`
- `grade`
- `mask_source`
- `selected_hit`
- `rle.size`
- `rle.counts` 的 hash

因此即使 `image_path/prompt` 不变，只要 RLE 内容变了，也不会误用旧缓存。

---

## 7. 初始化与断点续训

当前训练逻辑分两段。

### 7.1 第一次运行某个新实验

如果当前 `RUN_NAME` 目录下不存在：

- `last.pt`
- `best.pt`

则：

- 从 `INIT_CHECKPOINT` 加载模型参数
- 只初始化模型
- 不恢复 optimizer
- 不恢复 scheduler
- 不恢复 history

### 7.2 已有当前实验 checkpoint

如果当前 `RUN_NAME` 目录下已存在：

- `last.pt`
  或
- `best.pt`

则：

- 优先 resume 当前实验
- 恢复：
  - `model`
  - `optimizer`
  - `scheduler`
  - `history`

因此现在不会在已有 `resume_ckpt` 时，先白加载一遍旧 `clipseg_v1/best.pt`。

---

## 8. 设备与运行时行为

训练脚本内部会把配置里的 `DEVICE` 统一解析为真正的 `torch.device`。

当前默认配置：

- `DEVICE = 0`

会被解析成：

- `cuda:0`

内部使用的是：

- `RUNTIME_DEVICE`

这样可以避免：

- `torch.load(..., map_location=DEVICE)` 直接拿 `int`
- 造成 `TypeError: 'int' object is not callable`

---

## 9. Loss 与 boundary weak supervision

当前 loss 主体仍保持原来的：

- `BCE`
- `Dice`

区别在于 manifest 模式使用：

- `record["sample_weight"]`

作为逐样本 loss 权重。

### 当前 floor 逻辑

- manifest 模式：
  - `MANIFEST_LOSS_WEIGHT_FLOOR = 0.1`
- 旧 `pred_json` 模式：
  - `LOSS_WEIGHT_FLOOR = 0.7`

这样做是为了：

- 避免 `C` 档权重被高 floor 强行抬高

### boundary weak supervision

当前仍保留原弱监督边界逻辑：

- `ENABLE_BOUNDARY_WEAK_SUPERVISION = True`
- `BOUNDARY_IGNORE_WIDTH = 1`
- `BOUNDARY_IGNORE_MIN_AREA = 64`

并带有当前 tiny 保护逻辑：

- `person/car/animal` 的小目标更少被 ignore
- `tree` 中性处理
- `building` 不靠 ignore band 保护 tiny

---

## 10. 验证流程

当前每个 epoch 结束后都会执行一次验证：

1. 先跑完整个 `train_epoch`
2. 然后立刻跑 `validate`
3. 记录：
   - `val_miou`
   - `val_dice`
   - per-class IoU
4. 更新 `last.pt`
5. 如果更优，更新 `best.pt`

### 当前验证集入口

- `VAL_PRED_JSON`
- `VAL_LIST`
- `use_manifest = False`
- `apply_score_filter = VAL_APPLY_SCORE_FILTER`

因此验证仍然是旧伪标签口径，不吃 manifest。

---

## 11. 终端输出与进度条

当前训练脚本有：

- `Train epoch` 进度条
- `Val` 进度条

并已经做过缩短处理：

- 固定 `ncols`
- `leave=False`

目的是减少终端刷屏。

如果后续仍觉得验证条太长，可以进一步改成：

- 只保留训练进度条
- 验证阶段只输出最终汇总日志

---

## 12. 当前实验产物

当前实验输出目录：

- `PROJECT / RUN_NAME`

即：

- `test/train_output/clipseg_manifest_ft_v1`

其中主要包括：

- `best.pt`
- `last.pt`
- `results.csv`
- `results.png`
- `clipseg_manifest_ft_v1.log`

---

## 13. 当前适用范围与下一步

当前 `clipseg_v2` 适合：

- 五类主线：
  - `person`
  - `car`
  - `building`
  - `tree`
  - `animal`
- 基于 manifest 的伪标签蒸馏微调

当前不适合直接拿来做：

- `computer`
- `trash can`

这类尾类专项。

如果后续要做尾类长尾补强，更合理的做法是：

1. 以当前 `clipseg_v2/best.pt` 为初始化
2. 单独开新实验目录
3. 改类别集合和数据入口
4. 不覆盖当前五类主线结果
