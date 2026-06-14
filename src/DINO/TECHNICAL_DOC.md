# DINO 技术文档

## 1. 路线定位

`src/DINO/` 这条线本质上是一个两阶段闭环方案：

1. 训练阶段：
   用伪标签中的实例 mask 和由 mask 提取出的 box，训练一个 `box-prompt EfficientSAM` 学生模型。
2. 推理阶段：
   先用 `GroundingDINO` 产出每类候选框，再把这些框送入训练好的 `EfficientSAM student`，输出 mask，最后编码成提交 JSON。

所以它不是“训练 DINO 模型本体”，而是：

- `GroundingDINO` 负责推理时找框
- `EfficientSAM student` 负责根据框做实例分割

## 2. 目录结构

当前目录主要文件如下：

- [train_DINO.py](/d:/nyz/raytron_project/src/DINO/train_DINO.py)
- [infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py)
- [run_prepare_all.sh](/d:/nyz/raytron_project/src/DINO/run_prepare_all.sh)
- `efficient_sam/`
- `models/`

其中：

- `train_DINO.py`：训练入口，负责数据构建、采样、训练、验证、checkpoint 保存。
- `infer_DINO.py`：推理入口，负责 `GroundingDINO -> EfficientSAM -> JSON`。
- `models/`：放 EfficientSAM 和 GroundingDINO 权重。

## 3. 模型结构

### 3.1 训练时实际训练的模型

训练脚本里真正构建的是 `EfficientSAM`：

```python
from efficient_sam.efficient_sam import build_efficient_sam
```

并通过：

```python
model = build_efficient_sam(...)
```

初始化模型。

训练输入不是点提示，而是 `box prompt`。脚本核心前向是：

```python
forward_efficient_sam(model, images, boxes)
```

即：

- 输入图像
- 输入框
- 输出 mask logits

### 3.2 推理时的整体闭环

推理脚本 [infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py) 的流程是：

1. 对测试图做与训练一致的预处理。
2. 用 `GroundingDINO` 对多类 prompt 产候选框。
3. 对检测框按类阈值过滤，并限制每类最大框数。
4. 把框 batch 送入 `EfficientSAM student`。
5. 对输出 mask 做二值化、最小面积过滤、mask NMS。
6. 把结果编码成 RLE，写出 JSON。

这也是该路线名称里带 `DINO` 的原因：DINO 只出现在推理找框环节。

## 4. 类别定义

这条线当前已经切到 11 类：

```python
CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "computer",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
]
```

训练和推理脚本都使用这一组类别顺序。

## 5. 数据输入

### 5.1 训练数据

[train_DINO.py](/d:/nyz/raytron_project/src/DINO/train_DINO.py) 顶部默认路径：

```python
TRAIN_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON   = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST      = ROOT / "test" / "trainval_list.txt"
VAL_LIST        = ROOT / "test" / "val_list.txt"
IMAGE_ROOT      = ROOT
```

含义：

- `trainval_label.json` 是 11 类全集训练伪标签来源
- `val_label.json` 继续作为伪标签验证来源
- `trainval_list.txt` / `val_list.txt` 决定训练/验证图片集合
- `IMAGE_ROOT` 是图像根目录

这里的改动点是：

- 原先 6 类旧伪标签训练，已经切换成 11 类新标签输入
- 原先 train split 训练，已经切换成 `trainval` 全集训练
- 抽样策略、损失、优化器和训练主流程保持不变

### 5.2 推理数据

[infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py) 默认读取：

```python
TEST_TASK_JSON = ROOT / "test" / "test_tasks.json"
TEST_LIST      = ROOT / "test" / "test_list.txt"
```

优先级是：

1. `test_tasks.json`
2. `test_list.txt`

## 6. 图像预处理

训练和推理都使用同一套“teacher 对齐”的灰度预处理逻辑，核心函数是：

- `load_teacher_aligned_gray`
- `letterbox_image`
- `letterbox_mask`

主要步骤包括：

1. 读图并转灰度
2. 判断伪彩图并转灰度
3. 判断 `blackHot` 极性并可能反色
4. 基于统计量做 CLAHE 对比度增强
5. 基于噪声估计做双边滤波或中值滤波
6. 基于模糊程度做锐化
7. letterbox 到固定尺寸

这保证了训练和推理的输入分布尽量一致。

## 7. 训练样本构造

训练用的数据集类是：

```python
class EfficientSAMBoxPromptDataset(Dataset)
```

核心思路：

1. 从伪标签 JSON 里读取每张图每个 prompt 的 mask。
2. 对 `hit=true` 的条目：
   - 读取 RLE
   - 解码 mask
   - 提取实例外接框
   - 生成 box prompt 训练样本
3. 对 `hit=false` 的条目：
   - 按 `NEGATIVE_SAMPLE_RATIO` 概率记录为负样本统计参考

注意一点：

脚本里明确打印：

```text
hit=false negative prompts are counted but not used for EfficientSAM box-prompt training.
```

也就是说这条线虽然读取了负样本配置，但它本质上仍然是以正样本 box-prompt 分割训练为主，不像一些 CLIPSeg 线那样直接把负样本当成常规二值分割样本参与训练。

## 8. 采样与重平衡

训练脚本提供几种重平衡机制：

### 8.1 置信度过滤

```python
APPLY_SCORE_FILTER = False
PROMPT_THRESHOLDS = {...}
```

如果打开，会按类别阈值过滤低分伪标签。

### 8.2 稀有类过采样

```python
RARE_OVERSAMPLE = {
    "animal": 3,
    "computer": 3,
    "trash can": 3,
    "window": 3,
    "door": 3,
    "fence": 3,
    "pole_light": 3,
}
```

在数据集构建时重复 rare 类正样本。

### 8.3 类别均衡采样器

```python
BALANCED_SAMPLER = True
```

训练时可通过：

```python
build_balanced_sampler(dataset, num_samples)
```

对 `dataset.samples` 按类别反频率做 `WeightedRandomSampler`。

### 8.4 验证集限样

```python
MAX_VAL_SAMPLES = 5000
```

并通过：

```python
limit_dataset_stratified(dataset, max_samples, seed=42, name="dataset")
```

对验证集做按类分层抽样，避免验证过慢。

## 9. 提示框处理

因为这条线训练的是 `box prompt -> mask`，所以 box 质量很重要。

脚本提供了几项 box 级增强：

- `BOX_EXPAND_RATIO`
- `BOX_EXPAND_RATIO_MIN`
- `BOX_EXPAND_RATIO_MAX`
- `BOX_JITTER_PROB`
- `BOX_JITTER_CENTER`
- `BOX_JITTER_SCALE`

对应函数有：

- `expand_box_xyxy`
- `jitter_box_xyxy`
- `letterbox_box_xyxy`
- `hflip_box_xyxy`

训练时会对 box 做一定随机扩张和抖动，让 student 对检测框误差更鲁棒。

## 10. 训练配置

[train_DINO.py](/d:/nyz/raytron_project/src/DINO/train_DINO.py) 顶部默认配置大致如下：

### 10.1 训练基础参数

```python
IMG_SIZE = 768
BATCH_SIZE = 4
EPOCHS = 5
NUM_WORKERS = 0
DEVICE = "cuda"
SEED = 42
BALANCED_SAMPLER = True
STEPS_PER_EPOCH = 2500
MAX_VAL_SAMPLES = 5000
FREEZE_IMAGE_ENCODER = True
```

注意：

- `NUM_WORKERS = 0` 是专门为 Windows 避免 DataLoader 卡死设的
- `FREEZE_IMAGE_ENCODER = True` 说明默认偏向只微调 decoder 相关部分

### 10.2 优化器

```python
LR = 1e-4
DECODER_LR = None
IMAGE_LR = None
WEIGHT_DECAY = 0.03
MODEL_TYPE = "vitt"
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.01
GRAD_CLIP = 1.0
```

### 10.3 Loss

```python
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
FOCAL_AUX_WEIGHT = 0.05
FOCAL_GAMMA = 1.5
FOCAL_POS_FACTOR = 1.0
FOCAL_NEG_FACTOR = 0.25
```

训练主损失由：

- BCE
- Dice
- 类别 focal auxiliary loss

组成。

对应函数是：

- `bce_dice_loss`
- `class_focal_aux_loss`

## 11. 验证指标

验证阶段核心函数：

```python
validate(model, loader, device, epoch, epochs)
```

计算内容包括：

- `iou/overall`
- `dice/overall`
- `iou/pos_only`
- `dice/pos_only`
- `iou/<class_name>`

训练日志会输出：

- `TrainLoss`
- `Main`
- `BCE`
- `Dice`
- `Focal`
- `ValLoss`
- `mIoU`
- `pos_mIoU`
- `Dice`
- `LR`
- `Per-class IoU`

## 12. Checkpoint 与输出

默认输出目录：

```python
OUTPUT_DIR = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1"
```

训练过程中会保存：

- `last.pt`
- `last_model_only.pt`
- `best.pt`
- `best_model_only.pt`
- `best_pos.pt`
- `best_pos_model_only.pt`
- `results.csv`
- `train.log`

其中：

- `best.pt` 按 overall mIoU 选
- `best_pos.pt` 按 pos-only mIoU 选
- `*_model_only.pt` 只保留模型权重，更适合推理部署

## 13. 推理流程

[infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py) 的推理链路是：

1. 读取测试图路径
2. 调 `GroundingDINO` 多类 prompt 检测
3. 按类阈值过滤检测框
4. 按类限制框数量
5. 调 `EfficientSAM student` 对每个框出 mask
6. 对 mask 做二值化
7. 做最小面积过滤
8. 做同类 mask NMS
9. 写出 JSON

### 13.1 DINO 阈值

```python
DINO_THRESHOLDS = {
    "person": 0.25,
    "car": 0.25,
    "building": 0.20,
    "tree": 0.20,
    "animal": 0.18,
    "computer": 0.18,
}
```

### 13.2 每类最多框数

```python
MAX_BOXES_PER_CLASS = {
    "person": 80,
    "car": 120,
    "building": 80,
    "tree": 100,
    "animal": 80,
    "computer": 40,
}
```

### 13.3 Mask 阈值与后处理

```python
MASK_THRESHOLDS = {
    "person": 0.50,
    "car": 0.50,
    "building": 0.50,
    "tree": 0.50,
    "animal": 0.45,
    "computer": 0.45,
}
```

```python
MIN_AREA = {
    "person": 8,
    "car": 8,
    "building": 16,
    "tree": 8,
    "animal": 4,
    "computer": 4,
}
```

```python
MASK_NMS_IOU = 0.65
```

## 14. 推理模型加载方式

推理用的是两步加载：

1. 先加载基础 `EfficientSAM` 权重
2. 再加载训练好的 `student checkpoint`

核心函数：

```python
load_student_model(student_ckpt_path, efficient_sam_ckpt, device)
```

默认 student 权重路径：

```python
STUDENT_CKPT = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1" / "best_model_only.pt"
```

## 15. 输出 JSON 格式

推理输出默认写到：

```python
OUTPUT_JSON = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1" / "pred_test_dino_efficientsam.json"
```

文档字符串里已经说明：

```text
输出 JSON 格式对齐项目现有 clipseg/Hanxue 推理格式。
```

也就是说这条线的结果可以和项目里其他推理结果保持统一消费方式。

## 16. 这条线的优点与限制

### 16.1 优点

- 把检测和分割拆开，推理链路清晰
- 训练阶段只需要 box-prompt 分割，目标明确
- 可复用 GroundingDINO 的开放词汇找框能力
- 验证与训练日志比较完整
- 训练和推理预处理保持一致

### 16.2 限制

- 训练本体并不是 DINO 联合训练，而是只训练 EfficientSAM student
- 推理质量强依赖 DINO 的找框质量
- 现在已经切到 11 类，但推理质量仍然强依赖 DINO 的找框质量
- 默认 `NUM_WORKERS = 0`，Windows 下速度会受限
- 负样本更多是统计和参考用途，不是完整负样本分割训练范式

## 17. 后续维护建议

如果后面继续调这条线，优先建议从这些参数下手：

- `DINO_THRESHOLDS`
- `MAX_BOXES_PER_CLASS`
- `MASK_THRESHOLDS`
- `MIN_AREA`
- `RARE_OVERSAMPLE`
- `NEGATIVE_SAMPLE_RATIO`
- `BOX_EXPAND_RATIO_*`
- `BOX_JITTER_*`

不建议一开始就改：

- 模型结构
- EfficientSAM backbone
- GroundingDINO 主体
- 输出 JSON 协议

因为这条线当前最大的收益点更可能来自：

- 框质量
- 框后过滤
- mask 二值化阈值
- rare 类采样与过采样策略
