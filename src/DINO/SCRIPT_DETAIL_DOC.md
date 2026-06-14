# DINO 脚本级实现说明

## 1. 文档目的

这份文档不是路线概览，而是“按脚本拆开”的实现说明，目标是方便代码审阅。

重点回答这些问题：

- 每个脚本负责什么
- 输入是什么
- 输出是什么
- 内部主流程怎么走
- 关键函数分别干什么
- 脚本之间怎么衔接
- 哪些地方容易误解

当前主要覆盖：

- [train_DINO.py](/d:/nyz/raytron_project/src/DINO/train_DINO.py)
- [infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py)
- [run_prepare_all.sh](/d:/nyz/raytron_project/src/DINO/run_prepare_all.sh)

同时会补充：

- `efficient_sam/`
- `models/`

在整个工程中的角色。

## 2. 总体关系

这条线不是单脚本独立工作，而是三段式：

1. `run_prepare_all.sh`
   负责把原始图像和 SAM3 mask 处理成训练可用的伪标签/标注文件。
2. `train_DINO.py`
   负责训练 `EfficientSAM student`。
3. `infer_DINO.py`
   负责推理时先用 `GroundingDINO` 找框，再让训练好的 `EfficientSAM student` 产 mask。

逻辑链路是：

```text
原始图像 + SAM3 mask
    -> run_prepare_all.sh 产训练标注
    -> train_DINO.py 训练 box-prompt EfficientSAM
    -> infer_DINO.py 用 DINO 出框 + EfficientSAM 出 mask
```

## 3. train_DINO.py

### 3.1 脚本职责

[train_DINO.py](/d:/nyz/raytron_project/src/DINO/train_DINO.py) 是这条线的训练主脚本。

虽然名字叫 `train_DINO.py`，但它真正训练的不是 GroundingDINO，而是：

- `EfficientSAM` student
- 输入提示是 box prompt
- box 来自伪标签 mask 提取出的外接框

GroundingDINO 不参与这个脚本里的训练。

### 3.2 顶部配置做了什么

脚本最前面直接定义了项目根目录和路径：

```python
ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
```

然后把当前目录加进 `sys.path`，保证本目录模块可导入：

```python
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
```

接着定义：

- 训练伪标签 JSON
- 验证伪标签 JSON
- 训练/验证图片列表
- 图像根目录
- EfficientSAM 权重路径
- 训练输出目录和各种 checkpoint 文件名

这意味着它是一个“单文件可运行”的训练入口，不依赖单独 config 文件。

### 3.3 这个脚本的输入

训练输入主要有两类：

1. 图像
2. 伪标签 JSON

默认路径：

```python
TRAIN_PRED_JSON = ROOT / "test" / "clean_rare" / "label" / "trainval_label.json"
VAL_PRED_JSON   = ROOT / "test" / "clean_rare" / "label" / "val_label.json"
TRAIN_LIST      = ROOT / "test" / "trainval_list.txt"
VAL_LIST        = ROOT / "test" / "val_list.txt"
IMAGE_ROOT      = ROOT
```

其中：

- JSON 里每张图按 prompt 存伪标签
- 训练已经切到 `11 类 + trainval 全集`
- 验证仍然读取 `val_label.json + val_list.txt`
- `trainval_list.txt / val_list.txt` 决定哪些图真的参与训练和验证

### 3.4 训练对象是什么

脚本里用：

```python
from efficient_sam.efficient_sam import build_efficient_sam
```

构造模型。

关键函数：

```python
def build_model(model_type: str, checkpoint_path: Path)
```

实现分成三步：

1. 根据 `model_type` 选择 `vitt` 或 `vits` 的编码器规模
2. 用 `build_efficient_sam(...)` 搭建网络结构
3. 从 `efficient_sam_vitt.pt` 加载基础权重

也就是说，训练模型一开始是预训练 EfficientSAM，再继续 fine-tune。

### 3.5 数据集类做了什么

训练数据集类是：

```python
class EfficientSAMBoxPromptDataset(Dataset)
```

它的职责是把“按类 prompt 存储的 mask 伪标签”变成“box prompt 分割训练样本”。

核心工作是：

1. 读取伪标签 JSON
2. 遍历每张图每个 prompt
3. 对 `hit=true` 的 mask：
   - 做阈值过滤
   - 解码 RLE
   - 去空 mask
   - 计算外接框
   - 缓存 mask
   - 构造成训练样本
4. 对 rare 类做过采样
5. 训练时做 box 扩张、box 抖动和水平翻转

这个类里最关键的方法是：

- `_build_samples`
- `__getitem__`

#### `_build_samples` 做什么

`_build_samples` 的工作是“从 JSON 生成样本表”。

每条正样本最终至少包含：

- `image_path`
- `mask_path`
- `bbox`
- `prompt`
- `sample_weight`
- `instance_area`

其中 `sample_weight` 不是固定 1，而是：

```python
sample_weight = max(score, self.loss_weight_floor)
```

这意味着 teacher score 会参与监督强度，但又不会低于下限。

#### `__getitem__` 做什么

`__getitem__` 的职责是把单条样本变成模型输入。

它会：

1. 读取原图
2. 读取对应实例 mask
3. 做灰度预处理
4. 对图像和 mask 统一 letterbox 到固定尺寸
5. 对 box 也做相同坐标变换
6. 训练时随机 hflip
7. 输出：
   - `image`
   - `mask`
   - `box`
   - `sample_weight`
   - `prompt`

### 3.6 负样本在这条线里是怎么处理的

虽然脚本定义了：

```python
NEGATIVE_SAMPLE_RATIO = 0.15
NEGATIVE_SAMPLE_WEIGHT = 0.30
```

而且数据集构建里也会统计 `hit=false` prompt，但脚本本身明确说明：

```text
hit=false negative prompts are counted but not used for EfficientSAM box-prompt training.
```

也就是说：

- 负样本配置存在
- 负样本会被统计
- 但训练主流程本质仍是正样本 box-prompt 分割

这点在审代码时很容易误判成“完整负样本训练”，实际上不是。

### 3.7 图像预处理如何实现

训练脚本里有一套比较完整的红外图像预处理链：

- `compute_skewness`
- `is_pseudo_color`
- `estimate_noise_sigma`
- `load_teacher_aligned_gray`

真正的入口是：

```python
load_teacher_aligned_gray(abs_path: Path)
```

它做的事情包括：

1. 读图
2. 伪彩图转灰度
3. 判断 blackHot 并可能反色
4. 根据对比度做 CLAHE
5. 根据噪声做双边滤波或中值滤波
6. 根据模糊程度做锐化

这是训练和推理对齐的关键。

### 3.8 box 如何增强

脚本对 box 做了额外鲁棒性增强。

相关函数：

- `clip_box_xyxy`
- `expand_box_xyxy`
- `jitter_box_xyxy`
- `letterbox_box_xyxy`
- `hflip_box_xyxy`

它们的作用：

- `clip_box_xyxy`
  把框裁回图像边界内
- `expand_box_xyxy`
  把框按比例扩张
- `jitter_box_xyxy`
  给框中心和尺度加随机扰动
- `letterbox_box_xyxy`
  把原图坐标框映射到 768 输入坐标系
- `hflip_box_xyxy`
  训练时水平翻转框

所以这条线不是拿 teacher 框原样训练，而是做了适度检测误差模拟。

### 3.9 loss 怎么组成

训练主损失由三部分组成：

1. BCE
2. Dice
3. focal auxiliary loss

对应实现：

- `dice_loss_with_logits`
- `bce_dice_loss`
- `class_focal_aux_loss`

最终训练时：

```python
main_loss, bce_loss, dice_loss = bce_dice_loss(...)
focal_loss = class_focal_aux_loss(...)
```

再组合成总 loss。

其中 focal 部分还会乘类别权重和样本权重，所以 rare 类会被进一步强调。

### 3.10 优化器和调度器怎么做

优化器入口：

```python
build_optimizer(model, decoder_lr, image_lr, weight_decay)
```

它会把：

- decoder 参数
- image encoder 参数

分到不同学习率组。

学习率调度器入口：

```python
build_warmup_cosine_scheduler(...)
```

采用：

- warmup
- cosine decay

的组合。

### 3.11 balanced sampler 怎么实现

当：

```python
BALANCED_SAMPLER = True
```

时，脚本会调用：

```python
build_balanced_sampler(dataset, num_samples)
```

基本逻辑是：

1. 统计每类样本数
2. 类越少，权重越高
3. 用 `WeightedRandomSampler` 构建每个 epoch 的采样分布

所以这条线的“类别均衡”不是删样本，而是“按反频率加权采样”。

### 3.12 验证集限样怎么做

验证集太大时，脚本不会简单截断前 N 条，而是调用：

```python
limit_dataset_stratified(dataset, max_samples, seed=42, name="dataset")
```

逻辑是：

1. 先按类别分组
2. 每类先拿基础配额
3. 剩余名额再从 leftovers 里补

这样比直接截断更稳，因为能尽量维持各类分布。

### 3.13 每个 epoch 的训练流程

主入口是：

```python
def train(args):
```

训练阶段完整流程：

1. 创建输出目录并初始化日志
2. 固定随机种子
3. 解析 device
4. 构建模型并可选冻结 image encoder
5. 构建 train/val dataset
6. 打印 dataset 统计
7. 对 val dataset 做分层限样
8. 构建 balanced sampler
9. 构建 train/val dataloader
10. 先做一次 `smoke_test_forward`
11. 构建 optimizer 和 scheduler
12. 如果存在 resume checkpoint，则恢复
13. 开始 epoch 循环
14. 每轮：
    - `train_one_epoch`
    - `validate`
    - 记录 history
    - 保存 `last`
    - 如果更优则保存 `best`
    - 如果 pos-only 更优则保存 `best_pos`
15. 最后写 `results.csv`

### 3.14 checkpoint 保存了什么

checkpoint 保存函数：

```python
save_checkpoint(path, model_only_path, epoch, model, optimizer, scheduler, history, best_overall_iou, best_pos_iou)
```

会保存两份：

1. 完整训练状态
2. 仅模型权重

所以最终有：

- `last.pt`
- `last_model_only.pt`
- `best.pt`
- `best_model_only.pt`
- `best_pos.pt`
- `best_pos_model_only.pt`

### 3.15 训练脚本输出什么指标

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

`results.csv` 里也会把这些主指标和每类 IoU 记录下来。

### 3.16 这个脚本最容易让人误解的点

有几个点要特别说明：

1. 它不训练 GroundingDINO。
   GroundingDINO 只出现在推理脚本。
2. 它训练的是 `EfficientSAM + box prompt`。
3. 它不是多类固定通道语义分割。
   更像实例级 box->mask 学习。
4. 负样本配置存在，但不是完整负样本 box-prompt 分割训练。
5. `FREEZE_IMAGE_ENCODER=True` 时，主要是在训练 decoder 相关部分。

## 4. infer_DINO.py

### 4.1 脚本职责

[infer_DINO.py](/d:/nyz/raytron_project/src/DINO/infer_DINO.py) 是推理闭环脚本。

职责是：

1. 用 `GroundingDINO` 找候选框
2. 用训练好的 `EfficientSAM student` 对这些框出 mask
3. 结果编码成 JSON

这就是“DINO + EfficientSAM”这个名字真正指的东西。

### 4.2 输入与输出

默认输入：

- `TEST_TASK_JSON`
- `TEST_LIST`
- `EFFICIENT_SAM_CKPT`
- `STUDENT_CKPT`

默认输出：

```python
OUTPUT_JSON = ROOT / "test" / "train_output" / "DINO_11cls_fullset_v1" / "pred_test_dino_efficientsam.json"
```

### 4.3 这个脚本如何加载测试图

入口函数：

```python
load_test_image_list(test_tasks_json, test_list)
```

逻辑是：

1. 如果 `test_tasks.json` 存在，优先从里面提取唯一 `image_path`
2. 否则退回 `test_list.txt`

所以它兼容：

- 任务 JSON 驱动
- 纯图片列表驱动

### 4.4 GroundingDINO 适配层是怎么实现的

核心函数：

```python
run_grounding_dino(image_path, class_names)
```

这是推理脚本里最关键的“外部检测器兼容层”。

它按优先级尝试两种实现：

#### 方式 A：HuggingFace transformers 版

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
```

优点：

- 更容易直接跑
- 不依赖本地单独 DINO 仓库结构

做法：

1. 首次调用时全局缓存 processor 和 model
2. 用 `. ".join(class_names) + "."` 拼成多类 prompt
3. 得到原始检测结果
4. 再根据类名做映射和筛选

#### 方式 B：独立 GroundingDINO 仓库 API

如果 HF 方式导入或推理失败，就尝试：

```python
from groundingdino.util.inference import Model as GDModel
from groundingdino.util.inference import predict as gd_predict
```

做法类似，也是：

1. 多类 prompt 合并
2. 出框
3. 再映射成项目内部 `class_name`

#### 如果都失败

会直接抛出异常，提示用户自己适配本地 DINO 安装方式。

所以这个函数本质上是一个“多实现后端适配器”。

### 4.5 检测框如何过滤

检测框过滤函数：

```python
filter_dino_detections(detections, class_names, thresholds, max_per_class)
```

按类做两步处理：

1. 阈值过滤
2. 每类只保留 top-k

也就是说，DINO 原始输出不会直接全部送入 EfficientSAM。

### 4.6 student 模型怎么加载

推理时模型加载分两步：

#### 第一步：加载基础 EfficientSAM

```python
build_model("vitt", efficient_sam_ckpt)
```

#### 第二步：加载 student checkpoint

```python
load_student_model(student_ckpt_path, efficient_sam_ckpt, device)
```

实现上是：

1. 先加载基础权重
2. 再加载训练出来的 student 权重
3. `strict=False`
4. 打印 `missing_keys / unexpected_keys`

所以推理脚本对 checkpoint 格式有一定容错性。

### 4.7 单图推理怎么实现

核心函数：

```python
infer_one_image(model, image_path, dino_detections, device, img_size=768, sam_batch_size=32)
```

流程非常明确：

1. 对图像做训练同款灰度预处理
2. letterbox 到 768
3. 把 DINO 原图框映射到 letterbox 坐标
4. 把同一张图复制成 batch，对一批框一起跑 EfficientSAM
5. 对每个框输出的 logits 做 sigmoid
6. 按类阈值二值化
7. unletterbox 回原图
8. 小连通域过滤
9. 编码成 RLE
10. 从 mask 回算 bbox 和 area
11. 对同类 mask 再做一次 mask NMS

最终每个 instance 至少包含：

- `class_name`
- `score`
- `bbox`
- `area`
- `rle`

### 4.8 为什么要做 mask NMS

函数：

```python
mask_nms(masks, scores, class_names, iou_threshold)
```

这是因为 DINO 可能给同一类目标出多个重叠框，EfficientSAM 也会产出高度重叠 mask。

如果不做 mask NMS：

- 同一目标可能重复提交多份 mask

当前实现只在“同类之间”做 NMS，不跨类抑制。

### 4.9 主函数怎么组织

推理主入口：

```python
main()
```

流程：

1. 解析参数
2. 初始化日志
3. 解析 device
4. 打印关键配置
5. 设置 DINO 后端使用的 device
6. 加载 student 模型
7. 加载测试图列表
8. 逐图推理
9. 统计实例数、空结果图、缺图、DINO 空检图
10. 写 JSON
11. 打印最终统计

### 4.10 输出 JSON 里有什么

输出文件不仅有结果，还附带诊断信息：

- `model_info`
- `timing`
- `stats`
- `results`

其中 `results` 是每张图的实例结果，`stats` 里有：

- 总图片数
- 总实例数
- 每类实例数
- 每类平均分
- 空结果图数量
- 缺失图片数量
- DINO 空检图片数量

所以这个脚本不是只生成提交文件，也顺带是一个推理诊断脚本。

### 4.11 这个脚本最容易误解的点

1. 它不是只跑 EfficientSAM。
   前面还有 DINO 检测。
2. 它的输出不是原生 COCO，而是项目内部统一 JSON。
3. GroundingDINO 不是固定一种实现，脚本里有适配层。
4. student checkpoint 和基础 EfficientSAM checkpoint 不是一回事，二者都要用。

## 5. run_prepare_all.sh

### 5.1 脚本职责

[run_prepare_all.sh](/d:/nyz/raytron_project/src/DINO/run_prepare_all.sh) 是一个数据准备流水线脚本。

它不参与训练和推理，而是负责：

- 把原始图像和 SAM3 mask 整理成训练标注
- 做基础检查
- 切分 train/val
- 导出 GroundingDINO 可用的 COCO 文件

### 5.2 它的输入

脚本中硬编码了：

```bash
IMAGE_DIR="images/image"
MASK_DIR="images/sam3_masks"
OUT_DIR="data/processed_sam3"
PROMPT="defect"
MIN_AREA=20
```

这说明这个脚本更像一套“原始数据预处理工具”，而不是当前 `test/...` 伪标签流程的一部分。

### 5.3 它的执行步骤

步骤分 4 段：

1. `prepare_sam3_pseudo_labels.py`
   - 读取原始图像和 mask
   - 生成 `annotations.json`
2. `check_processed_sam3.py`
   - 检查生成标注是否合理
3. `split_annotations.py`
   - 生成 `train_annotations.json`
   - 生成 `val_annotations.json`
4. `export_groundingdino_coco.py`
   - 分别把 train/val 标注导出成 GroundingDINO 可用的 COCO JSON

### 5.4 这个脚本的价值

它说明这条路线最初设计时，不只是面向当前 11 类 `trainval_label.json / val_label.json` 这套伪标签格式，也兼容过更早的旧任务 JSON，并支持从“原始 mask 文件夹”走起。

可以把它理解成：

- 更底层的数据生产脚本
- 当前训练/推理脚本所依赖数据的一条来源链

## 6. efficient_sam/ 目录的角色

这个目录的职责是提供：

- EfficientSAM 的模型定义
- 构建函数

训练脚本和推理脚本都会从这里导入 `build_efficient_sam`。

所以它是这条线的“核心模型实现目录”。

## 7. models/ 目录的角色

`models/` 目录不是 Python 逻辑主入口，而是模型权重与相关资源存放目录。

当前至少涉及：

- `efficient_sam` 权重
- `groundingdino` 权重

训练时主要用：

- `efficient_sam_vitt.pt`

推理时主要用：

- `efficient_sam_vitt.pt`
- `groundingdino_swint_ogc.pth`
- 训练产出的 `best_model_only.pt`

## 8. 脚本间依赖关系

### 8.1 train_DINO.py 的外部依赖

- `efficient_sam.efficient_sam`
- `torch`
- `opencv`
- `numpy`
- `pycocotools`（RLE 解码时可能需要）

### 8.2 infer_DINO.py 的额外依赖

除了训练脚本那套依赖外，还依赖：

- `transformers` 或独立 `groundingdino`
- `PIL`
- `pycocotools`
- `scipy.ndimage`（连通域过滤时可选使用）

### 8.3 run_prepare_all.sh 的外部依赖

它依赖外部 `scripts/` 目录中的若干数据处理脚本。

也就是说，这个 shell 脚本不是自足的，必须在对应项目结构下才能工作。

## 9. 审代码时建议重点关注的点

如果是给别人审，我建议重点看这些位置：

### 9.1 训练侧

- `EfficientSAMBoxPromptDataset._build_samples`
- `EfficientSAMBoxPromptDataset.__getitem__`
- `build_balanced_sampler`
- `limit_dataset_stratified`
- `train_one_epoch`
- `validate`
- `save_checkpoint`

### 9.2 推理侧

- `run_grounding_dino`
- `filter_dino_detections`
- `load_student_model`
- `infer_one_image`
- `mask_nms`

### 9.3 风险点

- DINO 后端兼容层是否与你实际环境一致
- HF 版 DINO 的 box 坐标解释是否完全正确
- `NUM_WORKERS=0` 导致的训练速度限制
- 负样本是否真正起到了预期作用
- 稀有类过采样是否会引起过拟合

## 10. 一句话总结

这套 `DINO` 代码的真实结构可以概括为：

```text
训练：用伪标签 mask 提取 box，训练一个 box-prompt EfficientSAM student
推理：用 GroundingDINO 找框，再让 student 产 mask，并输出项目统一 JSON
```

所以名字里虽然有 `DINO`，但训练主体是 `EfficientSAM student`，而 `DINO` 主要负责推理时的开放词汇找框。
