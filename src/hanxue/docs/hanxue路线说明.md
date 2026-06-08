# Hanxue 路线说明

## 1. 路线定位

`src/hanxue` 现在的定位是：

- 作为 `CLIPSeg` 的**同协议对照学生模型**
- 输入仍然是：`图像 + text prompt`
- 输出仍然是：`单通道二值分割 mask`
- 训练监督仍然使用：`SAM3 teacher 伪标签`

这条路线不是官方完整 `EfficientSAM3` 推理链路，而是：

- 图像编码器：`EfficientSAM ViT-T image encoder`
- 文本编码器：`Chinese-CLIP text encoder`
- 融合解码器：`FiLMFusionDecoder`

也就是一个“EfficientSAM backbone student”。

## 2. 和 CLIPSeg 的同构原则

为了让实验可比，这条路线现在和 `CLIPSeg` 保持以下同构：

- 主线类别一致：`person / car / building / tree / animal`
- teacher 伪标签输入一致：
  - `test/prompt_test_output/train_tasks/pred_train_tasks.json`
  - `test/prompt_test_output/val_tasks1/pred_val_tasks1.json`
- 训练/验证划分一致：
  - `test/train_list.txt`
  - `test/val_list.txt`
- 伪标签阈值一致：
  - `person/car/building = 0.70`
  - `tree/animal = 0.60`
- 训练前图像预处理一致：
  - 伪彩图转灰度
  - 黑热/白热统一
  - 动态 CLAHE
  - 动态去噪
  - 动态锐化
  - 等比例缩放 + padding 到 `1024 x 1024`
- 样本策略一致：
  - 可选 score 二次过滤，默认关闭
  - 纳入 `hit=false` 负样本
  - 负样本比例 `0.15`
  - 负样本权重 `0.30`
  - 稀有类过采样
- 验证评估一致：
  - 按类阈值算 `IoU / Dice / Precision / Recall`
  - 额外提供独立推理评估脚本

## 3. 目录与脚本功能

### 3.1 配置

文件：[config_hanxue.py](D:/nyz/raytron_project/src/hanxue/config_hanxue.py)

作用：

- 集中管理五类主线配置
- 管理 teacher 伪标签路径
- 管理训练超参数
- 管理图像预处理阈值
- 管理 prompt 增强、负样本、过采样策略

这是整个 `hanxue` 路线的统一入口。

### 3.2 数据集

文件：[data/dataset.py](D:/nyz/raytron_project/src/hanxue/data/dataset.py)

作用：

- 读取 prompt 风格 teacher 伪标签 JSON
- 按 `train_list.txt / val_list.txt` 过滤样本
- 将 `rle` 解码为 mask，并缓存为 PNG
- 构造：
  - 正样本：`hit=true`
  - 负样本：`hit=false`
- 执行和 `CLIPSeg` 同构的图像预处理
- 执行：
  - prompt 动态增强
  - 水平翻转
  - 稀有类过采样
- 输出训练所需张量：
  - `image`
  - `mask`
  - `input_ids`
  - `attention_mask`
  - `sample_weight`

### 3.3 模型主体

文件：

- [models/image_encoder.py](D:/nyz/raytron_project/src/hanxue/models/image_encoder.py)
- [models/text_encoder.py](D:/nyz/raytron_project/src/hanxue/models/text_encoder.py)
- [models/fusion_decoder.py](D:/nyz/raytron_project/src/hanxue/models/fusion_decoder.py)
- [models/custom_sam_model.py](D:/nyz/raytron_project/src/hanxue/models/custom_sam_model.py)

作用：

#### `image_encoder.py`

- 加载 `EfficientSAM ViT-T` 权重
- 只保留 `image_encoder`
- 补齐 `pixel_mean / pixel_std` 归一化

#### `text_encoder.py`

- 加载本地 `Chinese-CLIP` 文本编码器
- 冻结文本参数
- 输出全局文本特征

#### `fusion_decoder.py`

- 用文本全局特征生成 FiLM 参数
- 调制图像特征
- 输出单通道分割 logits

#### `custom_sam_model.py`

- 把图像编码器、文本编码器、解码器拼起来
- 统一训练与推理调用接口

### 3.4 训练

文件：[train.py](D:/nyz/raytron_project/src/hanxue/train.py)

作用：

- 加载 `InfraredPromptDataset`
- 构建 `CustomSAMWorldModel`
- 按 `decoder_lr / image_lr` 分组优化
- 执行：
  - warmup
  - cosine 学习率调度
  - BCE + Dice + Focal 混合损失
- 每个 epoch 做验证
- 输出：
  - `best.pt`
  - `last.pt`
  - `results.csv`
  - 训练日志

### 3.5 统一推理入口

文件：[inference.py](D:/nyz/raytron_project/src/hanxue/inference.py)

作用：

- 现在统一承担两类推理任务：
  - `submit` 模式：提交/部署推理
  - `eval` 模式：验证集推理评估
- 负责：
  - 任务读取
  - 图像预处理
  - mask 恢复到原图尺寸
  - RLE 编码
  - 提交格式 `predictions.json` 输出
  - 验证格式：
    - `*_predictions.json`
    - `pred_*_hanxue.json`
    - `*_meta.json`

也就是说，原先单独的 `infer_eval.py` 已经并入这里，顶层脚本收敛成：

- `train.py`
- `inference.py`

## 4. 实验流程

### 步骤 1：生成五类任务 JSON

来源脚本：

- [gen_val_tasks.py](D:/nyz/raytron_project/src/prompt/gen_val_tasks.py)

输出：

- `test/json/train_tasks.json`
- `test/json/val_tasks1.json`

### 步骤 2：生成 SAM3 teacher 伪标签

来源脚本：

- [generate_sam3_labels.py](D:/nyz/raytron_project/src/prompt/generate_sam3_labels.py)

输出：

- `pred_train_tasks.json`
- `pred_val_tasks1.json`

### 步骤 3：训练 hanxue 学生模型

运行：

- [train.py](D:/nyz/raytron_project/src/hanxue/train.py)

输入：

- teacher 伪标签
- 五类任务划分

输出：

- `test/train_output/hanxue_v1/best.pt`

### 步骤 4：做验证推理评估

运行：

- [inference.py](D:/nyz/raytron_project/src/hanxue/inference.py)

输出：

- `test/inference_eval/hanxue_v1/...`

运行方式：

```powershell
python src/hanxue/inference.py --mode eval --tasks test/json/val_tasks1.json
```

### 步骤 5：统一错误分析

来源脚本：

- [analyze_prompt_predictions.py](D:/nyz/raytron_project/src/prompt/analyze_prompt_predictions.py)

作用：

- 统计假阳性
- 统计漏检
- 统计低对比表现
- 定位整图级坏样本

## 5. 这条路线保留了什么，改造了什么

### 保留的部分

- `EfficientSAM` 图像编码器主干
- `Chinese-CLIP` 文本编码器
- `FiLMFusionDecoder`
- `CustomSAMWorldModel`
- 原有 `inference.py` 的任务/RLE 输出骨架

### 改造的部分

- 补齐了缺失的数据集实现
- 对齐到五类主线
- 对齐到 `CLIPSeg` 的 teacher 伪标签协议
- 对齐到 `CLIPSeg` 的训练/验证划分
- 对齐到 `CLIPSeg` 的预处理、负样本、过采样、阈值逻辑
- 增加了完整训练记录和独立推理评估链路

## 6. 当前边界

这条路线现在已经能和 `CLIPSeg` 做**同协议对比实验**，但它仍然不是官方完整 `EfficientSAM3`：

- 图像端来自 `EfficientSAM`
- 文本端和解码端是自定义的

所以这条线的比较意义是：

- 比较 `EfficientSAM backbone student` 和 `CLIPSeg` 在同一 teacher、同一任务、同一评估协议下的效果差异

而不是比较“官方 EfficientSAM3 和 CLIPSeg”的原生能力。
