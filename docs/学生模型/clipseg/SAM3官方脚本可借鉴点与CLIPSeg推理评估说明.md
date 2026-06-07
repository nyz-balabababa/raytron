# SAM3官方脚本可借鉴点与CLIPSeg推理评估说明

## 一、目的

这份文档用于说明两件事：

1. 官方 `sam3/` 目录里，哪些训练/推理脚本或模块对当前项目有直接参考价值
2. 当前新增的 `CLIPSeg` 推理评估脚本为什么这样实现，它和官方脚本的关系是什么

当前主线是：

- teacher：**官方 SAM3**
- student：**CLIPSeg**
- 主线类别：`person / car / building / tree / animal`

因此，这里的重点不是“复用官方整套训练系统”，而是**借官方协议和数据组织方式，为当前 student 路线服务**。

---

## 二、官方目录里最值得借鉴的部分

### 2.1 官方推理入口

路径：

- [inference.py](D:/nyz/sam3/选手示例工程/to_Df - 官方/inference.py)

这份脚本最有价值的不是模型本身，而是它的**任务组织协议**：

1. 读取 `task json`
2. 按 `image_path` 分组
3. 同一张图汇总 `unique_prompts`
4. 对每个 prompt 推理
5. 输出 `ann_id -> rle`
6. 对未命中的 prompt 输出空掩码

这套协议和比赛的提交格式直接对齐，也适合后续做验证集评估。

当前项目里：

- `generate_sam3_labels.py` 复用了这套官方 SAM3 推理骨架
- `clipseg_infer_eval.py` 则复用了它的**外层任务协议和输出格式**

也就是说，当前 `CLIPSeg` 推理评估脚本虽然不是跑 `SAM3`，但它的任务组织方式是向官方推理入口看齐的。

---

### 2.2 官方训练数据组织

路径：

- [coco_json_loaders.py](D:/nyz/sam3/选手示例工程/to_Df - 官方/sam3/train/data/coco_json_loaders.py)

这里最值得借鉴的是：

- 每张图不是只保留一个类别
- 而是组织成 **image × query(prompt/category)** 的训练单元
- 并且支持 `include_negatives=True`

这和我们当前对 `CLIPSeg` 的理解是一致的：

- 一张图里，某个 prompt 命中，是正样本
- 某个 prompt 不命中，也应该作为负样本存在

所以当前 `clipseg_train.py` 里把 `hit=false` 纳入训练，并不是随意加的，而是和官方 query-based 训练思路一致。

---

### 2.3 官方训练 YAML / Trainer 系统

路径示例：

- [roboflow_v100_eval.yaml](D:/nyz/sam3/选手示例工程/to_Df - 官方/sam3/train/configs/roboflow_v100/roboflow_v100_eval.yaml)
- [roboflow_v100_full_ft_100_images.yaml](D:/nyz/sam3/选手示例工程/to_Df - 官方/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml)

这些文件能提供的信息主要是：

- 官方训练是完整的 `Trainer + Dataset + Transforms + Matcher + Loss + Optimizer` 系统
- 训练输入也是 query-oriented
- 验证、日志、优化器、layer decay 等都做成了完整框架

这部分的价值在于：

- 有助于理解官方是怎么训练 `SAM3` 的
- 如果后面要重新训练/微调 `SAM3`，这些配置有参考意义

但对当前项目的 `CLIPSeg` 路线来说，这套系统**不能直接拿来用**。

原因很简单：

- 它服务的是 `SAM3` 自己的模型结构
- 当前 student 模型是 HuggingFace 的 `CLIPSeg`
- 两者的输入接口、loss 头、decoder 和推理方式都不一样

因此这里只能借思路，不能直接迁移代码。

---

## 三、当前 CLIPSeg 推理评估脚本为什么不直接照抄官方

当前脚本路径：

- [clipseg_infer_eval.py](D:/nyz/raytron_project/src/train/clipseg/clipseg_infer_eval.py)

它不能直接照抄官方 `inference.py`，原因不是“官方写得不好”，而是**底层模型完全不同**。

### 3.1 官方 `inference.py` 是 SAM3 原生推理

官方脚本内部依赖：

- `build_sam3_image_model`
- `Sam3Processor`
- `processor.set_text_prompt(...)`
- `state["masks"] / state["scores"]`

这些都是 `SAM3` 专属接口。

`CLIPSeg` 没有这套 processor/state 机制，也没有实例级 proposal 再聚合的流程。

所以对于 `CLIPSeg`，不能直接复用：

- `Sam3Processor`
- `aggregate_prompt_prediction(...)`
- 官方实例级 mask 合并逻辑

---

### 3.2 当前 `clipseg_infer_eval.py` 借的是“外层协议”

当前 `CLIPSeg` 评估脚本真正借鉴官方的部分有：

1. `task json -> image 分组`
2. 同图聚合 `unique_prompts`
3. `ann_id -> rle` 的输出协议
4. 空结果补空掩码
5. `predictions.json` 的整体结构

这部分是**模型无关的协议层**，适合直接复用。

---

### 3.3 当前 `clipseg_infer_eval.py` 自己实现的是“内层推理”

因为 student 是 `CLIPSeg`，所以脚本内部必须自己做：

1. 加载 `best.pt`
2. 复用当前 `clipseg_train.py` 的图像预处理
3. 对每个 prompt 做 tokenizer 编码
4. 前向得到 `logits`
5. 按类阈值二值化
6. 从 pad 后空间裁回原图尺寸
7. 转为 RLE

也就是说：

- **外层协议**向官方对齐
- **内层推理**向当前训练脚本对齐

这才是当前 student 评估脚本最合理的实现方式。

---

## 四、当前脚本相对官方脚本的关键差异

### 4.1 模型不同

官方：

- `SAM3`
- processor 驱动
- 文本 prompt 进入 `Sam3Processor`
- 输出实例 proposal，再按阈值合并

当前：

- `CLIPSeg`
- `CLIPSegProcessor.tokenizer`
- 图像单次编码 + 文本条件前向
- 输出单通道 logits，直接阈值化

---

### 4.2 预处理不同

官方 `inference.py` 对输入图像基本是直接 `RGB` 读取。

当前 `CLIPSeg` 评估脚本为了和训练保持一致，额外做了：

- 伪彩转灰
- 黑热/白热统一
- CLAHE
- 去噪
- 锐化
- 1024 resize + pad
- CLIP 标准化

这是必须保留的，因为当前学生模型训练时就是按这套分布学的。

---

### 4.3 阈值逻辑不同

官方 `inference.py` 默认更像统一阈值推理。

当前 `CLIPSeg` 评估脚本直接复用训练主线阈值：

- `person = 0.70`
- `car = 0.70`
- `building = 0.70`
- `tree = 0.60`
- `animal = 0.60`

原因是当前我们不是在追求“一个通用官方默认值”，而是在追求：

**验证集推理评估尽量对齐当前训练设定和 teacher 输出规则。**

---

## 五、当前结论

对当前项目最有价值的不是“把官方训练推理脚本整套搬过来”，而是：

1. 借官方 `inference.py` 的任务组织和输出协议
2. 借官方 `coco_json_loaders.py` 的 query / negative 组织思路
3. 在 student 侧，用自己的模型前向逻辑实现与之兼容的评估脚本

因此，当前 `clipseg_infer_eval.py` 的实现原则是正确的：

- 没有重新发明协议
- 没有错误套用 `SAM3` 内部接口
- 让 `CLIPSeg` 的验证推理结果可以直接接现有分析链路

---

## 六、后续建议

当前 `CLIPSeg` 路线建议保留三层脚本分工：

1. **teacher 伪标签生成**
   - `src/prompt/generate_sam3_labels.py`

2. **student 训练**
   - `src/train/clipseg/clipseg_train.py`

3. **student 推理评估**
   - `src/train/clipseg/clipseg_infer_eval.py`

这样：

- 训练指标用于选 checkpoint
- 推理评估用于看真实部署表现
- 两者不再混为一谈

如果后面要继续整理，最适合补的是：

- `CLIPSeg` 推理评估脚本的运行方式
- 与 `analyze_prompt_predictions.py` 的联动命令
- 评估产物目录结构说明
