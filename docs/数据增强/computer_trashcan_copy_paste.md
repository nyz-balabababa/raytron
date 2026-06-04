# YOLO-World 尾类离线 Copy-Paste 脚本说明

> 对应脚本: `src/tools/offline_copy_paste_tail.py`

## 一、脚本目的

这个脚本的定位很明确：

- 面向当前 `YOLO-World + FastSAM` 的尾类增强
- 只做 **离线 Copy-Paste**
- 当前主要服务两类：
  - `computer`
  - `trash can`

它不会修改原始伪标签文件，也不会自动改训练配置。  
它的作用是：

1. 从 `d5&7` 伪标签中抽高质量前景实例
2. 从全训练集里抽“不含该类”的同域背景图
3. 离线合成增强图
4. 同步生成：
   - YOLO 检测训练需要的 `bbox` 标签
   - 以后给 `CLIPSeg` 使用的 `pred_json`

一句话：

**这个脚本是“尾类离线增广数据生成器”，不是训练脚本。**

---

## 二、默认输入

脚本默认读三份输入。

### 2.1 前景伪标签

```python
DEFAULT_FG_JSON = test/prompt_test_output/train_d5&7_tasks/pred_train_d5&7_tasks.json
```

作用：

- 从 `d5&7` 尾类专项伪标签里抽 `computer / trash can` 的高质量前景实例

### 2.2 背景伪标签

```python
DEFAULT_BG_JSON = test/prompt_test_output/train_tasks/new_train.json
```

作用：

- 作为背景样本池的索引来源
- 同时也是最终 `pred_copy_paste_merged.json` 的底稿

也就是说，默认输出的 merged json 本质是：

**`new_train.json + Copy-Paste 新样本`**

### 2.3 训练列表

```python
DEFAULT_TRAIN_LIST = test/train_list.txt
```

作用：

- 用来限制背景图只从当前训练集范围内抽取

---

## 三、整体流程

脚本流程分成 6 步。

### 3.1 提取前景池

从 `pred_train_d5&7_tasks.json` 中读取 `computer / trash can` 命中样本，对每个目标：

1. 按 `score` 做阈值过滤
2. RLE 解码成二值 mask
3. 连通域拆分成单实例
4. 按面积、边界、尺寸再做筛选
5. 把通过筛选的实例存到前景池

前景池里保存的信息包括：

- 类别名
- 来源图片
- score
- 裁剪框
- 单实例 mask 的 RLE
- 实例面积
- 数据源 `source`

### 3.2 构建背景池

从 `new_train.json` 中读取所有训练图，只保留：

- 在 `train_list.txt` 中
- 且 **不含目标类** 的图片

分别为 `computer` 和 `trash can` 建背景池。

### 3.3 抽取前景与背景

每次生成一个样本时：

1. 从前景池中随机选一个实例
2. 从目标类对应的背景池中选一张背景图

其中 `computer` 和 `trash can` 的背景选择策略不同：

- `computer`
  - 优先抽与前景同 `source` 的背景
  - 场景更保守
- `trash can`
  - 不要求同 `source`
  - 背景更宽松

### 3.4 对前景做轻量变换

当前只做保守增强：

- 随机水平翻转
- 按类别配置做轻量缩放

不做：

- 大旋转
- 颜色增强
- 几何剧烈扰动

### 3.5 放置与粘贴

放置前会先构建背景图中已有 prompt 的占用 mask。  
然后在背景中随机找位置，要求：

- 不贴边
- 与已有目标重叠不超过阈值
- 尝试次数不超过 `MAX_PLACEMENT_ATTEMPTS`

粘贴方式：

- 先把前景灰度图扩成 RGB
- 用前景 mask 生成 alpha
- 对 alpha 做高斯模糊
- 用 alpha 混合到背景图

也就是：

**不是硬切贴上去，而是做了轻微边缘羽化。**

### 3.6 同步生成训练与伪标签输出

每张增强图生成后，会同步写出：

1. 增强图 `jpg`
2. YOLO 检测 `bbox` 标签
3. `pred_json` 记录
4. `manifest.csv`

---

## 四、关键参数

## 4.1 全局参数

```python
RNG_SEED = 42
MAX_PLACEMENT_ATTEMPTS = 40
EDGE_BLUR_SIGMA = 1.2
MIN_MARGIN = 4
```

含义：

- `RNG_SEED`
  - 随机种子，保证可复现
- `MAX_PLACEMENT_ATTEMPTS`
  - 每张背景图最多尝试 40 次找粘贴位置
- `EDGE_BLUR_SIGMA`
  - 前景边缘羽化强度
- `MIN_MARGIN`
  - 与图像边缘的最小距离

## 4.2 类别参数

脚本里通过 `TARGET_SPECS` 单独控制每个类。

### `trash can`

```python
"trash can": {
    "num_samples": 400,
    "min_score": 0.65,
    "min_area_px": 120,
    "max_area_ratio": 0.12,
    "min_side_px": 10,
    "scale_range": (0.85, 1.20),
    "same_source_only": False,
    "max_overlap_ratio": 0.05,
}
```

解释：

- `num_samples = 400`
  - 默认生成 400 张 `trash can` 增强样本
- `min_score = 0.65`
  - 伪标签置信度低于 0.65 的前景不入池
- `min_area_px = 120`
  - 太小的实例不做前景
- `max_area_ratio = 0.12`
  - 面积过大、接近整图的实例不做前景
- `min_side_px = 10`
  - 宽高太小的实例过滤
- `scale_range = (0.85, 1.20)`
  - 粘贴时轻量缩放
- `same_source_only = False`
  - 背景不要求与前景来自同一 source
- `max_overlap_ratio = 0.05`
  - 与背景已有目标最大重叠比例

### `computer`

```python
"computer": {
    "num_samples": 150,
    "min_score": 0.60,
    "min_area_px": 100,
    "max_area_ratio": 0.08,
    "min_side_px": 8,
    "scale_range": (0.90, 1.10),
    "same_source_only": True,
    "max_overlap_ratio": 0.03,
}
```

解释：

- `num_samples = 150`
  - 默认生成 150 张 `computer` 增强样本
- `min_score = 0.60`
  - 比 `trash can` 稍低
- `max_area_ratio = 0.08`
  - 对大块误检更严格
- `scale_range = (0.90, 1.10)`
  - 缩放更保守
- `same_source_only = True`
  - 优先保留更接近的场景上下文
- `max_overlap_ratio = 0.03`
  - 对重叠更严格

---

## 五、命令行参数

脚本支持命令行覆写默认配置。

### 基础参数

```python
--fg-json
--bg-json
--train-list
--output-dir
--seed
```

### 指定增强类别

```python
--targets computer "trash can"
```

可选值：

- `computer`
- `trash can`

### 单独指定生成数量

```python
--num-trash-can
--num-computer
```

这两个参数会覆盖 `TARGET_SPECS` 里的默认值。

例如：

```powershell
D:\anconda3\envs\rayton\python.exe src\tools\offline_copy_paste_tail.py --num-trash-can 400 --num-computer 150
```

只生成 `trash can`：

```powershell
D:\anconda3\envs\rayton\python.exe src\tools\offline_copy_paste_tail.py --targets "trash can" --num-trash-can 300
```

---

## 六、输出文件

默认输出目录：

```text
test/copy_paste_tail/
```

脚本会生成以下文件。

## 6.1 `images/`

目录：

```text
test/copy_paste_tail/images/
```

内容：

- 增强后的新图片
- 文件名形如：
  - `cp_computer_00000.jpg`
  - `cp_trash_can_00001.jpg`

作用：

- 给 `YOLO-World` 训练时作为新样本图

## 6.2 `labels/`

目录：

```text
test/copy_paste_tail/labels/
```

内容：

- 与 `images/` 一一对应的 YOLO 检测标签
- 格式是：
  - `class_id cx cy w h`

作用：

- 直接给 YOLO 检测训练使用

注意：

- 如果是重新跑 `world_train_v4.py`，这批标签通常不是必需拷贝的
- 因为 `world_train_v4.py` 可以根据 `pred_json` 重新生成 bbox

## 6.3 `pred_copy_paste_only.json`

路径：

```text
test/copy_paste_tail/pred_copy_paste_only.json
```

内容：

- **只包含新生成的 Copy-Paste 样本**
- 每条记录仍保持项目统一格式：
  - `image_path`
  - `prompts`
  - `hit / score / instances / rle`

作用：

- 单独查看这批增强样本
- 与原伪标签分开管理
- 可用于尾类专项实验

## 6.4 `pred_copy_paste_merged.json`

路径：

```text
test/copy_paste_tail/pred_copy_paste_merged.json
```

内容：

- `DEFAULT_BG_JSON` 中的全部记录
- 再加上 Copy-Paste 新生成的样本

默认情况下它等价于：

**`new_train.json + Copy-Paste 新样本`**

作用：

- 后续给 `CLIPSeg` 接入最方便
- 也适合做“原训练集 + 增强样本”的统一训练源

注意：

- 这里的 `merged` 不是类别合并
- 只是样本级追加

## 6.5 `manifest.csv`

路径：

```text
test/copy_paste_tail/manifest.csv
```

内容：

- 新样本路径
- 目标类别
- 前景来源图
- 背景来源图
- 前景 score
- 缩放比例
- 前景面积

作用：

- 追踪样本来源
- 事后排查坏样本
- 分析哪类组合更有效

---

## 七、进度条

脚本当前已经加了 `tqdm` 进度条，主要显示：

- `提取前景池`
- `生成 trash can`
- `生成 computer`

便于观察：

- 当前增强进度
- 是否出现大量样本因为放置失败被跳过

---

## 八、如何接到后续训练

### 接到 `YOLO-World v4`

不建议直接只用 `pred_copy_paste_only.json`。  
通常应该用：

- 原 `d5&7` 伪标签
- 加上 `pred_copy_paste_only.json`

也就是生成一份新的：

- `pred_train_d5&7_tasks_with_copy_paste.json`
- `train_list_d5&7_with_copy_paste.txt`

### 接到 `CLIPSeg`

更方便的做法通常是直接使用：

- `pred_copy_paste_merged.json`

因为 `CLIPSeg` 更接近直接消费 `pred_json`。

---

## 九、一句话总结

`offline_copy_paste_tail.py` 是一个专门面向 `computer / trash can` 尾类的离线数据增强脚本：它从 `d5&7` 抽高质量前景、从全训练集抽同域背景，生成增强图、YOLO bbox 标签，以及既能给 `world` 用、也能给 `CLIPSeg` 接入的 `pred_json`。 
