# A-mid 技术文档

## 1. 路线定位

`A-mid` 是 `src/ESAM-CCLIP-11` 体系下新增的一条 11 类优化分支，目标是放在：

- `方案A rare recall版`
- `A-lite 保 old 精度版`

之间，作为一个折中版本。

它的设计目标不是改模型结构，而是只通过：

- 采样比例
- rare oversample
- class weights
- 负样本比例
- 默认热启动权重
- 默认 text cache / sweep 输出路径

去调整 old 类和 rare 类之间的平衡。

## 2. 设计原则

`A-mid` 遵守以下限制：

- 不改模型结构
- 不改 decoder 结构
- 不改 11 类类别顺序
- 不改官方推理输出格式
- 不换 EfficientSAM / ChineseCLIP backbone
- 不引入新依赖
- 不新增新 loss / 新 head / 多尺度训练
- 不解冻 image encoder
- 不解冻 text encoder

这条线本质上仍然是：

- `ESAM + ChineseCLIP + decoder-only`
- `freeze image encoder`
- `freeze text encoder`
- `train decoder only`

## 3. 目录结构

当前目录为：

- [config_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/config_esam_cclip_11.py)
- [train_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/train_esam_cclip_11.py)
- [sweep_thresholds_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/sweep_thresholds_11.py)
- [common_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/common_esam_cclip_11.py)
- [dataset_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/dataset_esam_cclip_11.py)
- [model_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/model_esam_cclip_11.py)
- [prompt_prototypes.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/prompt_prototypes.py)
- [run_train_mid.sh](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/run_train_mid.sh)
- [run_sweep_mid.sh](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/run_sweep_mid.sh)

## 4. 实现方式

`A-mid` 现在已经调整为一条独立训练线，不再通过 wrapper 去调用根线脚本。

具体分成两层：

### 4.1 本地独立配置

[config_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/config_esam_cclip_11.py) 是 `A-mid` 的独立配置文件。

它显式定义了：

- 路径
- 类别
- prompt prototype
- threshold grid
- rare oversample
- class weights
- 训练默认参数

这部分不依赖 `A` 或 `A-lite` 的配置文件。

### 4.2 本地训练 / sweep 入口

- [train_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/train_esam_cclip_11.py)
- [sweep_thresholds_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/sweep_thresholds_11.py)

这两个脚本现在都直接放在 `A-mid` 目录内独立运行：

- 在文件顶部提供 `A-mid` 自己的用户配置区
- 训练参数默认走 `A-mid` 本地配置
- 支持 IDE 直接运行
- 支持命令行覆盖默认参数

这样做的目的：

- 不再依赖根线入口
- 不再出现跨目录调度
- 保留原有进度条
- 保留 GPU 调用逻辑
- 保留原有日志和 checkpoint 行为

### 4.3 本地核心模块

- `common_esam_cclip_11.py`
- `dataset_esam_cclip_11.py`
- `model_esam_cclip_11.py`
- `prompt_prototypes.py`

当前这几个文件已经直接存在于 `A-mid` 目录中，训练和 sweep 只导入本地副本。

这样做的目的：

- 让 `A-mid` 目录结构完整
- 导入路径稳定
- 避免修改 `A-mid` 时连带影响根线

### 4.4 单一引用来源

当前 `A-mid` 的代码引用策略已经统一为：

- 配置文件只使用 `A-mid` 自己目录下的 [config_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/config_esam_cclip_11.py)
- 训练入口只运行 `A-mid` 自己目录下的 [train_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/train_esam_cclip_11.py)
- sweep 入口只运行 `A-mid` 自己目录下的 [sweep_thresholds_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/sweep_thresholds_11.py)
- 公共模块只导入 `A-mid` 自己目录下的本地文件

也就是说，`A-mid` 已经不再依赖 `src/ESAM-CCLIP-11` 或 `A-lite` 的代码执行链。

## 5. 类别定义

`A-mid` 固定使用以下 11 类：

```python
CLASSES = [
    "person",
    "car",
    "building",
    "tree",
    "animal",
    "trash can",
    "window",
    "door",
    "fence",
    "pole_light",
    "motorcycle",
]
```

类别顺序不允许改动。

## 6. 数据与路径

当前默认数据路径：

- `TRAIN_JSON = test/clean_rare/label/all-nocomputer.json`
- `VAL_JSON = test/clean_rare/label/val-nocomputer.json`
- `ALL_JSON = test/clean_rare/label/all-nocomputer.json`
- `TRAIN_LIST = test/trainval_list.txt`
- `VAL_LIST = test/val_list.txt`

当前 `ROOT` 定义为：

```python
ROOT = Path(__file__).resolve().parents[2]
```

适用于目录结构：

```text
project_root/
  src/
    ESAM-CCLIP-11-mid/
      config_esam_cclip_11.py
```

## 7. A-mid 核心配置

### 7.1 run name

```python
RUN_NAME = "ESAM-CCLIP-11-rare-balanced-mid"
MODEL_TYPE = "esam_chineseclip_decoder_11_rare_balanced_mid"
```

### 7.2 冻结策略

```python
FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
TRAIN_DECODER_ONLY = True
USE_PROMPT_PROTOTYPE = True
```

### 7.3 学习率与训练默认参数

```python
DECODER_LR = 1e-4
IMAGE_LR = 0.0
TEXT_LR = 0.0
EPOCHS = 4
BATCH_SIZE = 4
WORKERS = 4
AMP = True
```

### 7.4 负样本配置

```python
NEGATIVE_SAMPLE_RATIO = 0.03
NEGATIVE_SAMPLE_WEIGHT = 0.20
INCLUDE_NEGATIVE_SAMPLES = True
VAL_INCLUDE_NEGATIVE_SAMPLES = False
```

### 7.5 old / rare 重平衡

```python
OLD_CLASS_SAMPLE_RATIO = 0.55
RARE_CLASS_KEEP_RATIO = 1.0
```

含义：

- old 类正样本适度降采样
- rare 类正样本全部保留

### 7.6 rare oversample

```python
RARE_OVERSAMPLE = {
    "trash can": 3,
    "window": 2,
    "door": 2,
    "fence": 3,
    "pole_light": 3,
    "motorcycle": 2,
}
```

### 7.7 class weights

```python
CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 1.05,
    "building": 1.0,
    "tree": 1.0,
    "animal": 1.0,
    "trash can": 2.0,
    "window": 1.7,
    "door": 1.6,
    "fence": 1.8,
    "pole_light": 2.0,
    "motorcycle": 1.9,
}
```

### 7.8 pole_light prompt alias

`pole_light` 的 prompt prototype 保留了多别名支持，至少包含：

```python
["pole_light", "pole light", "street light", "lamp", "light pole", "路灯", "灯杆"]
```

## 8. 训练流程

训练入口是 [train_esam_cclip_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/train_esam_cclip_11.py)。

它会先覆盖这些默认项：

```python
DEFAULT_RESUME_BEST = OUTPUT_ROOT / "ESAM-CCLIP-11-fastfinetune" / "final_fullset.pt"
DEFAULT_RESUME = None
DEFAULT_NO_VAL = True
DEFAULT_NO_TRAIN_SPLIT_FILTER = True
DEFAULT_NO_AUTO_RESUME = True
DEFAULT_REBUILD_TEXT_CACHE = True
DEFAULT_TEXT_CACHE_PATH = TEXT_CACHE_PATH.with_name("text_emb_11_rare_balanced_mid.pt")
DEFAULT_RUN_NAME = RUN_NAME
```

这表示：

- 默认从 `ESAM-CCLIP-11-fastfinetune/final_fullset.pt` 热启动
- 默认是全集训练直提交流程
- 默认不自动续训
- 默认重建 `A-mid` 自己的 text cache

### 8.1 必保留日志

训练过程中保留以下关键日志，方便确认真的跑的是 `A-mid`：

```text
loaded checkpoint path=%s
no_auto_resume=%s
negative_sample_ratio=%s
old_class_sample_ratio=%s
rare_class_keep_ratio=%s
rare_oversample=%s
class_weights=%s
old_classes=%s
rare_classes=%s
```

### 8.2 训练输出

默认输出目录：

```text
test/train_output/ESAM-CCLIP-11-rare-balanced-mid/
```

主要输出包括：

- `last.pt`
- `final_fullset.pt`
- `config_used.json`
- `results.csv`
- `results.png`

如果开启 `--train_eval_after`，还会额外输出：

- `train_diagnostic_metrics.json`
- `train_diagnostic_per_class.csv`

## 9. 数据集重平衡逻辑

`A-mid` 当前不重写 dataset 主逻辑，而是复用现有 `dataset_esam_cclip_11.py` 实现。

目标上要求它保留以下行为：

1. 正样本权重：

```python
"sample_weight": max(score, LOSS_WEIGHT_FLOOR)
```

2. old 类正样本按 `OLD_CLASS_SAMPLE_RATIO` 随机保留
3. rare 类正样本按 `RARE_CLASS_KEEP_RATIO` 保留
4. rare 类按 `RARE_OVERSAMPLE` 复制
5. 负样本按 `NEGATIVE_SAMPLE_RATIO` 采样
6. 负样本权重使用 `NEGATIVE_SAMPLE_WEIGHT`

另外需要观察这些日志是否正常输出：

```text
positive stats raw
positive stats after rebalance
rebalance summary
negative stats
```

如果后续发现底层 dataset 还没有完整覆盖这套日志格式，应当优先在底层公共 dataset 上补齐，而不是在 `A-mid` 单独分叉重写。

## 10. 阈值 sweep

阈值搜索入口是 [sweep_thresholds_11.py](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/sweep_thresholds_11.py)。

默认配置：

```python
DEFAULT_CHECKPOINT = Path("test/train_output/ESAM-CCLIP-11-rare-balanced-mid/final_fullset.pt")
DEFAULT_TEXT_CACHE_PATH = TEXT_CACHE_PATH.with_name("text_emb_11_rare_balanced_mid.pt")
DEFAULT_OUTPUT_DIR = Path("test/train_output/threshold_sweep_esam_11_rare_balanced_mid")
DEFAULT_WRITE_BACK_CHECKPOINT = True
DEFAULT_WRITE_BACK_PATH = Path("model/submit-rsam-rare-balanced-mid/sam3.pt")
DEFAULT_REBUILD_TEXT_CACHE = True
DEFAULT_MAX_SAMPLES_PER_CLASS = 500
```

### 10.1 采样方式

支持：

- `--max_samples_per_class`

并保留了“每类收满即提前停止”的逻辑，用来避免扫完整个 val。

### 10.2 概率计算稳定化

保留 stable sigmoid：

```python
logit = np.clip(sample["logit"], -50.0, 50.0)
prob = 1.0 / (1.0 + np.exp(-logit))
```

### 10.3 写回 checkpoint

sweep 结束后会把最佳阈值和后处理配置写回 checkpoint，并保存到：

```text
model/submit-rsam-rare-balanced-mid/sam3.pt
```

## 11. 一键运行脚本

### 11.1 训练

[run_train_mid.sh](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/run_train_mid.sh)

```bash
#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/ESAM-CCLIP-11-mid/train_esam_cclip_11.py \
  --epochs 3 \
  --batch_size 4 \
  --workers 4 \
  --train_eval_after \
  --train_eval_max_samples 3000
```

### 11.2 sweep

[run_sweep_mid.sh](/d:/nyz/raytron_project/src/ESAM-CCLIP-11-mid/run_sweep_mid.sh)

```bash
#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/ESAM-CCLIP-11-mid/sweep_thresholds_11.py \
  --checkpoint test/train_output/ESAM-CCLIP-11-rare-balanced-mid/final_fullset.pt \
  --write_back_checkpoint \
  --write_back_path model/submit-rsam-rare-balanced-mid/sam3.pt \
  --max_samples_per_class 500 \
  --rebuild_text_cache
```

## 12. 与 A / A-lite 的区别

可以简单理解为：

- `A`：更偏 rare recall，old 类压得更重
- `A-lite`：更偏保 old 精度，rare 拉升更克制
- `A-mid`：中间折中

当前 `A-mid` 相对 `A-lite` 的主要变化：

- `NEGATIVE_SAMPLE_RATIO` 从更低值适度提高/调整为中间档
- `NEGATIVE_SAMPLE_WEIGHT` 调整为中间档
- `OLD_CLASS_SAMPLE_RATIO = 0.55`
- `RARE_CLASS_KEEP_RATIO = 1.0`
- rare oversample 仍保留，但弱于更激进 rare 版
- class weights 对 rare 类有提升，但不走极端

## 13. 当前已完成与未完成

### 已完成

- 新增 `A-mid` 独立目录
- 新增 `A-mid` 独立配置
- 新增训练 / sweep 入口
- 支持 IDE 直接运行
- 保留 GPU 和进度条路径
- 保留 shell 一键脚本

### 未额外改动

- 没有改模型结构
- 没有改 decoder
- 没有改 backbone
- 没有改 inference 接口
- 没有改输出 JSON 格式
- 没有引入新依赖

## 14. 维护建议

后续如果继续调 `A-mid`，建议优先只动这些参数：

- `NEGATIVE_SAMPLE_RATIO`
- `NEGATIVE_SAMPLE_WEIGHT`
- `OLD_CLASS_SAMPLE_RATIO`
- `RARE_OVERSAMPLE`
- `CLASS_WEIGHTS`
- `DEFAULT_MAX_SAMPLES_PER_CLASS`

不建议优先去动：

- 模型结构
- backbone
- text encoder
- 解冻策略

因为这条线的目标本来就是“配置级中间版”，不是结构创新版。
