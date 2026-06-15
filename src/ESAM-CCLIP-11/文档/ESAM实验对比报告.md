# ESAM 实验对比报告

本文基于 `D:\nyz\raytron_project\test\train_output` 下当前已有的 ESAM 训练产物、日志、`results.csv` 和已存在的 sweep 结果整理。  
目的不是复述所有细节，而是把几条主线的改动、参数、结果和当前结论压成一份可回看的实验记录。

## 1. 结论先看

当前最强的本地同口径 sweep 结果仍然是 `lite` 主线：

- `threshold_sweep_esam_11_hybrid_lite`: `best_metric = 0.578131`

目前几条关键线的判断：

- `ESAM-CCLIP-11-fastfinetune`：11 类全集热启动起点，有价值，但当时直接提交分数低，说明“基础权重还行，提交链路不成熟”。
- `ESAM-CCLIP-11-old-balanced`：在 `fastfinetune` 基础上继续全集训练，属于第一条成功主线的核心拐点。
- `ESAM-CCLIP-11-lite-fullset-polish`：在 `old-balanced` 基础上做更稳的全集轻抛光，是当前最接近最终提交逻辑的一条线。
- `ESAM-CCLIP-11-split-control`：第二条线的起点，分集热启动。
- `ESAM-CCLIP-11-split-bridge-stage1`：路线二第一阶段有效，`all11/rare` 都有稳步提升，是 split 线最有希望的基础权重。
- `ESAM-CCLIP-11-split-bridge-satge2`：路线二第二阶段能略提 old，但 rare 回落明显。
- `ESAM-CCLIP-11-stage1-rare-rescue`：在 `stage1` 上继续做 rare rescue，这次没有达到预期，整体略退。
- `ESAM-CCLIP-11-stage1-mild-rare-rescue`：比激进版 rare rescue 稍稳，但仍然没有超过原始 `stage1`。
- `ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep`：目前 split 体系下最好看的一次后续修复，`all11/old5/rare` 同时抬升。
- `ESAM-CCLIP-11-text-realign-1ep-bs8`：在 `stage2_mild_rare_rescue` 后做 1 轮 text realign，当前 val 结果继续小幅正向，主要收益在 rare。
- `ESAM-unfreeze-stage1-lastnorm-1ep`：训练验证集指标看起来有提升，但后续 hybrid sweep 没吃出来，不适合直接判成主线成功。
- `ESAM-CCLIP-11-stage1-unfreeze-recalibrate-1ep-bs8`：在 `stage1 unfreeze` 后再做 decoder-only 校准，验证集有修复，但仍未超过 `lite` 主线。

## 2. 训练线总览

### 2.1 路线一：全集热启动 -> old-balanced -> lite-fullset-polish

#### A. `ESAM-CCLIP-11-fastfinetune`

定位：

- 3 轮 11 类全集热启动
- 作为后续 `old-balanced` / `rare-balanced` 的共同起点

核心参数：

- `epochs=3`
- `no_val=True`
- `no_train_split_filter=True`
- `decoder_lr=1e-4`
- `negative_sample_ratio=0.05`
- `resume_best=6类 best.pt`
- `train_decoder_only=True`

现有可记录结果：

- `results.csv` 只有训练损失，无 val mIoU
- 最后一轮训练损失：
  - `train_loss=0.2148`
  - `train_bce=0.0312`
  - `train_dice=0.3046`
  - `train_focal=0.0102`

分析：

- 这条线是“11 类全集热启动基底”。
- 你后面回忆里提到它当时直接提交只有约 `48` 分，这和后续 `66+` 并不矛盾。
- 更合理的解释是：这条权重本身提供了 11 类基础能力，但当时的推理脚本、阈值、后处理、类别 alias、提交链路还远没打磨好。

#### B. `ESAM-CCLIP-11-old-balanced`

定位：

- 在 `fastfinetune/final_fullset.pt` 上继续做全集训练
- 目标是保住 old 类，同时温和拉起 rare

核心参数：

- `epochs=4`
- `no_val=True`
- `no_train_split_filter=True`
- `decoder_lr=1e-4`
- `negative_sample_ratio=0.02`
- `old_class_sample_ratio=0.65`
- `rare_class_keep_ratio=1.0`
- 稀有类重采样：
  - `trash can=3`
  - `window=2`
  - `door=2`
  - `fence=3`
  - `pole_light=3`
  - `motorcycle=2`
- 稀有类权重：
  - `trash can=1.8`
  - `window=1.5`
  - `door=1.5`
  - `fence=1.7`
  - `pole_light=1.9`
  - `motorcycle=1.7`

现有可记录结果：

- 最后一轮训练损失：
  - `train_loss=0.2201`
  - `train_bce=0.0275`
  - `train_dice=0.3075`
  - `train_focal=0.0093`

后续 sweep 结果：

- `threshold_sweep_lite_recall`: `best_metric=0.570989`
- `threshold_sweep_esam_11_hybrid_lite`: `best_metric=0.578131`

`hybrid_lite` 关键统计：

- `old5_avg=0.655563`
- `rare6_avg=0.513604`

分析：

- 这是路线一里真正“把分拉上来”的关键节点。
- `old-balanced` 不是最终最稳的提交版本，但它是后面 `lite-fullset-polish` 的核心基础。

#### C. `ESAM-CCLIP-11-rare-balanced`

定位：

- 同样在 `fastfinetune/final_fullset.pt` 上继续全集训练
- 但比 `old-balanced` 更 aggressively 地压 old、推 rare

核心参数：

- `epochs=4`
- `no_val=True`
- `no_train_split_filter=True`
- `decoder_lr=1e-4`
- `negative_sample_ratio=0.05`
- `old_class_sample_ratio=0.45`
- `rare_class_keep_ratio=1.0`

现有可记录结果：

- 最后一轮训练损失：
  - `train_loss=0.2864`
  - `train_bce=0.0259`
  - `train_dice=0.3375`
  - `train_focal=0.0088`

分析：

- 从配置上就能看出，这条线明显比 `old-balanced` 更偏 rare。
- 你后续整体判断也是 `old-balanced` 最终分更高，这和参数方向是一致的。
- 当前材料里没有一份明确的 rare-balanced sweep summary 被保留下来，因此这里不对它补不存在的最终本地分。

#### D. `ESAM-CCLIP-11-lite-fullset-polish`

定位：

- 在 `old-balanced/final_fullset.pt` 基础上继续做全集轻抛光
- 属于路线一的后处理前最终训练步骤

核心参数：

- `preset=lite_fullset_polish`
- `epochs=3`
- `no_val=True`
- `no_train_split_filter=True`
- `decoder_lr=5e-5`
- `negative_sample_ratio=0.01`
- `negative_sample_weight=0.10`
- `old_class_sample_ratio=1.0`
- `rare_balance_enabled=False`
- 稀有类重采样：
  - `trash can=2`
  - `fence=2`
  - `pole_light=2`
  - `motorcycle=2`
- 稀有类权重：
  - `trash can=1.6`
  - `window=1.3`
  - `door=1.3`
  - `fence=1.4`
  - `pole_light=1.6`
  - `motorcycle=1.5`

现有可记录结果：

- 最后一轮训练损失：
  - `train_loss=0.1596`
  - `train_bce=0.0291`
  - `train_dice=0.2563`
  - `train_focal=0.0097`

分析：

- 这条线和 `old-balanced` 不是横向替代关系，而是纵向接续关系。
- `old-balanced` 更像“先把方向拉出来”，`lite-fullset-polish` 更像“在这个方向上降低学习率、减轻负样本和重平衡，再做一次稳定抛光”。
- 当前最强的本地 sweep 结果仍然是围绕这条主线得到的，因此它是现阶段最强提交基线。

---

### 2.2 路线二：split-control -> stage1 -> stage2

#### A. `ESAM-CCLIP-11-split-control`

定位：

- 第二条线起点
- 分集热启动

核心参数：

- `epochs=5`
- `no_val=False`
- `no_train_split_filter=False`
- `decoder_lr=1e-4`
- `negative_sample_ratio=0.02`
- `resume_best=6类 best.pt`

最后一轮验证结果：

- `val_miou_all11=0.584616`
- `val_miou_old5=0.663018`
- `val_miou_rare=0.405665`

关键类别：

- `trash can=0.4308`
- `window=0.2928`
- `door=0.4810`
- `fence=0.5038`
- `pole_light=0.4145`
- `motorcycle=0.5360`

后续 safe sweep：

- `threshold_sweep_esam_11_mid_safe`: `best_metric=0.536587`

分析：

- 这条线说明 split 热启动本身能成型，但本地 sweep 指标不如后面的 stage1。
- 它更像路线二的起步权重，而不是最终竞争力最强的提交权重。

#### B. `ESAM-CCLIP-11-split-bridge-stage1`

定位：

- 路线二第一阶段
- 从 `split-control/best_all11.pt` 继续 split train/val 续训
- 目的是得到更干净的 split bridge 权重

核心参数：

- `preset=split_bridge_stage1`
- `epochs=3`
- `no_val=False`
- `no_train_split_filter=False`
- `decoder_lr=5e-5`
- `negative_sample_ratio=0.02`
- `negative_sample_weight=0.15`
- `old_class_sample_ratio=1.0`
- `rare_balance_enabled=False`
- 稀有类重采样：
  - `trash can=2`
  - `fence=2`
  - `pole_light=2`
  - `motorcycle=2`

逐轮验证结果：

- Epoch1:
  - `all11=0.583716`
  - `old5=0.664482`
  - `rare=0.399374`
- Epoch2:
  - `all11=0.586203`
  - `old5=0.664797`
  - `rare=0.406816`
- Epoch3:
  - `all11=0.586968`
  - `old5=0.663296`
  - `rare=0.412754`

分析：

- 这是第二条线里最值得保留的阶段。
- 它的特点不是“绝对总分最高”，而是：
  - `all11` 稳步涨
  - `rare` 稳步涨
  - `old5` 基本稳住
- 所以后面所有围绕 split 的继续优化，最合理的出发点都是它的 `best_all11.pt`。

#### C. `ESAM-CCLIP-11-split-bridge-satge2`

说明：

- 实际目录名里有个拼写错误：`satge2`
- 但它对应的 run name 是 `ESAM-CCLIP-11-split-bridge-fullset`

定位：

- 路线二第二阶段
- 从 `stage1/best_all11.pt` 继续做 fullset 收尾

核心参数：

- `preset=split_bridge_stage2_fullset`
- `epochs=3`
- `no_val=True`
- `no_train_split_filter=True`
- `decoder_lr=3e-5`
- `negative_sample_ratio=0.01`
- `negative_sample_weight=0.10`
- `old_class_sample_ratio=1.0`
- `rare_balance_enabled=False`

现有可记录训练结果：

- 最后一轮训练损失：
  - `train_loss=0.1627`
  - `train_bce=0.0298`
  - `train_dice=0.2599`
  - `train_focal=0.0099`

后续 sweep：

- `threshold_sweep_esam_11_hybrid_stage2`: `best_metric=0.562322`

`hybrid_stage2` 关键统计：

- `old5_avg=0.656215`
- `rare6_avg=0.484078`

分析：

- 和 `lite_hybrid` 比，`stage2` 的 old 只有很小幅度波动式提升，rare 则出现明显回落。
- 具体概括就是：
  - old 并没有形成决定性优势
  - rare 几乎全线落后于 `lite`
- 因此 `stage2` 更像“把 stage1 学到的 rare 往 fullset 主流分布重新拉回去”。

---

### 2.3 路线二后续尝试

#### A. `ESAM-CCLIP-11-stage1-rare-rescue`

定位：

- 在 `stage1/best_all11.pt` 上继续做温和 rare rescue

核心参数：

- `preset=stage1_rare_rescue`
- 实际本次只跑了 `epochs=1`
- `no_val=False`
- `no_train_split_filter=False`
- `decoder_lr=2e-5`
- `negative_sample_ratio=0.02`
- `negative_sample_weight=0.15`
- `old_class_sample_ratio=0.85`
- `rare_balance_enabled=True`
- 稀有类重采样：
  - `trash can/window/door/fence/pole_light/motorcycle = 2`
- 稀有类权重：
  - `trash can=1.7`
  - `window=1.55`
  - `door=1.45`
  - `fence=1.65`
  - `pole_light=1.75`
  - `motorcycle=1.6`

本次结果：

- `all11=0.585691`
- `old5=0.662306`
- `rare=0.410819`

对比原始 `stage1`：

- `all11`: `0.586968 -> 0.585691`
- `old5`: `0.663296 -> 0.662306`
- `rare`: `0.412754 -> 0.410819`

分析：

- 没有炸，但也没有达到“rare rescue”的目标。
- 更准确地说，它把 `stage1` 原本较好的平衡轻微扰乱了。
- 这说明 `stage1` 已经接近一个比较好的平衡点，继续往 rare 方向轻推未必有效。

#### B. `ESAM-CCLIP-11-stage1-mild-rare-rescue`

定位：

- 这是在 `stage1/best_all11.pt` 基础上做的一次温和版 rare rescue
- 相比 `stage1_rare_rescue`，它更保守：
  - old 类下采样更轻
  - 负样本更少
  - rare oversample 不再特别猛

核心参数：

- `preset=stage1_rare_rescue`
- `resume_best=ESAM-CCLIP-11-split-bridge-stage1/best_all11.pt`
- `epochs=1`
- `no_val=False`
- `no_train_split_filter=False`
- `decoder_lr=2e-5`
- `negative_sample_ratio=0.005`
- `negative_sample_weight=0.05`
- `old_class_sample_ratio=0.95`
- `rare_balance_enabled=True`
- 稀有类重采样：
  - `trash can/window/door/fence/pole_light/motorcycle = 2`
- 稀有类权重：
  - `trash can=1.7`
  - `window=1.55`
  - `door=1.45`
  - `fence=1.65`
  - `pole_light=1.75`
  - `motorcycle=1.6`

本次结果：

- `all11=0.585825`
- `old5=0.662475`
- `rare=0.410875`

对比 `stage1`：

- `all11`: `0.586968 -> 0.585825`，`-0.001143`
- `old5`: `0.663296 -> 0.662475`，`-0.000821`
- `rare`: `0.412754 -> 0.410875`，`-0.001879`

对比 `stage1_rare_rescue`：

- `all11`: `0.585691 -> 0.585825`，`+0.000134`
- `old5`: `0.662306 -> 0.662475`，`+0.000169`
- `rare`: `0.410819 -> 0.410875`，`+0.000056`

分析：

- 这条比之前更激进的 `stage1_rare_rescue` 略稳一点，但本质结论没有变。
- 也就是说：
  - 在 `stage1` 基础上直接做 rare rescue，不管是激进版还是 mild 版，这次都没有超过原始 `stage1`
  - 区别只在于 mild 版“退得更少”

#### C. `ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep`

定位：

- 这是在 `stage2/final_fullset.pt` 基础上继续做的一次温和 rare rescue
- 实际上是“stage2 -> 回到 split train/val -> mild rare repair”

核心参数：

- `preset=rare_repair_lite`
- `resume_best=ESAM-CCLIP-11-split-bridge-stage2/final_fullset.pt`
- `epochs=1`
- `no_val=False`
- `no_train_split_filter=False`
- `decoder_lr=1e-5`
- `negative_sample_ratio=0.005`
- `negative_sample_weight=0.05`
- `old_class_sample_ratio=0.95`
- `rare_balance_enabled=True`
- 稀有类重采样：
  - `trash can=3`
  - `window=2`
  - `door=2`
  - `fence=3`
  - `pole_light=3`
  - `motorcycle=2`
- 稀有类权重：
  - `trash can=1.6`
  - `window=1.3`
  - `door=1.3`
  - `fence=1.4`
  - `pole_light=1.6`
  - `motorcycle=1.5`

本次结果：

- `all11=0.592521`
- `old5=0.665135`
- `rare=0.426782`

关键类别：

- `trash can=0.4799`
- `window=0.3090`
- `door=0.5011`
- `fence=0.5347`
- `pole_light=0.4338`
- `motorcycle=0.5578`

对比 `stage1`：

- `all11`: `0.586968 -> 0.592521`，`+0.005552`
- `old5`: `0.663296 -> 0.665135`，`+0.001839`
- `rare`: `0.412754 -> 0.426782`，`+0.014028`

对比 `stage1_rare_rescue`：

- `all11`: `0.585691 -> 0.592521`，`+0.006830`
- `old5`: `0.662306 -> 0.665135`，`+0.002828`
- `rare`: `0.410819 -> 0.426782`，`+0.015963`

分析：

- 这是目前 split 口径下非常亮眼的一次修复。
- 和 `stage1` 相比，它不是只抬 rare、牺牲 old，而是：
  - `all11` 明显涨
  - `old5` 也涨
  - `rare` 也有比较明显的提升
- 这说明：
  - `stage2` 本身并不是“完全没救”
  - 更准确地说，`stage2` 在 fullset 收尾后把 rare 压下去了，但如果再切回 split 并做温和 rare repair，仍然能把一部分 rare 和整体指标拉回来
- 当前最值得做的下一步不是否定它，而是：
  - 先对这条权重做 sweep
  - 再和 `stage1`、`lite` 做同口径比较

#### D. `ESAM-unfreeze-stage1-lastnorm-1ep`

定位：

- 路线三
- 在 `stage1` 基础上尝试 partial image encoder unfreeze
- 这次是 `last_norm` 档位，1 epoch

现有可记录结果：

- 训练目录里保留了 `best_all11.pt`
- `results.csv` 只有 1 轮
- 验证结果：
  - `all11=0.588655`
  - `old5=0.663594`
  - `rare=0.417608`

对比 `stage1`：

- `all11`: `+0.00169`
- `old5`: `+0.00030`
- `rare`: `+0.00485`

后续 sweep：

- `threshold_sweep_stage1_unfreeze_hybrid`: `best_metric=0.548433`
- `old5_avg=0.653185`
- `rare6_avg=0.461141`

分析：

- 训练集验证指标看起来是有提升的。
- 但后续 `hybrid` sweep 没把这个收益转出来。
- 更合理的解释不是“训练一定失败”，而是：
  - 解冻后 logit 分布变了
  - 当前 sweep 网格和这条线可能不完全匹配
- 但就现有材料看，它还不适合作为当前主线继续押注。

#### E. `ESAM-CCLIP-11-stage1-unfreeze-recalibrate-1ep-bs8`

定位：

- 路线三-A
- 在 `ESAM-unfreeze-stage1-lastnorm-1ep/best_all11.pt` 基础上
- 重新冻结 image encoder，只训练 decoder 1 轮

核心参数：

- `preset=unfreeze_recalibrate`
- `resume_best=ESAM-unfreeze-stage1-lastnorm-1ep/best_all11.pt`
- `epochs=1`
- `batch_size=8`
- `decoder_lr=5e-6`
- `image_lr=0.0`
- `text_lr=0.0`
- `negative_sample_ratio=0.005`
- `negative_sample_weight=0.05`
- `rare_balance_enabled=False`
- `old_class_sample_ratio=1.0`
- `rare_class_keep_ratio=1.0`

本次结果：

- `all11=0.590115`
- `old5=0.664354`
- `rare=0.420668`

对比 `ESAM-unfreeze-stage1-lastnorm-1ep`：

- `all11`: `0.588655 -> 0.590115`，`+0.001460`
- `old5`: `0.663594 -> 0.664354`，`+0.000760`
- `rare`: `0.417608 -> 0.420668`，`+0.003060`

分析：

- 这说明“解冻后再回冻、只训 decoder 做 recalibrate”这个思路是成立的。
- 它确实把 `stage1 unfreeze` 的收益进一步兑现了一点。
- 但这条线目前仍属于 split 体系的追赶线，还没有超过 `lite` 主线。

#### F. `ESAM-CCLIP-11-text-realign-1ep-bs8`

定位：

- 路线二后续的文本泛化分支
- 在 `ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep/best_all11.pt` 基础上
- 开 prompt alias / text realign
- 冻结 image encoder 和 text encoder
- 只训练 decoder 1 轮

核心参数：

- `preset=text_realign_1ep`
- `resume_best=ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep/best_all11.pt`
- `epochs=1`
- `batch_size=8`
- `decoder_lr=5e-6`
- `negative_sample_ratio=0.005`
- `negative_sample_weight=0.05`
- `augment_prompt=True`
- `prompt_alias_train=True`
- `prompt_alias_prob=1.0`
- `val_augment_prompt=False`

本次结果：

- `all11=0.593284`
- `old5=0.664603`
- `rare=0.430502`

对比 `stage2_mild_rare_rescue_1ep`：

- `all11`: `0.592521 -> 0.593284`，`+0.000763`
- `old5`: `0.665135 -> 0.664603`，`-0.000532`
- `rare`: `0.426782 -> 0.430502`，`+0.003719`

关键类别变化：

- `trash can`: `0.4799 -> 0.4968`
- `window`: `0.3090 -> 0.3191`
- `door`: `0.5011 -> 0.5025`
- `motorcycle`: `0.5578 -> 0.5593`

分析：

- 这条线是当前“文本泛化/alias realign”方向的正向验证。
- old5 有极轻微回落，但 rare 的提升更明确，尤其 `trash can` 和 `window`。
- 从 split val 口径看，它比原始 `stage2_mild_rare_rescue_1ep` 更均衡一点。
- 后续真正判断它有没有提交价值，仍然要看对应 sweep。

## 3. Sweep 结果总对比

当前几个最关键、可以同口径直接比较的 sweep 结果如下：

| 结果 | `best_metric` | 备注 |
|---|---:|---|
| `threshold_sweep_esam_11_hybrid_lite` | `0.578131` | 当前最强 lite 主线 |
| `threshold_sweep_lite_recall` | `0.570989` | 旧 lite recall 版本 |
| `threshold_sweep_esam_11_hybrid_stage2` | `0.562322` | 路线二 stage2 |
| `threshold_sweep_stage1_unfreeze_hybrid` | `0.548433` | 路线三 unfreeze |
| `threshold_sweep_esam_11_mid_safe` | `0.536587` | split-control / mid safe |

### 3.1 哪些结果距离 lite 主线在 0.03 以内

以当前最高的 `lite_hybrid = 0.578131` 为基准：

- `stage2_hybrid = 0.562322`
  - 差值 `0.015809`
- `stage1_unfreeze_hybrid = 0.548433`
  - 差值 `0.029698`

在 `0.03` 以内的只有这两条。

### 3.2 stage2 和 lite 的结构差异

`stage2_hybrid`：

- `old5_avg=0.656215`
- `rare6_avg=0.484078`

`lite_hybrid`：

- `old5_avg=0.655563`
- `rare6_avg=0.513604`

结论：

- `stage2` 的 old 只有极小幅度提升
- `rare` 比 `lite` 低了约 `0.0295`
- 也就是说：**stage2 相比 lite，是 old 小幅波动式提升，但 rare 明显下降**

### 3.3 stage1 / rare-rescue / unfreeze 的对比

训练验证口径下：

| 线 | `all11` | `old5` | `rare` |
|---|---:|---:|---:|
| `stage1` | `0.586968` | `0.663296` | `0.412754` |
| `stage1_rare_rescue` | `0.585691` | `0.662306` | `0.410819` |
| `stage1_mild_rare_rescue` | `0.585825` | `0.662475` | `0.410875` |
| `stage2_mild_rare_rescue_1ep` | `0.592521` | `0.665135` | `0.426782` |
| `stage2_text_realign_1ep` | `0.593284` | `0.664603` | `0.430502` |
| `unfreeze_stage1_lastnorm_1ep` | `0.588655` | `0.663594` | `0.417608` |
| `stage1_unfreeze_recalibrate_1ep` | `0.590115` | `0.664354` | `0.420668` |

结论：

- `rare_rescue` 这次没有救起来，反而整体轻微退步
- `stage1_mild_rare_rescue` 比激进版 `stage1_rare_rescue` 略稳，但仍然没超过原始 `stage1`
- `stage2_mild_rare_rescue_1ep` 是目前 split 口径下最好看的一次后续修复结果
- `stage2_text_realign_1ep` 在 `stage2_mild` 基础上继续小幅抬高了 `all11` 和 `rare`
- `unfreeze` 在验证集上看起来有提升
- `stage1_unfreeze_recalibrate_1ep` 说明“解冻后回冻校准”方向是正向的
- 但 `unfreeze` 体系整体的后续 sweep 还没有吃出足够收益，因此还不宜当主线

### 3.4 四条 split 后续修复线并排看

这里单独把最容易混的四条线放一起：

| 线 | 基础权重 | 方向 | `all11` | `old5` | `rare` | 结论 |
|---|---|---|---:|---:|---:|---|
| `stage1` | `split-control` | split bridge 主线 | `0.586968` | `0.663296` | `0.412754` | 路线二最好的基础权重 |
| `stage1_rare_rescue` | `stage1` | 较激进 rare rescue | `0.585691` | `0.662306` | `0.410819` | 不如原始 stage1 |
| `stage1_mild_rare_rescue` | `stage1` | 温和 rare rescue | `0.585825` | `0.662475` | `0.410875` | 比激进版略稳，但仍不如原始 stage1 |
| `stage2_mild_rare_rescue_1ep` | `stage2` | stage2 后回 split 温和修复 | `0.592521` | `0.665135` | `0.426782` | 当前 split 口径下最亮眼的后续修复 |
| `stage2_text_realign_1ep` | `stage2_mild_rare_rescue_1ep` | 文本泛化 / alias realign | `0.593284` | `0.664603` | `0.430502` | 在 stage2_mild 上继续小幅抬 all11 和 rare |

这一组最重要的结论：

- `stage1` 本身是好基础，这个判断没变。
- 但“直接在 stage1 上做 rare rescue”这条思路，这两版都没有证实有效。
- 反而是“先有 stage2 fullset，再切回 split 做 mild rare rescue”这次最成功。
- 在 `stage2_mild` 基础上继续做 `text_realign_1ep`，当前 split val 口径继续是正向的。
- 所以现在更值得跟进的是：
  - `stage2_mild_rare_rescue_1ep` 的 sweep 表现
  - `stage2_text_realign_1ep` 的 sweep 表现
  - 而不是继续在 `stage1_rare_rescue` 上加码

## 4. 当前阶段的整体判断

### 4.1 已经被验证有效的链路

- `fastfinetune -> old-balanced -> lite-fullset-polish -> lite/hybrid sweep`

这是目前最成熟、最稳定的一条主线。

### 4.2 split 线里最有希望的基础

- `ESAM-CCLIP-11-split-bridge-stage1`

原因：

- 比 `split-control` 更稳
- rare 在涨
- old 基本守住
- 是后续 split 细修最合理的起点

### 4.3 当前不建议继续强推的方向

- 继续加强 `stage1_rare_rescue`
- 原样继续 `stage2` 再跑多轮 fullset

原因：

- rare rescue 这次已经证明收益不稳定
- stage2 当前方向会把 rare 往回压

### 4.4 如果继续优化 stage1，更合理的方向

从现有结果看，更合理的不是继续“救 rare”，而是：

- 围绕 `stage1` 做更保 old 的轻抛光
- 或给 `stage1` / `unfreeze` 单独配更贴合的 sweep 网格

更简化地说：

- `stage1` 是好基础
- 但后续优化方向应该偏 **保 old + 轻抛光**
- 而不是继续 **压 old 去推 rare**

## 5. 现阶段建议排序

按“已验证的可提交竞争力”排：

1. 路线一 `lite` 主线
   - 代表 sweep：`threshold_sweep_esam_11_hybrid_lite`
2. 路线二 `stage2`
   - 代表 sweep：`threshold_sweep_esam_11_hybrid_stage2`
3. 路线三 `stage1 unfreeze`
   - 训练指标可看，sweep 暂时不够强
4. `stage1_rare_rescue`
   - 当前版本不建议继续强化

如果只看 split train/val 指标的后续优化潜力，当前可以改成：

1. `ESAM-CCLIP-11-stage2-mild-rare-rescue-1ep`
2. `ESAM-CCLIP-11-split-bridge-stage1`
3. `ESAM-unfreeze-stage1-lastnorm-1ep`
4. `ESAM-CCLIP-11-stage1-mild-rare-rescue`
5. `ESAM-CCLIP-11-stage1-rare-rescue`

按“作为继续实验的基础权重”排：

1. `ESAM-CCLIP-11-split-bridge-stage1`
2. `ESAM-CCLIP-11-old-balanced`
3. `ESAM-CCLIP-11-lite-fullset-polish`

这里的排序含义不同：

- `lite` 是当前最强提交基线
- `stage1` 是当前最值得继续做 split 细修的基础

## 6. 备注

- `ESAM-CCLIP-11-split-bridge-satge2` 目录名有拼写错误，实际对应 `stage2 fullset`。
- `ESAM-unfreeze-stage1-lastnorm-1ep` 的 `best_all11.pt` 比其他权重大一些，和打开 image encoder 参与训练有关。
- 这份报告只使用当前项目里能直接读到的训练目录和 sweep summary；不存在的 sweep 结果没有补写。
