# YOLO 基线与数据增强实验总结

## 1. 目的与范围

这部分实验的定位不是最终提交主线，而是作为 ESAM / CLIPSeg 之外的对照基线，用来回答三类问题：

1. 红外伪标签是否足以支撑一个闭集学生模型训练。
2. 开放词汇检测加零样本分割这条路线，对少样本类和新类是否更友好。
3. 各类数据增强，尤其是 `computer`、`trash can`、`cable connector` 这类尾类增强，是否真的能带来收益。

本总结基于以下三类材料整理：

- 文档：`docs/数据增强/*.md`、`docs/学生模型/yolov11/*.md`、`docs/学生模型/yoloworld_fastsam/*.md`
- 代码：`src/train/yolo/*`、`src/train/world/*`、`src/tools/offline_*`
- 训练产物：`test/train_output/yolo/*`、`test/train_output/效果不理想/*`

## 2. 实验主线概览

整体上可以分成四组实验：

1. `YOLOv11-seg` 闭集分割基线
2. `YOLO-World + FastSAM` 开放词汇两阶段基线
3. `d5&7` 尾类专项实验
4. 离线数据增强实验：copy-paste、裁剪放大、极小目标 person 裁剪

其中用户之前提到的 “v23”，对应的是：

- `yoloworld_fastsam_v2`
- `yoloworld_fastsam_v3`

这两版是 YOLO-World 主线里最关键的一次“激进增强版 -> 稳定回退版”的演进。

## 3. 代码与实验目录对应关系

### 3.1 YOLOv11 闭集基线

- 代码：
  - `src/train/yolo/config_yolo.py`
  - `src/train/yolo/yolov11_seg.py`
- 文档：
  - `docs/学生模型/yolov11/yolo11n-seg.md`
  - `docs/学生模型/yolov11/yolo11s_seg_v3_summary.md`
- 输出：
  - `test/train_output/yolo/yolov11/yolo11s_seg_v3_stage1`

### 3.2 YOLO-World + FastSAM 主线

- v1:
  - 代码：`src/train/world/world_train.py`
  - 输出：`test/train_output/yolo/yoloworld_fastsam_v1`
- v2:
  - 代码：`src/train/world/world_train_v2.py`
  - 配置：`src/train/world/config_world_v2.py`
  - 输出：`test/train_output/yolo/yoloworld_fastsam_v2`
- v3:
  - 代码：`src/train/world/world_train_v3.py`
  - 配置：`src/train/world/config_world_v3.py`
  - 输出：`test/train_output/yolo/yoloworld_fastsam_v3`
- v4:
  - 代码：`src/train/world/world_train_v4.py`
  - 配置：`src/train/world/config_world_v4.py`
  - 输出：
    - `test/train_output/yolo/yoloworld_fastsam_v4_d5d7_3cls(分层抽+1.0)`
    - `test/train_output/yolo/yoloworld_fastsam_v4_d5d7_3cls(3+分层1.0)`
    - `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls`
    - `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls(0.5)`
    - `test/train_output/yolo/v4_d5d7_focal loss`
- v5:
  - 代码：`src/train/world/world_train_v5.py`
  - 配置：`src/train/world/config_world_v5.py`
  - 输出：`test/train_output/yolo/yoloworld_fastsam_v5_8cls_verify`

### 3.3 数据增强工具

- copy-paste：
  - 文档：`docs/数据增强/computer_trashcan_copy_paste.md`
  - 工具：`src/tools/offline_copy_paste_tail.py`
- 尾类裁剪增强：
  - 文档：`docs/数据增强/computer_trashcan_裁剪增强方案.md`
  - 工具：`src/tools/offline_crop_tail.py`
- person 极小目标裁剪增强：
  - 文档：`docs/数据增强/person_data1_极小目标裁剪增强方案.md`
  - 工具：`src/tools/offline_crop_person_data1.py`

## 4. YOLOv11 闭集基线

## 4.1 实验目的

YOLOv11 这条线的目标不是追最终榜单成绩，而是验证：

- SAM3 伪标签能否直接训练一个闭集学生模型；
- 红外图像上的伪标签是否具备可学习性；
- 作为与 ESAM / CLIPSeg 的对照，闭集检测分割模型能到什么水平。

## 4.2 主要设置

来自 `config_yolo.py` 和总结文档：

- 类别：`person, car, building, tree, animal, computer`
- 训练源：SAM3 伪标签 `pred_train_tasks.json`、`pred_val_tasks1.json`
- 图像尺寸：640
- 优化器：AdamW
- label smoothing：0.1
- dropout：0.1
- 数据增强：`copy_paste=0.3`，`scale=0.3`，`mosaic=0`
- 过采样：`animal=3x`，`computer=10x`
- 两阶段原计划：
  - stage1：150 epoch，lr=1e-3
  - stage2：50 epoch，lr=5e-5

实际本次运行主要完成了 stage1，并在约第 82 个 epoch 手动停止。

## 4.3 结果

文档记录的关键结果：

- 训练集约 `41158` 张，验证集约 `4104` 张
- 最佳 epoch 约为 `62`
- best Box mAP50：`0.65145`
- best Box mAP50-95：`0.52555`
- last Box mAP50：`0.64451`
- last Box mAP50-95：`0.51888`
- last Precision：`0.74172`
- last Recall：`0.62372`

## 4.4 结论

这条线说明：

- 闭集学生模型确实能从伪标签中学到稳定目标；
- 但它天然不具备开放词汇优势；
- 对新增类别、提示词变化和尾类扩展的支持不如 ESAM / YOLO-World 体系灵活。

所以它更像一个“伪标签可学性验证基线”，而不是最终开放词汇主线。

## 5. YOLO-World + FastSAM 主线演进

## 5.1 v1：最初的开放词汇两阶段基线

### 做了什么

`yoloworld_fastsam_v1` 是最纯净的两阶段开放词汇基线：

1. YOLO-World 做开放词汇检测
2. FastSAM 对检测框做零样本分割

设置相对简单：

- 6 类：`person, car, building, tree, animal, computer`
- 训练/验证都用 full train/val
- `animal=3x`，`computer=2x`
- 没有太多复杂技巧

### 结果

- best mAP50：`0.68782`
- best mAP50-95：`0.57601`
- last mAP50：`0.68555`
- last mAP50-95：`0.57202`
- Precision：`0.79102`
- Recall：`0.64401`

两阶段 mask mIoU：

- person：`0.5176`
- car：`0.5845`
- building：`0.5842`
- tree：`0.4316`
- animal：`0.4893`
- computer：`0.3218`
- overall：`0.5276`

### 结论

v1 证明了这条两阶段路线是可行的，但整体 mask 质量一般，尤其 old 类还没有跑出很强的上限。

## 5.2 v2：激进技巧叠加版

### 做了什么

`yoloworld_fastsam_v2` 是一次明显偏“激进优化”的版本。核心改动包括：

- 多尺度推理扩展到 `person/tree/animal`
- `computer` 阈值降到 `0.4`
- `computer` 过采样提高到 `1+4x`
- `cls=1.0`、`box=10.0`
- `label_smoothing=0.05`
- FastSAM 动态输入尺寸
- cross-scale NMS
- set_classes / CLIP cache 等工程优化

### 结果

- last mAP50：`0.67634`
- last mAP50-95：`0.56402`
- Precision：`0.75072`
- Recall：`0.63424`

两阶段 mask mIoU：

- person：`0.6187`
- car：`0.6813`
- building：`0.6008`
- tree：`0.4068`
- animal：`0.6179`
- computer：`0.3122`
- overall：`0.5844`

### 结论

v2 的整体 mIoU 相比 v1 提升很明显，说明多尺度和少样本强化是有效的。

但它的问题也很明确：

- 配置过于激进；
- `computer` 的过采样和阈值放松虽然增强了学习信号，但对语义稳定性不够友好；
- 文档和日志里同义词泛化表现并不理想，说明这条线可能在“检测指标上升”的同时，牺牲了部分开放词汇语义泛化。

## 5.3 v3：稳定回退版

### 做了什么

`yoloworld_fastsam_v3` 可以理解成对 v2 的“回调稳态版”：

- 只保留 `person`、`animal` 的多尺度
- 去掉 `tree` 多尺度
- `cls/box/label_smoothing` 回到默认
- `computer` 阈值从 `0.4` 回到 `0.5`
- `computer` 过采样从 `1+4x` 降到 `1+1x`
- 保留 FastSAM 动态尺寸、CLIP 缓存、多尺度 NMS

### 结果

- last mAP50：`0.6819`
- last mAP50-95：`0.57229`
- Precision：`0.77964`
- Recall：`0.63783`

两阶段 mask mIoU：

- person：`0.6172`
- car：`0.6810`
- building：`0.6121`
- tree：`0.4075`
- animal：`0.6228`
- computer：`0.2896`
- overall：`0.5864`

### 与 v2 的对比

- overall：`0.5844 -> 0.5864`
- building：`0.6008 -> 0.6121`
- animal：`0.6179 -> 0.6228`
- person：基本持平
- tree：基本持平
- computer：`0.3122 -> 0.2896`，略掉

### 结论

这说明 v3 的策略更像：

- 放弃对 `computer` 的过激强化；
- 换取整体类别间更均衡、更稳定的效果；
- 从主线角度看，v3 是比 v2 更成熟的 6 类 YOLO-World 基线。

换句话说，用户之前提到的 “v23”，本质上是：

- v2：激进技巧叠加版
- v3：回退到更稳、更可信的主线版本

## 6. d5&7 尾类专项实验

这部分是整套 YOLO 实验里最重要的“尾类验证区”。它不是为了做全量提交，而是专门回答：

- `trash can`、`cable connector` 这类类别到底学不学得动；
- 负样本比例到底该不该加；
- 数据增强是否真的有帮助。

主要类别是：

- `computer`
- `trash can`
- `cable connector`

其中 `computer` 是老稀有类，`trash can` 和 `cable connector` 是新增尾类。

## 6.1 纯正样本基线

目录可对应：

- `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls`
- `test/train_output/yolo/yoloworld_fastsam_v4_d5d7_3cls(分层抽+1.0)` 的早期 block 也复现了同一组结果

### 主要设置

- 只做 3 类专项训练
- 使用 d5&7 专项伪标签
- `computer` / `cable connector` 多尺度
- 不引入额外负样本压制

### 结果

- mAP50：`0.49354`
- mAP50-95：`0.40109`
- Precision：`0.60588`
- Recall：`0.48648`

mask mIoU：

- computer：`0.3612`
- trash can：`0.4432`
- cable connector：`0.2801`
- overall：`0.3823`

### 结论

这是尾类 3 类实验里最重要的参考基准。它说明：

- `trash can` 和 `cable connector` 并非完全学不会；
- 在当前伪标签质量下，先把正样本学好，比盲目引入大量负样本更重要。

## 6.2 随机负样本 1.5x

目录：

- `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls`
- 或 `.../yoloworld_fastsam_v4_d5d7_3cls(0.5)` 中对应 block

### 结果

- mAP50：`0.3013`
- mAP50-95：`0.26002`
- Precision：`0.40593`
- Recall：`0.35757`

mask mIoU：

- computer：`0.3318`
- trash can：`0.0985`
- cable connector：`0.2180`
- overall：`0.1562`

### 结论

这是非常典型的“负样本压过头”案例。

最明显的信号就是：

- `trash can` 从 `0.4432` 直接掉到 `0.0985`
- overall 从 `0.3823` 掉到 `0.1562`

说明当前尾类阶段，过强的随机负样本会让模型优先学会“不报”，而不是“报准”。

## 6.3 随机负样本 1.0x

目录：

- `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls`

### 结果

- mAP50：`0.3386`
- mAP50-95：`0.28251`
- Precision：`0.47499`
- Recall：`0.38769`

mask mIoU：

- computer：`0.2958`
- trash can：`0.4045`
- cable connector：`0.2203`
- overall：`0.3345`

### 结论

相比 1.5x，1.0x 已经明显回稳，说明：

- 少量负样本并非完全无效；
- 但它仍然没有超过纯正样本基线 `0.3823`；
- 因此在这批尾类数据上，负样本不是主增益来源。

## 6.4 负样本 0.5x

目录：

- `test/train_output/效果不理想/yoloworld_fastsam_v4_d5d7_3cls(0.5)`

### 结果

- mAP50：`0.33241`
- mAP50-95：`0.29481`
- Precision：`0.44205`
- Recall：`0.39865`

mask mIoU：

- computer：`0.3435`
- trash can：`0.3676`
- cable connector：`0.2378`
- overall：`0.3222`

### 结论

0.5x 比 1.5x 好很多，也比 stratified 版本稳定，但仍不如纯正样本 `0.3823`。

这说明结论比较一致：

- 尾类阶段负样本可以少量存在；
- 但不能成为训练主导；
- 当前最强信号仍然来自正样本本身。

## 6.5 分层抽样负样本 1.0x

目录：

- `test/train_output/yolo/yoloworld_fastsam_v4_d5d7_3cls(3+分层1.0)`

### 结果

- mAP50：`0.32911`
- mAP50-95：`0.27542`
- Precision：`0.38847`
- Recall：`0.42988`

mask mIoU：

- computer：`0.3140`
- trash can：`0.2960`
- cable connector：`0.1604`
- overall：`0.2518`

### 结论

分层抽样的直觉是“抽更有代表性的负样本”，但在这组实验里结果反而更差，说明：

- 这些“更有代表性”的负样本，对当前尾类模型来说可能过难；
- 它们并没有帮助模型学到更强的区分边界；
- 反而进一步压制了尾类召回。

## 6.6 focal loss + class weight 分支

目录：

- `test/train_output/yolo/v4_d5d7_focal loss`

### 主要设置

从历史日志看，这一支做了：

- stricter 阈值：
  - computer=`0.6`
  - trash can=`0.65`
  - cable connector=`0.6`
- focal loss：`gamma=1.5, alpha=0.25`
- class weights：
  - computer=`2.0`
  - trash can=`3.0`
  - cable connector=`1.0`

### 结果

- mAP50：`0.30234`
- mAP50-95：`0.26432`
- Precision：`0.38907`
- Recall：`0.39257`

mask mIoU：

- computer：`0.3629`
- trash can：`0.4156`
- cable connector：`0.2986`
- overall：`0.3886`

### 结论

这个数值看起来比纯正样本 `0.3823` 还高，但必须强调：

- 它的验证集是更严格、也更小的一版，正样本只有 `82` 张；
- 因此它和 `143` 张验证集的 v4 主表不能简单横向比较。

更合理的表述是：

- focal loss 分支在更干净、更严格的数据切片上表现出了潜力；
- 但由于评估集不同，不能据此断言它绝对优于纯正样本基线。

## 6.7 v5：把尾类并回主线的 8 类验证

目录：

- `test/train_output/yolo/yoloworld_fastsam_v5_8cls_verify`

### 做了什么

这条线尝试把：

- 原 6 类主线
- `trash can`
- `cable connector`

合成一个 8 类主线，观察尾类并回后会不会“带着一起涨”。

### 结果

- mAP50：`0.57485`
- mAP50-95：`0.47373`
- Precision：`0.67145`
- Recall：`0.55571`

mask mIoU：

- person：`0.6181`
- car：`0.6823`
- building：`0.6029`
- tree：`0.4077`
- animal：`0.6360`
- computer：`0.3307`
- trash can：`0.0803`
- cable connector：`0.2084`
- overall：`0.5782`

### 结论

这条线的意义非常明确：

- 老主线 6 类基本没有崩；
- 甚至 animal / computer 还不错；
- 但新增尾类在 fullset 主线里被严重稀释，尤其 `trash can` 只剩 `0.0803`。

这说明：

- “把尾类直接并回大主线”并不会自动成功；
- 主类分布太强时，弱类信号会被淹没；
- 这与后面 ESAM 里用户看到的“fullset 会压 rare”现象，在机制上是相通的。

## 7. 数据增强实验

## 7.1 copy-paste：computer / trash can 尾类复制粘贴

### 实现方式

文档与工具：

- `docs/数据增强/computer_trashcan_copy_paste.md`
- `src/tools/offline_copy_paste_tail.py`

核心思路：

- 从 `d5&7` 中挑高质量前景实例
- 从 full train 中挑不含目标类的背景图
- 离线粘贴生成增强图
- 同步生成 YOLO 标签、merged json/list、manifest

这套实现本身是完整的，工程上没问题。

### 实验结果

#### COPY_paste_v1

- 训练正样本：
  - computer=`231`
  - trash can=`504`
  - cable connector=`47`
- mAP50：`0.34545`
- mAP50-95：`0.313`

mask mIoU：

- computer：`0.3074`
- trash can：`0.2799`
- cable connector：`0.1183`
- overall：`0.2560`

#### COPY_paste_v2

- 训练正样本：
  - computer=`78`
  - trash can=`221`
  - cable connector=`47`
- mAP50：`0.30853`
- mAP50-95：`0.27373`

mask mIoU：

- computer：`0.3546`
- trash can：`0.3192`
- cable connector：`0.1042`
- overall：`0.2873`

### 结论

copy-paste 的结论比较清楚：

- 生成数量确实上去了；
- 但最终 mask 质量没有提升，反而明显差于纯正样本基线。

原因很可能包括：

- 红外图像的热分布对边缘和上下文很敏感；
- paste 后的局部统计特征与真实场景不一致；
- 即便做了 alpha blending 和模糊边缘，也无法完全消除“拼接感”。

因此这条线更适合写成：

- “已实现并完成实验验证”
- “结果不理想，未纳入最终主线”

## 7.2 尾类裁剪增强：比 copy-paste 更合理的替代思路

文档与工具：

- `docs/数据增强/computer_trashcan_裁剪增强方案.md`
- `src/tools/offline_crop_tail.py`

### 核心思路

不是把目标粘贴到别的背景，而是在原图内部做“裁剪-放大-重采样”：

- 保留原始热分布
- 保留局部上下文
- 让小目标在训练输入中占更大比例

这比 copy-paste 更符合红外图像特点。

### 当前状态

这条线的设计和工具是完整的，但在本次检查的 `test/train_output/yolo` 与 `test/train_output/效果不理想` 中，没有找到一个非常明确、可直接汇总成表格的完成训练结果目录。

因此论文里更适合写成：

- 已完成方案设计与脚本实现
- 动机明确：保留热分布与局部上下文
- 但当前材料中尚未形成一套与 copy-paste 同级别的完整 benchmark 输出

## 7.3 person 极小目标裁剪增强

文档与工具：

- `docs/数据增强/person_data1_极小目标裁剪增强方案.md`
- `src/tools/offline_crop_person_data1.py`

### 核心思路

只针对 `data1` 中极小的 `person` 做局部裁剪放大：

- 面积阈值很小
- crop 以 person bbox 为中心
- 适当保留附近上下文与相邻 person
- 输出统一 patch 尺寸

### 结论

这是一条很有针对性的“小目标增强”工具链，适合在论文里作为方法设计写进去。

但和尾类裁剪增强一样，在当前被检查的训练产物里，还没有找到一个清晰闭环的“最终结果目录”，因此更适合写成：

- “已实现的数据增强方案”
- “用于后续实验”
- “当前材料中暂无完整对比表结果”

## 8. “效果不理想”目录到底记录了什么

`test/train_output/效果不理想` 不是无意义的废目录，它其实记录了两类很关键的负结果：

1. `COPY_paste_v1 / COPY_paste_v2`
   - 证明 copy-paste 在红外尾类任务上没有带来预期收益
2. `yoloworld_fastsam_v4_d5d7_3cls*`
   - 证明负样本比例一旦加重，尾类很容易被压死

从研究角度，这部分非常有价值，因为它明确告诉我们：

- 不是所有“常见 CV 增强”都适合红外伪标签尾类任务；
- 增强/采样设计必须围绕热分布、上下文一致性和正负样本失衡来做；
- 负结果本身就是论文里非常重要的实验结论。

## 9. 关键结果总表

### 9.1 主线基线对比

| 实验 | 类别数 | 核心特点 | overall mIoU |
| --- | --- | --- | --- |
| YOLO-World + FastSAM v1 | 6 | 最初两阶段开放词汇基线 | `0.5276` |
| YOLO-World + FastSAM v2 | 6 | 激进技巧叠加、多尺度更强、computer 强化 | `0.5844` |
| YOLO-World + FastSAM v3 | 6 | 回退稳态版，主线最平衡 | `0.5864` |
| YOLO-World + FastSAM v5 | 8 | 把尾类并回 fullset 主线 | `0.5782` |

### 9.2 尾类专项对比

| 实验 | 主要设置 | computer | trash can | cable connector | overall |
| --- | --- | ---: | ---: | ---: | ---: |
| v4 纯正样本 | 3 类专项，不加负样本压制 | `0.3612` | `0.4432` | `0.2801` | `0.3823` |
| v4 随机负样本 1.0x | 适量随机负样本 | `0.2958` | `0.4045` | `0.2203` | `0.3345` |
| v4 负样本 0.5x | 更轻负样本 | `0.3435` | `0.3676` | `0.2378` | `0.3222` |
| v4 分层抽样 1.0x | family/bucket 分层负样本 | `0.3140` | `0.2960` | `0.1604` | `0.2518` |
| v4 随机负样本 1.5x | 负样本偏重 | `0.3318` | `0.0985` | `0.2180` | `0.1562` |
| v4 focal-loss 分支 | 更严格阈值 + focal + class weight | `0.3629` | `0.4156` | `0.2986` | `0.3886` |

说明：

- `focal-loss` 结果来自更小更严格的验证切片，不能和 `143` 张验证集主表直接一一对比。

### 9.3 数据增强对比

| 实验 | 类型 | overall |
| --- | --- | ---: |
| COPY_paste_v1 | copy-paste 尾类增强 | `0.2560` |
| COPY_paste_v2 | copy-paste 尾类增强 | `0.2873` |
| v4 纯正样本 | 无 copy-paste 的专项基线 | `0.3823` |

这张表已经足够支持一个明确结论：

- 本项目中，copy-paste 没有带来收益，反而显著低于不做 copy-paste 的专项基线。

## 10. 综合分析

## 10.1 v23 到底说明了什么

v2 和 v3 的演进是这组 YOLO 实验最值得写进论文的部分之一。

它说明：

- 激进的 trick 叠加可以快速把指标拉上去；
- 但开放词汇模型需要兼顾语义稳定性，不能只看检测侧数值；
- 回退到更稳的配置后，整体 mask mIoU 反而更高。

因此论文里可以把 v2 -> v3 写成：

- “从激进增强到稳态配置的模型演进”
- “说明开放词汇主线需要在召回提升与语义泛化之间做平衡”

## 10.2 尾类更怕什么

从 v4 全部实验看，尾类最怕的不是“样本少”本身，而是：

1. 过强负样本压制
2. 主类分布吞没
3. 破坏热分布的一般性增强

这三点分别对应：

- 随机负样本 1.5x 崩盘
- v5 merge-back 后 `trash can` 被稀释到 `0.0803`
- copy-paste 明显不如纯正样本基线

## 10.3 什么增强更符合红外特性

从现有结果推断，红外任务里更合理的增强策略不是 copy-paste，而是：

- 原图内裁剪放大
- 小目标局部重采样
- 保留真实热分布与局部上下文

这也解释了为什么：

- `computer_trashcan_裁剪增强方案`
- `person_data1_极小目标裁剪增强方案`

在设计上明显比 copy-paste 更谨慎。

## 10.4 YOLO 这块最终给 ESAM / CLIPSeg 带来了什么

虽然 YOLO-World / YOLOv11 并不是最终提交主线，但这批实验给后续 ESAM 路线提供了非常重要的经验：

1. old 类和 rare 类的优化方向应该拆开看
2. rare 类不适合被大主线直接吞并
3. 负样本与 hard negative 需要非常克制
4. 红外图像上数据增强要优先保护热分布一致性

这些经验和后来 ESAM 里用户观察到的：

- fullset 会压 rare
- stage1 split 更适合当 base
- old 保住时分数更稳

在实验逻辑上是相互印证的。

## 11. 论文写作建议

如果要把这一块写进论文，建议用下面的表述框架：

### 11.1 闭集与开放词汇基线

- 先给出 YOLOv11 闭集分割基线
- 再给出 YOLO-World + FastSAM 的开放词汇两阶段基线
- 说明开放词汇路线在整体 mIoU 上更有潜力

### 11.2 版本演进

- 用 v1 / v2 / v3 展示模型从“基础版 -> 激进增强版 -> 稳定回退版”的演进
- 强调 v3 是更稳定、更可复现的主线

### 11.3 尾类与数据增强

- 用 v4 / v5 说明尾类学习难点
- 用 copy-paste 负结果说明并非所有增强都有效
- 用 crop-based 方案说明你们后续更符合红外先验的改进方向

### 11.4 负结果价值

建议不要回避 `效果不理想` 里的实验，反而应把它们写成：

- 对增强策略和负样本机制的系统性排错
- 为最终 ESAM 主线提供设计依据的对比实验

## 12. 最终结论

基于当前材料，可以得出下面几条较稳的结论：

1. YOLO-World + FastSAM 明显优于最初的基础开放词汇基线，v3 是 6 类主线上最稳的一版。
2. v2/v3 说明激进技巧叠加未必最优，稳态配置对开放词汇模型更重要。
3. 尾类专项实验表明 rare 类可以学到，但极易被过强负样本和主类分布压制。
4. copy-paste 在本项目的红外尾类场景下效果不理想，不适合进入主线。
5. 裁剪放大型增强更符合红外数据特点，是更合理的后续方向。
6. 这批 YOLO 与数据增强实验虽然不是最终提交方案，但为后续 ESAM 主线提供了明确的经验边界和设计依据。
