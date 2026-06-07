# SAM3 官方代码整理与伪标签生成方案

本文针对目录 `D:\nyz\sam3\选手示例工程\to_Df - 官方\sam3` 做两件事：

1. 说明哪些模块适合直接复用改写
2. 说明哪些更适合作为知识参考

同时给出本项目里新的 SAM3 伪标签生成脚本落点和使用建议。

---

## 1. 目录结论

对当前项目最有价值的，不是把整套官方代码都搬进来，而是分层使用。

### 1.1 可直接复用改写

这部分适合直接作为你生成伪标签的核心链路。

| 文件 | 价值 | 当前建议 |
|---|---|---|
| `to_Df - 官方/inference.py` | 官方单图文本分割入口 | **直接作为伪标签脚本骨架** |
| `sam3/model_builder.py` | 官方模型构建入口 | **直接复用** |
| `sam3/model/sam3_image_processor.py` | 官方图像推理处理器 | **直接复用** |
| `sam3/eval/coco_writer.py` | 官方 COCO/RLE 导出思路 | 结果格式参考 |

这四个文件已经足够支撑“图片 + prompt -> mask/RLE”的 teacher 伪标签链路。

### 1.2 适合知识参考

这部分不是当前伪标签脚本的直接依赖，但对后续训练、格式理解和更深改造有价值。

| 文件/目录 | 价值 | 当前建议 |
|---|---|---|
| `sam3/train/data/sam3_image_dataset.py` | 官方图像任务数据组织方式 | 参考 query/image/segment 结构 |
| `sam3/train/data/coco_json_loaders.py` | 官方 COCO + 文本 query 读取方式 | 参考 annotation/query 设计 |
| `sam3/model/*` | 模型结构细节 | 后续蒸馏、特征对齐时再深入 |
| `sam3/train/loss/*` | 官方 loss 设计 | 未来若做 SAM3 微调再看 |
| `sam3/eval/*` | 评测/导出/后处理 | 需要 COCO 风格评测时参考 |

### 1.3 当前阶段价值较低

| 文件/目录 | 原因 |
|---|---|
| `sam3/agent/*` | 面向 agentic segmentation，多轮交互，不是你当前 teacher 批量打伪标签需要的 |
| `sam3/perflib/*` | 更偏算子/性能优化，不是当前瓶颈 |
| `sam3/model/sam3_video_*`、`tracker_*` | 当前任务是单图，不是视频跟踪 |
| `sam3/train/configs/*` | 训练配置多而重，当前目标不是官方训练重跑 |

---

## 2. 当前最适合的改写方式

不建议继续把 `prompt_test` 当成“唯一入口”，因为它已经混入了本地项目自己的实验逻辑。

更稳的方式是：

1. 保留官方 `inference.py` 的核心推理逻辑
2. 在外面只包一层：
   - 任务 JSON 读取
   - 你已经验证过有效的红外预处理
   - 按类阈值过滤
   - 训练所需的 `pred_*.json` 输出
   - `kept_images.txt / skipped_*.json` 这样的数据清单

这样做的好处是：

- teacher 始终是官方 SAM3，不会和学生模型混线
- 预处理保留你前期已经验证过的收益
- 输出格式仍兼容你现有训练链路

---

## 3. 是否继续沿用前期预处理

结论：**有必要。**

依据来自 `docs/前期准备/训练前工作流程总结.md` 和 `docs/前期准备/质量分析报告.md`：

1. 数据源异质性很高  
   分辨率、亮度、对比度、极性、噪声水平差异都明显，不预处理时 teacher 输入分布本身不稳定。

2. 伪彩、黑热/白热混杂是明确存在的问题  
   尤其 `_pair`、`blackHot`、部分 `data5/data7` 图像，若不统一极性，很容易把 prompt 响应搞乱。

3. 低对比、低 SNR 图对 mask 完整性影响明显  
   CLAHE、轻量降噪、适度锐化在前期实验里已经证明有正向作用。

因此，新脚本建议继续沿用：

- 伪彩转灰
- 黑热/白热统一
- 动态 CLAHE
- 动态降噪
- 动态锐化

但不建议在伪标签阶段做新的强增强或激进 resize。

---

## 4. 是否要卡掉一部分低质量伪标签

结论：**有必要，但不建议一刀切硬删。**

更稳的做法是两层控制：

### 4.1 输入图层面

对明显坏图做可配置跳过：

- 来自 `clean_outliers_list.txt` 的彩色异常图、文字/UI 叠加图
- 缺失文件
- 推理阶段直接异常的图片

这里不要直接在代码里“永久删除数据”，而是：

- 生成 `kept_images.txt`
- 生成 `skipped_*.json`

后续训练决定是否只用 `kept_images.txt`。

### 4.2 伪标签层面

对每个 prompt 使用输出阈值过滤：

- 保留 raw 统计，便于分析 teacher 激活分布
- 真正写入训练伪标签时，用更保守的按类阈值

这样能同时兼顾：

- 分析时不丢信息
- 训练时减少噪声标签

---

## 5. 新脚本落点

已新增脚本：

- [generate_sam3_pseudo_labels.py](D:/nyz/raytron_project/src/prompt/generate_sam3_pseudo_labels.py)

实现方式：

1. **模型与处理器**  
   直接复用官方：
   - `sam3.model_builder.build_sam3_image_model`
   - `sam3.model.sam3_image_processor.Sam3Processor`

2. **mask 聚合**  
   直接改写自官方 `inference.py` 的 `aggregate_prompt_prediction`

3. **预处理**  
   沿用你之前 `prompt_test` 里验证过的红外预处理逻辑

4. **输出格式**  
   对齐已有 `prompt_test_output/pred_*.json`

5. **质量控制**  
   支持：
   - `--skip-list`
   - `--processor-conf-threshold`
   - `--default-threshold`
   - `--output-thresholds-json`

---

## 6. 使用建议

推荐先按下面顺序跑：

1. 在验证集上先生成一版新的 SAM3 伪标签
2. 对比旧版 `pred_val_tasks1.json`
3. 观察：
   - 每类激活率
   - 哪些图被 skip
   - 低对比图是否更完整
   - `person/tree/computer/trash can` 这些敏感类是否更稳

建议命令形态：

```powershell
D:\anconda3\envs\rayton\python.exe src\prompt\generate_sam3_pseudo_labels.py `
  --tasks test\json\val_tasks1.json `
  --image-root . `
  --output-root test\prompt_test_output `
  --official-root "D:\nyz\sam3\选手示例工程\to_Df - 官方" `
  --checkpoint "D:\nyz\sam3\选手示例工程\to_Df - 官方\model\sam3.pt" `
  --require-cuda
```

如果你暂时不想跳过异常图，可以显式去掉 skip list：

```powershell
... --skip-list ""
```

---

## 7. 当前判断

对你现在这条主线，最稳的 teacher 伪标签策略是：

1. **teacher 固定为官方 SAM3**
2. **保留前期验证过有效的红外预处理**
3. **坏图过滤做成可配置，不在代码里硬删原始数据**
4. **按类阈值输出训练伪标签**
5. **配套保留 kept/skipped 清单，方便后续训练切分**

这比继续混用 `prompt_test` 和当前根目录 `inference.py` 更清楚，也更不容易把 teacher/student 角色搞乱。
