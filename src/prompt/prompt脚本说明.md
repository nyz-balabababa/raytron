# prompt 脚本说明

`src/prompt` 目录下这些脚本分工不同，建议按“任务生成 -> teacher 伪标签生成 -> 坏图名单维护 -> 可视化抽查”的顺序理解，不要混用。

当前脚本包括：

- `gen_val_tasks.py`
- `generate_sam3_labels.py`
- `build_unified_denylist.py`
- `vis_val_browser.py`
- `prompt_test(9).py`

---

## 1. gen_val_tasks.py

### 作用

用于从一份图片列表生成标准任务 JSON。

它回答的问题是：

- 怎样把 `train_list.txt` / `val_list.txt` 转成统一任务格式
- 怎样按 prompt 展开成 `每张图 × 每个类别 = 一条任务`
- 怎样在生成任务时直接跳过 denylist 里的坏图

### 典型输入

- 图片列表 txt
- prompt 列表
- 可选 skip list

默认配置里常见的是：

- `test/val_list.txt`
- 五类 prompt：`person / car / building / tree / animal`
- `noisy_data/unified_denylist.txt`

### 典型输出

- `test/json/*.json`
- 对应的 `*_meta.json`

输出任务格式为：

```json
[
  {"ann_id": 1, "image_path": "...", "text_prompt": "..."}
]
```

### 适用场景

- 为 teacher 伪标签生成准备任务文件
- 为提交推理或验证推理准备任务文件
- 想把图片列表和坏图过滤规则打通

### 不适合的场景

它不负责推理，也不负责伪标签分析。

如果你已经有任务 JSON，后续要生成 teacher 伪标签，应使用 `generate_sam3_labels.py`。

---

## 2. generate_sam3_labels.py

### 作用

用于基于官方 `SAM3` 推理链路生成 `prompt_test_output` 风格伪标签。

设计目标是：

- 尽量复用官方 `build_sam3_image_model + Sam3Processor`
- 保留红外图像预处理
- 输出兼容现有训练流程的 `pred_*.json / stats_*.csv / kept_*.txt`

### 典型输入

- `tasks json`
- 官方 `SAM3` 代码根目录
- `sam3.pt`
- 可选坏图 skip list

### 典型输出

默认输出到：

- `test/prompt_test_output/<tasks_name>/pred_*.json`
- `stats_*.csv`
- `kept_*.txt`
- `skipped_*.json`
- `meta_*.json`

### 当前脚本特点

- 支持按类阈值覆盖
- 支持坏图 skip list
- 支持红外预处理开关
- 统计时保留 `raw_stats_threshold`
- 输出格式对齐现有 teacher 伪标签格式

### 适用场景

- 生成当前主版本 teacher 伪标签
- 比较不同预处理/阈值设置下的 teacher 输出
- 为 `refine_pseudo_labels.py` 和训练脚本提供输入

### 注意事项

它是当前更正式的 teacher 伪标签生成工具。  
如果你已经转到官方 `SAM3` 链路，优先使用它，不建议继续把 `prompt_test(9).py` 当主线生成器。

---

## 3. build_unified_denylist.py

### 作用

用于合成统一坏图名单。

它会把多个来源的坏图信息合并成一份最终 denylist，供任务生成和 teacher 伪标签生成阶段共同使用。

### 典型输入来源

1. `test/clean_outliers.csv`
   - 只取历史清洗规则里的 `is_color` / `is_text_overlay`
2. `noisy_data/`
   - 视作人工维护的坏图目录
3. `analyze_prompt_predictions.py` 输出 JSON
   - 读取 `all_prompts_wrong_images[].image_path`

### 典型输出

- `noisy_data/unified_denylist.txt`
- `noisy_data/unified_denylist_meta.json`

### 适用场景

- 把人工坏图、颜色异常图、文本覆盖图统一管理
- 为 `gen_val_tasks.py` 和 `generate_sam3_labels.py` 提供 skip list
- 避免每个脚本各自维护一套坏图逻辑

### 不适合的场景

它不负责判断伪标签质量，也不做 mask 分析。

如果你是要对 teacher 质量做 A/B/C 分档，应使用 `src/label/refine_pseudo_labels.py`。

---

## 4. vis_val_browser.py

### 作用

这是一个交互式伪标签浏览器，用于人工抽查：

- 当前伪标签
- 旧伪标签
- 新伪标签
- 分歧样本列表

### 当前支持的查看内容

- 原图
- mask 叠加
- bbox 叠加
- 按文件夹筛选
- 按类别筛选
- 按命中/高分/低分/漏检筛选
- 分歧样本 TopK 浏览

### 典型输入

- `TASK_JSON`
- `PRED_JSON`
- `OLD_PRED_JSON`
- `NEW_PRED_JSON`
- `DIFF_SAMPLES_TXT`

并且它兼容两种预测格式：

- prompt-style 伪标签 JSON
- 提交格式 `predictions.json`

### 适用场景

- 抽查 `pseudo_A/B/C`
- 抽查旧 teacher vs 新 teacher 的差异
- 检查 mask 转 bbox 是否合理
- 浏览 `compare_pseudo_labels.py` 输出的分歧样本

### 注意事项

这是人工浏览和诊断工具，不是批处理分析脚本。  
如果你想系统统计新旧 teacher 的差异，应优先用 `src/label/compare_pseudo_labels.py`，浏览器用于后续人工确认。

---

## 5. prompt_test(9).py

### 作用

这是较早的 prompt 测试与评估脚本，特点是：

- 自动发现 `test/` 下任务 JSON
- 推理前执行自适应预处理
- 输出 pred JSON、统计 CSV、可视化结果和预处理日志

### 当前脚本特点

- 带设备检查，支持 `REQUIRE_CUDA`
- 带任务白名单 `TASK_JSON_WHITELIST`
- 按类输出阈值
- 会做预处理缓存和可视化

### 适用场景

- 早期 prompt 实验
- 快速试单类或小范围任务
- 调试预处理是否有效

### 当前定位

它更像历史实验脚本，不建议作为当前主线 teacher 伪标签生成入口。  
当前更推荐使用 `generate_sam3_labels.py`。

---

## 这几个脚本怎么配合

推荐顺序：

1. `build_unified_denylist.py`
   - 先统一坏图名单

2. `gen_val_tasks.py`
   - 再从图片列表生成任务 JSON

3. `generate_sam3_labels.py`
   - 基于任务 JSON 生成 teacher 伪标签

4. `vis_val_browser.py`
   - 最后人工抽查结果和分歧样本

如果只是做旧实验复现或小范围 prompt 调试，可以额外使用：

5. `prompt_test(9).py`
   - 作为历史实验脚本参考

---

## 一句话区分

- `gen_val_tasks.py`：**图片列表 -> 任务 JSON**
- `generate_sam3_labels.py`：**任务 JSON -> teacher 伪标签**
- `build_unified_denylist.py`：**多来源坏图 -> 统一 denylist**
- `vis_val_browser.py`：**人工浏览当前/新旧伪标签**
- `prompt_test(9).py`：**旧版 prompt 实验与调试脚本**

---

## 当前主线建议

如果当前主线是五类：

- `person`
- `car`
- `building`
- `tree`
- `animal`

建议：

1. 先用 `build_unified_denylist.py` 合成统一坏图名单
2. 用 `gen_val_tasks.py` 从 `train_list.txt / val_list.txt` 生成任务
3. 用 `generate_sam3_labels.py` 生成当前主版本 teacher 伪标签
4. 再把结果交给 `src/label/refine_pseudo_labels.py` 做 `A/B/C` 分档
5. 用 `vis_val_browser.py` 做抽查，而不是靠肉眼直接扫 JSON

`prompt_test(9).py` 建议只保留为历史实验参考，不作为当前主线入口。
