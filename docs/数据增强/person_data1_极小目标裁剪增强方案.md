# data1 person 极小目标裁剪增强方案

## 1. 目标

这条增强线只服务一个目的：

- 提升 `data1` 中 `person` 的极小目标召回

同时明确避免两件事：

- 不破坏其他数据源中正常尺度 `person` 的分布
- 不改验证集，只回灌训练集

---

## 2. 适用数据

当前脚本：

- [offline_crop_person_data1.py](/C:/raytron_project/src/tools/offline_crop_person_data1.py)

默认输入：

- `test/prompt_test_output/train(person)/pred_train(person).json`
- `test/train_list_d1.txt`

默认输出：

- `test/crop_person_data1/`
- `test/prompt_test_output/train(person)/pred_train(person)_with_crop.json`
- `test/train_list_d1_with_crop.txt`

---

## 3. 小目标门槛

不是对所有 `data1/person` 都做裁剪增强，只增强足够小的一批实例。

默认门槛：

- `bbox_area_ratio < 0.001`
- `max(bbox_w, bbox_h) < 20`

这里的 `bbox_area_ratio` 指：

- `bbox_area / image_area`

这样做的原因：

- `data1` 的 `person` 多为俯视小人，通常“窄而高”
- 只用 `8x8` 或只看单边尺寸都不稳
- 面积占比 + 最大边限制更适合这类目标

---

## 4. 裁剪策略

这条线不是“紧框裁人”，而是“以人框为中心的上下文裁剪”。

默认规则：

- 以 `person bbox` 为中心
- 做正方形裁剪
- 外扩范围：`2.5x ~ 3.5x`
- 轻微中心扰动：`0.06`
- 最小原始裁剪边长：`48`
- 输出固定尺寸 patch：`160 x 160`

目标：

- 让小人被放大
- 但周围仍保留路面、背景、邻近结构

这样学到的是“小人出现在什么环境里”，而不是单独的人体亮点纹理。

---

## 5. 保留规则

主目标需要同时满足：

- `score >= 0.70`
- `keep_ratio >= 0.80`
- 裁剪后主目标像素数不低于最小阈值
- 裁剪后主目标占 crop 面积比例不低于 `0.015`

另外：

- patch 中其他 `person` 连通域允许保留
- 但会按 `context_keep_ratio = 0.60` 过滤裁残碎片

这意味着：

- 可以保留邻近小人，维持真实密集场景
- 不会把只剩一点边角的碎片人强行写回标签

---

## 6. 输出内容

脚本会生成：

- `images/`：裁剪后的训练图
- `labels/`：YOLO bbox 标签，类别固定为 `person`
- `pred_crop_only.json`：只包含新裁剪样本
- `pred_crop_merged.json`：原训练伪标签 + 裁剪样本
- `manifest.csv`：每张样本的来源、裁剪框、目标占比等
- `source_usage.csv`：每个源实例被复用次数
- `preview/`：原图裁剪框 + patch + mask 叠加预览

---

## 7. 使用建议

推荐顺序：

1. 先用默认门槛跑 `50 ~ 100` 张小样本
2. 先看 `preview/`
3. 确认放大效果和上下文合理后，再放大到 `400` 张左右
4. 训练时使用 merged 版本，而不是只用 crop-only

即：

- 用 `pred_train(person)_with_crop.json`
- 配 `train_list_d1_with_crop.txt`

不要只用裁剪图训练，否则容易把输入分布过度推向局部 patch。

---

## 8. 一句话总结

这条线的核心不是“做 person 裁剪增强”，而是：

- **只对 `data1`**
- **只对足够小的 `person`**
- **只做保留上下文的局部放大**

它本质上是一个 `person@data1` 的极小目标专项增强方案。
