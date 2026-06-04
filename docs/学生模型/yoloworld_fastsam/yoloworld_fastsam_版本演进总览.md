# YOLO-World + FastSAM 版本演进总览

## 一、路线总览

这条路线的核心思路一直没变：

1. 用 `SAM3` 伪标签提供监督
2. 用 `YOLO-World` 学开放词汇检测
3. 用 `FastSAM` 在推理阶段补分割
4. 最终用两阶段 `mask mIoU` 判断路线是否有效

但 `v1 → v5` 的目标并不相同：

- `v1`：证明两阶段路线能跑通
- `v2`：尝试叠加技巧，拉高小目标和稀有类效果
- `v3`：回收无效技巧，稳定 6 类主线
- `v4`：切到 `d5&7`，专项验证 3 个尾类
- `v5`：把尾类并回 8 类主线，验证数据链路和并回效果

一句话理解：

- `v3` 是 6 类稳定基线
- `v4` 是 3 类尾类实验线
- `v5` 是 8 类并回验证线

---

## 二、版本关系表

| 版本 | 主要目标 | 类别 | 数据来源 | 结论定位 |
|---|---|---|---|---|
| `v1` | 路线打通 | 6 类 | 全训练集 `train_tasks / val_tasks1` | 两阶段方案可运行 |
| `v2` | 叠加改进技巧 | 6 类 | 全训练集 `train_tasks / val_tasks1` | 部分技巧有效，部分有副作用 |
| `v3` | 收敛稳定基线 | 6 类 | 全训练集 `train_tasks / val_tasks1` | 当前 6 类主线基线 |
| `v4` | 尾类专项验证 | 3 类 `computer / trash can / cable connector` | `d5&7` 专项伪标签 | 验证尾类是否可学 |
| `v5` | 并回主线验证 | 8 类 = 6 类主线 + 2 类尾类 | 主线伪标签 + `d5&7` 尾类伪标签 | 验证尾类并回主线后的效果 |

---

## 三、各版本定位

## 3.1 v1：路线验证版

`v1` 的意义主要是证明以下事情成立：

- `YOLO-World` 可以在你这批红外图上完成开放词汇检测微调
- `FastSAM` 可以接 `YOLO-World` 的框做零样本分割
- `SAM3` 伪标签能够被转成检测训练所需的 `bbox`

这一版更像原型验证，不强调细节最优。

## 3.2 v2：技巧扩展版

`v2` 在 `v1` 的基础上加入了多种增强型策略，例如：

- `person / tree / animal` 多尺度推理
- 增大 `cls / box / label_smoothing`
- `computer` 降阈值 + 高倍率过采样
- FastSAM 动态 `imgsz`
- CLIP 离线缓存与训练曲线监控

`v2` 的价值不在于最终成为主线，而在于帮你们筛选出：

- 什么是有效技巧
- 什么会把 CLIP 语义泛化拉坏

## 3.3 v3：6 类稳定基线

`v3` 是对 `v2` 结果做“减法”后的稳定版本。

它保留了：

- `person / animal` 多尺度
- `animal` 稀有类过采样
- FastSAM 动态 `imgsz`
- CLIP 离线缓存

它回退了：

- `cls / box / label_smoothing` 手动增强
- `computer` 的激进阈值与高倍率过采样
- `tree` 多尺度

所以 `v3` 的角色很清楚：

- 不是继续追新技巧
- 而是提供一版稳定、可复现、能代表 6 类主线能力的基线

## 3.4 v4：d5&7 三类尾类实验线

`v4` 不再关注 6 类主线，而是把问题收缩到 3 个尾类：

- `computer`
- `trash can`
- `cable connector`

数据也切到 `d5&7`：

- 训练：`pred_train_d5&7_tasks.json`
- 验证：`pred_val_d5&7_tasks.json`

`v4` 不是最终提交方案，它的用途是：

1. 验证这三类是否真的可学
2. 验证负样本策略是否有帮助
3. 验证阈值和过采样对尾类是否有价值

目前从已有总结看：

- 这三类是可学的
- 纯正样本版最好
- 负样本策略不是当前主提升项
- `trash can` 和 `cable connector` 对数据清洗更敏感

## 3.5 v5：8 类并回主线验证版

`v5` 的目标不是刷新最佳成绩，而是验证：

- 6 类主线伪标签和 2 类尾类伪标签能否正确合并
- 尾类并回主线后，训练链路和评估链路是否正常
- 尾类会不会在主线里被稀释掉

`v5` 的类别组成：

- 主线 6 类：`person / car / building / tree / animal / computer`
- 新增 2 类：`trash can / cable connector`

数据来源是混合的：

- 原 6 类来自全训练集 `train_tasks / val_tasks1`
- 新 2 类来自 `d5&7` 的专项伪标签

这也是 `v5` 和 `v4` 最大的区别：

- `v4` 是单独做尾类
- `v5` 是把尾类塞回主线，观察主线稀释效应

---

## 四、数据来源演进

## 4.1 主线 6 类

`v1 / v2 / v3` 的数据源基本一致：

- 训练：`test/prompt_test_output/train_tasks/pred_train_tasks.json`
- 验证：`test/prompt_test_output/val_tasks1/pred_val_tasks1.json`

这条线代表“全局 6 类主线”。

## 4.2 d5&7 尾类线

`v4` 独立引入：

- 训练：`test/prompt_test_output/train_d5&7_tasks/pred_train_d5&7_tasks.json`
- 验证：`test/prompt_test_output/val_d5&7_tasks/pred_val_d5&7_tasks.json`

这条线代表“尾类专项实验线”。

## 4.3 混合并回线

`v5` 则采用双源合并：

- `base_full_train / base_full_val`
- `tail_d5d7_train / tail_d5d7_val`

它是一个典型的“标签链路验证版”，不是纯粹的同源训练版。

---

## 五、关键策略演进

## 5.1 多尺度策略

| 版本 | 多尺度类别 | 结论 |
|---|---|---|
| `v1` | 无或较少 | 基线 |
| `v2` | `person / tree / animal` | `person / animal` 有效，`tree` 负收益 |
| `v3` | `person / animal` | 收敛后的稳定设置 |
| `v4` | `computer / cable connector` | 面向更小、更稀有尾类 |
| `v5` | 仍保持 `person / animal` | 主线验证版，暂不把尾类也塞进多尺度 |

这反映出你们后面已经不再把多尺度当通用招，而是开始按类使用。

## 5.2 过采样策略

| 版本 | 重点类 | 结论 |
|---|---|---|
| `v2` | `animal / computer` | `computer 5x` 不稳定 |
| `v3` | `animal 3x`、`computer 2x` | 回到保守值 |
| `v4` | `computer / cable connector / trash can` | 尾类按类单独调倍率 |
| `v5` | `animal / computer / cable connector` | 仍偏主线配置，尾类容易被稀释 |

## 5.3 阈值策略

版本越往后，阈值策略越体现出一个趋势：

- 前期按主线 6 类统一处理
- 后期开始按尾类单独设阈值
- 再往后已经演变成“伪标签质量比训练技巧更关键”

---

## 六、当前阶段的实际结论

到现在为止，这条路线的结论已经比较清楚。

## 6.1 主线结论

`YOLO-World + FastSAM` 作为 6 类开放词汇分割验证路线是成立的。  
`v3` 可以作为当前这条线的稳定基线。

## 6.2 尾类结论

`v4` 已经证明：

- `computer / trash can / cable connector` 不是完全不可学
- 尾类更依赖伪标签清洗，而不是更重的训练技巧
- `trash can` 与 `cable connector` 对样本干净度更敏感

## 6.3 并回主线结论

`v5` 暴露的问题不是“尾类完全没信号”，而是：

- 尾类在 8 类主线里被严重稀释
- 主线与尾类的数据源不一致
- 主线检测指标和尾类分割质量不一定同步提升

所以 `v5` 更像一条“验证并回可行性”的工程线，而不是当前最值得继续深挖的性能线。

---

## 七、如何使用这些版本

如果你的目标是不同的，应该参考不同版本。

### 想看 6 类主线基线

看：

- `world_train_v3.py`
- `config_world_v3.py`
- `yoloworld_fastsam_v3.md`

### 想看 3 类尾类专项实验

看：

- `world_train_v4.py`
- `config_world_v4.py`
- `yoloworld_fastsam_v4_d5d7_3cls_summary.md`

### 想看尾类并回主线的实现方式

看：

- `world_train_v5.py`
- `config_world_v5.py`

---

## 八、当前建议

如果后面还继续沿这条线推进，优先级建议是：

1. 以 `v3` 作为 6 类主线参考实现
2. 以 `v4` 作为尾类实验主线
3. 把 `v5` 当成并回验证工具，而不是当前主优化对象
4. 后续更多精力放在：
   - 伪标签质量
   - 尾类筛样本
   - 类别是否值得继续保留

---

## 九、一句话总结

`YOLO-World + FastSAM` 这条路线已经从 `v1` 的“能不能跑”演进到 `v3` 的“6 类稳定基线”、`v4` 的“尾类专项验证”和 `v5` 的“尾类并回主线验证”；当前最清晰的主线结构是：**`v3` 管 6 类基线，`v4` 管尾类实验，`v5` 管并回验证。**
