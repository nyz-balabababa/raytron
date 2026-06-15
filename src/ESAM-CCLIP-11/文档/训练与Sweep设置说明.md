# ESAM-CCLIP-11 训练与 Sweep 设置说明

本文档只说明当前 `src/ESAM-CCLIP-11` 代码里的实际 `train` 和 `sweep` 设置，不展开讲历史路线来源。需要看路线选择时，再结合：

- [当前可尝试路线总结.md](/d:/nyz/raytron_project/src/ESAM-CCLIP-11/文档/当前可尝试路线总结.md)
- [运行说明文档.md](/d:/nyz/raytron_project/src/ESAM-CCLIP-11/文档/运行说明文档.md)
- [优化路线实现说明.md](/d:/nyz/raytron_project/src/ESAM-CCLIP-11/文档/优化路线实现说明.md)

## 1. 当前训练默认底座

当前根线训练入口是：

- `src/ESAM-CCLIP-11/train_esam_cclip_11.py`

当前配置底座来自：

- `src/ESAM-CCLIP-11/config_esam_cclip_11.py`

默认训练基线是：

- `RUN_NAME = ESAM-CCLIP-11-split-control`
- `IMG_SIZE = 768`
- `BATCH_SIZE = 4`
- `EPOCHS = 5`
- `DECODER_LR = 1e-4`
- `IMAGE_LR = 0.0`
- `WEIGHT_DECAY = 0.03`
- `WARMUP_EPOCHS = 1`
- `MIN_LR_RATIO = 0.01`
- `NEGATIVE_SAMPLE_RATIO = 0.02`
- `NEGATIVE_SAMPLE_WEIGHT = 0.20`
- `OLD_CLASS_SAMPLE_RATIO = 1.0`
- `RARE_CLASS_KEEP_RATIO = 1.0`

默认模型训练策略是：

- 冻结 `image encoder`
- 冻结 `text encoder`
- 只训练 `decoder`
- 默认不开 `image cache`
- 默认使用 `prompt prototype + text cache`

默认后处理基线 `POSTPROCESS_DEFAULT`：

- `person/car`: `min_area=16`
- `building`: `min_area=128`, `fill_holes=True`
- `tree`: `min_area=64`, `fill_holes=True`
- `animal`: `min_area=8`
- `trash can/window`: `min_area=4`
- `door`: `min_area=8`
- `fence`: `min_area=2`
- `pole_light`: `min_area=1`
- `motorcycle`: `min_area=4`

## 2. 当前训练 preset

当前支持的 preset 有：

- `base`
- `lite_fullset_polish`
- `split_bridge_stage1`
- `split_bridge_stage2_fullset`
- `text_realign_1ep`
- `unfreeze_recalibrate`
- `stage1_rare_rescue`
- `rare_repair_lite`
- `partial_unfreeze_lastnorm`
- `partial_unfreeze_last1`

### `base`

含义：

- 不额外覆盖配置，直接用 `config_esam_cclip_11.py` 默认值。

适合：

- 调试
- 手工传全套参数
- 不想用预设时

### `lite_fullset_polish`

当前设置：

- `run_name = ESAM-CCLIP-11-lite-fullset-polish`
- `no_val = True`
- `no_train_split_filter = True`
- `epochs = 3`
- `decoder_lr = 5e-5`
- `warmup_epochs = 1`
- `negative_sample_ratio = 0.01`
- `negative_sample_weight = 0.10`
- `rare_balance_enabled = False`
- `old_class_sample_ratio = 1.0`
- `rare_class_keep_ratio = 1.0`
- `text_cache_path = test/cache/text_emb_11_lite_polish.pt`
- `rare_oversample` 沿用基础配置

含义：

- 这是当前最稳的 `lite + fullset` 主线。
- 不走验证集选 best，最终主要看 `final_fullset.pt`。

### `split_bridge_stage1`

当前设置：

- `run_name = ESAM-CCLIP-11-split-bridge-stage1`
- `train_json = test/clean_rare/label/train_label.json`
- `val_json = test/clean_rare/label/val_label.json`
- `train_list = test/train_list.txt`
- `val_list = test/val_list.txt`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 3`
- `decoder_lr = 5e-5`
- `warmup_epochs = 1`
- `negative_sample_ratio = 0.02`
- `negative_sample_weight = 0.15`
- `rare_balance_enabled = False`
- `text_cache_path = test/cache/text_emb_11_split_bridge.pt`

含义：

- 路线二 Stage1。
- 先用分集/分级数据做干净续训。
- 这条线建议显式传 `--resume_best` 或 `--resume`。

### `split_bridge_stage2_fullset`

当前设置：

- `run_name = ESAM-CCLIP-11-split-bridge-fullset`
- `train_json = test/clean_rare/label/trainval_label.json`
- `no_val = True`
- `no_train_split_filter = True`
- `epochs = 3`
- `decoder_lr = 3e-5`
- `warmup_epochs = 1`
- `negative_sample_ratio = 0.01`
- `negative_sample_weight = 0.10`
- `rare_balance_enabled = False`
- `text_cache_path = test/cache/text_emb_11_split_bridge.pt`

含义：

- 路线二 Stage2。
- 从 Stage1 权重接 fullset。
- 最终主要看 `final_fullset.pt`。

### `text_realign_1ep`

当前设置：

- `run_name = ESAM-CCLIP-11-text-realign-1ep`
- `train_json = test/clean_rare/label/train_label.json`
- `val_json = test/clean_rare/label/val_label.json`
- `train_list = test/train_list.txt`
- `val_list = test/val_list.txt`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 1`
- `decoder_lr = 5e-6`
- `image_lr = 0.0`
- `text_lr = 0.0`
- `warmup_epochs = 0`
- `min_lr_ratio = 0.5`
- `negative_sample_ratio = 0.005`
- `negative_sample_weight = 0.05`
- `rare_balance_enabled = False`
- `old_class_sample_ratio = 1.0`
- `rare_class_keep_ratio = 1.0`
- `augment_prompt = True`
- `prompt_alias_train = True`
- `prompt_alias_prob = 1.0`
- `val_augment_prompt = False`
- `text_cache_path = test/cache/text_emb_11_text_realign_alias.pt`
- `freeze_image_encoder = True`
- `freeze_text_encoder = True`
- `train_decoder_only = True`

含义：

- 这是在强底座权重上做 1 轮 decoder-only 文本对齐校准的 preset。
- 训练时会随机使用同类 prompt alias。
- val 仍然用 canonical class / prototype 做稳定评估。
- 典型用途是：
  - `stage2_mild_rare_rescue_1ep/best_all11.pt`
  - `-> text_realign_1ep`
  - `-> 再 sweep`

### `unfreeze_recalibrate`

当前设置：

- `run_name = ESAM-CCLIP-11-unfreeze-recalibrate`
- `train_json = test/clean_rare/label/train_label.json`
- `val_json = test/clean_rare/label/val_label.json`
- `train_list = test/train_list.txt`
- `val_list = test/val_list.txt`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 1`
- `decoder_lr = 1e-5`
- `image_lr = 0.0`
- `text_lr = 0.0`
- `warmup_epochs = 0`
- `min_lr_ratio = 0.5`
- `negative_sample_ratio = 0.005`
- `negative_sample_weight = 0.05`
- `rare_balance_enabled = False`
- `old_class_sample_ratio = 1.0`
- `rare_class_keep_ratio = 1.0`
- `text_cache_path = test/cache/text_emb_11_unfreeze_recalibrate.pt`
- `freeze_image_encoder = True`
- `freeze_text_encoder = True`
- `train_decoder_only = True`
- `unfreeze_image_mode = none`

含义：

- 这是路线三的“解冻后再冻结，只训 decoder 做 1 轮校准”的 preset。
- 典型用途是：
  - `stage1_unfreeze_lastnorm_1ep/best_all11.pt`
  - `-> unfreeze_recalibrate`
  - `-> 看解冻收益能不能通过 decoder 校准兑现`

### `stage1_rare_rescue`

当前设置：

- `run_name = ESAM-CCLIP-11-stage1-rare-rescue`
- `train_json = test/clean_rare/label/train_label.json`
- `val_json = test/clean_rare/label/val_label.json`
- `train_list = test/train_list.txt`
- `val_list = test/val_list.txt`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 2`
- `decoder_lr = 2e-5`
- `image_lr = 0.0`
- `text_lr = 0.0`
- `warmup_epochs = 1`
- `min_lr_ratio = 0.3`
- `negative_sample_ratio = 0.02`
- `negative_sample_weight = 0.15`
- `old_class_sample_ratio = 0.85`
- `rare_class_keep_ratio = 1.0`
- `rare_balance_enabled = True`
- `text_cache_path = test/cache/text_emb_11_stage1_rare_rescue.pt`
- `rare_oversample`：
  - `trash can/window/door/fence/pole_light/motorcycle = 2`
- `class_weights` 温和抬 rare：
  - `trash can=1.7`
  - `window=1.55`
  - `door=1.45`
  - `fence=1.65`
  - `pole_light=1.75`
  - `motorcycle=1.6`

含义：

- 这是在 `stage1` 基础上做 split 口径的温和 rare rescue。
- 会额外保存 `best_rescue.pt`。

### `rare_repair_lite`

当前设置：

- `run_name = ESAM-CCLIP-11-rare-repair-lite`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 3`
- `decoder_lr = 5e-5`
- `warmup_epochs = 1`
- `negative_sample_ratio = 0.01`
- `negative_sample_weight = 0.10`
- `rare_balance_enabled = True`
- `old_class_sample_ratio = 0.80`
- `rare_class_keep_ratio = 1.0`
- `text_cache_path = test/cache/text_emb_11_rare_repair_lite.pt`
- `rare_oversample` 更激进：
  - `trash can: 3`
  - `window: 2`
  - `door: 2`
  - `fence: 3`
  - `pole_light: 3`
  - `motorcycle: 2`

含义：

- 更偏 rare 修复。
- 当前不是默认主线。

### `partial_unfreeze_lastnorm`

当前设置：

- `run_name = ESAM-CCLIP-11-partial-unfreeze-lastnorm`
- `train_json = train_label.json`
- `val_json = val_label.json`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 2`
- `decoder_lr = 2e-5`
- `image_lr = 5e-7`
- `min_lr_ratio = 0.3`
- `negative_sample_ratio = 0.02`
- `negative_sample_weight = 0.15`
- `unfreeze_image_mode = last_norm`

含义：

- 路线三常用入口。
- 只小范围打开 image encoder 尾部 `norm/neck/adapter/output/proj` 参数。

### `partial_unfreeze_last1`

当前设置：

- `run_name = ESAM-CCLIP-11-partial-unfreeze-last1`
- `train_json = train_label.json`
- `val_json = val_label.json`
- `no_val = False`
- `no_train_split_filter = False`
- `epochs = 1`
- `decoder_lr = 1e-5`
- `image_lr = 5e-7`
- `min_lr_ratio = 0.5`
- `negative_sample_ratio = 0.02`
- `negative_sample_weight = 0.15`
- `unfreeze_image_mode = last1`

含义：

- 比 `last_norm` 更激进。
- 只建议做短实验。

## 3. 当前训练脚本常用参数

当前训练脚本已经兼容以下常用参数：

- `--preset`
- `--resume_best`
- `--resume`
- `--resume_weights_only`
- `--no_auto_resume`
- `--no_val`
- `--with_val`
- `--no_train_split_filter`
- `--use_train_split_filter`
- `--train_json`
- `--val_json`
- `--all_json`
- `--train_list`
- `--val_list`
- `--text_cache_path`
- `--rebuild_text_cache`
- `--epochs`
- `--decoder_lr`
- `--image_lr`
- `--negative_sample_ratio`
- `--negative_sample_weight`
- `--old_class_sample_ratio`
- `--rare_class_keep_ratio`
- `--augment_prompt`
- `--prompt_alias_train`
- `--prompt_alias_prob`
- `--val_augment_prompt`
- `--unfreeze_image_mode`
- `--print_trainable_params`

几个最重要的行为：

- `--with_val` 等价于 `no_val=False`
- `--use_train_split_filter` 等价于 `no_train_split_filter=False`
- 如果 `args.no_val=True`，训练结束会保存 `final_fullset.pt`
- 如果 `args.no_val=False`，训练结束会保存 `final_split.pt`
- 如果 `--resume_best` 已传，脚本不会再用 `run_dir/last.pt` 抢优先级

## 4. 当前 partial unfreeze 设置

当前支持：

- `none`
- `last_norm`
- `last1`
- `last2`

规则是：

- `text_encoder` 始终冻结
- `decoder` 始终训练
- `image_encoder` 默认全冻
- 只有传 `--unfreeze_image_mode != none` 时才小范围打开

当前实现细节：

- `last_norm`
  - 只打开 image encoder 末尾匹配到的 `norm/neck/adapter/output/proj`
  - 目前比更早版本更保守，只开尾部一小段匹配参数
- `last1`
  - 尝试打开最后一个 block
- `last2`
  - 尝试打开最后两个 block

当前保护逻辑：

- 如果开启 partial unfreeze，但 `image_encoder trainable params = 0`，直接报错
- 如果 `text_encoder trainable params > 0`，直接报错
- 如果 `image_lr > decoder_lr`，会打 warning
- 如果 `last_norm` 打开的 image 参数明显过大，会打 warning

## 5. 当前训练产物命名

训练目录一般在：

- `test/train_output/<run_name>/`

当前常见产物：

- `last.pt`
- `best_all11.pt`
- `best_old5.pt`
- `best_rare.pt`
- `best_pos_only.pt`
- `final_split.pt`
- `final_fullset.pt`
- `<run_name>.log`

使用规则：

- 有验证集时：
  - 优先比较 `best_all11.pt / best_old5.pt / best_rare.pt`
- fullset 无验证时：
  - 重点看 `final_fullset.pt`

## 6. 当前 sweep 脚本入口

当前 sweep 入口是：

- `src/ESAM-CCLIP-11/sweep_thresholds_11.py`

当前默认：

- `DEFAULT_SWEEP_MODE = best`
- `DEFAULT_MAX_SAMPLES_PER_CLASS = 500`

`max_samples_per_class` 规则：

- `500` 或 `1000` 表示 reservoir sampling
- `0` 表示验证集全量

当前 sweep 还额外支持文本融合相关参数：

- `--prompt_fusion_mode`
  - `prototype`
  - `raw`
  - `blend`
- `--raw_prompt_weight`
- `--prompt_match_mode`
  - `exact`
  - `soft`

默认保持旧行为：

- `prompt_fusion_mode = prototype`
- `raw_prompt_weight = 0.0`
- `prompt_match_mode = exact`

如果 sweep 用了这些参数，写回 checkpoint 时也会同步写：

- `prompt_fusion_mode`
- `raw_prompt_weight`
- `prompt_match_mode`

这样新版 `inference.py` 能直接按 checkpoint metadata 复现同样的文本特征逻辑。

当前 sweep 输出 checkpoint 还支持两种写法：

- `--write_back_path`
- `--out_checkpoint`

如果两个同时传：

- 以 `--out_checkpoint` 为准
- 脚本会打印 warning

写回 checkpoint 时，当前会同步写：

- `prompt_thresholds`
- `val_thresholds`
- `postprocess`
- `postprocess_cfg`
- `sweep_mode`
- `sweep_metric`
- `img_size`

## 7. 当前 sweep 模式说明

当前保留并支持的模式：

- `best`
- `safe`
- `recall`
- `fine_recall`
- `hybrid`
- `hybrid_safe`

### `best`

说明：

- 当前默认模式。
- 偏 old 类稳定性。
- 阈值更贴近你历史 66.4 那套思路。

特点：

- old 类更稳
- rare 只轻微放宽
- 更适合作为默认提交候选

### `safe`

说明：

- 现在只是兼容别名。
- 内部会自动映射到 `best`。

### `recall`

说明：

- 较粗的 recall 网格。
- rare 类整体更积极。

特点：

- 搜索范围比 `best` 更偏召回
- 但 old 类也可能被带偏

### `fine_recall`

说明：

- 是 `recall` 的细网格版本。
- rare 类阈值更密，`min_area` 更小。

特点：

- 对 rare 更激进
- 会继续扫 `topk_components`
- 更适合榨 rare，不一定适合保 old

### `hybrid`

说明：

- old 类走 `best`
- rare 类走 `fine_recall`

特点：

- 不再粗暴二选一
- 适合想同时守 old、补 rare 的情况

### `hybrid_safe`

说明：

- old5 类固定使用当前 route2 hybrid 最优配置
- rare 类只允许在当前阈值及更高阈值里搜索
- rare 类 `min_area / topk_components` 固定，不再扩大搜索范围

固定 old 配置：

- `person`: `threshold=0.45`, `min_area=24`
- `car`: `threshold=0.50`, `min_area=32`
- `building`: `threshold=0.42`, `min_area=128`
- `tree`: `threshold=0.40`, `min_area=160`
- `animal`: `threshold=0.28`, `min_area=32`

rare 保守阈值网格：

- `trash can`: `[0.23, 0.25, 0.27, 0.30]`
- `window`: `[0.30, 0.32, 0.35, 0.38]`
- `door`: `[0.30, 0.32, 0.35]`
- `fence`: `[0.25, 0.27, 0.30]`
- `pole_light`: `[0.20, 0.22, 0.25, 0.28]`
- `motorcycle`: `[0.35, 0.37, 0.40]`

rare 固定 postprocess：

- `trash can`: `min_area=8`
- `window`: `min_area=8`
- `door`: `min_area=16`
- `fence`: `min_area=4`, `topk_components=3`
- `pole_light`: `min_area=4`
- `motorcycle`: `min_area=8`

适合：

- 路线二 fullset 权重
- 怀疑 rare 阈值过低导致官方测试假阳性偏多时

## 8. 当前 sweep 输出内容

输出目录里一般会有：

- `best_thresholds.json`
- `best_postprocess.json`
- `threshold_sweep_summary.csv`
- `sweep_summary.json`
- `sweep_summary_terminal.txt`
- `sweep_run.log`

当前 summary 会记录：

- `sweep_mode`
- `checkpoint`
- `max_samples_per_class`
- `best_metric`
- `sample_counts`
- 每类：
  - `best_threshold`
  - `best_min_area`
  - `best_topk_components`
  - `best_iou`
  - `precision`
  - `recall`
  - `pred_area`
  - `gt_area`

`hybrid_safe` 额外会打印：

- `old classes are fixed`
- `rare classes threshold-only conservative sweep`
- `fixed old cfg`
- `rare threshold grid`

## 9. 现在怎么选 train 和 sweep

当前实操建议：

- 训练主线优先：
  - `lite_fullset_polish`
  - `split_bridge_stage1 + split_bridge_stage2_fullset`
- 小实验：
  - `partial_unfreeze_lastnorm`
- sweep 优先级：
  1. `best`
  2. `hybrid`
  3. `fine_recall`
  4. `hybrid_safe`

使用经验：

- `best` 更稳
- `hybrid` 更适合“old 不想掉、rare 还想补”
- `fine_recall` 更像 rare 冲分线
- `hybrid_safe` 更像保守 hedge 版本

## 10. 推荐命令入口

查看训练帮助：

```powershell
python src/ESAM-CCLIP-11/train_esam_cclip_11.py --help
```

查看 sweep 帮助：

```powershell
python src/ESAM-CCLIP-11/sweep_thresholds_11.py --help
```

如果要看现成命令模板，直接看：

- [命令汇总.txt](/d:/nyz/raytron_project/src/ESAM-CCLIP-11/命令汇总.txt)
