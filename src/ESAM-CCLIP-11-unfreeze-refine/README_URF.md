# ESAM-CCLIP-11cls URF

这条线的目标是做一条独立于主线的冲榜实验线：

- `ESAM-CCLIP-11cls balanced partial unfreeze + zero-init refine head`

核心原则：

- 不修改原 `src/ESAM-CCLIP-11/`
- 原最高分线继续保留，作为保底提交线
- 新线所有实验都放在 `src/ESAM-CCLIP-11-unfreeze-refine/`

## 目的

这条线不是替代原最高分线，而是在不破坏原主线的前提下，尝试两类增益：

- `balanced partial unfreeze`
  只小范围打开 EfficientSAM image encoder 后段参数，目标是微调视觉表征，但避免把 backbone 全量带偏。
- `zero-init refine head`
  在原 coarse logit 上叠加一个零初始化 residual refine head，目标是做边界和局部细节修正，同时保持旧 checkpoint 热启动稳定。

## 不改原线

- 原目录 `src/ESAM-CCLIP-11/` 不做任何修改。
- 原最高分 checkpoint 和原提交逻辑可以继续直接使用。
- `URF` 线所有训练、sweep、inference 都从新目录独立运行。

## 两条 Pipeline

- `unfreeze_then_refine`
- `refine_then_unfreeze`

### Pipeline A: `unfreeze_then_refine`

顺序：

1. `A1_unfreeze`
   - preset: `partial_unfreeze_balanced_safe`
   - 作用：先做 1 epoch 的 balanced partial unfreeze，轻量打开 image encoder 后段，优先看 old5 是否稳定。

2. `A2_recalibrate`
   - preset: `balanced_recalibrate`
   - 作用：重新冻结 image encoder，只校准 decoder，把 A1 带来的轻微漂移收回来。

3. `A3_refine`
   - preset: `zero_init_refine`
   - 作用：在已校准的 coarse 模型上接 zero-init refine head，观察 old5/rare 是否同时有改善。

### Pipeline B: `refine_then_unfreeze`

顺序：

1. `B1_refine`
   - preset: `zero_init_refine`
   - 作用：先只加 refine head，确认 refine 本身不会明显拉低 old5。

2. `B2_unfreeze`
   - preset: `partial_unfreeze_balanced_safe`
   - 作用：保留 refine head 的同时做 1 epoch partial unfreeze，测试“refine 后再开 backbone”是否更稳。

3. `B3_recalibrate`
   - preset: `balanced_recalibrate_refine`
   - 作用：回冻 image encoder，联合校准 decoder + refine head，减少 B2 的波动。

## Stage 作用

- `partial_unfreeze_balanced_safe`
  - old5 为主，rare 轻量参与
  - 只允许小范围解冻 image encoder 后段
  - 目标是“稳 old5，不做 rare-heavy”

- `balanced_recalibrate`
  - 完全冻结 image encoder / text encoder
  - 只调 decoder
  - 目标是收回 partial unfreeze 的偏移

- `zero_init_refine`
  - 打开 refine head
  - 最后一层零初始化
  - 目标是先验证 refine 是否带来净收益

- `balanced_recalibrate_refine`
  - 冻结 backbone
  - 联合调 decoder + refine head
  - 目标是做 refine 阶段后的保守校准

## Refine Head

当前实现和模型文件一致：

- 输入：`coarse_logit_up` + `gray_image`
- 结构：
  - `Conv2d(2,16,3,padding=1)`
  - `ReLU`
  - `Conv2d(16,16,3,padding=1)`
  - `ReLU`
  - `Conv2d(16,1,1)`
- 输出：
  - `final_logit = coarse_logit_up + residual_logit`

安全点：

- 最后一层 `1x1 Conv` 权重和 bias 为零初始化
- 旧 checkpoint 用 `strict=False` 加载
- `use_refine_head=False` 时，输出逻辑等价于原线 coarse 输出

## 训练命令示例

### 单段训练

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/train_esam_cclip_11_urf.py ^
  --preset partial_unfreeze_balanced_safe ^
  --resume D:\nyz\raytron_project\test\train_output\YOUR_BASE\best.pt ^
  --resume_weights_only ^
  --no_auto_resume ^
  --output_dir D:\nyz\raytron_project\test\train_output ^
  --device cuda ^
  --batch_size 8 ^
  --num_workers 8
```

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/train_esam_cclip_11_urf.py ^
  --preset zero_init_refine ^
  --resume D:\nyz\raytron_project\test\train_output\YOUR_STAGE\best.pt ^
  --resume_weights_only ^
  --no_auto_resume ^
  --use_refine_head ^
  --train_refine_head ^
  --output_dir D:\nyz\raytron_project\test\train_output
```

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/train_esam_cclip_11_urf.py `
  --preset balanced_recalibrate `
  --resume runs/URF_A_manual/A1_unfreeze/A1_unfreeze/best.pt `
  --resume_weights_only `
  --no_auto_resume `
  --output_dir runs/URF_A_manual/A2_recalibrate `
  --run_name A2_recalibrate `
  --pipeline_name A `
  --pipeline_stage A2_recalibrate `
  --no_use_refine_head `
  --device cuda `
  --num_workers 4 `
  --batch_size 8
```

### pipelineA:

```powershell
partial_unfreeze_balanced_safe
-> freeze recalibrate
-> zero_init_refine
```

### pipeline B：

```
zero_init_refine
-> partial_unfreeze_balanced_safe
-> balanced_recalibrate_refine
```



### 跑整条 Pipeline

`unfreeze_then_refine`:

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/run_pipeline_urf.py ^
  --pipeline unfreeze_then_refine ^
  --base_ckpt D:\nyz\raytron_project\test\train_output\YOUR_BASE\best.pt ^
  --output_root D:\nyz\raytron_project\test\train_output ^
  --device cuda ^
  --batch_size 8 ^
  --num_workers 8
```

`refine_then_unfreeze`:

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/run_pipeline_urf.py ^
  --pipeline refine_then_unfreeze ^
  --base_ckpt D:\nyz\raytron_project\test\train_output\YOUR_BASE\best.pt ^
  --output_root D:\nyz\raytron_project\test\train_output ^
  --device cuda ^
  --batch_size 8 ^
  --num_workers 8
```

只看将执行的命令：

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/run_pipeline_urf.py ^
  --pipeline refine_then_unfreeze ^
  --base_ckpt D:\nyz\raytron_project\test\train_output\YOUR_BASE\best.pt ^
  --dry_run
```

## Sweep 命令示例

平衡目标：

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/sweep_thresholds_11_urf.py ^
  --checkpoint D:\nyz\raytron_project\test\train_output\YOUR_URF_STAGE\best.pt ^
  --objective balanced ^
  --output_dir D:\nyz\raytron_project\test\train_output\threshold_sweep_urf_balanced ^
  --out_checkpoint D:\nyz\raytron_project\model\submit-urf-balanced\sam3.pt
```

old5 保守目标：

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/sweep_thresholds_11_urf.py ^
  --checkpoint D:\nyz\raytron_project\test\train_output\YOUR_URF_STAGE\best.pt ^
  --objective old5_safe ^
  --output_dir D:\nyz\raytron_project\test\train_output\threshold_sweep_urf_old5_safe ^
  --out_checkpoint D:\nyz\raytron_project\model\submit-urf-old5-safe\sam3.pt
```

提交保守目标：

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/sweep_thresholds_11_urf.py ^
  --checkpoint D:\nyz\raytron_project\test\train_output\YOUR_URF_STAGE\best.pt ^
  --objective submit_safe ^
  --output_dir D:\nyz\raytron_project\test\train_output\threshold_sweep_urf_submit_safe ^
  --out_checkpoint D:\nyz\raytron_project\model\submit-urf-submit-safe\sam3.pt
```

输出会包含：

- `thresholds.json`
- `sweep_report.md`
- `sweep_summary.json`
- `best_thresholds.json`
- `best_postprocess.json`
- 写回后的 swept checkpoint

## Inference 命令示例

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/inference_urf.py ^
  --tasks D:\nyz\raytron_project\test\test_tasks.json ^
  --images_root D:\nyz\raytron_project\test ^
  --output D:\nyz\raytron_project\test\predictions_urf.json ^
  --model_dir D:\nyz\raytron_project\model ^
  --checkpoint D:\nyz\raytron_project\model\submit-urf-balanced\sam3.pt
```

如果需要额外指定 sweep 产物：

```powershell
python src/ESAM-CCLIP-11-unfreeze-refine/inference_urf.py ^
  --tasks D:\nyz\raytron_project\test\test_tasks.json ^
  --images_root D:\nyz\raytron_project\test ^
  --output D:\nyz\raytron_project\test\predictions_urf.json ^
  --model_dir D:\nyz\raytron_project\model ^
  --checkpoint D:\nyz\raytron_project\model\submit-urf-balanced\sam3.pt ^
  --threshold_json D:\nyz\raytron_project\test\train_output\threshold_sweep_urf_balanced\thresholds.json
```

## 注意事项

- `base_ckpt` 必须优先使用当前最高分底座：
  - 当前 `66.462` 对应 checkpoint
  - 或 `lite_stage2_submit` 对应 checkpoint
- `partial unfreeze` 只建议先跑 `1 epoch`
- `refine` 先跑 `1 epoch`，先看 `old5 / rare / all11`，再决定是否继续
- 任意候选模型在提交前都必须重新 `sweep`
- `URF` 线的结论必须和原主线对照看，不能只看单次波动
- 如果 `partial unfreeze` 后 old5 明显掉点，优先走 `recalibrate` 收一轮，不要直接继续加长训练
