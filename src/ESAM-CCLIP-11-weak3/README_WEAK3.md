# WEAK3

当前目录在原 ESAM 11 类基础上验证 `weak window + weak door + weak pole_light`。

核心策略：

- 训练仍保持 11 类，不删除 `window / door / pole_light`
- 仅在训练阶段弱化三类样本保留比例和 loss 权重
- 推理与提交格式保持官方 11 类不变

当前默认 weak3 配置：

- `window keep_ratio=0.10`
- `door keep_ratio=0.35`
- `pole_light keep_ratio=0.50`
- `window final_loss_weight=0.10`
- `door final_loss_weight=0.35`
- `pole_light final_loss_weight=0.50`

注意：

- `WEAK_CLASS_LOSS_WEIGHT` 只保留给日志和兼容字段
- dataset 不再把 `WEAK_CLASS_LOSS_WEIGHT` 再乘到 `sample_weight`
- 最终弱类 loss 权重来源只有 `CLASS_WEIGHTS`

推荐命令

0. 静态检查

```powershell
python -m py_compile src\ESAM-CCLIP-11-weak3\config_esam_cclip_11.py
python -m py_compile src\ESAM-CCLIP-11-weak3\dataset_esam_cclip_11.py
python -m py_compile src\ESAM-CCLIP-11-weak3\train_esam_cclip_11.py
python -m py_compile src\ESAM-CCLIP-11-weak3\sweep_thresholds_11.py
python -m py_compile src\ESAM-CCLIP-11-weak3\inference.py
```

1. dry run 数据集统计

```powershell
python src\ESAM-CCLIP-11-weak3\train_esam_cclip_11.py `
  --train_json test\clean_rare\label\train_label.json `
  --val_json test\clean_rare\label\val_label.json `
  --train_list test\train_list.txt `
  --val_list test\val_list.txt `
  --dry_run_dataset_stats `
  --run_name ESAM-CCLIP-11-weak3-dryrun
```

2. split 1 epoch 诊断

```powershell
python src\ESAM-CCLIP-11-weak3\train_esam_cclip_11.py `
  --train_json test/clean_rare/label/train_label.json `
  --val_json test/clean_rare/label/val_label.json `
  --train_list test/train_list.txt `
  --val_list test/val_list.txt `
  --resume_best "<6CLS_BEST_CHECKPOINT>" `
  --epochs 1 `
  --batch_size 4 `
  --workers 4 `
  --run_name ESAM-CCLIP-11-weak3-restart6-e1 `
  --no_auto_resume `
  --rebuild_text_cache
```

3. fullset 训练

```powershell
python src\ESAM-CCLIP-11-weak3\train_esam_cclip_11.py `
  --train_json test/clean_rare/label/trainval_label.json `
  --no_train_split_filter `
  --no_val `
  --resume_best "<6CLS_BEST_CHECKPOINT>" `
  --epochs 3 `
  --batch_size 4 `
  --workers 4 `
  --run_name ESAM-CCLIP-11-weak3-restart6-fullset-e3 `
  --no_auto_resume `
  --rebuild_text_cache
```

4. weak3_safe sweep

```powershell
python src\ESAM-CCLIP-11-weak3\sweep_thresholds_11.py `
  --checkpoint test/train_output/ESAM-CCLIP-11-weak3-restart6-fullset-e3/final_fullset.pt `
  --sweep_mode weak3_safe `
  --objective balanced `
  --max_samples_per_class 500 `
  --write_back_checkpoint `
  --write_back_path model/submit-rsam-weak3/sam3.pt
```

5. 从 66.46 checkpoint 继续微调

```powershell
python src\ESAM-CCLIP-11-weak3\train_esam_cclip_11.py `
  --train_json test/clean_rare/label/train_label.json `
  --val_json test/clean_rare/label/val_label.json `
  --train_list test/train_list.txt `
  --val_list test/val_list.txt `
  --resume_best "<BEST66_CHECKPOINT>" `
  --resume_weights_only `
  --reset_history `
  --epochs 1 `
  --batch_size 4 `
  --workers 4 `
  --run_name ESAM-CCLIP-11-weak3-from6646-e1 `
  --no_auto_resume `
  --rebuild_text_cache
```
