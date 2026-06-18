# ESAM-CCLIP-11 运行说明

## 当前目录结构

- 根目录公共核心：
  - `common_esam_cclip_11.py`
  - `config_esam_cclip_11.py`
  - `dataset_esam_cclip_11.py`
  - `model_esam_cclip_11.py`
  - `prompt_prototypes.py`
  - `path_utils.py`
- 根目录主入口：
  - `train_esam_cclip_11.py`
  - `sweep_thresholds_11.py`
- `legacy/`
  - `A/`：保留 A 线现有脚本
  - `A-lite/`：保留 A-lite 线现有脚本
- `文档/`
  - 当前整理说明、路线说明、历史文档
- `inference.py`
  - 提交推理脚本，本次整理不修改

## 当前主入口

- 训练主入口：`python src/ESAM-CCLIP-11/train_esam_cclip_11.py --help`
- Sweep 主入口：`python src/ESAM-CCLIP-11/sweep_thresholds_11.py --help`

根目录 `sweep_thresholds_11.py` 现在保留新版 sweep 逻辑，支持：

- `--sweep_mode safe`
- `--sweep_mode recall`
- `--max_samples_per_class`
- checkpoint 写回

## legacy 目录

- `legacy/A/`
  - 保留 A 线历史训练脚本、配置和 sweep
- `legacy/A-lite/`
  - 保留 A-lite 线历史训练脚本、配置和 sweep

这些目录只做路径和入口兼容修复，不改原有训练策略。

可用自检命令：

- `python src/ESAM-CCLIP-11/legacy/A/train_esam_cclip_11.py --help`
- `python src/ESAM-CCLIP-11/legacy/A/sweep_thresholds_11.py --help`
- `python src/ESAM-CCLIP-11/legacy/A-lite/train_esam_cclip_11.py --help`
- `python src/ESAM-CCLIP-11/legacy/A-lite/sweep_thresholds_11.py --help`

## 推理脚本

- `inference.py` 不要动。
- 如果后续训练脚本或公共模块再调整，优先通过根目录兼容层解决，不要直接改推理入口。

## ESAM-CCLIP-11-mid

- `src/ESAM-CCLIP-11-mid` 这次不在整理范围内。
- mid 目录继续独立维护。
