# DINO + ESAM 混合推理脚本说明

## 1. 脚本定位

`infer_DINO_ESAM_hybrid.py` 是一条**只改推理、不重训 ESAM** 的混合路线：

```text
GroundingDINO 找框
-> 按框裁 ROI
-> 用现有 ESAM checkpoint 在 ROI 内做文本分割
-> 回贴原图
-> 输出提交 JSON
```

它的目标不是替代 `src/ESAM-CCLIP-11/inference.py`，而是在 `src/DINO/` 下提供一个独立可运行的脚本，验证：

- DINO 先缩小搜索范围后，ESAM 对小目标/长尾类是否更稳
- 不改训练、不改 ESAM 原脚本，能不能快速形成一条新提交线

---

## 2. 能不能写完直接提交

不能。

写完脚本后，至少还要做两步：

1. 本地跑一次推理，确认能正常输出 `predictions.json`
2. 把提交时需要的依赖文件整理齐，再放进比赛镜像/提交目录

也就是说，这个脚本是**提交入口脚本**，但不是“写完文件就自动能提交”。

---

## 3. 这个脚本和原 ESAM 推理的关系

这个脚本**不修改** `src/ESAM-CCLIP-11/inference.py`。

它也**不 import `src/ESAM-CCLIP-11` 目录下的代码**。  
需要的 ESAM 逻辑已经独立封装在脚本内部，只依赖：

- `src/hanxue/models/image_encoder.py`
- `src/hanxue/models/text_encoder.py`
- `src/hanxue/models/fusion_decoder.py`

所以它是放在 `src/DINO/` 下的一条独立推理线。

---

## 4. 运行前需要的文件

至少要有这些资源：

### ESAM 权重侧

- ESAM checkpoint  
  例如：
  `test/train_output/ESAM-CCLIP-11-fastfinetune/final_fullset.pt`

- Chinese-CLIP tokenizer 目录  
  例如：
  `src/hanxue/weights/chinese_clip/`

- EfficientSAM backbone 权重  
  例如：
  `src/hanxue/weights/efficient_sam/efficient_sam_vitt.pt`

- text cache  
  例如：
  `test/cache/text_emb_11.pt`

### DINO 侧

脚本支持两种 DINO 后端：

1. `hf`
   使用 HuggingFace `IDEA-Research/grounding-dino-base`

2. `local`
   使用本地 `groundingdino` 仓库 API

当前仓库里已经有：

- `src/DINO/models/groundingdino/groundingdino_swint_ogc.pth`

但如果你要走 `local` 后端，还需要本地 `groundingdino` Python 包和 config 文件。

---

## 5. 输入输出协议

### 输入

任务 JSON 要包含：

- `ann_id`
- `image_path`
- `text_prompt` 或 `prompt` 或 `text`

### 输出

输出是：

```json
{
  "predictions": [
    {
      "ann_id": 123,
      "rle": {
        "size": [H, W],
        "counts": "..."
      }
    }
  ]
}
```

和原提交协议一致。

---

## 6. 主流程

对每张图、每条唯一 prompt，脚本会做：

1. 用 GroundingDINO 根据 prompt 找框
2. 按分数排序并截断到 `max_boxes_per_prompt`
3. 对每个框做 `box_expand_ratio` 扩张
4. 从原图灰度图中裁 ROI
5. 用 ESAM checkpoint 对 ROI 做文本分割
6. 把 ROI mask 贴回原图
7. 所有框的结果做并集
8. 用 checkpoint 中保存的阈值和后处理参数转成最终 mask
9. 编码成 RLE

如果传了：

```bash
--fallback_full_image
```

那么 DINO 没找到框时，会退回一次全图 ESAM 分割。

---

## 7. 推荐本地运行命令

### 先做本地 dry run

```powershell
python src/DINO/infer_DINO_ESAM_hybrid.py `
  --tasks test/test_tasks.json `
  --image_root . `
  --output test/predictions_dino_esam_hybrid.json `
  --checkpoint test/train_output/ESAM-CCLIP-11-fastfinetune/final_fullset.pt `
  --tokenizer_dir src/hanxue/weights/chinese_clip `
  --efficient_sam_ckpt src/hanxue/weights/efficient_sam/efficient_sam_vitt.pt `
  --text_cache_path test/cache/text_emb_11.pt `
  --save_debug_json
```

如果你本地没有 `test/test_tasks.json`，就换成你自己的任务 JSON。

### 如果 DINO 框特别容易漏

可以先试：

```powershell
python src/DINO/infer_DINO_ESAM_hybrid.py `
  --tasks test/test_tasks.json `
  --image_root . `
  --output test/predictions_dino_esam_hybrid.json `
  --checkpoint test/train_output/ESAM-CCLIP-11-fastfinetune/final_fullset.pt `
  --tokenizer_dir src/hanxue/weights/chinese_clip `
  --efficient_sam_ckpt src/hanxue/weights/efficient_sam/efficient_sam_vitt.pt `
  --text_cache_path test/cache/text_emb_11.pt `
  --dino_box_threshold 0.15 `
  --box_expand_ratio 0.15 `
  --fallback_full_image `
  --save_debug_json
```

---

## 8. 主要参数

### DINO 相关

- `--dino_backend`
  `auto / hf / local`

- `--hf_model_id`
  HF 模型名，默认 `IDEA-Research/grounding-dino-base`

- `--groundingdino_config`
  本地 GroundingDINO config 路径

- `--groundingdino_ckpt`
  本地 GroundingDINO 权重路径

- `--dino_box_threshold`
  DINO 框阈值，未命中已知类映射时用这个默认值

- `--dino_text_threshold`
  DINO 文本匹配阈值

- `--max_boxes_per_prompt`
  每条 prompt 最多保留多少个框

- `--box_expand_ratio`
  ROI 裁剪前对框做多少比例扩张

### ESAM 相关

- `--checkpoint`
  ESAM checkpoint 路径

- `--tokenizer_dir`
  Chinese-CLIP tokenizer 目录

- `--efficient_sam_ckpt`
  EfficientSAM backbone 权重

- `--text_cache_path`
  text cache 路径

- `--default_mask_threshold`
  prompt 不在 checkpoint 类别映射内时，ESAM 默认 mask 阈值

### 调试相关

- `--fallback_full_image`
  DINO 没框时，退回全图 ESAM 一次

- `--strict`
  某张图失败时直接报错退出，不自动输出空 mask

- `--save_debug_json`
  额外保存 debug 信息，包括每条 prompt 找到多少框、最终 mask 面积等

---

## 9. 推荐调参顺序

优先从这些参数开始试：

1. `--dino_box_threshold`
2. `--box_expand_ratio`
3. `--max_boxes_per_prompt`
4. `--fallback_full_image`

建议先把混合线用在这些类上观察收益：

- `animal`
- `trash can`
- `pole_light`
- `motorcycle`
- `door`
- `window`
- `fence`

这些类通常比 `person/car/building/tree` 更容易从 DINO proposal 中受益。

---

## 10. 本地验证通过后再做什么

本地跑通之后，再做这两件事：

1. 把提交时真正要用的资源整理进固定目录  
   至少包括：
   - `sam3.pt` 或你的 ESAM checkpoint
   - tokenizer
   - `text_emb_11.pt`
   - `efficient_sam_vitt.pt`
   - DINO 需要的模型资源

2. 用比赛目录约定再跑一遍  
   比如最终环境里常见的是：
   - 任务 JSON 在 `/raytron/test/test_tasks.json`
   - 输出写到 `/raytron/test/predictions.json`

---

## 11. 一句话总结

这条脚本的本质是：

```text
不重训 ESAM
只在推理阶段把 GroundingDINO 接到 ESAM 前面
用 DINO 缩小搜索范围，用 ESAM 保留最终分割质量
```

所以它是当前最适合快速验证收益的一条独立推理线。
