# =============================================================================
# raytron 赛题二 提交镜像 —— CLIPSeg
# =============================================================================
# 构建流程:
#   1. python src/export_model.py          ← 导出模型到 model/submit/
#   2. docker build -t raytron-submit .
#   3. docker save raytron-submit | gzip > raytron-submit.tar.gz
# =============================================================================

FROM supervisely/sam3:1.0.6

# ── 系统依赖（OpenCV 运行时）──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── 安装 CLIPSeg 额外依赖 ──
# PyTorch / CUDA / pycocotools 已由基础镜像提供，只需补充:
#   transformers  → CLIPSegProcessor + CLIPSegForImageSegmentation
#   opencv-python → 图像预处理 (灰度读取/反色/resize/pad)
RUN pip install --no-cache-dir \
    transformers>=4.46 \
    opencv-python-headless>=4.8 \
    huggingface_hub

# ── 模型（/raytron/code/model/）──
# model/submit/ 由 src/export_model.py 生成，包含 sam3.pt + config.json + tokenizer 等
COPY model/submit/ /raytron/code/model/

# ── 推理脚本 ──
COPY inference.py /raytron/code/inference.py

# ── 验证 ──
RUN echo "=== 关键文件 ===" \
    && ls -lh /raytron/code/inference.py \
    && ls -lh /raytron/code/model/sam3.pt \
    && ls -lh /raytron/code/model/config.json \
    && echo "=== Python 环境 ===" \
    && python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" \
    && python -c "from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation; print('CLIPSeg OK')" \
    && python -c "import cv2; print(f'OpenCV {cv2.__version__}')" \
    && python -c "import pycocotools; print('pycocotools OK')" \
    && echo "=== 推理脚本语法 ===" \
    && python -c "import py_compile; py_compile.compile('/raytron/code/inference.py', doraise=True); print('OK')" \
    && echo "=== 构建完成 ==="

WORKDIR /raytron
