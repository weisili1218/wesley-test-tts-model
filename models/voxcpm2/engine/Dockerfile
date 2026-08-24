# syntax=docker/dockerfile:1.7
# ===========================================================================
# VoxCPM2 (OpenBMB)  ── NVIDIA DGX Spark / GB10 (linux/arm64, Blackwell sm_121)
#
# 2B 參數、30 種語言、48kHz 輸出的 tokenizer-free diffusion AR TTS。
# 四顆引擎裡唯一同時支援「音檔克隆 / 文字描述造音色 / 什麼都不給」三種模式的。
#
# GB10 上要處理的：
#   1. torch 沒有能跑 sm_121 的預設 build → cu130 index 的 2.11.0 aarch64 wheel
#   2. torchcodec 需要系統 FFmpeg 動態函式庫 → apt 裝 ffmpeg
#   3. 預設會去 ModelScope 抓 ZipEnhancer 降噪模型 → server.py 用
#      load_denoiser=False 關掉，離線 image 才不會卡住
#
# 一定要在 GB10 上原生 build，不要用 --platform 走 QEMU emulation。
# ===========================================================================

FROM ubuntu:24.04

ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG MODEL_REPO=openbmb/VoxCPM2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    HF_HOME=/opt/hf \
    PATH=/opt/env/bin:$PATH

# ffmpeg 那組是 torchcodec 要的（它 dlopen libavcodec/libavformat）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3 python3-venv python3-dev \
        ffmpeg libsndfile1 \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/env && pip install --no-cache-dir -U pip setuptools wheel

# 先裝 torch，voxcpm 的 torch>=2.5 相依才不會從 PyPI 把別的版本蓋回來
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install huggingface_hub \
    && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='/models/voxcpm2')"

COPY server.py /opt/server/server.py

# --- 冒煙測試 ---------------------------------------------------------------
RUN python -c "\
import torch, torchcodec; \
from voxcpm import VoxCPM; \
import os; \
assert os.path.exists('/models/voxcpm2/audiovae.pth'), 'audiovae 沒抓到'; \
print('torch', torch.__version__); \
print('voxcpm import OK')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ENGINE_NAME=voxcpm2 \
    MODEL_PATH=/models/voxcpm2 \
    ENGINE_MODES=clone,design,default \
    VOXCPM_OPTIMIZE=0 \
    VOXCPM_CFG=2.0 \
    VOXCPM_TIMESTEPS=10

EXPOSE 8080
WORKDIR /opt/server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
