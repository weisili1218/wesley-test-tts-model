# syntax=docker/dockerfile:1.7
# ===========================================================================
# CosyVoice2-0.5B  ──  NVIDIA DGX Spark / GB10 (linux/arm64, Blackwell sm_121)
#
# 跟官方 CosyVoice 的 Docker 完全不同，因為官方是 x86_64 + CUDA 12.1。
# 在 GB10 上要處理的是：
#   1. torch 2.3.1+cu121 沒有 aarch64 build，而且不認得 sm_121  → 用 cu130 2.11.0
#   2. onnxruntime-gpu 在 PyPI 沒有 aarch64 wheel               → 用 CPU 版
#   3. ttsfrd 只有 x86_64 wheel                                → fallback 到 wetext
#
# 好消息：官方新版已經把文字正規化從 WeTextProcessing(pynini) 換成 wetext(kaldifst)，
# kaldifst 有 aarch64 wheel，所以這版**不需要 micromamba/conda**，Ubuntu 內建的
# Python 3.12 直接用就好，build 比舊的 BreezyVoice 版快很多。
#
# 一定要在 GB10 上原生 build，不要用 --platform 走 QEMU emulation。
# ===========================================================================

FROM ubuntu:24.04

ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
# GB10 是 Blackwell sm_121，必須 CUDA 13 的 wheel。
# 想改純 CPU 測試：--build-arg TORCH_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG COSYVOICE_REF=main
ARG MODEL_REPO=FunAudioLLM/CosyVoice2-0.5B

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    HF_HOME=/opt/hf \
    PATH=/opt/env/bin:$PATH \
    PYTHONPATH=/opt/cosyvoice:/opt/cosyvoice/third_party/Matcha-TTS \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --- 系統套件 ---------------------------------------------------------------
# build-essential + python3-dev 是為了 pyworld（只有 sdist，要現場編）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3 python3-venv python3-dev \
        ffmpeg sox libsox-dev \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/env && pip install --no-cache-dir -U pip setuptools wheel

# --- PyTorch（CUDA 13 / aarch64）--------------------------------------------
# 先裝 torch，後面的相依才不會從 PyPI 把 CPU 版 torch 蓋回來。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

# --- 其餘 Python 相依 --------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --- CosyVoice 原始碼（含 Matcha-TTS submodule）------------------------------
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/cosyvoice \
    && cd /opt/cosyvoice \
    && git checkout "${COSYVOICE_REF}" \
    && git submodule update --init --recursive \
    && rm -rf .git third_party/Matcha-TTS/.git

# --- 把權重烘進 image（之後就能完全離線跑）----------------------------------
# 用 python API 而不是 huggingface-cli / hf，因為那兩個指令名稱在 hub 版本之間換過。
RUN --mount=type=cache,target=/root/.cache/pip pip install huggingface_hub \
    && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='/models/cosyvoice2')"

COPY server.py /opt/server/server.py

# --- 冒煙測試：build 階段就確認能 import，壞掉不會產出壞 image ---------------
RUN python -c "\
import torch, onnxruntime, torchaudio, whisper; \
from wetext import Normalizer; \
from cosyvoice.cli.cosyvoice import CosyVoice2; \
print('torch', torch.__version__); \
print('ort  ', onnxruntime.__version__); \
Normalizer(remove_erhua=False); \
print('wetext OK / CosyVoice2 import OK')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ENGINE_NAME=cosyvoice2 \
    ENGINE_MODES=clone \
    COSYVOICE_CLASS=CosyVoice2 \
    MODEL_PATH=/models/cosyvoice2

EXPOSE 8080
WORKDIR /opt/server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
