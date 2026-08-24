# syntax=docker/dockerfile:1.7
# ===========================================================================
# Qwen3-TTS-12Hz-1.7B-CustomVoice  ── NVIDIA DGX Spark / GB10
#                                      (linux/arm64, Blackwell sm_121)
#
# 這顆是四個引擎裡唯一「不靠上傳音檔克隆」的：CustomVoice 走的是內建 9 個精選
# 音色 + 自然語言指令控制。所以 gateway 會把它的音色當成 preset 類型處理。
#
# GB10 上要處理的：
#   1. torch 沒有能跑 sm_121 的預設 build  → cu130 index 的 2.11.0 aarch64 wheel
#   2. flash-attn 在 aarch64+Blackwell 要現場編譯，很久又常失敗 → 改用 sdpa
#
# 一定要在 GB10 上原生 build，不要用 --platform 走 QEMU emulation。
# ===========================================================================

FROM ubuntu:24.04

ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
# 想要真正的「3 秒上傳克隆」就把這行換成 Qwen/Qwen3-TTS-12Hz-1.7B-Base，
# 想要純文字描述造音色就換 Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign。
ARG MODEL_REPO=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    HF_HOME=/opt/hf \
    PATH=/opt/env/bin:$PATH

# sox / libsox 是 qwen-tts 相依的 pysox 需要的系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3 python3-venv python3-dev \
        ffmpeg sox libsox-dev \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/env && pip install --no-cache-dir -U pip setuptools wheel

# 先裝 torch，qwen-tts 的 torchaudio 相依才不會從 PyPI 把 CPU 版蓋回來
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --- 把權重烘進 image ---------------------------------------------------------
# speech_tokenizer/ 就在同一個 repo 底下，所以只要抓這一包，不用另外抓
# Qwen3-TTS-Tokenizer-12Hz。給本地目錄時 from_pretrained 不會再連網。
RUN --mount=type=cache,target=/root/.cache/pip pip install huggingface_hub \
    && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='/models/qwen3-tts')"

COPY server.py /opt/server/server.py

# --- 冒煙測試 ---------------------------------------------------------------
RUN python -c "\
import torch, transformers; \
from qwen_tts import Qwen3TTSModel; \
import os; \
assert os.path.exists('/models/qwen3-tts/speech_tokenizer/config.json'), 'speech_tokenizer 沒抓到'; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('qwen-tts import OK')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ENGINE_NAME=qwen3-tts \
    MODEL_PATH=/models/qwen3-tts \
    QWEN_ATTN=sdpa \
    QWEN_DTYPE=bfloat16 \
    QWEN_DEFAULT_SPEAKER=Vivian \
    ENGINE_MODES=preset

EXPOSE 8080
WORKDIR /opt/server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
