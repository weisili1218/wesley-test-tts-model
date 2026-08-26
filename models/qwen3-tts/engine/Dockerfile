# syntax=docker/dockerfile:1.7
# ===========================================================================
# Qwen3-TTS-12Hz-1.7B-CustomVoice  ── NVIDIA DGX Spark / GB10
#                                      (linux/arm64, Blackwell sm_121)
#
# 這一包掛的是 Base checkpoint：純克隆模型，吃一段參考音檔 + 逐字稿。
# 沒有內建音色 —— 那是 CustomVoice checkpoint 才有的（見下面 MODEL_REPO 註解）。
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
# 這兩個 checkpoint 換著掛都可以，換的時候記得一起改下面的 ENGINE_MODES：
#   -Base         上傳音檔克隆，沒有內建音色  → ENGINE_MODES=clone    （目前這個）
#   -CustomVoice  內建 9 個精選音色，不能克隆  → ENGINE_MODES=preset
# server.py 兩種 mode 都實作了，換 checkpoint 不用改程式，只有這兩個值要對上。
# （還有一個 -VoiceDesign 走純文字描述造音色，但它的 API 跟這兩個不同，
#   server.py 沒實作，要用得自己補一條 mode=design 的分支。）
ARG MODEL_REPO=Qwen/Qwen3-TTS-12Hz-1.7B-Base

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
# 權重先落在 cache mount 的 HF cache，再複製進 image，各包共用同一份 cache。
# 不能直接寫 snapshot_download(local_dir=...) —— 帶了 local_dir 就會繞過 cache
# （hub 0.36 的 docstring：the cache_dir will not be used），那樣每次這層失效
# 都會重抓一次權重。copytree 的 symlinks=False 把 cache 裡指向 blobs/ 的符號
# 連結解成實體檔案，所以 image 仍然自給自足，跑的時候不依賴 cache mount。
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,id=hf-hub,target=/hf-cache,sharing=locked \
    pip install huggingface_hub \
    && HF_HUB_CACHE=/hf-cache python -c "\
import shutil; \
from huggingface_hub import snapshot_download; \
shutil.copytree(snapshot_download('${MODEL_REPO}'), '/models/qwen3-tts', \
                symlinks=False, dirs_exist_ok=True)"

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
    ENGINE_MODES=clone

EXPOSE 8080
WORKDIR /opt/server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
