# syntax=docker/dockerfile:1.7
# ===========================================================================
# TTS Gateway ── qwen3-tts 這一包的對外入口
#
# 做三件事：
#   1. 音色庫（/v1/voices）：上傳參考音檔 → 正規化成 16k 單聲道 → 給一個 voice_id
#   2. OpenAI 相容合成（/v1/audio/speech）
#   3. 引擎前面一個 semaphore 排隊，避免同時打爆沒有 request queue 的引擎
#
# 這個 image 不需要 GPU，也不含任何模型權重，所以很小（<200MB），
# 改一行重 build 只要兩分鐘，不會動到旁邊那顆十幾 GB 的引擎 image。
#
# app.py 四包完全一致，不寫死引擎名稱 —— 掛哪顆全看 ENGINES 環境變數，
# 底下的預設值只是 compose 沒給的時候的備援。
# ===========================================================================

FROM python:3.12-slim-bookworm

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# ffmpeg 用來做上傳音檔正規化與輸出格式轉換
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app.py /opt/gateway/app.py
WORKDIR /opt/gateway

ENV VOICES_DIR=/voices \
    ENGINES="qwen3-tts=http://engine:8080" \
    DEFAULT_ENGINE=qwen3-tts

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
