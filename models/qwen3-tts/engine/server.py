"""
Qwen3-TTS-12Hz-1.7B-CustomVoice engine HTTP server。

跟另外三顆引擎不一樣的地方（很重要）：
  CustomVoice 這個 checkpoint 走的是「內建精選音色 + 指令控制」，
  不是「上傳一段音檔克隆」。所以：
    - mode=preset  → 用內建 speaker（Vivian / Serena / Uncle_Fu / ...），這是主要用法
    - mode=clone   → 會嘗試 generate_voice_clone；CustomVoice 權重通常不支援，
                     失敗時回 400 並建議改用 CosyVoice/VoxCPM 那三顆
  要真正的 3 秒克隆請改掛 Qwen/Qwen3-TTS-12Hz-1.7B-Base（改 MODEL_REPO 重 build）。

環境變數：
  ENGINE_NAME / MODEL_PATH / QWEN_ATTN / QWEN_DTYPE / QWEN_DEFAULT_SPEAKER
"""
import io
import os
import logging

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

ENGINE_NAME = os.environ.get("ENGINE_NAME", "qwen3-tts")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/qwen3-tts")
# sdpa 是 aarch64 上唯一不用現場編譯就能用的實作。想換 flash_attention_2
# 要先自己在 image 裡把 flash-attn 編起來。
ATTN = os.environ.get("QWEN_ATTN", "sdpa")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("QWEN_DTYPE", "bfloat16")
]
DEFAULT_SPEAKER = os.environ.get("QWEN_DEFAULT_SPEAKER", "Vivian")
# CustomVoice checkpoint 只有 preset 音色。換成 Qwen3-TTS-12Hz-1.7B-Base 重 build 時，
# 把這個環境變數設成 "preset,clone"，gateway 就會自動把克隆音色也路由過來。
MODES = [m.strip() for m in os.environ.get("ENGINE_MODES", "preset").split(",") if m.strip()]

app = FastAPI(title=f"{ENGINE_NAME} engine")
_model = None
_presets: list[str] = []


def get_model():
    global _model, _presets
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("loading Qwen3-TTS from %s (device=%s attn=%s)", MODEL_PATH, device, ATTN)
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map=device, dtype=DTYPE, attn_implementation=ATTN
        )
        _presets = _model.get_supported_speakers() or []
        log.info("loaded, presets=%s", _presets)
    return _model


class SynthRequest(BaseModel):
    text: str
    mode: str = "preset"
    ref_audio_path: str | None = None
    ref_text: str | None = None
    description: str | None = None
    instruct: str | None = None
    speaker: str | None = None
    language: str | None = None
    speed: float = 1.0


@app.get("/health")
def health():
    return {
        "engine": ENGINE_NAME,
        "model_path": MODEL_PATH,
        "loaded": _model is not None,
        "sample_rate": None,
        "modes": MODES,
        "needs_ref_audio": False,
        "presets": _presets,
    }


@app.get("/presets")
def presets():
    """gateway 用這支把 Qwen 的內建音色併進 /v1/voices 的清單。"""
    m = get_model()
    return {
        "speakers": m.get_supported_speakers() or [],
        "languages": m.get_supported_languages() or [],
    }


def _to_wav(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    model = get_model()
    # language=None 會讓模型自動判斷語言；有指定就照指定的走。
    language = req.language or "Auto"
    # description（音色文字描述）跟 instruct（語氣指示）對這顆模型都是 instruct。
    instruct = " ".join(x for x in (req.description, req.instruct) if x) or None

    try:
        if req.mode == "clone":
            if not req.ref_audio_path or not os.path.exists(req.ref_audio_path):
                raise HTTPException(400, f"參考音檔不存在：{req.ref_audio_path}")
            try:
                wavs, sr = model.generate_voice_clone(
                    text=req.text,
                    language=language,
                    ref_audio=req.ref_audio_path,
                    ref_text=req.ref_text,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    400,
                    "Qwen3-TTS-CustomVoice 這個 checkpoint 不支援上傳音檔克隆"
                    f"（底層錯誤：{e}）。請改用 cosyvoice2 / fun-cosyvoice3 / voxcpm2，"
                    "或改掛 Qwen3-TTS-12Hz-1.7B-Base 重 build。",
                )
        else:
            speaker = req.speaker or DEFAULT_SPEAKER
            supported = model.get_supported_speakers() or []
            if supported and speaker not in supported:
                raise HTTPException(
                    400, f"speaker「{speaker}」不在支援清單，可用的有：{supported}"
                )
            wavs, sr = model.generate_custom_voice(
                text=req.text, language=language, speaker=speaker, instruct=instruct
            )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(500, f"{ENGINE_NAME} 合成失敗：{e}")

    return Response(
        content=_to_wav(wavs[0], sr),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr), "X-Engine": ENGINE_NAME},
    )


@app.post("/warmup")
def warmup():
    m = get_model()
    return {"loaded": True, "presets": m.get_supported_speakers() or []}
