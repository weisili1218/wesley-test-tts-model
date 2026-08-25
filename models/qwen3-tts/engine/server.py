"""
Qwen3-TTS-12Hz-1.7B engine HTTP server。

這支對兩個 checkpoint 都適用，能力由 ENGINE_MODES 宣告（Dockerfile 設）：

  -Base（目前掛的）      ENGINE_MODES=clone
    mode=clone   → generate_voice_clone(ref_audio, ref_text)，上傳一段參考音檔克隆
    mode=preset  → 400，這個 checkpoint 沒有內建 speaker

  -CustomVoice           ENGINE_MODES=preset
    mode=preset  → generate_custom_voice(speaker, instruct)，9 個內建精選音色
    mode=clone   → 400，這個 checkpoint 不會克隆

兩邊都吃 instruct（語氣／風格指令）。換 checkpoint 只要改 Dockerfile 的
MODEL_REPO + ENGINE_MODES 重 build，這支不用動。

環境變數：
  ENGINE_NAME / MODEL_PATH / QWEN_ATTN / QWEN_DTYPE / QWEN_DEFAULT_SPEAKER
  ENGINE_MODES —— gateway 靠它決定要把哪種音色路由過來，一定要跟 checkpoint 對上
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
# 只有 CustomVoice checkpoint 用得到；Base 沒有內建 speaker，這個值是死的。
DEFAULT_SPEAKER = os.environ.get("QWEN_DEFAULT_SPEAKER", "Vivian")
# 要跟 MODEL_REPO 對上：Base → "clone"，CustomVoice → "preset"。
# gateway 完全照這個宣告來路由，不寫死任何引擎能力。
MODES = [m.strip() for m in os.environ.get("ENGINE_MODES", "clone").split(",") if m.strip()]

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
    # 沒給就用宣告的第一個 mode，這樣直接戳引擎除錯時不用每次帶
    mode: str = MODES[0] if MODES else "clone"
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
        # Base 只會克隆，沒有參考音檔就發不出聲音；CustomVoice 反之
        "needs_ref_audio": "clone" in MODES and "preset" not in MODES,
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
            if "clone" not in MODES:
                raise HTTPException(
                    400,
                    f"這個 checkpoint（{MODEL_PATH}）沒有宣告 clone 能力。"
                    "要上傳音檔克隆請把 Dockerfile 的 MODEL_REPO 換成 "
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base、ENGINE_MODES 設成 clone 重 build，"
                    "或改用 cosyvoice2 / fun-cosyvoice3 / voxcpm2。",
                )
            if not req.ref_audio_path or not os.path.exists(req.ref_audio_path):
                raise HTTPException(400, f"參考音檔不存在：{req.ref_audio_path}")
            # 註：generate_voice_clone 沒有 instruct 參數，所以 clone 模式下
            # 語氣指令（gateway 的 instructions）拿不到，只有 preset 模式吃得到。
            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=language,
                ref_audio=req.ref_audio_path,
                ref_text=req.ref_text,
            )
        else:
            supported = model.get_supported_speakers() or []
            if not supported:
                raise HTTPException(
                    400,
                    f"這個 checkpoint（{MODEL_PATH}）沒有內建音色，不能用 mode=preset。"
                    "內建音色是 Qwen3-TTS-12Hz-1.7B-CustomVoice 才有的；"
                    "目前這包請上傳參考音檔走 clone（POST /v1/voices）。",
                )
            speaker = req.speaker or DEFAULT_SPEAKER
            if speaker not in supported:
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
