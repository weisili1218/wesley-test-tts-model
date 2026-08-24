"""
VoxCPM2 (OpenBMB) engine HTTP server。

VoxCPM2 是四顆裡面能力最全的：三種音色來源都支援。
  mode=clone   → reference_wav_path 克隆；再給逐字稿就升級成 ultimate cloning
                 （相似度最高，官方建議 reference 跟 prompt 給同一段）
  mode=design  → 不用任何音檔，把音色描述寫在文字最前面的括號裡，
                 例如 "(一位溫柔的年輕女性)你好，歡迎使用。"
  mode=default → 什麼都不給，模型自己生一個音色

環境變數：
  ENGINE_NAME / MODEL_PATH / VOXCPM_OPTIMIZE / VOXCPM_CFG / VOXCPM_TIMESTEPS
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

from voxcpm import VoxCPM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

ENGINE_NAME = os.environ.get("ENGINE_NAME", "voxcpm2")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/voxcpm2")
# optimize=True 會做 torch.compile，穩定跑起來之後開它可以明顯加速，
# 但在 aarch64 + Blackwell 上第一次編譯很久、偶爾會炸，所以預設關掉。
OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "0") == "1"
DEFAULT_CFG = float(os.environ.get("VOXCPM_CFG", "2.0"))
DEFAULT_TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))
# 對外宣告支援哪些音色來源，gateway 靠這個決定路由。
MODES = [m.strip() for m in os.environ.get("ENGINE_MODES", "clone,design,default").split(",") if m.strip()]

app = FastAPI(title=f"{ENGINE_NAME} engine")
_model = None


def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading VoxCPM2 from %s (device=%s optimize=%s)", MODEL_PATH, device, OPTIMIZE)
        _model = VoxCPM.from_pretrained(
            MODEL_PATH,
            # 開著的話啟動時會去 ModelScope 抓 ZipEnhancer，離線 image 會卡死
            load_denoiser=False,
            local_files_only=True,
            optimize=OPTIMIZE,
            device=device,
        )
        log.info("loaded, sample_rate=%d", _model.tts_model.sample_rate)
    return _model


class SynthRequest(BaseModel):
    text: str
    mode: str = "clone"
    ref_audio_path: str | None = None
    ref_text: str | None = None
    description: str | None = None
    instruct: str | None = None
    speaker: str | None = None
    language: str | None = None
    speed: float = 1.0
    cfg_value: float | None = None
    inference_timesteps: int | None = None


@app.get("/health")
def health():
    return {
        "engine": ENGINE_NAME,
        "model_path": MODEL_PATH,
        "loaded": _model is not None,
        "sample_rate": _model.tts_model.sample_rate if _model else None,
        "modes": MODES,
        "needs_ref_audio": False,
        "presets": [],
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    model = get_model()

    # VoxCPM2 的風格/音色控制全部走「文字開頭的括號」這個介面，
    # 所以 description(音色描述) 跟 instruct(語氣指示) 都併進 prefix。
    prefix = "".join(f"({x})" for x in (req.description, req.instruct) if x)
    text = prefix + req.text

    kwargs = {
        "text": text,
        "cfg_value": req.cfg_value if req.cfg_value is not None else DEFAULT_CFG,
        "inference_timesteps": (
            req.inference_timesteps if req.inference_timesteps is not None else DEFAULT_TIMESTEPS
        ),
    }

    if req.mode == "clone":
        if not req.ref_audio_path:
            raise HTTPException(400, "mode=clone 需要 ref_audio_path")
        if not os.path.exists(req.ref_audio_path):
            raise HTTPException(400, f"參考音檔不存在：{req.ref_audio_path}")
        kwargs["reference_wav_path"] = req.ref_audio_path
        if req.ref_text:
            # 有逐字稿就走 ultimate cloning：同一段音檔同時當 reference 跟 prompt
            kwargs["prompt_wav_path"] = req.ref_audio_path
            kwargs["prompt_text"] = req.ref_text
    elif req.mode == "design":
        if not prefix:
            raise HTTPException(400, "mode=design 需要 description（音色的文字描述）")
    # mode=default 什麼都不加，模型自己生一個音色

    try:
        wav = model.generate(**kwargs)
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(500, f"{ENGINE_NAME} 合成失敗：{e}")

    sr = model.tts_model.sample_rate
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr), "X-Engine": ENGINE_NAME},
    )


@app.post("/warmup")
def warmup():
    m = get_model()
    return {"loaded": True, "sample_rate": m.tts_model.sample_rate}
