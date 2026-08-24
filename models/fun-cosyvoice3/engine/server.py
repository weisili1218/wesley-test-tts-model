"""
CosyVoice engine HTTP server（CosyVoice2 / Fun-CosyVoice3 共用同一份程式）。

這支只做一件事：把 CosyVoice 的推論介面包成 gateway 認得的統一 HTTP 契約。
所有音色管理（上傳、命名、刪除）都在 gateway，這裡只負責「給我 wav 路徑跟文字，
還你音訊」。

契約：
  GET  /health      → 這顆引擎的能力宣告
  POST /synthesize  → audio/wav

環境變數：
  ENGINE_NAME     引擎代號，會出現在 /health 跟 gateway 的 model 欄位
  MODEL_PATH      模型權重目錄
  COSYVOICE_CLASS CosyVoice2 或 CosyVoice3
  COSYVOICE_FP16  1 = 用 fp16（有 CUDA 時預設開）
"""
import io
import os
import logging

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

ENGINE_NAME = os.environ.get("ENGINE_NAME", "fun-cosyvoice3")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/fun-cosyvoice3")
CLASS_NAME = os.environ.get("COSYVOICE_CLASS", "CosyVoice3")

# CosyVoice 的 prompt 音檔一律吃 16k 單聲道，這是它 frontend 的硬性要求。
PROMPT_SR = 16000
# 對外宣告支援哪些音色來源，gateway 靠這個決定路由。
MODES = [m.strip() for m in os.environ.get("ENGINE_MODES", "clone").split(",") if m.strip()]

app = FastAPI(title=f"{ENGINE_NAME} engine")
_model = None


def get_model():
    """第一次呼叫時才載入模型，讓容器可以先起來、healthcheck 再慢慢等。"""
    global _model
    if _model is None:
        fp16 = os.environ.get("COSYVOICE_FP16", "1") == "1" and torch.cuda.is_available()
        cls = {"CosyVoice2": CosyVoice2, "CosyVoice3": CosyVoice3}[CLASS_NAME]
        log.info("loading %s from %s (fp16=%s)", CLASS_NAME, MODEL_PATH, fp16)
        _model = cls(MODEL_PATH, load_trt=False, load_vllm=False, fp16=fp16)
        log.info("loaded, sample_rate=%d", _model.sample_rate)
    return _model


class SynthRequest(BaseModel):
    text: str
    # CosyVoice 只支援 clone：它的每條推論路徑都要參考音檔。
    #   有逐字稿  → zero-shot（音色最像）
    #   沒逐字稿  → cross-lingual
    #   有 instruct/description → instruct2（風格控制，仍然要參考音檔當底）
    # 純文字描述造音色（design）做不到，那要用 voxcpm2。
    mode: str = "clone"
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
        "impl": CLASS_NAME,
        "model_path": MODEL_PATH,
        "loaded": _model is not None,
        "sample_rate": _model.sample_rate if _model else None,
        # gateway 靠這兩個欄位決定「這顆引擎能不能用這種音色」
        "modes": MODES,
        "needs_ref_audio": True,
        "presets": [],
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    model = get_model()

    if not req.ref_audio_path:
        raise HTTPException(
            400,
            f"{ENGINE_NAME} 是 zero-shot 克隆模型，一定要參考音檔。"
            f"請先用 POST /v1/voices 建立一個 clone 音色再指定它。",
        )
    if not os.path.exists(req.ref_audio_path):
        raise HTTPException(400, f"參考音檔不存在：{req.ref_audio_path}")

    prompt_wav = load_wav(req.ref_audio_path, PROMPT_SR)

    # instruct 跟 description 都是「要模型怎麼講」的自然語言，合成一句丟給 instruct2。
    style = " ".join(x for x in (req.description, req.instruct) if x)

    try:
        if style:
            # 有風格指示 → instruct2（CosyVoice2/3 才有）
            it = model.inference_instruct2(
                req.text, style, prompt_wav, stream=False, speed=req.speed
            )
        elif req.ref_text:
            # 有逐字稿 → zero-shot，音色相似度最好
            it = model.inference_zero_shot(
                req.text, req.ref_text, prompt_wav, stream=False, speed=req.speed
            )
        else:
            # 沒逐字稿 → cross-lingual（不需要逐字稿，但相似度略差）
            it = model.inference_cross_lingual(
                req.text, prompt_wav, stream=False, speed=req.speed
            )
        chunks = [out["tts_speech"] for out in it]
    except Exception as e:  # 讓 gateway 拿到看得懂的錯誤，而不是 500 空白
        log.exception("synthesis failed")
        raise HTTPException(500, f"{ENGINE_NAME} 合成失敗：{e}")

    if not chunks:
        raise HTTPException(500, "模型沒有產出音訊")

    speech = torch.concat(chunks, dim=1)
    buf = io.BytesIO()
    torchaudio.save(buf, speech, model.sample_rate, format="wav")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(model.sample_rate), "X-Engine": ENGINE_NAME},
    )


@app.post("/warmup")
def warmup():
    """讓 compose 起來之後可以主動把模型先載進 GPU，避免第一個請求等 60 秒。"""
    m = get_model()
    return {"loaded": True, "sample_rate": m.sample_rate}
