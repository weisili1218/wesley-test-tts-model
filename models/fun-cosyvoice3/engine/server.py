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
  COSYVOICE_MAX_REF_SEC
                  參考音檔的硬上限秒數，預設 30（上游 frontend 的 assert）
  COSYVOICE3_SYSTEM_PROMPT
                  只有 CosyVoice3 用得到。接在 <|endofprompt|> 前面的 system prompt，
                  預設跟官方 model card 一致
"""
import io
import os
import logging
import threading

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

ENGINE_NAME = os.environ.get("ENGINE_NAME", "fun-cosyvoice3")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/fun-cosyvoice3")
CLASS_NAME = os.environ.get("COSYVOICE_CLASS", "CosyVoice3")

# 對外宣告支援哪些音色來源，gateway 靠這個決定路由。
MODES = [m.strip() for m in os.environ.get("ENGINE_MODES", "clone").split(",") if m.strip()]

# 參考音檔的硬上限，gateway 靠這個在建立音色時就擋掉過長的檔案。
# 來源是 frontend._extract_speech_token 的
#   assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
# 它不會自己截斷，超過就是 AssertionError —— 而那個錯誤會在合成時才出現，
# 使用者只會看到音色建得起來、但每一次合成都失敗。
MAX_REF_SEC = float(os.environ.get("COSYVOICE_MAX_REF_SEC", "30"))

# CosyVoice3 的 LLM 硬性要求輸入裡有 <|endofprompt|>（token 151646）：
#   cosyvoice/llm/llm.py:479   assert 151646 in text
# 沒有的話 AssertionError 會在 llm_job 那條執行緒炸掉（cli/model.py:121），主執行緒
# p.join() 之後拿到空的 speech token list，flow 產出 0 幀 mel，最後表現成 HiFi-GAN 的
#   RuntimeError: Calculated padded input size per channel: (3). Kernel size: (4).
# 那個 3 是 f0_predictor 第一層 CausalConv1d(kernel=4) 的 causal padding
# （transformer/convolution.py:177），不是 mel 長度 —— mel 長度是 0，所以不管輸入
# 幾個字都會看到同一個 3。
#
# 前綴要掛在哪個參數上，三條路徑各不相同，照官方 model card 的 Basic Usage：
#   zero_shot      → prompt_text 前面
#   cross_lingual  → tts_text 前面（frontend 會把 prompt_text 整個刪掉）
#   instruct2      → instruct_text 結尾
#
# CosyVoice2 走的是 Qwen2LM，沒有這個檢查，也沒有用這種輸入訓練過，注入只會多出
# 無意義的 text token，所以要用 COSYVOICE_CLASS 擋住。
EOP = "<|endofprompt|>"
SYSTEM_PROMPT = os.environ.get("COSYVOICE3_SYSTEM_PROMPT", "You are a helpful assistant.")
IS_CV3 = CLASS_NAME == "CosyVoice3"

app = FastAPI(title=f"{ENGINE_NAME} engine")
_model = None
# /synthesize 跟 /warmup 都是 sync def，FastAPI 會把它們丟到 threadpool 跑，兩者
# 可以真的同時執行 —— gateway 那個 per-engine semaphore 只擋 /synthesize，warmup
# 是直接打的。沒有這把鎖的話，「先 warmup 再合成」如果撞在一起，兩條 thread 會
# 同時看到 _model is None 而各載一份權重。GB10 是 unified memory，多出來的那份
# 跟整台機器搶同一池，這正是最容易 OOM 的路徑。
_model_lock = threading.Lock()


def get_model():
    """第一次呼叫時才載入模型，讓容器可以先起來、healthcheck 再慢慢等。"""
    global _model
    if _model is not None:          # 已載入就不必進鎖，合成路徑不會被序列化
        return _model
    with _model_lock:
        if _model is None:          # double-checked：等到鎖的那條可能已經被載好了
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
        "max_ref_sec": MAX_REF_SEC,
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

    # 直接給路徑，不要先 load_wav 成張量。這個版本的 frontend_zero_shot 會把
    # prompt_wav 原封不動丟進 load_wav()（frontend.py:121），也就是它自己要負責
    # 讀檔並重取樣到 24k。傳張量進去會變成 torchaudio.load(tensor)，torchcodec
    # 只吃 uint8 的編碼位元組，float 張量會 video_tensor must be kUInt8。
    # 舊版 CosyVoice 的 prompt_speech_16k 收張量，這支是照舊版 API 寫的。
    prompt_wav = req.ref_audio_path

    # instruct 跟 description 都是「要模型怎麼講」的自然語言，合成一句丟給 instruct2。
    style = " ".join(x for x in (req.description, req.instruct) if x)

    try:
        if style:
            # 有風格指示 → instruct2（CosyVoice2/3 才有）
            instruct = f"{SYSTEM_PROMPT} {style}{EOP}" if IS_CV3 else style
            it = model.inference_instruct2(
                req.text, instruct, prompt_wav, stream=False, speed=req.speed
            )
            chunks = [out["tts_speech"] for out in it]
        elif req.ref_text:
            # 有逐字稿 → zero-shot，音色相似度最好
            ref_text = f"{SYSTEM_PROMPT}{EOP}{req.ref_text}" if IS_CV3 else req.ref_text
            it = model.inference_zero_shot(
                req.text, ref_text, prompt_wav, stream=False, speed=req.speed
            )
            chunks = [out["tts_speech"] for out in it]
        elif IS_CV3:
            # 沒逐字稿 → cross-lingual（不需要逐字稿，但相似度略差）。
            # 這條路徑的前綴只能掛在 tts_text 上，而 text_normalize 看到 <| |> 就會
            # 整段跳過正規化「與斷句」（cli/frontend.py:130），長文會變成一整段丟給
            # LLM，跟 zero-shot 路徑行為不一致。所以先用 frontend 自己的斷句切好，
            # 再逐句掛前綴。切完的句子已含 <| |>，inference_cross_lingual 內部那次
            # text_normalize 會原樣返回，不會重複處理。
            sentences = model.frontend.text_normalize(
                req.text, split=True, text_frontend=True
            ) or [req.text]
            chunks = [
                out["tts_speech"]
                for s in sentences
                for out in model.inference_cross_lingual(
                    f"{SYSTEM_PROMPT}{EOP}{s}", prompt_wav, stream=False, speed=req.speed
                )
            ]
        else:
            # CosyVoice2 的 cross-lingual：不注入前綴，斷句交給 inference 自己做
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
    # 不能用 torchaudio.save：2.11 把 save 委派給 torchcodec 之後，format 參數被
    # 忽略、改由副檔名決定，寫進沒有副檔名的 BytesIO 會 Couldn't allocate
    # AVFormatContext。soundfile 支援 file-like 物件，qwen3-tts / voxcpm2 也是這樣寫。
    sf.write(
        buf,
        speech.squeeze(0).cpu().numpy(),
        model.sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
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
