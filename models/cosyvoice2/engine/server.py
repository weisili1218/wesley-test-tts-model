"""
CosyVoice engine HTTP server（CosyVoice2 / Fun-CosyVoice3 共用同一份程式）。

這支只做一件事：把 CosyVoice 的推論介面包成 gateway 認得的統一 HTTP 契約。
所有音色管理（上傳、命名、刪除）都在 gateway，這裡只負責「給我 wav 路徑跟文字，
還你音訊」。

契約：
  GET  /health           → 這顆引擎的能力宣告
  POST /synthesize       → audio/wav
  POST /voices/prepare   → 先把一個音色的參考特徵抽好放著（gateway 建音色時打）
  POST /voices/forget    → 把它從快取拿掉（gateway 刪音色 / 改逐字稿時打）
  POST /warmup           → 載模型；帶參考音檔的話順便建快取 + 合成一句

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
  COSYVOICE_VOICE_CACHE_SIZE
                  參考音檔特徵快取要放幾份，預設 32，設 0 關掉
"""
import hashlib
import io
import os
import logging
import threading
from collections import OrderedDict

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


# ---------------------------------------------------------------------------
# 參考音檔特徵快取
# ---------------------------------------------------------------------------
# 每合成一句，上游的 frontend 都會把參考音檔重抽一輪特徵：
#     _extract_speech_token   whisper mel + speech_tokenizer_v3 ONNX（跑 CPU）
#     _extract_spk_embedding  kaldi fbank + campplus ONNX（跑 CPU）
#     _extract_speech_feat    24k mel
# 同一個音色的這三樣每次都一模一樣，而長文會被斷成好幾句、批次會跑好幾百列 ——
# 等於同樣的東西算了好幾百遍，而且是在 CPU 上。
#
# 上游給了正規的解法：add_zero_shot_spk() 把上面那包 model_input 存進
# frontend.spk2info[spk_id]，之後 inference 帶 zero_shot_spk_id 就直接取用。
#
# ⚠ key 一定要含 prompt_text，這是這段唯一的陷阱：
#   frontend_zero_shot 在 zero_shot_spk_id != '' 時會走
#       model_input = {**self.spk2info[zero_shot_spk_id]}
#   —— 傳進去的 prompt_text 參數**被完全忽略**。而三條路徑的 prompt_text 不一樣：
#       zero_shot      正規化後的逐字稿
#       cross_lingual  ''（frontend_cross_lingual 寫死）
#       instruct2      instruct 字串本身
#   如果只用參考音檔當 key，帶 instructions 的請求就會拿到逐字稿那份快取，
#   語氣指令被靜靜吃掉、聽起來像沒生效。所以 (音檔, prompt_text) 一起當 key。
#
# 快取內容是 CUDA 張量，一份約 1 MB 上下（30 秒音檔的 24k mel 是大宗），
# 32 份約 30 MB —— 在 unified memory 上這點量無所謂，但還是用 LRU 收著。
VOICE_CACHE_SIZE = int(os.environ.get("COSYVOICE_VOICE_CACHE_SIZE", "32"))
# spk_id -> (ref_audio_path, invalidation_key)。OrderedDict 當 LRU 用。
_voice_cache: OrderedDict[str, tuple] = OrderedDict()
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0}


def _norm_prompt(model, text: str) -> str:
    """inference_zero_shot 會先正規化 prompt_text 再送進 frontend，快取要對齊那個結果。"""
    try:
        return model.frontend.text_normalize(text, split=False, text_frontend=True)
    except Exception:
        return text


def _cached_spk_id(model, ref_audio_path: str, prompt_text: str) -> str | None:
    """回傳可以拿去當 zero_shot_spk_id 的字串；建不起來就回 None。

    回 None 的話呼叫端傳 '' 進去，就是原本每句重抽的老路，結果完全一樣、只是慢。
    """
    if VOICE_CACHE_SIZE <= 0:
        return None
    try:
        st = os.stat(ref_audio_path)
    except OSError:
        return None
    # mtime/size 進 key：gateway 覆蓋同一個 voice_xxx.wav 時快取會自動失效
    key = (st.st_mtime_ns, st.st_size)
    spk_id = "cache-" + hashlib.sha1(
        f"{ref_audio_path}\0{prompt_text}".encode("utf-8")
    ).hexdigest()[:16]

    with _cache_lock:
        entry = _voice_cache.get(spk_id)
        if entry and entry[1] == key and spk_id in model.frontend.spk2info:
            _voice_cache.move_to_end(spk_id)
            _cache_stats["hits"] += 1
            return spk_id

        _cache_stats["misses"] += 1
        try:
            # 內部就是 frontend_zero_shot('', prompt_text, wav, sample_rate, '')，
            # 也就是跟沒快取時完全同一條抽特徵的路徑（含 24k 的 feat/token 對齊）。
            model.add_zero_shot_spk(prompt_text, ref_audio_path, spk_id)
        except Exception as e:
            log.warning("音色特徵快取建不起來（%s），這次改走每句重抽的路徑", e)
            model.frontend.spk2info.pop(spk_id, None)
            _voice_cache.pop(spk_id, None)
            return None

        _voice_cache[spk_id] = (ref_audio_path, key)
        _voice_cache.move_to_end(spk_id)
        while len(_voice_cache) > VOICE_CACHE_SIZE:
            old_id, _ = _voice_cache.popitem(last=False)
            model.frontend.spk2info.pop(old_id, None)
    return spk_id


def _forget_voice(ref_audio_path: str) -> int:
    """把某個參考音檔的所有快取條目丟掉（一個音色可能有逐字稿版跟 instruct 版好幾份）。"""
    dropped = 0
    with _cache_lock:
        for spk_id, (path, _key) in list(_voice_cache.items()):
            if path == ref_audio_path:
                _voice_cache.pop(spk_id, None)
                if _model is not None:
                    _model.frontend.spk2info.pop(spk_id, None)
                dropped += 1
    return dropped


def _cache_info() -> dict:
    return {"size": len(_voice_cache), "capacity": VOICE_CACHE_SIZE, **_cache_stats}


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
        # gateway 靠這個欄位知道這顆引擎收 /voices/prepare
        "supports_voice_prepare": True,
        "voice_cache": _cache_info(),
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

    # 每條路徑各自算出「它會餵給 frontend_zero_shot 的 prompt_text」，拿去換一個
    # zero_shot_spk_id。換不到就傳 ''，等於走原本每句重抽特徵的路，結果一模一樣。
    try:
        if style:
            # 有風格指示 → instruct2（CosyVoice2/3 才有）
            instruct = f"{SYSTEM_PROMPT} {style}{EOP}" if IS_CV3 else style
            # instruct2 的 prompt_text 就是 instruct 本身，而且上游不會正規化它。
            # 所以快取是綁 (音檔, 這句 instruct)：批次 CSV 整欄同樣的語氣指令會命中，
            # 換一句 instruct 就是另一份快取 —— 這正是要的，不會互相污染。
            spk = _cached_spk_id(model, prompt_wav, instruct)
            it = model.inference_instruct2(
                req.text, instruct, prompt_wav, zero_shot_spk_id=spk or "",
                stream=False, speed=req.speed,
            )
            chunks = [out["tts_speech"] for out in it]
        elif req.ref_text:
            # 有逐字稿 → zero-shot，音色相似度最好
            ref_text = f"{SYSTEM_PROMPT}{EOP}{req.ref_text}" if IS_CV3 else req.ref_text
            spk = _cached_spk_id(model, prompt_wav, _norm_prompt(model, ref_text))
            it = model.inference_zero_shot(
                req.text, ref_text, prompt_wav, zero_shot_spk_id=spk or "",
                stream=False, speed=req.speed,
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
            # cross_lingual 的 prompt_text 被 frontend_cross_lingual 寫死成 ''，
            # 所以快取 key 的 prompt_text 也是 ''。逐句迴圈更是快取的重點客戶：
            # 沒快取的話 N 句就抽 N 次特徵。
            spk = _cached_spk_id(model, prompt_wav, "") or ""
            chunks = [
                out["tts_speech"]
                for s in sentences
                for out in model.inference_cross_lingual(
                    f"{SYSTEM_PROMPT}{EOP}{s}", prompt_wav, zero_shot_spk_id=spk,
                    stream=False, speed=req.speed,
                )
            ]
        else:
            # CosyVoice2 的 cross-lingual：不注入前綴，斷句交給 inference 自己做
            it = model.inference_cross_lingual(
                req.text, prompt_wav, zero_shot_spk_id=_cached_spk_id(model, prompt_wav, "") or "",
                stream=False, speed=req.speed,
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


# ---------------------------------------------------------------------------
# 音色特徵：先建好放著
# ---------------------------------------------------------------------------
class PrepareRequest(BaseModel):
    ref_audio_path: str
    ref_text: str | None = None
    # 預設不為了建快取去載模型 —— 那要 30-90 秒，會讓 ./voice.sh add 莫名其妙卡住。
    # 沒載模型就直接說「沒建」，反正第一次合成時會自己建，只是慢那一次。
    load_model: bool = False


@app.post("/voices/prepare")
def prepare_voice(req: PrepareRequest):
    """把一個音色的參考特徵先抽好存進快取，之後合成就不用再抽。

    gateway 在 POST /v1/voices 建立音色時會打這支。回 200 但 prepared=false 是
    正常結果（例如引擎還沒載模型），不是錯誤 —— 這只是先做而已，沒做也不影響正確性。
    """
    if not os.path.exists(req.ref_audio_path):
        raise HTTPException(400, f"參考音檔不存在：{req.ref_audio_path}")
    if _model is None and not req.load_model:
        return {"prepared": False,
                "reason": "引擎還沒載模型。先打 POST /v1/warmup 再建音色就會直接建好；"
                          "不打也可以，第一次合成時會自己建。",
                "voice_cache": _cache_info()}

    model = get_model()
    # 建哪一份要跟這個音色實際會走的路徑對上：有逐字稿走 zero-shot，沒有走
    # cross-lingual，兩者的 prompt_text 不同（見 _cached_spk_id 的說明）。
    if req.ref_text:
        ref_text = f"{SYSTEM_PROMPT}{EOP}{req.ref_text}" if IS_CV3 else req.ref_text
        prompt_text = _norm_prompt(model, ref_text)
        path_name = "zero_shot"
    else:
        prompt_text = ""
        path_name = "cross_lingual"

    spk_id = _cached_spk_id(model, req.ref_audio_path, prompt_text)
    return {
        "prepared": spk_id is not None,
        "path": path_name,
        "reason": None if spk_id else "抽特徵失敗，合成時會再試一次",
        "voice_cache": _cache_info(),
        # 帶 instructions 的合成走 instruct2，prompt_text 是那句 instruct 而不是
        # 逐字稿，所以那份快取只能等第一次用到時才建 —— 這裡先建不了。
        "note": "instructions 的快取是按 instruct 內容分開存的，第一次用到才會建",
    }


@app.post("/voices/forget")
def forget_voice(req: PrepareRequest):
    """把這個參考音檔的所有快取條目丟掉。gateway 刪音色 / 改逐字稿時打。"""
    dropped = _forget_voice(req.ref_audio_path)
    return {"dropped": dropped, "voice_cache": _cache_info()}


@app.post("/warmup")
def warmup(req: PrepareRequest | None = None):
    """讓 compose 起來之後可以主動把模型先載進 GPU，避免第一個請求等 60 秒。

    gateway 會把音色庫裡最近建立的 clone 音色一起帶進來；有帶的話就順便把它的
    特徵建好、再真的合成一句短的丟掉，把 CUDA kernel 的首次配置也一起付掉。
    """
    m = get_model()
    warmed = False
    if req and req.ref_audio_path and os.path.exists(req.ref_audio_path):
        try:
            if req.ref_text:
                ref_text = f"{SYSTEM_PROMPT}{EOP}{req.ref_text}" if IS_CV3 else req.ref_text
                spk = _cached_spk_id(m, req.ref_audio_path, _norm_prompt(m, ref_text)) or ""
                list(m.inference_zero_shot("你好。", ref_text, req.ref_audio_path,
                                           zero_shot_spk_id=spk, stream=False))
            else:
                spk = _cached_spk_id(m, req.ref_audio_path, "") or ""
                text = f"{SYSTEM_PROMPT}{EOP}你好。" if IS_CV3 else "你好。"
                list(m.inference_cross_lingual(text, req.ref_audio_path,
                                               zero_shot_spk_id=spk, stream=False))
            warmed = True
        except Exception as e:
            # 暖機失敗不算錯，頂多第一句慢一點，不該讓 gateway 的 warmup 掛掉
            log.warning("暖機合成失敗（%s），只載了權重", e)
    return {"loaded": True, "warmed": warmed, "sample_rate": m.sample_rate,
            "voice_cache": _cache_info()}
