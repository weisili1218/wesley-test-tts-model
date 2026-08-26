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

--- 生成參數 ---------------------------------------------------------------

qwen-tts 的每個 generate_* 除了自己文件列的參數，還吃 HF Transformers
model.generate 的 kwargs。不給的話會沿用 checkpoint generation_config.json
的預設（temperature 0.9 / top_k 50 / top_p 1.0 / repetition_penalty 1.05）
—— 0.9 對自回歸 TTS 偏高，破音、吞字、整段重複唸都是這個溫度來的。

這裡用三檔 profile 取代那組預設（QWEN_PROFILE，預設 balanced），profile 除了
取樣參數還帶「斷句積極度」和「重試次數」，所以三檔的速度差異是真的：

              temp  top_p top_k rep_pen  斷句上限  重試
  fast        0.70  0.85   40    1.10      60      0
  balanced    0.80  0.90   50    1.10     120      1
  quality     0.85  0.95   50    1.05     200      2

單項可以用 QWEN_TEMPERATURE / QWEN_TOP_P / QWEN_TOP_K /
QWEN_REPETITION_PENALTY / QWEN_SPLIT_MAX_CHARS / QWEN_RETRIES 蓋掉 profile，
每個請求也可以自己帶（見 SynthRequest）。改 profile 只要重啟容器，不用重 build。

另外三件事：
  * 參考音檔特徵有 LRU 快取（create_voice_clone_prompt），同一個音色連續合成
    不用每次重抽 speech token + x-vector —— 這是最大的提速項。
  * 長文自動斷句後 batch 推論，避免長序列自回歸漂移，也順便加速。
  * 產出明顯不對（吞字／鬼打牆）時換 seed 重生，回應帶 X-Retries。

環境變數：
  ENGINE_NAME / MODEL_PATH / QWEN_ATTN / QWEN_DTYPE / QWEN_DEFAULT_SPEAKER
  ENGINE_MODES —— gateway 靠它決定要把哪種音色路由過來，一定要跟 checkpoint 對上
  QWEN_PROFILE / QWEN_TEMPERATURE / QWEN_TOP_P / QWEN_TOP_K /
  QWEN_REPETITION_PENALTY / QWEN_SPLIT_MAX_CHARS / QWEN_RETRIES / QWEN_SEED
  QWEN_MAX_NEW_TOKENS_CAP / QWEN_PROMPT_CACHE_SIZE / QWEN_MAX_BATCH /
  QWEN_JOIN_SILENCE_MS
"""
import inspect
import io
import os
import logging
import random
from collections import OrderedDict

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

# ---------------------------------------------------------------------------
# 參數 profile
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    "fast": {
        "temperature": 0.70, "top_p": 0.85, "top_k": 40, "repetition_penalty": 1.10,
        "split_max_chars": 60, "retries": 0,
    },
    "balanced": {
        "temperature": 0.80, "top_p": 0.90, "top_k": 50, "repetition_penalty": 1.10,
        "split_max_chars": 120, "retries": 1,
    },
    "quality": {
        "temperature": 0.85, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05,
        "split_max_chars": 200, "retries": 2,
    },
}
DEFAULT_PROFILE = os.environ.get("QWEN_PROFILE", "balanced").strip().lower()
if DEFAULT_PROFILE not in PROFILES:
    log.warning("QWEN_PROFILE=%s 不認得，改用 balanced。可用：%s",
                DEFAULT_PROFILE, sorted(PROFILES))
    DEFAULT_PROFILE = "balanced"


def _env_num(name: str, cast):
    """環境變數沒設就回 None —— 要能分辨「沒設」和「設成 0」。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError:
        log.warning("%s=%r 不是合法數值，忽略", name, raw)
        return None


# profile 之上的環境變數覆蓋。None = 不覆蓋，照 profile 走。
ENV_OVERRIDES = {
    "temperature": _env_num("QWEN_TEMPERATURE", float),
    "top_p": _env_num("QWEN_TOP_P", float),
    "top_k": _env_num("QWEN_TOP_K", int),
    "repetition_penalty": _env_num("QWEN_REPETITION_PENALTY", float),
    "split_max_chars": _env_num("QWEN_SPLIT_MAX_CHARS", int),
    "retries": _env_num("QWEN_RETRIES", int),
}
DEFAULT_SEED = _env_num("QWEN_SEED", int)

# 12Hz tokenizer → 12 個 token ≈ 1 秒音訊。上限是「鬼打牆時幾秒內斷掉」的保險，
# 正常情況模型碰到 EOS 就停，這個值不會讓它變快。2048 是官方評測用的值。
CODEC_HZ = 12
MAX_NEW_TOKENS_CAP = _env_num("QWEN_MAX_NEW_TOKENS_CAP", int) or 2048
PROMPT_CACHE_SIZE = _env_num("QWEN_PROMPT_CACHE_SIZE", int) or 32
# compose.yaml 已經寫明 mem_limit 擋不住 GPU 側的 unified memory，batch 開太大
# 會把整台吃掉。段落多於這個數就分批送。
MAX_BATCH = _env_num("QWEN_MAX_BATCH", int) or 8
JOIN_SILENCE_MS = _env_num("QWEN_JOIN_SILENCE_MS", int)
if JOIN_SILENCE_MS is None:
    JOIN_SILENCE_MS = 120

app = FastAPI(title=f"{ENGINE_NAME} engine")
_model = None
_presets: list[str] = []
# 參考音檔特徵快取。key 帶 mtime/size，gateway 覆蓋同一個 voice_xxx.wav 時會自動失效。
_prompt_cache: OrderedDict = OrderedDict()
_cache_stats = {"hits": 0, "misses": 0}
# qwen-tts 0.1.1 有沒有 create_voice_clone_prompt，第一次用到時才探測
_clone_prompt_ok: bool | None = None
# generate_* 吃不吃 HF generate kwargs，第一次被打槍後就記住，不要每次都試
_gen_kwargs_ok = True


def get_model():
    global _model, _presets
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("loading Qwen3-TTS from %s (device=%s attn=%s dtype=%s)",
                 MODEL_PATH, device, ATTN, DTYPE)
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, device_map=device, dtype=DTYPE, attn_implementation=ATTN
        )
        _presets = _model.get_supported_speakers() or []
        log.info("loaded, profile=%s params=%s presets=%s",
                 DEFAULT_PROFILE, resolve_params(None, {}), _presets)
    return _model


# ---------------------------------------------------------------------------
# 參數解析：profile → 環境變數 → 單一請求，後面蓋前面
# ---------------------------------------------------------------------------
def resolve_params(profile: str | None, overrides: dict) -> dict:
    name = (profile or DEFAULT_PROFILE).strip().lower()
    if name not in PROFILES:
        raise HTTPException(400, f"沒有這個 profile：{name}。可用的有：{sorted(PROFILES)}")
    params = dict(PROFILES[name])
    for key, val in ENV_OVERRIDES.items():
        if val is not None:
            params[key] = val
    for key, val in overrides.items():
        if val is not None:
            params[key] = val
    params["profile"] = name
    params["retries"] = max(0, int(params["retries"]))
    params["split_max_chars"] = max(20, int(params["split_max_chars"]))
    return params


def _gen_kwargs(params: dict, texts: list[str]) -> dict:
    """params 裡只有這幾個是要傳給 model.generate 的，其他是這支自己用的。"""
    # batch 裡最長的那段決定 token 上限 —— 短的碰到 EOS 自己會停。
    longest = max(texts, key=len) if texts else ""
    return {
        "temperature": float(params["temperature"]),
        "top_p": float(params["top_p"]),
        "top_k": int(params["top_k"]),
        "repetition_penalty": float(params["repetition_penalty"]),
        "max_new_tokens": params.get("max_new_tokens") or token_budget(longest),
        "do_sample": True,
    }


# ---------------------------------------------------------------------------
# 文字長度 → 音訊秒數 / token 預算
# ---------------------------------------------------------------------------
def est_seconds(text: str) -> float:
    """純唸稿的估計秒數（不含任何餘裕），中文約 3.5 字/秒、拉丁字母約 12 字元/秒。"""
    cjk = sum(
        1 for c in text
        if "㐀" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힯"
    )
    return cjk / 3.5 + (len(text) - cjk) / 12.0


def token_budget(text: str) -> int:
    """估計秒數 + 1.5 秒頭尾餘裕，再乘 2.2 的安全係數換成 token 數。"""
    return max(128, min(MAX_NEW_TOKENS_CAP, int((est_seconds(text) + 1.5) * CODEC_HZ * 2.2)))


# ---------------------------------------------------------------------------
# 斷句
# ---------------------------------------------------------------------------
_SENT_END = "。！？；!?;\n"
_CLAUSE_END = "，,、：:"


def split_text(text: str, max_chars: int) -> list[str]:
    """切成不超過 max_chars 的段落。短文字原封不動回傳，行為跟以前一樣。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # 1. 先在句末標點切開（標點留在前一段）
    sentences, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _SENT_END:
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    # 2. 單句本身就超長 → 在逗號切；連逗號都沒有（長串數字、英文）就硬切
    pieces: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        sub, buf = [], ""
        for ch in sent:
            buf += ch
            if ch in _CLAUSE_END and len(buf) >= max_chars * 0.6:
                sub.append(buf)
                buf = ""
        if buf:
            sub.append(buf)
        for part in sub:
            while len(part) > max_chars:
                pieces.append(part[:max_chars])
                part = part[max_chars:]
            if part:
                pieces.append(part)

    # 3. 貪婪合併回 max_chars —— 段落越少接縫越少，語調越連貫
    out, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > max_chars:
            out.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        out.append(cur)
    return out or [text]


# ---------------------------------------------------------------------------
# 參考音檔特徵快取
# ---------------------------------------------------------------------------
def _clone_prompt_supported(model) -> bool:
    """qwen-tts 釘在 0.1.1，官方 README 是 main 分支的，不保證有這支 API。"""
    global _clone_prompt_ok
    if _clone_prompt_ok is None:
        ok = hasattr(model, "create_voice_clone_prompt")
        if ok:
            try:
                params = inspect.signature(model.generate_voice_clone).parameters
                ok = "voice_clone_prompt" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                ok = False
        _clone_prompt_ok = ok
        log.info("參考音檔特徵快取：%s", "可用" if ok else
                 "這版 qwen-tts 沒有 create_voice_clone_prompt，每次都會重抽特徵")
    return _clone_prompt_ok


def get_clone_prompt(model, ref_audio_path: str, ref_text: str | None):
    """回傳可重複使用的 clone prompt；不支援就回 None，呼叫端退回原本的路徑。"""
    if not _clone_prompt_supported(model):
        return None
    st = os.stat(ref_audio_path)
    key = (ref_audio_path, st.st_mtime_ns, st.st_size, ref_text or "")
    if key in _prompt_cache:
        _prompt_cache.move_to_end(key)
        _cache_stats["hits"] += 1
        return _prompt_cache[key]

    _cache_stats["misses"] += 1
    try:
        items = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path, ref_text=ref_text, x_vector_only_mode=False
        )
    except Exception as e:
        # 抽特徵失敗不該讓整個請求掛掉 —— 退回讓 generate 自己處理參考音檔
        global _clone_prompt_ok
        _clone_prompt_ok = False
        log.warning("create_voice_clone_prompt 失敗（%s），之後一律走原本的路徑", e)
        return None

    _prompt_cache[key] = items
    while len(_prompt_cache) > PROMPT_CACHE_SIZE:
        _prompt_cache.popitem(last=False)
    return items


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def _call_generate(fn, call: dict, kwargs: dict):
    """帶取樣參數呼叫；這版 qwen-tts 不吃的話退回不帶，並記住不要再試。"""
    global _gen_kwargs_ok
    if _gen_kwargs_ok and kwargs:
        try:
            return fn(**call, **kwargs)
        except TypeError as e:
            _gen_kwargs_ok = False
            log.warning("%s 不吃這組 generate kwargs（%s），"
                        "之後一律用 checkpoint 的預設取樣參數", fn.__name__, e)
    return fn(**call)


def _seed(value: int | None) -> int:
    """回傳這次實際用的 seed，讓呼叫端重試時可以往上加。"""
    used = value if value is not None else random.randrange(2**31)
    torch.manual_seed(used)
    return used


def _looks_broken(wav: np.ndarray, sr: int, text: str, max_new_tokens: int) -> str | None:
    """吞字／鬼打牆的偵測。沒問題回 None，有問題回原因字串。"""
    got = len(wav) / float(sr) if sr else 0.0
    # 生到 token 上限還沒 EOS —— 12 token ≈ 1 秒，貼著上限就是沒正常收尾
    if got >= (max_new_tokens / CODEC_HZ) * 0.95:
        return f"生到 token 上限還沒收尾（{got:.1f}s）"
    est = est_seconds(text)
    if est < 1.0:
        # 太短的句子估不準（一個「好」跟一句「好的」差很多），不判定
        return None
    ratio = got / est
    if ratio < 0.4:
        return f"音訊只有 {got:.1f}s，估計要 {est:.1f}s，像是吞字"
    if ratio > 2.5:
        return f"音訊長達 {got:.1f}s，估計只要 {est:.1f}s，像是重複唸"
    return None


def _synth_clone(model, segments: list[str], language: str, ref_audio_path: str,
                 ref_text: str | None, params: dict) -> tuple[list[np.ndarray], int, int]:
    """把 segments 一次 batch 生完，壞掉的段落換 seed 單獨重生。"""
    prompt_items = get_clone_prompt(model, ref_audio_path, ref_text)
    base_seed = params.get("seed", DEFAULT_SEED)

    def call_for(texts: list[str]) -> dict:
        call = {"text": texts, "language": [language] * len(texts)}
        if prompt_items is not None:
            call["voice_clone_prompt"] = prompt_items
        else:
            call["ref_audio"] = ref_audio_path
            call["ref_text"] = ref_text
        return call

    wavs: list[np.ndarray] = []
    sr = 0
    # 分批：段落多於 MAX_BATCH 就切開送，避免一次把 unified memory 吃光
    for start in range(0, len(segments), MAX_BATCH):
        chunk = segments[start:start + MAX_BATCH]
        kwargs = _gen_kwargs(params, chunk)
        _seed(base_seed)
        out, sr = _call_generate(model.generate_voice_clone, call_for(chunk), kwargs)
        wavs.extend(np.asarray(w) for w in out[:len(chunk)])

    # 逐段驗收，壞的換 seed 重生（單段送，不影響其他段）
    retries_used = 0
    for i, text in enumerate(segments):
        if retries_used >= params["retries"]:
            break
        for attempt in range(params["retries"] - retries_used):
            why = _looks_broken(wavs[i], sr, text, _gen_kwargs(params, [text])["max_new_tokens"])
            if why is None:
                break
            retries_used += 1
            log.warning("第 %d 段重生（第 %d 次）：%s", i + 1, retries_used, why)
            kwargs = _gen_kwargs(params, [text])
            _seed((base_seed if base_seed is not None else 0) + retries_used)
            out, sr = _call_generate(model.generate_voice_clone, call_for([text]), kwargs)
            wavs[i] = np.asarray(out[0])

    return wavs, sr, retries_used


def _join(wavs: list[np.ndarray], sr: int) -> np.ndarray:
    """段落之間插一小段靜音再串起來，接縫才不會擠在一起。"""
    if len(wavs) == 1:
        return wavs[0]
    gap = np.zeros(int(sr * JOIN_SILENCE_MS / 1000.0), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, w in enumerate(wavs):
        if i:
            out.append(gap)
        out.append(np.asarray(w, dtype=np.float32))
    return np.concatenate(out)


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
    # --- 生成參數。全部選填，沒給就照 QWEN_PROFILE 那一檔走 ---
    profile: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    split_max_chars: int | None = None
    retries: int | None = None
    seed: int | None = None


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
        # 沒有參考音檔長度上限；gateway 靠這個欄位決定要不要在建立音色時擋。
        "max_ref_sec": None,
        "presets": _presets,
        # 調參時用這段確認容器現在到底吃的是哪一組值，不用翻 log
        "profile": DEFAULT_PROFILE,
        "profiles": sorted(PROFILES),
        "params": resolve_params(None, {}),
        "prompt_cache": {
            "supported": _clone_prompt_ok,
            "size": len(_prompt_cache),
            "capacity": PROMPT_CACHE_SIZE,
            **_cache_stats,
        },
        "max_batch": MAX_BATCH,
        "max_new_tokens_cap": MAX_NEW_TOKENS_CAP,
        "generate_kwargs_supported": _gen_kwargs_ok,
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

    params = resolve_params(req.profile, {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
        "split_max_chars": req.split_max_chars,
        "retries": req.retries,
        "seed": req.seed,
    })
    retries_used = 0

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
            segments = split_text(req.text, params["split_max_chars"])
            if len(segments) > 1:
                log.info("長文斷成 %d 段（上限 %d 字），batch %d 筆一送",
                         len(segments), params["split_max_chars"], MAX_BATCH)
            wav_list, sr, retries_used = _synth_clone(
                model, segments, language, req.ref_audio_path, req.ref_text, params
            )
            wav = _join(wav_list, sr)
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
            segments = split_text(req.text, params["split_max_chars"])
            wav_list: list[np.ndarray] = []
            sr = 0
            for start in range(0, len(segments), MAX_BATCH):
                chunk = segments[start:start + MAX_BATCH]
                _seed(params.get("seed", DEFAULT_SEED))
                out, sr = _call_generate(
                    model.generate_custom_voice,
                    {"text": chunk, "language": [language] * len(chunk),
                     "speaker": [speaker] * len(chunk),
                     "instruct": [instruct] * len(chunk) if instruct else None},
                    _gen_kwargs(params, chunk),
                )
                wav_list.extend(np.asarray(w) for w in out[:len(chunk)])
            wav = _join(wav_list, sr)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(500, f"{ENGINE_NAME} 合成失敗：{e}")

    return Response(
        content=_to_wav(wav, sr),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sr),
            "X-Engine": ENGINE_NAME,
            "X-Profile": params["profile"],
            "X-Segments": str(len(segments)),
            "X-Retries": str(retries_used),
        },
    )


class WarmupRequest(BaseModel):
    # 有給參考音檔就真的合成一句丟掉，把 CUDA kernel 打熱 —— 只載權重的話
    # 第一個真實請求還是要付首次配置的錢。
    ref_audio_path: str | None = None
    ref_text: str | None = None


@app.post("/warmup")
def warmup(req: WarmupRequest | None = None):
    m = get_model()
    generated = False
    if req and req.ref_audio_path and os.path.exists(req.ref_audio_path) and "clone" in MODES:
        try:
            params = resolve_params(None, {"retries": 0})
            _synth_clone(m, ["你好。"], "Auto", req.ref_audio_path, req.ref_text, params)
            generated = True
        except Exception as e:
            # 暖機失敗不算錯 —— 頂多第一句慢一點，不該讓 gateway 的 warmup 掛掉
            log.warning("暖機合成失敗（%s），只載了權重", e)
    return {
        "loaded": True,
        "warmed": generated,
        "presets": m.get_supported_speakers() or [],
        "profile": DEFAULT_PROFILE,
    }
