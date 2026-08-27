"""
TTS Gateway ── 引擎前面的統一入口。

這一包是「一個模型一個 docker」的版本：底下只掛一顆引擎，由 compose 的
`ENGINES` 環境變數指定（格式 `名稱=網址`，多顆用逗號分隔）。程式本身不寫死
任何引擎名稱，所以同一份 app.py 四包共用，日後要合併成多引擎也只是改環境變數。

為什麼引擎跟 gateway 要分成兩個容器：
  1. 引擎之間的相依互相衝突（qwen-tts 釘 transformers==4.57.3，CosyVoice 要
     4.51.3），本來就不能塞同一個 venv。拆開之後每包各自乾淨。
  2. 音色管理不該塞在引擎裡。引擎只負責「給我 wav 路徑跟文字，還我音訊」，
     音色的上傳、命名、保存、刪除全部集中在這裡。
  3. 引擎自己沒有 request queue，同時打多個請求會擠在一起。這裡用 semaphore 排隊。
  4. gateway 不用 GPU、不含權重，image 很小（<200MB），改一行重 build 只要兩分鐘。

音色（voice）有三種型別，因為各模型的音色來源本來就不一樣：
  clone   上傳一段參考音檔 → 克隆。cosyvoice2 / fun-cosyvoice3 / voxcpm2 支援。
  design  純文字描述音色，不需要音檔。voxcpm2 支援。
  preset  模型內建的精選音色。qwen3-tts 的 9 個 speaker 屬於這種，唯讀。

這一包實際支援哪幾種，看引擎 /health 回報的 modes，gateway 不寫死。
"""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Header, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/voices"))
REGISTRY = VOICES_DIR / "voices.json"
DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "cosyvoice2")
API_KEY = os.environ.get("API_KEY", "").strip()
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))
# 建立音色時把參考特徵推給引擎先算好用的 timeout。抽特徵只要一兩秒，
# 給 120 秒是留給「30 秒音檔 + ONNX 跑 CPU」的最壞情況。
PREPARE_TIMEOUT = float(os.environ.get("PREPARE_TIMEOUT", "120"))

# 參考音檔一律轉成單聲道，取樣率由 REF_SR 決定（預設 16k，各家模型的最大公因數：
# CosyVoice 的 load_wav 要求 >=16k，VoxCPM2 官方也是收 16k reference）。
#
# ⚠ CosyVoice2 / Fun-CosyVoice3 這兩包請在 compose 設 REF_SR=24000。原因是同一個
# 檔案會被上游拿去抽三種特徵，而三條路徑要的取樣率不一樣：
#     _extract_speech_token  → load_wav(wav, 16000)   speech tokenizer
#     _extract_spk_embedding → load_wav(wav, 16000)   campplus 音色向量
#     _extract_speech_feat   → load_wav(wav, 24000)   ← flow 的 prompt mel
# 給 16k 不會報錯（load_wav 只 assert sample_rate >= 16000），但第三條會把 16k
# **升採樣**到 24k —— 那份 mel 就是 flow matching 的聲學 prompt，8 kHz 以上整片是空的，
# 產出的聲音會繼承這個被砍掉高頻的頻譜包絡，聽起來悶、沒有齒音跟空氣感。
# 存 24k 則三條路徑都正確：前兩條自己降頻到 16k，第三條剛好命中。
REF_SR = int(os.environ.get("REF_SR", "16000"))

# 上傳音檔的長度建議值。太短音色抓不準，太長對音色沒幫助還拖慢每次合成。
# 這是「建議」，超過只會 warning；真正會讓合成失敗的硬上限由引擎自己宣告，
# 見 /health 的 max_ref_sec 跟下面的 FALLBACK_MAX_REF_SEC。
MIN_REF_SEC = 2.0
MAX_REF_SEC = 30.0


def _parse_engines() -> dict[str, str]:
    """ENGINES="name=url,name=url" → {name: url}"""
    out: dict[str, str] = {}
    for item in os.environ.get("ENGINES", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, url = item.partition("=")
        if name and url:
            out[name.strip()] = url.strip().rstrip("/")
    return out


ENGINES = _parse_engines()
# 每顆引擎同時只放一個請求進去 —— 引擎端沒有 queue，並發只會互相拖慢並吃爆記憶體。
_engine_locks = {name: asyncio.Semaphore(1) for name in ENGINES}
_registry_lock = asyncio.Lock()
# 引擎的能力宣告（支援哪些 mode、有哪些內建音色）有快取，
# 不然每次列音色都要往引擎打一次。
_caps_cache: dict[str, dict] = {}
_caps_cache_at = 0.0
CAPS_TTL = 300.0

app = FastAPI(
    title="TTS Gateway",
    description="TTS 引擎的統一入口，含音色庫。實際掛哪顆引擎看 ENGINES 環境變數。",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 認證（設了 API_KEY 才啟用）
# ---------------------------------------------------------------------------
def require_auth(authorization: str | None = Header(default=None)):
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(401, "缺少或錯誤的 Authorization header")


# ---------------------------------------------------------------------------
# 音色庫的讀寫
# ---------------------------------------------------------------------------
def _load_registry() -> dict[str, dict]:
    if not REGISTRY.exists():
        return {}
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 檔案壞掉時保留現場，不要靜靜地把使用者的音色清單清成空的
        broken = REGISTRY.with_suffix(f".broken-{int(time.time())}.json")
        shutil.copy2(REGISTRY, broken)
        raise HTTPException(500, f"voices.json 解析失敗，已備份到 {broken}")


def _save_registry(data: dict[str, dict]) -> None:
    """先寫暫存檔再 rename，避免寫到一半斷電留下半個檔案。"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REGISTRY)


# ---------------------------------------------------------------------------
# ffmpeg 包裝
# ---------------------------------------------------------------------------
def _probe_duration(path: Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise HTTPException(400, f"讀不出這個音檔的長度，格式可能不支援：{p.stderr.strip()[:200]}")
    try:
        return float(p.stdout.strip())
    except ValueError:
        raise HTTPException(400, "讀不出這個音檔的長度，格式可能不支援")


def _normalize_ref_audio(src: Path, dst: Path) -> None:
    """任意格式 → REF_SR 單聲道 16-bit wav。"""
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(REF_SR),
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise HTTPException(400, f"音檔轉檔失敗：{p.stderr.strip()[-300:]}")


def _convert_wav(wav: bytes, fmt: str) -> tuple[bytes, str]:
    """引擎一律回 wav；要別的格式時在這裡轉。"""
    if fmt == "wav":
        return wav, "audio/wav"
    codec = {
        "mp3": (["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"], "audio/mpeg"),
        "flac": (["-c:a", "flac", "-f", "flac"], "audio/flac"),
        "opus": (["-c:a", "libopus", "-b:a", "96k", "-f", "opus"], "audio/opus"),
        "aac": (["-c:a", "aac", "-b:a", "192k", "-f", "adts"], "audio/aac"),
    }.get(fmt)
    if codec is None:
        raise HTTPException(400, f"不支援的 response_format：{fmt}")
    args, mime = codec
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", *args, "pipe:1"],
        input=wav, capture_output=True,
    )
    if p.returncode != 0:
        raise HTTPException(500, f"輸出格式轉換失敗：{p.stderr.decode(errors='replace')[-300:]}")
    return p.stdout, mime


# ---------------------------------------------------------------------------
# 引擎呼叫
# ---------------------------------------------------------------------------
async def _engine_get(name: str, path: str, timeout: float = 10.0):
    url = ENGINES[name] + path
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()


async def _engine_synthesize(name: str, payload: dict) -> bytes:
    if name not in ENGINES:
        raise HTTPException(404, f"沒有這顆引擎：{name}。可用的有：{sorted(ENGINES)}")
    url = ENGINES[name] + "/synthesize"
    async with _engine_locks[name]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
            try:
                r = await c.post(url, json=payload)
            except httpx.RequestError as e:
                raise HTTPException(503, f"連不到引擎 {name}（{e}）。它可能還在載模型。")
    if r.status_code >= 400:
        # 把引擎的錯誤原封不動往上傳，不要包成無資訊的 500
        detail = r.text
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(r.status_code, f"[{name}] {detail}")
    return r.content


async def _engine_voice_op(engine: str, op: str, voice: dict,
                           load_model: bool = False) -> dict:
    """把音色的參考音檔推給引擎做 prepare / forget。

    這是純粹的最佳化：引擎那邊會把 speech token、x-vector、prompt mel 這些
    「同一個音色每次合成都算一模一樣」的東西抽好存起來，之後合成就直接取用。
    沒做也不影響正確性，引擎第一次用到時會自己補 —— 所以這裡**任何失敗都不會**
    讓建立音色的請求失敗，只會回一段說明放進 warnings。

    引擎沒實作這兩支路由（voxcpm2，或舊版 image）會拿到 404/405，一樣當成
    「沒做」處理，不會壞。
    """
    if engine not in ENGINES:
        return {"prepared": False, "reason": f"沒有這顆引擎：{engine}"}
    if not voice.get("audio_path"):
        return {"prepared": False, "reason": "這個音色沒有參考音檔"}
    payload = {"ref_audio_path": voice["audio_path"],
               "ref_text": voice.get("transcript"),
               "load_model": load_model}
    try:
        # 走跟合成同一個 semaphore：抽特徵也是 GPU 工作，不要跟合成擠在一起
        async with _engine_locks[engine]:
            async with httpx.AsyncClient(timeout=PREPARE_TIMEOUT) as c:
                r = await c.post(f"{ENGINES[engine]}/voices/{op}", json=payload)
    except httpx.RequestError as e:
        return {"prepared": False, "reason": f"連不到引擎 {engine}（{e}）"}
    if r.status_code in (404, 405):
        return {"prepared": False, "reason": f"引擎 {engine} 沒有實作音色特徵快取"}
    if r.status_code >= 400:
        return {"prepared": False, "reason": r.text[:200]}
    try:
        return r.json()
    except Exception:
        return {"prepared": False, "reason": "引擎回的不是 JSON"}


async def _prepare_warning(result: dict) -> str | None:
    """prepare 沒成功時給使用者一句看得懂的話；成功就回 None。"""
    if result.get("prepared"):
        return None
    reason = result.get("reason") or "原因不明"
    return (f"音色特徵還沒先建好（{reason}）。這不影響能不能用，"
            "只是第一次合成會多花幾秒抽特徵。")


async def _get_caps() -> dict[str, dict]:
    """把引擎的 /health 收集起來（有 TTL 快取）。

    引擎自己宣告支援哪些 mode，gateway 不寫死。這樣之後把 qwen3-tts 換成
    Base checkpoint（會克隆）時，只要改引擎的 ENGINE_MODES 就好，這裡不用動。
    """
    global _caps_cache, _caps_cache_at
    if _caps_cache and time.time() - _caps_cache_at < CAPS_TTL:
        return _caps_cache
    out: dict[str, dict] = {}
    for name in ENGINES:
        try:
            info = await _engine_get(name, "/health")
            out[name] = {"modes": info.get("modes") or [], "presets": info.get("presets") or [],
                         "max_ref_sec": info.get("max_ref_sec")}
        except Exception:
            # 引擎還沒起來就先當作沒能力，不要讓整個列表掛掉
            out[name] = {"modes": [], "presets": [], "max_ref_sec": None}
    _caps_cache, _caps_cache_at = out, time.time()
    return out


async def _get_presets() -> dict[str, list[str]]:
    return {name: c["presets"] for name, c in (await _get_caps()).items()}


# ---------------------------------------------------------------------------
# 音色型別 → 支援的引擎
# ---------------------------------------------------------------------------
# 引擎還沒起來時用的靜態備援（跟各 server.py 的 /health 宣告一致）。
# 這裡四顆都列著，但只有這一包實際掛的那顆會被查到。
FALLBACK_MODES = {
    "cosyvoice2": ["clone"],
    "fun-cosyvoice3": ["clone"],
    "voxcpm2": ["clone", "design", "default"],
    # qwen3-tts 目前掛 Base checkpoint（純克隆）。換回 CustomVoice 的話這裡是 ["preset"]，
    # 不過這張表只在引擎連不上、拿不到 /health 時才會用到，正常都是照引擎宣告的走。
    "qwen3-tts": ["clone"],
}

# 參考音檔的硬上限（秒）。None = 這顆引擎沒有上限。
# 正常走引擎 /health 宣告的 max_ref_sec，這張表只在引擎連不上時當備援 ——
# 但這條路徑一定要有備援：引擎沒起來時放行一個過長的音檔，使用者會拿到 201，
# 然後每一次合成都失敗，而且錯誤訊息完全看不出是音檔太長。
FALLBACK_MAX_REF_SEC = {
    # CosyVoice 的 frontend._extract_speech_token 有
    #   assert speech.shape[1] / 16000 <= 30
    # 超過就是 AssertionError，它不會自己截斷。
    "cosyvoice2": 30.0,
    "fun-cosyvoice3": 30.0,
    "voxcpm2": None,
    "qwen3-tts": None,
}


def _max_ref_sec(engine: str, caps: dict[str, dict] | None = None) -> float | None:
    got = (caps or {}).get(engine, {}).get("max_ref_sec")
    return got if got is not None else FALLBACK_MAX_REF_SEC.get(engine)


def _compatible_engines(voice: dict, caps: dict[str, dict] | None = None) -> list[str]:
    if voice["type"] == "preset":
        return [voice["engine"]]  # preset 只有自己那顆引擎能用
    out = []
    for name in ENGINES:
        modes = (caps or {}).get(name, {}).get("modes") or FALLBACK_MODES.get(name, [])
        if voice["type"] in modes:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# 音色 API
# ---------------------------------------------------------------------------
class VoiceUpdate(BaseModel):
    name: str | None = None
    transcript: str | None = None
    description: str | None = None
    language: str | None = None
    default_engine: str | None = None


class DesignVoice(BaseModel):
    name: str = Field(..., description="音色名稱，例如「溫柔女聲」")
    description: str = Field(..., description="音色的自然語言描述，例如「一位溫柔的年輕女性，語速偏慢」")
    language: str | None = None
    default_engine: str = DEFAULT_ENGINE


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _hdr(value: str) -> str:
    """HTTP header 只吃 latin-1。音色名稱/ID 可能含中文，這裡先過濾掉。"""
    return value.encode("latin-1", "replace").decode("latin-1")


def _public(voice: dict, caps: dict[str, dict] | None = None) -> dict:
    """對外的音色表示：不暴露容器內的絕對路徑。"""
    out = {k: v for k, v in voice.items() if k != "audio_path"}
    out["has_audio"] = bool(voice.get("audio_path"))
    out["compatible_engines"] = _compatible_engines(voice, caps)
    return out


@app.post("/v1/voices", dependencies=[Depends(require_auth)], status_code=201)
async def create_voice(
    file: UploadFile = File(..., description="參考音檔，建議 5-15 秒、單人、無背景音樂"),
    name: str = Form(..., description="音色名稱"),
    transcript: str = Form("", description="參考音檔的逐字稿。強烈建議填，音色相似度差很多"),
    language: str = Form(""),
    default_engine: str = Form(DEFAULT_ENGINE),
):
    """上傳一段參考音檔，建立一個 clone 音色。

    逐字稿請盡量填。不填的話 CosyVoice 會走 cross-lingual 路徑、VoxCPM2 會少掉
    ultimate cloning，兩者的音色相似度都會明顯下降。
    """
    if default_engine not in ENGINES:
        raise HTTPException(400, f"default_engine 不存在：{default_engine}。可用：{sorted(ENGINES)}")

    voice_id = _new_id("voice")
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    dst = VOICES_DIR / f"{voice_id}.wav"

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "up").suffix or ".bin", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        _normalize_ref_audio(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)

    duration = _probe_duration(dst)
    warnings = []
    if duration < MIN_REF_SEC:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f"參考音檔只有 {duration:.1f} 秒，至少要 {MIN_REF_SEC} 秒才抓得到音色")
    hard_limit = _max_ref_sec(default_engine, await _get_caps())
    if hard_limit is not None and duration > hard_limit:
        dst.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"參考音檔 {duration:.1f} 秒，超過 {default_engine} 的上限 {hard_limit:.0f} 秒。"
            f"這顆引擎不會自動截斷，硬建下去會變成音色建得起來、但每一次合成都失敗。"
            f"請先剪到 {hard_limit:.0f} 秒以內（建議 5-15 秒）再上傳。",
        )
    if duration > MAX_REF_SEC:
        warnings.append(f"參考音檔 {duration:.1f} 秒偏長，建議 5-15 秒；太長對音色沒有幫助，"
                        "還會讓每次合成都多花時間重抽特徵")
    if not transcript.strip():
        warnings.append("沒有給逐字稿，音色相似度會下降。可以之後用 PATCH /v1/voices/{id} 補上")

    voice = {
        "id": voice_id,
        "name": name,
        "type": "clone",
        "audio_path": str(dst),
        "transcript": transcript.strip() or None,
        "description": None,
        "language": language.strip() or None,
        "default_engine": default_engine,
        "duration_sec": round(duration, 2),
        "sample_rate": REF_SR,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    async with _registry_lock:
        reg = _load_registry()
        reg[voice_id] = voice
        _save_registry(reg)

    # 音色存好了，順手把參考特徵推給引擎先算起來放著 —— 之後每一次合成都不用再抽。
    # 引擎還沒載模型的話它會直接說「沒建」（不會為了這個去載 30-90 秒的模型），
    # 那就等第一次合成時自己建。任何失敗都只是變成一句 warning。
    prep = await _engine_voice_op(default_engine, "prepare", voice)
    warn = await _prepare_warning(prep)
    if warn:
        warnings.append(warn)

    return JSONResponse(
        {"voice": _public(voice), "prepared": bool(prep.get("prepared")), "warnings": warnings},
        status_code=201,
    )


@app.post("/v1/voices/design", dependencies=[Depends(require_auth)], status_code=201)
async def create_design_voice(body: DesignVoice):
    """用一段文字描述建立音色，不需要參考音檔。目前只有 voxcpm2 支援。"""
    caps = await _get_caps()
    design_engines = [n for n in ENGINES
                      if "design" in (caps.get(n, {}).get("modes") or FALLBACK_MODES.get(n, []))]
    if body.default_engine not in design_engines:
        raise HTTPException(
            400, f"design 型別的音色只有這些引擎支援：{design_engines}"
        )
    voice_id = _new_id("design")
    voice = {
        "id": voice_id,
        "name": body.name,
        "type": "design",
        "audio_path": None,
        "transcript": None,
        "description": body.description,
        "language": body.language,
        "default_engine": body.default_engine,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    async with _registry_lock:
        reg = _load_registry()
        reg[voice_id] = voice
        _save_registry(reg)
    return JSONResponse({"voice": _public(voice)}, status_code=201)


@app.get("/v1/voices", dependencies=[Depends(require_auth)])
async def list_voices(type: str | None = None, engine: str | None = None):
    """列出所有音色：自建的（clone/design）+ 引擎內建的（preset）。"""
    reg = _load_registry()
    caps = await _get_caps()
    voices = [_public(v, caps) for v in reg.values()]

    for eng, c in caps.items():
        speakers = c["presets"]
        for spk in speakers:
            voices.append(
                _public({
                    "id": f"preset_{eng}_{spk}",
                    "name": spk,
                    "type": "preset",
                    "engine": eng,
                    "speaker": spk,
                    "description": None,
                    "language": None,
                    "default_engine": eng,
                    "readonly": True,
                })
            )

    if type:
        voices = [v for v in voices if v["type"] == type]
    if engine:
        voices = [v for v in voices if engine in v["compatible_engines"]]
    voices.sort(key=lambda v: (v["type"], v["name"]))
    return {"object": "list", "data": voices}


@app.get("/v1/voices/{voice_id}", dependencies=[Depends(require_auth)])
async def get_voice(voice_id: str):
    voice = await _resolve_voice(voice_id)
    return {"voice": _public(voice)}


@app.patch("/v1/voices/{voice_id}", dependencies=[Depends(require_auth)])
async def update_voice(voice_id: str, body: VoiceUpdate):
    """改名字、補逐字稿、換預設引擎。preset 音色是唯讀的。"""
    async with _registry_lock:
        reg = _load_registry()
        if voice_id not in reg:
            raise HTTPException(404, f"找不到音色 {voice_id}（內建 preset 音色不能改）")
        voice = reg[voice_id]
        for field in ("name", "transcript", "description", "language", "default_engine"):
            val = getattr(body, field)
            if val is not None:
                if field == "default_engine" and val not in ENGINES:
                    raise HTTPException(400, f"default_engine 不存在：{val}")
                voice[field] = val
        reg[voice_id] = voice
        _save_registry(reg)

    # 逐字稿是 zero-shot 那份快取的一部分（引擎那邊 prompt_text 也進了 key），
    # 改了就得把舊的丟掉重建，不然新的逐字稿要等 LRU 淘汰才會生效。
    prepared = None
    if body.transcript is not None:
        await _engine_voice_op(voice.get("default_engine") or DEFAULT_ENGINE, "forget", voice)
        prep = await _engine_voice_op(voice.get("default_engine") or DEFAULT_ENGINE,
                                      "prepare", voice)
        prepared = bool(prep.get("prepared"))
    out = {"voice": _public(voice)}
    if prepared is not None:
        out["prepared"] = prepared
    return out


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_auth)])
async def delete_voice(voice_id: str):
    async with _registry_lock:
        reg = _load_registry()
        if voice_id not in reg:
            raise HTTPException(404, f"找不到音色 {voice_id}（內建 preset 音色不能刪）")
        voice = reg.pop(voice_id)
        _save_registry(reg)
    # 先叫引擎放掉快取（那裡面是 GPU 張量），再刪檔案
    for name in ENGINES:
        await _engine_voice_op(name, "forget", voice)
    if voice.get("audio_path"):
        Path(voice["audio_path"]).unlink(missing_ok=True)
    return {"deleted": True, "id": voice_id}


@app.post("/v1/voices/{voice_id}/prepare", dependencies=[Depends(require_auth)])
async def prepare_voice(voice_id: str, engine: str | None = None, load_model: bool = False):
    """叫引擎把這個音色的參考特徵抽好放著，之後合成就不用再抽。

    建立音色時已經自動做過一次，這支是給這幾種情況補打的：
      * 引擎重啟過 —— 快取放在引擎的記憶體裡，容器一重啟就沒了
      * 建音色的時候引擎還沒載模型（那時會跳過，回應的 warnings 有寫）
      * 音色是從別台機器整個 work/voices/ 複製過來的，沒經過 POST /v1/voices

    `load_model=true` 會允許引擎為了建快取去載模型（要等 30-90 秒）；
    預設 false，引擎沒載模型就直接說沒建。
    """
    voice = await _resolve_voice(voice_id)
    if voice["type"] != "clone":
        raise HTTPException(400, f"只有 clone 型別的音色有參考特徵可以建，這個是 {voice['type']}")
    eng = engine or voice.get("default_engine") or DEFAULT_ENGINE
    result = await _engine_voice_op(eng, "prepare", voice, load_model=load_model)
    return {"engine": eng, "voice_id": voice["id"], **result}


@app.post("/v1/voices/{voice_id}/preview", dependencies=[Depends(require_auth)])
async def preview_voice(voice_id: str, engine: str | None = None, text: str | None = None):
    """用這個音色合成一句短句試聽，確認音色對不對再拿去跑批次。"""
    voice = await _resolve_voice(voice_id)
    text = text or "這是音色試聽，用來確認聲音是不是你要的。"
    eng = engine or voice.get("default_engine") or DEFAULT_ENGINE
    payload = _build_payload(voice, eng, text, None, 1.0, None, await _get_caps())
    wav = await _engine_synthesize(eng, payload)
    return Response(wav, media_type="audio/wav",
                    headers={"X-Engine": _hdr(eng), "X-Voice-Id": _hdr(voice["id"])})


# ---------------------------------------------------------------------------
# 音色解析：把使用者給的字串（id / 名字 / preset speaker）變成一筆音色紀錄
# ---------------------------------------------------------------------------
async def _resolve_voice(ref: str) -> dict:
    reg = _load_registry()
    if ref in reg:
        return reg[ref]

    # preset_<engine>_<speaker>
    if ref.startswith("preset_"):
        rest = ref[len("preset_"):]
        for eng in sorted(ENGINES, key=len, reverse=True):
            if rest.startswith(eng + "_"):
                spk = rest[len(eng) + 1:]
                return {"id": ref, "name": spk, "type": "preset", "engine": eng,
                        "speaker": spk, "default_engine": eng, "description": None}

    # 用名字找自建音色
    by_name = [v for v in reg.values() if v["name"] == ref]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise HTTPException(400, f"有 {len(by_name)} 個音色都叫「{ref}」，請改用 voice id")

    # 直接給 preset speaker 名字，例如 voice="Vivian"
    for eng, speakers in (await _get_presets()).items():
        if ref in speakers:
            return {"id": f"preset_{eng}_{ref}", "name": ref, "type": "preset", "engine": eng,
                    "speaker": ref, "default_engine": eng, "description": None}

    raise HTTPException(404, f"找不到音色「{ref}」。用 GET /v1/voices 看有哪些可以用。")


def _build_payload(voice: dict, engine: str, text: str, instruct: str | None,
                   speed: float, language: str | None,
                   caps: dict[str, dict] | None = None,
                   gen: dict | None = None) -> dict:
    """把一筆音色紀錄翻譯成目標引擎看得懂的 /synthesize 請求。"""
    compatible = _compatible_engines(voice, caps)
    if engine not in compatible:
        raise HTTPException(
            400,
            f"音色「{voice['name']}」（{voice['type']} 型別）不能用在引擎 {engine}。"
            f"可以用的引擎：{compatible}",
        )
    payload = {
        "text": text,
        "mode": voice["type"],
        "instruct": instruct,
        "speed": speed,
        "language": language or voice.get("language"),
    }
    if voice["type"] == "clone":
        payload["ref_audio_path"] = voice["audio_path"]
        payload["ref_text"] = voice.get("transcript")
    elif voice["type"] == "design":
        payload["description"] = voice["description"]
    elif voice["type"] == "preset":
        payload["speaker"] = voice["speaker"]
    # 生成參數只放有值的 —— None 不要送，讓引擎端的 profile 決定
    for key, val in (gen or {}).items():
        if val is not None:
            payload[key] = val
    # type=default 什麼都不帶，讓引擎自己決定
    return payload


# ---------------------------------------------------------------------------
# OpenAI 相容合成
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    input: str = Field(..., description="要合成的文字")
    model: str = Field(DEFAULT_ENGINE, description="引擎名稱。這一包只有一顆，不給就用 DEFAULT_ENGINE")
    voice: str | None = Field(None, description="voice id、音色名稱，或內建 speaker 名稱")
    instructions: str | None = Field(None, description="語氣指示，例如「用開心一點的語氣說」")
    response_format: Literal["wav", "mp3", "flac", "opus", "aac"] = "wav"
    speed: float = 1.0
    language: str | None = None
    # 以下是生成參數，全部選填，不給就照引擎自己的預設走（qwen3-tts 是
    # QWEN_PROFILE 那一檔）。只有 qwen3-tts 認得，其他三顆引擎的 SynthRequest
    # 沒有這些欄位，pydantic 會直接忽略，所以帶了也不會壞。
    profile: str | None = Field(None, description="qwen3-tts：fast / balanced / quality")
    temperature: float | None = Field(None, description="越低越穩、越高語調越活。TTS 建議 0.7-0.9")
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = Field(None, description="調高可以擋「整段重複唸」，建議 1.05-1.15")
    seed: int | None = Field(None, description="固定 seed 可以重現同一次生成，也方便 re-roll 不滿意的版本")


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def create_speech(body: SpeechRequest):
    if not body.input.strip():
        raise HTTPException(400, "input 是空的")

    # OpenAI 官方的 model 名稱直接對應到預設引擎，方便既有 client 不用改
    engine = {"tts-1": DEFAULT_ENGINE, "tts-1-hd": DEFAULT_ENGINE,
              "gpt-4o-mini-tts": DEFAULT_ENGINE}.get(body.model, body.model)
    if engine not in ENGINES:
        raise HTTPException(404, f"沒有這顆引擎：{body.model}。可用：{sorted(ENGINES)}")

    if body.voice:
        voice = await _resolve_voice(body.voice)
    else:
        voice = await _default_voice_for(engine)

    gen = {k: getattr(body, k) for k in
           ("profile", "temperature", "top_p", "top_k", "repetition_penalty", "seed")}
    payload = _build_payload(voice, engine, body.input, body.instructions,
                             body.speed, body.language, await _get_caps(), gen)
    wav = await _engine_synthesize(engine, payload)
    data, mime = _convert_wav(wav, body.response_format)
    return Response(
        data, media_type=mime,
        headers={"X-Engine": _hdr(engine), "X-Voice-Id": _hdr(voice["id"]),
                 "X-Voice-Type": _hdr(voice["type"])},
    )


async def _default_voice_for(engine: str) -> dict:
    """沒指定 voice 的時候，挑一個這顆引擎能用的。"""
    presets = (await _get_presets()).get(engine) or []
    if presets:
        spk = presets[0]
        return {"id": f"preset_{engine}_{spk}", "name": spk, "type": "preset",
                "engine": engine, "speaker": spk, "default_engine": engine, "description": None}
    modes = (await _get_caps()).get(engine, {}).get("modes") or FALLBACK_MODES.get(engine, [])
    if "default" in modes:
        return {"id": "builtin_default", "name": "default", "type": "default",
                "default_engine": engine, "description": None}
    # CosyVoice 那兩顆一定要參考音檔，這裡挑最近建立的 clone 音色
    reg = _load_registry()
    clones = sorted([v for v in reg.values() if v["type"] == "clone"],
                    key=lambda v: v.get("created_at", ""), reverse=True)
    if clones:
        return clones[0]
    raise HTTPException(
        400,
        f"{engine} 需要一個參考音檔音色，但音色庫是空的。"
        "請先 POST /v1/voices 上傳一段參考音檔。",
    )


# ---------------------------------------------------------------------------
# 其他
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models():
    data = []
    for name in sorted(ENGINES):
        try:
            info = await _engine_get(name, "/health")
            info["status"] = "ready" if info.get("loaded") else "idle"
        except Exception as e:
            info = {"engine": name, "status": "unreachable", "error": str(e)}
        info.update({"id": name, "object": "model", "owned_by": "local"})
        data.append(info)
    return {"object": "list", "data": data}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "engines": sorted(ENGINES), "voices_dir": str(VOICES_DIR)}


def _newest_clone() -> dict | None:
    """音色庫裡最近建立的 clone 音色。暖機要真的合成一句就得有參考音檔。"""
    clones = sorted([v for v in _load_registry().values() if v["type"] == "clone"],
                    key=lambda v: v.get("created_at", ""), reverse=True)
    return clones[0] if clones else None


@app.post("/v1/warmup", dependencies=[Depends(require_auth)])
async def warmup(engine: str | None = None):
    """主動把模型載進 GPU。第一次合成要等 30-90 秒載模型，先打這支比較不會嚇到。

    音色庫裡有 clone 音色的話會順便帶進去，讓引擎真的合成一句短的再丟掉 ——
    只載權重的話，第一個真實請求還是要付 CUDA kernel 首次配置的錢。
    引擎端沒實作這個 body 的話會直接忽略，不影響原本的行為。
    """
    targets = [engine] if engine else list(ENGINES)
    voice = _newest_clone()
    body = {
        "ref_audio_path": voice.get("audio_path") if voice else None,
        "ref_text": voice.get("transcript") if voice else None,
    }
    out = {}
    for name in targets:
        if name not in ENGINES:
            out[name] = {"error": "unknown engine"}
            continue
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
                r = await c.post(ENGINES[name] + "/warmup", json=body)
                out[name] = r.json()
        except Exception as e:
            out[name] = {"error": str(e)}
    _caps_cache.clear()
    return out
