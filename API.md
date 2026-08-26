# API 總表 — 四包共通

四包的 `gateway/app.py` **byte 完全相同**，所以對外的 API 只有一份。
差別只有兩件事：**打哪個 port**，以及**底下那顆引擎宣告支援哪些 mode**。

各包的 `DEPLOY.md` Step 9 是該包的簡表；這份是跨四包的完整版。

```
你的 client ──HTTP──▶ gateway :1800X ──HTTP──▶ engine :8080（容器內）
                      音色庫 + 排隊              吃 GPU、只做推論
                      OpenAI 相容               不對外
```

| 包 | gateway port | 引擎除錯 port | `model` 名稱 | 支援的 mode |
|---|---|---|---|---|
| cosyvoice2 | 18001 | 18081 | `cosyvoice2` | `clone` |
| fun-cosyvoice3 | 18002 | 18082 | `fun-cosyvoice3` | `clone` |
| qwen3-tts | 18003 | 18083 | `qwen3-tts` | `clone` |
| voxcpm2 | 18004 | 18084 | `voxcpm2` | `clone` / `design` / `default` |

> 路由四包都有（同一份 app.py），引擎沒能力的那幾條會回 **400 + 人看得懂的原因**，
> 不會是空白的 500。

---

## 認證與連線

| | 設在哪 | 說明 |
|---|---|---|
| `API_KEY` | compose.yaml 的 gateway `environment` | 有設才啟用。所有 `/v1/*` 都要帶 `Authorization: Bearer <key>` |
| `TTS_GATEWAY` | 你的 shell | `voice.sh` / `synth.sh` / `batch.py` 用，預設 `http://localhost:1800X` |
| `TTS_API_KEY` | 你的 shell | 同上，腳本會自己帶成 Bearer header |

不需要 token 的：`/healthz`、`/docs`、`/openapi.json`。
沒帶或帶錯 → **401**。

---

## Gateway API

### `GET /healthz`

免認證。確認 gateway 這個容器活著（**不代表模型載好了**，那要看 `/v1/models`）。

```json
{"status": "ok", "engines": ["voxcpm2"], "voices_dir": "/voices"}
```

---

### `GET /v1/models`

引擎狀態。逐顆去打引擎的 `/health`，**不吃快取**。

```json
{"object": "list", "data": [{
  "id": "voxcpm2", "object": "model", "owned_by": "local",
  "engine": "voxcpm2", "status": "ready",
  "loaded": true, "sample_rate": 48000,
  "modes": ["clone", "design", "default"],
  "needs_ref_audio": false, "presets": []
}]}
```

`status`：`ready`（模型已載入）/ `idle`（容器活著但還沒載模型）/ `unreachable`（連不到，會多一個 `error` 欄位）。

---

### `POST /v1/warmup`

把模型載進 GPU。**第一次合成要等 30–90 秒載模型，起服務後先打這支**。
可加 `?engine=<name>`（這一包只有一顆，通常不用）。順便清掉能力快取。

```bash
curl -X POST http://localhost:18004/v1/warmup
# {"voxcpm2": {"loaded": true, "sample_rate": 48000}}
```

---

### `POST /v1/audio/speech` — 合成（OpenAI 相容）

Body 是 JSON，回 **audio binary**（不是 JSON）。

| 欄位 | 型別 | 預設 | 說明 |
|---|---|---|---|
| `input` | string | 必填 | 要合成的文字。空字串 → 400 |
| `model` | string | `DEFAULT_ENGINE` | 引擎名稱。`tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` 會自動對應到預設引擎，既有 OpenAI client 不用改 |
| `voice` | string\|null | null | voice id、音色名稱，或內建 speaker 名稱。不給則自動挑（見下） |
| `instructions` | string\|null | null | 語氣指示，例如「用開心一點的語氣說」 |
| `response_format` | enum | `wav` | `wav` / `mp3` / `flac` / `opus` / `aac`。非 wav 由 gateway 用 ffmpeg 轉 |
| `speed` | float | 1.0 | **只有 CosyVoice 那兩顆真的會用**，見下面的參數效力表 |
| `language` | string\|null | null | **只有 qwen3-tts 會用**。不給則沿用音色上的 `language` |
| `profile` | string\|null | null | **只有 qwen3-tts 會用**。`fast` / `balanced` / `quality`，只這一次生效 |
| `temperature` / `top_p` / `top_k` / `repetition_penalty` | number\|null | null | **只有 qwen3-tts 會用**。蓋掉 profile 的對應欄位 |
| `seed` | int\|null | null | **只有 qwen3-tts 會用**。固定住可重現同一次生成 |

> 這五個生成參數欄位目前**只加在 qwen3-tts 那一包的 `gateway/app.py`**。
> `models/*/gateway/app.py` 是四份各自獨立的檔案（不是 symlink），這次沒有同步過去。
> 之後要同步也不會壞：其他三顆引擎的 `SynthRequest` 沒有這些欄位，pydantic 預設
> 會忽略多餘欄位。

回應 header：

| Header | 內容 |
|---|---|
| `X-Engine` | 實際用了哪顆引擎 |
| `X-Voice-Id` | 實際用了哪個音色 |
| `X-Voice-Type` | `clone` / `design` / `preset` / `default` |

> header 只吃 latin-1，中文音色名稱在 `X-Voice-Id` 會變成 `?`。要看實際音色請用回應 body 之外的 `GET /v1/voices`，不要 parse 這個 header。

**沒給 `voice` 時 gateway 怎麼挑**（`_default_voice_for`）：

1. 引擎有內建 preset → 用第一個（目前四包都沒有 preset，這條走不到）
2. 引擎支援 `default` mode → 讓模型自己生一個音色（voxcpm2 走這條）
3. 否則挑**最近建立的 clone 音色**（CosyVoice 兩顆 + qwen3-tts 走這條）
4. 都沒有 → **400**，要你先 `POST /v1/voices` 上傳參考音檔

```bash
curl -X POST http://localhost:18002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"今天天氣真好。","model":"fun-cosyvoice3","voice":"小美","response_format":"mp3"}' \
  --output out.mp3
```

---

### 音色 API

音色（voice）有三種 type，因為各模型的音色來源本來就不一樣：

| type | 怎麼來的 | 哪幾包能用 | 可改可刪 |
|---|---|---|---|
| `clone` | 上傳參考音檔 | 四包都可以 | ✅ |
| `design` | 純文字描述，不用音檔 | voxcpm2 | ✅ |
| `preset` | 模型內建 | 目前沒有（qwen3-tts 換成 Base checkpoint 後就沒有內建音色了）| ❌ 唯讀 |

自建音色存在 `work/voices/`：`voices.json` 是索引，`<voice_id>.wav` 是正規化後的參考音檔。整個資料夾複製走就能搬機器，也可以複製給另外三包用。

#### `POST /v1/voices` — 上傳參考音檔建 clone 音色

`multipart/form-data`，回 **201**。

| 欄位 | 必填 | 說明 |
|---|---|---|
| `file` | ✅ | 參考音檔。任意格式，gateway 用 ffmpeg 轉成 **16k 單聲道 16-bit wav** |
| `name` | ✅ | 音色名稱 |
| `transcript` | — | 參考音檔的逐字稿。**強烈建議填**，見下 |
| `language` | — | 語言標記 |
| `default_engine` | — | 預設 `DEFAULT_ENGINE`；不存在的引擎 → 400 |

長度檢查：**< 2 秒直接 400**（音色抓不準，wav 會刪掉）；**> 30 秒過關但回 warning**（建議 5–15 秒，部分模型會自行截斷）。

```json
{"voice": {"id": "voice_a1b2c3d4e5f6", "name": "小美", "type": "clone",
           "transcript": "...", "duration_sec": 8.3, "sample_rate": 16000,
           "has_audio": true, "compatible_engines": ["fun-cosyvoice3"],
           "created_at": "2026-08-21T03:00:00Z"},
 "warnings": ["沒有給逐字稿，音色相似度會下降。..."]}
```

> **逐字稿為什麼重要**：不填的話 CosyVoice 會退到 cross-lingual 路徑、VoxCPM2 會少掉 ultimate cloning，兩者音色相似度都明顯下降。忘了填可以事後 `PATCH` 補。

回應**不會**有 `audio_path`（容器內的絕對路徑不對外），改用 `has_audio`。

#### `POST /v1/voices/design` — 用文字描述造音色

JSON，回 **201**。只有引擎宣告支援 `design` 才能用（目前是 voxcpm2），否則 400。

```bash
curl -X POST http://localhost:18004/v1/voices/design \
  -H 'Content-Type: application/json' \
  -d '{"name":"溫柔女聲","description":"一位溫柔的年輕女性，語速偏慢，咬字清楚"}'
```

欄位：`name`、`description`（皆必填）、`language`、`default_engine`。

#### `GET /v1/voices` — 列音色

自建的（clone/design）+ 引擎內建的（preset）一起列。

query：`?type=clone|design|preset`、`?engine=<name>`（過濾 `compatible_engines` 含這顆的）。

```json
{"object": "list", "data": [{"id": "...", "name": "...", "type": "clone",
  "has_audio": true, "compatible_engines": ["voxcpm2"]}]}
```

> **preset 音色要 warmup 之後才看得到**（目前四包都沒有 preset，這段留給換回 CustomVoice 時參考）。引擎的內建 speaker 清單是模型載入時才填的，容器剛起來時 `/health` 的 `presets` 是空的。而 gateway 對引擎能力有 **300 秒快取**，所以冷啟動後太早打會看到空清單、而且會空 5 分鐘。先 `POST /v1/warmup`（它會順手清快取）。

#### `GET /v1/voices/{id}` / `PATCH` / `DELETE`

- `PATCH`：可改 `name`、`transcript`、`description`、`language`、`default_engine`。**preset 是唯讀的，打了回 404。**
- `DELETE`：連 wav 一起刪，回 `{"deleted": true, "id": "..."}`。

#### `POST /v1/voices/{id}/preview` — 試聽

參數走 **query string**，不是 body：`?text=<試聽文字>`、`?engine=<name>`。
回 wav。固定用 `speed=1.0`、不帶 `instructions`。

---

### 音色怎麼被解析（`voice` 欄位可以塞什麼）

`_resolve_voice` 依序試：

1. 完全符合的 **voice id**（`voice_xxx` / `design_xxx`）
2. `preset_<engine>_<speaker>` 格式
3. **自建音色的名稱**。撞名（兩個以上）→ **400**，叫你改用 id
4. 直接給 **preset speaker 名稱**，例如 `"Vivian"`（目前四包都沒有 preset，這條走不到）
5. 都找不到 → **404**

---

## 參數在各引擎的實際效力

同一個欄位在四顆引擎的下場不一樣。這張表是實際讀 `engine/server.py` 得到的：

| 欄位 | cosyvoice2 / fun-cosyvoice3 | qwen3-tts | voxcpm2 |
|---|---|---|---|
| `speed` | ✅ 傳進 inference | ❌ 忽略 | ❌ 忽略 |
| `language` | ❌ 忽略 | ✅ 用；不給則 `Auto` 自動判斷 | ❌ 忽略 |
| `instructions` | ✅ 走 `inference_instruct2`（**會蓋掉 zero-shot 路徑**） | ❌ 忽略（`generate_voice_clone` 沒這個參數）| ✅ 併成文字最前面的括號 prefix |
| 音色 `description` | 與 `instructions` 合併成同一段風格文字 | 同左 | 同左（併進 prefix） |
| `response_format` | gateway 統一用 ffmpeg 轉，四包一致 | 同左 | 同左 |
| `profile` / `temperature` / `top_p` / `top_k` / `repetition_penalty` / `seed` | ❌ 忽略 | ✅ 傳進 `model.generate` 的 kwargs，見該包 DEPLOY.md 的「生成參數」 | ❌ 忽略 |

> **CosyVoice 的坑**：給了 `instructions` 就會走 instruct2，**不再走 zero-shot**，音色相似度會跟不給時不一樣。要最像原音就別給 `instructions`。

輸出取樣率由引擎決定（看 `/v1/models` 的 `sample_rate`）；voxcpm2 是 48kHz。

---

## 引擎內部契約（`:808X`，預設不對外）

四顆引擎都實作同一組介面。**正常情況不要直接打** —— 引擎自己沒有 request queue，繞過 gateway 的 semaphore 同時打兩個請求可能把記憶體吃爆。要除錯就把 compose 裡 engine 的 `ports` 取消註解。

| 方法 | 路徑 | 回應 |
|---|---|---|
| GET | `/health` | 能力宣告：`engine` / `loaded` / `sample_rate` / `modes` / `needs_ref_audio` / `presets` / `max_ref_sec`（參考音檔的硬上限秒數，`null` = 沒有上限。gateway 在 `POST /v1/voices` 就用它擋掉過長的音檔）|
| POST | `/synthesize` | `audio/wav`，header 帶 `X-Sample-Rate`、`X-Engine` |
| POST | `/warmup` | 主動載模型 |
| GET | `/presets` | **只有 qwen3-tts 有**：`{"speakers": [...], "languages": [...]}` |

### `POST /synthesize` 的 body

gateway 由 `_build_payload` 組出來，四顆共通欄位：

```
text, mode, ref_audio_path, ref_text, description, instruct, speaker, language, speed
```

qwen3-tts 那一包的 gateway 還會把有值的生成參數一起放進來（`None` 的不放，
讓引擎端的 profile 決定）：

```
profile, temperature, top_p, top_k, repetition_penalty, seed
```

`mode` 決定帶哪幾個：

| mode | 額外帶的欄位 |
|---|---|
| `clone` | `ref_audio_path`、`ref_text` |
| `design` | `description` |
| `preset` | `speaker` |
| `default` | 什麼都不帶，模型自己決定 |

**voxcpm2 的引擎多兩個欄位**：`cfg_value`、`inference_timesteps`。
⚠️ **gateway 的 `SpeechRequest` 沒有這兩個欄位，所以打 `/v1/audio/speech` 調不到**。
要改只能改 compose 的 `VOXCPM_CFG` / `VOXCPM_TIMESTEPS` 環境變數（預設 `2.0` / `10`），或直接打引擎。

### 各引擎的推論路徑

**cosyvoice2 / fun-cosyvoice3**（同一份 server.py，靠 `COSYVOICE_CLASS` 切）：

| 條件 | 走哪條 |
|---|---|
| 有 `instruct` 或 `description` | `inference_instruct2`（風格控制，仍要參考音檔當底） |
| 有 `ref_text` | `inference_zero_shot` — **音色最像** |
| 都沒有 | `inference_cross_lingual` — 不用逐字稿，相似度略差 |

沒有 `ref_audio_path` 一律 400：這是 zero-shot 克隆模型，每條路徑都要參考音檔。

**qwen3-tts**：目前掛 `Qwen3-TTS-12Hz-1.7B-Base`，`mode=clone` → `generate_voice_clone(ref_audio, ref_text)`。沒有 `ref_audio_path` 回 400。`ref_text`（逐字稿）不是必填但**強烈建議**，模型拿它去對齊參考音檔。`mode=preset` → 400，Base 沒有內建 speaker（那是 CustomVoice checkpoint 才有的，見該包 DEPLOY.md 的「換模型」）。

**voxcpm2**：`clone` 給了 `ref_text` 會**同一段音檔同時當 reference 跟 prompt**，升級成 ultimate cloning（相似度最高）；`design` 把描述包成 `(描述)文字` 的 prefix；`default` 什麼都不加。

### 引擎相關的環境變數

| 引擎 | 變數 |
|---|---|
| 共通 | `ENGINE_NAME`、`MODEL_PATH`、`ENGINE_MODES` |
| cosyvoice ×2 | `COSYVOICE_CLASS`（`CosyVoice2`/`CosyVoice3`）、`COSYVOICE_FP16`<br>`COSYVOICE_MAX_REF_SEC`（參考音檔硬上限秒數，預設 30，來自上游 `frontend._extract_speech_token` 的 assert）<br>`COSYVOICE3_SYSTEM_PROMPT`（只有 `CosyVoice3` 用得到。CosyVoice3 的 LLM 硬性要求輸入含 `<\|endofprompt\|>`，這是接在它前面那段 system prompt，預設 `You are a helpful assistant.`，跟官方 model card 一致）|
| qwen3-tts | `QWEN_ATTN`（預設 `sdpa`，aarch64 上唯一免現場編譯的）、`QWEN_DTYPE`、`QWEN_DEFAULT_SPEAKER`（只有 CustomVoice checkpoint 用得到）<br>生成參數：`QWEN_PROFILE`（`fast`/`balanced`/`quality`，預設 `balanced`）、`QWEN_TEMPERATURE`、`QWEN_TOP_P`、`QWEN_TOP_K`、`QWEN_REPETITION_PENALTY`、`QWEN_SPLIT_MAX_CHARS`、`QWEN_RETRIES`、`QWEN_SEED`（後七個不設 = 照 profile 走）<br>效能：`QWEN_MAX_NEW_TOKENS_CAP`（2048）、`QWEN_PROMPT_CACHE_SIZE`（32）、`QWEN_MAX_BATCH`（8）、`QWEN_JOIN_SILENCE_MS`（120）|
| voxcpm2 | `VOXCPM_OPTIMIZE`（torch.compile，預設關）、`VOXCPM_CFG`、`VOXCPM_TIMESTEPS` |

> `ENGINE_MODES` 是 gateway 路由的依據，gateway **不寫死**任何引擎能力。
> qwen3-tts 從 CustomVoice 換成 Base 就是這樣做的 —— 把 `preset` 改成 `clone`，
> gateway 自動改把克隆音色路由過來，`app.py` 一行都沒動。
> 兩個 checkpoint 的能力互斥，不要寫成 `preset,clone`。

gateway 端：`ENGINES`（`名稱=網址`，逗號分隔）、`DEFAULT_ENGINE`、`API_KEY`、`REQUEST_TIMEOUT`（預設 600 秒）、`VOICES_DIR`。

---

## CLI 包裝對照

腳本底下就是 curl，這張表是對照關係：

| 指令 | 打的端點 |
|---|---|
| `./voice.sh list` | `GET /v1/voices` |
| `./voice.sh add <名稱> <音檔> [逐字稿]` | `POST /v1/voices` (multipart) |
| `./voice.sh design <名稱> "<描述>"` | `POST /v1/voices/design` |
| `./voice.sh show <id>` | `GET /v1/voices/{id}` |
| `./voice.sh transcript <id> "<逐字稿>"` | `PATCH /v1/voices/{id}` |
| `./voice.sh preview <id> [文字]` | `POST /v1/voices/{id}/preview` → `work/results/preview.wav` |
| `./voice.sh rm <id>` | `DELETE /v1/voices/{id}` |
| `./synth.sh "<文字>" [音色] [輸出檔名]` | `POST /v1/audio/speech` → `work/results/` |
| `python3 scripts/batch.py <csv>` | 逐列打 `POST /v1/audio/speech` |

`voice.sh design` 在不支援的三包裡指令還在，但會拿到 400。

**batch.py 的 CSV 欄位**：`text`（必填）、`output`（必填，副檔名會自己補）、`voice`、`instruct`。
參數：`--output-dir` / `--base` / `--api-key` / `--voice`（覆蓋 CSV）/ `--format` / `--speed` / `--timeout` / `--concurrency` / `--overwrite`。
已存在的輸出檔會跳過，重跑不會重做。只用標準函式庫，不用建 venv。

> `--concurrency` 調大幾乎沒有意義：gateway 每顆引擎一個 semaphore，本來就會排隊。

---

## 錯誤碼

| 碼 | 什麼時候 |
|---|---|
| **400** | `input` 空、音檔太短（< 2 秒）/ 轉檔失敗 / 讀不出長度、`response_format` 不支援、音色 type 跟引擎不相容、音色名稱撞名、`default_engine` 不存在、design 給了不支援的引擎、speaker 不在清單、音色庫是空的但引擎需要參考音檔 |
| **401** | 有設 `API_KEY` 但 Authorization header 缺或錯 |
| **404** | 找不到音色 / 找不到引擎 / 想改或刪 preset 音色 |
| **500** | 引擎合成失敗（原始錯誤會原封不動往上傳）、`voices.json` 解析失敗（會先備份成 `voices.broken-<ts>.json` 才報錯，不會把你的音色清單洗成空的） |
| **503** | 連不到引擎 —— 通常是還在載模型 |

引擎的錯誤會被包成 `[<engine>] <原始訊息>` 往上傳，不會變成無資訊的空白 500。

---

## 幾個實作上的落差，先講清楚

| | 狀況 |
|---|---|
| **能力快取 300 秒** | gateway 對引擎 `/health` 有 `CAPS_TTL=300` 的快取。引擎剛起來時 `presets` 是空的，這個空值會被快取住。`POST /v1/warmup` 會清掉它 |
| **voxcpm2 的取樣旋鈕打不到** | `cfg_value` / `inference_timesteps` 只在引擎層，gateway 的 `SpeechRequest` 沒有對應欄位。只能靠環境變數 |
| **qwen3-tts 的 `GET /presets` 沒被用到** | server.py 的註解說「gateway 用這支」，但 gateway 實際是從 `/health` 的 `presets` 欄位拿。這支端點目前只有手動除錯時有用 |
| **中文音色名稱在 header 會變 `?`** | `_hdr()` 做 latin-1 過濾，`X-Voice-Id` / `X-Voice-Type` 對中文名稱不可靠 |
| **gateway 有四份副本** | 改完一包要同步：`for d in models/*/gateway; do cp models/cosyvoice2/gateway/app.py "$d/app.py"; done` |
| **音色庫不共用** | 每包自己的 `work/voices/`。要共用就整個目錄複製過去 |

互動式文件（FastAPI 自動產生，免 token）：`http://localhost:1800X/docs`。
