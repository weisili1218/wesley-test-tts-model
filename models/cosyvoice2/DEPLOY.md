# CosyVoice2-0.5B — 部署到 DGX Spark / GB10

CosyVoice 的前一代，純 zero-shot 克隆。要跟 Fun-CosyVoice3 做 A/B 比較才留著這顆。

**這是一包完全獨立的部署單元**：一顆引擎 + 一個自己的 gateway，跟 `models/` 底下
另外三包零依賴。整個資料夾複製到別台機器就能單獨跑，不用管其他模型。

- 對外入口： `http://localhost:18001`
- 引擎除錯用： `http://localhost:18081`（正常不用碰）
- HF repo： `FunAudioLLM/CosyVoice2-0.5B`

---

## 這一包有什麼

```
cosyvoice2/
├── engine/                 引擎：Dockerfile + requirements.txt + server.py
│                           吃 GPU、把權重烘進 image，約 12-14 GB
├── gateway/                音色庫 + OpenAI 相容 API，不用 GPU，image <200MB
├── compose.yaml            engine + gateway 兩個 service
├── preflight.sh            build 前的環境檢查
├── check-conflicts.sh      跟機器上「原本在跑的東西」有沒有衝突（唯讀）
├── voice.sh                音色管理的指令包裝
├── synth.sh                單句合成的指令包裝
├── scripts/batch.py        批次合成（只用標準函式庫，不用 pip install）
├── DEPLOY.md               這份文件
└── work/
    ├── data/               參考音檔、批次 CSV 放這
    ├── voices/             音色庫（gateway 可寫，引擎唯讀）
    └── results/            產出的音檔
```

架構：

```
你 ──HTTP:18001──▶ gateway ──內部網路──▶ engine :8080
                  音色庫                cosyvoice2
              work/voices/
```

引擎只負責「給我 wav 路徑跟文字，還我音訊」，音色的上傳、命名、保存、刪除全部在
gateway。`gateway/app.py` 四包完全一樣，不寫死引擎名稱 —— 掛哪顆全看 compose 的
`ENGINES` 環境變數，所以改了一包直接複製到另外三包就好。

> **音色庫不共用。** 每包有自己的 `work/voices/`。要讓另一包也用同一批音色，
> 把整個 `work/voices/` 目錄複製過去即可（裡面是 `voices.json` 加幾個 wav）。

---

## 這顆模型能做什麼

| | CosyVoice2-0.5B |
|---|---|
| HF repo | `FunAudioLLM/CosyVoice2-0.5B` |
| 參數量 | 0.5B |
| 輸出取樣率 | 24 kHz |
| 上傳音檔克隆（clone） | ✅ |
| 純文字描述造音色（design） | ❌ |
| 內建精選音色（preset） | ❌ |
| 引擎的 default 模式（不靠任何音色資料發聲） | ❌ |
| 語氣／風格指令（`instructions`） | ✅ |
| `speed` 參數 | ✅ 有效 |

**合成時不給 `voice` 會怎樣：** 自動挑**最近建立的 clone 音色**。音色庫是空的就回 400 要你先上傳。

**什麼時候選這顆：** 想比較新舊兩代克隆品質；或 Fun-CosyVoice3 在你的素材上表現不如預期時的備案。

**起來之後第一件事：** 這顆**沒有內建音色**，音色庫是空的就不能合成。第一件事一定是先上傳一段參考音檔。

---

## ⚠️ 這台 GB10 上還跑著別的東西嗎

如果是**你獨占的機器**，跳過這節直接看 Step 0。

如果機器上**已經在跑別人的服務或訓練工作**，先做完這三件事再往下 —— 這一包的預設值
是為了「方便測試」調的，對共用機器不夠保守。

### 1. 先跑衝突偵測（唯讀，不會改任何東西）

```bash
bash check-conflicts.sh
```

會告訴你：8 個 port（gateway 18001-18004、engine 18081-18084）誰佔著、既有的 docker
容器／compose 專案／image tag 有沒有跟這四包撞名、GPU 上現在有哪些 process 吃了多少
記憶體、`/var/lib/docker` 剩多少、docker daemon 的 default runtime 與 live-restore。

> 這支腳本檢查的是**四包全部**的 port，不只這一包 —— 就算你只打算跑這一包，
> 知道另外三個 port 的狀況也不吃虧。

### 2. 預設值已經調成保守的了

不用改任何檔案，`compose.yaml` 出廠就是：

| 設定 | 預設 | 意思 |
|---|---|---|
| gateway 的 `ports` | `127.0.0.1:18001:8000` | **只有這台機器自己連得到**，同網段的人連不到 |
| engine 的 `ports` | 註解掉 | 引擎不對外。直接打它會繞過 gateway 的排隊機制（引擎自己沒有 request queue），同時兩個請求進去可能把記憶體吃爆 |
| 兩個 service 的 `restart` | `"no"` | 機器重開或 docker daemon 重啟後**不會自動復活**佔住 port |

在 GB10 上（SSH 進去）跑 `./synth.sh`、`./voice.sh`、`scripts/batch.py` 都正常，
因為它們打的就是 `localhost:18001`。

**要從別台機器連**（例如你的筆電、或另一個服務要呼叫它）：

```bash
BIND_ADDR=0.0.0.0 docker compose up -d
```

這時候等於對整個網段開放，而且**預設沒有認證** —— 請一併把 `compose.yaml` 裡
gateway 的 `API_KEY` 取消註解、換成你自己的字串，然後在客戶端設
`export TTS_API_KEY=<同一個字串>`。

**要除錯直接戳引擎**：把 `compose.yaml` 裡 engine 的 `ports:` 跟下面那行取消註解，
就會露出 `18081`。用完記得註解回去。

### 3. 一次只起一包

`deploy.resources.reservations.devices` 的 `count: 1` **不是配額**，只是「把這顆 GPU
露給容器」。GB10 是 unified memory，GPU 配置吃的就是系統那塊 RAM，跟機器上原本的
GPU 工作**共搶同一池**，沒有任何隔離 —— 誰後配置誰 OOM。

好消息是模型**懶載入**：容器起來但還沒收到請求時不吃 GPU 記憶體，只佔 port 和一個
idle 的 python process。真正吃記憶體是在你打 `/v1/warmup` 或第一次合成之後。

---

## Step 0 — 環境檢查

```bash
cd cosyvoice2
bash preflight.sh
```

檢查七件事：架構、驅動版本、Docker、**容器內看不看得到 GPU**、磁碟、記憶體、
build 要連的網域。全部 `[OK]` 才往下走。最關鍵的兩行：

```bash
uname -m          # 要是 aarch64
nvidia-smi        # driver 主版本要 ≥ 580（CUDA 13 的 wheel 需要）
```

`preflight.sh` 看的是「這台機器夠不夠格 build」，`check-conflicts.sh` 看的是
「會不會踩到別人」。兩支互補，共用機器兩支都跑。

> DGX OS 出廠就裝好 Docker 和 NVIDIA Container Toolkit，Step 1、2 可以跳過。
> 只有自己重灌 Ubuntu 24.04 才需要做。

---

## Step 1 — 安裝 Docker（只在自己裝 Ubuntu 的情況需要）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                        docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker "$USER" && newgrp docker
```

注意 `arch=arm64`，不是 amd64。

---

## Step 2 — 安裝 NVIDIA Container Toolkit（只在自己裝 Ubuntu 的情況需要）

> **🔴 機器上有別的容器在跑的話，先不要照做。**
>
> 底下這段有兩行會動到 **host 全域設定**：
> `nvidia-ctk runtime configure` 改寫 `/etc/docker/daemon.json`，
> `systemctl restart docker` 會**把那台機器上所有容器一起彈掉**（除非 daemon 開了
> `live-restore`）。
>
> **先確認到底需不需要做：**
>
> ```bash
> docker info --format '{{.DefaultRuntime}}'    # 已經是 nvidia 就不用做
> docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L   # 過了就不用做
> ```
>
> DGX OS 出廠就裝好了，正常情況**整段跳過**。`check-conflicts.sh` 第 3 節也會
> 告訴你 default runtime 是什麼、live-restore 開了沒。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 驗證（這步過了才有意義往下走）
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
# 應該印出 GPU 0: NVIDIA GB10 ...
```

---

## Step 3 — 把這一包放上去

只要複製這一個資料夾，不用整包 `models/` 都搬：

```bash
# 從你的筆電
scp -r models/cosyvoice2 user@gb10:~/

# 在 GB10 上
cd ~/cosyvoice2
chmod +x preflight.sh check-conflicts.sh voice.sh synth.sh
```

---

## Step 4 — Build

**一定要在 GB10 上原生 build**，不要在 x86 機器上 `--platform linux/arm64`
（那會走 QEMU emulation，慢到不能用）。

> **共用機器注意**：build 會吃滿所有 CPU core 和網路頻寬（pip + HuggingFace 下載
> 好幾 GB），而且 `/var/lib/docker` 撐爆的話會**影響那台機器上所有容器**，不只這一包。
> 別人有工作在跑的話，錯開時間或先跟人講一聲。

```bash
cd ~/cosyvoice2
tmux new -s ttsbuild          # 強烈建議，SSH 斷線不會白做
DOCKER_BUILDKIT=1 docker compose build 2>&1 | tee build.log
```

分開 build（想先確認 gateway 沒問題的話）：

```bash
DOCKER_BUILDKIT=1 docker compose build gateway    # 兩分鐘
DOCKER_BUILDKIT=1 docker compose build engine     # 慢的是這個
```

**預期：30-45 分鐘，image 約 12-14 GB。** 時間花在：

| 階段 | 大概時間 | 在做什麼 |
|---|---|---|
| apt + venv | 1-2 分 | Ubuntu 24.04 內建 Python 3.12，不用 conda |
| pip install torch | 5-15 分 | torch + torchaudio + `nvidia-*-cu13`，約 6 GB |
| pip install 其他相依 | 5-15 分 | 看 `engine/requirements.txt` |
| 下載模型權重 | 5-20 分 | 從 HF 抓 `FunAudioLLM/CosyVoice2-0.5B` |
| 冒煙測試 | <1 分 | build 階段就 import 一次，壞掉不會產出壞 image |

`engine/Dockerfile` 最後有冒煙測試，build log 尾巴應該看到：

```
torch 2.11.0+cu130
ort   1.23.2
wetext OK / CosyVoice2 import OK
```

`COSYVOICE_REF` 可以釘 CosyVoice 原始碼的 commit，讓 build 可重現。

---

## Step 5 — 起服務並驗證 GPU

```bash
docker compose up -d
docker compose ps
```

> 共用機器上**一次只起一包**。四包同時暖機約 13 GB 權重（2+2+4+5），
> 在 unified memory 上跟別人的工作搶同一池，沒有隔離。

先確認 CUDA 真的通：

```bash
docker compose run --rm --entrypoint python engine -c "
import torch
print('torch      :', torch.__version__)
print('cuda avail :', torch.cuda.is_available())
print('device     :', torch.cuda.get_device_name(0))
print('capability :', torch.cuda.get_device_capability(0))
"
```

`capability` 要印出 `(12, 1)`（= sm_121）。

模型是**第一次收到請求時才載入**，所以容器起來很快但第一句會等 30-90 秒。
先主動暖機比較不會嚇到：

```bash
curl -X POST http://localhost:18001/v1/warmup
curl -s http://localhost:18001/v1/models | python3 -m json.tool
```

---

## Step 6 — 音色

### 用指令包裝（最省事）

```bash
# 上傳參考音檔建立克隆音色。音檔什麼格式都行，gateway 會用 ffmpeg 轉成 16k 單聲道
./voice.sh add 小美 work/data/xiaomei.m4a "這裡放這段音檔的逐字稿。"

# 看有哪些音色
./voice.sh list

# 試聽，確認音色對不對再拿去跑批次 → work/results/preview.wav
./voice.sh preview 小美 "這是試聽的句子。"

# 事後補逐字稿（很值得補）
./voice.sh transcript voice_a1b2c3d4e5f6 "補上的逐字稿"

# 刪除（wav 檔會一起刪掉）
./voice.sh rm voice_a1b2c3d4e5f6
```

**逐字稿請盡量給。** 不給的話CosyVoice 會退回 cross-lingual 路徑，
音色相似度會明顯下降。上傳時沒給，回應的 `warnings` 會提醒你。

參考音檔的建議：**5-15 秒、單人、乾淨無背景音樂**。短於 2 秒會被拒絕，
超過 30 秒會警告（模型可能自行截斷）。

### 直接打 API

```bash
curl -X POST http://localhost:18001/v1/voices \
  -F "name=小美" \
  -F "file=@work/data/xiaomei.wav" \
  -F "transcript=這裡放這段音檔的逐字稿。" \
  -F "language=zh"

curl http://localhost:18001/v1/voices
curl -X DELETE http://localhost:18001/v1/voices/voice_a1b2c3d4e5f6
```


音色實際存在 `work/voices/`：`voices.json` 是索引，`<voice_id>.wav` 是正規化後的
參考音檔。整個資料夾複製走就能搬到別台機器，或給另外三包用。

---

## Step 7 — 合成

### 指令包裝

```bash
./synth.sh "歡迎使用語音合成服務。"                   # 用最近建立的音色
./synth.sh "歡迎使用語音合成服務。" 小美              # 指定音色
./synth.sh "歡迎使用語音合成服務。" 小美 out-001.wav  # 指定輸出檔名
# → work/results/
```

### OpenAI 相容 API

```bash
curl -X POST http://localhost:18001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "cosyvoice2",
        "input": "冷氣團南下，北部轉涼。",
        "voice": "小美",
        "instructions": "用平穩的新聞播報語氣",
        "response_format": "wav"
      }' \
  --output speech.wav
```

| 欄位 | 說明 |
|---|---|
| `model` | 這一包只有 `cosyvoice2`。也接受 `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts`，都會對應到它 |
| `voice` | voice id 或你取的音色名稱。不給就自動挑最近建立的 clone 音色 |
| `instructions` | 語氣指示。這顆吃 |
| `response_format` | `wav` / `mp3` / `flac` / `opus` / `aac`，非 wav 由 gateway 用 ffmpeg 轉 |
| `speed` | ✅ 有效 |
| `language` | 選填，不給就讓模型自己判斷 |

回應的 header 會標明實際用了什麼：`X-Engine`、`X-Voice-Id`、`X-Voice-Type`。

Python 端可以直接用 openai 套件：

```python
from openai import OpenAI
# 在 GB10 上跑就用 localhost；從別台機器連的話，服務要先用
# BIND_ADDR=0.0.0.0 docker compose up -d 起來，這裡再換成 http://gb10:18001/v1
client = OpenAI(base_url="http://localhost:18001/v1", api_key="不檢查的話隨便填")
client.audio.speech.create(model="cosyvoice2", voice="小美",
                           input="測試一下。").stream_to_file("out.wav")
```

---

## Step 8 — 批次合成

準備 `work/data/batch.csv`（`work/data/batch.csv.example` 有範例）。
這一包只有一顆引擎，所以 CSV **沒有 engine 欄位**：

```csv
text,output,voice,instruct
今天天氣真好，適合出門走走。,out-001,小美,
下週三下午三點開會，請準時參加。,out-002,小美,用平穩的語氣說
歡迎收聽本集節目，我是主持人。,out-003,小美,用開朗一點的語氣說
```

- `text` 和 `output` 必填，其他可省略
- `output` 不用寫副檔名，會自己補
- **已經存在的輸出檔會跳過**，重跑不會重做；要重來就先刪掉舊檔（或加 `--overwrite`）
- 有任何一列失敗，離開碼是 1，最後會列出是哪幾行、為什麼

```bash
python3 scripts/batch.py work/data/batch.csv
python3 scripts/batch.py work/data/batch.csv --format mp3
python3 scripts/batch.py work/data/batch.csv --voice 小美
```

`scripts/batch.py` 只用 Python 標準函式庫，直接跑就好，不用建 venv。

`--concurrency` 在這一包幾乎沒有意義 —— 只有一顆引擎，gateway 端本來就會排隊
（一個 semaphore），因為引擎自己沒有 request queue，硬併發只會互相拖慢又吃爆記憶體。

另開一個 terminal 看 GPU：`watch -n1 nvidia-smi`

---

## Step 9 — API 一覽

| 方法 | 路徑 | 用途 | 這一包 |
|---|---|---|---|
| GET | `/healthz` | gateway 活著嗎（不需要 token） | ✅ |
| GET | `/v1/models` | 引擎狀態、支援的 mode、內建音色 | ✅ |
| POST | `/v1/warmup` | 主動把模型載進 GPU | ✅ |
| POST | `/v1/audio/speech` | 合成（OpenAI 相容） | ✅ |
| GET | `/v1/voices` | 列音色，可加 `?type=` | ✅ |
| POST | `/v1/voices` | 上傳參考音檔建立 clone 音色（multipart） | ✅ |
| POST | `/v1/voices/design` | 用文字描述建立 design 音色（JSON） | ❌ 不支援 |
| GET | `/v1/voices/{id}` | 單一音色 | ✅ |
| PATCH | `/v1/voices/{id}` | 改名 / 補逐字稿 | ✅ |
| DELETE | `/v1/voices/{id}` | 刪除，連 wav 一起 | ✅ |
| POST | `/v1/voices/{id}/preview` | 試聽，可加 `?text=` | ✅ |

互動式文件在 `http://localhost:18001/docs`。

> 標成「不支援」的那幾條路由還是存在（gateway 四包共用同一份 app.py），
> 只是這顆引擎沒有對應能力，打了會回 400 並附上人看得懂的原因，不會是空白的 500。
>
> 這顆只吃 `clone`，音色庫是空的時候直接合成會回 400 要你先上傳參考音檔。

**要加保護**就在 `compose.yaml` 的 gateway 設 `API_KEY`，之後所有 `/v1/*` 都要帶
`Authorization: Bearer <key>`（`/healthz` 不用）。腳本則設環境變數：

```bash
export TTS_GATEWAY=http://gb10:18001      # 前提：服務用 BIND_ADDR=0.0.0.0 起來
export TTS_API_KEY=sk-xxxx
```

在 GB10 上直接跑腳本的話不用設 `TTS_GATEWAY`，預設就是 `http://localhost:18001`。

---

## Step 10 — 疑難排解

| 症狀 | 原因 / 處理 |
|---|---|
| `failed to import ttsfrd, use wetext instead` | **正常的**，不是錯誤。arm64 沒有 ttsfrd，程式會自動 fallback 到 wetext |
| `Specified provider 'CUDAExecutionProvider' is not in available provider names` | 也是正常 warning。ONNX 那幾個小模型跑 CPU，不影響結果 |
| 第一次請求等很久 | 模型是懶載入。先打 `POST /v1/warmup` |
| `連不到引擎 cosyvoice2`（503） | 引擎還在載模型或根本沒起來。`docker compose ps` 跟 `docker compose logs engine` |
| `torch.cuda.is_available()` 是 False | Step 2 沒做完，或 driver < 580 |
| `no kernel image is available for execution on the device` | 裝到非 cu130 的 torch。確認 `torch.__version__` 尾巴是 `+cu130` |
| 中文變亂碼 | `PYTHONUTF8=1` 沒設。Dockerfile 已經設在 `ENV`，除非你改過 |
| build 到一半磁碟滿了 | **共用主機請勿用 `docker system prune -af --volumes`** —— `-a` 會刪掉別人停用中容器的 image、`--volumes` 會刪掉別人的資料 volume。改用只清 build cache 的 `docker builder prune`，或指名 `docker image rm <ID>`。再確認空間（見 preflight 第 5 項） |
| port 已經被佔用 | 四包的 port 刻意錯開（gateway 18001-18004、引擎 18081-18084）。真的撞到就改 compose 的 `ports` |
| `voices.json 解析失敗` | 檔案壞了，gateway 會先備份成 `voices.broken-<時間>.json` 再報錯，可以手動修 |
| `_pickle.UnpicklingError` / `weights_only` 相關 | torch ≥2.6 改了 `torch.load` 預設值。Dockerfile 已設 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `cosyvoice2 是 zero-shot 克隆模型，一定要參考音檔` | 這顆沒有內建音色。先 `./voice.sh add` 建一個 |
| 產出的聲音不像參考語者 | 逐字稿沒給或給錯；或參考音檔太短／有背景音。建議 5-15 秒、乾淨單人 |

---

## Step 11 — 之後的維運

**備份 image**（換機器不用重 build）：

```bash
docker save tts-cosyvoice2:gb10 | gzip > tts-cosyvoice2-gb10.tar.gz
docker save tts-cosyvoice2-gateway:gb10 | gzip > tts-cosyvoice2-gateway-gb10.tar.gz
# 另一台機器
gunzip -c tts-cosyvoice2-gb10.tar.gz | docker load
```

**清空間**：`docker image prune` **清不掉 build cache**。Dockerfile 裡有三個
`--mount=type=cache,target=/root/.cache/pip`，那些留在 builder 裡，要另外清：

```bash
docker builder prune          # 清 build cache（下次 build 會變慢，但不會壞）
docker system df              # 先看看各類佔多少
```

**備份音色庫**（比 image 重要得多，這是你自己的資料）：

```bash
tar czf voices-backup-$(date +%F).tar.gz work/voices/
```

**換模型**：改 `engine/Dockerfile` 的 `ARG MODEL_REPO` 再重 build，其他三包完全不受影響：

```bash
docker compose build --build-arg MODEL_REPO=<新的 HF repo> engine
```

**改 gateway**：`gateway/app.py` 四包一模一樣。改完記得同步：

```bash
for d in ../*/gateway; do cp gateway/app.py "$d/app.py"; done
```

**想改回四顆共用一個 gateway**：gateway 完全由 `ENGINES` 環境變數驅動，
把四顆引擎都寫進去就好，程式碼一行都不用改：

```yaml
ENGINES: "cosyvoice2=http://cosyvoice2:8080,fun-cosyvoice3=http://fun-cosyvoice3:8080,\
qwen3-tts=http://qwen3-tts:8080,voxcpm2=http://voxcpm2:8080"
```

---

## 備案：GPU 版怎麼都 build 不起來

arm64 + CUDA 13 這條路比較新。真的卡住就先用 CPU 版把流程跑通：

```bash
docker compose build --build-arg TORCH_INDEX=https://pypi.org/simple engine
```

CosyVoice2 是 0.5B，GB10 有 20 顆 ARM core，CPU 跑批次可以接受。
