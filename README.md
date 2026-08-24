# TTS on DGX Spark / GB10 — 一個模型一包 Docker

四顆 TTS 模型，**每顆各自一個完全獨立的資料夾**，各自一份 Dockerfile、compose、
gateway、腳本與文件。任何一包單獨複製到別台機器就能跑，不用管其他三包。

```
models/
├── cosyvoice2/       CosyVoice2-0.5B            gateway :8001  引擎 :8081
├── fun-cosyvoice3/   Fun-CosyVoice3-0.5B-2512   gateway :8002  引擎 :8082
├── qwen3-tts/        Qwen3-TTS-1.7B CustomVoice gateway :8003  引擎 :8083
└── voxcpm2/          VoxCPM2 (OpenBMB, 2B)      gateway :8004  引擎 :8084
```

每一包裡面都長一樣：

```
<模型名>/
├── engine/           引擎：Dockerfile + requirements.txt + server.py（吃 GPU，權重烘在 image 裡）
├── gateway/          音色庫 + OpenAI 相容 API（不用 GPU，image <200MB）
├── compose.yaml      engine + gateway 兩個 service
├── preflight.sh      build 前的環境檢查（這台機器夠不夠格 build）
├── check-conflicts.sh 衝突偵測（會不會踩到機器上原本在跑的東西，唯讀）
├── voice.sh          音色管理
├── synth.sh          單句合成
├── scripts/batch.py  批次合成（只用標準函式庫）
├── DEPLOY.md         這一包的完整部署文件 ← 從這裡開始讀
└── work/             data/ 素材、voices/ 音色庫、results/ 產出
```

四包對外的 API 完全一樣（`gateway/app.py` byte 相同），差別只有 port 跟底下那顆引擎
支援哪些音色模式。跨四包的完整 API 參考在 [`API.md`](API.md)；
只要看某一包的簡表就翻它自己的 `DEPLOY.md` Step 9。

---

## 怎麼開始

挑一包，進去，照著它的 `DEPLOY.md` 做：

```bash
cd models/fun-cosyvoice3
bash check-conflicts.sh           # 機器上有別的東西在跑的話，先跑這支
bash preflight.sh                 # 環境檢查，全部 [OK] 才往下
docker compose build              # 30-60 分鐘，一定要在 GB10 上原生 build
docker compose up -d
curl -X POST http://localhost:8002/v1/warmup
```

Port 刻意錯開，記憶體夠的話四包可以同時起來。**預設只綁 `127.0.0.1`**，
要從別台機器連就用 `BIND_ADDR=0.0.0.0 docker compose up -d`。

---

## 該挑哪一包

| | 選它的理由 | 要先準備參考音檔嗎 |
|---|---|---|
| **fun-cosyvoice3** | 克隆品質最好。只想試一顆的話就試這顆 | 要（5-15 秒乾淨單人） |
| **cosyvoice2** | 前一代，拿來跟上面那顆做 A/B 比較 | 要 |
| **qwen3-tts** | 內建 9 個精選音色，**開箱即用**，最快看到結果 | 不用 |
| **voxcpm2** | 功能最全：48kHz、30 種語言，可用一句文字描述造音色 | 不用（三種模式都吃） |

**第一次上機建議的順序**：先 `qwen3-tts`（不用準備素材，最快確認整條路通不通），
再 `fun-cosyvoice3`（實際要用的克隆品質），其他兩顆看需求。

---

## ⚠️ 這台 GB10 上還跑著別的東西嗎

獨占的機器跳過這節。**共用**的話先跑衝突偵測（唯讀，不會改任何東西）：

```bash
bash check-conflicts.sh
```

會查：8 個 port 誰佔著（含是哪個 process）、既有 docker 容器／compose 專案／image tag
有沒有跟這四包撞名、GPU 上現在誰在跑吃了多少、`/var/lib/docker` 剩多少、
docker daemon 的 default runtime 與 live-restore。四包裡各有一份相同的副本。

**`compose.yaml` 的預設值已經調成保守的了**，不用改任何檔案：

| 設定 | 預設 | 意思 |
|---|---|---|
| gateway `ports` | `127.0.0.1:800X:8000` | 只有機器自己連得到，同網段的人連不到 |
| engine `ports` | 註解掉 | 引擎不對外（直接打它會繞過 gateway 的排隊機制） |
| `restart` | `"no"` | 機器重開後不會自動復活佔住 port |

SSH 進 GB10 跑 `./synth.sh`、`./voice.sh`、`batch.py` 都正常。**要從別台機器連**：

```bash
BIND_ADDR=0.0.0.0 docker compose up -d    # 這時請一併把 compose 裡的 API_KEY 打開
```

還有兩件事要自己注意：

| | 風險 | 怎麼處理 |
|---|---|---|
| **不要照 `DEPLOY.md` Step 2 做** | 那段的 `sudo systemctl restart docker` 會**把整台機器上所有容器一起彈掉**，`nvidia-ctk runtime configure` 還會改寫 `/etc/docker/daemon.json` | DGX OS 出廠就裝好了。先 `docker info --format '{{.DefaultRuntime}}'`，是 `nvidia` 就整段跳過 |
| **GPU 沒有隔離** | `count: 1` 不是配額。GB10 是 unified memory，GPU 吃的就是系統 RAM，跟別人的工作搶同一池，誰後配置誰 OOM | **一次只起一包**。模型是懶載入，容器起來沒收到請求時不吃 GPU 記憶體 |

各包 `DEPLOY.md` 開頭有完整版。

---

## 這樣拆的代價，先講清楚

| | 說明 |
|---|---|
| 磁碟 | 每包的引擎 image 12-18 GB。四包全 build 要留 150 GB，一包一包做則 60-70 GB。`/var/lib/docker` 撐爆會影響**整台機器上所有容器**；`docker image prune` 清不掉 build cache，要 `docker builder prune` |
| gateway 有四份副本 | `gateway/app.py` **四包 byte 相同**。改完一包要同步：`for d in models/*/gateway; do cp models/cosyvoice2/gateway/app.py "$d/app.py"; done` |
| 音色庫不共用 | 每包有自己的 `work/voices/`。要共用就把整個目錄複製過去（`voices.json` + 幾個 wav） |
| 沒有跨模型的統一入口 | 四個 port 各打各的。要一個入口就見下面 |

## 想改回「一個 gateway 接四顆」

gateway 完全由 `ENGINES` 環境變數驅動，程式碼一行都不用改 —— 起一個 gateway，
把四顆引擎的位址都寫進去就好：

```yaml
ENGINES: "cosyvoice2=http://cosyvoice2:8080,fun-cosyvoice3=http://fun-cosyvoice3:8080,\
qwen3-tts=http://qwen3-tts:8080,voxcpm2=http://voxcpm2:8080"
DEFAULT_ENGINE: fun-cosyvoice3
```

（前提是四顆引擎在同一個 docker network 上，且服務名稱對得起來。）

---

## 為什麼一定要拆成四個 image

不是為了好看，是因為相依真的衝突：

| 引擎 | 關鍵相依 | 衝突點 |
|---|---|---|
| CosyVoice2 / Fun-CosyVoice3 | `transformers==4.51.3` + `wetext` | 官方釘死 4.51.3 |
| Qwen3-TTS | `transformers==4.57.3` | `qwen-tts` 套件硬性釘死，跟上面直接對撞 |
| VoxCPM2 | `funasr` + `torchcodec` + `gradio>=6` | 自己一大包相依樹 |

塞在同一個 venv 裡一定會有人被降級。

四顆共通的 GB10 處理方式（各包的 Dockerfile 開頭都有詳細註解）：

- torch 用 `download.pytorch.org/whl/cu130` 的 **2.11.0 aarch64 wheel** —— GB10 是
  Blackwell sm_121，官方釘的 2.3.1+cu121 沒有 aarch64 build 也不認得 sm_121
- onnxruntime 用 **CPU 版** —— PyPI 沒有 aarch64 的 GPU wheel
- CosyVoice 那兩顆 fallback 到 **wetext**（ttsfrd 只有 x86_64 wheel）
- 權重在 build 階段就烘進 image，跑的時候 `HF_HUB_OFFLINE=1` 完全離線
- **一定要在 GB10 上原生 build**，不要用 `--platform` 走 QEMU emulation
