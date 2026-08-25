#!/usr/bin/env bash
# =============================================================================
# check-conflicts.sh — 上機衝突偵測（唯讀）
#
# 用途：在「已經在跑別的東西」的 GB10 / DGX Spark 上，先確認把 models/ 底下這
#       四包 TTS（gateway 18001-18004、engine 18081-18084）build 起來、跑起來，
#       會不會踩到既有的服務、容器、image、GPU 工作或磁碟。
#
# 用法：
#       bash check-conflicts.sh              # 一般使用者
#       sudo bash check-conflicts.sh         # 想看到「別人的」process 名稱就加 sudo
#
# 保證唯讀：本腳本只做 ss / lsof / docker ps|images|network ls|info|system df /
#           nvidia-smi 查詢 / df / cat。不會 pull、不會 run、不會 prune、
#           不會改任何設定，也不會啟動任何容器。
#
# 相依：只要 bash。ss（或 netstat / lsof）、docker、nvidia-smi 沒有的話會降級
#       成 [WARN] 而不是掛掉。不需要 python、不需要 jq。
# =============================================================================
set -u

# ---- 輸出 -------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_B=$'\033[1m'; C_0=$'\033[0m'
else
  C_G=''; C_Y=''; C_R=''; C_B=''; C_0=''
fi

N_OK=0; N_WARN=0; N_FAIL=0
ok()   { printf '  %s[OK]%s   %s\n'   "$C_G" "$C_0" "$1"; N_OK=$((N_OK+1)); }
warn() { printf '  %s[WARN]%s %s\n'   "$C_Y" "$C_0" "$1"; N_WARN=$((N_WARN+1)); }
bad()  { printf '  %s[FAIL]%s %s\n'   "$C_R" "$C_0" "$1"; N_FAIL=$((N_FAIL+1)); }
info() { printf '         %s\n' "$1"; }
sect() { printf '\n%s=== %s ===%s\n' "$C_B" "$1" "$C_0"; }

GATEWAY_PORTS="18001 18002 18003 18004"
ENGINE_PORTS="18081 18082 18083 18084"
ALL_PORTS="$GATEWAY_PORTS $ENGINE_PORTS"
PROJECTS="cosyvoice2 fun-cosyvoice3 qwen3-tts voxcpm2"
IMAGE_TAGS="tts-cosyvoice2:gb10 tts-cosyvoice2-gateway:gb10
tts-fun-cosyvoice3:gb10 tts-fun-cosyvoice3-gateway:gb10
tts-qwen3-tts:gb10 tts-qwen3-tts-gateway:gb10
tts-voxcpm2:gb10 tts-voxcpm2-gateway:gb10"

HAVE_DOCKER=0
DOCKER_OK=0
HAVE_NVSMI=0

printf '%s' "$C_B"
echo "==========================================================================="
echo " TTS 四包（gateway 18001-18004 / engine 18081-18084）上機衝突檢查 — 唯讀"
echo "==========================================================================="
printf '%s' "$C_0"
echo " 時間     : $(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"
echo " 主機     : $(hostname 2>/dev/null || echo unknown)"
echo " 使用者   : $(id -un 2>/dev/null) (uid=$(id -u 2>/dev/null))"
echo " 架構     : $(uname -m 2>/dev/null) / $(uname -r 2>/dev/null)"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release 2>/dev/null
  echo " OS       : ${PRETTY_NAME:-unknown}"
fi

# =============================================================================
sect "0. 前置：工具可用性"
# =============================================================================
SS_CMD=""
if command -v ss >/dev/null 2>&1; then SS_CMD="ss"
elif command -v netstat >/dev/null 2>&1; then SS_CMD="netstat"
elif command -v lsof >/dev/null 2>&1; then SS_CMD="lsof"
fi
if [ -n "${SS_CMD}" ]; then ok "找得到 ${SS_CMD}，可以查 port 佔用"
else warn "沒有 ss / netstat / lsof，port 檢查會退回只用 docker 的資訊，可能漏掉非容器的服務"; fi

if command -v docker >/dev/null 2>&1; then
  HAVE_DOCKER=1
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    DOCKER_OK=1
    ok "docker 可用（server $(docker version --format '{{.Server.Version}}' 2>/dev/null)）"
  else
    bad "有 docker 指令但連不上 daemon（沒權限或服務沒起來）。後面的 docker 檢查會被跳過。"
  fi
else
  bad "找不到 docker 指令"
fi

if command -v nvidia-smi >/dev/null 2>&1; then HAVE_NVSMI=1; ok "nvidia-smi 可用"
else bad "找不到 nvidia-smi，GPU 相關檢查會被跳過"; fi

if [ "$(id -u 2>/dev/null)" != "0" ]; then
  info "提醒：非 root 執行，ss 只看得到「你自己」的 process 名稱。"
  info "      要知道別人的服務是誰佔著 port，請改用 sudo 再跑一次。"
fi

# =============================================================================
sect "1. Host port 佔用（18001-18004 gateway / 18081-18084 engine）"
# =============================================================================
listeners_for() {
  # $1 = port；印出 raw 的 listening 行（可能多行：IPv4 + IPv6）
  _p="$1"
  case "${SS_CMD}" in
    ss)
      ss -ltnp 2>/dev/null | tail -n +2 | awk -v p="$_p" '
        { addr=$4; n=split(addr, a, ":"); if (a[n] == p) print }'
      ;;
    netstat)
      netstat -ltnp 2>/dev/null | awk -v p="$_p" '
        /^tcp/ { addr=$4; n=split(addr, a, ":"); if (a[n] == p) print }'
      ;;
    lsof)
      lsof -nP -iTCP:"$_p" -sTCP:LISTEN 2>/dev/null | tail -n +2
      ;;
    *) : ;;
  esac
}

docker_publisher_for() {
  # $1 = port；找有沒有既有容器已經把這個 host port publish 出去
  [ "$DOCKER_OK" -eq 1 ] || return 0
  docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}' 2>/dev/null \
    | awk -F'|' -v p=":$1->" '$3 ~ p { print "      ↳ 既有容器： " $1 "  (image " $2 ")\n        ports: " $3 }'
}

PORT_BUSY=0
for port in $ALL_PORTS; do
  case " $GATEWAY_PORTS " in *" ${port} "*) role="gateway 對外入口";; *) role="engine 除錯用";; esac
  out="$(listeners_for "${port}")"
  if [ -n "$out" ]; then
    PORT_BUSY=$((PORT_BUSY+1))
    bad "port ${port}（${role}）已經被佔用"
    printf '%s\n' "$out" | sed 's/^/        /'
    dp="$(docker_publisher_for "${port}")"
    [ -n "$dp" ] && printf '%s\n' "$dp"
  else
    ok "port ${port}（${role}）目前沒人聽"
  fi
done

if [ "$PORT_BUSY" -gt 0 ]; then
  info ""
  info "有 port 被佔用時的處理："
  info "  * 撞到的是 engine port（18081-18084）→ 直接把 compose.yaml 裡 engine 的"
  info "    ports: 兩行註解掉。gateway 走內部網路一樣連得到，不需要對外。"
  info "  * 撞到的是 gateway port（18001-18004）→ 改成沒人用的號碼，例如"
  info "    ports: - \"127.0.0.1:18001:8000\""
  info "  * 已知會撞的常見服務：Triton Inference Server 預設 8000/8001/8002"
  info "    （HTTP/gRPC/metrics），vLLM 預設 8000，很多 dev server 用 8080/8081。"
fi

# 附帶：把 8000-8100 之間所有 listening port 列出來，方便挑替代號碼
if [ -n "${SS_CMD}" ] && [ "${SS_CMD}" = "ss" ]; then
  info ""
  info "參考：目前 8000-8100 之間所有 LISTEN 的 port —"
  busy_range="$(ss -ltn 2>/dev/null | tail -n +2 | awk '
    { n=split($4,a,":"); p=a[n]+0; if (p>=8000 && p<=8100) print p }' | sort -un | tr '\n' ' ')"
  if [ -n "$busy_range" ]; then info "  $busy_range"; else info "  （都是空的）"; fi
fi

# =============================================================================
sect "2. Docker 命名衝突（compose 專案 / 容器 / image tag / network）"
# =============================================================================
if [ "$DOCKER_OK" -eq 1 ]; then

  # --- 2a. 既有 compose 專案 --------------------------------------------------
  EXIST_PROJ="$(docker ps -a --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null \
                 | grep -v '^$' | sort -u)"
  if [ -n "$EXIST_PROJ" ]; then
    info "這台機器上既有的 compose 專案："
    printf '%s\n' "$EXIST_PROJ" | sed 's/^/        - /'
  else
    info "這台機器上目前沒有 compose 專案。"
  fi
  for p in $PROJECTS; do
    if printf '%s\n' "$EXIST_PROJ" | grep -qx "$p"; then
      bad "compose 專案名稱「${p}」已經存在！docker compose up 會把既有的容器當成自己的並重建／刪掉。"
      docker ps -a --filter "label=com.docker.compose.project=$p" \
        --format '        ↳ {{.Names}}  {{.Image}}  {{.Status}}' 2>/dev/null
      info "        處理：改用 -p 換專案名，例如  docker compose -p tts-$p up -d"
    else
      ok "compose 專案名稱「${p}」沒有被佔用"
    fi
  done

  # --- 2b. 容器名稱 ----------------------------------------------------------
  NAME_HIT=0
  ALL_NAMES="$(docker ps -a --format '{{.Names}}' 2>/dev/null)"
  for p in $PROJECTS; do
    for s in engine gateway; do
      for n in "${p}-${s}-1" "${p}_${s}_1"; do
        if printf '%s\n' "$ALL_NAMES" | grep -qx "$n"; then
          bad "容器名稱「${n}」已經存在（會被 compose 重建）"
          NAME_HIT=$((NAME_HIT+1))
        fi
      done
    done
  done
  [ "$NAME_HIT" -eq 0 ] && ok "compose 會產生的 8 個容器名稱都沒有撞到既有容器"

  # --- 2c. image tag ---------------------------------------------------------
  TAG_HIT=0
  ALL_TAGS="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null)"
  for t in $IMAGE_TAGS; do
    if printf '%s\n' "$ALL_TAGS" | grep -qx "$t"; then
      warn "image tag「${t}」已經存在，build 會把這個 tag 蓋過去（舊 image 變成 dangling）"
      docker images --format '        ↳ {{.Repository}}:{{.Tag}}  {{.ID}}  {{.CreatedSince}}  {{.Size}}' 2>/dev/null \
        | grep -F "$t " || true
      TAG_HIT=$((TAG_HIT+1))
    fi
  done
  [ "$TAG_HIT" -eq 0 ] && ok "8 個 image tag（tts-*:gb10）都沒有撞到既有 image"

  # --- 2d. network -----------------------------------------------------------
  NET_HIT=0
  ALL_NETS="$(docker network ls --format '{{.Name}}' 2>/dev/null)"
  for p in $PROJECTS; do
    if printf '%s\n' "$ALL_NETS" | grep -qx "${p}_default"; then
      warn "docker network「${p}_default」已經存在"
      NET_HIT=$((NET_HIT+1))
    fi
  done
  [ "$NET_HIT" -eq 0 ] && ok "四個 <專案>_default network 都不存在，會是新建的"

  info ""
  info "既有 docker network 的網段（新建 4 個 bridge network 會再吃掉 4 段，"
  info "如果跟公司內網／VPN 網段重疊會造成該網段連不上）："
  for n in $(docker network ls --format '{{.Name}}' 2>/dev/null); do
    sub="$(docker network inspect "$n" \
            --format '{{range .IPAM.Config}}{{.Subnet}} {{end}}' 2>/dev/null)"
    [ -n "$sub" ] && info "  $(printf '%-28s' "$n") $sub"
  done
  POOL="$(docker info --format '{{range .DefaultAddressPools}}{{.Base}}/{{.Size}} {{end}}' 2>/dev/null)"
  [ -n "$POOL" ] && info "  預設位址池： $POOL"

  # --- 2e. 目前跑著的容器總覽 -------------------------------------------------
  info ""
  RUNNING_N="$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${RUNNING_N:-0}" -gt 0 ]; then
    warn "這台機器上目前有 ${RUNNING_N} 個容器在跑 —— 動 docker daemon 之前務必確認這些是誰的："
    docker ps --format '        {{.Names}}  |  {{.Image}}  |  {{.Status}}  |  {{.Ports}}' 2>/dev/null
    info ""
    info "  特別注意：DEPLOY.md Step 2 有一行 sudo systemctl restart docker，"
    info "  在 DGX OS 上 nvidia-container-toolkit 已經預裝好，那整個 Step 2 要跳過。"
    info "  真的重啟 daemon 的話，上面這些容器會全部被重啟。"
  else
    ok "目前沒有其他容器在跑"
  fi
else
  warn "跳過 docker 命名衝突檢查（連不上 daemon）"
fi

# =============================================================================
sect "3. Docker daemon 設定（default runtime / live-restore / root dir）"
# =============================================================================
if [ "$DOCKER_OK" -eq 1 ]; then
  DEF_RT="$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"
  RTS="$(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)"
  LR="$(docker info --format '{{.LiveRestoreEnabled}}' 2>/dev/null)"
  CGD="$(docker info --format '{{.CgroupDriver}}' 2>/dev/null)"
  ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)"
  info "DefaultRuntime : ${DEF_RT:-unknown}"
  info "可用 runtimes  : ${RTS:-unknown}"
  info "LiveRestore    : ${LR:-unknown}"
  info "CgroupDriver   : ${CGD:-unknown}"
  info "DockerRootDir  : ${ROOT:-/var/lib/docker}"

  # 注意：「沒有註冊 nvidia runtime」不等於「不能用 GPU」。
  # Docker 19.03 起有原生 GPU 支援：--gpus / deploy.resources.devices 會走
  # DeviceRequests，由 nvidia-container-runtime-hook 掛載，容器 runtime 仍是 runc。
  # DGX OS 出廠就是這種設定。只看 runtime 清單會誤判成 FAIL，而照那個 FAIL 去
  # nvidia-ctk runtime configure + systemctl restart docker，正是本腳本一再警告
  # 不要做的事（會把機器上所有容器彈掉）。
  # 這裡改用唯讀的方式找證據 —— 本腳本承諾不 run 容器，所以不做實測。
  case "$RTS" in
    *nvidia*)
      ok "nvidia runtime 已註冊，四包的 GPU 保留不需要再改 daemon 設定"
      ;;
    *)
      GPU_HOOK=""
      command -v nvidia-container-runtime-hook >/dev/null 2>&1 && GPU_HOOK=1
      GPU_PROOF="$(docker ps -q 2>/dev/null | while read -r c; do
            docker inspect "$c" --format '{{if .HostConfig.DeviceRequests}}{{.Name}}{{end}}' 2>/dev/null
          done | grep -v '^$' | sed 's|^/||' | head -3 | tr '\n' ' ')"
      if [ -n "$GPU_PROOF" ]; then
        ok "沒註冊 nvidia runtime，但這台走的是 Docker 原生 GPU 支援（DeviceRequests）"
        info "        實證：同機已有容器正以此機制使用 GPU —— ${GPU_PROOF}"
        info "        不需要改 daemon 設定，更不要 systemctl restart docker。"
        info "        要親自確認就跑 preflight.sh，它會實際起一個容器測 nvidia-smi。"
      elif [ -n "$GPU_HOOK" ]; then
        warn "沒註冊 nvidia runtime，但找得到 nvidia-container-runtime-hook —— 這台很可能"
        info "        是走 Docker 原生 GPU 支援（DGX OS 預設）。請跑 preflight.sh 實測確認。"
        info "        測得過就不需要改 daemon 設定，也不要 systemctl restart docker。"
      else
        bad "沒註冊 nvidia runtime，也找不到 nvidia-container-runtime-hook。"
        info "        先跑 preflight.sh 實測 docker run --gpus all；真的不行再跟機器擁有者"
        info "        確認 —— 改 daemon 設定要 restart docker，會影響機器上所有容器。"
      fi
      ;;
  esac

  if [ "$DEF_RT" = "nvidia" ]; then
    warn "DefaultRuntime 是 nvidia：連 build 步驟跟 gateway 容器都會拿到 GPU 掛載；"
    info "        compose 裡 engine 的 NVIDIA_VISIBLE_DEVICES: all 在這種設定下會直接"
    info "        看到全部 GPU，繞過任何 count: 1 的限制。建議把那一行拿掉。"
  else
    ok "DefaultRuntime 是 ${DEF_RT:-runc}（非 nvidia），只有明確要求 GPU 的容器才拿得到 GPU"
  fi

  if [ "$LR" = "true" ]; then
    ok "live-restore 開著：重啟 docker daemon 時既有容器不會被殺"
  else
    warn "live-restore 沒開：一旦 restart docker daemon，機器上所有容器都會被重啟。不要跑 DEPLOY.md Step 2 的 systemctl restart docker。"
  fi

  if [ -r /etc/docker/daemon.json ]; then
    info ""
    info "/etc/docker/daemon.json 內容（唯讀顯示）："
    sed 's/^/        /' /etc/docker/daemon.json
  else
    info "（/etc/docker/daemon.json 不存在或沒有讀取權限）"
  fi
else
  warn "跳過 docker daemon 設定檢查"
fi

# =============================================================================
sect "4. GPU 現況（誰在用、用了多少）"
# =============================================================================
if [ "$HAVE_NVSMI" -eq 1 ]; then
  info "GPU 清單："
  nvidia-smi --query-gpu=index,name,driver_version,compute_mode,persistence_mode,memory.total,memory.used,utilization.gpu \
             --format=csv 2>/dev/null | sed 's/^/        /'

  DRV_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  DRV_MAJ="$(printf '%s' "$DRV_FULL" | cut -d. -f1 | tr -dc '0-9')"
  if [ -n "$DRV_MAJ" ] && [ "$DRV_MAJ" -ge 580 ] 2>/dev/null; then
    ok "driver ${DRV_FULL} ≥ 580，支援 CUDA 13 的 wheel，不需要動 host driver"
  else
    bad "driver ${DRV_FULL:-unknown} < 580。這四包用 cu130 的 torch，需要 ≥580。"
    info "        升級 driver 會影響機器上「所有」既有的 GPU 工作 —— 這是最不該擅自做的變更，"
    info "        一定要先跟機器擁有者確認。"
  fi

  CM="$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1)"
  case "$CM" in
    *Default*) ok "compute mode = Default，多個 process 可以同時開 CUDA context" ;;
    *Exclusive*) bad "compute mode = ${CM}：一次只允許一個 process 開 CUDA context。四包的 engine 會互搶，也可能直接卡住既有工作。" ;;
    *Prohibited*) bad "compute mode = Prohibited，沒有 process 能用 GPU" ;;
    *) warn "讀不出 compute mode（${CM:-空})" ;;
  esac

  info ""
  info "目前在 GPU 上的 process："
  APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)"
  if [ -n "$APPS" ]; then
    printf '%s\n' "$APPS" | sed 's/^/        /'
    APPS_N="$(printf '%s\n' "$APPS" | grep -c . )"
    warn "有 ${APPS_N} 個 process 正在用 GPU。四包 engine 各自會再開一個 CUDA context"
    info "        （每個 context 本身就吃數百 MB ~ 1 GB），加上權重："
    info "          cosyvoice2 ~2GB / fun-cosyvoice3 ~2GB / qwen3-tts ~4GB / voxcpm2 ~5GB"
    info "        四包全部暖機起來大約 +20~25 GB。"
  else
    ok "目前沒有 process 在用 GPU（或這台機器的 nvidia-smi 不回報 compute apps）"
    info "        註：GB10 是 unified memory，部分 driver/韌體版本的 compute-apps 查詢會回空的。"
    info "        請一併看下面第 5 項的系統記憶體用量。"
  fi

  info ""
  info "完整 nvidia-smi（含 process 表）："
  nvidia-smi 2>/dev/null | sed 's/^/        /'
else
  warn "跳過 GPU 檢查（沒有 nvidia-smi）"
fi

# =============================================================================
sect "5. 記憶體（GB10 是 unified memory：GPU 吃的就是這一塊）"
# =============================================================================
if [ -r /proc/meminfo ]; then
  MT_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
  MA_KB="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  MT_GB=$(( ${MT_KB:-0} / 1024 / 1024 ))
  MA_GB=$(( ${MA_KB:-0} / 1024 / 1024 ))
  info "MemTotal     : ${MT_GB} GB"
  info "MemAvailable : ${MA_GB} GB"
  info ""
  info "GB10 沒有獨立 VRAM：GPU 配置的記憶體直接從這 ${MT_GB} GB 扣。"
  info "所以四包 engine OOM 的時候，不是只有自己拿到 CUDA OOM，而是整台機器進入"
  info "記憶體壓力，OOM killer 可能去殺「別人」的 process。這點跟一般獨顯不一樣。"
  info ""
  info "四包全部暖機（POST /v1/warmup）之後的粗估常駐量：約 20-25 GB。"
  if [ "$MA_GB" -ge 40 ]; then
    ok "剩餘可用 ${MA_GB} GB，四包同時暖機的空間夠"
  elif [ "$MA_GB" -ge 25 ]; then
    warn "剩餘可用 ${MA_GB} GB。四包全開會很緊，建議一次只 warmup 一包。"
  else
    bad "剩餘可用只有 ${MA_GB} GB，跟既有工作搶記憶體的風險很高。先確認別人在跑什麼。"
  fi
  info ""
  info "目前吃記憶體最多的 10 個 process（RSS）："
  ps -eo rss,pid,user,comm --sort=-rss 2>/dev/null | head -11 \
    | awk 'NR==1{printf "        %10s %8s %-12s %s\n","RSS(KB)","PID","USER","COMMAND"; next}
           {printf "        %10s %8s %-12s %s\n",$1,$2,$3,$4}'
else
  warn "讀不到 /proc/meminfo（不是 Linux？）"
fi

# =============================================================================
sect "6. 磁碟：/var/lib/docker（撐爆會影響整台機器上所有容器）"
# =============================================================================
DOCKER_ROOT="/var/lib/docker"
if [ "$DOCKER_OK" -eq 1 ]; then
  R="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)"
  [ -n "$R" ] && DOCKER_ROOT="$R"
fi
info "Docker root dir : $DOCKER_ROOT"
DF_TARGET="${DOCKER_ROOT}"
df -Pk "${DF_TARGET}" >/dev/null 2>&1 || DF_TARGET="/"
df -h "${DF_TARGET}" 2>/dev/null | sed 's/^/        /'
# df -Pk 是 POSIX 格式（欄位不會換行），GNU / BSD 都吃，比 --output=avail 可攜。
AVAIL_G="$(df -Pk "${DF_TARGET}" 2>/dev/null | awk 'NR==2 { printf "%d", $4/1024/1024 }')"
AVAIL_G="${AVAIL_G:-0}"
info ""
info "需求：engine image 每包 12-18 GB，四包合計約 55-60 GB；再加上 BuildKit 的"
info "      pip cache mount（--mount=type=cache,target=/root/.cache/pip，四包共用"
info "      同一份，torch cu130 + 各家相依）大約 15-25 GB。四包全 build 建議留 150 GB。"
if [ "$AVAIL_G" -ge 150 ]; then ok "剩餘 ${AVAIL_G} GB，四包全 build 都夠"
elif [ "$AVAIL_G" -ge 70 ]; then warn "剩餘 ${AVAIL_G} GB：一次 build 一包還行，四包全上會滿。"
elif [ "$AVAIL_G" -ge 40 ]; then warn "剩餘 ${AVAIL_G} GB：只夠一包，而且很緊。"
else bad "剩餘只有 ${AVAIL_G} GB，build 到一半塞爆 /var/lib/docker 會讓機器上「所有」容器一起出事。"; fi
info ""
info "注意：DEPLOY.md 疑難排解建議的 docker system prune -af --volumes 會刪掉這台"
info "      機器上所有沒在用的 image 跟所有沒掛載的 volume —— 那是別人的資料。"
info "      共用機器上請改用只影響自己的： docker builder prune  或"
info "      docker image rm tts-<模型>:gb10"

if [ "$DOCKER_OK" -eq 1 ]; then
  info ""
  info "docker system df（目前已經用掉多少）："
  docker system df 2>/dev/null | sed 's/^/        /'
  info "        ↑ Build Cache 那一列就是 pip cache mount 住的地方，"
  info "          docker image prune 清不掉它，要 docker builder prune。"
fi

# =============================================================================
sect "7. CPU / build 期間的側效應"
# =============================================================================
NPROC="$(nproc 2>/dev/null || echo '?')"
info "CPU 核心數 : $NPROC"
[ -r /proc/loadavg ] && info "loadavg    : $(cut -d' ' -f1-3 /proc/loadavg)"
info ""
info "build 期間會做的事（每包 30-60 分鐘）："
info "  * apt-get install build-essential（要編 pyworld）→ 吃滿 CPU"
info "  * pip 下載 torch 2.11.0+cu130 + nvidia-*-cu13（約 6 GB）→ 吃滿頻寬"
info "  * git clone --recursive FunAudioLLM/CosyVoice（cosyvoice2 / fun-cosyvoice3）"
info "  * huggingface snapshot_download 權重 2/2/4/5 GB"
info "  四包合計對外下載量大約 40-60 GB。同機有別的工作在跑就別四包一起 build。"
info ""
info "build 不需要 root，但需要在 docker 群組裡 —— 而 docker 群組等同 root 權限。"
if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  ok "目前使用者已經在 docker 群組裡"
else
  if [ "$(id -u 2>/dev/null)" = "0" ]; then ok "目前是 root"
  else warn "目前使用者不在 docker 群組。加進去（usermod -aG docker）等於給這個帳號 root 權限，要機器擁有者同意。"; fi
fi

# =============================================================================
sect "8. 對外連線（build 需要）"
# =============================================================================
if command -v curl >/dev/null 2>&1; then
  for host in github.com pypi.org huggingface.co download.pytorch.org; do
    if curl -sS -o /dev/null -m 8 "https://${host}" >/dev/null 2>&1; then ok "${host} 連得到"
    else bad "連不到 ${host}（build 會失敗）"; fi
  done
else
  warn "沒有 curl，跳過連線檢查"
fi

# =============================================================================
sect "總結"
# =============================================================================
printf '  %s[OK]%s %d   %s[WARN]%s %d   %s[FAIL]%s %d\n\n' \
  "$C_G" "$C_0" "$N_OK" "$C_Y" "$C_0" "$N_WARN" "$C_R" "$C_0" "$N_FAIL"

cat <<'SUMMARY'
  上機前務必自己再確認三件事（這支腳本查不到）：
    1. 機器上跑的東西是誰的、能不能被打擾？（第 2 節列的容器 + 第 4 節的 GPU process）
    2. 千萬不要跑 DEPLOY.md Step 2 的  sudo systemctl restart docker
       —— DGX OS 已經預裝設定好 nvidia-container-toolkit，那步是給自己重灌 Ubuntu 用的。
    3. 千萬不要跑 DEPLOY.md 疑難排解的  docker system prune -af --volumes
       —— 那會刪掉別人的 image 跟 volume。

  建議在 up 之前先改 compose.yaml：
    * restart: unless-stopped  →  restart: "no"      （避免重開機自動佔住 GPU 跟 port）
    * engine 的 ports: 18081-18084 整段刪掉             （gateway 走內部網路就連得到）
    * gateway 的 "1800X:8000" →  "127.0.0.1:1800X:8000" （不要對整個網段開沒有認證的 API）
    * engine 的 NVIDIA_VISIBLE_DEVICES: all 刪掉      （deploy.reservations 已經給了 GPU）
SUMMARY

if [ "$N_FAIL" -gt 0 ]; then exit 1; fi
exit 0
