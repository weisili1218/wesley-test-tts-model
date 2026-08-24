#!/usr/bin/env bash
# 在「目標機器」上先跑這支，確認環境符合預期再開始 build。
# 用法： bash preflight.sh
#
# 這支只檢查 cosyvoice2 這一包要用到的東西。四包的檢查項目一樣，
# 差別只在磁碟/記憶體的建議值 —— 這裡算的是「只跑這一包」。
set -u

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAILED=1; }
FAILED=0

echo "=== 檢查對象： cosyvoice2（0.5B，權重約 2 GB）==="
echo

echo "=== 1. 架構與作業系統 ==="
ARCH=$(uname -m)
echo "  uname -m       : $ARCH"
[ -f /etc/os-release ] && . /etc/os-release && echo "  OS             : ${PRETTY_NAME:-unknown}"
if [ "$ARCH" = "aarch64" ]; then ok "arm64，符合 GB10"; else
  warn "不是 aarch64（${ARCH}）。這份 Dockerfile 是為 GB10/arm64 寫的。"
fi

echo
echo "=== 2. NVIDIA 驅動與 GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${DRV:-0}" -ge 580 ] 2>/dev/null; then ok "driver ${DRV}，支援 CUDA 13"; else
    warn "driver 主版本 ${DRV}，CUDA 13 的 wheel 需要 580 以上，請先更新驅動。"
  fi
else
  bad "找不到 nvidia-smi，請先安裝 NVIDIA driver。"
fi

echo
echo "=== 3. Docker ==="
if command -v docker >/dev/null 2>&1; then
  docker version --format '  client {{.Client.Version}} / server {{.Server.Version}}' 2>/dev/null \
    || bad "docker 指令存在但連不上 daemon（權限或服務沒起來）"
  docker compose version >/dev/null 2>&1 && ok "docker compose 可用" \
    || bad "沒有 docker compose v2（需要 docker-compose-plugin）"
else
  bad "沒有安裝 docker"
fi

echo
echo "=== 4. NVIDIA Container Toolkit（容器內看得到 GPU 嗎）==="
if command -v docker >/dev/null 2>&1; then
  if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/tmp/_gpu.$$ 2>&1; then
    sed 's/^/  /' /tmp/_gpu.$$; ok "容器內看得到 GPU"
  else
    bad "容器內看不到 GPU，請安裝/設定 nvidia-container-toolkit"
    sed 's/^/  | /' /tmp/_gpu.$$ | head -5
  fi
  rm -f /tmp/_gpu.$$
fi

echo
echo "=== 5. 磁碟空間 ==="
echo "  這一包的引擎 image 約 12-14 GB（torch+CUDA 函式庫約 6 GB、權重 2 GB），"
echo "  gateway 不到 0.2 GB。加上 build cache，只跑這一包建議留 60 GB。"
echo "  （四包全部 build 的話要 150 GB，那時候請一包一包做。）"
df -h /var/lib/docker 2>/dev/null | sed 's/^/  /' || df -h / | sed 's/^/  /'
AVAIL=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9')
[ -z "${AVAIL:-}" ] && AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "${AVAIL:-0}" -ge 60 ]; then ok "剩餘 ${AVAIL}G"
elif [ "${AVAIL:-0}" -ge 30 ]; then warn "只剩 ${AVAIL}G，build 會很緊。先 docker system prune -af 清一下。"
else bad "只剩 ${AVAIL}G，不夠 build。先清理或加掛磁碟。"; fi

echo
echo "=== 6. 記憶體 ==="
TOTAL_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
TOTAL_GB=$((TOTAL_KB / 1024 / 1024))
echo "  MemTotal       : ${TOTAL_GB} GB"
echo "  cosyvoice2 常駐約 2 GB 權重（GB10 是 unified memory，CPU/GPU 共用）。"
if [ "$TOTAL_GB" -ge 32 ]; then ok "夠跑這一包"
elif [ "$TOTAL_GB" -ge 16 ]; then warn "勉強夠，同時別再起別包的引擎"
else bad "記憶體太小"; fi

echo
echo "=== 7. 對外連線（build 需要）==="
for host in github.com pypi.org huggingface.co download.pytorch.org; do
  if curl -sS -o /dev/null -m 8 "https://$host" 2>/dev/null; then ok "$host"; else bad "連不到 $host"; fi
done

echo
if [ "$FAILED" -eq 0 ]; then
  echo "全部通過，可以 docker compose build 了。"
else
  echo "有項目失敗，先處理上面的 [FAIL] 再 build。"
  exit 1
fi
