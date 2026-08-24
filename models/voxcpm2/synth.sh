#!/usr/bin/env bash
# 單句合成。底下就是打這一包 gateway 的 /v1/audio/speech。
#
#   ./synth.sh "要合成的文字" [音色] [輸出檔名]
#
# 這一包只有 voxcpm2 一顆引擎，所以不用（也不能）指定引擎 —— 舊版那個
# 第三個「引擎」參數在這裡變成輸出檔名了。
#
# 例：
#   ./synth.sh "今天天氣真好。"                        什麼都不給，讓模型自己決定聲音
#   ./synth.sh "今天天氣真好。" 小美                   用克隆音色
#   ./synth.sh "今天天氣真好。" 溫柔女聲 out-001.wav   用文字描述造的音色
#
# 這顆三種模式都吃（clone / design / 什麼都不給），輸出 48kHz。
#
# 輸出一律落在 work/results/。要打別台機器就設 TTS_GATEWAY。
set -euo pipefail

TEXT="${1:?請給要合成的文字}"
VOICE="${2:-}"
OUT="${3:-output.wav}"

BASE="${TTS_GATEWAY:-http://localhost:8004}"
# 陣列刻意不留空：set -u 底下展開空陣列在舊版 bash 會報 unbound variable
AUTH=(-H "X-Client: tts-scripts")
[ -n "${TTS_API_KEY:-}" ] && AUTH+=(-H "Authorization: Bearer ${TTS_API_KEY}")

mkdir -p work/results

BODY=$(python3 - "$TEXT" "$VOICE" <<'PY'
import json, sys
text, voice = sys.argv[1], sys.argv[2]
body = {"input": text, "model": "voxcpm2", "response_format": "wav"}
if voice:
    body["voice"] = voice
print(json.dumps(body, ensure_ascii=False))
PY
)

HTTP=$(curl -sS "${AUTH[@]}" -X POST "$BASE/v1/audio/speech" \
  -H 'Content-Type: application/json' -d "$BODY" \
  --output "work/results/$OUT" --write-out '%{http_code}')

if [ "$HTTP" != "200" ]; then
  echo "合成失敗（HTTP ${HTTP}）：" >&2
  cat "work/results/$OUT" >&2; echo >&2
  rm -f "work/results/$OUT"
  exit 1
fi

echo "完成： work/results/$OUT"
