#!/usr/bin/env bash
# 單句合成。底下就是打這一包 gateway 的 /v1/audio/speech。
#
#   ./synth.sh "要合成的文字" [音色] [輸出檔名]
#
# 這一包只有 qwen3-tts 一顆引擎，所以不用（也不能）指定引擎 —— 舊版那個
# 第三個「引擎」參數在這裡變成輸出檔名了。
#
# 例：
#   ./synth.sh "今天天氣真好。"                              用最近建立的克隆音色
#   ./synth.sh "今天天氣真好。" "$VOICE_ID"                   指定音色
#   ./synth.sh "今天天氣真好。" "$VOICE_ID" out-001.wav       指定音色與輸出檔名
#
# 第二個參數是 POST /v1/voices 回傳的 voice.id（voice_xxxxxxxxxxxx）。填 add 當初給的
# 名稱也行，但撞名會回 400 叫你改用 id —— 自動化腳本一律用 id。./voice.sh list 可以查。
#
# 這顆掛的是 Base checkpoint，沒有內建音色 —— 音色庫空的時候會回 400。
# 先 ./voice.sh add "<名稱>" <音檔> "<逐字稿>" 建一個，再回來合成。
#
# 輸出一律落在 work/results/。要打別台機器就設 TTS_GATEWAY。
set -euo pipefail

TEXT="${1:?請給要合成的文字}"
VOICE="${2:-}"
OUT="${3:-output.wav}"

BASE="${TTS_GATEWAY:-http://localhost:18003}"
# 陣列刻意不留空：set -u 底下展開空陣列在舊版 bash 會報 unbound variable
AUTH=(-H "X-Client: tts-scripts")
[ -n "${TTS_API_KEY:-}" ] && AUTH+=(-H "Authorization: Bearer ${TTS_API_KEY}")

mkdir -p work/results

BODY=$(python3 - "$TEXT" "$VOICE" <<'PY'
import json, sys
text, voice = sys.argv[1], sys.argv[2]
body = {"input": text, "model": "qwen3-tts", "response_format": "wav"}
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
