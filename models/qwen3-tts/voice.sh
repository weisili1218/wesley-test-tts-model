#!/usr/bin/env bash
# 音色庫的操作包裝，底下就是打這一包 gateway 的 /v1/voices。
#
#   ./voice.sh list                                 列出 9 個內建音色
#   ./voice.sh show <voice_id>                      看單一音色
#   ./voice.sh preview <voice_id> [試聽文字]        試聽，存成 work/results/preview.wav
#
# Qwen3-TTS 的 CustomVoice checkpoint 只有內建音色（preset，唯讀），不能上傳克隆、
# 也不能用文字描述造音色，所以沒有 add / design / transcript / rm 這幾條。
# 真的要克隆的話，把 engine/Dockerfile 的 MODEL_REPO 換成
# Qwen/Qwen3-TTS-12Hz-1.7B-Base、ENGINE_MODES 改成 preset,clone 重 build，
# 再從別包把 add 那段複製過來。
#
# 這一包只有 qwen3-tts 一顆引擎，所以指令裡不用指定引擎（舊版 preview 的
# 「引擎」參數在這裡拿掉了，第二個參數直接是試聽文字）。
#
# 音色存在 work/voices/：voices.json 是索引，<voice_id>.wav 是正規化後的參考音檔。
# 整個資料夾複製走就能搬到別台機器，也可以複製給另外三包用。
set -euo pipefail

BASE="${TTS_GATEWAY:-http://localhost:18003}"
# 陣列刻意不留空：set -u 底下展開空陣列在舊版 bash 會報 unbound variable
AUTH=(-H "X-Client: tts-scripts")
[ -n "${TTS_API_KEY:-}" ] && AUTH+=(-H "Authorization: Bearer ${TTS_API_KEY}")

pretty() { python3 -m json.tool 2>/dev/null || cat; }

cmd="${1:-list}"; shift || true

case "$cmd" in
  list)
    curl -sS "${AUTH[@]}" "$BASE/v1/voices" | pretty
    ;;
  show)
    curl -sS "${AUTH[@]}" "$BASE/v1/voices/${1:?請給 voice_id}" | pretty
    ;;
  preview)
    ID="${1:?請給 voice_id}"; TXT="${2:-}"
    Q=$(python3 -c '
import sys, urllib.parse
print("?" + urllib.parse.urlencode({"text": sys.argv[1]}) if sys.argv[1] else "")' "$TXT")
    mkdir -p work/results
    HTTP=$(curl -sS "${AUTH[@]}" -X POST "$BASE/v1/voices/$ID/preview$Q" \
      --output work/results/preview.wav --write-out '%{http_code}')
    if [ "$HTTP" != "200" ]; then
      echo "試聽失敗（HTTP ${HTTP}）：" >&2; cat work/results/preview.wav >&2; echo >&2
      rm -f work/results/preview.wav; exit 1
    fi
    echo "完成： work/results/preview.wav"
    ;;
  *)
    sed -n '2,11p' "$0"
    exit 1
    ;;
esac
