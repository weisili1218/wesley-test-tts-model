#!/usr/bin/env bash
# 音色庫的操作包裝，底下就是打這一包 gateway 的 /v1/voices。
#
#   ./voice.sh list                                 列出所有音色
#   ./voice.sh add <名稱> <音檔> [逐字稿]            上傳參考音檔建立克隆音色
#   ./voice.sh design <名稱> "<音色文字描述>"        不用音檔，純文字描述造音色
#   ./voice.sh show <voice_id>                      看單一音色
#   ./voice.sh transcript <voice_id> "<逐字稿>"     事後補逐字稿
#   ./voice.sh preview <voice_id> [試聽文字]        試聽，存成 work/results/preview.wav
#   ./voice.sh rm <voice_id>                        刪除
#
# VoxCPM2 是四包裡唯一三種模式都吃的：克隆（add）、文字描述（design）、
# 什麼都不給（synth.sh 不帶音色）。
# 音檔可以是任意格式（wav/mp3/m4a/flac...），gateway 會用 ffmpeg 轉成 16k 單聲道。
# 逐字稿請盡量給 —— 不給的話少掉 ultimate cloning，音色相似度明顯下降。
#
# 這一包只有 voxcpm2 一顆引擎，所以指令裡不用指定引擎（舊版 preview 的
# 「引擎」參數在這裡拿掉了，第二個參數直接是試聽文字）。
#
# 音色存在 work/voices/：voices.json 是索引，<voice_id>.wav 是正規化後的參考音檔。
# 整個資料夾複製走就能搬到別台機器，也可以複製給另外三包用。
set -euo pipefail

BASE="${TTS_GATEWAY:-http://localhost:8004}"
# 陣列刻意不留空：set -u 底下展開空陣列在舊版 bash 會報 unbound variable
AUTH=(-H "X-Client: tts-scripts")
[ -n "${TTS_API_KEY:-}" ] && AUTH+=(-H "Authorization: Bearer ${TTS_API_KEY}")

pretty() { python3 -m json.tool 2>/dev/null || cat; }

cmd="${1:-list}"; shift || true

case "$cmd" in
  list)
    curl -sS "${AUTH[@]}" "$BASE/v1/voices" | pretty
    ;;
  add)
    NAME="${1:?用法： ./voice.sh add <名稱> <音檔> [逐字稿]}"
    FILE="${2:?請給參考音檔路徑}"
    TEXT="${3:-}"
    [ -f "$FILE" ] || { echo "找不到檔案：$FILE" >&2; exit 1; }
    [ -z "$TEXT" ] && echo "提醒：沒給逐字稿，音色相似度會下降。之後可以用 ./voice.sh transcript 補。" >&2
    curl -sS "${AUTH[@]}" -X POST "$BASE/v1/voices" \
      -F "name=$NAME" -F "file=@$FILE" -F "transcript=$TEXT" | pretty
    ;;
  design)
    NAME="${1:?用法： ./voice.sh design <名稱> \"<音色描述>\"}"
    DESC="${2:?請給音色的文字描述}"
    curl -sS "${AUTH[@]}" -X POST "$BASE/v1/voices/design" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"name":sys.argv[1],"description":sys.argv[2]}))' "$NAME" "$DESC")" \
      | pretty
    ;;
  show)
    curl -sS "${AUTH[@]}" "$BASE/v1/voices/${1:?請給 voice_id}" | pretty
    ;;
  transcript)
    ID="${1:?請給 voice_id}"; TEXT="${2:?請給逐字稿}"
    curl -sS "${AUTH[@]}" -X PATCH "$BASE/v1/voices/$ID" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"transcript":sys.argv[1]}))' "$TEXT")" \
      | pretty
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
  rm)
    curl -sS "${AUTH[@]}" -X DELETE "$BASE/v1/voices/${1:?請給 voice_id}" | pretty
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 1
    ;;
esac
