#!/usr/bin/env bash
# 音色庫的操作包裝，底下就是打這一包 gateway 的 /v1/voices。
#
#   ./voice.sh list                                 列出所有音色
#   ./voice.sh add <名稱> <音檔> [逐字稿]            上傳參考音檔建立克隆音色
#   ./voice.sh show <voice_id>                      看單一音色
#   ./voice.sh transcript <voice_id> "<逐字稿>"     事後補逐字稿
#   ./voice.sh preview <voice_id> [試聽文字]        試聽，存成 work/results/preview.wav
#   ./voice.sh prepare <voice_id>                   叫引擎把參考特徵先抽好（建音色時已自動做過）
#   ./voice.sh rm <voice_id>                        刪除
#
# 這一包掛的是 Qwen3-TTS-12Hz-1.7B-Base：純克隆，沒有內建音色，所以第一件事一定是
# add 一個。音檔可以是任意格式（wav/mp3/m4a/flac...），gateway 會用 ffmpeg 轉成
# 16k 單聲道。長度建議 5-15 秒，短於 2 秒會被擋掉。
# 逐字稿請盡量給 —— Qwen 的 generate_voice_clone 會拿它去對齊參考音檔，不給相似度會掉。
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
  add)
    NAME="${1:?用法： ./voice.sh add <名稱> <音檔> [逐字稿]}"
    FILE="${2:?請給參考音檔路徑}"
    TEXT="${3:-}"
    [ -f "$FILE" ] || { echo "找不到檔案：$FILE" >&2; exit 1; }
    [ -z "$TEXT" ] && echo "提醒：沒給逐字稿，音色相似度會下降。之後可以用 ./voice.sh transcript 補。" >&2
    curl -sS "${AUTH[@]}" -X POST "$BASE/v1/voices" \
      -F "name=$NAME" -F "file=@$FILE" -F "transcript=$TEXT" | pretty
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
  prepare)
    # 建立音色時 gateway 已經自動推過一次了。這支是給這幾種情況補打的：
    # 引擎重啟過（快取在引擎的記憶體裡，重啟就沒了）、建音色時引擎還沒載模型、
    # 或音色是整個 work/voices/ 從別台機器複製過來的。
    # load_model=true 會讓引擎為了建快取去載模型（要等 30-90 秒）。
    ID="${1:?請給 voice_id}"; LOAD="${2:-false}"
    curl -sS "${AUTH[@]}" -X POST \
      "$BASE/v1/voices/$ID/prepare?load_model=$LOAD" | pretty
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
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
