#!/usr/bin/env python3
"""
批次合成：讀一份 CSV，逐列打這一包的 gateway，把音檔寫到輸出資料夾。

只用 Python 標準函式庫，所以在 GB10 上直接 `python3 scripts/batch.py` 就能跑，
不用建 venv、不用 pip install。

這一包只有 qwen3-tts 一顆引擎，所以 CSV 沒有 engine 欄位、也沒有 --engine 參數
（舊版多引擎才需要）。要比較不同模型就到別包的資料夾各跑一次。

CSV 欄位（第一列是標題，順序不拘，缺的欄位用預設值）：

    text        必填，要合成的文字
    output      必填，輸出檔名（不用寫副檔名，會自己補）
    voice       音色：內建 speaker 名稱，例如 Vivian（./voice.sh list 看全部）
    instruct    語氣指示，例如「用開心一點的語氣說」

範例：

    text,output,voice,instruct
    歡迎收聽本集節目，我是主持人。,out-001,Vivian,用開朗一點的語氣說
    下週三下午三點開會，請準時參加。,out-002,Ethan,

用法：

    python3 scripts/batch.py work/data/batch.csv
    python3 scripts/batch.py work/data/batch.csv --format mp3
    python3 scripts/batch.py work/data/batch.csv --voice Vivian

已經存在的輸出檔會跳過，重跑不會重做；要重新產生就先刪掉舊檔（或加 --overwrite）。
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENGINE = "qwen3-tts"
FORMAT_EXT = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "opus": ".opus", "aac": ".aac"}


def synth(base, api_key, payload, dst, timeout):
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"連不到 gateway：{e.reason}") from None
    # 先寫暫存檔再 rename，中途中斷不會留下半個檔案被下次誤判成「已完成」
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(audio)
    os.replace(tmp, dst)
    return len(audio)


def main():
    ap = argparse.ArgumentParser(description=f"批次合成（{ENGINE}）")
    ap.add_argument("csv_file", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("work/results"))
    ap.add_argument("--base", default=os.environ.get("TTS_GATEWAY", "http://localhost:18003"))
    ap.add_argument("--api-key", default=os.environ.get("TTS_API_KEY", ""))
    ap.add_argument("--voice", default=None, help="覆蓋 CSV 裡的 voice 欄位")
    ap.add_argument("--format", default="wav", choices=sorted(FORMAT_EXT))
    ap.add_argument("--speed", type=float, default=1.0, help="這顆引擎會忽略 speed，留著只是為了 CSV 相容")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同時幾列。這一包只有一顆引擎，gateway 端本來就會排隊"
                         "（每顆引擎一個 semaphore），所以調大幾乎沒有意義")
    ap.add_argument("--overwrite", action="store_true", help="不要跳過已存在的輸出檔")
    args = ap.parse_args()

    if not args.csv_file.exists():
        sys.exit(f"找不到 CSV：{args.csv_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ext = FORMAT_EXT[args.format]

    with args.csv_file.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("CSV 沒有資料列")
    missing = {"text", "output"} - set(rows[0])
    if missing:
        sys.exit(f"CSV 缺少必要欄位：{sorted(missing)}")

    jobs, skipped = [], 0
    for i, row in enumerate(rows, start=2):  # 2 = 扣掉標題列的實際行號
        text = (row.get("text") or "").strip()
        name = (row.get("output") or "").strip()
        if not text or not name:
            print(f"  第 {i} 行： text 或 output 是空的，跳過")
            continue
        dst = args.output_dir / (name if name.endswith(ext) else name + ext)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        payload = {"input": text, "model": ENGINE,
                   "response_format": args.format, "speed": args.speed}
        voice = args.voice or (row.get("voice") or "").strip()
        instruct = (row.get("instruct") or "").strip()
        if voice:
            payload["voice"] = voice
        if instruct:
            payload["instructions"] = instruct
        jobs.append((i, payload, dst))

    print(f"共 {len(rows)} 列，要做 {len(jobs)} 列，跳過已存在的 {skipped} 列")
    if not jobs:
        return

    failures = []

    def run(job):
        i, payload, dst = job
        try:
            size = synth(args.base, args.api_key, payload, dst, args.timeout)
            print(f"  [OK]   第 {i} 行 → {dst}（{size / 1024:.0f} KB）")
        except Exception as e:
            print(f"  [FAIL] 第 {i} 行 → {e}")
            failures.append((i, str(e)))

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(run, jobs))

    print(f"\n完成 {len(jobs) - len(failures)} / {len(jobs)}，輸出在 {args.output_dir}")
    if failures:
        print("失敗的列：")
        for i, err in failures:
            print(f"  第 {i} 行： {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
