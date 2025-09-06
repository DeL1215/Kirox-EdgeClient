#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import threading
import traceback
import argparse
from typing import Optional

from task_manager import TaskManager, TaskRequest
from agent_request import send_to_llm
from vad_trigger import VADMicStreamer, play_wav_default_out

# （可選）若沒有 create_instance.py 也不影響執行
try:
    from create_instance import create_instance
except Exception:
    def create_instance(*args, **kwargs):
        return {"ok": False, "reason": "create_instance not available"}

# === 預設 Agent 設定（可被 CLI 覆蓋）===
PROMPT_STYLE = "你是一位具有幽默感和真誠情感的文字創作者，擅長用簡短又帶有情緒衝擊的語句回應問題。你的回答可能帶點傷感，也可能搞笑，但總是貼近人心。"
VOICE_NAME   = "Jay"
TTS_LANG     = "zh"
BOTID        = "d478758a-da9c-42b8-adfc-4ccc794fc046"

# 可選：細調 TTS 首音延遲/效能（視你的 Coqui backend 而定）
TTS_OUTPUT_DEVICE_INDEX: Optional[int] = None
TTS_FRAMES_PER_BUFFER: Optional[int]   = None

# === 錄音/VAD 預設參數（可被 CLI 覆蓋）===
SOURCE_MODE = "input"
INPUT_DEVICE_NAME: Optional[str] = "GENERAL WEBCAM"
INPUT_DEVICE_INDEX: Optional[int] = None

VAD_AGGRESSIVENESS = 2
TARGET_PROC_RATE   = 16000
MAX_SEGMENT_SEC    = 30.0
FRAME_MS           = 10
START_TRIGGER_FRAMES = 3
END_TRIGGER_FRAMES   = 12
RMS_THRESHOLD        = 1600

# 僅供（可選）重播整段檔案時使用；現在我們用邊收邊播，預設不用
OUTPUT_DEVICE_NAME: Optional[str] = None
OUTPUT_DEVICE_INDEX: Optional[int] = None

PLAYBACK_GUARD = threading.Event()

# ===== 手動覆蓋播放裝置（建議放在 import 後、進入主程式前）=====
# 你的裝置清單裡，螢幕喇叭 "VG27AQ3A" 是 index 4（請依自己實機為準）
os.environ["EDGE_OUT_INDEX"] = os.environ.get("EDGE_OUT_INDEX", "4")
os.environ["EDGE_OUT_NAME"]  = os.environ.get("EDGE_OUT_NAME",  "VG27AQ3A")
os.environ["EDGE_FORCE_DEVICE"] = os.environ.get("EDGE_FORCE_DEVICE", "1")  # 嚴格使用指定裝置
os.environ["EDGE_STEREO"] = os.environ.get("EDGE_STEREO", "1")              # 1ch → 2ch 上混
os.environ["EDGE_OUT_GAIN"] = os.environ.get("EDGE_OUT_GAIN", "1.0")        # 播放音量倍率（1.0~2.0）


async def on_done(req: TaskRequest, res: dict) -> None:
    parsed = res.get("parsed", {})
    audio_file = res.get("audio_file", "")
    print(f"[DONE] kind={req.kind} message={parsed.get('Message')} tool={parsed.get('Tool')} action={parsed.get('Action')}")
    if audio_file and os.path.exists(audio_file):
        print(f"[DONE] TTS saved to: {audio_file}")
        # 已串流播放，不再重播整段音檔；若要驗證檔案，可解註下列段落：
        # await asyncio.to_thread(play_wav_default_out, audio_file, OUTPUT_DEVICE_NAME, OUTPUT_DEVICE_INDEX)


async def on_error(req: TaskRequest, err: Exception) -> None:
    print(f"[ERROR] kind={getattr(req, 'kind', '?')} payload_keys={list(getattr(req, 'payload', {}).keys())}")
    print(f"[ERROR] type={type(err).__name__} msg={err}")
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    print(f"[ERROR] traceback:\n{tb}")


def _apply_env_overrides(args: argparse.Namespace) -> None:
    """
    將 CLI 參數轉成環境變數，交給 agent_request.py 的播放器使用。
    - EDGE_OUT_INDEX / EDGE_OUT_NAME / EDGE_STEREO
    """
    if args.out_index is not None:
        os.environ["EDGE_OUT_INDEX"] = str(args.out_index)
    if args.out_name:
        os.environ["EDGE_OUT_NAME"] = args.out_name
    os.environ["EDGE_STEREO"] = "1" if args.stereo else "0"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kirox Edge Client — 手動指定輸出裝置與各種超參數"
    )

    # 連線/Agent
    p.add_argument("--uri", default="wss://agent.xbotworks.com/ws", help="WebSocket URI（例如 ws://127.0.0.1:9600/ws）")
    p.add_argument("--botid", default=BOTID, help="Agent/使用者識別 ID")
    p.add_argument("--prompt-style", default=PROMPT_STYLE, help="Agent 的提示風格（PROMPT_STYLE）")
    p.add_argument("--voice-name", default=VOICE_NAME, help="TTS 聲音名稱（對應 voice_map）")
    p.add_argument("--tts-lang", default=TTS_LANG, help="TTS 語言代碼")
    p.add_argument("--tts-out-index", type=int, default=TTS_OUTPUT_DEVICE_INDEX, help="TTS 後端的輸出裝置 index（Coqui 直出播放用；通常 None）")
    p.add_argument("--tts-fpb", type=int, default=TTS_FRAMES_PER_BUFFER, help="TTS backend frames_per_buffer（數小先出快，但吃 CPU）")

    # 輸出喇叭（最重要）
    p.add_argument("--out-index", type=int, default=None, help="（建議）指定目前要用的喇叭 sounddevice index（覆蓋 Edge 播放器）")
    p.add_argument("--out-name", type=str, default=None, help="（建議）用子字串匹配喇叭名稱（例如 'VG27AQ3A'）")
    p.add_argument("--stereo", action="store_true", default=True, help="1ch 自動上混 2ch（預設啟用）")
    p.add_argument("--mono", dest="stereo", action="store_false", help="停用上混，維持 1ch")

    # VAD / 錄音
    p.add_argument("--input-name", default=INPUT_DEVICE_NAME, help="輸入裝置名稱子字串（優先）")
    p.add_argument("--input-index", type=int, default=INPUT_DEVICE_INDEX, help="輸入裝置 index（覆蓋名稱搜尋）")
    p.add_argument("--vad", type=int, default=VAD_AGGRESSIVENESS, help="WebRTC VAD aggressiveness 0~3")
    p.add_argument("--frame-ms", type=int, default=FRAME_MS, help="VAD 框長毫秒（常見 10/20/30）")
    p.add_argument("--start-frames", type=int, default=START_TRIGGER_FRAMES, help="起音觸發：連續有聲幀數")
    p.add_argument("--end-frames", type=int, default=END_TRIGGER_FRAMES, help="結束觸發：連續無聲幀數")
    p.add_argument("--max-sec", type=float, default=MAX_SEGMENT_SEC, help="單段音訊最長秒數")
    p.add_argument("--proc-rate", type=int, default=TARGET_PROC_RATE, help="VAD 處理採樣率（8k/16k/32k/48k）")
    p.add_argument("--rms-thresh", type=int, default=RMS_THRESHOLD, help="RMS 門檻，避免微弱底噪觸發")

    # 其他
    p.add_argument("--outputs-dir", default="outputs", help="入/出音檔儲存資料夾")
    p.add_argument("--timeout", type=float, default=180.0, help="單次請求逾時秒數")
    p.add_argument("--debug", action="store_true", help="開啟詳細除錯（傳遞到 Edge 播放器）")

    return p


async def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 將 CLI 參數映射到環境變數（Edge 播放器讀取）
    _apply_env_overrides(args)

    # 1) 先用 REST upsert（可略過，亦可在 WS 首包 upsert）
    try:
        _ = create_instance(
            botid=args.botid,
            prompt_style=args.prompt_style,
            voice_name=args.voice_name,
            language=args.tts_lang,
            output_device_index=args.tts_out_index,
            frames_per_buffer=args.tts_fpb,
            server_url="https://agent.xbotworks.com/create_instance",
        )
    except Exception as e:
        print(f"[WARN] create_instance failed: {e}")

    # 2) 任務管理器
    tm = TaskManager(
        send_coroutine=send_to_llm,
        timeout_seconds=float(args.timeout),
        on_done=on_done,
        on_error=on_error
    )
    loop = asyncio.get_running_loop()

    # 3) 啟動 VAD 錄音器
    vad = VADMicStreamer(
        task_manager=tm,
        loop=loop,
        outputs_dir=args.outputs_dir,
        uri=args.uri,
        vad_aggressiveness=int(args.vad),
        frame_ms=int(args.frame_ms),
        start_trigger_frames=int(args.start_frames),
        end_trigger_frames=int(args.end_frames),
        max_segment_sec=float(args.max_sec),
        target_proc_rate=int(args.proc_rate),
        rms_threshold=int(args.rms_thresh),
        botid=args.botid,
        source_mode=SOURCE_MODE,
        input_device_name=args.input_name,
        input_device_index=args.input_index,
        playback_guard=PLAYBACK_GUARD,
    )
    vad.start()
    print("[VAD] Started. Speak... (Ctrl+C to exit)")
    print(f"[CFG] WS={args.uri}  OUT_INDEX={os.getenv('EDGE_OUT_INDEX')}  OUT_NAME={os.getenv('EDGE_OUT_NAME')}  "
          f"STEREO={os.getenv('EDGE_STEREO')}  GAIN={os.getenv('EDGE_OUT_GAIN')}  DEBUG={args.debug}")

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vad.stop()
        await tm.wait_idle()


if __name__ == "__main__":
    asyncio.run(main())
