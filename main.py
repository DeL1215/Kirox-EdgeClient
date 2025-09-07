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
from vad_trigger import VADMicStreamer  # 已換 Silero VAD
# 可選：沒有 create_instance 也不影響
try:
    from create_instance import create_instance
except Exception:
    def create_instance(*args, **kwargs):
        return {"ok": False, "reason": "create_instance not available"}

# ===== 預設值（可被 CLI 覆蓋） =====
PROMPT_STYLE = "你是一位具有幽默感和真誠情感的文字創作者，擅長用簡短又帶有情緒衝擊的語句回應問題。"
VOICE_NAME   = "Jay"
TTS_LANG     = "zh"
BOTID        = "d478758a-da9c-42b8-adfc-4ccc794fc046"

# 錄音 / VAD（Silero 友善預設）
SOURCE_MODE            = "input"
INPUT_DEVICE_NAME      = "GENERAL WEBCAM"
INPUT_DEVICE_INDEX     = None
FRAME_MS               = 32        # 32ms@16k=512 samples，滿足 Silero 最小需求
START_TRIGGER_FRAMES   = 3
END_TRIGGER_FRAMES     = 10        # 稍長，讓結束更穩定
TARGET_PROC_RATE       = 16000
RMS_THRESHOLD          = 1200      # 抑制遠處環境聲
MAX_SEGMENT_SEC        = 30.0

# Edge 播放器（喇叭輸出）環境變數的預設值
os.environ.setdefault("EDGE_OUT_INDEX", "4")
os.environ.setdefault("EDGE_OUT_NAME",  "VG27AQ3A")
os.environ.setdefault("EDGE_FORCE_DEVICE", "1")  # 嚴格使用指定裝置
os.environ.setdefault("EDGE_STEREO", "1")        # 1ch→2ch 上混
os.environ.setdefault("EDGE_OUT_GAIN", "1.0")    # 音量倍率

PLAYBACK_GUARD = threading.Event()


async def on_done(req: TaskRequest, res: dict) -> None:
    """任務完成回呼：印關鍵資訊；TTS 已串流播放，故不再重播整段。"""
    parsed = res.get("parsed", {})
    audio_file = res.get("audio_file", "")
    print(f"[DONE] kind={req.kind} message={parsed.get('Message')} tool={parsed.get('Tool')} action={parsed.get('Action')}")
    if audio_file and os.path.exists(audio_file):
        print(f"[DONE] TTS saved to: {audio_file}")


async def on_error(req: TaskRequest, err: Exception) -> None:
    """任務錯誤回呼：印堆疊供除錯。"""
    print(f"[ERROR] kind={getattr(req, 'kind', '?')} payload_keys={list(getattr(req, 'payload', {}).keys())}")
    print(f"[ERROR] type={type(err).__name__} msg={err}")
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    print(f"[ERROR] traceback:\n{tb}")


def _apply_env_overrides(args: argparse.Namespace) -> None:
    """
    將 CLI 參數映射到 Edge 播放器的環境變數。
    - EDGE_OUT_INDEX / EDGE_OUT_NAME / EDGE_STEREO
    """
    if args.out_index is not None:
        os.environ["EDGE_OUT_INDEX"] = str(args.out_index)
    if args.out_name:
        os.environ["EDGE_OUT_NAME"] = args.out_name
    os.environ["EDGE_STEREO"] = "1" if args.stereo else "0"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kirox Edge Client — 精簡參數（Silero VAD 版）")

    # 連線 / Agent
    p.add_argument("--uri", default="wss://agent.xbotworks.com/ws", help="WebSocket URI（例：ws://127.0.0.1:9600/ws）")
    p.add_argument("--botid", default=BOTID, help="Agent/使用者識別 ID")
    p.add_argument("--prompt-style", default=PROMPT_STYLE, help="Agent 的提示風格")
    p.add_argument("--voice-name", default=VOICE_NAME, help="TTS 聲音名稱")
    p.add_argument("--tts-lang", default=TTS_LANG, help="TTS 語言代碼")
    # Edge 播放器輸出（選填，但強烈建議指定）
    p.add_argument("--out-index", type=int, default=None, help="（建議）指定喇叭 sounddevice index")
    p.add_argument("--out-name", type=str, default=None, help="（建議）喇叭名稱子字串（例：'VG27AQ3A'）")
    p.add_argument("--stereo", action="store_true", default=True, help="1ch 自動上混 2ch（預設啟用）")
    p.add_argument("--mono", dest="stereo", action="store_false", help="停用上混，維持 1ch")

    # 錄音 / VAD
    p.add_argument("--input-name", default=INPUT_DEVICE_NAME, help="輸入裝置名稱子字串（優先）")
    p.add_argument("--input-index", type=int, default=INPUT_DEVICE_INDEX, help="輸入裝置 index（覆蓋名稱搜尋）")
    p.add_argument("--frame-ms", type=int, default=FRAME_MS, help="音框毫秒；Silero 最低需 sr/31.25 取樣（16k→≥32ms）")
    p.add_argument("--start-frames", type=int, default=START_TRIGGER_FRAMES, help="起音觸發（連續有聲幀數）")
    p.add_argument("--end-frames", type=int, default=END_TRIGGER_FRAMES, help="結束觸發（連續無聲幀數）")
    p.add_argument("--proc-rate", type=int, default=TARGET_PROC_RATE, help="VAD 處理採樣率（建議 16000）")
    p.add_argument("--rms-thresh", type=int, default=RMS_THRESHOLD, help="RMS 門檻，過低會誤觸；過高會漏檢")
    p.add_argument("--max-sec", type=float, default=MAX_SEGMENT_SEC, help="單段音訊最長秒數")

    # 其他
    p.add_argument("--outputs-dir", default="outputs", help="入/出音檔儲存資料夾")
    p.add_argument("--timeout", type=float, default=180.0, help="單次請求逾時秒數")
    p.add_argument("--debug", action="store_true", help="開啟詳細除錯（傳遞到 Edge 播放器）")
    return p


async def main():
    args = _build_arg_parser().parse_args()
    _apply_env_overrides(args)

    # 1) 可選：先 upsert 一個 TTS/Agent 實例（也可在 WS 首包做）
    try:
        _ = create_instance(
            botid=args.botid,
            prompt_style=args.prompt_style,
            voice_name=args.voice_name,
            language=args.tts_lang,
            output_device_index=None,  # 由 Edge 播放器讀環境變數決定
            frames_per_buffer=None,
            server_url="https://agent.xbotworks.com/create_instance",
        )
    except Exception as e:
        print(f"[WARN] create_instance failed: {e}")

    # 2) 任務管理器
    tm = TaskManager(send_coroutine=send_to_llm, timeout_seconds=float(args.timeout),
                     on_done=on_done, on_error=on_error)
    loop = asyncio.get_running_loop()

    # 3) 啟動 VAD 錄音器（Silero 版）
    vad = VADMicStreamer(
        task_manager=tm,
        loop=loop,
        outputs_dir=args.outputs_dir,
        uri=args.uri,
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
        # Silero 遲滯門檻（可依環境再調）
        start_prob=0.60,
        end_prob=0.35,
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
