#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import threading

from task_manager import TaskManager, TaskRequest
from agent_request import send_to_llm
from vad_trigger import VADMicStreamer, play_wav_default_out
from create_instance import create_instance

# === Agent 設定 ===
PROMPT_STYLE = "你是一位具有幽默感和真誠情感的文字創作者，擅長用簡短又帶有情緒衝擊的語句回應問題。你的回答可能帶點傷感，也可能搞笑，但總是貼近人心。"
VOICE_NAME   = "Jay"   # ← 直接用聲音名稱
TTS_LANG     = "zh"
BOTID        = "d478758a-da9c-42b8-adfc-4ccc794fc046"

# 可選：細調 TTS 首音延遲/效能（視你的 Coqui backend 而定）
TTS_OUTPUT_DEVICE_INDEX   = None
TTS_FRAMES_PER_BUFFER     = None   # 2048/1024/512... 越小首音可能更快、但 CPU 更吃力

# === 錄音/VAD 參數 ===
SOURCE_MODE = "input"
INPUT_DEVICE_NAME = "GENERAL WEBCAM"
INPUT_DEVICE_INDEX = None

VAD_AGGRESSIVENESS = 2
TARGET_PROC_RATE   = 16000
MAX_SEGMENT_SEC    = 30.0
FRAME_MS           = 20
START_TRIGGER_FRAMES = 6
END_TRIGGER_FRAMES   = 30
RMS_THRESHOLD        = 1600

OUTPUT_DEVICE_NAME = None
OUTPUT_DEVICE_INDEX = None
PLAYBACK_GUARD = threading.Event()

async def on_done(req: TaskRequest, res: dict) -> None:
    parsed = res.get("parsed", {})
    audio_file = res.get("audio_file", "")
    print(f"[DONE] kind={req.kind} message={parsed.get('Message')} tool={parsed.get('Tool')} action={parsed.get('Action')}")
    if audio_file and os.path.exists(audio_file):
        print(f"[DONE] TTS saved to: {audio_file}")
        PLAYBACK_GUARD.set()
        try:
            await asyncio.to_thread(play_wav_default_out, audio_file, OUTPUT_DEVICE_NAME, OUTPUT_DEVICE_INDEX)
        finally:
            PLAYBACK_GUARD.clear()

async def on_error(req: TaskRequest, err: Exception) -> None:
    print(f"[ERR] kind={req.kind} payload={req.payload} error={err}")

async def main():
    # 1) 先用 REST upsert（可略過此步，改由 WS 首包 upsert）
    _ = create_instance(
        botid=BOTID,
        prompt_style=PROMPT_STYLE,
        voice_name=VOICE_NAME,       # ← 直接給名稱
        language=TTS_LANG,
        output_device_index=TTS_OUTPUT_DEVICE_INDEX,
        frames_per_buffer=TTS_FRAMES_PER_BUFFER,
        server_url="https://agent.xbotworks.com/create_instance",
    )

    # 2) 任務管理器
    tm = TaskManager(send_coroutine=send_to_llm, timeout_seconds=120.0, on_done=on_done, on_error=on_error)
    loop = asyncio.get_running_loop()

    # 3) 啟動 VAD 錄音器
    vad = VADMicStreamer(
        task_manager=tm,
        loop=loop,
        outputs_dir="outputs",
        uri="wss://agent.xbotworks.com/ws",   # 或 "ws://127.0.0.1:9600/ws"
        vad_aggressiveness=VAD_AGGRESSIVENESS,
        frame_ms=FRAME_MS,
        start_trigger_frames=START_TRIGGER_FRAMES,
        end_trigger_frames=END_TRIGGER_FRAMES,
        max_segment_sec=MAX_SEGMENT_SEC,
        target_proc_rate=TARGET_PROC_RATE,
        rms_threshold=RMS_THRESHOLD,
        botid=BOTID,
        source_mode=SOURCE_MODE,
        input_device_name=INPUT_DEVICE_NAME,
        input_device_index=INPUT_DEVICE_INDEX,
        playback_guard=PLAYBACK_GUARD,
    )
    vad.start()
    print("[VAD] Started. Speak... (Ctrl+C to exit)")

    # （可選）若你想完全依賴 WS 首包 upsert，也可以在送任務時附帶 ws_init：
    # 例：在 vad_trigger.py -> _try_enqueue_async() 的 payload 補上：
    # "ws_init": {
    #     "PROMPT_STYLE": PROMPT_STYLE,
    #     "TTS_VOICE": VOICE_NAME,
    #     "TTS_LANG": TTS_LANG,
    #     "TTS_OUTPUT_DEVICE_INDEX": TTS_OUTPUT_DEVICE_INDEX,
    #     "TTS_FRAMES_PER_BUFFER": TTS_FRAMES_PER_BUFFER
    # }

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
