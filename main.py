#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================
# 先全域鎖定音訊裝置（一定要在任何音訊相關 import 前）
# =======================
import sys
import logging

# 建議：最早就設好 logging（含音訊裝置選取 log）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    print("[FATAL] 未安裝 sounddevice（pip install sounddevice）", e)
    sys.exit(1)

# 你要的裝置設定（優先 index，其次 name 子字串）
INPUT_DEVICE_INDEX  = None
INPUT_DEVICE_NAME   = "GENERAL WEBCAM"   # 例：你的 USB Webcam 麥克風
OUTPUT_DEVICE_INDEX = 4                  # 你的 #4 → VG27AQ3A (hw:2,3)
OUTPUT_DEVICE_NAME  = "VG27AQ3A"

def _resolve_device(index, name, want_input):
    if index is not None:
        return index
    if name:
        for i, d in enumerate(sd.query_devices()):
            if want_input and d.get("max_input_channels", 0) > 0 and name in d["name"]:
                return i
            if not want_input and d.get("max_output_channels", 0) > 0 and name in d["name"]:
                return i
    # 回退：仍保持原本預設
    return sd.default.device[0 if want_input else 1]

try:
    in_dev  = _resolve_device(INPUT_DEVICE_INDEX,  INPUT_DEVICE_NAME,  True)
    out_dev = _resolve_device(OUTPUT_DEVICE_INDEX, OUTPUT_DEVICE_NAME, False)
    sd.default.device = (in_dev, out_dev)
    sd.default.channels = 2  # 播放端固定 2 聲道（上混由播放端做）
    # 不在這裡鎖 samplerate，讓實際開流依音檔/串流決定
    din  = sd.query_devices(in_dev)
    dout = sd.query_devices(out_dev)
    log.info(f"[AudioPreset] in={in_dev} [{din['name']}], out={out_dev} [{dout['name']}]")
except Exception as e:
    log.exception("設定全域音訊裝置失敗")
    sys.exit(1)

# =======================
# 之後才 import 其他模組
# =======================
import asyncio
import traceback

from task_manager import TaskManager
from agent_request import send_to_llm
from vad_trigger import VADMicStreamer

# （可選）後端預建 TTS/Agent
try:
    from create_instance import create_instance
except Exception:
    def create_instance(*args, **kwargs):
        return {"ok": False, "reason": "create_instance not available"}


# ============ 你只需要改這些常數 ============

# 連線/身分
WS_URI       = "wss://agent.xbotworks.com/ws"
BOTID        = "d478758a-da9c-42b8-adfc-4ccc794fc046"
PROMPT_STYLE = "你是一位具有幽默感和真誠情感的文字創作者，擅長用簡短又帶有情緒衝擊的語句回應問題。"
VOICE_NAME   = "Jay"
TTS_LANG     = "zh"

# 一鍵開關：保存錄音與 TTS 檔案（True=全部保存；False=全部不保存）
SAVE_OUTPUTS = True  # 先開啟，方便你用 aplay 驗證輸出檔案內容

# 錄音/VAD
FRAME_MS               = 32
START_TRIGGER_FRAMES   = 3
END_TRIGGER_FRAMES     = 10
TARGET_PROC_RATE       = 16000
RMS_THRESHOLD          = 1200
MAX_SEGMENT_SEC        = 30.0
START_PROB             = 0.60
END_PROB               = 0.35
INPUT_DEVICE_INDEX_CLI = INPUT_DEVICE_INDEX     # 沿用上面已鎖的（保留擴展彈性）
INPUT_DEVICE_NAME_CLI  = INPUT_DEVICE_NAME

# 本地播放（必開：本需求明確要本地裝置播放）
ENABLE_LOCAL_PLAYBACK  = True
OUTPUT_DEVICE_INDEX_CLI = OUTPUT_DEVICE_INDEX   # 沿用上面已鎖的
OUTPUT_DEVICE_NAME_CLI  = OUTPUT_DEVICE_NAME
STEREO_UPMIX           = True          # 1ch → 2ch
OUT_GAIN               = 1.0           # 播放增益
FRAMES_PER_BUFFER      = 2048          # HDMI/DP 建議加大，必要可 4096

# 其他
OUTPUTS_DIR            = "outputs"
TIMEOUT_SECONDS        = 180.0


# ============ 回呼（打印用） ============

async def on_done(req, res: dict) -> None:
    parsed = res.get("parsed", {})
    logging.getLogger("callback").info(
        f"[DONE] kind={getattr(req,'kind','?')} "
        f"msg={parsed.get('Message')} tool={parsed.get('Tool')}"
    )

async def on_error(req, err: Exception) -> None:
    clog = logging.getLogger("callback")
    clog.error(f"[ERROR] kind={getattr(req, 'kind', '?')} type={type(err).__name__} msg={err}")
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    clog.error(f"[TRACE]\n{tb}")


# ============ 進入點 ============

async def main():
    # 可選：預建 TTS/Agent（若後端支援；把輸出裝置資訊也傳過去）
    try:
        resp = create_instance(
            botid=BOTID,
            prompt_style=PROMPT_STYLE,
            voice_name=VOICE_NAME,
            language=TTS_LANG,
            output_device_index=OUTPUT_DEVICE_INDEX_CLI,
            frames_per_buffer=FRAMES_PER_BUFFER,
            server_url="https://agent.xbotworks.com/create_instance",
        )
        logging.getLogger("agent").info(f"[Agent] create_instance resp={resp}")
    except Exception as e:
        logging.getLogger("agent").warning(f"[Agent] create_instance failed: {e}")

    tm = TaskManager(
        send_coroutine=send_to_llm,
        timeout_seconds=float(TIMEOUT_SECONDS),
        on_done=on_done,
        on_error=on_error
    )
    loop = asyncio.get_running_loop()

    # 設定並啟動 VAD 流
    vad = VADMicStreamer(
        task_manager=tm,
        loop=loop,
        outputs_dir=OUTPUTS_DIR,
        uri=WS_URI,
        botid=BOTID,

        # 語音/風格參數（傳給後端）
        voice_name=VOICE_NAME,
        language=TTS_LANG,
        prompt_style=PROMPT_STYLE,

        # 錄音/VAD
        frame_ms=int(FRAME_MS),
        start_trigger_frames=int(START_TRIGGER_FRAMES),
        end_trigger_frames=int(END_TRIGGER_FRAMES),
        max_segment_sec=float(MAX_SEGMENT_SEC),
        target_proc_rate=int(TARGET_PROC_RATE),
        rms_threshold=int(RMS_THRESHOLD),
        start_prob=float(START_PROB),
        end_prob=float(END_PROB),

        # 輸入裝置（與前面的全域設定一致）
        input_device_name=INPUT_DEVICE_NAME_CLI,
        input_device_index=INPUT_DEVICE_INDEX_CLI,

        # 本地播放與輸出裝置（與前面的全域設定一致）
        enable_local_playback=bool(ENABLE_LOCAL_PLAYBACK),
        out_device_index=OUTPUT_DEVICE_INDEX_CLI,
        out_device_name=OUTPUT_DEVICE_NAME_CLI,
        stereo_upmix=bool(STEREO_UPMIX),
        out_gain=float(OUT_GAIN),
        frames_per_buffer=FRAMES_PER_BUFFER,

        # 保存總開關
        save_outputs=bool(SAVE_OUTPUTS),
    )
    vad.start()

    log.info("[VAD] Started. Speak... (Ctrl+C to exit)")
    log.info(f"[CFG] WS={WS_URI} SAVE_OUTPUTS={SAVE_OUTPUTS} "
             f"OUT_DEV_IDX={OUTPUT_DEVICE_INDEX_CLI} OUT_DEV_NAME={OUTPUT_DEVICE_NAME_CLI} "
             f"FPB={FRAMES_PER_BUFFER}")

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        vad.stop()
        await tm.wait_idle()


if __name__ == "__main__":
    asyncio.run(main())
