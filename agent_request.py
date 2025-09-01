import asyncio
import websockets
import json
import re
from typing import Optional, Dict, Any

def parse_llm_output(output_text: str) -> Dict[str, Optional[str]]:
    try:
        data = json.loads(output_text)
        return {"Message": data.get("Message"), "Tool": data.get("Tool"), "Action": data.get("Action")}
    except json.JSONDecodeError:
        pass
    try:
        m = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not m:
            raise ValueError("找不到 JSON 區塊")
        data = json.loads(m.group(0))
        return {"Message": data.get("Message"), "Tool": data.get("Tool"), "Action": data.get("Action")}
    except Exception as e:
        print(f"[剖析錯誤] 無法解析 LLM 輸出：{e}")
        return {"Message": None, "Tool": None, "Action": None}

async def send_audio_to_llm_ws(
    uri: str,
    botid: str,
    audio_path: str,
    out_path: str,
    ws_init: Optional[Dict[str, Any]] = None,   # ← 新增：首包 upsert 參數（可空）
) -> Dict[str, Any]:
    """
    流程：
      1. 首包 TEXT JSON：{"botid": ..., 其它 upsert 欄位...}
      2. 傳音訊 binary
      3. TEXT JSON：{"end": true}
      4. 收 {"text": ...}、再收 binary TTS，直到 {"done": true}
    ws_init 可傳：
      - PROMPT_STYLE / TTS_VOICE / TTS_LANG / TTS_OUTPUT_DEVICE_INDEX / TTS_FRAMES_PER_BUFFER / image
    """
    first_json = {"botid": botid}
    if ws_init:
        first_json.update({k: v for k, v in ws_init.items() if v is not None})

    async with websockets.connect(uri, max_size=None) as ws:
        # 1) 首包
        await ws.send(json.dumps(first_json))

        # 2) 音訊
        with open(audio_path, "rb") as f:
            await ws.send(f.read())

        # 3) 結束旗標
        await ws.send(json.dumps({"end": True}))

        # 4) 收第一段文字
        first = await ws.recv()
        outer = json.loads(first)
        inner_text = outer.get("text", "")
        parsed = parse_llm_output(inner_text if isinstance(inner_text, str) else str(inner_text))

        # 5) 收 TTS 串流
        with open(out_path, "wb") as fout:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    fout.write(msg)
                else:
                    d = json.loads(msg)
                    if d.get("done"):
                        break
            fout.flush()

    return {"parsed": parsed, "raw_first_text": inner_text, "audio_file": out_path}

# 任務統一入口
async def send_to_llm(request) -> Dict[str, Any]:
    kind = request.kind
    payload = request.payload

    if kind == "audio":
        uri = payload.get("uri", "ws://127.0.0.1:9600/ws")
        botid = payload["botid"]
        audio_path = payload["audio_path"]
        out_path = payload["out_path"]
        ws_init = payload.get("ws_init")  # ← 轉交進去（可 None）
        return await send_audio_to_llm_ws(uri, botid, audio_path, out_path, ws_init=ws_init)

    if kind == "image":
        return {"parsed": {"Message": "Image handled (placeholder)", "Tool": None, "Action": None},
                "raw_first_text": "", "audio_file": payload.get("out_path", "")}

    raise ValueError(f"Unsupported request kind: {kind}")
