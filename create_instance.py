import requests
from typing import Optional, Dict, Any

def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    """移除 None 值，避免不小心覆蓋 server 端既有設定。"""
    return {k: v for k, v in d.items() if v is not None}

def create_instance(
    botid: str,
    prompt_style: Optional[str] = None,
    voice_name: Optional[str] = None,                 # ← 改用 voice name
    language: Optional[str] = "zh",
    output_device_index: Optional[int] = None,
    frames_per_buffer: Optional[int] = None,
    server_url: str = "http://127.0.0.1:9600/create_instance",
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    呼叫 FastAPI 的 /create_instance。
    - 若實例不存在：建立（使用提供的參數）
    - 若實例已存在：更新（僅更新你提供的欄位）
    回傳 server 的 JSON（dict），失敗時回 None。
    """
    payload = _compact({
        "botid": botid,
        "PROMPT_STYLE": prompt_style,
        "TTS_VOICE": voice_name,
        "TTS_LANG": language,
        "TTS_OUTPUT_DEVICE_INDEX": output_device_index,
        "TTS_FRAMES_PER_BUFFER": frames_per_buffer,
    })

    try:
        resp = requests.post(server_url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        print(f"[連線錯誤] {e}")
        return None

    try:
        result = resp.json()
    except Exception:
        print(f"[回應非 JSON] 狀態碼: {resp.status_code}, 內容: {resp.text}")
        return None

    status = result.get("status")
    msg = result.get("message", "")
    if status == "ok":
        print(f"[成功] {msg}")
    else:
        print(f"[回應] status={status} message={msg}")
    return result
