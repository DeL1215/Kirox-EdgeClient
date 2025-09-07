# agent_request.py
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
import time
import wave
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import websockets

# --- 播放（可缺省） ---
try:
    import sounddevice as sd  # type: ignore
    _HAS_SD = True
except Exception as _e:  # pragma: no cover
    sd = None
    _HAS_SD = False
    print("[agent_request] 警告：未安裝 sounddevice（pip install sounddevice）。將僅落檔不即時播放。", _e)

# --- 系統查詢（PulseAudio / PipeWire） ---
import subprocess
import shutil


def _run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _prefer_pulse_hostapi() -> Optional[int]:
    """在 sounddevice hostapis 中尋找 PulseAudio / PipeWire hostapi 的 index。"""
    if not _HAS_SD:
        return None
    try:
        for idx, ha in enumerate(sd.query_hostapis()):
            n = (ha.get("name") or "").lower()
            if "pulse" in n or "pipewire" in n:
                return idx
    except Exception:
        pass
    return None


def _pulse_default_sink_name() -> Optional[str]:
    """取得目前系統 Default Sink 名稱。"""
    if shutil.which("pactl") is None:
        return None
    ver = _run_cmd(["pactl", "--version"]).lower()
    if "pulseaudio" in ver or "pipewire" in ver:
        name = _run_cmd(["pactl", "get-default-sink"]).strip()
        if name:
            return name
    info = _run_cmd(["pactl", "info"])
    for line in info.splitlines():
        if line.lower().startswith("default sink:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _pulse_sink_description(name: str) -> Optional[str]:
    """由 sink 名稱找對應的 Description（人類可讀），好用來對映 sounddevice 裝置名。"""
    out = _run_cmd(["pactl", "list", "sinks"])
    if not out:
        return None
    cur = None
    desc = None
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            cur = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            desc = line.split(":", 1)[1].strip()
            if cur == name:
                return desc
    return None


def _sd_output_index_by_name_substr(substr: str) -> Optional[int]:
    """以名稱子字串在 sounddevice 裝置清單中找一個可輸出裝置 index。"""
    if not _HAS_SD:
        return None
    ss = (substr or "").lower()
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                if ss in (d.get("name") or "").lower():
                    return i
    except Exception:
        pass
    return None


def _system_default_output_index() -> Optional[int]:
    """
    嘗試把 Ubuntu/PA/PW 的 Default Sink 映射為 sounddevice 裝置 index。
    """
    if not _HAS_SD:
        return None
    sink = _pulse_default_sink_name()
    if sink:
        desc = _pulse_sink_description(sink)
        if desc:
            idx = _sd_output_index_by_name_substr(desc)
            if idx is not None:
                return idx
    ha = _prefer_pulse_hostapi()
    if ha is not None:
        try:
            ha_info = sd.query_hostapis()[ha]
            idx = ha_info.get("default_output_device", None)
            if isinstance(idx, int) and idx >= 0:
                return idx
        except Exception:
            pass
    return None


# =========================
# 工具：解析 LLM 文字輸出
# =========================
def parse_llm_output(output_text: str) -> Dict[str, Optional[str]]:
    """嘗試把回覆中的 JSON 區塊剖析出來。失敗則回空欄位。"""
    try:
        data = json.loads(output_text)
        return {"Message": data.get("Message"), "Tool": data.get("Tool")}
    except json.JSONDecodeError:
        pass
    try:
        m = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not m:
            raise ValueError("找不到 JSON 區塊")
        data = json.loads(m.group(0))
        return {"Message": data.get("Message"), "Tool": data.get("Tool")}
    except Exception:
        return {"Message": None, "Tool": None}


# ==========================================
# 串流播放器：支援 pcm_s16le / f32le + 取樣率/裝置回退
# ==========================================
class _StreamingPCMPlayer:
    """
    串流 PCM 播放器：
      - 支援 'pcm_s16le' / 'f32le'
      - 裝置選擇策略（加入 EDGE_FORCE_DEVICE 與增益）：
          * 若 EDGE_FORCE_DEVICE=1：優先用 EDGE_OUT_INDEX；若找不到，再用 EDGE_OUT_NAME；
            都找不到時才回退 default。
          * 若未啟用 FORCE：候選順序為 [default, sys_default_index, EDGE_OUT_INDEX, EDGE_OUT_NAME, 第一個可輸出].
      - 取樣率先用 source sr（例如 24000），失敗 fallback 到 [48000,44100,32000,16000,8000]
      - 1ch 可上混 2ch（EDGE_STEREO!=0）
      - 以 EDGE_OUT_GAIN（float）調整播放音量（預設 1.0）
    """
    def __init__(self, in_fmt: str, sample_rate: int, channels: int, debug: bool = False):
        self.in_fmt = in_fmt
        self.in_sr = int(sample_rate)
        self.in_ch = int(channels)
        self.debug = debug

        self.stream = None
        self.out_sr: Optional[int] = None
        self.out_ch: Optional[int] = None
        self._prev_last: Optional[np.ndarray] = None

        # 增益
        try:
            self._gain = float(os.environ.get("EDGE_OUT_GAIN", "1.0"))
        except Exception:
            self._gain = 1.0

        if not _HAS_SD:
            if self.debug:
                print("[player] sounddevice 不可用 -> 跳過播放")
            return

        # ---- 環境變數覆蓋 ----
        env_idx_raw = os.environ.get("EDGE_OUT_INDEX")
        env_name = os.environ.get("EDGE_OUT_NAME")
        force_stereo = os.environ.get("EDGE_STEREO", "1") != "0"
        force_dev = os.environ.get("EDGE_FORCE_DEVICE", "0") == "1"

        env_idx: Optional[int] = None
        if env_idx_raw is not None:
            try:
                env_idx = int(env_idx_raw)
            except Exception:
                env_idx = None

        # ---- 建立候選裝置序列 ----
        candidates: List[Optional[int]] = []

        if force_dev:
            # 嚴格使用指定裝置
            if env_idx is not None:
                candidates = [env_idx]
            else:
                # 嘗試用名稱匹配
                hint = _sd_output_index_by_name_substr(env_name) if env_name else None
                if hint is not None:
                    candidates = [hint]
                else:
                    # 找不到指定，最後才允許 default
                    candidates = [None]
        else:
            # 非嚴格：default -> 系統 default index -> 環境 index -> 名稱匹配 -> 第一個可輸出
            candidates.append(None)
            sys_idx = _system_default_output_index()
            if sys_idx is not None and sys_idx not in candidates:
                candidates.append(sys_idx)
            if env_idx is not None and env_idx not in candidates:
                candidates.append(env_idx)
            if env_name:
                hint_idx = _sd_output_index_by_name_substr(env_name)
                if hint_idx is not None and hint_idx not in candidates:
                    candidates.append(hint_idx)
            # 兜底
            try:
                for i, d in enumerate(sd.query_devices()):
                    if d.get("max_output_channels", 0) > 0:
                        if i not in candidates:
                            candidates.append(i)
                            break
            except Exception:
                pass

        if self.debug:
            tags = [("default" if c is None else str(c)) for c in candidates]
            print(f"[player] device candidates: {tags}")
            try:
                print("[player] Output device list:")
                for i, d in enumerate(sd.query_devices()):
                    print(f"  #{i}: {d.get('name')} (out={d.get('max_output_channels',0)})")
            except Exception as e:
                print(f"[player] query_devices failed: {e}")

        # 目標輸出聲道
        target_out_ch = 2 if (force_stereo and self.in_ch == 1) else self.in_ch

        # ---- 試開 stream：先試各候選 device，再在每個 device 試不同 sr ----
        last_err = None
        for dev in candidates:
            for sr in [self.in_sr, 48000, 44100, 32000, 16000, 8000]:
                try:
                    st = sd.OutputStream(
                        samplerate=sr,
                        channels=target_out_ch,
                        dtype="float32",
                        blocksize=0,
                        latency="low",
                        device=dev
                    )
                    st.start()
                    self.stream = st
                    self.out_sr = sr
                    self.out_ch = target_out_ch
                    if self.debug:
                        tag_dev = "default" if dev is None else f"dev={dev}"
                        tag_sr = "(match)" if sr == self.in_sr else "(fallback)"
                        print(f"[player] OutputStream opened sr={sr} ch={target_out_ch} {tag_dev} {tag_sr}")
                    break
                except Exception as e:
                    last_err = e
                    if self.debug:
                        tag_dev = "default" if dev is None else f"dev={dev}"
                        print(f"[player] open failed sr={sr} {tag_dev}: {e}")
                    continue
            if self.stream is not None:
                break

        if self.stream is None:
            raise last_err or RuntimeError("No output sample rate/device available")

    def _resample_linear(self, f32: np.ndarray) -> np.ndarray:
        if self.out_sr == self.in_sr or f32.shape[0] < 2:
            return f32
        if self._prev_last is not None:
            f32 = np.concatenate([self._prev_last, f32], axis=0)

        x = np.arange(f32.shape[0], dtype=np.float32)
        M = int(round((f32.shape[0] - 1) * self.out_sr / self.in_sr)) + 1
        x_new = np.linspace(0.0, f32.shape[0] - 1, num=M, endpoint=True, dtype=np.float32)

        if f32.shape[1] == 1:
            y = np.interp(x_new, x, f32[:, 0]).astype(np.float32)
            out = y.reshape(-1, 1)
        else:
            cols = [np.interp(x_new, x, f32[:, c]).astype(np.float32) for c in range(f32.shape[1])]
            out = np.stack(cols, axis=1)

        if self._prev_last is not None and out.shape[0] > 0:
            out = out[1:, :]
        self._prev_last = f32[-1:, :]
        return out

    def _upmix_if_needed(self, f32: np.ndarray) -> np.ndarray:
        if f32.ndim != 2:
            return f32
        if f32.shape[1] == 1 and self.out_ch == 2:
            return np.repeat(f32, 2, axis=1)
        return f32

    def feed(self, chunk: bytes):
        if self.stream is None or not chunk:
            return

        if self.in_fmt == "pcm_s16le":
            arr = np.frombuffer(chunk, dtype=np.int16)
            frames = arr.reshape(-1, self.in_ch) if self.in_ch > 1 else arr.reshape(-1, 1)
            f32 = frames.astype(np.float32) / 32768.0
        elif self.in_fmt == "f32le":
            arr = np.frombuffer(chunk, dtype=np.float32)
            frames = arr.reshape(-1, self.in_ch) if self.in_ch > 1 else arr.reshape(-1, 1)
            f32 = frames
        else:
            return

        if f32.size == 0:
            return

        # 增益
        if self._gain != 1.0:
            f32 = np.clip(f32 * self._gain, -1.0, 1.0)

        f32 = self._upmix_if_needed(f32)
        if self.out_sr != self.in_sr:
            f32 = self._resample_linear(f32)

        if f32.size:
            self.stream.write(f32)

    def close(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            if self.debug:
                print("[player] closed")


# ==========================================
# 邊收邊寫正確 WAV（統一存 16-bit PCM）
# ==========================================
class _WavSink:
    def __init__(self, path: str, input_fmt: str, sample_rate: int, channels: int, debug: bool = False):
        self.path = path
        self.in_fmt = input_fmt
        self.sr = int(sample_rate)
        self.ch = int(channels)
        self.debug = debug

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._wf = wave.open(self.path, "wb")
        self._wf.setnchannels(self.ch)
        self._wf.setsampwidth(2)      # 統一存 16-bit
        self._wf.setframerate(self.sr)
        self._frames = 0

        if self.debug:
            print(f"[sink] open wav: {self.path}, sr={self.sr}, ch={self.ch}")

    def write(self, chunk: bytes):
        if not chunk:
            return
        if self.in_fmt == "pcm_s16le":
            data = chunk
        elif self.in_fmt == "f32le":
            f32 = np.frombuffer(chunk, dtype=np.float32)
            if self.ch > 1:
                f32 = f32.reshape(-1, self.ch)
            s16 = np.clip(np.rint(f32 * 32767.0), -32768, 32767).astype(np.int16)
            data = s16.tobytes()
        else:
            return

        self._wf.writeframes(data)
        self._frames += len(data) // 2  # 16-bit
        if self.debug:
            print(f"[sink] wrote {len(data)} bytes (total frames={self._frames})")

    def close(self):
        try:
            self._wf.close()
            if self.debug:
                sz = os.path.getsize(self.path)
                dur = self._frames / float(self.sr) if self.sr else 0.0
                print(f"[sink] closed: size={sz} bytes, ~{dur:.3f}s")
        except Exception as e:
            if self.debug:
                print(f"[sink] close error: {e}")


# ==========================================
# WebSocket：把音訊送到 LLM 伺服器並接收串流
# ==========================================
async def send_audio_to_llm_ws(
    uri: str,
    botid: str,
    audio_path: str,
    out_path: str,
    ws_init: Optional[Dict[str, Any]] = None,
    *,
    on_action=None,            # ← 新增：收到 {"action": ...} 時回呼
    on_text=None,
    on_audio_format=None,
    on_audio_chunk=None,
    on_stream_end=None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    典型時序（新版 server）：
      1) TEXT：{"botid": ..., 可選 upsert 欄位}
      2) BINARY：上傳使用者音訊
      3) TEXT：{"end": true}
      4) TEXT：{"action": ...}            # 先送動作（可能是字串或 JSON 物件）
      5) TEXT：{"text": "..."}            # 文字回覆
      6) TEXT：{"audio_format":"pcm_s16le","sample_rate":24000,"channels":1}
      7) BINARY：音訊分塊（多包）
      8) TEXT：{"done": true}
    """
    first_json = {"botid": botid}
    if ws_init:
        first_json.update({k: v for k, v in ws_init.items() if v is not None})

    if debug:
        print(f"[client] 即將連到 {uri}")

    t_all0 = time.time()
    parsed_text: Dict[str, Any] = {}
    raw_first_text = ""
    raw_action = None
    parsed_action = None

    # 音訊格式（等伺服器宣告後確定）
    in_fmt: Optional[str] = None
    in_sr: Optional[int] = None
    in_ch: Optional[int] = None

    player: Optional[_StreamingPCMPlayer] = None
    sink: Optional[_WavSink] = None

    async with websockets.connect(uri, max_size=None) as ws:
        t_connected = time.time()
        if debug:
            print(f"[client] 已連線，用時 {t_connected - t_all0:.3f}s")

        # 1) 首包
        await ws.send(json.dumps(first_json))
        if debug:
            print(f"[client] 已送首包 upsert：{first_json}")

        # 2) 使用者音訊
        with open(audio_path, "rb") as f:
            data_up = f.read()
        await ws.send(data_up)
        if debug:
            print(f"[client] 已送音訊 {len(data_up)} bytes (from {audio_path})")

        # 3) end
        await ws.send(json.dumps({"end": True}))
        if debug:
            print("[client] 已送 end:true")

        # 4+) 接收迴圈
        bytes_recv_audio = 0
        t_first_audio = None
        t_after_end = time.time()

        if debug:
            print("[client] 進入接收階段…")

        while True:
            msg = await ws.recv()

            # --- 音訊分塊 ---
            if isinstance(msg, bytes):
                if in_fmt is None:
                    # 未宣告格式就來了 → 預設並警告（理論上 server 會先宣告）
                    in_fmt = "pcm_s16le"
                    in_sr = 24000
                    in_ch = 1
                    if debug:
                        print("[client] 警告：尚未收到 audio_format 宣告就收到 bytes，採預設 pcm_s16le@24000x1")
                    # 建播放器 / 寫檔器
                    try:
                        player = _StreamingPCMPlayer(in_fmt, in_sr, in_ch, debug=debug)
                    except Exception as e:
                        player = None
                        if debug:
                            print(f"[player] 建立失敗：{e}")
                    sink = _WavSink(out_path, in_fmt, in_sr, in_ch, debug=debug)

                if t_first_audio is None:
                    t_first_audio = time.time()
                    if debug:
                        print(f"[client] 收到首個音訊分塊，距離送 end={t_first_audio - t_after_end:.3f}s")
                        hx = " ".join(f"{b:02x}" for b in msg[:16])
                        print(f"[debug] first 16 bytes: {hx}")

                bytes_recv_audio += len(msg)

                # 若提供 on_audio_chunk，預設播放器會被覆蓋；不傳則直接播放
                if on_audio_chunk:
                    await on_audio_chunk(msg)
                else:
                    if player:
                        player.feed(msg)

                # 邊收邊寫 WAV
                if sink:
                    sink.write(msg)

                if debug and (bytes_recv_audio % (64 * 1024) < len(msg)):
                    kbps = (bytes_recv_audio / max(1e-6, (time.time() - (t_first_audio or time.time())))) / 1024.0
                    print(f"[client] 音訊累積 {bytes_recv_audio} bytes, ~{kbps:.1f} KB/s")
                continue

            # --- TEXT JSON ---
            try:
                d = json.loads(msg)
            except Exception:
                if debug:
                    s = str(msg)
                    print(f"[client] 未知文字訊息：{s[:120]}...")
                continue

            # 先送的 Action
            if "action" in d:
                raw_action = d.get("action")
                parsed_action = None
                # 可能是 dict 或字串
                if isinstance(raw_action, dict):
                    parsed_action = raw_action
                else:
                    try:
                        parsed_action = json.loads(raw_action)
                    except Exception:
                        parsed_action = None

                if debug:
                    short = str(raw_action)[:160].replace("\n", " ")
                    print(f"[client] action 收到：{short}")

                if on_action:
                    try:
                        await on_action(raw_action, parsed_action)
                    except Exception as e:
                        if debug:
                            print(f"[client] on_action 回呼錯誤：{e}")
                continue

            # 文字回覆
            if "text" in d:
                raw_first_text = d.get("text") or ""
                parsed_text = parse_llm_output(raw_first_text if isinstance(raw_first_text, str) else str(raw_first_text))
                if debug:
                    s = raw_first_text if isinstance(raw_first_text, str) else str(raw_first_text)
                    s_short = s[:120].replace("\n", " ").replace("\r", " ")
                    print(f"[client] 文字回覆就緒（len={len(s)}），連線後耗時 {time.time() - t_connected:.3f}s")
                    print(f"[CB] text(len={len(s)}): {s_short}...")
                if on_text:
                    try:
                        await on_text(raw_first_text)
                    except Exception as e:
                        if debug:
                            print(f"[client] on_text 回呼錯誤：{e}")
                continue

            # 音訊格式宣告
            if "audio_format" in d:
                in_fmt = str(d.get("audio_format") or "pcm_s16le").lower()
                in_sr = int(d.get("sample_rate") or 24000)
                in_ch = int(d.get("channels") or 1)
                if debug:
                    print(f"[client] 收到音訊格式宣告：{d}")
                if on_audio_format:
                    try:
                        await on_audio_format(d)
                    except Exception as e:
                        if debug:
                            print(f"[client] on_audio_format 回呼錯誤：{e}")

                # 準備播放器與寫檔器
                try:
                    player = _StreamingPCMPlayer(in_fmt, in_sr, in_ch, debug=debug)
                except Exception as e:
                    player = None
                    if debug:
                        print(f"[player] 建立失敗：{e}")
                sink = _WavSink(out_path, in_fmt, in_sr, in_ch, debug=debug)
                continue

            # 結束
            if d.get("done") is True:
                total = time.time() - t_all0
                if debug:
                    print(f"[client] 收到 done=true，總耗時 {total:.3f}s；上傳 {len(data_up)} bytes，下載音訊 {bytes_recv_audio} bytes")
                if on_stream_end:
                    try:
                        await on_stream_end()
                    except Exception as e:
                        if debug:
                            print(f"[client] on_stream_end 回呼錯誤：{e}")
                break

            if debug:
                print(f"[client] 其他文字訊息：{d}")

        # 收尾
        if sink:
            sink.close()
        if player:
            player.close()

    return {
        "parsed": parsed_text,          # 從文字回覆解析出的 {Message, Tool}
        "raw_first_text": raw_first_text,
        "audio_file": out_path,
        "action": parsed_action,        # 若 action 可解析為 JSON，這裡給 dict；否則 None
        "raw_action": raw_action,       # 原始 action（可能是字串或 dict）
    }


# ==========================================
# 任務統一入口（給 TaskManager 用）
# ==========================================
async def send_to_llm(request) -> Dict[str, Any]:
    """
    與 TaskManager 的介面一致。
    """
    kind = getattr(request, "kind", None)
    payload = getattr(request, "payload", {}) or {}
    debug = getattr(request, "debug", False)

    if kind == "audio":
        uri = payload.get("uri", "ws://127.0.0.1:9600/ws")
        botid = payload["botid"]
        audio_path = payload["audio_path"]
        out_path = payload["out_path"]
        ws_init = payload.get("ws_init")

        return await send_audio_to_llm_ws(
            uri=uri,
            botid=botid,
            audio_path=audio_path,
            out_path=out_path,
            ws_init=ws_init,
            on_action=getattr(request, "on_action", None),        # ← 新增轉接
            on_text=getattr(request, "on_text", None),
            on_audio_format=getattr(request, "on_audio_format", None),
            # 不傳 on_audio_chunk 以確保 default 播放器會 feed()
            on_stream_end=getattr(request, "on_stream_end", None),
            debug=debug,
        )

    if kind == "image":
        return {
            "parsed": {"Message": "Image handled (placeholder)", "Tool": None},
            "raw_first_text": "",
            "audio_file": payload.get("out_path", ""),
            "action": None,
            "raw_action": None,
        }

    raise ValueError(f"Unsupported request kind: {kind}")
