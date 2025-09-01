#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import wave
import queue
import threading
import platform
import subprocess
import shutil
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import webrtcvad
import asyncio

from task_manager import TaskRequest, TaskManager


# ====== 系統工具：PulseAudio / PipeWire 查詢與裝置尋找 ======

def _run_cmd(cmd: list[str]) -> str:
    """執行系統指令，回傳 stdout（失敗回空字串）。"""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""

def _pulse_defaults() -> Tuple[Optional[str], Optional[str]]:
    """
    取得 PulseAudio/PipeWire 的預設輸出/輸入名稱：
    回傳 (Default Sink, Default Source)，找不到則回 (None, None)。
    """
    if shutil.which("pactl") is None:
        return None, None
    info = _run_cmd(["pactl", "info"])
    sink = src = None
    for line in info.splitlines():
        low = line.lower()
        if low.startswith("default sink:"):
            sink = line.split(":", 1)[1].strip() or None
        elif low.startswith("default source:"):
            src = line.split(":", 1)[1].strip() or None
    return sink, src

def _prefer_pulse_hostapi() -> Optional[int]:
    """回傳 sounddevice 中 PulseAudio / PipeWire 的 hostapi index（若存在）。"""
    try:
        for idx, ha in enumerate(sd.query_hostapis()):
            n = (ha.get("name") or "").lower()
            if "pulse" in n or "pipewire" in n:
                return idx
    except Exception:
        pass
    return None

def _find_sd_input_index_by_name_substr(name_substr: str, prefer_hostapi: Optional[int]) -> Optional[int]:
    """
    用名稱子字串在 sounddevice 找『可輸入』裝置 index（優先指定 hostapi）。
    回傳第一個符合者；找不到回 None。
    """
    if not name_substr:
        return None
    name_substr = name_substr.lower()
    devices = sd.query_devices()
    # 先優先 Pulse/PipeWire
    if prefer_hostapi is not None:
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == prefer_hostapi:
                if name_substr in (d.get("name") or "").lower():
                    return i
    # 再全域搜尋
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            if name_substr in (d.get("name") or "").lower():
                return i
    return None

def _find_sd_output_index_by_sink_name(sink_name: Optional[str]) -> Optional[int]:
    """
    用 Default Sink 名稱子字串找到『可輸出』裝置 index；若 sink_name 為 None 或找不到回 None。
    """
    if not sink_name:
        return None
    sink_name = sink_name.lower()
    devices = sd.query_devices()
    prefer_ha = _prefer_pulse_hostapi()
    # 先優先 Pulse/PipeWire hostapi
    if prefer_ha is not None:
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == prefer_ha:
                if sink_name in (d.get("name") or "").lower():
                    return i
    # 再全域搜尋
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            if sink_name in (d.get("name") or "").lower():
                return i
    return None


# ====== 播放工具：阻塞播放 + 尾端緩衝，避免截尾 ======

def play_wav_default_out(path: str,
                         output_device_name: Optional[str] = None,
                         output_device_index: Optional[int] = None) -> None:
    """
    播放 WAV（16-bit PCM），預設用系統 Default Sink；也可手動指定輸出裝置。
    - output_device_index 優先，其次 output_device_name（子字串匹配）。
    - 都未指定時嘗試以 pactl Default Sink 對應 sounddevice 裝置；失敗則讓 sounddevice 用系統預設。
    以 blocking 播放，尾端加 stop + 30ms 緩衝避免 0.1 秒被吃掉。
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        samplerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise RuntimeError(f"目前僅示範 16-bit PCM WAV，偵測到 {sampwidth*8} bits")

    data = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)

    # 選輸出裝置
    dev_idx = None
    try:
        if output_device_index is not None:
            dev_idx = int(output_device_index)
        elif output_device_name:
            name = output_device_name.lower()
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0:
                    if name in (d.get("name") or "").lower():
                        dev_idx = i
                        break
        else:
            default_sink, _ = _pulse_defaults()
            dev_idx = _find_sd_output_index_by_sink_name(default_sink)
    except Exception:
        dev_idx = None

    sd.play(np.copy(data), samplerate, device=dev_idx, blocking=True)
    sd.stop()
    time.sleep(0.03)  # 尾端保險


# ====== 重採樣：mono int16 -> mono int16 ======

def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or src_sr == dst_sr:
        return x.astype(np.int16, copy=False)
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    n_dst = int(round(len(x) * dst_sr / src_sr))
    t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False, dtype=np.float64)
    y = np.interp(t_dst, t_src, x.astype(np.float64))
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


# ====== VAD + 錄音主類別 ======

class VADMicStreamer:
    """
    預設（Linux/Ubuntu）自動抓『可用輸入裝置（麥克風）』：
      1) 若有 Default Source（pactl）→ 優先選；
      2) 否則掃描所有「有輸入通道」裝置，選第一個能開啟的；
      3) 開串流失敗自動降階 48k/44.1k/32k/16k/8k。
    支援手動指定 input_device_name / input_device_index。
    播放期間以 playback_guard 暫停偵測並清空佇列，避免自我回錄。
    """
    def __init__(
        self,
        task_manager: TaskManager,
        loop: asyncio.AbstractEventLoop,
        outputs_dir: str = "outputs",
        uri: str = "ws://127.0.0.1:9600/ws",
        vad_aggressiveness: int = 2,
        frame_ms: int = 20,
        start_trigger_frames: int = 5,
        end_trigger_frames: int = 25,
        max_segment_sec: float = 15.0,
        mic_keyword: Optional[str] = None,           # 舊參數：可留空
        fallback_device: Optional[int] = None,       # 舊參數：可留空
        target_proc_rate: int = 16000,
        rms_threshold: int = 500,
        botid: Optional[str] = None,
        # 新增：裝置選擇（以輸入為主）與播放防護
        source_mode: str = "input",                  # ← 預設只抓輸入（麥克風）
        input_device_name: Optional[str] = None,     # 指定名稱子字串（手動）
        input_device_index: Optional[int] = None,    # 指定 index（手動）
        playback_guard: Optional[threading.Event] = None,  # 播放期間暫停 VAD
    ) -> None:
        self.tm = task_manager
        self.loop = loop
        self.outputs_dir = outputs_dir
        self.uri = uri
        self.proc_rate = target_proc_rate            # VAD 處理率（8/16/32/48k 之一）
        self.dtype = "int16"
        self.source_mode = source_mode
        self.playback_guard = playback_guard or threading.Event()

        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_ms = frame_ms
        self.proc_frame_samples = int(self.proc_rate * self.frame_ms / 1000)
        self.start_trigger_frames = start_trigger_frames
        self.end_trigger_frames = end_trigger_frames
        self.max_segment_sec = max_segment_sec
        self.rms_threshold = rms_threshold

        # 舊選項（最低優先，用於完全無 Pulse/PipeWire 環境時）
        self.mic_keyword = mic_keyword
        self.fallback_device = fallback_device

        os.makedirs(self.outputs_dir, exist_ok=True)
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._proc_buf = np.zeros((0,), dtype=np.int16)
        self.botid = botid

        # 選擇輸入裝置（以「輸入」為唯一目標）
        if platform.system() == "Linux":
            self.input_rate, self.input_device, self.channels, self.kind = self._pick_linux_input_only(
                input_device_name, input_device_index
            )
        else:
            self.input_rate, self.input_device, self.channels, self.kind = self._pick_nonlinux(
                input_device_name, input_device_index
            )

        self._log_selected_device()

    # ---------- 裝置挑選（Linux，僅輸入） ----------
    def _pick_linux_input_only(self, input_device_name: Optional[str], input_device_index: Optional[int]) -> Tuple[int, Optional[int], int, str]:
        prefer_ha = _prefer_pulse_hostapi()

        # (1) 手動指定優先
        if input_device_index is not None:
            return self._validate_rates(input_device_index, prefer_stereo=False) + ("manual",)
        if input_device_name:
            idx = _find_sd_input_index_by_name_substr(input_device_name, prefer_ha)
            if idx is not None:
                return self._validate_rates(idx, prefer_stereo=False) + ("manual",)

        # (2) 有 Default Source（pactl）則優先選
        _, default_source = _pulse_defaults()
        if default_source:
            idx = _find_sd_input_index_by_name_substr(default_source, prefer_ha)
            if idx is not None:
                return self._validate_rates(idx, prefer_stereo=False) + ("mic",)

        # (3) 強韌 fallback：掃描所有可輸入裝置，選第一個能成功配置的
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                sr, idx2, ch = self._validate_rates(i, prefer_stereo=False)
                try:
                    sd.check_input_settings(device=idx2, samplerate=sr, channels=ch, dtype=self.dtype)
                    return sr, idx2, ch, "mic"
                except Exception:
                    continue

        # (4) 最後手段：試 device=None 搭配常見 sr
        for sr in [48000, 44100, 32000, 16000, 8000]:
            try:
                sd.check_input_settings(device=None, samplerate=sr, channels=1, dtype=self.dtype)
                return sr, None, 1, "mic"
            except Exception:
                pass

        # (5) 仍不行，硬選第一個有輸入通道的 index（配 44.1k）
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                return 44100, i, 1, "mic"

        return 44100, None, 1, "unknown"

    # ---------- 裝置挑選（非 Linux：保守） ----------
    def _pick_nonlinux(self, input_device_name: Optional[str], input_device_index: Optional[int]) -> Tuple[int, Optional[int], int, str]:
        if input_device_index is not None:
            return self._validate_rates(input_device_index, prefer_stereo=False) + ("manual",)
        if input_device_name:
            idx = _find_sd_input_index_by_name_substr(input_device_name, _prefer_pulse_hostapi())
            if idx is not None:
                return self._validate_rates(idx, prefer_stereo=False) + ("manual",)
        try:
            sd.check_input_settings(device=None, samplerate=self.proc_rate, channels=1, dtype=self.dtype)
            return self.proc_rate, None, 1, "mic"
        except Exception:
            for sr in [48000, 44100, 32000, 16000, 8000]:
                try:
                    sd.check_input_settings(device=None, samplerate=sr, channels=1, dtype=self.dtype)
                    return sr, None, 1, "mic"
                except Exception:
                    continue
            return 44100, None, 1, "unknown"

    # ---------- 測試 (sr, ch) 可用性並回傳最合適組合 ----------
    def _validate_rates(self, dev_index: int, prefer_stereo: bool) -> Tuple[int, int, int]:
        """
        幫指定裝置挑一組可用的 (samplerate, device_index, channels)。
        mic 以 1ch 為主；失敗會回退到 1ch 與不同 sr。
        """
        try_rates = [48000, 44100, self.proc_rate, 32000, 16000, 8000]
        ch_try = [1, 2]  # 麥克風優先 1ch
        for ch in ch_try:
            for sr in try_rates:
                try:
                    sd.check_input_settings(device=dev_index, samplerate=sr, channels=ch, dtype=self.dtype)
                    return sr, dev_index, ch
                except Exception:
                    continue
        return self.proc_rate, dev_index, 1

    # ---------- 日誌 ----------
    def _log_selected_device(self) -> None:
        try:
            devices = sd.query_devices()
            print("[Audio] Available devices:")
            for i, d in enumerate(devices):
                print(f"  #{i}: {d.get('name')} (in={d.get('max_input_channels', 0)}, out={d.get('max_output_channels', 0)}, hostapi={d.get('hostapi')})")
        except Exception:
            print("[Audio] Could not list devices.")
        print(f"[Audio] Selected input device: {self.input_device}, samplerate={self.input_rate}, target_proc_rate={self.proc_rate}, channels={self.channels}, kind={self.kind}")

    # ---------- PortAudio 串流開啟（含降階重試） ----------
    def _open_stream_with_fallback(self):
        """
        優先用 self.input_rate 開串流；失敗就依序嘗試 48k/44.1k/32k/16k/8k，
        並自動調整 blocksize，直到成功為止（同時更新 self.input_rate）。
        """
        trial_rates = [self.input_rate, 48000, 44100, 32000, 16000, 8000]
        for sr in trial_rates:
            try:
                sd.check_input_settings(device=self.input_device, samplerate=sr, channels=self.channels, dtype=self.dtype)
                blocksize = int(sr * self.frame_ms / 1000)
                stream = sd.InputStream(
                    samplerate=sr,
                    channels=self.channels,
                    dtype=self.dtype,
                    blocksize=blocksize,
                    callback=self._audio_callback,
                    device=self.input_device
                )
                stream.__enter__()
                self.input_rate = sr
                print(f"[Audio] Opened stream at samplerate={sr}, device={self.input_device}")
                return stream
            except Exception:
                continue
        raise RuntimeError("No valid input sample rate for the selected device.")

    # ---------- 音訊回呼：支援多聲道 -> mono（平均） ----------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # 可視需要印出 XRUN 狀態
            pass
        if indata.ndim == 1:
            mono = indata.astype(np.int16, copy=False)
        else:
            mono_f32 = indata.astype(np.float32, copy=False).mean(axis=1)
            mono = np.clip(np.rint(mono_f32), -32768, 32767).astype(np.int16)
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            pass

    # ---------- 控制 ----------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    # ---------- 工具 ----------
    @staticmethod
    def _write_wav(path: str, pcm16: np.ndarray, sample_rate: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

    def _new_out_path(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.outputs_dir, f"seg_{ts}_{int(time.time()*1000)%100000}.wav")

    # ---------- 主迴圈 ----------
    def _run(self) -> None:
        stream = self._open_stream_with_fallback()
        try:
            in_speech = False
            voiced_count = 0
            unvoiced_count = 0
            seg_samples: List[np.ndarray] = []
            seg_start_t = 0.0

            while not self._stop.is_set():
                # 播放期間暫停偵測、清空佇列，避免自我回錄
                if self.playback_guard.is_set():
                    in_speech = False
                    voiced_count = 0
                    unvoiced_count = 0
                    seg_samples = []
                    try:
                        while True:
                            self._q.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.01)
                    continue

                try:
                    chunk_in = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 轉為 VAD 處理採樣率
                chunk_proc = _resample_linear(chunk_in, self.input_rate, self.proc_rate)

                if self._proc_buf.size > 0:
                    chunk_proc = np.concatenate([self._proc_buf, chunk_proc], axis=0)
                    self._proc_buf = np.zeros((0,), dtype=np.int16)

                offset = 0
                while offset + self.proc_frame_samples <= len(chunk_proc):
                    frame = chunk_proc[offset:offset + self.proc_frame_samples]
                    offset += self.proc_frame_samples

                    # RMS 過濾（能量太小視為靜音）
                    rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                    if rms < self.rms_threshold:
                        is_speech = False
                    else:
                        is_speech = self.vad.is_speech(frame.tobytes(), sample_rate=self.proc_rate)

                    if not in_speech:
                        if is_speech:
                            voiced_count += 1
                            if voiced_count >= self.start_trigger_frames:
                                in_speech = True
                                seg_samples = [frame.copy()]
                                seg_start_t = time.time()
                                unvoiced_count = 0
                                voiced_count = 0
                        else:
                            voiced_count = 0
                    else:
                        seg_samples.append(frame.copy())
                        if is_speech:
                            unvoiced_count = 0
                        else:
                            unvoiced_count += 1

                        end_by_silence = (unvoiced_count >= self.end_trigger_frames)
                        end_by_length = ((time.time() - seg_start_t) >= self.max_segment_sec)

                        if end_by_silence or end_by_length:
                            pcm = np.concatenate(seg_samples, axis=0) if seg_samples else np.zeros((0,), dtype=np.int16)
                            out_path = self._new_out_path()
                            self._write_wav(out_path, pcm, self.proc_rate)

                            ok = self._try_enqueue_async(out_path)
                            if not ok:
                                try:
                                    os.remove(out_path)
                                except Exception:
                                    pass

                            in_speech = False
                            voiced_count = 0
                            unvoiced_count = 0
                            seg_samples = []

                if offset < len(chunk_proc):
                    self._proc_buf = chunk_proc[offset:].copy()
        finally:
            stream.__exit__(None, None, None)

    # ---------- 推送任務 ----------
    def _try_enqueue_async(self, wav_path: str) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            self.tm.try_start(TaskRequest(
                kind="audio",
                payload={
                    "uri": self.uri,
                    "audio_path": wav_path,
                    "out_path": os.path.join(self.outputs_dir, f"resp_{os.path.basename(wav_path)}"),
                    "botid": self.botid
                }
            )),
            self.loop
        )
        try:
            return future.result()
        except Exception:
            return False
