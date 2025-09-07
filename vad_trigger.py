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
import asyncio
import torch  # ← 取代 webrtcvad，改用 Silero VAD

from task_manager import TaskRequest, TaskManager


# ====== 系統工具：PulseAudio / PipeWire 查詢與裝置尋找 ======

def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""

def _pulse_defaults() -> Tuple[Optional[str], Optional[str]]:
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
    try:
        for idx, ha in enumerate(sd.query_hostapis()):
            n = (ha.get("name") or "").lower()
            if "pulse" in n or "pipewire" in n:
                return idx
    except Exception:
        pass
    return None

def _find_sd_input_index_by_name_substr(name_substr: str, prefer_hostapi: Optional[int]) -> Optional[int]:
    if not name_substr:
        return None
    name_substr = name_substr.lower()
    devices = sd.query_devices()
    if prefer_hostapi is not None:
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == prefer_hostapi:
                if name_substr in (d.get("name") or "").lower():
                    return i
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            if name_substr in (d.get("name") or "").lower():
                return i
    return None

def _find_sd_output_index_by_sink_name(sink_name: Optional[str]) -> Optional[int]:
    if not sink_name:
        return None
    sink_name = sink_name.lower()
    devices = sd.query_devices()
    prefer_ha = _prefer_pulse_hostapi()
    if prefer_ha is not None:
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == prefer_ha:
                if sink_name in (d.get("name") or "").lower():
                    return i
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            if sink_name in (d.get("name") or "").lower():
                return i
    return None


# ====== 播放工具：阻塞播放 + 尾端緩衝，避免截尾 ======

def play_wav_default_out(path: str,
                         output_device_name: Optional[str] = None,
                         output_device_index: Optional[int] = None) -> None:
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


# ====== VAD + 錄音主類別（Silero 版本） ======

class VADMicStreamer:
    """
    以 Silero VAD 取代 webrtcvad，仍保持與原版 class/參數相容。
    特性：
      - 對「人聲」更敏感，較不會被環境雜音誤觸發
      - 支援啟動/結束不同門檻（hysteresis），配合連續幀數更穩定
    """
    def __init__(
        self,
        task_manager: TaskManager,
        loop: asyncio.AbstractEventLoop,
        outputs_dir: str = "outputs",
        uri: str = "wss://agent.xbotworks.com/ws",
        # ↓↓↓ 這些參數保留相容，但內部改以 Silero 概念處理 ↓↓↓
        vad_aggressiveness: int = 2,  # 無實質影響（保留參數以相容）
        frame_ms: int = 30,           # 建議 30ms，Silero 在 20~30ms 表現穩定
        start_trigger_frames: int = 3,
        end_trigger_frames: int = 12,
        max_segment_sec: float = 15.0,
        mic_keyword: Optional[str] = None,
        fallback_device: Optional[int] = None,
        target_proc_rate: int = 16000,   # Silero 模型建議 16k
        rms_threshold: int = 500,        # 先做能量門檻，濾掉底噪
        botid: Optional[str] = None,
        source_mode: str = "input",
        input_device_name: Optional[str] = None,
        input_device_index: Optional[int] = None,
        playback_guard: Optional[threading.Event] = None,
        # Silero 門檻（機率，0~1）：建議 start > end，形成遲滯
        start_prob: float = 0.60,
        end_prob: float = 0.35,
    ) -> None:
        self.tm = task_manager
        self.loop = loop
        self.outputs_dir = outputs_dir
        self.uri = uri
        self.proc_rate = target_proc_rate
        self.dtype = "int16"
        self.source_mode = source_mode
        self.playback_guard = playback_guard or threading.Event()

        # ========== Silero VAD 初始化 ==========
        # 會自動下載並快取；CPU 即可滿足即時需求
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.silero_model, self._silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        self.silero_model.eval().to(self.device)
        # utils: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.frame_ms = int(frame_ms)
        self.proc_frame_samples = int(self.proc_rate * self.frame_ms / 1000)
        # Silero 要求：每次送入的 chunk 長度 >= sr / 31.25
        self._silero_min_samples = int(np.ceil(self.proc_rate / 31.25))  # 16k → 512
        if self.proc_frame_samples < self._silero_min_samples:
            self.proc_frame_samples = self._silero_min_samples
            self.frame_ms = int(round(1000 * self.proc_frame_samples / self.proc_rate))

        self.start_trigger_frames = start_trigger_frames
        self.end_trigger_frames = end_trigger_frames
        self.max_segment_sec = max_segment_sec
        self.rms_threshold = rms_threshold

        self.mic_keyword = mic_keyword
        self.fallback_device = fallback_device

        os.makedirs(self.outputs_dir, exist_ok=True)
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._proc_buf = np.zeros((0,), dtype=np.int16)
        self.botid = botid

        # 選擇輸入裝置
        if platform.system() == "Linux":
            self.input_rate, self.input_device, self.channels, self.kind = self._pick_linux_input_only(
                input_device_name, input_device_index
            )
        else:
            self.input_rate, self.input_device, self.channels, self.kind = self._pick_nonlinux(
                input_device_name, input_device_index
            )

        self._log_selected_device()

    def _pick_linux_input_only(self, input_device_name: Optional[str], input_device_index: Optional[int]) -> Tuple[int, Optional[int], int, str]:
        prefer_ha = _prefer_pulse_hostapi()

        if input_device_index is not None:
            return self._validate_rates(input_device_index, prefer_stereo=False) + ("manual",)
        if input_device_name:
            idx = _find_sd_input_index_by_name_substr(input_device_name, prefer_ha)
            if idx is not None:
                return self._validate_rates(idx, prefer_stereo=False) + ("manual",)

        _, default_source = _pulse_defaults()
        if default_source:
            idx = _find_sd_input_index_by_name_substr(default_source, prefer_ha)
            if idx is not None:
                return self._validate_rates(idx, prefer_stereo=False) + ("mic",)

        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                sr, idx2, ch = self._validate_rates(i, prefer_stereo=False)
                try:
                    sd.check_input_settings(device=idx2, samplerate=sr, channels=ch, dtype=self.dtype)
                    return sr, idx2, ch, "mic"
                except Exception:
                    continue

        for sr in [48000, 44100, 32000, 16000, 8000]:
            try:
                sd.check_input_settings(device=None, samplerate=sr, channels=1, dtype=self.dtype)
                return sr, None, 1, "mic"
            except Exception:
                pass

        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                return 44100, i, 1, "mic"

        return 44100, None, 1, "unknown"

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

    def _validate_rates(self, dev_index: int, prefer_stereo: bool) -> Tuple[int, int, int]:
        try_rates = [48000, 44100, self.proc_rate, 32000, 16000, 8000]
        ch_try = [1, 2]
        for ch in ch_try:
            for sr in try_rates:
                try:
                    sd.check_input_settings(device=dev_index, samplerate=sr, channels=ch, dtype=self.dtype)
                    return sr, dev_index, ch
                except Exception:
                    continue
        return self.proc_rate, dev_index, 1

    def _log_selected_device(self) -> None:
        try:
            devices = sd.query_devices()
            print("[Audio] Available devices:")
            for i, d in enumerate(devices):
                print(f"  #{i}: {d.get('name')} (in={d.get('max_input_channels', 0)}, out={d.get('max_output_channels', 0)}, hostapi={d.get('hostapi')})")
        except Exception:
            print("[Audio] Could not list devices.")
        print(f"[Audio] Selected input device: {self.input_device}, samplerate={self.input_rate}, target_proc_rate={self.proc_rate}, channels={self.channels}, kind={self.kind}")
        print(f"[Audio] VAD frame_ms={self.frame_ms}  frame_samples={self.proc_frame_samples}  (silero_min={self._silero_min_samples})")


    def _open_stream_with_fallback(self):
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

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
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

    def _int16_to_float(self, x_int16: np.ndarray) -> np.ndarray:
        # 轉為 [-1, 1] 的 float32，Silero 輸入格式
        return (x_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

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

                # 轉為 VAD 處理採樣率（int16）
                chunk_proc = _resample_linear(chunk_in, self.input_rate, self.proc_rate)

                if self._proc_buf.size > 0:
                    chunk_proc = np.concatenate([self._proc_buf, chunk_proc], axis=0)
                    self._proc_buf = np.zeros((0,), dtype=np.int16)

                offset = 0
                while offset + self.proc_frame_samples <= len(chunk_proc):
                    frame_i16 = chunk_proc[offset:offset + self.proc_frame_samples]
                    offset += self.proc_frame_samples

                    # 先做 RMS 能量過濾（int16），可迅速排除底噪
                    rms = np.sqrt(np.mean(frame_i16.astype(np.float32) ** 2))
                    if rms < self.rms_threshold:
                        speech_prob = 0.0
                    else:
                        # Silero：把 int16 → float32 [-1,1]，再丟進模型得出人聲機率
                        frame_f32 = self._int16_to_float(frame_i16)
                        with torch.no_grad():
                            tens = torch.from_numpy(frame_f32).to(self.device)
                            speech_prob = float(self.silero_model(tens, self.proc_rate).item())

                    # 遲滯判定：start_prob / end_prob + 連續幀數
                    is_speech = speech_prob >= (self.start_prob if not in_speech else self.end_prob)

                    if not in_speech:
                        if is_speech:
                            voiced_count += 1
                            if voiced_count >= self.start_trigger_frames:
                                in_speech = True
                                seg_samples = [frame_i16.copy()]
                                seg_start_t = time.time()
                                unvoiced_count = 0
                                voiced_count = 0
                                print(f"[VAD] ▶ start (sr={self.proc_rate}, frame_ms={self.frame_ms}, p≈{speech_prob:.2f})")
                        else:
                            voiced_count = 0
                    else:
                        seg_samples.append(frame_i16.copy())
                        if is_speech:
                            unvoiced_count = 0
                        else:
                            unvoiced_count += 1

                        end_by_silence = (unvoiced_count >= self.end_trigger_frames)
                        end_by_length = ((time.time() - seg_start_t) >= self.max_segment_sec)

                        if end_by_silence or end_by_length:
                            dur = time.time() - seg_start_t
                            pcm = np.concatenate(seg_samples, axis=0) if seg_samples else np.zeros((0,), dtype=np.int16)
                            out_path = self._new_out_path()
                            self._write_wav(out_path, pcm, self.proc_rate)
                            fsz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                            reason = "silence" if end_by_silence else "max_len"
                            print(f"[VAD] ■ end ({reason}), seg_dur={dur:.2f}s, frames={len(pcm)}, wav={out_path} ({fsz} bytes)")

                            ok = self._try_enqueue_async(out_path, seg_dur=dur, file_size=fsz)
                            if not ok:
                                print("[VAD] enqueue rejected (busy). Drop segment.")
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

    def _try_enqueue_async(self, wav_path: str, *, seg_dur: float, file_size: int) -> bool:
        t0 = time.time()

        async def _on_text(s: str):
            _preview = s[:120].replace("\n", " ").replace("\r", " ")
            print(f"[CB] text(len={len(s)}): {_preview}...")

        async def _on_fmt(d: dict):
            # 串流音訊要開始了：先拉起 playback_guard 避免自錄到自己的播音
            sr = d.get("sample_rate")
            ch = d.get("channels")
            fmt = d.get("audio_format")
            print(f"[CB] audio_format: fmt={fmt}, sr={sr}, ch={ch} -> set playback_guard")
            self.playback_guard.set()

        async def _on_end():
            print("[CB] stream_end -> clear playback_guard")
            self.playback_guard.clear()

        req = TaskRequest(
            kind="audio",
            payload={
                "uri": self.uri,
                "audio_path": wav_path,
                "out_path": os.path.join(self.outputs_dir, f"resp_{os.path.basename(wav_path)}"),
                "botid": self.botid,
            },
            timeout_seconds=180.0,
            on_text=_on_text,
            on_audio_format=_on_fmt,
            # 關鍵：不要傳 on_audio_chunk，否則會覆蓋預設播放路徑
            on_stream_end=_on_end,
            debug=True,
        )

        print(f"[ENQUEUE] try_start -> file={wav_path} ({file_size} bytes), seg_dur={seg_dur:.2f}s")
        future = asyncio.run_coroutine_threadsafe(self.tm.try_start(req), self.loop)

        try:
            accepted = future.result(timeout=1.0)
        except Exception as e:
            print(f"[ENQUEUE] exception: {e}")
            return False

        t1 = time.time()
        print(f"[ENQUEUE] accepted={accepted}, latency={(t1 - t0):.3f}s")
        return bool(accepted)
