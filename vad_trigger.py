# -*- coding: utf-8 -*-
# vad_trigger.py
import os
import time
import wave
import queue
import asyncio
import threading
import tempfile
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import torch  # Silero VAD

from task_manager import TaskRequest, TaskManager


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or src_sr == dst_sr:
        return x.astype(np.int16, copy=False)
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    n_dst = int(round(len(x) * dst_sr / src_sr))
    t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False, dtype=np.float64)
    y = np.interp(t_dst, t_src, x.astype(np.float64))
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _find_sd_device_index_by_name_substr(name_substr: Optional[str], want_output: bool) -> Optional[int]:
    if not name_substr:
        return None
    name_substr = name_substr.lower()
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            ch_key = "max_output_channels" if want_output else "max_input_channels"
            if d.get(ch_key, 0) > 0 and name_substr in (d.get("name") or "").lower():
                return i
    except Exception:
        pass
    return None


class _SDPlayer:
    def __init__(self,
                 device_index: Optional[int],
                 device_name: Optional[str],
                 stereo_upmix: bool,
                 out_gain: float,
                 frames_per_buffer: Optional[int]):
        self._dev_index_cfg = device_index
        self._dev_name_cfg = device_name
        self.stereo_upmix = bool(stereo_upmix)
        self.out_gain = float(out_gain)
        self.blocksize = int(frames_per_buffer) if frames_per_buffer else 0
        self.stream: Optional[sd.OutputStream] = None
        self.out_channels = 0
        self.sr = 0

    def _resolve_device_index(self) -> Optional[int]:
        if self._dev_index_cfg is not None:
            return int(self._dev_index_cfg)
        idx = _find_sd_device_index_by_name_substr(self._dev_name_cfg, want_output=True)
        return idx  # 可能為 None → 用系統預設

    def open(self, samplerate: int, in_channels: int):
        self.sr = int(samplerate)
        self.out_channels = 2 if (self.stereo_upmix and in_channels == 1) else max(1, in_channels)
        dev_idx = self._resolve_device_index()
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.out_channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=dev_idx
        )
        self.stream.start()

    def write(self, chunk_bytes: bytes, in_channels: int):
        if not chunk_bytes or not self.stream:
            return
        x = np.frombuffer(chunk_bytes, dtype=np.int16)
        ch = max(1, in_channels)
        x = x.reshape(-1, ch)
        if self.stereo_upmix and ch == 1 and self.out_channels == 2:
            x = np.repeat(x, 2, axis=1)
        if self.out_gain != 1.0:
            x = (x.astype(np.float32) * self.out_gain).clip(-32768, 32767).astype(np.int16)
        self.stream.write(x)

    def close(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class VADMicStreamer:
    """
    - 可指定輸入/輸出裝置（index 優先、名稱子字串次之、否則系統預設）。
    - SAVE_OUTPUTS 控制是否保留錄音片段與 TTS 檔案。
    - 本地播放期間啟用 guard 避免自錄。
    - 不依賴任何環境變數；所有參數由 main.py 傳入。
    """
    def __init__(
        self,
        task_manager: TaskManager,
        loop: asyncio.AbstractEventLoop,
        # 後端連線
        outputs_dir: str,
        uri: str,
        botid: Optional[str],
        # 額外語音/風格（由 main 提供，不在此預設）
        voice_name: Optional[str] = None,
        language: Optional[str] = None,
        prompt_style: Optional[str] = None,

        # 錄音/VAD
        frame_ms: int = 32,
        start_trigger_frames: int = 3,
        end_trigger_frames: int = 10,
        max_segment_sec: float = 30.0,
        target_proc_rate: int = 16000,
        rms_threshold: int = 1200,
        start_prob: float = 0.60,
        end_prob: float = 0.35,
        input_device_name: Optional[str] = None,
        input_device_index: Optional[int] = None,

        # 本地播放
        enable_local_playback: bool = True,
        out_device_index: Optional[int] = None,
        out_device_name: Optional[str] = None,
        stereo_upmix: bool = True,
        out_gain: float = 1.0,
        frames_per_buffer: Optional[int] = None,

        # 保存總開關
        save_outputs: bool = True,
    ) -> None:
        self.tm = task_manager
        self.loop = loop
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.uri = uri
        self.botid = botid
        self.voice_name = voice_name
        self.language = language
        self.prompt_style = prompt_style

        # VAD / 音訊狀態
        self.proc_rate = int(target_proc_rate)
        self.dtype = "int16"
        self.frame_ms = int(frame_ms)
        self.proc_frame_samples = int(self.proc_rate * self.frame_ms / 1000)
        self._silero_min_samples = int(np.ceil(self.proc_rate / 31.25))  # 16k → 512
        if self.proc_frame_samples < self._silero_min_samples:
            self.proc_frame_samples = self._silero_min_samples
            self.frame_ms = int(round(1000 * self.proc_frame_samples / self.proc_rate))
        self.start_trigger_frames = int(start_trigger_frames)
        self.end_trigger_frames = int(end_trigger_frames)
        self.max_segment_sec = float(max_segment_sec)
        self.rms_threshold = int(rms_threshold)
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)

        # 本地播放
        self.enable_local_playback = bool(enable_local_playback)
        self.player_cfg = dict(
            device_index=out_device_index,
            device_name=out_device_name,
            stereo_upmix=stereo_upmix,
            out_gain=out_gain,
            frames_per_buffer=frames_per_buffer,
        )

        # 保存
        self.save_outputs = bool(save_outputs)

        # I/O
        self.input_device_name = input_device_name
        self.input_device_index = input_device_index

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._proc_buf = np.zeros((0,), dtype=np.int16)

        self.playback_guard = threading.Event()

        # Silero VAD 初始化
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.silero_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        self.silero_model.eval().to(self.device)

        # 選擇輸入裝置
        self.input_rate, self.input_device, self.channels = self._pick_input_device()
        self._log_selected_device()

    # ---------- 輸入裝置 ----------
    def _pick_input_device(self) -> Tuple[int, Optional[int], int]:
        # index > 名稱子字串 > 預設（嘗試多 SR）
        if self.input_device_index is not None:
            try:
                sd.check_input_settings(device=self.input_device_index, samplerate=self.proc_rate, channels=1, dtype=self.dtype)
                return self.proc_rate, self.input_device_index, 1
            except Exception:
                pass

        idx = _find_sd_device_index_by_name_substr(self.input_device_name, want_output=False) if self.input_device_name else None
        if idx is not None:
            try:
                sd.check_input_settings(device=idx, samplerate=self.proc_rate, channels=1, dtype=self.dtype)
                return self.proc_rate, idx, 1
            except Exception:
                pass

        for sr in [self.proc_rate, 48000, 44100, 32000, 16000]:
            try:
                sd.check_input_settings(device=None, samplerate=sr, channels=1, dtype=self.dtype)
                return sr, None, 1
            except Exception:
                continue
        return 44100, None, 1

    def _log_selected_device(self) -> None:
        try:
            devices = sd.query_devices()
            print("[Audio] Devices:")
            for i, d in enumerate(devices):
                print(f"  #{i}: {d.get('name')} (in={d.get('max_input_channels', 0)} out={d.get('max_output_channels', 0)})")
        except Exception:
            pass
        print(f"[Audio] Input device idx={self.input_device} sr={self.input_rate} ch=1")

    # ---------- Stream ----------
    def _open_stream_with_fallback(self):
        trial_rates = [self.input_rate, 48000, 44100, 32000, 16000]
        for sr in trial_rates:
            try:
                sd.check_input_settings(device=self.input_device, samplerate=sr, channels=1, dtype=self.dtype)
                blocksize = int(sr * self.frame_ms / 1000)
                stream = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype=self.dtype,
                    blocksize=blocksize,
                    callback=self._audio_callback,
                    device=self.input_device
                )
                stream.__enter__()
                self.input_rate = sr
                return stream
            except Exception:
                continue
        raise RuntimeError("No valid input sample rate for selected device")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        if indata.ndim == 1:
            mono = indata.astype(np.int16, copy=False)
        else:
            mono = np.clip(np.rint(indata.astype(np.float32).mean(axis=1)), -32768, 32767).astype(np.int16)
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            pass

    # ---------- Public ----------
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

    # ---------- File helpers ----------
    @staticmethod
    def _write_wav(path: str, pcm16: np.ndarray, sample_rate: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

    def _new_out_path(self, prefix: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.outputs_dir, f"{prefix}_{ts}_{int(time.time()*1000)%100000}.wav")

    # ---------- Core loop ----------
    def _run(self) -> None:
        stream = self._open_stream_with_fallback()
        try:
            in_speech = False
            voiced_count = 0
            unvoiced_count = 0
            seg_samples: List[np.ndarray] = []
            seg_start_t = 0.0

            while not self._stop.is_set():
                # 播放期間避免自錄
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

                # 轉為 VAD 採樣率
                chunk_proc = _resample_linear(chunk_in, self.input_rate, self.proc_rate)

                if self._proc_buf.size > 0:
                    chunk_proc = np.concatenate([self._proc_buf, chunk_proc], axis=0)
                    self._proc_buf = np.zeros((0,), dtype=np.int16)

                offset = 0
                while offset + self.proc_frame_samples <= len(chunk_proc):
                    frame_i16 = chunk_proc[offset:offset + self.proc_frame_samples]
                    offset += self.proc_frame_samples

                    # 粗略 RMS 篩噪
                    rms = np.sqrt(np.mean(frame_i16.astype(np.float32) ** 2))
                    if rms < self.rms_threshold:
                        speech_prob = 0.0
                    else:
                        # Silero 機率
                        frame_f32 = (frame_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                        with torch.no_grad():
                            tens = torch.from_numpy(frame_f32).to(self.device)
                            speech_prob = float(self.silero_model(tens, self.proc_rate).item())

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
                                print(f"[VAD] ▶ start p≈{speech_prob:.2f}")
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

                            # 寫入檔案（保存或暫存）
                            if self.save_outputs:
                                wav_path = self._new_out_path("seg")
                                delete_input_after = False
                            else:
                                fd, wav_path = tempfile.mkstemp(prefix="seg_", suffix=".wav", dir=self.outputs_dir)
                                os.close(fd)
                                delete_input_after = True

                            self._write_wav(wav_path, pcm, self.proc_rate)
                            size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
                            why = "silence" if end_by_silence else "max_len"
                            print(f"[VAD] ■ end ({why}) dur={dur:.2f}s frames={len(pcm)} wav={wav_path} ({size} bytes)")

                            ok = self._send_to_backend_and_play(
                                wav_path,
                                delete_input_after=delete_input_after
                            )
                            if not ok and delete_input_after:
                                try:
                                    os.remove(wav_path)
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

    # ---------- Send to backend + local playback ----------
    def _send_to_backend_and_play(self, wav_path: str, *, delete_input_after: bool) -> bool:
        t0 = time.time()

        # 由 agent_request.py 依 out_path 落檔；本地只負責播放
        if self.save_outputs:
            tts_out_path = self._new_out_path("resp")
            delete_tts_after = False
        else:
            fd, tts_out_path = tempfile.mkstemp(prefix="resp_", suffix=".wav", dir=self.outputs_dir)
            os.close(fd)
            delete_tts_after = True

        player = _SDPlayer(**self.player_cfg) if self.enable_local_playback else None
        fmt_sr = 16000
        fmt_ch = 1

        async def _on_text(s: str):
            preview = s[:120].replace("\n", " ").replace("\r", " ")
            print(f"[CB] text: {preview}...")

        async def _on_fmt(d: dict):
            nonlocal fmt_sr, fmt_ch
            fmt_sr = int(d.get("sample_rate") or 16000)
            fmt_ch = int(d.get("channels") or 1)
            print(f"[CB] audio_format sr={fmt_sr} ch={fmt_ch} -> guard ON")
            self.playback_guard.set()
            if player:
                player.open(fmt_sr, fmt_ch)

        async def _on_chunk(b: bytes):
            if player:
                player.write(b, fmt_ch)

        async def _on_end():
            if player:
                player.close()
            print("[CB] stream_end -> guard OFF")
            self.playback_guard.clear()
            if delete_input_after:
                try:
                    os.remove(wav_path)
                    print(f"[CLEANUP] removed input: {wav_path}")
                except Exception:
                    pass
            if delete_tts_after:
                try:
                    os.remove(tts_out_path)
                    print(f"[CLEANUP] removed tts: {tts_out_path}")
                except Exception:
                    pass

        # ★ 關鍵：補 out_path，避免 agent_request.py 的 KeyError
        payload = {
            "uri": self.uri,
            "audio_path": wav_path,
            "botid": self.botid,
            "out_path": tts_out_path,
        }
        if self.voice_name is not None:
            payload["voice_name"] = self.voice_name
        if self.language is not None:
            payload["language"] = self.language
        if self.prompt_style is not None:
            payload["prompt_style"] = self.prompt_style

        req = TaskRequest(
            kind="audio",
            payload=payload,
            timeout_seconds=180.0,
            on_text=_on_text,
            on_audio_format=_on_fmt,
            on_audio_chunk=_on_chunk,
            on_stream_end=_on_end,
            debug=True,
        )

        print(f"[ENQUEUE] start -> file={wav_path}")
        future = asyncio.run_coroutine_threadsafe(self.tm.try_start(req), self.loop)
        try:
            accepted = future.result(timeout=1.0)
        except Exception as e:
            print(f"[ENQUEUE] exception: {e}")
            return False

        print(f"[ENQUEUE] accepted={accepted}, latency={(time.time() - t0):.3f}s")
        return bool(accepted)
