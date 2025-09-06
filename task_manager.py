# task_manager.py
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

# 串流友善：每個請求可帶回呼
OnTextCb = Callable[[str], Awaitable[None]]
OnChunkCb = Callable[[bytes], Awaitable[None]]
OnFmtCb = Callable[[Dict[str, Any]], Awaitable[None]]
OnEndCb = Callable[[], Awaitable[None]]

@dataclass
class TaskRequest:
    """
    代表一次完整的請求負載。
    - kind: "audio" / "image" / ...
    - payload: 實際要送給下游(send_coroutine)的資料
    - timeout_seconds: 覆寫 TaskManager 預設逾時；None 表示使用預設
    - on_text:         收到伺服器先行的文字訊息（AI 回覆文字）時觸發
    - on_audio_format: 收到音訊格式宣告時觸發（例如 {"audio_format":"pcm_s16le","sample_rate":24000,"channels":1}）
    - on_audio_chunk:  收到音訊分塊(bytes)時觸發（若提供則覆蓋預設播放器）
    - on_stream_end:   串流結束(收到 {"done": true}) 時觸發
    - debug:           是否輸出詳細偵錯訊息
    """
    kind: str
    payload: Dict[str, Any]
    timeout_seconds: Optional[float] = None
    on_text: Optional[OnTextCb] = None
    on_audio_format: Optional[OnFmtCb] = None
    on_audio_chunk: Optional[OnChunkCb] = None
    on_stream_end: Optional[OnEndCb] = None
    debug: bool = False


class TaskManager:
    """
    單一通道、嚴格互斥的 Task Manager：
    - 同時間只允許「一個」請求在飛行（in-flight）。
    - 當 in-flight 時，任何 try_start() 呼叫都會被拒絕並回傳 False。
    - 透過注入的 send_coroutine 執行實際傳輸（與 LLM 溝通）。
    - 可設定全域逾時、完成/失敗回呼；亦支援 TaskRequest 層級覆寫逾時與串流回呼。
    """

    def __init__(
        self,
        send_coroutine: Callable[[TaskRequest], Awaitable[Dict[str, Any]]],
        *,
        timeout_seconds: float = 120.0,
        on_done: Optional[Callable[[TaskRequest, Dict[str, Any]], Awaitable[None]]] = None,
        on_error: Optional[Callable[[TaskRequest, Exception], Awaitable[None]]] = None,
    ) -> None:
        self._send = send_coroutine
        self._timeout = timeout_seconds
        self._on_done = on_done
        self._on_error = on_error

        self._busy_flag: bool = False
        self._lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task] = None
        self._current_req: Optional[TaskRequest] = None

    @property
    def is_busy(self) -> bool:
        return self._busy_flag

    async def try_start(self, request: TaskRequest) -> bool:
        """
        嘗試開始一個新請求。
        - 若當前忙碌，直接回 False（丟棄這次觸發）。
        - 若可開始，回 True 並背景執行。
        """
        async with self._lock:
            if self._busy_flag:
                if request.debug:
                    print("[TaskManager] busy=True，拒絕新請求")
                return False
            self._busy_flag = True
            self._current_req = request
            self._current_task = asyncio.create_task(self._run(request))
            if request.debug:
                print("[TaskManager] 接受新請求，開始背景執行")
            return True

    async def _run(self, request: TaskRequest) -> None:
        started_at = time.time()
        timeout = request.timeout_seconds if request.timeout_seconds is not None else self._timeout
        if request.debug:
            print(f"[TaskManager] _run() start, timeout={timeout}s, kind={request.kind}")

        try:
            result = await asyncio.wait_for(self._send(request), timeout=timeout)
            if request.debug:
                print(f"[TaskManager] _run() done in {time.time()-started_at:.3f}s")
            if self._on_done:
                try:
                    await self._on_done(request, result)
                except Exception as cb_err:
                    print(f"[TaskManager] on_done callback error: {cb_err}")
        except Exception as e:
            if request.debug:
                print(f"[TaskManager] _run() exception after {time.time()-started_at:.3f}s -> {e}")
            if self._on_error:
                try:
                    await self._on_error(request, e)
                except Exception as cb_err:
                    print(f"[TaskManager] on_error callback error: {cb_err}")
        finally:
            async with self._lock:
                self._busy_flag = False
                self._current_task = None
                self._current_req = None
            if request.debug:
                print("[TaskManager] 狀態已清空，允許下一筆請求")

    async def cancel_inflight(self) -> bool:
        async with self._lock:
            if not self._current_task or self._current_task.done():
                return False
            self._current_task.cancel()
            return True

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                if not self._busy_flag:
                    return
                task = self._current_task
            if task:
                try:
                    await task
                except Exception:
                    pass
            await asyncio.sleep(0.01)
