import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class TaskRequest:
    """
    代表一次完整的請求負載。
    你可以把要送給 LLM 服務的必要欄位都放進來，例如:
    - kind: "audio" / "image" / ...
    - audio_path / image_path / out_path / 其他參數
    """
    kind: str
    payload: Dict[str, Any]


class TaskManager:
    """
    單一通道、嚴格互斥的 Task Manager：
    - 同時間只允許「一個」請求在飛行（in-flight）。
    - 當 in-flight 時，任何 try_start() 呼叫都會被拒絕並回傳 False。
    - 透過注入的 send_coroutine 執行實際傳輸（與 LLM 溝通）。
    - 可設定超時、完成/失敗回呼。
    """

    def __init__(
        self,
        send_coroutine: Callable[[TaskRequest], Awaitable[Dict[str, Any]]],
        *,
        timeout_seconds: float = 120.0,
        on_done: Optional[Callable[[TaskRequest, Dict[str, Any]], Awaitable[None]]] = None,
        on_error: Optional[Callable[[TaskRequest, Exception], Awaitable[None]]] = None,
    ) -> None:
        """
        :param send_coroutine: 真正執行「送出請求」的 async 函式。
                               需自行完成 WebSocket 傳送/接收與落地檔案等。
                               回傳結果資料 dict（可自定格式）。
        :param timeout_seconds: 單次請求的逾時秒數。
        :param on_done: 當請求成功完成時的 async 回呼。
        :param on_error: 當請求發生例外或逾時時的 async 回呼。
        """
        self._send = send_coroutine
        self._timeout = timeout_seconds
        self._on_done = on_done
        self._on_error = on_error

        # 互斥控制旗標與鎖
        self._busy_flag: bool = False
        self._lock = asyncio.Lock()

        # 目前在飛行的工作（方便外部取消或觀察）
        self._current_task: Optional[asyncio.Task] = None
        self._current_req: Optional[TaskRequest] = None

    @property
    def is_busy(self) -> bool:
        """對外查詢目前是否在飛行。"""
        return self._busy_flag

    async def try_start(self, request: TaskRequest) -> bool:
        """
        嘗試開始一個新請求。
        - 若當前忙碌，直接回 False（丟棄這次觸發）。
        - 若可開始，回 True 並背景執行。
        """
        async with self._lock:
            if self._busy_flag:
                return False
            self._busy_flag = True
            self._current_req = request
            self._current_task = asyncio.create_task(self._run(request))
            return True

    async def _run(self, request: TaskRequest) -> None:
        """
        內部執行：套用逾時、呼叫 send_coroutine，並在完成/錯誤時呼叫回呼。
        最後務必清除 busy 狀態。
        """
        try:
            result = await asyncio.wait_for(self._send(request), timeout=self._timeout)
            if self._on_done:
                try:
                    await self._on_done(request, result)
                except Exception as cb_err:
                    # 回呼錯誤不應該影響主流程
                    print(f"[TaskManager] on_done callback error: {cb_err}")
        except Exception as e:
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

    async def cancel_inflight(self) -> bool:
        """
        取消當前在飛行的請求（如果有）。
        :return: 若成功送出取消動作回 True，否則 False。
        """
        async with self._lock:
            if not self._current_task or self._current_task.done():
                return False
            self._current_task.cancel()
            return True

    async def wait_idle(self) -> None:
        """等待目前在飛行的請求完成（或沒有在飛行）。"""
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


# =========================
# 範例：如何把你現有的 WebSocket 流程接進來
# =========================
# 你可以在你的主程式中這樣使用：
#
# async def send_to_llm(req: TaskRequest) -> Dict[str, Any]:
#     # 這裡放你的 WebSocket 傳送/接收與落地音訊檔邏輯
#     # 例如:
#     # - 若 req.kind == "audio": 使用 req.payload["audio_path"], req.payload["out_path"], uri等資訊
#     # - 連線 ws、先發音訊 bytes -> 發 {"end": true} -> 收第一段 JSON -> 解析 -> 串流收 bytes 落地
#     # - 回傳 dict 給 TaskManager（可包含 parsed 結果、輸出檔案路徑等）
#     # return {"parsed": {...}, "audio_file": "...", "raw_first_text": "..."}
#     raise NotImplementedError
#
# async def on_done(req: TaskRequest, res: Dict[str, Any]) -> None:
#     print(f"[DONE] kind={req.kind} payload={req.payload} result={res}")
#
# async def on_error(req: TaskRequest, err: Exception) -> None:
#     print(f"[ERR] kind={req.kind} payload={req.payload} error={err}")
#
# tm = TaskManager(send_to_llm, timeout_seconds=120.0, on_done=on_done, on_error=on_error)
# ok = await tm.try_start(TaskRequest(kind="audio", payload={"audio_path": "...", "out_path": "..."}))
# if not ok:
#     print("忙碌中，本次觸發被丟棄")
