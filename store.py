"""store.py – In-memory analytics store for the proxy dashboard."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class RequestRecord:
    req_id: str
    ts: float                    # unix timestamp
    model: str
    stream: bool
    num_messages: int
    num_tools: int
    status: str = "pending"      # "pending" | "ok" | "error"
    error_code: Optional[str] = None
    duration_ms: Optional[float] = None
    text_chars: int = 0
    chunk_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "ts": self.ts,
            "model": self.model,
            "stream": self.stream,
            "num_messages": self.num_messages,
            "num_tools": self.num_tools,
            "status": self.status,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
            "text_chars": self.text_chars,
            "chunk_count": self.chunk_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class LogEntry:
    ts: float
    level: str      # "INFO" | "ERROR" | "WARN"
    req_id: Optional[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "level": self.level,
            "req_id": self.req_id,
            "message": self.message,
        }


class AnalyticsStore:
    """Thread-safe (asyncio) in-memory store for requests and logs."""

    MAX_REQUESTS = 500
    MAX_LOGS = 1000

    def __init__(self) -> None:
        self._requests: Deque[RequestRecord] = deque(maxlen=self.MAX_REQUESTS)
        self._logs: Deque[LogEntry] = deque(maxlen=self.MAX_LOGS)
        self._total_requests = 0
        self._total_errors = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._start_time = time.time()
        self._lock = asyncio.Lock()

        # SSE subscribers: set of asyncio.Queue
        self._log_subscribers: List[asyncio.Queue] = []
        self._request_subscribers: List[asyncio.Queue] = []

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    async def add_request(self, record: RequestRecord) -> None:
        async with self._lock:
            self._requests.append(record)
            self._total_requests += 1
        await self._broadcast_request(record)

    async def update_request(self, req_id: str, **kwargs: Any) -> None:
        async with self._lock:
            for r in self._requests:
                if r.req_id == req_id:
                    for k, v in kwargs.items():
                        setattr(r, k, v)
                    if kwargs.get("status") == "error":
                        self._total_errors += 1
                    self._total_input_tokens += kwargs.get("input_tokens", 0)
                    self._total_output_tokens += kwargs.get("output_tokens", 0)
                    await self._broadcast_request(r)
                    break

    # ------------------------------------------------------------------
    # Log tracking
    # ------------------------------------------------------------------

    async def add_log(self, level: str, req_id: Optional[str], message: str) -> None:
        entry = LogEntry(ts=time.time(), level=level, req_id=req_id, message=message)
        async with self._lock:
            self._logs.append(entry)
        await self._broadcast_log(entry)

    # ------------------------------------------------------------------
    # Analytics snapshot
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            recent = list(self._requests)
        
        durations = [r.duration_ms for r in recent if r.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0

        model_counts: Dict[str, int] = {}
        active_streams = 0
        for r in recent:
            model_counts[r.model] = model_counts.get(r.model, 0) + 1
            if r.stream and r.status == "pending":
                active_streams += 1

        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_duration_ms": round(avg_duration, 1),
            "active_streams": active_streams,
            "model_counts": model_counts,
            "recent_requests": [r.to_dict() for r in list(self._requests)[-50:][::-1]],
        }

    async def get_logs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [e.to_dict() for e in list(self._logs)[-200:][::-1]]

    # ------------------------------------------------------------------
    # SSE pub/sub
    # ------------------------------------------------------------------

    def subscribe_logs(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._log_subscribers.append(q)
        return q

    def unsubscribe_logs(self, q: asyncio.Queue) -> None:
        try:
            self._log_subscribers.remove(q)
        except ValueError:
            pass

    def subscribe_requests(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._request_subscribers.append(q)
        return q

    def unsubscribe_requests(self, q: asyncio.Queue) -> None:
        try:
            self._request_subscribers.remove(q)
        except ValueError:
            pass

    async def _broadcast_log(self, entry: LogEntry) -> None:
        dead = []
        for q in self._log_subscribers:
            try:
                q.put_nowait(entry.to_dict())
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe_logs(q)

    async def _broadcast_request(self, record: RequestRecord) -> None:
        dead = []
        for q in self._request_subscribers:
            try:
                q.put_nowait(record.to_dict())
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe_requests(q)


# Global singleton
store = AnalyticsStore()
