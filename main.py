"""main.py – FastAPI application: Anthropic-to-OpenAI/NVIDIA proxy with dashboard."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from config import ServerConfig, load_config
from models import (
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from store import RequestRecord, store
from translation import convert_anthropic_to_openai, convert_openai_to_anthropic

# ---------------------------------------------------------------------------
# Logging – bridge Python logging into our analytics store
# ---------------------------------------------------------------------------

class StoreHandler(logging.Handler):
    """Sends log records to the analytics store (fire-and-forget via asyncio)."""

    def emit(self, record: logging.LogRecord) -> None:
        import asyncio
        msg = self.format(record)
        level = record.levelname
        # Extract req_id from message if present
        req_id: Optional[str] = None
        if msg.startswith("[req_"):
            end = msg.find("]")
            if end != -1:
                req_id = msg[1:end]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(store.add_log(level, req_id, msg))
        except Exception:
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
_root = logging.getLogger()
_store_handler = StoreHandler()
_store_handler.setFormatter(logging.Formatter("%(message)s"))
_root.addHandler(_store_handler)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

cfg: ServerConfig = load_config()
app = FastAPI(title="claude-nvidia-proxy-python", docs_url=None, redoc_url=None)

DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

# Load model mappings
def _load_models() -> dict[str, str]:
    """Load model mappings from model-map.json"""
    try:
        model_map_path = Path(__file__).parent / "model-map.json"
        if model_map_path.exists():
            with open(model_map_path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load model-map.json: %s", e)
    return {}

MODELS = _load_models()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req_id() -> str:
    return f"req_{int(time.time_ns())}"


def _json_error(status: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"type": "proxy_error", "code": code, "message": code}},
    )


def _check_auth(request: Request) -> bool:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[len("bearer "):].strip()
        return hmac.compare_digest(token, cfg.server_api_key)
    x_api_key = request.headers.get("x-api-key", "").strip()
    if x_api_key:
        return hmac.compare_digest(x_api_key, cfg.server_api_key)
    return False


def _trunc(s: str, max_chars: int) -> str:
    if max_chars == 0:
        return "(disabled)"
    runes = list(s)
    if len(runes) > max_chars:
        return "".join(runes[:max_chars]) + "...(truncated)"
    return s


def _map_finish_reason(finish: Optional[str]) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }
    return mapping.get(finish or "", "end_turn")


async def _log_forward(
    req_id: str,
    anthropic_req: AnthropicMessageRequest,
    openai_req: OpenAIChatCompletionRequest,
) -> None:
    summary = {
        "model": anthropic_req.model,
        "max_tokens": anthropic_req.max_tokens,
        "stream": anthropic_req.stream,
        "messages": len(anthropic_req.messages),
        "tools": len(anthropic_req.tools or []),
    }
    logger.info("[%s] inbound summary=%s", req_id, _trunc(json.dumps(summary), cfg.log_body_max))
    logger.info("[%s] forward url=%s", req_id, cfg.upstream_url)
    logger.info(
        "[%s] forward headers=%s",
        req_id,
        _trunc(
            json.dumps({"Content-Type": "application/json", "Authorization": "Bearer <redacted>"}),
            cfg.log_body_max,
        ),
    )
    body_dict = openai_req.model_dump(exclude_none=True)
    logger.info("[%s] forward body=%s", req_id, _trunc(json.dumps(body_dict), cfg.log_body_max))


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "claude-nvidia-proxy-python",
        "upstream_url": cfg.upstream_url,
        "auth_enabled": bool(cfg.server_api_key),
    })


@app.get("/api/stats")
async def stats() -> JSONResponse:
    return JSONResponse(await store.get_stats())


@app.get("/api/logs")
async def logs_history() -> JSONResponse:
    return JSONResponse(await store.get_logs())


@app.get("/api/stream/logs")
async def stream_logs(request: Request) -> StreamingResponse:
    """SSE endpoint: pushes live log entries to the dashboard."""

    async def generator() -> AsyncGenerator[str, None]:
        q = store.subscribe_logs()
        try:
            # Send last 50 logs immediately on connect
            recent = await store.get_logs()
            for entry in reversed(recent[:50]):
                yield f"data: {json.dumps(entry)}\n\n"
            # Then stream new entries
            while True:
                if await request.is_disconnected():
                    break
                try:
                    import asyncio
                    entry = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(entry)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            store.unsubscribe_logs(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stream/requests")
async def stream_requests(request: Request) -> StreamingResponse:
    """SSE endpoint: pushes live request updates to the dashboard."""

    async def generator() -> AsyncGenerator[str, None]:
        q = store.subscribe_requests()
        try:
            # Send recent requests immediately on connect
            snap = await store.get_stats()
            for rec in reversed(snap["recent_requests"][:20]):
                yield f"data: {json.dumps(rec)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    import asyncio
                    rec = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(rec)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            store.unsubscribe_requests(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """List available models in OpenAI format"""
    models_list = []
    for model_name in MODELS.keys():
        models_list.append({
            "id": model_name,
            "object": "model",
            "owned_by": "nvidia-proxy",
            "permission": []
        })

    # Sort by model name for consistent output
    models_list.sort(key=lambda m: m["id"])

    return JSONResponse({
        "object": "list",
        "data": models_list
    })


# ---------------------------------------------------------------------------
# Proxy route
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    req_id = _req_id()
    t_start = time.time()

    if cfg.server_api_key and not _check_auth(request):
        logger.warning("[%s] inbound unauthorized", req_id)
        await store.add_log("WARN", req_id, f"[{req_id}] inbound unauthorized")
        return _json_error(401, "unauthorized")

    try:
        body = await request.json()
        anthropic_req = AnthropicMessageRequest.model_validate(body)
    except Exception as exc:
        logger.error("[%s] invalid inbound json: %s", req_id, exc)
        await store.add_log("ERROR", req_id, f"[{req_id}] invalid inbound json: {exc}")
        return _json_error(400, "invalid_json")

    if not anthropic_req.model.strip():
        logger.warning("[%s] missing model", req_id)
        return _json_error(400, "missing_model")

    # Record to store
    record = RequestRecord(
        req_id=req_id,
        ts=t_start,
        model=anthropic_req.model,
        stream=anthropic_req.stream,
        num_messages=len(anthropic_req.messages),
        num_tools=len(anthropic_req.tools or []),
    )
    await store.add_request(record)

    try:
        openai_req = convert_anthropic_to_openai(anthropic_req)
    except Exception as exc:
        logger.error("[%s] request conversion failed: %s", req_id, exc)
        await store.update_request(req_id, status="error", error_code="request_conversion_failed",
                                   duration_ms=round((time.time() - t_start) * 1000, 1))
        return _json_error(400, "request_conversion_failed")

    await _log_forward(req_id, anthropic_req, openai_req)

    if anthropic_req.stream:
        return await _handle_stream(req_id, t_start, openai_req, anthropic_req.model)
    return await _handle_non_stream(req_id, t_start, openai_req, anthropic_req.model)


# ---------------------------------------------------------------------------
# Non-streaming handler
# ---------------------------------------------------------------------------


async def _handle_non_stream(
    req_id: str, t_start: float, openai_req: OpenAIChatCompletionRequest, original_model: str = ""
) -> Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.provider_api_key}",
    }
    body = openai_req.model_dump(exclude_none=True)

    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            resp = await client.post(cfg.upstream_url, headers=headers, json=body)
    except Exception as exc:
        logger.error("[%s] upstream request failed: %s", req_id, exc)
        await store.update_request(req_id, status="error", error_code="upstream_request_failed",
                                   duration_ms=round((time.time() - t_start) * 1000, 1))
        return _json_error(502, "upstream_request_failed")

    duration_ms = round((time.time() - t_start) * 1000, 1)
    logger.info("[%s] upstream status=%d duration_ms=%s", req_id, resp.status_code, duration_ms)

    if resp.status_code < 200 or resp.status_code >= 300:
        raw = resp.text
        if cfg.log_body_max:
            logger.error("[%s] upstream error body=%s", req_id, _trunc(raw, cfg.log_body_max))
        await store.update_request(req_id, status="error", error_code=f"upstream_{resp.status_code}",
                                   duration_ms=duration_ms)
        return Response(content=raw, status_code=resp.status_code, media_type="application/json")

    try:
        openai_resp = OpenAIChatCompletionResponse.model_validate(resp.json())
    except Exception as exc:
        logger.error("[%s] invalid upstream json: %s", req_id, exc)
        await store.update_request(req_id, status="error", error_code="invalid_upstream_json",
                                   duration_ms=duration_ms)
        return _json_error(502, "invalid_upstream_json")

    anthropic_resp: AnthropicMessageResponse = convert_openai_to_anthropic(openai_resp, original_model)
    input_tok = anthropic_resp.usage.input_tokens
    output_tok = anthropic_resp.usage.output_tokens
    text_chars = sum(len(b.get("text", "")) for b in anthropic_resp.content if isinstance(b, dict) and b.get("type") == "text")

    await store.update_request(
        req_id,
        status="ok",
        duration_ms=duration_ms,
        input_tokens=input_tok,
        output_tokens=output_tok,
        text_chars=text_chars,
    )
    return JSONResponse(content=anthropic_resp.model_dump())


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------


async def _handle_stream(
    req_id: str, t_start: float, openai_req: OpenAIChatCompletionRequest, original_model: str = ""
) -> StreamingResponse:
    openai_req.stream = True
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.provider_api_key}",
    }
    body = openai_req.model_dump(exclude_none=True)

    async def event_generator() -> AsyncGenerator[str, None]:
        import asyncio

        chunk_count = 0
        text_chars = 0
        tool_delta_chunks = 0
        tool_args_chars = 0
        finish_reason_seen = ""
        preview_buf: list[str] = []
        saw_done = False

        tool_states: dict[int, dict[str, Any]] = {}
        next_cb_index = 0
        current_cb_index = -1
        current_block_type = ""
        has_text_block = False

        def assign_cb() -> int:
            nonlocal next_cb_index
            idx = next_cb_index
            next_cb_index += 1
            return idx

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", cfg.upstream_url, headers=headers, json=body) as resp:
                    logger.info("[%s] upstream status=%d (stream)", req_id, resp.status_code)

                    if resp.status_code < 200 or resp.status_code >= 300:
                        raw = (await resp.aread()).decode("utf-8", errors="replace")
                        if cfg.log_body_max:
                            logger.error("[%s] upstream error body=%s", req_id, _trunc(raw, cfg.log_body_max))
                        await store.update_request(
                            req_id, status="error", error_code=f"upstream_{resp.status_code}",
                            duration_ms=round((time.time() - t_start) * 1000, 1),
                        )
                        err = json.dumps({"error": {"type": "proxy_error", "code": "upstream_error", "message": raw}})
                        yield f"data: {err}\n\n"
                        return

                    message_id = f"msg_{int(time.time() * 1000)}"
                    response_model = original_model if original_model else openai_req.model
                    payload = json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": message_id, "type": "message", "role": "assistant",
                            "model": response_model, "content": [], "stop_reason": None,
                            "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    })
                    yield f"event: message_start\ndata: {payload}\n\n"

                    queue: asyncio.Queue = asyncio.Queue(maxsize=500)

                    async def producer() -> None:
                        try:
                            async for line in resp.aiter_lines():
                                await queue.put(line)
                        finally:
                            await queue.put(None)

                    producer_task = asyncio.create_task(producer())

                    while True:
                        line = await queue.get()
                        if line is None:
                            break
                        line = line.rstrip("\r\n")
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            saw_done = True
                            break

                        try:
                            chunk = OpenAIChatCompletionChunk.model_validate_json(data)
                        except Exception:
                            continue

                        if not chunk.choices:
                            continue

                        chunk_count += 1
                        choice = chunk.choices[0]
                        delta = choice.delta

                        # --- Tool call deltas ---
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                tool_delta_chunks += 1
                                ti = tc.index if tc.index is not None else 0
                                state = tool_states.get(ti)
                                tc_id = (tc.id or "").strip() or f"call_{int(time.time()*1000)}_{ti}"
                                tc_name = ""
                                if tc.function:
                                    tc_name = (tc.function.name or "").strip() or f"tool_{ti}"

                                if state is None:
                                    if current_cb_index >= 0:
                                        p = json.dumps({"type": "content_block_stop", "index": current_cb_index})
                                        yield f"event: content_block_stop\ndata: {p}\n\n"
                                        current_cb_index = -1
                                        current_block_type = ""
                                    cb_idx = assign_cb()
                                    state = {"cb_index": cb_idx, "id": tc_id, "name": tc_name}
                                    tool_states[ti] = state
                                    p = json.dumps({
                                        "type": "content_block_start", "index": cb_idx,
                                        "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name, "input": {}},
                                    })
                                    yield f"event: content_block_start\ndata: {p}\n\n"
                                    current_cb_index = cb_idx
                                    current_block_type = "tool_use"
                                else:
                                    if not state["id"] and tc_id:
                                        state["id"] = tc_id
                                    if not state["name"] and tc_name:
                                        state["name"] = tc_name
                                    current_cb_index = state["cb_index"]
                                    current_block_type = "tool_use"

                                if tc.function and tc.function.arguments:
                                    tool_args_chars += len(tc.function.arguments)
                                    p = json.dumps({
                                        "type": "content_block_delta", "index": state["cb_index"],
                                        "delta": {"type": "input_json_delta", "partial_json": tc.function.arguments},
                                    })
                                    yield f"event: content_block_delta\ndata: {p}\n\n"

                        # --- Text delta ---
                        if delta.content:
                            text_chars += len(delta.content)
                            if cfg.log_stream_preview_max > 0:
                                remaining = cfg.log_stream_preview_max - sum(len(p) for p in preview_buf)
                                if remaining > 0:
                                    preview_buf.append(delta.content[:remaining])

                            if current_block_type and current_block_type != "text":
                                if current_cb_index >= 0:
                                    p = json.dumps({"type": "content_block_stop", "index": current_cb_index})
                                    yield f"event: content_block_stop\ndata: {p}\n\n"
                                    current_cb_index = -1
                                    current_block_type = ""

                            if not has_text_block:
                                has_text_block = True
                                cb_idx = assign_cb()
                                p = json.dumps({
                                    "type": "content_block_start", "index": cb_idx,
                                    "content_block": {"type": "text", "text": ""},
                                })
                                yield f"event: content_block_start\ndata: {p}\n\n"
                                current_cb_index = cb_idx
                                current_block_type = "text"

                            p = json.dumps({
                                "type": "content_block_delta", "index": current_cb_index,
                                "delta": {"type": "text_delta", "text": delta.content},
                            })
                            yield f"event: content_block_delta\ndata: {p}\n\n"

                        # --- Finish reason ---
                        if choice.finish_reason:
                            finish_reason_seen = choice.finish_reason
                            stop_reason = _map_finish_reason(choice.finish_reason)
                            p = json.dumps({
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                            })
                            yield f"event: message_delta\ndata: {p}\n\n"

                    await producer_task

                    # Close any open block
                    if current_cb_index >= 0:
                        p = json.dumps({"type": "content_block_stop", "index": current_cb_index})
                        yield f"event: content_block_stop\ndata: {p}\n\n"

                    if not finish_reason_seen:
                        p = json.dumps({
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                        })
                        yield f"event: message_delta\ndata: {p}\n\n"

                    p = json.dumps({"type": "message_stop"})
                    yield f"event: message_stop\ndata: {p}\n\n"

                    duration_ms = round((time.time() - t_start) * 1000, 1)
                    if cfg.log_stream_preview_max > 0:
                        preview = "".join(preview_buf)
                        logger.info(
                            "[%s] stream summary chunks=%d text_chars=%d tool_delta_chunks=%d "
                            "tool_args_chars=%d finish_reason=%r saw_done=%s preview=%r",
                            req_id, chunk_count, text_chars, tool_delta_chunks,
                            tool_args_chars, finish_reason_seen, saw_done, preview,
                        )
                    else:
                        logger.info(
                            "[%s] stream summary chunks=%d text_chars=%d finish_reason=%r saw_done=%s",
                            req_id, chunk_count, text_chars, finish_reason_seen, saw_done,
                        )

                    await store.update_request(
                        req_id,
                        status="ok",
                        duration_ms=duration_ms,
                        text_chars=text_chars,
                        chunk_count=chunk_count,
                    )

        except Exception as exc:
            logger.error("[%s] stream proxy error: %s", req_id, exc)
            await store.update_request(
                req_id, status="error", error_code="stream_error",
                duration_ms=round((time.time() - t_start) * 1000, 1),
            )
            err = json.dumps({"error": {"type": "proxy_error", "code": "stream_error", "message": str(exc)}})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("listening on %s:%d", cfg.host, cfg.port)
    logger.info("upstream: %s", cfg.upstream_url)
    logger.info("dashboard: http://%s:%d/", cfg.host, cfg.port)
    if cfg.server_api_key:
        logger.info("inbound auth: enabled")
    else:
        logger.info("inbound auth: disabled (SERVER_API_KEY not set)")

    uvicorn.run(app, host=cfg.host, port=cfg.port)
