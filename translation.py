"""translation.py – Convert between Anthropic and OpenAI request/response formats."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from models import (
    AnthropicContentBlock,
    AnthropicImageSource,
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    AnthropicUsage,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)


# ---------------------------------------------------------------------------
# Model mapping
# ---------------------------------------------------------------------------

_MODEL_MAP: Dict[str, str] = {}

def _load_model_map() -> Dict[str, str]:
    global _MODEL_MAP
    if _MODEL_MAP:
        return _MODEL_MAP

    model_map_path = Path(__file__).parent / "model-map.json"
    if model_map_path.exists():
        try:
            with open(model_map_path) as f:
                _MODEL_MAP = json.load(f)
        except Exception:
            _MODEL_MAP = {}
    return _MODEL_MAP

def _map_model(model: str) -> str:
    mapping = _load_model_map()
    return mapping.get(model, model)

def _reverse_map_model(actual_model: str) -> str:
    """Reverse-map the actual model back to the requested model name."""
    mapping = _load_model_map()
    for requested, actual in mapping.items():
        if actual == actual_model:
            return requested
    return actual_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_system_text(system: Any) -> str:
    """Return plain text for the Anthropic 'system' field (str or block list)."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        return _join_text_blocks([AnthropicContentBlock(**b) if isinstance(b, dict) else b for b in system])
    return ""


def _join_text_blocks(blocks: List[AnthropicContentBlock]) -> str:
    parts: List[str] = []
    for blk in blocks:
        if blk.type == "text" and blk.text:
            parts.append(blk.text)
    return "\n".join(parts)


def _map_finish_reason(finish: Optional[str]) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }
    if not finish:
        return "end_turn"
    return mapping.get(finish, "end_turn")


def _convert_tool_choice(tc: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI format."""
    if not isinstance(tc, dict):
        return tc
    typ = tc.get("type", "")
    if typ in ("auto", "none", "required"):
        return typ
    if typ == "tool":
        name = tc.get("name", "")
        if not name:
            return "auto"
        return {"type": "function", "function": {"name": name}}
    return tc


# ---------------------------------------------------------------------------
# Anthropic → OpenAI request
# ---------------------------------------------------------------------------


def _parse_user_blocks(raw_content: Any) -> List[AnthropicContentBlock]:
    if isinstance(raw_content, list):
        return [AnthropicContentBlock(**b) if isinstance(b, dict) else b for b in raw_content]
    return []


def _user_blocks_to_openai_messages(blocks: List[AnthropicContentBlock]) -> List[Any]:
    out: List[Any] = []

    # tool_result blocks → OpenAI "tool" role messages
    for blk in blocks:
        if blk.type != "tool_result" or not (blk.tool_use_id or "").strip():
            continue
        content_str = ""
        if blk.content is not None:
            if isinstance(blk.content, str):
                content_str = blk.content
            else:
                content_str = json.dumps(blk.content)
        out.append({"role": "tool", "tool_call_id": blk.tool_use_id, "content": content_str})

    # text / image blocks → user message
    parts: List[Any] = []
    for blk in blocks:
        if blk.type == "text" and blk.text:
            parts.append({"type": "text", "text": blk.text})
        elif blk.type == "image" and blk.source:
            src: AnthropicImageSource = blk.source
            url = ""
            if src.type == "base64":
                if src.media_type and src.data:
                    # validate base64
                    try:
                        base64.b64decode(src.data)
                        url = f"data:{src.media_type};base64,{src.data}"
                    except Exception:
                        pass
            elif src.type == "url" and src.url:
                url = src.url
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})

    if not parts:
        out.append({"role": "user", "content": ""})
    elif len(parts) == 1 and parts[0].get("type") == "text":
        out.append({"role": "user", "content": parts[0]["text"]})
    else:
        out.append({"role": "user", "content": parts})

    return out


def _assistant_blocks_to_openai_message(blocks: List[AnthropicContentBlock]) -> Any:
    text = _join_text_blocks(blocks)
    tool_calls: List[Any] = []

    for blk in blocks:
        if blk.type != "tool_use":
            continue
        blk_id = (blk.id or "").strip()
        blk_name = (blk.name or "").strip()
        if not blk_id or not blk_name:
            continue
        args = "{}"
        if blk.input is not None:
            args = json.dumps(blk.input) if not isinstance(blk.input, str) else blk.input
        tool_calls.append({
            "id": blk_id,
            "type": "function",
            "function": {"name": blk_name, "arguments": args},
        })

    msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def convert_anthropic_to_openai(req: AnthropicMessageRequest) -> OpenAIChatCompletionRequest:
    """Convert Anthropic request to OpenAI format, mapping the model."""
    messages: List[Any] = []

    sys_text = _extract_system_text(req.system).strip()
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for m in req.messages:
        if isinstance(m, dict):
            role = m.get("role", "").strip()
            content = m.get("content")
        else:
            role = m.role.strip()
            content = m.content

        if not role:
            continue

        # content can be a plain string or a list of blocks
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            blocks = [AnthropicContentBlock(**b) if isinstance(b, dict) else b for b in content]
            if role == "user":
                messages.extend(_user_blocks_to_openai_messages(blocks))
            elif role == "assistant":
                messages.append(_assistant_blocks_to_openai_message(blocks))
            else:
                messages.append({"role": role, "content": _join_text_blocks(blocks)})
        else:
            messages.append({"role": role, "content": str(content) if content is not None else ""})

    tools: Optional[List[Any]] = None
    if req.tools:
        tools = []
        for t in req.tools:
            params = t.input_schema
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params,
                },
            })

    tool_choice: Any = None
    if req.tool_choice is not None:
        tool_choice = _convert_tool_choice(req.tool_choice)

    return OpenAIChatCompletionRequest(
        model=_map_model(req.model),
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        tools=tools,
        tool_choice=tool_choice,
    )


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response (non-streaming)
# ---------------------------------------------------------------------------


def convert_openai_to_anthropic(resp: OpenAIChatCompletionResponse, original_model: str = "") -> AnthropicMessageResponse:
    content: List[Any] = []
    finish_reason: Optional[str] = None

    if resp.choices:
        ch = resp.choices[0]
        finish_reason = ch.finish_reason
        msg = ch.message
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments
                parsed: Any = {}
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except Exception:
                        parsed = {}
                elif isinstance(raw_args, dict):
                    parsed = raw_args
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": parsed,
                })

    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    if resp.usage:
        u = resp.usage
        if u.prompt_tokens_details and isinstance(u.prompt_tokens_details, dict):
            cache_read = u.prompt_tokens_details.get("cached_tokens", 0)
        input_tokens = u.prompt_tokens - cache_read
        output_tokens = u.completion_tokens

    # Use original model name if provided (mask the response)
    response_model = original_model if original_model else (resp.model or "")

    return AnthropicMessageResponse(
        id=f"msg_{int(time.time() * 1000)}",
        type="message",
        role="assistant",
        model=response_model,
        content=content,
        stop_reason=_map_finish_reason(finish_reason),
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI streaming → Anthropic SSE events
# ---------------------------------------------------------------------------


def stream_openai_to_anthropic_events(
    model: str,
    lines: Generator[str, None, None],
    log_stream_preview_max: int = 256,
):
    """
    Yield (event_name, payload_dict) tuples by translating OpenAI SSE chunks
    into Anthropic SSE events, exactly mirroring the Go proxyStream logic.
    """
    message_id = f"msg_{int(time.time() * 1000)}"

    yield "message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    chunk_count = 0
    text_chars = 0
    tool_delta_chunks = 0
    tool_args_chars = 0
    finish_reason: Optional[str] = None
    preview_buf: List[str] = []
    saw_done = False

    # Track per-tool-index state
    tool_states: Dict[int, Dict[str, Any]] = {}  # index → {cb_index, id, name}
    next_cb_index = 0
    current_cb_index = -1
    current_block_type = ""  # "text" | "tool_use"
    has_text_block = False

    def assign_cb_index() -> int:
        nonlocal next_cb_index
        idx = next_cb_index
        next_cb_index += 1
        return idx

    def close_current_block():
        nonlocal current_cb_index, current_block_type
        if current_cb_index >= 0:
            yield_buf.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": current_cb_index,
            }))
            current_cb_index = -1
            current_block_type = ""

    yield_buf: List[Any] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
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
                tool_index = tc.index if tc.index is not None else 0
                state = tool_states.get(tool_index)

                tc_id = (tc.id or "").strip() or f"call_{int(time.time()*1000)}_{tool_index}"
                tc_name = ""
                if tc.function:
                    tc_name = (tc.function.name or "").strip() or f"tool_{tool_index}"

                if state is None:
                    close_current_block()
                    cb_idx = assign_cb_index()
                    state = {"cb_index": cb_idx, "id": tc_id, "name": tc_name}
                    tool_states[tool_index] = state
                    yield_buf.append(("content_block_start", {
                        "type": "content_block_start",
                        "index": cb_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": {},
                        },
                    }))
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
                    args_part = tc.function.arguments
                    tool_args_chars += len(args_part)
                    yield_buf.append(("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state["cb_index"],
                        "delta": {"type": "input_json_delta", "partial_json": args_part},
                    }))

        # --- Text delta ---
        if delta.content:
            text_chars += len(delta.content)
            if log_stream_preview_max > 0:
                remaining = log_stream_preview_max - sum(len(p) for p in preview_buf)
                if remaining > 0:
                    preview_buf.append(delta.content[:remaining])

            if current_block_type and current_block_type != "text":
                close_current_block()

            if not has_text_block:
                has_text_block = True
                cb_idx = assign_cb_index()
                yield_buf.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": cb_idx,
                    "content_block": {"type": "text", "text": ""},
                }))
                current_cb_index = cb_idx
                current_block_type = "text"

            yield_buf.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": current_cb_index,
                "delta": {"type": "text_delta", "text": delta.content},
            }))

        # --- Finish reason ---
        if choice.finish_reason:
            finish_reason = choice.finish_reason
            stop_reason = _map_finish_reason(choice.finish_reason)
            yield_buf.append(("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }))

        # Yield everything accumulated so far
        for item in yield_buf:
            yield item
        yield_buf.clear()

    # Drain yield_buf (shouldn't be needed after loop, but just in case)
    for item in yield_buf:
        yield item
    yield_buf.clear()

    # Close any open content block
    if current_cb_index >= 0:
        yield "content_block_stop", {
            "type": "content_block_stop",
            "index": current_cb_index,
        }

    # Ensure message_delta is always emitted
    if not finish_reason:
        yield "message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    yield "message_stop", {"type": "message_stop"}
