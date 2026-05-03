"""models.py – Pydantic models for Anthropic and OpenAI request/response payloads."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Anthropic request types
# ---------------------------------------------------------------------------


class AnthropicImageSource(BaseModel):
    type: str  # "base64" | "url"
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicContentBlock(BaseModel):
    type: str  # "text" | "image" | "tool_use" | "tool_result"
    # text
    text: Optional[str] = None
    # image
    source: Optional[AnthropicImageSource] = None
    # tool_use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None
    # tool_result
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None  # str or list of blocks


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Any] = None


class AnthropicMessageRequest(BaseModel):
    model: str
    max_tokens: int = 1024
    temperature: Optional[float] = None
    stream: bool = False
    system: Optional[Any] = None  # str or list of blocks
    messages: List[Any]  # list of {role, content} dicts
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Any] = None
    thinking: Optional[Any] = None


# ---------------------------------------------------------------------------
# OpenAI request types
# ---------------------------------------------------------------------------


class OpenAIFunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Any] = None


class OpenAITool(BaseModel):
    type: str = "function"
    function: OpenAIFunctionDef


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[Any]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None


# ---------------------------------------------------------------------------
# OpenAI response types (non-streaming)
# ---------------------------------------------------------------------------


class OpenAIFunction(BaseModel):
    name: str
    arguments: Any = None


class OpenAIToolCall(BaseModel):
    id: str
    type: str
    function: OpenAIFunction


class OpenAIMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAIChoice(BaseModel):
    message: OpenAIMessage
    finish_reason: Optional[str] = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[Any] = None


class OpenAIChatCompletionResponse(BaseModel):
    id: str = ""
    model: str = ""
    choices: List[OpenAIChoice] = Field(default_factory=list)
    usage: Optional[OpenAIUsage] = None


# ---------------------------------------------------------------------------
# OpenAI streaming chunk types
# ---------------------------------------------------------------------------


class OpenAIDeltaToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class OpenAIDeltaToolCall(BaseModel):
    index: int = 0
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[OpenAIDeltaToolCallFunction] = None


class OpenAIDelta(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIDeltaToolCall]] = None


class OpenAIStreamChoice(BaseModel):
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIChatCompletionChunk(BaseModel):
    model: Optional[str] = None
    choices: List[OpenAIStreamChoice] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Anthropic response types
# ---------------------------------------------------------------------------


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[Any]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage
