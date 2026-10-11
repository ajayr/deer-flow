"""Models for the OpenAI-compatible API: the subset of the OpenAI schema this API serves.

The response models describe what the router sends. ``ChatCompletionRequest`` only documents the request: the router reads
the raw body and applies ``params.POLICY``, so a malformed request is answered in the OpenAI error envelope, not with a 422.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelObject(BaseModel):
    """One entry of ``GET /models``: an OpenAI ``model`` object plus the fields Open WebUI-style clients read."""

    id: str = Field(..., description="Id to send as `model` in a request: `<model>` or `<model>:<mode>`")
    object: Literal["model"] = "model"
    created: int = Field(..., description="Unix time the Gateway process started")
    owned_by: str = "deerflow"
    name: str = Field(..., description="Display name; a mode variant carries its mode in parentheses")
    description: str | None = Field(default=None, description="The model's description; omitted when it has none")
    is_multimodal: bool = Field(..., description="Whether the model accepts images; false until image input is supported")
    context_length: int = Field(..., description="Advertised context window; the same constant for every entry")


class ModelList(BaseModel):
    """Response of ``GET /models``."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /chat/completions``. Only the fields listed here change what happens; the rest follow ``params.POLICY``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="An id from `GET /models`: `<model>` or `<model>:<mode>`")
    messages: list[dict[str, Any]] = Field(..., description="The whole conversation so far, as `{role, content}` objects with `user` or `assistant` roles and text content. It must start and end with a user message.")
    stream: bool = Field(default=False, description="Send the answer as server-sent events")
    stream_options: dict[str, Any] | None = Field(default=None, description="`include_usage: true` adds a final chunk with token usage")
    reasoning_effort: str | None = Field(default=None, description="Honored for a model whose reasoning contract has an effort control; ignored (and reported) otherwise")


class CompletionMessage(BaseModel):
    """The assistant message of a non-streaming answer."""

    role: Literal["assistant"] = "assistant"
    content: str = Field(..., description="The answer")
    reasoning_content: str | None = Field(default=None, description="The agent's progress and reasoning; omitted when there is none")


class CompletionChoice(BaseModel):
    """The only choice of a non-streaming answer."""

    index: int = 0
    message: CompletionMessage
    finish_reason: Literal["stop"] = "stop"


class CompletionUsage(BaseModel):
    """Tokens the run used; zero when the provider reported none."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Response of ``POST /chat/completions`` with `stream: false`."""

    id: str = Field(..., description="`chatcmpl-` followed by the run id")
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = Field(..., description="The model id the request named")
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ErrorBody(BaseModel):
    """The `error` object of the OpenAI error envelope."""

    message: str
    type: Literal["invalid_request_error", "server_error"]
    param: str | None = None
    code: str


class ErrorResponse(BaseModel):
    """An error before a response has started, in the OpenAI error envelope."""

    error: ErrorBody
