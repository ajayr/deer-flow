"""The non-streaming answer to a chat completion.

Pure: no FastAPI, no I/O. A request with ``stream: false`` consumes the run through the same translator as a streaming
one, so the body is built from what the translator sent, and a run that fails answers with the same error codes as an HTTP
status instead of an in-band frame.
"""

from __future__ import annotations

from typing import Any

from app.gateway.openai_compat.errors import InBandError, OpenAIError
from app.gateway.openai_compat.translator import Usage

# A failure that sending the message again may fix is a bad gateway: the provider failed, or the stream from the run was
# lost. A run somebody stopped is a conflict. Anything the table does not know is the server's own error.
_STATUS = {"run_error": 502, "empty_response": 502, "stream_gap": 502, "run_interrupted": 409, "internal_error": 500}


def failure_status(error: InBandError) -> int:
    """The HTTP status for a run that failed, when the response has not started yet."""
    return _STATUS.get(error.code, 500)


def failure_error(error: InBandError) -> OpenAIError:
    """The error a failed non-streaming run raises: the in-band error's code and message under an HTTP status."""
    return OpenAIError(failure_status(error), error.code, error.message, error_type=error.type)


def completion_body(*, completion_id: str, model: str, created: int, content: str, reasoning: str, usage: Usage | None) -> dict[str, Any]:
    """The ``chat.completion`` object for a run that finished.

    ``usage`` is zero when the provider reported none, because OpenAI clients expect the object to be there.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    totals = usage or Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": totals.prompt_tokens, "completion_tokens": totals.completion_tokens, "total_tokens": totals.total_tokens},
    }
