"""Errors of the OpenAI-compatible API, in the OpenAI error envelope.

Two kinds travel in the same envelope:

* ``OpenAIError`` is raised before a response has started. The router turns it into an HTTP
  status and a JSON body, so every message is short and fixed enough to read in a client.
* ``InBandError`` is what a stream says when it fails after it has started sending. A response
  that has already sent its headers cannot change its status code, so the failure travels as
  one more ``data:`` payload and the stream closes without ``[DONE]``.

Messages carry nothing from a failure itself: no stack traces, provider URLs or keys reach the
client. Conduit prefixes its own text and truncates at 512 characters, so a message stays well
under that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Longest message the router sends. Conduit truncates at 512 characters and prefixes its own text.
MAX_MESSAGE_CHARS = 400

#: Longest slice of a client-supplied value that a message quotes back.
MAX_QUOTED_CHARS = 40

INVALID_REQUEST_TYPE = "invalid_request_error"
SERVER_ERROR_TYPE = "server_error"


def _fit(message: str) -> str:
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    return message[: MAX_MESSAGE_CHARS - 3] + "..."


def quote(value: object) -> str:
    """A client-supplied value, shortened and printable, for use inside a message."""
    # Cleaning maps one character to one, so only the head can reach the message.
    head = str(value)[: MAX_QUOTED_CHARS + 1]
    text = "".join(char if char.isprintable() else " " for char in head)
    if len(text) > MAX_QUOTED_CHARS:
        text = text[: MAX_QUOTED_CHARS - 3] + "..."
    return f"'{text}'"


class OpenAIError(Exception):
    """An error answered with an HTTP status and the OpenAI error envelope."""

    def __init__(self, status_code: int, code: str, message: str, *, param: str | None = None, error_type: str = INVALID_REQUEST_TYPE) -> None:
        self.message = _fit(message)
        super().__init__(self.message)
        self.status_code = status_code
        self.code = code
        self.param = param
        self.error_type = error_type

    def body(self) -> dict[str, Any]:
        """The OpenAI error envelope for this error."""
        return {"error": {"message": self.message, "type": self.error_type, "param": self.param, "code": self.code}}


def invalid_request(message: str, *, param: str | None = None) -> OpenAIError:
    """The body is malformed: not JSON, not an object, or a field of the wrong type."""
    return OpenAIError(400, "invalid_request", message, param=param)


def unsupported_parameter(param: str, message: str) -> OpenAIError:
    """A parameter this API does not support at all."""
    return OpenAIError(400, "unsupported_parameter", message, param=param)


def unsupported_value(param: str, message: str) -> OpenAIError:
    """A supported parameter with a value this API does not support."""
    return OpenAIError(400, "unsupported_value", message, param=param)


def invalid_message(message: str, *, param: str = "messages") -> OpenAIError:
    """A message in the history breaks a message rule."""
    return OpenAIError(400, "invalid_message", message, param=param)


def context_length_exceeded(message: str) -> OpenAIError:
    """The history is larger than the router is willing to ingest."""
    return OpenAIError(400, "context_length_exceeded", message, param="messages")


def model_not_found(message: str) -> OpenAIError:
    """The model or variant does not exist, or the caller may not use it."""
    return OpenAIError(404, "model_not_found", message, param="model")


def internal_error(message: str = "The server could not process the request. Try again.") -> OpenAIError:
    """A failure before streaming started."""
    return OpenAIError(500, "internal_error", message, error_type=SERVER_ERROR_TYPE)


@dataclass(frozen=True, slots=True)
class InBandError:
    """One error a started stream can end with."""

    code: str
    message: str
    type: str = SERVER_ERROR_TYPE

    def body(self) -> dict[str, Any]:
        """The OpenAI error envelope for this error."""
        return {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}


RUN_ERROR = InBandError("run_error", "The model provider failed while answering. Send the message again to retry.")
EMPTY_RESPONSE = InBandError("empty_response", "The agent finished without producing an answer. Send the message again to retry.")
