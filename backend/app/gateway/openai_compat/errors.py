"""In-band errors: what a stream says when it fails after it has started sending.

A response that has already sent its headers cannot change its status code, so the failure
travels as one more ``data:`` payload in the OpenAI error envelope and the stream closes
without ``[DONE]``. The messages are fixed, short and carry nothing from the failure itself:
no stack traces, provider URLs or keys reach the client, and Conduit prefixes its own text
and truncates at 512 characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InBandError:
    """One error a started stream can end with."""

    code: str
    message: str
    type: str = "server_error"

    def body(self) -> dict[str, Any]:
        """The OpenAI error envelope for this error."""
        return {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}


RUN_ERROR = InBandError("run_error", "The model provider failed while answering. Send the message again to retry.")
EMPTY_RESPONSE = InBandError("empty_response", "The agent finished without producing an answer. Send the message again to retry.")
