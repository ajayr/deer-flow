"""Shared readers for ``messages-tuple`` stream events.

:func:`accumulate_stream_text` turns one ``messages-tuple`` event into the
displayable assistant text it carries. Code that forwards streamed text to a
client (the IM channel manager is the first user) should go through these
helpers, so that every such path applies the same rule for what counts as
assistant output: :func:`is_assistant_stream_type`.

Standard library only; like everything under ``deerflow`` this module never
imports ``app``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_text_content(content: Any) -> str:
    """Extract text from a streaming payload content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    nested = block.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
        return "".join(parts)
    if isinstance(content, Mapping):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def merge_stream_text(existing: str, chunk: str) -> str:
    """Merge either delta text or cumulative text into a single snapshot."""
    if not chunk:
        return existing
    if not existing:
        return chunk
    # Cumulative re-delivery: strictly longer and starts with existing.
    if len(chunk) > len(existing) and chunk.startswith(existing):
        return chunk
    # Everything else is a delta — always append, even when the delta
    # happens to match the buffer suffix (e.g. 'hel' + 'l') or equals
    # the buffer (CJK reduplication: '谢' + '谢' = '谢谢'). Channels feed
    # only delta ('messages-tuple') events to this function; 'values'
    # snapshots are consumed via a separate branch, so a same-content
    # delta (chunk == existing) still represents a fresh token to keep.
    return existing + chunk


def extract_stream_message_id(payload: Any, metadata: Any) -> str | None:
    """Best-effort extraction of the streamed AI message identifier."""
    candidates = [payload, metadata]
    if isinstance(payload, Mapping):
        candidates.append(payload.get("kwargs"))

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in ("id", "message_id"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def stream_payload_type(payload: Mapping[str, Any]) -> str:
    """Resolve the message ``type`` of one ``messages-tuple`` payload.

    Two payload shapes reach this function and they name the message type in
    different places:

    * The shape DeerFlow's own gateway emits (``runtime/serialization.py``
      calls ``model_dump()``): ``type`` is the LangChain literal directly --
      ``"ai"`` / ``"AIMessageChunk"`` / ``"human"`` / ``"tool"`` / ``"system"``.
    * LangChain's ``to_json()`` constructor shape, which
      ``extract_stream_message_id`` and the content extraction below already
      accommodate: the wrapper's own ``type`` is the literal string
      ``"constructor"`` and the real class name is the last element of the
      ``id`` path (``["langchain", "schema", "messages", "AIMessageChunk"]``),
      with the constructor kwargs under ``kwargs``.

    Reading only the top level would classify every constructor-shaped payload
    as ``"constructor"``, which an allowlist rejects (safe) but which would
    also mean hidden context and assistant output are treated identically --
    so the class name is resolved properly instead of guessed.
    """
    raw_type = payload.get("type")
    if isinstance(raw_type, str) and raw_type and raw_type != "constructor":
        return raw_type
    kwargs = payload.get("kwargs")
    if isinstance(kwargs, Mapping):
        nested = kwargs.get("type")
        if isinstance(nested, str) and nested:
            return nested
    lc_path = payload.get("id")
    if isinstance(lc_path, (list, tuple)) and lc_path:
        tail = lc_path[-1]
        if isinstance(tail, str) and tail:
            return tail
    return raw_type if isinstance(raw_type, str) else ""


def is_assistant_stream_type(payload_type: str) -> bool:
    """Is this message type assistant output, i.e. text a client may be shown?

    An ALLOWLIST, deliberately.  The previous denylist ("reject anything whose
    type contains 'tool'") published every other message type, and DeerFlow
    writes hidden model context into the ``messages`` channel as ordinary
    messages: ``DynamicContextMiddleware`` injects the ``<memory>`` block as a
    hidden ``HumanMessage`` (``type == "human"``) and rewrites the user's own
    turn into a new ``HumanMessage``, and ``DurableContextMiddleware`` injects a
    hidden ``<durable_context_data>`` ``HumanMessage``.  LangGraph fans state
    writes out on the ``messages-tuple`` stream, so all of those reached the
    channel as if they were the assistant's reply -- proved live on a Buzz
    relay, where each streaming update is an immutable public Nostr event and a
    later corrective edit cannot unpublish the leaked one.

    The accepted spellings are the ones assistant output actually carries:
    LangChain serializes ``AIMessage.type`` as ``"ai"`` and
    ``AIMessageChunk.type`` as ``"AIMessageChunk"``; ``"assistant"`` is the
    OpenAI-style spelling a foreign runtime may use.  Matching is by prefix
    rather than substring because a substring test is not safe here -- ordinary
    English words contain "ai" ("chain", "domain"), so ``"ai" in type`` would
    admit a future/foreign type name by accident, which is exactly the class of
    mistake this allowlist exists to prevent.  No LangChain message type other
    than the AI ones begins with "ai" or "assistant".
    """
    normalized = payload_type.strip().lower()
    return normalized.startswith(("ai", "assistant"))


def accumulate_stream_text(
    buffers: dict[str, str],
    current_message_id: str | None,
    event_data: Any,
) -> tuple[str | None, str | None]:
    """Convert a ``messages-tuple`` event into the latest displayable AI text.

    Only assistant output is displayable.  Hidden human/system context (memory
    facts, durable context, the middleware-rewritten echo of the user's own
    message) and tool traffic must never be forwarded to a client; see
    :func:`is_assistant_stream_type`.

    A bare ``str`` payload -- previously accepted here and buffered under the
    current message id -- carries no type information at all, so it cannot be
    attributed to the assistant.  Nothing in DeerFlow produces it (the gateway
    always serializes a ``messages-tuple`` chunk as ``[message_dict, metadata]``
    via ``runtime/serialization.py::serialize_messages_tuple``), and a runtime
    that did emit raw text deltas would emit hidden context the same way, with
    no way to tell them apart.  Under an allowlist an unattributable payload is
    dropped rather than published.
    """
    payload = event_data
    metadata: Any = None
    if isinstance(event_data, (list, tuple)):
        if event_data:
            payload = event_data[0]
        if len(event_data) > 1:
            metadata = event_data[1]

    if not isinstance(payload, Mapping):
        return None, current_message_id

    if not is_assistant_stream_type(stream_payload_type(payload)):
        return None, current_message_id

    text = extract_text_content(payload.get("content"))
    if not text and isinstance(payload.get("kwargs"), Mapping):
        text = extract_text_content(payload["kwargs"].get("content"))
    if not text:
        return None, current_message_id

    message_id = extract_stream_message_id(payload, metadata) or current_message_id or "__default__"
    buffers[message_id] = merge_stream_text(buffers.get(message_id, ""), text)
    return buffers[message_id], message_id
