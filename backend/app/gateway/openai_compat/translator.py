"""Translate the frames of a DeerFlow run into OpenAI ``chat.completion.chunk`` payloads.

Pure: no I/O, no clock, no FastAPI. The router feeds it what the run's stream bridge
publishes (``messages`` and ``custom`` frames, ``error``, ``end``) together with the current
monotonic time, and sends what comes back. Every rule about what a client may see lives here
so recorded streams can test it.

What an OpenAI-style client receives is a conversation, not an agent trace:

* the final answer is ``content``;
* reasoning, the agent's narration before a tool call, one short line per tool and per
  subagent state change are ``reasoning_content``;
* tool arguments and outputs are never forwarded, and tool calls are never emitted as
  ``tool_calls`` (Conduit turns them into an error);
* an ``ask_clarification`` question or a sandbox network approval request is the end of the
  answer: its text is ``content`` and the user's next message is the reply.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.gateway.openai_compat.errors import EMPTY_RESPONSE, RUN_ERROR, InBandError
from deerflow.agents.interaction_policy import ASK_CLARIFICATION_TOOL_NAME
from deerflow.runtime.messages_stream import extract_stream_message_id, extract_text_content, is_assistant_stream_type, stream_payload_type

#: A message's first text is held this long, or until ``narration_window_chars`` characters
#: have arrived, before it is released as answer text (see ``StreamTranslator``).
NARRATION_HOLD_SECONDS = 1.0

#: Without a non-comment output for this long, the stream says it is still working, because
#: Conduit's idle timer on parsed events (5 minutes) is not reset by SSE comments.
SILENCE_NOTICE_SECONDS = 120.0

DEFAULT_NARRATION_WINDOW_CHARS = 200

_MESSAGE_EVENTS = frozenset({"messages", "messages-tuple"})

# The bridge publishes the human-input tool messages with these markers: the worker's own
# detection reads the same artifact (``runtime/runs/worker.py``), and the sandbox's source
# name is ``NETWORK_POLICY_HUMAN_INPUT_SOURCE`` in ``sandbox/middleware.py``.
_HUMAN_INPUT_KIND = "human_input_request"
_NETWORK_APPROVAL_SOURCE = "sandbox_network"

#: A plain-text reply cannot grant a network approval: the grant needs a structured
#: ``human_input_response`` that this API never forwards.
NETWORK_APPROVAL_CLOSING_LINE = "This can only be approved in the DeerFlow web UI; a reply here cannot grant it."

#: The custom event ``LLMErrorHandlingMiddleware`` publishes between two attempts at one model call.
_MODEL_RETRY_EVENT = "llm_retry"

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_SUBAGENT_STATES = {
    "task_started": "started",
    "task_running": "running",
    "task_completed": "done",
    "task_failed": "failed",
    "task_cancelled": "cancelled",
    "task_timed_out": "timed out",
}
_SUBAGENT_TERMINAL = frozenset({"done", "failed", "cancelled", "timed out"})

_TOOL_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.:\-]")
_TOOL_NAME_MAX_CHARS = 64


@dataclass(frozen=True, slots=True)
class Usage:
    """Run totals for the final ``usage`` chunk."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """One ``chat.completion.chunk`` payload."""

    body: dict[str, Any]

    @property
    def delta(self) -> Mapping[str, Any]:
        choices = self.body["choices"]
        return choices[0]["delta"] if choices else {}


@dataclass(frozen=True, slots=True)
class Failed:
    """The stream ends here with an in-band error and no ``[DONE]``."""

    error: InBandError


@dataclass(frozen=True, slots=True)
class Done:
    """``[DONE]``: the last thing a stream sends."""


DONE = Done()

Output = Chunk | Failed | Done


def to_sse(output: Output) -> str:
    """The SSE ``data:`` frame for one output."""
    if isinstance(output, Done):
        return "data: [DONE]\n\n"
    body = output.body if isinstance(output, Chunk) else output.error.body()
    return "data: " + json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n\n"


class _LeadingThink:
    """Splits an answer that opens with ``<think>...</think>`` while it streams.

    Only a block at the very start of a message counts. The web UI also finds blocks later in
    the text, outside code spans; a streaming splitter that did the same would have to guess
    whether an opening tag in the middle of an answer is the model thinking or the answer
    talking about the tag, and a wrong guess would swallow the rest of the answer into the
    reasoning. Text that might still turn out to be part of a tag stays buffered.
    """

    def __init__(self) -> None:
        self._state: Literal["start", "inside", "plain"] = "start"
        self._buffer = ""
        self._trim_answer = False

    def feed(self, text: str) -> tuple[str, str]:
        """Return the ``(reasoning, answer)`` text this chunk completes."""
        if not text:
            return "", ""
        if self._state == "plain":
            return "", self._answer(text)
        self._buffer += text
        if self._state == "start":
            stripped = self._buffer.lstrip()
            if not stripped or (_THINK_OPEN.startswith(stripped) and stripped != _THINK_OPEN):
                return "", ""
            if not stripped.startswith(_THINK_OPEN):
                answer, self._buffer, self._state = self._buffer, "", "plain"
                return "", answer
            self._state, self._buffer = "inside", stripped[len(_THINK_OPEN) :]
        end = self._buffer.find(_THINK_CLOSE)
        if end >= 0:
            reasoning, rest = self._buffer[:end], self._buffer[end + len(_THINK_CLOSE) :]
            self._state, self._buffer, self._trim_answer = "plain", "", True
            return reasoning, self._answer(rest)
        # The end of the buffer may be the start of the closing tag.
        keep = len(_THINK_CLOSE) - 1
        reasoning, self._buffer = self._buffer[:-keep], self._buffer[-keep:]
        return reasoning, ""

    def flush(self) -> tuple[str, str]:
        """Release what is still buffered at the end of a message."""
        buffered, state = self._buffer, self._state
        self._buffer, self._state = "", "plain"
        if state == "inside":
            # An unclosed block: the rest of the message is reasoning, as in the web UI.
            return buffered, ""
        return "", buffered

    def _answer(self, text: str) -> str:
        if self._trim_answer:
            text = text.lstrip("\r\n")
            if text:
                self._trim_answer = False
        return text


def split_leading_think(text: str) -> tuple[str, str]:
    """``(reasoning, answer)`` of a complete message that may open with a ``<think>`` block."""
    splitter = _LeadingThink()
    reasoning, answer = splitter.feed(text)
    tail_reasoning, tail_answer = splitter.flush()
    return reasoning + tail_reasoning, answer + tail_answer


def _thinking_text(content: Any) -> str:
    """Reasoning carried as Anthropic-style ``thinking`` content blocks."""
    if not isinstance(content, list):
        return ""
    return "".join(block["thinking"] for block in content if isinstance(block, Mapping) and block.get("type") == "thinking" and isinstance(block.get("thinking"), str))


def _calls_tools(message: Mapping[str, Any]) -> bool:
    return any(message.get(key) for key in ("tool_call_chunks", "tool_calls", "invalid_tool_calls"))


def _human_input_request(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``human_input`` artifact of a tool message that asks the user something."""
    artifact = message.get("artifact")
    request = artifact.get("human_input") if isinstance(artifact, Mapping) else None
    if isinstance(request, Mapping) and request.get("kind") == _HUMAN_INPUT_KIND:
        return request
    return None


def _human_input_text(message: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    text = extract_text_content(message.get("content")).strip()
    if request.get("source") == _NETWORK_APPROVAL_SOURCE:
        text = f"{text}\n\n{NETWORK_APPROVAL_CLOSING_LINE}".strip()
    return text


def _is_error_fallback(message: Mapping[str, Any]) -> bool:
    kwargs = message.get("additional_kwargs")
    return isinstance(kwargs, Mapping) and bool(kwargs.get("deerflow_error_fallback"))


def _unsent(sent: str, stored: str) -> str:
    """The end of ``stored`` that ``sent`` has not covered; empty when ``sent`` is not the start of ``stored``.

    The comparison ignores whitespace, so a differently wrapped copy of the same text counts as sent.
    """
    expected = "".join(sent.split())
    matched = 0
    for index, char in enumerate(stored):
        if matched == len(expected):
            return stored[index:]
        if char.isspace():
            continue
        if char != expected[matched]:
            return ""
        matched += 1
    return ""


def _safe_tool_name(name: str) -> str:
    return _TOOL_NAME_UNSAFE.sub("_", name.strip())[:_TOOL_NAME_MAX_CHARS] or "tool"


def stored_answer(messages: Sequence[Mapping[str, Any]]) -> str:
    """The answer a finished turn delivers, rebuilt from the turn's stored messages.

    ``messages`` are the messages after the turn's user message, as dicts (the form
    ``deerflow.runtime.serialization.serialize_lc_object`` produces). The classification is
    the translator's: AI messages without tool calls, joined by a blank line, plus the text of
    a human-input request. Error fallback messages are skipped, and a turn whose only text was
    narration falls back to the last of it, as the live stream does.
    """
    parts: list[str] = []
    narration = ""
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        kind = stream_payload_type(message)
        if is_assistant_stream_type(kind):
            if _is_error_fallback(message):
                continue
            _, answer = split_leading_think(extract_text_content(message.get("content")))
            answer = answer.strip()
            if not answer:
                continue
            if _calls_tools(message):
                narration = answer
            else:
                parts.append(answer)
        elif kind == "tool":
            request = _human_input_request(message)
            if request is not None:
                text = _human_input_text(message, request)
                if text:
                    parts.append(text)
    return "\n\n".join(parts) if parts else narration


@dataclass(slots=True)
class _Message:
    """State of the AI message being streamed."""

    id: str | None
    #: This message follows earlier answer text, so its own starts a new paragraph.
    fresh: bool
    think: _LeadingThink = field(default_factory=_LeadingThink)
    mode: Literal["undecided", "content", "reasoning"] = "undecided"
    held: list[str] = field(default_factory=list)
    held_chars: int = 0
    held_since: float = 0.0
    text: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Subagent:
    ordinal: int
    state: str | None = None


class StreamTranslator:
    """Turns one run's frames into the chunks of one chat completion.

    The caller drives it: ``open()`` first, then ``feed()`` for every frame, ``tick()`` whenever
    ``next_deadline()`` passes, and ``finish()`` after ``end`` once it has checked the run's
    final status. Every method returns the outputs to send, in order. After an output of type
    ``Failed`` or ``Done`` the translator is ``closed`` and accepts nothing more.

    **Narration.** A stream cannot retract text, and a model's text is narration (reasoning)
    rather than answer when a tool call follows it in the same message. So each AI message's
    first ``narration_window_chars`` characters, or its first ``NARRATION_HOLD_SECONDS``, are
    held back: a tool call inside that window routes the held text to ``reasoning_content``,
    and otherwise it is released as ``content`` and streams live. Narration longer than the
    window leaks into ``content``. A window of 0 streams every AI text as ``content``. Text in
    the same frame as a tool call is narration whatever its length.

    **Retries.** When the error-handling middleware retries a model call it publishes a
    ``llm_retry`` event; the text the abandoned attempt still held is dropped, so the retry's
    answer is not sent after a fragment of the attempt that failed. Text the attempt had already
    sent stays: a stream cannot retract it, and the retry's answer starts a new paragraph.

    **Rewritten messages.** Middleware can rewrite the last AI message in place (it strips the
    tool calls of a response the provider stopped for safety, or of a run caught in a loop, and
    appends an explanation). The rewrite reuses the message id, so LangGraph never streams it.
    ``finish(stored=...)`` takes the answer the thread stored and sends whatever part of it the
    stream did not carry.
    """

    def __init__(self, *, completion_id: str, model: str, created: int, narration_window_chars: int = DEFAULT_NARRATION_WINDOW_CHARS, include_usage: bool = False) -> None:
        self._id = completion_id
        self._model = model
        self._created = created
        self._window = narration_window_chars
        self._include_usage = include_usage
        self._now = 0.0
        self._started = False
        self._closed = False
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._answered = False
        self._reasoning_mid_line = False
        self._last_output_at = 0.0
        self._last_real_output_at = 0.0
        self._message: _Message | None = None
        self._last_ai_text = ""
        self._seen_calls: set[tuple[str | None, str, object]] = set()
        self._tool_names: dict[str, str] = {}
        self._subagents: dict[str, _Subagent] = {}

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def content_text(self) -> str:
        """Everything sent so far as ``content``."""
        return "".join(self._content)

    @property
    def reasoning_text(self) -> str:
        """Everything sent so far as ``reasoning_content``."""
        return "".join(self._reasoning)

    def open(self, *, now: float) -> list[Output]:
        """The first chunk, which carries the role."""
        self._now = now
        self._started = True
        self._touch(real=True)
        return [self._chunk({"role": "assistant"})]

    def feed(self, event: str, data: Any, *, now: float) -> list[Output]:
        """Translate one frame of the run's stream."""
        self._now = now
        if self._closed:
            return []
        if event in _MESSAGE_EVENTS:
            outputs = self._on_message(data)
        elif event == "custom":
            outputs = self._on_custom(data)
        elif event == "error":
            return self._fail(RUN_ERROR)
        elif event == "end":
            outputs = self._close_message()
        else:
            # metadata, values, updates, and anything under a subgraph namespace.
            outputs = []
        if self._closed:
            return outputs
        return outputs + self._expire_hold()

    def tick(self, *, now: float) -> list[Output]:
        """Time-driven outputs: a held message that is due, and the still-working notice."""
        self._now = now
        if self._closed:
            return []
        return self._expire_hold() + self._silence_notice()

    def next_deadline(self) -> float | None:
        """When ``tick()`` next has something to do, on the clock ``now`` is read from."""
        if self._closed or not self._started:
            return None
        hold = self._hold_deadline()
        silence = self._silence_deadline()
        return silence if hold is None else min(hold, silence)

    def finish(self, *, now: float, usage: Usage | None = None, failure: InBandError | None = None, stored: str | None = None) -> list[Output]:
        """End the stream after ``end``, once the run's final status is known.

        ``failure`` is the router's verdict from the run record (a run that errored, timed out
        or was interrupted); without one the run is complete. ``stored`` is the answer the
        thread kept for this turn (``stored_answer`` of its messages): whatever part of it the
        stream did not carry is sent as ``content``, so a message a middleware rewrote after it
        streamed still reaches the client. A run that still has no answer text re-sends the last
        AI text as ``content``, and ends with ``empty_response`` when there is none, because a
        client treats a stream with no content as broken. The usage chunk is sent only when
        ``include_usage`` was set and ``usage`` is known.
        """
        self._now = now
        if self._closed:
            return []
        outputs = self._close_message()
        if failure is not None:
            self._closed = True
            return [*outputs, Failed(failure)]
        outputs += self._reconcile(stored)
        if not self._answered:
            fallback = self._last_ai_text.strip()
            if not fallback:
                self._closed = True
                return [*outputs, Failed(EMPTY_RESPONSE)]
            outputs += self._emit_content(fallback)
        outputs.append(self._chunk({}, "stop"))
        if self._include_usage and usage is not None:
            outputs.append(self._usage_chunk(usage))
        outputs.append(DONE)
        self._closed = True
        return outputs

    # -- frames -------------------------------------------------------------------------

    def _on_message(self, data: Any) -> list[Output]:
        payload, metadata = data, None
        if isinstance(data, (list, tuple)):
            payload = data[0] if data else None
            metadata = data[1] if len(data) > 1 else None
        if not isinstance(payload, Mapping):
            return []
        kind = stream_payload_type(payload)
        if is_assistant_stream_type(kind):
            return self._on_ai(payload, metadata)
        if kind == "tool":
            return self._on_tool(payload)
        # Human, system and anything else: hidden memory and durable context ride this
        # channel as human messages, so only the types handled above are ever forwarded.
        return []

    def _on_ai(self, payload: Mapping[str, Any], metadata: Any) -> list[Output]:
        if isinstance(metadata, Mapping) and metadata.get("langgraph_node") == "tools":
            # A model called from inside a tool (a scanner, a classifier) is not the agent speaking.
            return []
        if _is_error_fallback(payload):
            return self._fail(RUN_ERROR)
        message, outputs = self._message_for(extract_stream_message_id(payload, metadata))
        content = payload.get("content")
        think_reasoning, answer = message.think.feed(extract_text_content(content))
        kwargs = payload.get("additional_kwargs")
        carried = kwargs.get("reasoning_content") if isinstance(kwargs, Mapping) else None
        outputs += self._emit_reasoning((carried if isinstance(carried, str) else "") + _thinking_text(content) + think_reasoning)
        calls_tools = _calls_tools(payload)
        outputs += self._route_text(message, answer, calls_tools=calls_tools)
        started_calls = self._new_tool_calls(payload, message.id)
        if calls_tools and message.mode != "content":
            outputs += self._release(message, "reasoning")
        for name in started_calls:
            outputs += self._emit_line(f"Tool {_safe_tool_name(name)}: running")
        if payload.get("chunk_position") == "last":
            outputs += self._close_message()
        return outputs

    def _on_tool(self, payload: Mapping[str, Any]) -> list[Output]:
        outputs = self._close_message()
        request = _human_input_request(payload)
        if request is not None:
            return outputs + self._emit_content(_human_input_text(payload, request), new_block=True)
        call_id = payload.get("tool_call_id")
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            name = self._tool_names.get(call_id, "") if isinstance(call_id, str) else ""
        outcome = "failed" if payload.get("status") == "error" else "done"
        return outputs + self._emit_line(f"Tool {_safe_tool_name(name)}: {outcome}")

    def _on_custom(self, data: Any) -> list[Output]:
        if not isinstance(data, Mapping):
            return []
        kind, task_id = data.get("type"), data.get("task_id")
        if kind == _MODEL_RETRY_EVENT:
            # The attempt that failed is over; its held text must not be sent in front of the retry's.
            self._message = None
            return []
        state = _SUBAGENT_STATES.get(kind) if isinstance(kind, str) else None
        if state is None or not isinstance(task_id, str) or not task_id:
            return []
        subagent = self._subagents.get(task_id)
        if subagent is None:
            subagent = self._subagents[task_id] = _Subagent(ordinal=len(self._subagents) + 1)
        if subagent.state in _SUBAGENT_TERMINAL or subagent.state == state or (state == "started" and subagent.state is not None):
            return []
        subagent.state = state
        return self._emit_line(f"Subagent {subagent.ordinal}: {state}")

    def _new_tool_calls(self, payload: Mapping[str, Any], message_id: str | None) -> list[str]:
        """Names of the tool calls this frame starts (the first frame of a call carries its name)."""
        started: list[str] = []
        for source in ("tool_call_chunks", "tool_calls"):
            entries = payload.get(source)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                call_id = entry.get("id") if isinstance(entry.get("id"), str) and entry.get("id") else None
                index = entry.get("index") if isinstance(entry.get("index"), int) else None
                keys = {(message_id, "id", call_id)} if call_id else set()
                if index is not None:
                    keys.add((message_id, "index", index))
                if not keys:
                    keys.add((message_id, "name", name))
                if call_id:
                    self._tool_names[call_id] = name
                if keys & self._seen_calls:
                    self._seen_calls |= keys
                    continue
                self._seen_calls |= keys
                if name != ASK_CLARIFICATION_TOOL_NAME:
                    started.append(name)
        return started

    # -- messages -----------------------------------------------------------------------

    def _message_for(self, message_id: str | None) -> tuple[_Message, list[Output]]:
        current = self._message
        if current is not None and (message_id is None or current.id is None or message_id == current.id):
            if current.id is None:
                current.id = message_id
            return current, []
        outputs = self._close_message()
        self._message = _Message(id=message_id, fresh=bool(self._content))
        return self._message, outputs

    def _close_message(self) -> list[Output]:
        """The current AI message is complete: whatever it still holds is its answer."""
        message = self._message
        if message is None:
            return []
        self._message = None
        reasoning, answer = message.think.flush()
        outputs = self._emit_reasoning(reasoning)
        outputs += self._route_text(message, answer)
        if message.mode == "undecided":
            outputs += self._release(message, "content")
        text = "".join(message.text)
        if text.strip():
            self._last_ai_text = text
        return outputs

    def _route_text(self, message: _Message, text: str, *, calls_tools: bool = False) -> list[Output]:
        if not text:
            return []
        message.text.append(text)
        if message.mode == "reasoning":
            return self._emit_reasoning(text)
        if message.mode == "content" or self._window <= 0:
            message.mode = "content"
            return self._emit_message_content(message, text)
        if not message.held:
            message.held_since = self._now
        message.held.append(text)
        message.held_chars += len(text)
        if message.held_chars >= self._window and not calls_tools:
            return self._release(message, "content")
        return []

    def _release(self, message: _Message, mode: Literal["content", "reasoning"]) -> list[Output]:
        """Decide what a message's held text is, and send it."""
        message.mode = mode
        text, message.held, message.held_chars = "".join(message.held), [], 0
        if mode == "reasoning":
            return self._emit_reasoning(text)
        return self._emit_message_content(message, text)

    def _hold_deadline(self) -> float | None:
        """When the current message's held text is due as ``content``, or None when it holds none.

        ``next_deadline()`` and ``tick()`` compare the clock with this one sum. An elapsed time
        can round below the hold (1.4 - 0.4 is 0.9999999999999999), and a tick at the reported
        deadline would then do nothing.
        """
        message = self._message
        if message is None or message.mode != "undecided" or not message.held:
            return None
        return message.held_since + NARRATION_HOLD_SECONDS

    def _silence_deadline(self) -> float:
        """When the still-working notice is due; one sum for the reason ``_hold_deadline`` gives."""
        return self._last_output_at + SILENCE_NOTICE_SECONDS

    def _expire_hold(self) -> list[Output]:
        message, deadline = self._message, self._hold_deadline()
        if message is None or deadline is None or self._now < deadline:
            return []
        return self._release(message, "content")

    def _silence_notice(self) -> list[Output]:
        if not self._started or self._now < self._silence_deadline():
            return []
        # Rounded to the microsecond: 128.2 - 8.2 is 119.99999999999999, two minutes all the same.
        minutes = max(1, int(round(self._now - self._last_real_output_at, 6) // 60))
        return self._emit_line(f"Still working... ({minutes} min)", real=False)

    def _fail(self, error: InBandError) -> list[Output]:
        """End the stream with an error, after sending what the model had already said."""
        outputs = self._close_message()
        self._closed = True
        return [*outputs, Failed(error)]

    def _reconcile(self, stored: str | None) -> list[Output]:
        """Send the part of the stored answer that the stream did not carry."""
        missing = _unsent(self.content_text, stored) if stored else ""
        if not missing.strip():
            return []
        return self._emit_content(missing if self._content else missing.lstrip())

    # -- output -------------------------------------------------------------------------

    def _touch(self, *, real: bool) -> None:
        self._last_output_at = self._now
        if real:
            self._last_real_output_at = self._now

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> Chunk:
        return Chunk({"id": self._id, "object": "chat.completion.chunk", "created": self._created, "model": self._model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})

    def _usage_chunk(self, usage: Usage) -> Chunk:
        return Chunk(
            {
                "id": self._id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self._model,
                "choices": [],
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens},
            }
        )

    def _emit_content(self, text: str, *, new_block: bool = False) -> list[Output]:
        if not text:
            return []
        if new_block and self._content:
            tail = "".join(self._content[-2:])[-2:]
            text = ("" if tail == "\n\n" else "\n" if tail.endswith("\n") else "\n\n") + text
        self._content.append(text)
        if text.strip():
            self._answered = True
        self._touch(real=True)
        return [self._chunk({"content": text})]

    def _emit_message_content(self, message: _Message, text: str) -> list[Output]:
        outputs = self._emit_content(text, new_block=message.fresh)
        if outputs:
            message.fresh = False
        return outputs

    def _emit_reasoning(self, text: str, *, real: bool = True) -> list[Output]:
        if not text:
            return []
        self._reasoning.append(text)
        self._reasoning_mid_line = not text.endswith("\n")
        self._touch(real=real)
        return [self._chunk({"reasoning_content": text})]

    def _emit_line(self, line: str, *, real: bool = True) -> list[Output]:
        """A status line of reasoning, always on a line of its own."""
        return self._emit_reasoning(("\n" if self._reasoning_mid_line else "") + line + "\n", real=real)
