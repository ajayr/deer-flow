"""Stream translator for the OpenAI-compatible API: run frames in, chat-completion chunks out.

These tests feed hand-built frames on a fake clock. The frames a real LangGraph run publishes
are replayed in ``test_openai_compat_translator_streams.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from support.conduit_rules import shape

from app.gateway.openai_compat import translator as translator_module
from app.gateway.openai_compat.errors import EMPTY_RESPONSE, RUN_ERROR, InBandError
from app.gateway.openai_compat.translator import (
    DONE,
    NARRATION_HOLD_SECONDS,
    NETWORK_APPROVAL_CLOSING_LINE,
    SILENCE_NOTICE_SECONDS,
    Failed,
    Output,
    StreamTranslator,
    Usage,
    split_leading_think,
    stored_answer,
    to_sse,
)
from deerflow.sandbox.middleware import NETWORK_POLICY_HUMAN_INPUT_SOURCE

# ── frame builders ───────────────────────────────────────────────────────────


def ai(
    content: Any = "", *, message_id: str | None = "ai-1", node: str = "model", position: str | None = None, reasoning: str | None = None, calls: Sequence[Mapping[str, Any]] = (), kind: str = "AIMessageChunk", **fields: Any
) -> list[Any]:
    """The ``data`` of a ``messages`` frame for an AI message: ``[message, metadata]``."""
    message = {
        "type": kind,
        "content": content,
        "id": message_id,
        "additional_kwargs": {} if reasoning is None else {"reasoning_content": reasoning},
        "tool_call_chunks": list(calls),
        "tool_calls": [],
        "invalid_tool_calls": [],
        "chunk_position": position,
        **fields,
    }
    return [message, {"langgraph_node": node}]


def call(name: str | None, call_id: str | None = "call_1", index: int | None = 0, args: str = "") -> dict[str, Any]:
    """One entry of a message's ``tool_call_chunks``."""
    return {"type": "tool_call_chunk", "name": name, "id": call_id, "index": index, "args": args}


def tool(name: str | None, content: str = "ok", *, call_id: str = "call_1", status: str = "success", artifact: Any = None) -> list[Any]:
    """The ``data`` of a ``messages`` frame for a tool result."""
    return [{"type": "tool", "name": name, "content": content, "tool_call_id": call_id, "status": status, "artifact": artifact, "id": "tool-1"}, {"langgraph_node": "tools"}]


def human_input(source: str) -> dict[str, Any]:
    """The artifact of a tool message that asks the user something."""
    return {"human_input": {"kind": "human_input_request", "source": source}}


class Run:
    """A translator on a fake clock that remembers everything it has sent."""

    def __init__(self, **options: Any) -> None:
        self.translator = StreamTranslator(completion_id="chatcmpl-1", model="deerflow:pro", created=1790000000, **options)
        self.now = 0.0
        self.outputs: list[Output] = self.translator.open(now=self.now)

    def _clock(self, at: float | None) -> float:
        self.now = self.now if at is None else at
        return self.now

    def feed(self, event: str, data: Any, *, at: float | None = None) -> list[Output]:
        new = self.translator.feed(event, data, now=self._clock(at))
        self.outputs += new
        return new

    def send(self, data: Any, *, at: float | None = None) -> list[Output]:
        return self.feed("messages", data, at=at)

    def tick(self, *, at: float) -> list[Output]:
        new = self.translator.tick(now=self._clock(at))
        self.outputs += new
        return new

    def finish(self, *, at: float | None = None, **kwargs: Any) -> list[Output]:
        new = self.translator.finish(now=self._clock(at), **kwargs)
        self.outputs += new
        return new


# ── open ─────────────────────────────────────────────────────────────────────


def test_open_sends_the_role_chunk_carrying_the_identity_of_the_completion():
    run = Run()

    assert run.outputs[0].body == {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1790000000,
        "model": "deerflow:pro",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    assert len(run.outputs) == 1


# ── narration: what the first words of a message are ─────────────────────────


def test_a_short_answer_is_held_until_its_message_closes_and_then_sent_as_content():
    run = Run()

    assert run.send(ai("Hello"), at=0.1) == []
    assert run.send(ai(", world.")) == []
    assert shape(run.send(ai(position="last"))) == [("content", "Hello, world.")]
    assert run.translator.content_text == "Hello, world."


def test_both_message_event_names_are_read():
    run = Run()

    run.feed("messages-tuple", ai("Hi"))

    assert shape(run.feed("messages-tuple", ai(position="last"))) == [("content", "Hi")]


def test_text_reaching_the_window_size_is_released_at_once_and_streams_live_afterwards():
    run = Run(narration_window_chars=10)

    assert run.send(ai("Hello")) == []
    assert shape(run.send(ai(", world!"))) == [("content", "Hello, world!")]
    assert shape(run.send(ai(" More."))) == [("content", " More.")]


def test_text_held_for_the_hold_time_is_released_by_tick():
    run = Run()
    run.send(ai("Hello"), at=5.0)

    assert run.translator.next_deadline() == 5.0 + NARRATION_HOLD_SECONDS
    assert run.tick(at=5.9) == []
    assert shape(run.tick(at=6.0)) == [("content", "Hello")]
    assert shape(run.send(ai(" there"))) == [("content", " there")]


def test_held_text_is_also_released_by_any_frame_that_arrives_after_the_hold_time():
    run = Run()
    run.send(ai("Hello"), at=0.0)

    assert shape(run.feed("metadata", {"run_id": "run-1"}, at=1.5)) == [("content", "Hello")]


def test_narration_followed_by_a_tool_call_in_the_same_message_is_reasoning():
    run = Run()
    run.send(ai("Let me search."))

    out = run.send(ai(calls=[call("web_search")]))

    assert shape(out) == [("reasoning", "Let me search."), ("reasoning", "\nTool web_search: running\n")]
    assert run.translator.content_text == ""


def test_narration_longer_than_the_window_leaks_into_content_before_its_tool_call():
    run = Run(narration_window_chars=20)
    narration = "I will look this up for you right away."

    assert shape(run.send(ai(narration))) == [("content", narration)]
    assert shape(run.send(ai(calls=[call("web_search")]))) == [("reasoning", "Tool web_search: running\n")]


def test_narration_that_streams_slower_than_the_hold_time_leaks_into_content():
    run = Run()
    run.send(ai("Let me "), at=0.0)

    assert shape(run.send(ai("search."), at=2.0)) == [("content", "Let me search.")]
    assert shape(run.send(ai(calls=[call("web_search")]), at=2.1)) == [("reasoning", "Tool web_search: running\n")]


def test_a_window_of_zero_streams_every_text_as_content():
    run = Run(narration_window_chars=0)

    assert shape(run.send(ai("Let me"))) == [("content", "Let me")]
    assert shape(run.send(ai(" search.", calls=[call("web_search")]))) == [("content", " search."), ("reasoning", "Tool web_search: running\n")]


@pytest.mark.parametrize(
    ("earlier", "separator"),
    [("First.", "\n\n"), ("First.\n", "\n"), ("First.\n\n", "")],
)
def test_a_message_that_follows_answer_text_starts_a_new_paragraph(earlier, separator):
    run = Run(narration_window_chars=0)
    run.send(ai(earlier, message_id="ai-1", position="last"))

    assert shape(run.send(ai("Second.", message_id="ai-2"))) == [("content", f"{separator}Second.")]


def test_a_new_message_id_starts_a_new_message_even_if_the_old_one_never_closed():
    run = Run(narration_window_chars=0)
    run.send(ai("First.", message_id="ai-1"))

    assert shape(run.send(ai("Second.", message_id="ai-2"))) == [("content", "\n\nSecond.")]


def test_text_still_held_when_another_message_begins_is_released_as_content():
    run = Run()
    run.send(ai("Hello", message_id="ai-1"))

    assert shape(run.send(ai("Again", message_id="ai-2"))) == [("content", "Hello")]


def test_a_message_without_an_id_continues_the_current_message():
    run = Run()
    run.send(ai("Hel", message_id=None))
    run.send(ai("lo", message_id="ai-1"))

    assert shape(run.send(ai(position="last", message_id=None))) == [("content", "Hello")]


def test_a_chunk_without_an_id_continues_the_message_before_it():
    run = Run()
    run.send(ai("Hel", message_id="ai-1"))
    run.send(ai("lo", message_id=None))

    assert shape(run.send(ai(position="last", message_id="ai-1"))) == [("content", "Hello")]


def test_text_exactly_the_size_of_the_window_is_released_at_once():
    run = Run(narration_window_chars=5)

    assert shape(run.send(ai("Hello"))) == [("content", "Hello")]


def test_narration_in_the_frame_that_calls_a_tool_is_reasoning_however_long_it_is():
    run = Run(narration_window_chars=20)
    narration = "I will look this up for you right away."

    out = run.send(ai(narration, kind="ai", tool_calls=[{"type": "tool_call", "name": "web_search", "id": "call_1", "args": {}}]))

    assert shape(out) == [("reasoning", narration), ("reasoning", "\nTool web_search: running\n")]
    assert run.translator.content_text == ""


def test_a_tool_call_the_model_garbled_still_makes_the_text_before_it_narration():
    run = Run()
    run.send(ai("Let me search."))

    out = run.send(ai(invalid_tool_calls=[{"type": "invalid_tool_call", "name": "web_search", "id": "call_1", "args": "{broken", "error": "bad json"}]))

    assert shape(out) == [("reasoning", "Let me search.")]


# ── a retried model call ─────────────────────────────────────────────────────


def _retry(**fields: Any) -> dict[str, Any]:
    """The custom event the error-handling middleware publishes between two attempts at one model call."""
    return {"type": "llm_retry", "attempt": 1, "max_attempts": 3, "wait_ms": 1000, "reason": "transient", "message": "provider detail sk-SECRET", **fields}


def test_a_retry_drops_the_text_the_abandoned_attempt_still_held():
    run = Run()
    run.send(ai("The capital of ", message_id="ai-1"))
    run.send(ai("France is", message_id="ai-1"))

    assert run.feed("custom", _retry()) == []
    run.send(ai("The capital of France ", message_id="ai-2"))
    run.send(ai("is Paris.", message_id="ai-2"))
    run.send(ai(position="last", message_id="ai-2"))

    assert run.translator.content_text == "The capital of France is Paris."
    assert "SECRET" not in "".join(to_sse(output) for output in run.outputs)


def test_text_the_abandoned_attempt_already_sent_stays_and_the_retry_starts_a_new_paragraph():
    run = Run(narration_window_chars=0)
    run.send(ai("The capital", message_id="ai-1"))
    run.feed("custom", _retry())

    out = run.send(ai("The capital of France is Paris.", message_id="ai-2", position="last"))

    assert shape(out) == [("content", "\n\nThe capital of France is Paris.")]


def test_a_retry_before_any_text_changes_nothing():
    run = Run()

    assert run.feed("custom", _retry()) == []
    run.send(ai("Hello", message_id="ai-2", position="last"))

    assert run.translator.content_text == "Hello"


# ── reasoning carriers ───────────────────────────────────────────────────────


def test_reasoning_content_in_additional_kwargs_streams_as_reasoning_at_once():
    run = Run()

    assert shape(run.send(ai(reasoning="Add the "))) == [("reasoning", "Add the ")]
    assert shape(run.send(ai(reasoning="numbers."))) == [("reasoning", "numbers.")]


def test_thinking_blocks_are_reasoning_and_text_blocks_are_the_answer():
    run = Run()
    content = [{"type": "thinking", "thinking": "Consider it."}, {"type": "text", "text": "Answer"}]

    assert shape(run.send(ai(content, position="last"))) == [("reasoning", "Consider it."), ("content", "Answer")]


def test_a_leading_think_block_is_reasoning_even_when_its_tags_are_split_across_chunks():
    run = Run()
    for piece in ["<thi", "nk>plan", " it</th", "ink>\n\nAnswer", " text"]:
        run.send(ai(piece))

    assert shape(run.send(ai(position="last"))) == [("content", "Answer text")]
    assert run.translator.reasoning_text == "plan it"


def test_a_think_tag_that_is_not_at_the_start_of_the_message_is_just_text():
    run = Run()
    run.send(ai("Use the <think> tag to reason."))

    assert shape(run.send(ai(position="last"))) == [("content", "Use the <think> tag to reason.")]
    assert run.translator.reasoning_text == ""


def test_an_unclosed_think_block_is_reasoning_to_the_end_of_the_message():
    run = Run()
    run.send(ai("<think>still thinking"))
    run.send(ai(position="last"))

    assert run.translator.reasoning_text == "still thinking"
    assert run.translator.content_text == ""


@pytest.mark.parametrize(
    "reasoning",
    [ai(reasoning="I should search the docs."), ai([{"type": "thinking", "thinking": "I should search the docs."}]), ai("<think>I should search the docs.</think>")],
    ids=["additional_kwargs", "thinking_block", "think_tag"],
)
def test_narration_after_reasoning_that_ends_mid_line_starts_a_paragraph_of_its_own(reasoning):
    run = Run()
    run.send(reasoning)
    run.send(ai("Let me look that up."))

    assert shape(run.send(ai(calls=[call("web_search")]))) == [("reasoning", "\n\nLet me look that up."), ("reasoning", "\nTool web_search: running\n")]
    assert run.translator.reasoning_text == "I should search the docs.\n\nLet me look that up.\nTool web_search: running\n"


# ── tool calls and results ───────────────────────────────────────────────────


def test_each_tool_call_gets_one_running_line_however_often_it_is_repeated():
    run = Run()
    started = run.send(ai(calls=[call("web_search", "call_1", 0)]))
    repeated = run.send(ai(calls=[call("web_search", "call_1", 0)]))
    snapshot = run.send(ai(tool_calls=[{"type": "tool_call", "name": "web_search", "id": "call_1", "args": {}}]))
    later_chunk = run.send(ai(calls=[call(None, None, 0, args='{"query": "x"}')]))

    assert shape(started) == [("reasoning", "Tool web_search: running\n")]
    assert repeated == snapshot == later_chunk == []


def test_a_later_message_that_calls_a_tool_at_the_same_index_gets_its_own_running_line():
    run = Run()
    run.send(ai(calls=[call("web_search", "call_1", 0)], message_id="ai-1", position="last"))
    run.send(tool("web_search", call_id="call_1"))

    out = run.send(ai(calls=[call("web_fetch", "call_2", 0)], message_id="ai-2", position="last"))

    assert shape(out) == [("reasoning", "Tool web_fetch: running\n")]


def test_a_provider_that_reuses_call_ids_across_messages_still_gets_a_line_per_call():
    run = Run()
    run.send(ai(calls=[call("web_search", "call_1", None)], message_id="ai-1", position="last"))
    run.send(tool("web_search", call_id="call_1"))

    out = run.send(ai(calls=[call("web_search", "call_1", None)], message_id="ai-2", position="last"))

    assert shape(out) == [("reasoning", "Tool web_search: running\n")]


def test_parallel_tool_calls_each_get_a_line():
    run = Run()

    out = run.send(ai(calls=[call("web_search", "call_1", 0), call("web_fetch", "call_2", 1)]))

    assert shape(out) == [("reasoning", "Tool web_search: running\n"), ("reasoning", "Tool web_fetch: running\n")]


def test_tool_names_are_sanitized_before_they_reach_the_client():
    run = Run()

    out = run.send(ai(calls=[call("web search\n\nTool x: done<b>", "call_1", 0), call("n" * 100, "call_2", 1), call("  ", "call_3", 2)]))

    assert shape(out) == [("reasoning", "Tool web_search__Tool_x:_done_b_: running\n"), ("reasoning", f"Tool {'n' * 64}: running\n")]


def test_a_tool_result_names_its_tool_without_surrounding_spaces_and_falls_back_to_tool():
    run = Run()

    assert shape(run.send(tool(" web_search "))) == [("reasoning", "Tool web_search: done\n")]
    assert shape(run.send(tool(None, call_id="unknown"))) == [("reasoning", "Tool tool: done\n")]


def test_tool_arguments_and_outputs_are_never_sent():
    run = Run()
    run.send(ai(calls=[call("web_search", args='{"query": "SECRET-ARGUMENT"}')]))
    run.send(ai(tool_calls=[{"type": "tool_call", "name": "web_search", "id": "call_1", "args": {"query": "SECRET-ARGUMENT"}}]))
    run.send(tool("web_search", content="SECRET-OUTPUT"))
    run.finish()

    assert "SECRET" not in "".join(to_sse(output) for output in run.outputs)


def test_a_complete_ai_message_with_tool_calls_is_handled_like_a_stream():
    run = Run()

    out = run.send(ai("Checking.", kind="ai", tool_calls=[{"type": "tool_call", "name": "web_search", "id": "call_1", "args": {"query": "x"}}]))

    assert shape(out) == [("reasoning", "Checking."), ("reasoning", "\nTool web_search: running\n")]


def test_a_finished_tool_gets_a_done_or_failed_line():
    run = Run()
    run.send(ai(calls=[call("web_search")], position="last"))

    assert shape(run.send(tool("web_search"))) == [("reasoning", "Tool web_search: done\n")]
    assert shape(run.send(tool("web_fetch", call_id="call_2", status="error"))) == [("reasoning", "Tool web_fetch: failed\n")]


def test_a_tool_message_without_a_name_uses_the_name_its_call_announced():
    run = Run()
    run.send(ai(calls=[call("fetch_page", "call_9")], position="last"))

    assert shape(run.send(tool(None, call_id="call_9"))) == [("reasoning", "Tool fetch_page: done\n")]


def test_a_tool_result_closes_the_message_that_called_it_even_without_a_last_marker():
    run = Run()
    run.send(ai("Looking.", calls=[call("web_search")], message_id=None))
    run.send(tool("web_search"))
    run.send(ai("The answer.", message_id=None))

    assert shape(run.finish()) == [("content", "The answer."), ("finish", "stop"), ("done", "")]


# ── questions for the user ───────────────────────────────────────────────────


def test_a_clarification_question_is_the_answer_and_its_tool_call_has_no_running_line():
    run = Run()

    called = run.send(ai("I need to check something first.", calls=[call("ask_clarification", "call_c")], position="last"))
    asked = run.send(tool("ask_clarification", "Which environment?", call_id="call_c", artifact=human_input("ask_clarification")))

    assert shape(called) == [("reasoning", "I need to check something first.")]
    assert shape(asked) == [("content", "Which environment?")]
    assert shape(run.finish())[-2:] == [("finish", "stop"), ("done", "")]


def test_the_text_of_a_question_is_trimmed():
    run = Run()

    out = run.send(tool("ask_clarification", "  Which one?\n", artifact=human_input("ask_clarification")))

    assert shape(out) == [("content", "Which one?")]


def test_a_network_approval_request_says_a_reply_cannot_grant_it():
    run = Run()

    out = run.send(tool("bash", "Network access to example.com needs approval.", artifact=human_input(NETWORK_POLICY_HUMAN_INPUT_SOURCE)))

    assert shape(out) == [("content", f"Network access to example.com needs approval.\n\n{NETWORK_APPROVAL_CLOSING_LINE}")]


def test_a_question_after_answer_text_starts_a_new_paragraph():
    run = Run(narration_window_chars=0)
    run.send(ai("Partial answer.", position="last"))

    out = run.send(tool("ask_clarification", "Which one?", artifact=human_input("ask_clarification")))

    assert shape(out) == [("content", "\n\nWhich one?")]


def test_a_tool_message_that_is_not_a_human_input_request_is_an_ordinary_result():
    run = Run()

    out = run.send(tool("web_search", artifact={"human_input": {"kind": "something_else"}}))

    assert shape(out) == [("reasoning", "Tool web_search: done\n")]


def test_the_human_input_markers_match_the_ones_the_runtime_publishes():
    assert translator_module._NETWORK_APPROVAL_SOURCE == NETWORK_POLICY_HUMAN_INPUT_SOURCE
    assert translator_module._HUMAN_INPUT_KIND == "human_input_request"


# ── subagents ────────────────────────────────────────────────────────────────


def _task(kind: str, task_id: str = "t1", **fields: Any) -> dict[str, Any]:
    return {"type": kind, "task_id": task_id, "description": "SECRET-DESCRIPTION", "model_name": "m", **fields}


def test_a_subagent_gets_one_line_per_state_change_and_nothing_after_it_finishes():
    run = Run()
    kinds = ["task_started", "task_running", "task_running", "task_completed", "task_running", "task_started"]

    lines = [shape(run.feed("custom", _task(kind, result="SECRET-RESULT"))) for kind in kinds]

    assert lines == [
        [("reasoning", "Subagent 1: started\n")],
        [("reasoning", "Subagent 1: running\n")],
        [],
        [("reasoning", "Subagent 1: done\n")],
        [],
        [],
    ]
    assert "SECRET" not in "".join(to_sse(output) for output in run.outputs)


def test_a_late_started_update_does_not_move_a_running_subagent_back():
    run = Run()
    run.feed("custom", _task("task_running"))

    assert run.feed("custom", _task("task_started")) == []


@pytest.mark.parametrize(
    ("kind", "state"),
    [("task_failed", "failed"), ("task_cancelled", "cancelled"), ("task_timed_out", "timed out")],
)
def test_every_way_a_subagent_can_end_is_named(kind, state):
    run = Run()

    assert shape(run.feed("custom", _task(kind))) == [("reasoning", f"Subagent 1: {state}\n")]


def test_subagents_are_numbered_in_the_order_they_first_appear():
    run = Run()
    run.feed("custom", _task("task_started", "b"))
    run.feed("custom", _task("task_started", "a"))

    assert shape(run.feed("custom", _task("task_failed", "a"))) == [("reasoning", "Subagent 2: failed\n")]
    assert shape(run.feed("custom", _task("task_completed", "b"))) == [("reasoning", "Subagent 1: done\n")]


def test_a_subagent_line_starts_on_a_line_of_its_own():
    run = Run()
    run.send(ai(reasoning="Thinking"))

    assert shape(run.feed("custom", _task("task_started"))) == [("reasoning", "\nSubagent 1: started\n")]


@pytest.mark.parametrize(
    "data",
    [None, "text", [], {"type": "progress", "task_id": "t1"}, {"type": "task_started"}, {"type": "task_started", "task_id": ""}, {"task_id": "t1"}, {"type": 3, "task_id": "t1"}, {"type": "task_started", "task_id": 7}],
)
def test_custom_frames_that_are_not_subagent_updates_are_ignored(data):
    assert Run().feed("custom", data) == []


# ── failures and frames that are never forwarded ─────────────────────────────


def test_an_error_event_ends_the_stream_with_a_fixed_message_and_nothing_follows():
    run = Run()
    run.send(ai("Partial"))

    out = run.feed("error", {"message": "provider exploded with key sk-SECRET"})

    assert shape(out) == [("content", "Partial"), ("error", "run_error")]
    assert out[-1] == Failed(RUN_ERROR)
    assert "sk-SECRET" not in "".join(to_sse(output) for output in out)
    assert run.translator.closed
    assert run.send(ai("More")) == []
    assert run.tick(at=500.0) == []
    assert run.finish() == []
    assert run.translator.next_deadline() is None


def test_the_error_fallback_message_ends_the_stream_instead_of_being_sent_as_an_answer():
    run = Run()

    out = run.send(ai("The model call failed: connection reset.", additional_kwargs={"deerflow_error_fallback": True}))

    assert out == [Failed(RUN_ERROR)]
    assert run.translator.content_text == ""


def test_text_from_a_model_called_inside_a_tool_is_not_the_agent_speaking():
    run = Run()

    assert run.send(ai("classification: safe", node="tools", position="last")) == []
    assert shape(run.finish()) == [("error", "empty_response")]


@pytest.mark.parametrize("kind", ["human", "HumanMessageChunk", "system", "SystemMessage", "function"])
def test_user_and_system_messages_are_never_forwarded(kind):
    run = Run()

    assert run.send([{"type": kind, "content": "hidden memory block", "id": "h1"}, {"langgraph_node": "model"}]) == []
    assert shape(run.finish()) == [("error", "empty_response")]


@pytest.mark.parametrize("event", ["values", "updates", "metadata", "debug", "messages|tools:abc", "custom|tools:abc", "end|ns"])
def test_other_events_and_events_from_a_namespace_are_ignored(event):
    run = Run()

    assert run.feed(event, ai("subgraph text", position="last")) == []
    assert run.feed(event, _task("task_started")) == []


@pytest.mark.parametrize("data", [None, "text", 3, [], [None], [{"type": "AIMessageChunk"}], [["not", "a", "mapping"], {}]])
def test_message_frames_of_an_unexpected_shape_are_ignored(data):
    run = Run()

    assert run.send(data) == []


# ── finish ───────────────────────────────────────────────────────────────────


def test_finish_sends_what_is_still_held_then_stop_and_done():
    run = Run()
    run.send(ai("Hello"))

    out = run.finish()

    assert shape(out) == [("content", "Hello"), ("finish", "stop"), ("done", "")]
    assert out[-1] is DONE
    assert run.translator.closed
    assert run.finish() == []
    assert run.feed("messages", ai("Late")) == []


@pytest.mark.parametrize(
    ("include_usage", "usage", "sent"),
    [(True, Usage(10, 5, 15), True), (False, Usage(10, 5, 15), False), (True, None, False)],
)
def test_the_usage_chunk_is_sent_only_when_requested_and_known(include_usage, usage, sent):
    run = Run(include_usage=include_usage)
    run.send(ai("Hello"))

    out = run.finish(usage=usage)

    if sent:
        assert shape(out) == [("content", "Hello"), ("finish", "stop"), ("usage", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}), ("done", "")]
        assert out[2].body["choices"] == []
        assert out[2].body["id"] == "chatcmpl-1"
    else:
        assert shape(out) == [("content", "Hello"), ("finish", "stop"), ("done", "")]


def test_a_failure_from_the_router_flushes_held_text_and_ends_without_done():
    run = Run()
    run.send(ai("Partial answer"))
    failure = InBandError("run_interrupted", "The run was interrupted. Send the message again.")

    out = run.finish(failure=failure)

    assert shape(out) == [("content", "Partial answer"), ("error", "run_interrupted")]
    assert run.translator.closed
    assert json.loads(to_sse(out[-1])[len("data: ") : -2])["error"]["code"] == "run_interrupted"


def test_a_failure_ignores_the_stored_answer():
    run = Run()
    failure = InBandError("run_interrupted", "The run was interrupted. Send the message again.")

    assert shape(run.finish(failure=failure, stored="An answer.")) == [("error", "run_interrupted")]


def test_finish_sends_the_stored_answer_when_a_middleware_rewrote_the_message_after_it_streamed():
    run = Run()
    run.send(ai("Let me write that file.", calls=[call("web_search")], position="last"))
    stored = "Let me write that file.\n\nThe model provider stopped this response, so no tool ran."

    out = run.finish(stored=stored)

    assert shape(out) == [("content", stored), ("finish", "stop"), ("done", "")]
    assert run.translator.content_text == stored


def test_finish_sends_only_the_part_of_the_stored_answer_the_stream_did_not_carry():
    run = Run(narration_window_chars=0)
    run.send(ai("The file is ready.", position="last"))

    out = run.finish(stored="The file is ready.\n\nThe provider cut the second step short.")

    assert shape(out) == [("content", "\n\nThe provider cut the second step short."), ("finish", "stop"), ("done", "")]
    assert run.translator.content_text == "The file is ready.\n\nThe provider cut the second step short."


def test_finish_sends_nothing_more_when_the_stream_already_carried_the_stored_answer():
    run = Run()
    run.send(ai("Hello,  world.", position="last"))

    assert shape(run.finish(stored="Hello, world.\n")) == [("finish", "stop"), ("done", "")]


@pytest.mark.parametrize(
    ("sent", "stored"),
    [
        ("Something else entirely.", "A different answer."),
        ("Narration that leaked.", "The real answer, which is quite a bit longer than the narration."),
    ],
)
def test_finish_leaves_the_stream_alone_when_it_is_not_the_start_of_the_stored_answer(sent, stored):
    run = Run()
    run.send(ai(sent, position="last"))

    assert shape(run.finish(stored=stored)) == [("finish", "stop"), ("done", "")]
    assert run.translator.content_text == sent


def test_finish_leaves_the_stream_alone_when_it_is_longer_than_the_stored_answer():
    run = Run()
    run.send(ai("The answer and then some.", position="last"))

    assert shape(run.finish(stored="The answer")) == [("finish", "stop"), ("done", "")]


def test_a_stored_answer_is_sent_without_leading_whitespace_when_the_stream_carried_nothing():
    run = Run()
    run.send(ai("Thinking.", calls=[call("web_search")], position="last"))

    assert shape(run.finish(stored="\n\nThe answer.")) == [("content", "The answer."), ("finish", "stop"), ("done", "")]


@pytest.mark.parametrize("stored", [None, "", "  \n"])
def test_finish_without_a_usable_stored_answer_behaves_as_before(stored):
    run = Run()
    run.send(ai("Hello", position="last"))

    assert shape(run.finish(stored=stored)) == [("finish", "stop"), ("done", "")]


def test_a_stored_answer_takes_the_place_of_the_narration_fallback():
    run = Run()
    run.send(ai("I could not find anything.", calls=[call("web_search")], position="last"))
    run.send(tool("web_search"))

    out = run.finish(stored="I could not find anything.\n\nNo result matched.")

    assert shape(out) == [("content", "I could not find anything.\n\nNo result matched."), ("finish", "stop"), ("done", "")]


def test_a_run_whose_only_text_was_narration_sends_it_again_as_the_answer():
    run = Run()
    run.send(ai("I could not find anything.", calls=[call("web_search")], position="last"))
    run.send(tool("web_search"))

    out = run.finish()

    assert shape(out) == [("content", "I could not find anything."), ("finish", "stop"), ("done", "")]
    assert run.translator.reasoning_text.startswith("I could not find anything.")


def test_a_run_with_no_text_ends_with_empty_response():
    run = Run()

    out = run.finish()

    assert out == [Failed(EMPTY_RESPONSE)]
    assert run.translator.closed


def test_a_run_that_only_reasoned_ends_with_empty_response():
    run = Run()
    run.send(ai(reasoning="Hmm.", position="last"))

    assert shape(run.finish()) == [("error", "empty_response")]


def test_whitespace_is_not_an_answer():
    run = Run()
    run.send(ai("  \n", position="last"))

    assert shape(run.finish()) == [("error", "empty_response")]


def test_the_end_event_releases_what_is_held():
    run = Run()
    run.send(ai("Hello"))

    assert shape(run.feed("end", None)) == [("content", "Hello")]


# ── silence ──────────────────────────────────────────────────────────────────


def test_next_deadline_is_none_before_open():
    translator = StreamTranslator(completion_id="c", model="m", created=1)

    assert translator.next_deadline() is None
    translator.open(now=0.0)
    assert translator.next_deadline() == SILENCE_NOTICE_SECONDS


def test_the_silence_timer_starts_when_the_stream_opens_whatever_the_clock_reads():
    translator = StreamTranslator(completion_id="c", model="m", created=1)
    translator.open(now=86400.0)

    assert translator.next_deadline() == 86400.0 + SILENCE_NOTICE_SECONDS
    assert translator.tick(now=86400.5) == []


def test_streaming_an_answer_keeps_the_silence_timer_from_firing():
    run = Run(narration_window_chars=0)

    for second in range(0, 600, 30):
        run.send(ai("word ", message_id="ai-1"), at=float(second))
        assert run.tick(at=second + 29.0) == []

    assert run.translator.reasoning_text == ""


def test_a_silent_run_says_it_is_still_working_every_two_minutes():
    run = Run()

    assert run.tick(at=119.9) == []
    assert shape(run.tick(at=120.0)) == [("reasoning", "Still working... (2 min)\n")]
    assert run.translator.next_deadline() == 240.0
    assert shape(run.tick(at=240.0)) == [("reasoning", "Still working... (4 min)\n")]


def test_real_output_restarts_the_silence_timer_and_the_minute_count():
    run = Run()
    run.send(ai(reasoning="thinking"), at=100.0)

    assert run.translator.next_deadline() == 100.0 + SILENCE_NOTICE_SECONDS
    assert run.tick(at=219.0) == []
    assert shape(run.tick(at=220.0)) == [("reasoning", "\nStill working... (2 min)\n")]


def test_the_still_working_notice_is_not_an_answer():
    run = Run()
    run.tick(at=SILENCE_NOTICE_SECONDS)

    assert shape(run.finish()) == [("error", "empty_response")]


def test_next_deadline_is_the_earlier_of_the_hold_and_the_silence_notice():
    early = Run()
    early.send(ai("Hello"), at=10.0)
    late = Run()
    late.send(ai("Hello"), at=SILENCE_NOTICE_SECONDS - 0.5)

    assert early.translator.next_deadline() == 10.0 + NARRATION_HOLD_SECONDS
    assert late.translator.next_deadline() == SILENCE_NOTICE_SECONDS


def test_a_tick_at_next_deadline_releases_held_text_on_a_clock_that_is_not_exact_in_binary():
    run = Run()
    run.send(ai("Hello"), at=0.4)

    assert run.translator.next_deadline() == 1.4
    assert shape(run.tick(at=run.translator.next_deadline())) == [("content", "Hello")]


def test_a_tick_at_next_deadline_sends_the_still_working_notice_on_a_clock_that_is_not_exact_in_binary():
    translator = StreamTranslator(completion_id="c", model="m", created=1)
    translator.open(now=8.2)

    assert shape(translator.tick(now=translator.next_deadline())) == [("reasoning", "Still working... (2 min)\n")]


def test_a_tick_at_next_deadline_always_has_something_to_do():
    for tenths in range(3000):
        start = tenths / 10
        held = StreamTranslator(completion_id="c", model="m", created=1)
        held.open(now=start)
        held.feed("messages", ai("Hello"), now=start + 0.2)
        silent = StreamTranslator(completion_id="c", model="m", created=1)
        silent.open(now=start)

        assert shape(held.tick(now=held.next_deadline())) == [("content", "Hello")], start
        assert shape(silent.tick(now=silent.next_deadline())) == [("reasoning", "Still working... (2 min)\n")], start


# ── wire format ──────────────────────────────────────────────────────────────


def test_to_sse_writes_one_compact_json_line_per_chunk_and_keeps_unicode():
    run = Run(narration_window_chars=0)
    chunk = run.send(ai("héllo ✓\nnext"))[0]

    frame = to_sse(chunk)

    assert frame.startswith('data: {"id":"chatcmpl-1","object":"chat.completion.chunk"')
    assert frame.endswith("\n\n")
    assert "\n" not in frame[:-2]
    assert "héllo ✓" in frame
    assert json.loads(frame[len("data: ") : -2])["choices"][0]["delta"] == {"content": "héllo ✓\nnext"}


def test_to_sse_ends_a_stream_with_done_and_wraps_an_error_in_the_openai_envelope():
    assert to_sse(DONE) == "data: [DONE]\n\n"
    assert json.loads(to_sse(Failed(RUN_ERROR))[len("data: ") : -2]) == {"error": {"message": RUN_ERROR.message, "type": "server_error", "param": None, "code": "run_error"}}


# ── split_leading_think ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>a</think>b", ("a", "b")),
        ("  <think>a</think>\n\nb", ("a", "b")),
        ("<think>a</think>", ("a", "")),
        ("<think>unclosed", ("unclosed", "")),
        ("plain", ("", "plain")),
        ("text <think>x</think>", ("", "text <think>x</think>")),
        ("<thi", ("", "<thi")),
        ("<thinking>not a tag</thinking>", ("", "<thinking>not a tag</thinking>")),
        ("", ("", "")),
    ],
)
def test_split_leading_think(text, expected):
    assert split_leading_think(text) == expected


# ── stored_answer ────────────────────────────────────────────────────────────


def _stored_ai(content: Any, *, tool_calls: Sequence[Mapping[str, Any]] = (), **fields: Any) -> dict[str, Any]:
    return {"type": "ai", "content": content, "tool_calls": list(tool_calls), "additional_kwargs": {}, **fields}


def _stored_tool(content: str, *, artifact: Any = None) -> dict[str, Any]:
    return {"type": "tool", "name": "ask_clarification", "content": content, "artifact": artifact, "tool_call_id": "call_1"}


_CALL = {"type": "tool_call", "name": "web_search", "id": "call_1", "args": {}}


def test_stored_answer_of_a_plain_reply_is_its_text():
    assert stored_answer([_stored_ai("Hello.")]) == "Hello."


def test_stored_answer_leaves_out_narration_tool_traffic_and_reasoning():
    messages = [
        _stored_ai("Let me look.", tool_calls=[_CALL]),
        {"type": "tool", "name": "web_search", "content": "3 results", "tool_call_id": "call_1"},
        _stored_ai([{"type": "thinking", "thinking": "Hmm"}, {"type": "text", "text": "<think>plan</think>\n\nThe docs are online."}]),
    ]

    assert stored_answer(messages) == "The docs are online."


def test_stored_answer_joins_several_replies_with_a_blank_line():
    assert stored_answer([_stored_ai("One."), _stored_ai("Two.")]) == "One.\n\nTwo."


def test_stored_answer_of_a_question_is_the_question_text():
    messages = [_stored_ai("Checking.", tool_calls=[_CALL]), _stored_tool("Which environment?", artifact=human_input("ask_clarification"))]

    assert stored_answer(messages) == "Which environment?"


def test_stored_answer_of_a_network_approval_carries_the_closing_line():
    messages = [_stored_tool("Needs approval.", artifact=human_input(NETWORK_POLICY_HUMAN_INPUT_SOURCE))]

    assert stored_answer(messages) == f"Needs approval.\n\n{NETWORK_APPROVAL_CLOSING_LINE}"


def test_stored_answer_of_a_turn_that_only_narrated_is_the_last_narration():
    messages = [_stored_ai("First thought.", tool_calls=[_CALL]), _stored_ai("Second thought.", tool_calls=[_CALL])]

    assert stored_answer(messages) == "Second thought."


def test_stored_answer_skips_error_fallback_messages_and_stray_entries():
    messages = [None, "text", {"no": "type"}, _stored_ai("Failed.", additional_kwargs={"deerflow_error_fallback": True}), _stored_ai("Recovered.")]

    assert stored_answer(messages) == "Recovered."


def test_stored_answer_of_nothing_is_empty():
    assert stored_answer([]) == ""
    assert stored_answer([_stored_ai("   "), _stored_ai([])]) == ""
