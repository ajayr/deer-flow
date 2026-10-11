"""Recorded run streams, translated: what a Conduit client receives for each kind of run.

``tests/fixtures/openai_compat/streams/*.json`` hold the frames a real LangGraph agent publishes
for each kind of run (see ``tests/support/stream_frames.py`` for what is real and what is scripted).
They are replayed here through the translator on a fake clock. A second test records every scenario
again and checks that the committed copy still translates exactly as the fresh one does, so the
fixtures cannot drift from what the installed LangGraph and LangChain actually emit, while a change
in a field the translator never reads does not fail an unrelated pull request. After a library
upgrade that changes the frames, regenerate them with
``DEERFLOW_WRITE_FIXTURES=1 pytest tests/test_openai_compat_translator_streams.py`` and review the diff.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from support.conduit_rules import assert_conduit_rules, collect, shape
from support.stream_frames import FIXTURE_DIR, SCENARIOS, Recording, load_recording, record_scenario

from app.gateway.openai_compat.translator import Output, StreamTranslator, Usage, stored_answer, to_sse

USAGE = Usage(prompt_tokens=12, completion_tokens=4, total_tokens=16)
ENDING = [("finish", "stop"), ("usage", {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}), ("done", "")]

#: Runs in which a middleware rewrote the last AI message after it streamed, so the stream alone
#: cannot carry the answer the thread stores.
REWRITTEN = {"safety_stop_after_narration", "safety_stop_without_narration"}


def translate(recording: Recording, *, step: float = 0.05, stored: bool = False, **options) -> tuple[StreamTranslator, list[Output]]:
    """Replay a recording with ``step`` seconds between frames, then finish the way the router does.

    With ``stored`` the router's second source is used as well: the answer the thread kept.
    """
    translator = StreamTranslator(completion_id="chatcmpl-run-1", model="deerflow:pro", created=1790000000, include_usage=True, **options)
    clock = 0.0
    outputs = translator.open(now=clock)
    for frame in recording.events:
        clock += step
        outputs += translator.feed(frame["event"], frame["data"], now=clock)
    clock += step
    outputs += translator.finish(now=clock, usage=USAGE, stored=stored_answer(recording.turn_messages) if stored else None)
    return translator, outputs


def _tool_message_text(recording: Recording) -> str:
    return next(message["content"] for message in recording.turn_messages if message["type"] == "tool")


# ── what each kind of run looks like to the client ───────────────────────────


def test_a_plain_answer_is_content():
    _, outputs = translate(load_recording("plain_answer"))

    assert shape(outputs) == [("role", "assistant"), ("content", "Hello, world."), *ENDING]


def test_narration_before_a_tool_call_and_the_tool_lines_are_reasoning_and_only_the_answer_is_content():
    _, outputs = translate(load_recording("tool_run_with_narration"))

    assert shape(outputs) == [
        ("role", "assistant"),
        ("reasoning", "Let me look that up."),
        ("reasoning", "\nTool web_search: running\n"),
        ("reasoning", "Tool web_search: done\n"),
        ("content", "The docs are online."),
        *ENDING,
    ]


def test_with_no_narration_window_the_narration_is_content_and_the_answer_starts_a_new_paragraph():
    translator, outputs = translate(load_recording("tool_run_with_narration"), narration_window_chars=0)

    assert shape(outputs) == [
        ("role", "assistant"),
        ("content", "Let me look "),
        ("content", "that up."),
        ("reasoning", "Tool web_search: running\n"),
        ("reasoning", "Tool web_search: done\n"),
        ("content", "\n\nThe docs "),
        ("content", "are online."),
        *ENDING,
    ]
    assert translator.content_text == "Let me look that up.\n\nThe docs are online."


def test_the_models_reasoning_is_reasoning_content():
    _, outputs = translate(load_recording("thinking"))

    assert shape(outputs) == [("role", "assistant"), ("reasoning", "Add the "), ("reasoning", "numbers."), ("content", "Four."), *ENDING]


def test_a_subagent_run_shows_numbered_status_lines_and_nothing_from_the_subagent():
    _, outputs = translate(load_recording("subagent_run"))

    assert shape(outputs) == [
        ("role", "assistant"),
        ("reasoning", "Delegating."),
        ("reasoning", "\nTool task: running\n"),
        ("reasoning", "Subagent 1: started\n"),
        ("reasoning", "Subagent 1: running\n"),
        ("reasoning", "Subagent 1: done\n"),
        ("reasoning", "Tool task: done\n"),
        ("content", "The subagent says 42."),
        *ENDING,
    ]
    text = "".join(str(value) for _, value in shape(outputs))
    assert "Working on it." not in text and "find the answer" not in text and "general-purpose" not in text


def test_a_clarification_question_is_the_answer():
    recording = load_recording("clarification")
    question = _tool_message_text(recording)

    _, outputs = translate(recording)

    assert "Which environment?" in question
    assert shape(outputs) == [("role", "assistant"), ("reasoning", "I need to check something first."), ("content", question), *ENDING]


def test_a_provider_failure_ends_the_stream_in_band_without_done():
    translator, outputs = translate(load_recording("provider_failure"))

    assert shape(outputs) == [("role", "assistant"), ("error", "run_error")]
    assert translator.closed


def test_a_retried_model_call_sends_only_the_answer_of_the_attempt_that_worked():
    recording = load_recording("model_call_retried")

    translator, outputs = translate(recording)

    assert any(frame["event"] == "custom" and frame["data"]["type"] == "llm_retry" for frame in recording.events)
    assert shape(outputs) == [("role", "assistant"), ("content", "The capital of France is Paris."), *ENDING]
    assert "SECRET" not in "".join(to_sse(output) for output in outputs)


def test_without_the_stored_answer_a_safety_stop_after_narration_looks_like_the_narration_was_the_answer():
    _, outputs = translate(load_recording("safety_stop_after_narration"))

    assert shape(outputs) == [
        ("role", "assistant"),
        ("reasoning", "Let me write that file."),
        ("reasoning", "\nTool web_search: running\n"),
        ("content", "Let me write that file."),
        *ENDING,
    ]


def test_the_stored_answer_gives_a_safety_stop_after_narration_its_explanation():
    recording = load_recording("safety_stop_after_narration")

    translator, outputs = translate(recording, stored=True)

    explanation = collect(outputs)[0]
    assert explanation == translator.content_text == stored_answer(recording.turn_messages)
    assert explanation.startswith("Let me write that file.\n\n") and "safety-related" in explanation
    assert shape(outputs)[-3:] == ENDING


def test_without_the_stored_answer_a_safety_stop_without_narration_ends_with_empty_response():
    _, outputs = translate(load_recording("safety_stop_without_narration"))

    assert shape(outputs) == [("role", "assistant"), ("reasoning", "Tool web_search: running\n"), ("error", "empty_response")]


def test_the_stored_answer_gives_a_safety_stop_without_narration_its_explanation():
    recording = load_recording("safety_stop_without_narration")

    translator, outputs = translate(recording, stored=True)

    explanation = collect(outputs)[0]
    assert explanation == translator.content_text == stored_answer(recording.turn_messages)
    assert explanation.startswith("The model provider stopped this response") and "safety-related" in explanation


# ── properties of every stream ───────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("window", [0, 10, 200])
@pytest.mark.parametrize("step", [0.05, 2.0])
@pytest.mark.parametrize("stored", [False, True])
def test_every_stream_obeys_the_client_rules_whatever_the_window_and_the_pace(name, window, step, stored):
    _, outputs = translate(load_recording(name), step=step, stored=stored, narration_window_chars=window)

    assert_conduit_rules(outputs)


@pytest.mark.parametrize("name", sorted(set(SCENARIOS) - {"provider_failure"}))
def test_with_the_stored_answer_the_client_ends_up_with_exactly_the_answer_the_thread_stores(name):
    recording = load_recording(name)

    translator, outputs = translate(recording, stored=True)

    assert collect(outputs)[0] == translator.content_text == stored_answer(recording.turn_messages)


@pytest.mark.parametrize("name", sorted(set(SCENARIOS) - {"provider_failure"} - REWRITTEN))
def test_the_streamed_answer_is_the_answer_the_thread_stores(name):
    recording = load_recording(name)

    translator, outputs = translate(recording)

    assert collect(outputs)[0] == translator.content_text == stored_answer(recording.turn_messages)


def test_a_failed_run_stores_no_answer():
    assert stored_answer(load_recording("provider_failure").turn_messages) == ""


# ── the fixtures are real ────────────────────────────────────────────────────


def test_every_scenario_has_a_committed_recording_and_no_recording_is_orphaned():
    assert sorted(path.stem for path in FIXTURE_DIR.glob("*.json")) == sorted(SCENARIOS)


def _what_a_client_sees(recording: Recording) -> list[tuple[str, ...]]:
    """The wire output of a recording under both ways of routing narration, and the answer the thread stores."""
    return [tuple(to_sse(output) for output in translate(recording, stored=True, narration_window_chars=window)[1]) + (stored_answer(recording.turn_messages),) for window in (0, 200)]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_committed_recording_still_translates_the_way_what_the_installed_libraries_publish_today_does(name):
    fresh = asyncio.run(record_scenario(name))
    path = FIXTURE_DIR / f"{name}.json"
    if os.environ.get("DEERFLOW_WRITE_FIXTURES") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh.to_json(), encoding="utf-8")

    assert _what_a_client_sees(load_recording(name)) == _what_a_client_sees(fresh), f"{path.name} is out of date: regenerate it with DEERFLOW_WRITE_FIXTURES=1 and review the diff"
