"""The loop that turns one run's stream into one response: waits, keepalives, endings and disconnects.

Every scenario runs on a virtual clock: the loop jumps to the next timer instead of sleeping, so a
stream that is quiet for two minutes takes no time and the times in the assertions are exact.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from app.gateway.openai_compat import driver as driver_module
from app.gateway.openai_compat.driver import (
    FAILED_RUN_WAIT_SECONDS,
    KEEPALIVE_FRAME,
    KEEPALIVE_SECONDS,
    Keepalive,
    RunStream,
    Settled,
    Step,
    failure_of,
    settle_run,
    to_wire,
    turn_messages,
    usage_of,
)
from app.gateway.openai_compat.errors import INTERNAL_ERROR, RUN_ERROR, RUN_INTERRUPTED, STREAM_GAP
from app.gateway.openai_compat.translator import DONE, SILENCE_NOTICE_SECONDS, Chunk, Failed, StreamTranslator, Usage
from deerflow.runtime import END_SENTINEL, HEARTBEAT_SENTINEL, DisconnectMode, RunRecord, RunStatus, StreamEvent, StreamGap

# ── a clock that does not wait ───────────────────────────────────────────────


class VirtualLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to the next timer instead of sleeping."""

    def __init__(self) -> None:
        super().__init__()
        self._virtual_now = 100.0
        real_select = self._selector.select

        def jump(timeout: float | None = None):
            if timeout is None:
                raise RuntimeError("nothing is scheduled and nothing is ready: the scenario would wait forever")
            if timeout:
                self._virtual_now += timeout
            return real_select(0)

        self._selector.select = jump

    def time(self) -> float:
        return self._virtual_now


def play(scenario):
    """Run a scenario coroutine to its end on a virtual clock and return what it returns."""
    loop = VirtualLoop()
    try:
        return loop.run_until_complete(scenario)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# ── scripted streams ─────────────────────────────────────────────────────────


@dataclass
class Probe:
    """What the scripted subscription saw of the loop."""

    #: Items the loop took and then came back for.
    resumed: int = 0
    #: The read in flight was cancelled.
    cancelled: bool = False
    #: The subscription was asked for something at all.
    started: bool = False


async def subscription(steps: tuple[Any, ...], probe: Probe) -> AsyncIterator[Any]:
    """A bridge subscription: items are yielded, numbers are seconds to wait, exceptions are raised."""
    probe.started = True
    try:
        for step in steps:
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, int | float):
                await asyncio.sleep(step)
                continue
            yield step
            probe.resumed += 1
    except asyncio.CancelledError:
        probe.cancelled = True
        raise


def ai(text: str, *, message_id: str = "ai-1", position: str | None = None) -> StreamEvent:
    """A ``messages`` frame for a piece of an AI message."""
    message = {
        "type": "AIMessageChunk",
        "content": text,
        "id": message_id,
        "additional_kwargs": {},
        "tool_call_chunks": [],
        "tool_calls": [],
        "invalid_tool_calls": [],
        "chunk_position": position,
    }
    return StreamEvent(id="1", event="messages", data=[message, {"langgraph_node": "model"}])


def label(step: Step) -> str:
    """One short word for what a step puts on the wire."""
    if isinstance(step, Keepalive):
        return "keepalive"
    if step is DONE:
        return "done"
    if isinstance(step, Failed):
        return f"failed:{step.error.code}"
    assert isinstance(step, Chunk)
    if not step.body["choices"]:
        return "usage"
    choice = step.body["choices"][0]
    if choice["finish_reason"] == "stop":
        return "stop"
    delta = choice["delta"]
    if "role" in delta:
        return "role"
    if "content" in delta:
        return f"content:{delta['content']}"
    return f"reasoning:{delta['reasoning_content']}"


class Client:
    """The end of the connection: it can go away between two checks."""

    def __init__(self) -> None:
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


class Scenario:
    """A stream, a translator and a client on the running loop's clock."""

    def __init__(self, *steps: Any, settled: Settled | None = None, include_usage: bool = False, window: int = 200, keepalive: float | None = None, settle_error: Exception | None = None) -> None:
        loop = asyncio.get_running_loop()
        self.loop = loop
        self.started_at = loop.time()
        self.probe = Probe()
        self.client = Client()
        self.settle_calls = 0
        self.translator = StreamTranslator(completion_id="chatcmpl-1", model="deerflow:pro", created=1790000000, narration_window_chars=window, include_usage=include_usage)

        async def settle() -> Settled:
            self.settle_calls += 1
            if settle_error is not None:
                raise settle_error
            return settled or Settled()

        options = {} if keepalive is None else {"keepalive_seconds": keepalive}
        self.stream = RunStream(self.translator, subscription(steps, self.probe), settle=settle, is_disconnected=self.client.is_disconnected, **options)

    async def run(self) -> list[tuple[float, str]]:
        """Every step with the time it was handed over."""
        return [(self.loop.time() - self.started_at, label(step)) async for step in self.stream.steps()]


def labels(timed: list[tuple[float, str]]) -> list[str]:
    return [name for _, name in timed]


# ── what the outcome helpers decide ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (RunStatus.error, RUN_ERROR),
        (RunStatus.timeout, RUN_ERROR),
        (RunStatus.interrupted, RUN_INTERRUPTED),
        (RunStatus.success, None),
        (RunStatus.running, None),
        (RunStatus.pending, None),
    ],
)
def test_a_run_that_did_not_finish_ends_its_stream_with_the_error_for_its_status(status, failure):
    assert failure_of(status) is failure


def test_a_run_nothing_is_known_about_has_no_usage_no_failure_and_no_stored_answer():
    assert Settled() == Settled(usage=None, failure=None, stored=None)


def test_token_totals_become_the_usage_chunk():
    assert usage_of(input_tokens=3, output_tokens=4, total_tokens=9) == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=9)  # the provider's total is kept, whatever it counts
    assert usage_of(input_tokens=1, output_tokens=0, total_tokens=1) == Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)


def test_a_provider_that_reported_no_tokens_has_no_usage():
    assert usage_of(input_tokens=0, output_tokens=0, total_tokens=0) is None


def _message(kind: str, message_id: str | None, content: str = "") -> dict[str, Any]:
    return {"type": kind, "id": message_id, "content": content}


def test_the_turn_is_what_the_thread_holds_after_the_users_message():
    state = [_message("human", "earlier"), _message("ai", "ai-0", "old"), _message("human", "u-2", "question"), _message("ai", "ai-1", "answer")]

    assert turn_messages(state, "u-2") == [_message("ai", "ai-1", "answer")]


def test_the_turn_is_every_message_after_the_users_not_only_the_next():
    state = [_message("human", "u-1"), _message("ai", "ai-1", "looking"), _message("tool", "t-1", "found"), _message("ai", "ai-2", "answer")]

    assert turn_messages(state, "u-1") == state[1:]


def test_the_state_is_searched_from_its_start():
    state = [_message("human", "u-1"), _message("ai", "ai-1"), _message("human", "u-1__user"), _message("ai", "ai-2")]

    assert turn_messages(state, "u-1") == state[1:]


def test_a_turn_with_nothing_after_the_users_message_is_empty_not_missing():
    assert turn_messages([_message("human", "u-1")], "u-1") == []


def test_the_turn_is_a_list_whatever_sequence_the_thread_state_is():
    assert turn_messages((_message("human", "u-1"), _message("ai", "ai-1", "answer")), "u-1") == [_message("ai", "ai-1", "answer")]


def test_the_first_turn_is_found_under_the_id_the_middleware_gave_the_users_text():
    state = [_message("system", "u-1", "reminder"), _message("human", "u-1__memory", "memory"), _message("human", "u-1__user", "question"), _message("ai", "ai-1", "answer")]

    assert turn_messages(state, "u-1") == [_message("ai", "ai-1", "answer")]


def test_a_message_that_is_not_the_users_is_not_the_start_of_the_turn_even_with_the_same_id():
    state = [_message("system", "u-1", "reminder"), _message("ai", "u-1", "answer")]

    assert turn_messages(state, "u-1") is None


def test_a_turn_that_cannot_be_found_is_none_and_odd_entries_are_skipped():
    assert turn_messages([_message("human", "other")], "u-1") is None
    assert turn_messages(["text", None, 3, _message("human", "u-1"), _message("ai", "ai-1", "answer")], "u-1") == [_message("ai", "ai-1", "answer")]  # type: ignore[list-item]
    assert turn_messages([], "u-1") is None


# ── settling a run ───────────────────────────────────────────────────────────


def _record(status: RunStatus = RunStatus.success, *, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0) -> RunRecord:
    return RunRecord(
        run_id="run-1",
        thread_id="t-1",
        assistant_id=None,
        status=status,
        on_disconnect=DisconnectMode.cancel,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class Thread:
    """The thread state a settling run reads."""

    def __init__(self, messages: Any = (), error: Exception | None = None) -> None:
        self.messages = messages
        self.error = error
        self.reads = 0

    async def read(self) -> Any:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return self.messages


def _settle(record: RunRecord, thread: Thread, user_message_id: str = "u-1") -> Settled:
    return asyncio.run(settle_run(record, user_message_id=user_message_id, read_messages=thread.read))


def test_a_finished_run_is_settled_with_its_token_totals_and_the_answer_its_thread_kept():
    thread = Thread([_message("human", "u-1", "question"), _message("ai", "ai-1", "answer")])

    settled = _settle(_record(input_tokens=3, output_tokens=4, total_tokens=9), thread)

    assert settled == Settled(usage=Usage(3, 4, 9), failure=None, stored="answer")
    assert thread.reads == 1


@pytest.mark.parametrize(("status", "failure"), [(RunStatus.error, RUN_ERROR), (RunStatus.timeout, RUN_ERROR), (RunStatus.interrupted, RUN_INTERRUPTED)])
def test_a_run_that_did_not_finish_is_settled_by_its_status_alone_and_its_thread_is_not_read(status, failure):
    thread = Thread([_message("human", "u-1", "question"), _message("ai", "ai-1", "answer")])

    settled = _settle(_record(status, input_tokens=3, output_tokens=4, total_tokens=9), thread)

    assert settled == Settled(usage=None, failure=failure, stored=None)
    assert thread.reads == 0


def test_a_run_that_reported_no_tokens_is_settled_without_usage_but_with_its_answer():
    thread = Thread([_message("human", "u-1", "question"), _message("ai", "ai-1", "answer")])

    assert _settle(_record(), thread) == Settled(usage=None, failure=None, stored="answer")


def test_the_stored_answer_is_only_what_the_thread_holds_after_the_users_message():
    thread = Thread([_message("human", "u-0", "old question"), _message("ai", "ai-0", "old answer"), _message("human", "u-1", "question"), _message("ai", "ai-1", "answer")])

    assert _settle(_record(), thread).stored == "answer"


def test_the_stored_answer_is_the_final_text_not_the_narration_around_a_tool_call():
    narration = {**_message("ai", "ai-1", "Let me look that up."), "tool_calls": [{"name": "web_search", "args": {}, "id": "call-1"}]}
    thread = Thread([_message("human", "u-1", "question"), narration, _message("tool", "tool-1", "found it"), _message("ai", "ai-2", "Here it is.")])

    assert _settle(_record(), thread).stored == "Here it is."


def test_every_answer_of_the_turn_is_stored_in_order():
    thread = Thread([_message("human", "u-1", "question"), _message("ai", "ai-1", "Part one."), _message("ai", "ai-2", "Part two.")])

    assert _settle(_record(), thread).stored == "Part one.\n\nPart two."


def test_a_turn_that_has_no_text_stored_has_an_empty_stored_answer_not_an_unknown_one():
    assert _settle(_record(), Thread([_message("human", "u-1", "question")])).stored == ""


def test_a_thread_that_does_not_hold_the_users_message_leaves_the_stored_answer_unknown():
    assert _settle(_record(input_tokens=1, output_tokens=1, total_tokens=2), Thread([_message("human", "other", "question")])) == Settled(usage=Usage(1, 1, 2), stored=None)


def test_a_thread_that_cannot_be_read_costs_only_the_stored_answer(caplog):
    thread = Thread(error=RuntimeError("state store down"))

    with caplog.at_level(logging.ERROR, logger=driver_module.logger.name):
        settled = _settle(_record(input_tokens=3, output_tokens=4, total_tokens=9), thread)

    assert settled == Settled(usage=Usage(3, 4, 9), failure=None, stored=None)
    assert any("settle" in record.getMessage() and record.exc_info for record in caplog.records)


def test_a_thread_state_that_is_not_a_list_of_messages_is_a_thread_that_cannot_be_read():
    assert _settle(_record(input_tokens=1, output_tokens=1, total_tokens=2), Thread(None)) == Settled(usage=Usage(1, 1, 2), stored=None)


def test_a_step_is_the_text_the_response_sends():
    chunk = Chunk({"id": "c", "choices": []})

    assert to_wire(Keepalive()) == KEEPALIVE_FRAME == ": keepalive\n\n"
    assert to_wire(DONE) == "data: [DONE]\n\n"
    assert to_wire(chunk) == 'data: {"id":"c","choices":[]}\n\n'
    assert to_wire(Failed(RUN_ERROR)).startswith('data: {"error":{"message":')


# ── the course of a response ─────────────────────────────────────────────────


def test_a_plain_answer_is_the_role_then_the_text_then_stop_and_done():
    async def scenario():
        run = Scenario(ai("Hel"), ai("lo", position="last"), END_SENTINEL)
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "stop", "done"]
    assert [at for at, _ in timed] == [0.0, 0.0, 0.0, 0.0]
    assert run.stream.ended == "done" and run.stream.failure is None
    assert run.translator.closed


def test_the_first_step_is_handed_over_before_the_subscription_is_read():
    async def scenario():
        run = Scenario(ai("Hello", position="last"), END_SENTINEL)
        steps = run.stream.steps()
        first = await anext(steps)
        started = run.probe.started
        await steps.aclose()
        return label(first), started

    assert play(scenario()) == ("role", False)


def test_nothing_is_read_after_the_end_of_the_run():
    async def scenario():
        run = Scenario(ai("Hello", position="last"), END_SENTINEL, ai("late", position="last"))
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed)[-1] == "done"
    assert run.probe.resumed == 1  # the frame before `end`; the loop never came back for another read


def test_the_heartbeats_of_the_bridge_are_not_forwarded():
    async def scenario():
        run = Scenario(HEARTBEAT_SENTINEL, ai("Hello", position="last"), HEARTBEAT_SENTINEL, END_SENTINEL)
        return await run.run()

    assert labels(play(scenario())) == ["role", "content:Hello", "stop", "done"]


def test_a_stream_that_says_nothing_sends_a_keepalive_every_five_seconds():
    async def scenario():
        run = Scenario(12, ai("Hello", position="last"), END_SENTINEL)
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert timed[:3] == [(0.0, "role"), (5.0, "keepalive"), (10.0, "keepalive")]
    assert labels(timed)[3:] == ["content:Hello", "stop", "done"]
    assert [at for at, _ in timed[3:]] == [12.0, 12.0, 12.0]
    assert not run.probe.cancelled  # the read that was waiting is the one that delivered the answer


def test_the_keepalive_interval_can_be_changed():
    async def scenario():
        return await Scenario(7, ai("Hello", position="last"), END_SENTINEL, keepalive=2.0).run()

    assert [at for at, name in play(scenario()) if name == "keepalive"] == [2.0, 4.0, 6.0]


def test_what_was_sent_restarts_the_wait_for_the_next_keepalive():
    async def scenario():
        return await Scenario(4, ai("Hi", position="last"), 4, END_SENTINEL, window=0).run()

    timed = play(scenario())

    assert labels(timed) == ["role", "content:Hi", "stop", "done"]  # the answer at 4 s and the end at 8 s are both inside five seconds of the last thing sent


def test_a_keepalive_follows_five_quiet_seconds_after_the_last_thing_sent_not_after_the_start():
    async def scenario():
        return await Scenario(4, ai("Hi", position="last"), 6, ai("Hello", message_id="ai-2", position="last"), END_SENTINEL, window=0).run()

    timed = play(scenario())

    assert [(at, name) for at, name in timed if name == "keepalive"] == [(9.0, "keepalive")]


def test_frames_that_send_nothing_do_not_postpone_the_keepalive():
    hidden = StreamEvent(id="2", event="values", data={})

    async def scenario():
        return await Scenario(3, hidden, 1.5, hidden, 4, ai("Hello", position="last"), END_SENTINEL).run()

    timed = play(scenario())

    assert timed[:3] == [(0.0, "role"), (5.0, "keepalive"), (8.5, "content:Hello")]


def test_text_the_translator_holds_is_released_when_its_hold_runs_out_not_when_the_next_frame_comes():
    async def scenario():
        return await Scenario(ai("Hi"), 3, END_SENTINEL).run()

    assert play(scenario())[:2] == [(0.0, "role"), (1.0, "content:Hi")]


def test_a_stream_quiet_for_two_minutes_says_it_is_still_working_even_without_a_keepalive_due():
    async def scenario():
        return await Scenario(SILENCE_NOTICE_SECONDS + 10, ai("Hello", position="last"), END_SENTINEL, keepalive=10_000).run()

    timed = play(scenario())

    assert timed[1] == (SILENCE_NOTICE_SECONDS, "reasoning:Still working... (2 min)\n")


def test_a_frame_that_arrives_while_the_loop_waits_is_answered_at_once():
    async def scenario():
        return await Scenario(2.5, ai("Hello", position="last"), 2.5, END_SENTINEL, keepalive=5).run()

    timed = play(scenario())

    assert (2.5, "content:Hello") in timed and timed[-1] == (5.0, "done")


# ── endings ──────────────────────────────────────────────────────────────────


def test_the_end_of_a_run_is_settled_and_its_facts_reach_the_last_chunks():
    async def scenario():
        run = Scenario(ai("Hel"), END_SENTINEL, settled=Settled(usage=Usage(10, 5, 15), stored="Hello"), include_usage=True)
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hel", "content:lo", "stop", "usage", "done"]
    assert run.settle_calls == 1
    assert run.stream.settled == Settled(usage=Usage(10, 5, 15), stored="Hello")


def test_a_run_whose_status_says_it_failed_ends_with_that_error_and_no_done():
    async def scenario():
        run = Scenario(ai("Partial"), END_SENTINEL, settled=Settled(failure=RUN_INTERRUPTED))
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Partial", "failed:run_interrupted"]
    assert run.stream.ended == "failed" and run.stream.failure is RUN_INTERRUPTED


ERROR_FRAME = StreamEvent(id="2", event="error", data={"message": "boom"})


def test_an_error_frame_fails_the_response_once_the_run_has_ended():
    async def scenario():
        run = Scenario(ERROR_FRAME, ai("never sent", position="last"), END_SENTINEL)
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    # What the run still says is read and dropped; its end is taken, and its facts are not needed to report the failure.
    assert labels(timed) == ["role", "failed:run_error"]
    assert run.stream.ended == "failed" and run.stream.failure is RUN_ERROR
    assert run.settle_calls == 0 and run.probe.resumed == 2 and not run.probe.cancelled


def test_a_failure_is_not_reported_before_the_run_has_ended():
    async def scenario():
        run = Scenario(ERROR_FRAME, 1.5, END_SENTINEL)
        return await run.run()

    assert play(scenario()) == [(0, "role"), (1.5, "failed:run_error")]


def test_the_end_of_a_failed_run_is_waited_for_no_longer_than_the_wait_allows(caplog):
    async def scenario():
        run = Scenario(ERROR_FRAME, 60, END_SENTINEL)
        timed = await run.run()
        return run, timed

    with caplog.at_level(logging.DEBUG, logger=driver_module.logger.name):
        run, timed = play(scenario())

    assert timed == [(0, "role"), (3, "failed:run_error")]
    assert run.stream.ended == "failed" and run.probe.cancelled
    assert caplog.records == []


def test_the_wait_for_a_failed_run_is_one_wait_not_one_per_frame():
    async def scenario():
        run = Scenario(ERROR_FRAME, 2, ai("late"), 2, END_SENTINEL)
        return await run.run()

    assert play(scenario()) == [(0, "role"), (3, "failed:run_error")]


def test_the_wait_for_a_failed_run_is_shorter_than_a_keepalive_so_the_response_is_never_quiet_for_long():
    assert 0 < FAILED_RUN_WAIT_SECONDS < KEEPALIVE_SECONDS


def test_a_failed_run_whose_stream_is_lost_while_it_ends_is_reported_at_once(caplog):
    async def scenario():
        gap = StreamGap(requested_event_id=None, earliest_available_event_id=None, latest_available_event_id=None)
        return [await Scenario(ERROR_FRAME, 1, gap, 60, END_SENTINEL).run(), await Scenario(ERROR_FRAME).run()]

    with caplog.at_level(logging.DEBUG, logger=driver_module.logger.name):
        timed = play(scenario())

    # A gap and a subscription that just stops both say that the end of the run will not arrive: they end the wait, quietly.
    assert timed == [[(0, "role"), (1, "failed:run_error")], [(0, "role"), (0, "failed:run_error")]]
    assert caplog.records == []


def test_a_bridge_that_fails_while_a_failed_run_ends_does_not_change_what_the_client_hears(caplog):
    async def scenario():
        run = Scenario(ERROR_FRAME, 1, RuntimeError("redis at 10.1.2.3 refused the connection"), END_SENTINEL)
        return await run.run()

    with caplog.at_level(logging.WARNING, logger=driver_module.logger.name):
        timed = play(scenario())

    assert timed == [(0, "role"), (1, "failed:run_error")]
    assert any("failed run was ending" in record.getMessage() and record.exc_info for record in caplog.records)


def test_a_gap_in_the_stream_ends_the_response_with_stream_gap():
    async def scenario():
        run = Scenario(ai("Hello"), StreamGap(requested_event_id="1", earliest_available_event_id="9", latest_available_event_id="12"), END_SENTINEL)
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "failed:stream_gap"]
    assert run.stream.ended == "failed" and run.stream.failure is STREAM_GAP
    assert run.settle_calls == 0


def test_a_subscription_that_ends_without_the_end_of_the_run_is_a_lost_stream_but_not_an_error_in_the_server(caplog):
    async def scenario():
        run = Scenario(ai("Hello"))
        timed = await run.run()
        return run, timed

    with caplog.at_level(logging.WARNING, logger=driver_module.logger.name):
        run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "failed:stream_gap"]
    assert run.stream.failure is STREAM_GAP and run.settle_calls == 0
    assert not caplog.records


def test_a_bridge_that_fails_is_a_lost_stream_and_the_client_hears_nothing_of_the_cause(caplog):
    async def scenario():
        run = Scenario(ai("Hello"), RuntimeError("redis at 10.1.2.3 refused the connection"))
        timed = await run.run()
        return run, timed

    with caplog.at_level(logging.ERROR, logger=driver_module.logger.name):
        run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "failed:stream_gap"]
    assert "10.1.2.3" not in repr(run.stream.failure) and "refused" not in repr(timed)
    assert any("bridge failed" in record.getMessage() and record.exc_info for record in caplog.records)


def test_a_failure_in_settling_does_not_lose_an_answer_that_was_already_streamed(caplog):
    async def scenario():
        run = Scenario(ai("Hello", position="last"), END_SENTINEL, settle_error=RuntimeError("state store down"), include_usage=True)
        timed = await run.run()
        return run, timed

    with caplog.at_level(logging.ERROR, logger=driver_module.logger.name):
        run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "stop", "done"]  # no usage chunk: nothing is known
    assert run.stream.settled == Settled()
    assert any("settle" in record.getMessage() and record.exc_info for record in caplog.records)


def test_a_failure_inside_the_loop_ends_the_response_with_an_internal_error_that_names_no_cause(caplog):
    async def scenario():
        run = Scenario(ai("Hello", position="last"), object(), END_SENTINEL)
        timed = await run.run()
        return run, timed

    with caplog.at_level(logging.ERROR, logger=driver_module.logger.name):
        run, timed = play(scenario())

    assert labels(timed) == ["role", "content:Hello", "failed:internal_error"]
    assert run.stream.ended == "failed" and run.stream.failure is INTERNAL_ERROR
    assert "attribute" not in repr(run.stream.failure)
    assert any("Streaming a run failed" in record.getMessage() and record.exc_info for record in caplog.records)  # the trace is in the server log, not in the response


def test_the_ending_is_known_before_the_last_step_is_handed_over():
    async def scenario():
        finished = Scenario(ai("Hello", position="last"), END_SENTINEL)
        failed = Scenario(ai("Hello"), StreamGap(requested_event_id=None, earliest_available_event_id=None, latest_available_event_id=None))
        seen = []
        for run in (finished, failed):
            async for step in run.stream.steps():
                if step is DONE or isinstance(step, Failed):
                    seen.append((run.stream.ended, run.stream.failure))
        return seen

    assert play(scenario()) == [("done", None), ("failed", STREAM_GAP)]


def test_the_response_stays_open_until_the_frame_that_ends_it():
    async def scenario():
        run = Scenario(ai("Hel"), 6, ai("lo", position="last"), END_SENTINEL, window=0)
        return [(label(step), run.stream.ended) async for step in run.stream.steps()]

    assert play(scenario()) == [("role", "open"), ("content:Hel", "open"), ("keepalive", "open"), ("content:lo", "open"), ("stop", "done"), ("done", "done")]


# ── disconnects ──────────────────────────────────────────────────────────────


def test_a_client_that_leaves_while_the_loop_waits_ends_it_at_the_next_wake_with_the_translator_open():
    async def scenario():
        run = Scenario(60, ai("Hello", position="last"), END_SENTINEL)
        run.loop.call_later(3, setattr, run.client, "gone", True)
        timed = await run.run()
        await asyncio.sleep(0)
        return run, timed

    run, timed = play(scenario())

    assert timed == [(0.0, "role"), (5.0, "keepalive")]  # the keepalive that woke the loop was already on its way
    assert run.stream.ended == "disconnected" and run.stream.failure is None and run.stream.settled is None
    assert not run.translator.closed and run.settle_calls == 0
    assert run.probe.cancelled  # the read in flight was abandoned


def test_a_client_that_left_before_the_first_wait_is_noticed_before_anything_is_read():
    async def scenario():
        run = Scenario(ai("Hello", position="last"), END_SENTINEL)
        run.client.gone = True
        timed = await run.run()
        return run, timed

    run, timed = play(scenario())

    assert labels(timed) == ["role"]
    assert run.stream.ended == "disconnected" and not run.probe.started


def test_a_client_that_leaves_after_a_chunk_is_noticed_before_the_next_frame_is_read():
    async def scenario():
        run = Scenario(ai("one", message_id="a", position="last"), ai("two", message_id="b", position="last"), END_SENTINEL, window=0)
        names = []
        async for step in run.stream.steps():
            names.append(label(step))
            if names[-1] == "content:one":
                run.client.gone = True
        return run, names

    run, names = play(scenario())

    assert names == ["role", "content:one"]
    assert run.stream.ended == "disconnected" and run.probe.resumed == 0  # the loop never came back for the second frame


def test_a_consumer_that_closes_the_generator_leaves_the_ending_open_and_the_read_cancelled():
    async def scenario():
        run = Scenario(60, ai("Hello", position="last"), END_SENTINEL)
        steps = run.stream.steps()
        await anext(steps)
        reader = asyncio.ensure_future(anext(steps))
        await asyncio.sleep(1)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await asyncio.sleep(0)
        return run

    run = play(scenario())

    assert run.stream.ended == "open"
    assert run.probe.cancelled


FALLBACK_FRAME = StreamEvent(id="2", event="messages", data=[{**ai("The model provider failed.").data[0], "additional_kwargs": {"deerflow_error_fallback": True}}, {"langgraph_node": "model"}])


@pytest.mark.parametrize("frame", [ERROR_FRAME, FALLBACK_FRAME], ids=["error event", "error message of the model"])
def test_a_consumer_that_closes_the_generator_while_a_failed_run_ends_leaves_a_failed_ending(frame):
    async def scenario():
        run = Scenario(frame, 60, END_SENTINEL)
        steps = run.stream.steps()
        await anext(steps)
        reader = asyncio.ensure_future(anext(steps))
        await asyncio.sleep(1)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await asyncio.sleep(0)
        return run

    run = play(scenario())

    # The router cancels a run only for a response that ended open; this one had already failed.
    assert run.stream.ended == "failed" and run.stream.failure is RUN_ERROR


def test_closing_the_generator_between_steps_cancels_nothing_that_is_not_there():
    async def scenario():
        run = Scenario(ai("Hello", position="last"), END_SENTINEL)
        steps = run.stream.steps()
        await anext(steps)
        await steps.aclose()
        return run

    run = play(scenario())

    assert run.stream.ended == "open" and not run.probe.cancelled
