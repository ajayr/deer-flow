"""The conversation ledger kept in thread metadata, exercised against the in-memory thread store."""

from __future__ import annotations

import asyncio
import logging

import pytest
from langgraph.store.memory import InMemoryStore

from app.gateway.openai_compat.conversation import Turn, text_hash
from app.gateway.openai_compat.ledger import CANDIDATE_LIMIT, CONV_KEY, TURNS_KEY, append_turns, find_candidates, record_answer
from deerflow.persistence.thread_meta.memory import THREADS_NS, MemoryThreadMetaStore
from deerflow.runtime.user_context import AUTO

pytestmark = pytest.mark.anyio

LEDGER_LOGGER = "app.gateway.openai_compat.ledger"
USER = "user-a"
OTHER = "user-b"
CONV = text_hash("hello")


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _InterleavingStore(MemoryThreadMetaStore):
    """Hands control to other tasks between a read and the write that follows it."""

    async def get(self, thread_id, *, user_id=AUTO):
        record = await super().get(thread_id, user_id=user_id)
        await asyncio.sleep(0)
        return record


class _FailingStore(MemoryThreadMetaStore):
    def __init__(self, store, *, fail_on):
        super().__init__(store)
        self.fail_on = fail_on

    async def get(self, thread_id, *, user_id=AUTO):
        if self.fail_on == "get":
            raise RuntimeError("database is down")
        return await super().get(thread_id, user_id=user_id)

    async def update_metadata(self, thread_id, metadata, *, touch=True, user_id=AUTO):
        if self.fail_on == "update_metadata":
            raise RuntimeError("database is down")
        await super().update_metadata(thread_id, metadata, touch=touch, user_id=user_id)


class _GatedStore(MemoryThreadMetaStore):
    """Holds reads of ``t1`` until ``t2``'s append has landed, so a lock shared by both threads would time out."""

    def __init__(self, store):
        super().__init__(store)
        self.t2_written = asyncio.Event()

    async def get(self, thread_id, *, user_id=AUTO):
        if thread_id == "t1":
            await asyncio.wait_for(self.t2_written.wait(), 1)
        return await super().get(thread_id, user_id=user_id)

    async def update_metadata(self, thread_id, metadata, *, touch=True, user_id=AUTO):
        await super().update_metadata(thread_id, metadata, touch=touch, user_id=user_id)
        if thread_id == "t2":
            self.t2_written.set()


@pytest.fixture
def store():
    return MemoryThreadMetaStore(InMemoryStore())


@pytest.fixture
def ledger_log(caplog):
    """``take()`` returns the warnings the ledger logged since the last call."""
    caplog.set_level(logging.WARNING, logger=LEDGER_LOGGER)

    def take():
        records = [record for record in caplog.records if record.name == LEDGER_LOGGER]
        caplog.clear()
        return records

    return take


def _turn(user: str, answer: str | None = None, *, t: float = 1.0) -> Turn:
    return Turn(u=text_hash(user), a=text_hash(answer) if answer is not None else None, t=t)


def _stored(turns):
    """What thread metadata holds for ``turns``: entries as JSON, anything else (a broken ledger) as it is."""
    if isinstance(turns, list):
        return [turn.to_json() if isinstance(turn, Turn) else turn for turn in turns]
    return turns


async def _thread(store, thread_id, *, user=USER, conv=None, turns=None, **other):
    metadata = dict(other)
    if conv is not None:
        metadata[CONV_KEY] = conv
    if turns is not None:
        metadata[TURNS_KEY] = _stored(turns)
    await store.create(thread_id, user_id=user, metadata=metadata)


async def _record(store, thread_id, *, user=USER):
    return await store.get(thread_id, user_id=user)


async def _ledger(store, thread_id, *, user=USER):
    return (await _record(store, thread_id, user=user))["metadata"][TURNS_KEY]


# ── the keys ─────────────────────────────────────────────────────────────────


def test_the_ledger_is_stored_under_the_documented_keys():
    assert (CONV_KEY, TURNS_KEY) == ("openai_conv", "openai_turns")


# ── find_candidates ──────────────────────────────────────────────────────────


async def test_there_are_no_candidates_for_a_message_nobody_sent(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")])

    assert await find_candidates(store, user_id=USER, conv=text_hash("something else")) == []


async def test_a_candidate_carries_its_thread_id_and_ledger(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello", "hi", t=5.0), _turn("more")])

    [candidate] = await find_candidates(store, user_id=USER, conv=CONV)

    assert candidate.thread_id == "t1"
    assert candidate.turns == (_turn("hello", "hi", t=5.0), _turn("more"))


async def test_candidates_come_most_recently_updated_first(store):
    for thread_id in ("t1", "t2", "t3"):
        await _thread(store, thread_id, conv=CONV, turns=[_turn("hello")])
    await store.update_metadata("t1", {"touched": True}, user_id=USER)

    assert [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)] == ["t1", "t3", "t2"]


async def test_a_pinned_thread_does_not_come_before_a_newer_one(store):
    await _thread(store, "older", conv=CONV, turns=[_turn("hello")], deerflow_pinned=True)
    await _thread(store, "newer", conv=CONV, turns=[_turn("hello")])

    assert [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)] == ["newer", "older"]


async def test_only_the_users_own_threads_are_candidates(store):
    await _thread(store, "mine", conv=CONV, turns=[_turn("hello")])
    await _thread(store, "theirs", user=OTHER, conv=CONV, turns=[_turn("hello")])

    assert [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)] == ["mine"]
    assert [c.thread_id for c in await find_candidates(store, user_id=OTHER, conv=CONV)] == ["theirs"]


async def test_threads_that_are_not_conversations_of_this_api_are_not_candidates(store):
    await _thread(store, "plain")
    await _thread(store, "other", conv=text_hash("other"), turns=[_turn("other")])

    assert await find_candidates(store, user_id=USER, conv=CONV) == []


async def test_an_archived_thread_can_still_be_continued(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")], deerflow_archived=True)

    assert [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)] == ["t1"]


async def test_a_thread_with_a_conversation_hash_but_no_ledger_is_an_empty_candidate(store):
    await _thread(store, "t1", conv=CONV)

    [candidate] = await find_candidates(store, user_id=USER, conv=CONV)

    assert (candidate.thread_id, candidate.turns) == ("t1", ())


@pytest.mark.parametrize("ledger", ["oops", {"u": "x"}, 5, [{"u": "x", "a": None, "t": 1}, "broken"], [{"u": 7, "a": None, "t": 1}]])
async def test_a_thread_whose_ledger_cannot_be_read_is_not_a_candidate(store, ledger):
    await _thread(store, "bad", conv=CONV, turns=ledger)
    await _thread(store, "good", conv=CONV, turns=[_turn("hello")])

    assert [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)] == ["good"]


async def test_only_the_most_recent_threads_are_considered(store):
    count = CANDIDATE_LIMIT + 5
    for index in range(count):
        await _thread(store, f"t{index:03d}", conv=CONV, turns=[_turn("hello")])

    found = [c.thread_id for c in await find_candidates(store, user_id=USER, conv=CONV)]

    assert len(found) == CANDIDATE_LIMIT
    assert found == [f"t{index:03d}" for index in range(count - 1, count - 1 - CANDIDATE_LIMIT, -1)]


# ── append_turns ─────────────────────────────────────────────────────────────


async def test_the_first_turns_of_a_thread_are_written_with_the_conversation_hash(store):
    await _thread(store, "t1")

    index = await append_turns(store, "t1", [_turn("hello", t=3.0)], user_id=USER, conv=CONV)

    assert index == 0
    record = await _record(store, "t1")
    assert record["metadata"][CONV_KEY] == CONV
    assert record["metadata"][TURNS_KEY] == [_turn("hello", t=3.0).to_json()]


async def test_later_turns_are_added_after_the_ones_before_and_the_index_says_where(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello", "hi"), _turn("more")])

    index = await append_turns(store, "t1", [_turn("third"), _turn("fourth")], user_id=USER)

    assert index == 2
    assert await _ledger(store, "t1") == [turn.to_json() for turn in (_turn("hello", "hi"), _turn("more"), _turn("third"), _turn("fourth"))]


async def test_appending_without_a_conversation_hash_leaves_the_stored_one_alone(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")])

    await append_turns(store, "t1", [_turn("more")], user_id=USER)

    assert (await _record(store, "t1"))["metadata"][CONV_KEY] == CONV


async def test_the_rest_of_the_threads_metadata_survives(store):
    await _thread(store, "t1", deerflow_pinned=True, note="keep")

    await append_turns(store, "t1", [_turn("hello")], user_id=USER, conv=CONV)

    metadata = (await _record(store, "t1"))["metadata"]
    assert (metadata["deerflow_pinned"], metadata["note"]) == (True, "keep")


async def test_appending_counts_as_activity(store):
    await _thread(store, "t1")
    before = (await _record(store, "t1"))["updated_at"]

    await append_turns(store, "t1", [_turn("hello")], user_id=USER, conv=CONV)

    assert (await _record(store, "t1"))["updated_at"] > before


async def test_a_thread_that_does_not_exist_is_not_created_by_appending(store, ledger_log):
    assert await append_turns(store, "nope", [_turn("hello")], user_id=USER, conv=CONV) is None
    assert await store.get("nope", user_id=USER) is None
    assert ledger_log() == []  # nothing went wrong: there was just nothing to append to


async def test_another_users_thread_is_left_alone(store, ledger_log):
    await _thread(store, "t1", user=OTHER, conv=CONV, turns=[_turn("hello")])

    assert await append_turns(store, "t1", [_turn("more")], user_id=USER) is None
    assert await _ledger(store, "t1", user=OTHER) == [_turn("hello").to_json()]
    assert ledger_log() == []


async def test_an_unreadable_ledger_is_not_overwritten(store, ledger_log):
    await _thread(store, "t1", conv=CONV, turns="oops")

    assert await append_turns(store, "t1", [_turn("more")], user_id=USER) is None
    assert await _ledger(store, "t1") == "oops"
    assert ledger_log() == []


async def test_a_thread_record_without_usable_metadata_starts_a_ledger(store):
    await _thread(store, "t1")
    record = dict((await store._store.aget(THREADS_NS, "t1")).value)
    record["metadata"] = None
    await store._store.aput(THREADS_NS, "t1", record)

    assert await append_turns(store, "t1", [_turn("hello")], user_id=USER, conv=CONV) == 0
    assert await _ledger(store, "t1") == [_turn("hello").to_json()]


async def test_a_failing_store_is_logged_and_reported_not_raised(store, ledger_log):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")])

    for fail_on in ("get", "update_metadata"):
        result = await append_turns(_FailingStore(store._store, fail_on=fail_on), "t1", [_turn("more")], user_id=USER)

        assert result is None
        [record] = ledger_log()
        assert "t1" in record.getMessage()
        assert record.exc_info is not None
    assert await _ledger(store, "t1") == [_turn("hello").to_json()]


async def test_two_appends_to_one_thread_do_not_overwrite_each_other(store):
    interleaving = _InterleavingStore(store._store)
    await _thread(interleaving, "t1", conv=CONV, turns=[])

    first, second = await asyncio.gather(
        append_turns(interleaving, "t1", [_turn("a")], user_id=USER),
        append_turns(interleaving, "t1", [_turn("b")], user_id=USER),
    )

    assert sorted([first, second]) == [0, 1]
    assert sorted(entry["u"] for entry in await _ledger(interleaving, "t1")) == sorted([_turn("a").u, _turn("b").u])


# ── record_answer ────────────────────────────────────────────────────────────


async def test_an_answer_is_recorded_on_the_turn_it_belongs_to(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello", "hi", t=1.0), _turn("more", t=2.0), _turn("last", t=3.0)])

    recorded = await record_answer(store, "t1", 1, user_hash=text_hash("more"), answer_hash=text_hash("the answer"), user_id=USER)

    assert recorded is True
    assert await _ledger(store, "t1") == [_turn("hello", "hi", t=1.0).to_json(), _turn("more", "the answer", t=2.0).to_json(), _turn("last", t=3.0).to_json()]


async def test_recording_an_answer_leaves_the_rest_of_the_metadata_and_the_recency_order_alone(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")], note="keep")
    before = await _record(store, "t1")

    await record_answer(store, "t1", 0, user_hash=text_hash("hello"), answer_hash=text_hash("hi"), user_id=USER)

    after = await _record(store, "t1")
    assert after["updated_at"] == before["updated_at"]
    assert (after["metadata"][CONV_KEY], after["metadata"]["note"]) == (CONV, "keep")


async def test_an_answer_is_not_recorded_on_a_turn_for_a_different_message(store, ledger_log):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello"), _turn("more")])

    assert await record_answer(store, "t1", 1, user_hash=text_hash("hello"), answer_hash=text_hash("x"), user_id=USER) is False
    assert await _ledger(store, "t1") == [_turn("hello").to_json(), _turn("more").to_json()]
    assert ledger_log() == []


@pytest.mark.parametrize("index", [-1, 2, 3, 99])
async def test_an_answer_is_not_recorded_outside_the_ledger(store, ledger_log, index):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello"), _turn("more")])

    assert await record_answer(store, "t1", index, user_hash=text_hash("more"), answer_hash=text_hash("x"), user_id=USER) is False
    assert await _ledger(store, "t1") == [_turn("hello").to_json(), _turn("more").to_json()]
    assert ledger_log() == []


async def test_the_last_turn_can_be_answered(store):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello"), _turn("more")])

    assert await record_answer(store, "t1", 1, user_hash=text_hash("more"), answer_hash=text_hash("x"), user_id=USER) is True


async def test_no_answer_is_recorded_for_a_missing_or_foreign_thread(store, ledger_log):
    await _thread(store, "t1", user=OTHER, conv=CONV, turns=[_turn("hello")])

    assert await record_answer(store, "nope", 0, user_hash=text_hash("hello"), answer_hash="x", user_id=USER) is False
    assert await record_answer(store, "t1", 0, user_hash=text_hash("hello"), answer_hash="x", user_id=USER) is False
    assert await _ledger(store, "t1", user=OTHER) == [_turn("hello").to_json()]
    assert ledger_log() == []


async def test_no_answer_is_recorded_in_an_unreadable_ledger(store, ledger_log):
    await _thread(store, "t1", conv=CONV, turns=["oops"])

    assert await record_answer(store, "t1", 0, user_hash=text_hash("hello"), answer_hash="x", user_id=USER) is False
    assert await _ledger(store, "t1") == ["oops"]
    assert ledger_log() == []


async def test_a_failing_store_while_recording_an_answer_is_logged_and_reported_not_raised(store, ledger_log):
    await _thread(store, "t1", conv=CONV, turns=[_turn("hello")])

    for fail_on in ("get", "update_metadata"):
        result = await record_answer(_FailingStore(store._store, fail_on=fail_on), "t1", 0, user_hash=text_hash("hello"), answer_hash="x", user_id=USER)

        assert result is False
        [record] = ledger_log()
        assert "t1" in record.getMessage()
        assert record.exc_info is not None
    assert await _ledger(store, "t1") == [_turn("hello").to_json()]


async def test_recording_an_answer_and_appending_the_next_turn_do_not_overwrite_each_other(store):
    interleaving = _InterleavingStore(store._store)
    await _thread(interleaving, "t1", conv=CONV, turns=[_turn("hello")])

    recorded, index = await asyncio.gather(
        record_answer(interleaving, "t1", 0, user_hash=text_hash("hello"), answer_hash=text_hash("hi"), user_id=USER),
        append_turns(interleaving, "t1", [_turn("more")], user_id=USER),
    )

    assert (recorded, index) == (True, 1)
    assert await _ledger(interleaving, "t1") == [_turn("hello", "hi").to_json(), _turn("more").to_json()]


async def test_different_threads_do_not_wait_for_each_other(store):
    gated = _GatedStore(store._store)
    await _thread(gated, "t1", conv=CONV, turns=[])
    await _thread(gated, "t2", conv=CONV, turns=[])

    indexes = await asyncio.gather(
        append_turns(gated, "t1", [_turn("a")], user_id=USER),
        append_turns(gated, "t2", [_turn("b")], user_id=USER),
    )

    assert indexes == [0, 0]
