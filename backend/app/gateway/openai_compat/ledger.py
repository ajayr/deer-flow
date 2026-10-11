"""The conversation ledger: which user messages a thread was sent, kept in the thread's own metadata.

Two metadata keys carry it. ``openai_conv`` is the hash of the thread's first user message, which is
what a client's resent history is looked up by. ``openai_turns`` lists one entry per user message the
thread was sent (see ``conversation.Turn``). Keeping the ledger in the thread's metadata ties it to the
thread: it is deleted with it, scoped to its owner, and visible next to the rest of its metadata.

Failures to write are logged and reported as ``None``/``False``, never raised. The ledger only improves
continuity: without it, the next request is treated as a new conversation and seeded again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from app.gateway.openai_compat.conversation import Candidate, Turn, parse_turns
from deerflow.runtime.keyed_lock import AsyncKeyedLockTable

if TYPE_CHECKING:
    from deerflow.persistence.thread_meta.base import ThreadMetaStore

logger = logging.getLogger(__name__)

#: Metadata key holding the hash of a thread's first user message.
CONV_KEY = "openai_conv"
#: Metadata key holding the thread's ledger.
TURNS_KEY = "openai_turns"
#: How many threads that share a first message are considered, most recently updated first.
CANDIDATE_LIMIT = 50

# A ledger is rewritten whole, so two writers on the same thread must not interleave their read and write.
# The lock holds only within one process and one event loop: writers in other processes or on other loops are not serialized.
_ledger_locks = AsyncKeyedLockTable[str]()


def _stored_turns(record: Mapping[str, Any] | None) -> tuple[Turn, ...] | None:
    """The ledger in a thread record: empty when it has none, ``None`` for no record or an unreadable ledger."""
    if record is None:
        return None
    metadata = record.get("metadata")
    return parse_turns(metadata.get(TURNS_KEY) if isinstance(metadata, Mapping) else None)


async def find_candidates(store: ThreadMetaStore, *, user_id: str, conv: str) -> list[Candidate]:
    """The user's threads that started with the first message ``conv`` is the hash of, most recently updated first.

    A thread whose ledger cannot be read is left out: it cannot be continued, only replaced.
    """
    rows = await store.search(metadata={CONV_KEY: conv}, limit=CANDIDATE_LIMIT, user_id=user_id)
    # search() lists pinned threads first; the tie-break here wants recency alone.
    rows = sorted(rows, key=lambda row: (str(row.get("updated_at") or ""), str(row.get("thread_id") or "")), reverse=True)
    candidates = []
    for row in rows:
        turns = _stored_turns(row)
        if turns is not None:
            candidates.append(Candidate(thread_id=row["thread_id"], turns=turns))
    return candidates


async def append_turns(store: ThreadMetaStore, thread_id: str, turns: Sequence[Turn], *, user_id: str, conv: str | None = None) -> int | None:
    """Add ``turns`` to the end of the thread's ledger and return the index of the first one, or ``None`` if that failed.

    ``conv`` is stored with the thread's first turns: it is what makes the thread findable.
    A thread that is missing, owned by someone else, or whose ledger is unreadable is left untouched.
    """
    try:
        async with _ledger_locks.hold(thread_id):
            stored = _stored_turns(await store.get(thread_id, user_id=user_id))
            if stored is None:
                return None
            update: dict[str, Any] = {TURNS_KEY: [turn.to_json() for turn in (*stored, *turns)]}
            if conv is not None:
                update[CONV_KEY] = conv
            await store.update_metadata(thread_id, update, user_id=user_id)
            return len(stored)
    except Exception:
        logger.warning("Could not record the conversation ledger of thread %s", thread_id, exc_info=True)
        return None


async def record_answer(store: ThreadMetaStore, thread_id: str, index: int, *, user_hash: str, answer_hash: str, user_id: str) -> bool:
    """Record the hash of the answer to the turn at ``index``, provided that turn is still the one for ``user_hash``.

    Returns whether the answer was recorded. The thread keeps its place in recency order: this is not new activity.
    """
    try:
        async with _ledger_locks.hold(thread_id):
            stored = _stored_turns(await store.get(thread_id, user_id=user_id))
            if stored is None or not 0 <= index < len(stored) or stored[index].u != user_hash:
                return False
            answered = [*stored[:index], replace(stored[index], a=answer_hash), *stored[index + 1 :]]
            await store.update_metadata(thread_id, {TURNS_KEY: [turn.to_json() for turn in answered]}, touch=False, user_id=user_id)
            return True
    except Exception:
        logger.warning("Could not record the answer in the conversation ledger of thread %s", thread_id, exc_info=True)
        return False
