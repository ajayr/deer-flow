"""Conversation identity for clients that resend their whole history on every turn.

Pure: no FastAPI, no I/O. A client such as Conduit sends no conversation id, and the Gateway
appends to a thread, so resending the history would duplicate it. This module decides which
stored thread a history continues. Each router-created thread carries a ledger: the hash of
every user message it was sent and, once known, the hash of the answer. A history continues a
thread when it is that thread's stored user messages plus exactly one new one; any other
history starts a new thread that is seeded with the client's earlier messages.

User messages must match exactly after whitespace normalization. Assistant hashes only break
ties, because clients reformat assistant text and exact matching would fork a thread every turn.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.gateway.openai_compat.errors import context_length_exceeded
from app.gateway.openai_compat.messages import ChatMessage

#: Hashes are stored as this many hex characters of a SHA-256, which is 128 bits.
_HASH_CHARS = 32

# Mirrors the local fallback title of ``TitleMiddleware``, so a seeded thread looks like an untitled one that was titled automatically.
_TITLE_CHARS = 50
_ELLIPSIS = "..."


def norm(text: str) -> str:
    """Whitespace-normalized text: runs of whitespace become one space, and the ends are trimmed."""
    return " ".join(text.split())


def text_hash(text: str) -> str:
    """Stable identity of a message's text, insensitive to whitespace."""
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()[:_HASH_CHARS]


@dataclass(frozen=True, slots=True)
class Exchange:
    """A user message and what the assistant said after it, as the client's history has it."""

    user: str
    assistant: str | None = None


def exchanges(messages: Sequence[ChatMessage]) -> tuple[Exchange, ...]:
    """Group a history into one exchange per user message.

    Assistant messages belong to the user message before them: several in a row are joined with
    a blank line, and one with no text is skipped. The history must start with a user message,
    which ``parse_messages`` guarantees.
    """
    grouped: list[tuple[str, list[str]]] = []
    for message in messages:
        if message.role == "user":
            grouped.append((message.text, []))
        elif not grouped:
            raise ValueError("a history starts with a user message")
        elif message.text.strip():
            grouped[-1][1].append(message.text)
    return tuple(Exchange(user=user, assistant="\n\n".join(replies) or None) for user, replies in grouped)


@dataclass(frozen=True, slots=True)
class Turn:
    """One entry of a thread's ledger."""

    #: Hash of the user message that started the turn.
    u: str
    #: Hash of the answer the client received; ``None`` until the stream ended normally.
    a: str | None
    #: When the turn started, in seconds since the epoch.
    t: float

    def to_json(self) -> dict[str, Any]:
        return {"u": self.u, "a": self.a, "t": self.t}

    @classmethod
    def from_json(cls, raw: object) -> Turn | None:
        """The entry, or ``None`` for anything that is not a well-formed one (thread metadata is editable)."""
        if not isinstance(raw, Mapping):
            return None
        u, a, t = raw.get("u"), raw.get("a"), raw.get("t")
        if not isinstance(u, str) or not (a is None or isinstance(a, str)) or isinstance(t, bool) or not isinstance(t, int | float):
            return None
        try:
            start = float(t)
        except OverflowError:  # an integer too large for a float
            return None
        return cls(u=u, a=a, t=start) if math.isfinite(start) else None


def parse_turns(raw: object) -> tuple[Turn, ...] | None:
    """The ledger stored in a thread's metadata: empty when there is none, ``None`` when it is not a list of well-formed entries."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return None
    turns: list[Turn] = []
    for item in raw:
        turn = Turn.from_json(item)
        if turn is None:
            return None
        turns.append(turn)
    return tuple(turns)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A stored thread whose first user message equals the history's, with its ledger."""

    thread_id: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class Match:
    """The candidate a history lines up with best."""

    candidate: Candidate
    #: How many of the history's first user messages equal the stored ones, in order.
    matched: int

    def continues(self, length: int) -> bool:
        """True when a history of ``length`` user messages is the stored conversation plus exactly one new message."""
        stored = len(self.candidate.turns)
        return length == stored + 1 and self.matched == stored


def best_match(history: Sequence[Exchange], candidates: Sequence[Candidate]) -> Match | None:
    """The candidate that shares the longest run of user messages with the history, or ``None``.

    Ties go to the candidate whose stored answers match more of the answers the client holds,
    then to the one that comes first in ``candidates``, which the caller orders by recency.
    A thread with an empty ledger never matches: it holds nothing the client could have seen.
    """
    users = [text_hash(exchange.user) for exchange in history]
    replies = [text_hash(exchange.assistant) if exchange.assistant else None for exchange in history]
    best: Match | None = None
    best_score = (0, 0)
    for candidate in candidates:
        if not candidate.turns:
            continue
        matched = 0
        while matched < min(len(candidate.turns), len(users)) and candidate.turns[matched].u == users[matched]:
            matched += 1
        answers = sum(1 for index in range(matched) if candidate.turns[index].a is not None and candidate.turns[index].a == replies[index])
        if best is None or (matched, answers) > best_score:
            best, best_score = Match(candidate, matched), (matched, answers)
    return best


def check_ingest(texts: Sequence[str], *, max_messages: int, max_chars: int) -> None:
    """Refuse text the router would have to write into a thread when there is too much of it.

    Raises a 400 ``context_length_exceeded``. Seeding never truncates, so the client learns
    that it must start a new chat instead of silently losing part of its history.
    """
    if len(texts) > max_messages:
        raise context_length_exceeded(f"This conversation has {len(texts)} messages, and at most {max_messages} can be imported into a new thread. Start a new chat.")
    chars = sum(len(text) for text in texts)
    if chars > max_chars:
        raise context_length_exceeded(f"This conversation has {chars} characters, and at most {max_chars} can be imported into a new thread. Start a new chat.")


def seed_messages(earlier: Sequence[Exchange]) -> list[dict[str, str]]:
    """The text-only human and ai messages that stand in for the exchanges before the new message."""
    seeded: list[dict[str, str]] = []
    for exchange in earlier:
        seeded.append({"role": "user", "content": exchange.user})
        if exchange.assistant:
            seeded.append({"role": "assistant", "content": exchange.assistant})
    return seeded


def imported_turns(earlier: Sequence[Exchange], *, now: float) -> list[Turn]:
    """Ledger entries for exchanges that were seeded rather than run."""
    return [Turn(u=text_hash(exchange.user), a=text_hash(exchange.assistant) if exchange.assistant else None, t=now) for exchange in earlier]


def seed_title(first_user_text: str, *, max_chars: int) -> str:
    """The title of a seeded thread: its first user message, normalized and cut to fit."""
    text = norm(first_user_text)
    limit = min(max_chars, _TITLE_CHARS)
    if len(text) <= limit:
        return text
    return text[: min(limit, max_chars - len(_ELLIPSIS))].rstrip() + _ELLIPSIS
