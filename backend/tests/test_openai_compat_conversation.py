"""Conversation identity: which stored thread a resent history continues, and what a new thread is seeded with."""

from __future__ import annotations

import pytest

from app.gateway.openai_compat.conversation import (
    Candidate,
    Exchange,
    Match,
    Turn,
    best_match,
    check_ingest,
    exchanges,
    imported_turns,
    norm,
    parse_turns,
    seed_messages,
    seed_title,
    text_hash,
)
from app.gateway.openai_compat.errors import OpenAIError
from app.gateway.openai_compat.messages import ChatMessage


def _user(text: str) -> ChatMessage:
    return ChatMessage("user", text)


def _assistant(text: str) -> ChatMessage:
    return ChatMessage("assistant", text)


def _turn(user: str, answer: str | None = None, *, t: float = 1.0) -> Turn:
    return Turn(u=text_hash(user), a=text_hash(answer) if answer is not None else None, t=t)


def _candidate(thread_id: str, *turns: Turn) -> Candidate:
    return Candidate(thread_id=thread_id, turns=turns)


def _history(*pairs: tuple[str, str | None]) -> tuple[Exchange, ...]:
    return tuple(Exchange(user=user, assistant=assistant) for user, assistant in pairs)


# ── hashing ──────────────────────────────────────────────────────────────────


def test_normalization_collapses_every_run_of_whitespace_and_trims_the_ends():
    assert norm("  hello \t\n world  again  ") == "hello world again"
    assert norm("") == ""
    assert norm(" \n ") == ""


def test_the_hash_is_the_first_128_bits_of_the_sha256_of_the_normalized_text():
    assert text_hash("hello world") == "b94d27b9934d3e08a52e52d7da7dabfa"
    assert text_hash("  hello \n world ") == "b94d27b9934d3e08a52e52d7da7dabfa"
    assert text_hash("") == "e3b0c44298fc1c149afbf4c8996fb924"


def test_different_text_has_a_different_hash():
    assert text_hash("Hello world") != text_hash("hello world")
    assert text_hash("hello world!") != text_hash("hello world")


def test_the_hash_of_non_latin_text_is_taken_over_utf8():
    assert text_hash("モデル") == text_hash("モデル ")
    assert len(text_hash("モデル")) == 32


# ── exchanges ────────────────────────────────────────────────────────────────


def test_each_user_message_gets_the_assistant_message_that_follows_it():
    history = exchanges([_user("one"), _assistant("two"), _user("three"), _assistant("four"), _user("five")])

    assert history == (Exchange("one", "two"), Exchange("three", "four"), Exchange("five", None))


def test_consecutive_assistant_messages_are_joined_with_a_blank_line():
    assert exchanges([_user("one"), _assistant("a"), _assistant("b"), _user("two")]) == (Exchange("one", "a\n\nb"), Exchange("two", None))


def test_consecutive_user_messages_are_exchanges_with_no_answer():
    assert exchanges([_user("one"), _user("two"), _assistant("answer"), _user("three")]) == (Exchange("one", None), Exchange("two", "answer"), Exchange("three", None))


def test_an_assistant_message_with_no_text_is_skipped():
    assert exchanges([_user("one"), _assistant(""), _assistant("  \n"), _user("two")]) == (Exchange("one", None), Exchange("two", None))
    assert exchanges([_user("one"), _assistant("a"), _assistant(" "), _assistant("b"), _user("two")])[0].assistant == "a\n\nb"


def test_text_is_kept_exactly_as_the_client_sent_it():
    assert exchanges([_user("  one \n two "), _assistant(" a  b "), _user("x")])[0] == Exchange("  one \n two ", " a  b ")


def test_a_history_that_starts_with_the_assistant_is_not_a_history():
    with pytest.raises(ValueError):
        exchanges([_assistant("Welcome!"), _user("hi")])


# ── the ledger entry ─────────────────────────────────────────────────────────


def test_a_turn_round_trips_through_json():
    turn = Turn(u="a" * 32, a="b" * 32, t=12.5)

    assert turn.to_json() == {"u": "a" * 32, "a": "b" * 32, "t": 12.5}
    assert Turn.from_json(turn.to_json()) == turn
    assert Turn.from_json(Turn(u="a" * 32, a=None, t=1.0).to_json()) == Turn(u="a" * 32, a=None, t=1.0)


def test_a_start_time_stored_as_an_integer_is_read_as_a_float():
    turn = Turn.from_json({"u": "x", "a": None, "t": 3})

    assert turn == Turn(u="x", a=None, t=3.0)
    assert isinstance(turn.t, float)


def test_a_missing_answer_key_means_no_answer_yet():
    assert Turn.from_json({"u": "x", "t": 1}) == Turn(u="x", a=None, t=1.0)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "u",
        ["u", None, 1],
        {},
        {"a": None, "t": 1},
        {"u": 5, "a": None, "t": 1},
        {"u": "x", "a": 5, "t": 1},
        {"u": "x", "a": None},
        {"u": "x", "a": None, "t": None},
        {"u": "x", "a": None, "t": "1"},
        {"u": "x", "a": None, "t": True},
        {"u": "x", "a": None, "t": 10**400},
        {"u": "x", "a": None, "t": float("nan")},
        {"u": "x", "a": None, "t": float("inf")},
    ],
)
def test_anything_that_is_not_a_well_formed_entry_is_not_a_turn(raw):
    assert Turn.from_json(raw) is None


def test_extra_keys_in_an_entry_are_ignored():
    assert Turn.from_json({"u": "x", "a": None, "t": 1, "state": "running", "run_id": "r"}) == Turn(u="x", a=None, t=1.0)


def test_a_missing_or_empty_ledger_is_empty():
    assert parse_turns(None) == ()
    assert parse_turns([]) == ()


def test_a_ledger_is_read_entry_by_entry():
    assert parse_turns([{"u": "x", "a": None, "t": 1}, {"u": "y", "a": "z", "t": 2}]) == (Turn("x", None, 1.0), Turn("y", "z", 2.0))


@pytest.mark.parametrize("raw", ["x", {"u": "x"}, 5, [{"u": "x", "a": None, "t": 1}, "broken"], [None]])
def test_a_ledger_with_anything_malformed_is_not_a_ledger(raw):
    assert parse_turns(raw) is None


# ── best_match ───────────────────────────────────────────────────────────────


def test_there_is_nothing_to_match_without_candidates():
    assert best_match(_history(("one", None)), []) is None


def test_a_history_that_is_the_stored_conversation_plus_one_message_continues_it():
    candidate = _candidate("t1", _turn("one", "a1"), _turn("two", "a2"))
    history = _history(("one", "a1"), ("two", "a2"), ("three", None))

    match = best_match(history, [candidate])

    assert match == Match(candidate, 2)
    assert match.continues(len(history))


@pytest.mark.parametrize(
    ("stored", "matched", "length", "expected"),
    [
        (2, 2, 3, True),
        (1, 1, 2, True),
        (2, 2, 2, False),  # the same last message again
        (2, 1, 2, False),  # the last message edited
        (2, 2, 4, False),  # a gap
        (3, 3, 3, False),
        (3, 2, 4, False),  # an older message edited
        (2, 1, 3, False),
        (2, 0, 3, False),
    ],
)
def test_a_history_continues_a_thread_only_when_it_adds_exactly_one_message_to_all_of_it(stored, matched, length, expected):
    candidate = Candidate("t", tuple(Turn(str(index), None, 1.0) for index in range(stored)))

    assert Match(candidate, matched).continues(length) is expected


def test_only_the_leading_run_of_equal_user_messages_counts():
    candidate = _candidate("t1", _turn("one"), _turn("edited"), _turn("three"))

    match = best_match(_history(("one", None), ("two", None), ("three", None), ("four", None)), [candidate])

    assert match.matched == 1
    assert not match.continues(4)


def test_a_ledger_longer_than_the_history_matches_at_most_the_whole_history():
    candidate = _candidate("t1", _turn("one"), _turn("two"), _turn("three"))

    match = best_match(_history(("one", None), ("two", None)), [candidate])

    assert match.matched == 2
    assert not match.continues(2)


def test_user_messages_match_after_whitespace_normalization():
    candidate = _candidate("t1", _turn("hello   world"))

    assert best_match(_history(("hello world", None), ("next", None)), [candidate]).continues(2)
    assert best_match(_history(("  hello \n world ", None), ("next", None)), [candidate]).continues(2)


def test_user_messages_differing_in_case_or_punctuation_do_not_match():
    candidate = _candidate("t1", _turn("hello world"))

    assert best_match(_history(("Hello world", None), ("next", None)), [candidate]).matched == 0
    assert best_match(_history(("hello world!", None), ("next", None)), [candidate]).matched == 0


def test_the_candidate_with_the_longest_run_wins():
    short = _candidate("short", _turn("one"), _turn("two"))
    long = _candidate("long", _turn("one"), _turn("two"), _turn("three"))
    history = _history(("one", None), ("two", None), ("three", None), ("four", None))

    assert best_match(history, [short, long]).candidate.thread_id == "long"
    assert best_match(history, [long, short]).candidate.thread_id == "long"


def test_a_thread_that_fits_the_history_beats_a_longer_one_that_diverged_earlier():
    fits = _candidate("fits", _turn("one"), _turn("two"))
    diverged = _candidate("diverged", _turn("one"), _turn("other"), _turn("x"), _turn("y"))
    history = _history(("one", None), ("two", None), ("three", None))

    match = best_match(history, [diverged, fits])

    assert match.candidate.thread_id == "fits"
    assert match.continues(3)


def test_a_tie_goes_to_the_candidate_whose_answers_match_more_of_what_the_client_holds():
    plain = _candidate("plain", _turn("one", "something else"), _turn("two", "also else"))
    matching = _candidate("matching", _turn("one", "a1"), _turn("two", "other"))
    history = _history(("one", "a1"), ("two", "a2"), ("three", None))

    assert best_match(history, [plain, matching]).candidate.thread_id == "matching"
    assert best_match(history, [matching, plain]).candidate.thread_id == "matching"


def test_answers_match_after_whitespace_normalization_too():
    reformatted = _candidate("reformatted", _turn("one", "a   1"), _turn("two", None))
    other = _candidate("other", _turn("one", "different"), _turn("two", None))
    history = _history(("one", " a 1\n"), ("two", None), ("three", None))

    assert best_match(history, [other, reformatted]).candidate.thread_id == "reformatted"


def test_a_stored_answer_missing_or_a_client_answer_missing_never_counts_as_a_match():
    unknown = _candidate("unknown", _turn("one"), _turn("two"))
    held = _candidate("held", _turn("one", "a1"), _turn("two", "a2"))
    history = _history(("one", None), ("two", None), ("three", None))

    # Nothing to compare on either side: the earlier candidate keeps the tie.
    assert best_match(history, [unknown, held]).candidate.thread_id == "unknown"
    assert best_match(history, [held, unknown]).candidate.thread_id == "held"


def test_the_run_of_user_messages_decides_before_the_answers_do():
    diverged = _candidate("diverged", _turn("one", "a1"), _turn("other", "a2"), _turn("x", "a3"))
    clean = _candidate("clean", _turn("one", "zzz"), _turn("two", "zzz"))
    history = _history(("one", "a1"), ("two", "a2"), ("three", "a3"), ("four", None))

    # diverged: a run of 1 with a matching answer. clean: a run of 2 and no matching answer.
    assert best_match(history, [diverged, clean]).candidate.thread_id == "clean"
    assert best_match(history, [clean, diverged]).candidate.thread_id == "clean"


def test_answers_beyond_the_matched_run_do_not_count():
    beyond = _candidate("beyond", _turn("one", "a1"), _turn("other", "a2"))
    plain = _candidate("plain", _turn("one", "a1"), _turn("else", "zzz"))
    history = _history(("one", "a1"), ("two", "a2"), ("three", None))

    # Both share a run of one message and one answer. The second answer of `beyond` equals the client's, but it comes after the user messages diverged.
    assert best_match(history, [plain, beyond]).candidate.thread_id == "plain"
    assert best_match(history, [beyond, plain]).candidate.thread_id == "beyond"


def test_a_tie_on_everything_goes_to_the_candidate_listed_first():
    first = _candidate("first", _turn("one", "a1"))
    second = _candidate("second", _turn("one", "a1"))
    history = _history(("one", "a1"), ("two", None))

    assert best_match(history, [first, second]).candidate.thread_id == "first"
    assert best_match(history, [second, first]).candidate.thread_id == "second"


def test_a_thread_with_an_empty_ledger_never_matches():
    empty = _candidate("empty")

    assert best_match(_history(("one", None)), [empty]) is None
    assert best_match(_history(("one", None), ("two", None)), [empty, _candidate("real", _turn("one"))]).candidate.thread_id == "real"


def test_a_candidate_that_shares_nothing_matches_with_a_run_of_zero_and_continues_nothing():
    match = best_match(_history(("one", None), ("two", None)), [_candidate("t1", _turn("other"))])

    assert match.matched == 0
    assert not match.continues(2)


# ── check_ingest ─────────────────────────────────────────────────────────────


def test_text_within_the_caps_is_accepted():
    check_ingest(["a", "b"], max_messages=2, max_chars=2)
    check_ingest([], max_messages=1, max_chars=1)


def test_one_message_too_many_is_a_400_that_says_what_to_do():
    with pytest.raises(OpenAIError) as caught:
        check_ingest(["a", "b", "c"], max_messages=2, max_chars=100)

    error = caught.value
    assert (error.status_code, error.code, error.param) == (400, "context_length_exceeded", "messages")
    assert "3 messages" in error.message and "at most 2" in error.message
    assert "Start a new chat" in error.message


def test_one_character_too_many_is_a_400_that_says_what_to_do():
    with pytest.raises(OpenAIError) as caught:
        check_ingest(["abc", "de"], max_messages=10, max_chars=4)

    error = caught.value
    assert (error.status_code, error.code, error.param) == (400, "context_length_exceeded", "messages")
    assert "5 characters" in error.message and "at most 4" in error.message
    assert "Start a new chat" in error.message


def test_the_characters_counted_are_the_ones_that_would_be_written_whitespace_included():
    with pytest.raises(OpenAIError):
        check_ingest(["a      b"], max_messages=5, max_chars=4)


def test_the_message_count_is_checked_before_the_characters():
    with pytest.raises(OpenAIError) as caught:
        check_ingest(["abcdef"] * 3, max_messages=2, max_chars=4)

    assert "3 messages" in caught.value.message and "characters" not in caught.value.message


# ── seeding ──────────────────────────────────────────────────────────────────


def test_seed_messages_are_the_text_of_each_exchange_in_order():
    earlier = _history(("one", "a1"), ("two", None), ("three", "a3"))

    assert seed_messages(earlier) == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "a3"},
    ]


def test_there_is_nothing_to_seed_for_a_first_message():
    assert seed_messages(()) == []
    assert imported_turns((), now=5.0) == []


def test_seeded_text_is_the_clients_text_not_its_normalization():
    assert seed_messages(_history(("  one \n two ", " a  b ")))[0]["content"] == "  one \n two "


def test_imported_turns_carry_the_hashes_and_one_start_time():
    earlier = _history(("one", "a1"), ("two", None), ("  three  ", "a  3"))

    assert imported_turns(earlier, now=7.5) == [
        Turn(u=text_hash("one"), a=text_hash("a1"), t=7.5),
        Turn(u=text_hash("two"), a=None, t=7.5),
        Turn(u=text_hash("three"), a=text_hash("a 3"), t=7.5),
    ]


# ── titles ───────────────────────────────────────────────────────────────────


def test_a_short_first_message_is_the_title_once_normalized():
    assert seed_title("  Plan   my\nweek ", max_chars=60) == "Plan my week"


def test_a_title_at_the_limit_is_kept_whole():
    assert seed_title("x" * 50, max_chars=60) == "x" * 50
    assert seed_title("x" * 8, max_chars=8) == "x" * 8


def test_a_longer_message_is_cut_to_fifty_characters_and_marked():
    title = seed_title("word " * 30, max_chars=60)

    assert title == ("word " * 30)[:50].rstrip() + "..."
    assert len(title) <= 60


def test_a_title_never_exceeds_the_configured_maximum():
    for max_chars in (10, 11, 20, 49, 50, 51, 60, 200):
        title = seed_title("abcdefghij " * 40, max_chars=max_chars)
        assert len(title) <= max_chars, (max_chars, title)
        assert title.endswith("...")


def test_the_ellipsis_never_follows_a_space():
    # The cut falls right after a space, in the fifty-character cut and in the shorter one that leaves room for the ellipsis.
    assert seed_title("a" * 49 + " " + "b" * 40, max_chars=60) == "a" * 49 + "..."
    assert seed_title("abcdef " + "b" * 40, max_chars=10) == "abcdef..."
