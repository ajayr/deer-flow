"""Direct tests for the shared ``messages-tuple`` stream helpers.

``tests/test_channels.py`` keeps the retained cases that call
``merge_stream_text`` and ``accumulate_stream_text`` directly, plus the one test
that proves hidden model context never reaches an IM channel through the real
manager. This module pins every helper on its own: the leaf helpers, which of
two competing keys wins, and the message-id / content-shape contract every
consumer relies on.
"""

from __future__ import annotations

import pytest

from deerflow.runtime.messages_stream import (
    accumulate_stream_text,
    extract_stream_message_id,
    extract_text_content,
    is_assistant_stream_type,
    merge_stream_text,
    stream_payload_type,
)

AI_CLASS_PATH = ["langchain", "schema", "messages", "AIMessageChunk"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        (["a", "b"], "ab"),
        ([{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}], "Hello"),
        ([{"content": "nested"}, "tail"], "nestedtail"),
        ([{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "shown"}], "shown"),
        ([{"text": 5}, 7, None], ""),
        ({"text": "from text"}, "from text"),
        ({"text": 3, "content": "from content"}, "from content"),
        ({"text": "wins", "content": "loses"}, "wins"),
        ("  padded \n", "  padded \n"),
        ([], ""),
        (None, ""),
        (42, ""),
    ],
)
def test_extract_text_content(content, expected):
    assert extract_text_content(content) == expected


@pytest.mark.parametrize(
    ("payload", "metadata", "expected"),
    [
        ({"id": "p-1"}, {"id": "m-1"}, "p-1"),
        ({"content": "x"}, {"message_id": "m-1"}, "m-1"),
        ({"message_id": "x-1"}, None, "x-1"),
        ({"type": "constructor", "kwargs": {"id": "k-1"}}, None, "k-1"),
        ({"id": ["langchain", "schema", "messages", "AIMessageChunk"], "kwargs": {"id": "k-2"}}, None, "k-2"),
        ({"id": "p-1", "message_id": "p-2"}, None, "p-1"),
        ({"id": "p-1", "kwargs": {"id": "k-1"}}, None, "p-1"),
        ({"id": ""}, {"id": 7}, None),
        ({}, {}, None),
        ("text", "meta", None),
        (None, None, None),
    ],
)
def test_extract_stream_message_id(payload, metadata, expected):
    assert extract_stream_message_id(payload, metadata) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "ai"}, "ai"),
        ({"type": "AIMessageChunk"}, "AIMessageChunk"),
        ({"type": "constructor", "kwargs": {"type": "human"}}, "human"),
        ({"type": "constructor", "id": ["langchain", "schema", "messages", "AIMessageChunk"]}, "AIMessageChunk"),
        ({"lc": 1, "id": ("langchain", "schema", "messages", "HumanMessage")}, "HumanMessage"),
        ({"type": "human", "kwargs": {"type": "ai"}}, "human"),
        ({"type": "constructor", "kwargs": {"type": "human"}, "id": AI_CLASS_PATH}, "human"),
        ({"type": "constructor"}, "constructor"),
        ({"content": "no type at all"}, ""),
        ({"type": 7}, ""),
    ],
)
def test_stream_payload_type(payload, expected):
    assert stream_payload_type(payload) == expected


@pytest.mark.parametrize("payload_type", ["ai", "AIMessageChunk", "AIMessage", "assistant", "Assistant", "  ai  "])
def test_is_assistant_stream_type_accepts_assistant_spellings(payload_type):
    assert is_assistant_stream_type(payload_type) is True


@pytest.mark.parametrize(
    "payload_type",
    ["human", "HumanMessage", "system", "SystemMessageChunk", "tool", "ToolMessageChunk", "function", "chain", "domain", "agent", "artifact", "a", "constructor", "", "   "],
)
def test_is_assistant_stream_type_rejects_everything_else(payload_type):
    """An allowlist: 'chain' and 'domain' contain 'ai' but are not assistant output, and neither is every word that merely starts with 'a'."""
    assert is_assistant_stream_type(payload_type) is False


@pytest.mark.parametrize(
    ("existing", "chunk", "expected"),
    [
        ("", "Hi", "Hi"),
        ("Hel", "lo", "Hello"),
        ("Hel", "Hello", "Hello"),
        ("ab", "ab", "abab"),
        ("谢", "谢", "谢谢"),
        ("hel", "l", "hell"),
        ("ab", "", "ab"),
        ("", "", ""),
    ],
)
def test_merge_stream_text(existing, chunk, expected):
    """A strictly longer chunk that starts with the buffer is a cumulative snapshot; anything else, even a chunk equal to the buffer, is a fresh delta."""
    assert merge_stream_text(existing, chunk) == expected


def _ai_event(content, *, message_id="ai-1", metadata=None):
    payload = {"type": "AIMessageChunk", "content": content}
    if message_id is not None:
        payload["id"] = message_id
    return [payload, metadata or {}]


def test_accumulate_stream_text_takes_the_message_id_from_metadata_when_the_payload_has_none():
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, None, _ai_event("Hi", message_id=None, metadata={"id": "meta-1"})) == ("Hi", "meta-1")
    assert buffers == {"meta-1": "Hi"}


def test_accumulate_stream_text_reuses_the_current_message_id_when_the_event_has_none():
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, "ai-7", _ai_event("x", message_id=None)) == ("x", "ai-7")
    assert buffers == {"ai-7": "x"}


def test_accumulate_stream_text_falls_back_to_the_default_buffer_key():
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, None, _ai_event("Hi", message_id=None)) == ("Hi", "__default__")
    assert buffers == {"__default__": "Hi"}


def test_accumulate_stream_text_reads_text_from_content_blocks():
    buffers: dict[str, str] = {}
    event = _ai_event([{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}])
    assert accumulate_stream_text(buffers, None, event) == ("Hello", "ai-1")


def test_accumulate_stream_text_contributes_nothing_for_non_text_blocks():
    buffers: dict[str, str] = {}
    event = _ai_event([{"type": "thinking", "thinking": "private"}])
    assert accumulate_stream_text(buffers, "ai-1", event) == (None, "ai-1")
    assert buffers == {}


def test_accumulate_stream_text_replaces_the_buffer_with_a_cumulative_snapshot():
    buffers: dict[str, str] = {}
    first_text, message_id = accumulate_stream_text(buffers, None, _ai_event("Hel"))
    second_text, _ = accumulate_stream_text(buffers, message_id, _ai_event("Hello"))
    assert (first_text, second_text) == ("Hel", "Hello")
    assert buffers == {"ai-1": "Hello"}


@pytest.mark.parametrize("event", [[], (), None, "Hello", 7, [None, {}], [{"type": "ai", "content": ""}, {}]])
def test_accumulate_stream_text_ignores_events_without_assistant_text(event):
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, "keep", event) == (None, "keep")
    assert buffers == {}


def test_accumulate_stream_text_accepts_a_tuple_event_and_a_bare_mapping():
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, None, ({"type": "ai", "id": "a-1", "content": "Hi"}, {})) == ("Hi", "a-1")
    assert accumulate_stream_text(buffers, "a-1", {"type": "ai", "id": "a-1", "content": "!"}) == ("Hi!", "a-1")


def test_accumulate_stream_text_accepts_an_event_that_has_no_metadata_element():
    assert accumulate_stream_text({}, None, [{"type": "ai", "id": "a-1", "content": "Hi"}]) == ("Hi", "a-1")


def test_accumulate_stream_text_writes_each_message_into_its_own_buffer():
    buffers: dict[str, str] = {}
    accumulate_stream_text(buffers, None, _ai_event("one", message_id="a"))
    assert accumulate_stream_text(buffers, "a", _ai_event("two", message_id="b")) == ("two", "b")
    assert buffers == {"a": "one", "b": "two"}


def test_accumulate_stream_text_reads_a_constructor_shaped_payload():
    """LangChain's ``to_json`` shape: the class name ends the ``id`` path, and the text and message id sit under ``kwargs``."""
    buffers: dict[str, str] = {}
    event = {"type": "constructor", "id": AI_CLASS_PATH, "kwargs": {"id": "k-1", "content": "Hi"}}
    assert accumulate_stream_text(buffers, None, event) == ("Hi", "k-1")
    assert buffers == {"k-1": "Hi"}


def test_accumulate_stream_text_prefers_top_level_content_over_kwargs_content():
    event = {"type": "ai", "id": "a-1", "content": "top", "kwargs": {"content": "nested"}}
    assert accumulate_stream_text({}, None, event) == ("top", "a-1")


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "human", "id": "m-1", "content": "<memory>Facts: ...</memory>"},
        {"type": "system", "id": "m-1", "content": "You are DeerFlow."},
        {"type": "tool", "id": "m-1", "content": "tool output"},
        {"type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "<memory>Facts: ...</memory>"}},
    ],
)
def test_accumulate_stream_text_never_buffers_hidden_or_tool_messages(payload):
    """The leak this module exists to prevent: hidden context and tool traffic are not assistant text, whatever shape carries them."""
    buffers: dict[str, str] = {}
    assert accumulate_stream_text(buffers, "keep", [payload, {}]) == (None, "keep")
    assert buffers == {}
