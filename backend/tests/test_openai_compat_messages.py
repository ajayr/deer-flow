"""The ``messages`` of a chat-completions request: what is accepted and what the router keeps of it."""

from __future__ import annotations

import json
import tracemalloc

import pytest

from app.gateway.openai_compat.errors import OpenAIError
from app.gateway.openai_compat.messages import ChatMessage, parse_messages


def _user(content="hi", **extra):
    return {"role": "user", "content": content, **extra}


def _assistant(content="hello", **extra):
    return {"role": "assistant", "content": content, **extra}


def _refusal(raw) -> OpenAIError:
    with pytest.raises(OpenAIError) as caught:
        parse_messages(raw)
    return caught.value


def _texts(raw) -> list[tuple[str, str]]:
    return [(message.role, message.text) for message in parse_messages(raw).messages]


# ── what is kept ─────────────────────────────────────────────────────────────


def test_a_conversation_is_kept_as_roles_and_text():
    parsed = parse_messages([_user("one"), _assistant("two"), _user("three")])

    assert parsed.messages == (ChatMessage("user", "one"), ChatMessage("assistant", "two"), ChatMessage("user", "three"))
    assert parsed.ignored == ()


def test_text_parts_are_joined_with_a_newline():
    raw = [_user([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}])]

    assert _texts(raw) == [("user", "first\nsecond")]


def test_a_message_without_content_has_no_text():
    assert _texts([_user("hi"), {"role": "assistant"}, _assistant(None), _user("again")]) == [("user", "hi"), ("assistant", ""), ("assistant", ""), ("user", "again")]


def test_text_is_kept_exactly_as_sent():
    assert _texts([_user("  spaced   out\n")]) == [("user", "  spaced   out\n")]


# ── the list itself ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [None, [], "hello", {"role": "user", "content": "hi"}, 3])
def test_messages_must_be_a_non_empty_list(raw):
    error = _refusal(raw)

    assert (error.status_code, error.code, error.param) == (400, "invalid_request", "messages")


def test_a_message_must_be_an_object():
    error = _refusal([_user(), "hello"])

    assert (error.code, error.param) == ("invalid_message", "messages[1]")


# ── roles ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["system", "developer"])
def test_a_system_prompt_is_refused_and_the_message_says_to_clear_it(role):
    error = _refusal([{"role": role, "content": "Be brief."}, _user()])

    assert (error.status_code, error.code, error.param) == (400, "invalid_message", "messages[0].role")
    assert error.message.startswith(f"{role.capitalize()} messages are not supported.")
    assert "Clear the system prompt in your client" in error.message


@pytest.mark.parametrize("role", ["tool", "function"])
def test_a_tool_message_is_refused(role):
    error = _refusal([_user(), {"role": role, "content": "42"}, _user("and?")])

    assert (error.code, error.param) == ("invalid_message", "messages[1].role")
    assert error.message.startswith(f"{role.capitalize()} messages are not supported.")
    assert "runs its own tools" in error.message


@pytest.mark.parametrize("role", [None, "", "moderator", 5, ["user"], {}])
def test_an_unknown_role_is_refused(role):
    error = _refusal([{"role": role, "content": "hi"}, _user()])

    assert (error.code, error.param) == ("invalid_message", "messages[0].role")


def test_the_conversation_must_start_with_a_user_message():
    error = _refusal([_assistant("Welcome!"), _user()])

    assert (error.code, error.param) == ("invalid_message", "messages[0].role")
    assert "start with a user message" in error.message


def test_the_last_message_must_be_a_user_message():
    error = _refusal([_user("one"), _assistant("two")])

    assert (error.code, error.param) == ("invalid_message", "messages[1]")
    assert "last message must be a user message" in error.message


@pytest.mark.parametrize("content", ["", "   \n\t", None, [{"type": "text", "text": "  "}], []])
def test_a_user_message_must_have_some_text(content):
    error = _refusal([_user("one"), _assistant("two"), _user(content)])

    assert (error.code, error.param) == ("invalid_message", "messages[2]")
    assert "needs some text" in error.message


def test_that_goes_for_an_earlier_user_message_too():
    error = _refusal([_user("one"), _assistant("two"), _user(""), _assistant("three"), _user("four")])

    assert (error.code, error.param) == ("invalid_message", "messages[2]")


def test_an_assistant_message_may_have_no_text():
    assert _texts([_user("one"), _assistant(""), _user("two")]) == [("user", "one"), ("assistant", ""), ("user", "two")]


# ── content ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("content", [5, {"type": "text", "text": "hi"}, True])
def test_content_must_be_a_string_or_a_list_of_parts(content):
    error = _refusal([_user(content)])

    assert (error.code, error.param) == ("invalid_message", "messages[0].content")


@pytest.mark.parametrize(("kind", "wording"), [("image_url", "Image input"), ("input_image", "Image input"), ("input_audio", "Audio input"), ("file", "File input")])
def test_a_part_that_is_not_text_is_refused_and_the_message_says_it_is_not_supported_yet(kind, wording):
    error = _refusal([_user([{"type": "text", "text": "look"}, {"type": kind, kind: {}}])])

    assert (error.code, error.param) == ("invalid_message", "messages[0].content[1].type")
    assert wording in error.message and "not supported yet" in error.message


def test_an_unknown_part_type_is_quoted_back_briefly():
    error = _refusal([_user([{"type": "x" * 100, "text": "hi"}])])

    assert error.param == "messages[0].content[0].type"
    assert "'" + "x" * 37 + "...'" in error.message
    assert len(error.message) < 200


@pytest.mark.parametrize("kind", [None, ["text"], {"type": "text"}])
def test_a_part_type_that_is_not_a_string_is_refused_not_crashed_on(kind):
    error = _refusal([_user([{"type": kind, "text": "hi"}])])

    assert (error.code, error.param) == ("invalid_message", "messages[0].content[0].type")
    assert error.message == "Each content part needs a string 'type'; only 'text' is supported."


def test_a_part_type_that_is_not_a_string_is_not_quoted_back():
    kind = [0] * 1_000_000
    tracemalloc.start()
    try:
        error = _refusal([_user([{"type": kind, "text": "hi"}])])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert error.param == "messages[0].content[0].type"
    assert peak < 1_000_000


@pytest.mark.parametrize(
    ("raw", "param"),
    [
        ([_user("a\ud800b")], "messages[0].content"),
        ([_user([{"type": "text", "text": "ok"}, {"type": "text", "text": "\udfff"}])], "messages[0].content"),
        ([_user(), _assistant("\ud83d"), _user()], "messages[1].content"),
    ],
)
def test_text_with_an_unpaired_surrogate_is_refused(raw, param):
    error = _refusal(raw)

    assert (error.code, error.param) == ("invalid_message", param)
    assert "unpaired surrogate" in error.message


@pytest.mark.parametrize("body", [b'[{"role": "user", "content": "a\xed\xa0\x80b"}]', '[{"role": "user", "content": "a\\ud800b"}]'])
def test_a_surrogate_that_json_decoding_lets_through_is_refused(body):
    assert _refusal(json.loads(body)).param == "messages[0].content"


def test_an_escaped_surrogate_pair_is_one_character_and_is_kept():
    assert _texts(json.loads('[{"role": "user", "content": "hi \\ud83d\\ude00"}]')) == [("user", "hi \U0001f600")]


def test_a_part_must_be_an_object_with_text():
    assert _refusal([_user(["hi"])]).param == "messages[0].content[0]"
    assert _refusal([_user([{"type": "text"}])]).param == "messages[0].content[0].text"
    assert _refusal([_user([{"type": "text", "text": 3}])]).param == "messages[0].content[0].text"


# ── fields that would change what the agent does ─────────────────────────────


@pytest.mark.parametrize(
    ("key", "value"),
    [("tool_calls", [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]), ("function_call", {"name": "f"}), ("tool_call_id", "call_1"), ("audio", {"id": "audio_1"})],
)
def test_a_field_that_would_change_what_the_agent_does_is_refused_with_its_position(key, value):
    error = _refusal([_user("one"), _assistant("two", **{key: value}), _user("three")])

    assert (error.status_code, error.code, error.param) == (400, "unsupported_parameter", f"messages[1].{key}")


def test_an_empty_list_of_tool_calls_is_harmless_and_not_reported():
    parsed = parse_messages([_user("one"), _assistant("two", tool_calls=[]), _user("three")])

    assert parsed.ignored == ()


@pytest.mark.parametrize("key", ["tool_calls", "function_call", "tool_call_id", "audio"])
def test_a_null_semantic_field_counts_as_absent(key):
    assert parse_messages([_user("one"), _assistant("two", **{key: None}), _user("three")]).ignored == ()


# ── fields that are ignored and reported ─────────────────────────────────────


def test_the_fields_clients_echo_back_are_ignored_and_reported_once():
    raw = [
        _user("one", name="alice"),
        _assistant("two", reasoning_content="thinking", reasoning="again", reasoning_details=[{"type": "x"}], refusal=None, annotations=[]),
        _user("three", name="alice"),
    ]

    parsed = parse_messages(raw)

    assert parsed.ignored == ("messages[].name", "messages[].reasoning_content", "messages[].reasoning", "messages[].reasoning_details", "messages[].annotations")
    assert [message.text for message in parsed.messages] == ["one", "two", "three"]


def test_an_unknown_message_key_is_ignored_and_reported_without_its_value():
    parsed = parse_messages([_user("one", id="msg_1", **{"odd key\r\n": 1}), _assistant("two"), _user("three")])

    assert parsed.ignored == ("messages[].id", "messages[].odd_key__")


def test_a_null_valued_key_is_not_reported():
    assert parse_messages([_user("one", name=None, extra=None)]).ignored == ()


def test_other_keys_of_a_text_part_are_ignored_and_reported_once():
    cached = {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    raw = [
        _user([cached, {"type": "text", "text": "again", "cache_control": {"type": "ephemeral"}, "citations": []}]),
        _assistant([{"type": "text", "text": "two", "annotations": []}]),
        _user([cached]),
    ]

    parsed = parse_messages(raw)

    assert parsed.ignored == ("messages[].content[].cache_control", "messages[].content[].citations", "messages[].content[].annotations")
    assert [message.text for message in parsed.messages] == ["hello\nagain", "two", "hello"]


def test_the_keys_of_a_message_are_reported_before_the_keys_of_its_parts():
    parsed = parse_messages([_user([{"type": "text", "text": "hi", "cache_control": {}}], name="alice")])

    assert parsed.ignored == ("messages[].name", "messages[].content[].cache_control")


def test_a_key_of_a_part_is_reported_without_its_value_and_made_safe_for_a_header():
    parsed = parse_messages([_user([{"type": "text", "text": "hi", "odd key\r\n": "a secret value"}])])

    assert parsed.ignored == ("messages[].content[].odd_key__",)


def test_a_null_valued_key_of_a_part_is_not_reported():
    assert parse_messages([_user([{"type": "text", "text": "hi", "cache_control": None}])]).ignored == ()


def test_the_type_and_the_text_of_a_part_are_not_reported():
    assert parse_messages([_user([{"type": "text", "text": "hi"}])]).ignored == ()
