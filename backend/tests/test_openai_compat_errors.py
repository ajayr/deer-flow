"""Errors of the OpenAI-compatible API: the envelope, the limits on what a message may say and the status and code each helper stands for."""

from __future__ import annotations

import dataclasses

import pytest

from app.gateway.openai_compat.errors import (
    EMPTY_RESPONSE,
    MAX_MESSAGE_CHARS,
    MAX_QUOTED_CHARS,
    RUN_ERROR,
    InBandError,
    OpenAIError,
    context_length_exceeded,
    internal_error,
    invalid_message,
    invalid_request,
    model_not_found,
    quote,
    unsupported_parameter,
    unsupported_value,
)

# ── the envelope ─────────────────────────────────────────────────────────────


def test_an_error_body_is_the_openai_envelope():
    error = OpenAIError(400, "invalid_request", "Bad.", param="model")

    assert error.body() == {"error": {"message": "Bad.", "type": "invalid_request_error", "param": "model", "code": "invalid_request"}}


def test_an_error_without_a_param_says_null():
    assert OpenAIError(400, "invalid_request", "Bad.").body()["error"]["param"] is None


def test_the_exception_text_is_the_client_message():
    error = OpenAIError(400, "invalid_request", "m" * 1000)

    assert str(error) == error.message


# ── the message limit ────────────────────────────────────────────────────────


def test_the_limit_leaves_room_for_the_text_conduit_adds_before_it_cuts_at_512():
    assert MAX_MESSAGE_CHARS <= 400 < 512


def test_a_message_at_the_limit_is_kept_whole():
    assert OpenAIError(400, "c", "m" * MAX_MESSAGE_CHARS).message == "m" * MAX_MESSAGE_CHARS


@pytest.mark.parametrize("length", [MAX_MESSAGE_CHARS + 1, MAX_MESSAGE_CHARS + 100, 5000])
def test_a_longer_message_is_cut_to_the_limit_and_says_so(length):
    text = ("abcdefghij" * 600)[:length]

    message = OpenAIError(400, "c", text).message

    assert len(message) == MAX_MESSAGE_CHARS
    assert message == text[: MAX_MESSAGE_CHARS - 3] + "..."


# ── the helpers ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "status", "code", "param", "kind"),
    [
        (invalid_request("m", param="stream"), 400, "invalid_request", "stream", "invalid_request_error"),
        (invalid_request("m"), 400, "invalid_request", None, "invalid_request_error"),
        (unsupported_parameter("tools", "m"), 400, "unsupported_parameter", "tools", "invalid_request_error"),
        (unsupported_value("n", "m"), 400, "unsupported_value", "n", "invalid_request_error"),
        (invalid_message("m"), 400, "invalid_message", "messages", "invalid_request_error"),
        (invalid_message("m", param="messages[2].role"), 400, "invalid_message", "messages[2].role", "invalid_request_error"),
        (context_length_exceeded("m"), 400, "context_length_exceeded", "messages", "invalid_request_error"),
        (model_not_found("m"), 404, "model_not_found", "model", "invalid_request_error"),
        (internal_error(), 500, "internal_error", None, "server_error"),
    ],
)
def test_each_helper_is_one_status_one_code_and_one_param(error, status, code, param, kind):
    assert (error.status_code, error.code, error.param, error.error_type) == (status, code, param, kind)
    assert error.body()["error"]["type"] == kind


def test_an_internal_error_says_nothing_about_its_cause_unless_told_to():
    assert internal_error().message == "The server could not process the request. Try again."
    assert internal_error("Try later.").message == "Try later."


# ── quote ────────────────────────────────────────────────────────────────────


def test_a_quoted_value_is_wrapped_in_single_quotes():
    assert quote("gpt-4o") == "'gpt-4o'"


def test_a_quoted_value_at_the_limit_is_kept_and_a_longer_one_is_cut():
    assert quote("x" * MAX_QUOTED_CHARS) == f"'{'x' * MAX_QUOTED_CHARS}'"
    assert quote("x" * (MAX_QUOTED_CHARS + 1)) == f"'{'x' * (MAX_QUOTED_CHARS - 3)}...'"


def test_a_value_that_is_not_text_is_quoted_by_its_repr():
    assert quote(5) == "'5'"
    assert quote(None) == "'None'"
    assert quote([1, "a"]) == "'[1, 'a']'"


def test_characters_that_do_not_print_become_spaces_so_a_value_cannot_break_a_line():
    assert quote("a\tb\r\nc\x00d") == "'a b  c d'"


def test_printable_text_in_other_scripts_is_kept():
    assert quote("モデル") == "'モデル'"


# ── in-band errors ───────────────────────────────────────────────────────────


def test_an_in_band_error_body_is_the_same_envelope_without_a_param():
    error = InBandError("run_error", "It failed.")

    assert error.body() == {"error": {"message": "It failed.", "type": "server_error", "param": None, "code": "run_error"}}


def test_an_in_band_error_cannot_be_changed_after_it_is_built():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RUN_ERROR.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(("error", "code"), [(RUN_ERROR, "run_error"), (EMPTY_RESPONSE, "empty_response")])
def test_the_fixed_in_band_errors_fit_a_client_and_leak_nothing(error, code):
    assert error.code == code
    assert len(error.message) <= MAX_MESSAGE_CHARS
    assert "http" not in error.message.lower() and "traceback" not in error.message.lower()
    assert "Send the message again" in error.message
