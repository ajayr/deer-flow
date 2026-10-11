"""The non-streaming answer to a chat completion: its body, and the HTTP status a failed run answers with."""

from __future__ import annotations

import pytest

from app.gateway.openai_compat import errors
from app.gateway.openai_compat.completion import completion_body, failure_error, failure_status
from app.gateway.openai_compat.errors import EMPTY_RESPONSE, INTERNAL_ERROR, RUN_ERROR, RUN_INTERRUPTED, STREAM_GAP, InBandError, OpenAIError
from app.gateway.openai_compat.schemas import ChatCompletion
from app.gateway.openai_compat.translator import Usage


def _body(**overrides):
    values = {"completion_id": "chatcmpl-1", "model": "deerflow:pro", "created": 1790000000, "content": "Hello", "reasoning": "", "usage": Usage(prompt_tokens=3, completion_tokens=4, total_tokens=9)}
    return completion_body(**{**values, **overrides})


# ── the body ─────────────────────────────────────────────────────────────────


def test_the_body_is_one_assistant_message_that_stopped_with_the_runs_usage():
    assert _body() == {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": "deerflow:pro",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 9},
    }


def test_the_reasoning_travels_as_reasoning_content_and_only_when_there_is_some():
    assert _body(reasoning="Tool web_search: done\n")["choices"][0]["message"] == {"role": "assistant", "content": "Hello", "reasoning_content": "Tool web_search: done\n"}
    assert "reasoning_content" not in _body(reasoning="")["choices"][0]["message"]


def test_the_usage_is_zero_when_the_provider_reported_none():
    assert _body(usage=None)["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_the_content_is_kept_exactly_as_the_translator_sent_it():
    assert _body(content="  Hello \n\n world ")["choices"][0]["message"]["content"] == "  Hello \n\n world "


@pytest.mark.parametrize("reasoning", ["", "Tool web_search: done\n"])
def test_the_body_is_what_the_documented_schema_describes(reasoning):
    """``ChatCompletion`` is what the OpenAPI schema and the API reference promise; the body may not say anything else."""
    body = _body(reasoning=reasoning)

    assert ChatCompletion.model_validate(body).model_dump(exclude_none=True) == body


# ── failures ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "status"),
    [(RUN_ERROR, 502), (EMPTY_RESPONSE, 502), (STREAM_GAP, 502), (RUN_INTERRUPTED, 409), (INTERNAL_ERROR, 500)],
)
def test_a_failed_run_answers_with_the_status_for_its_code(error, status):
    assert failure_status(error) == status


def test_every_in_band_error_the_package_defines_has_a_status_of_its_own():
    defined = {value.code for value in vars(errors).values() if isinstance(value, InBandError)}

    assert defined == {"run_error", "empty_response", "run_interrupted", "stream_gap", "internal_error"}
    assert {code: failure_status(InBandError(code, "m")) for code in defined} == {"run_error": 502, "empty_response": 502, "stream_gap": 502, "run_interrupted": 409, "internal_error": 500}


def test_a_code_without_a_status_is_a_server_error():
    assert failure_status(InBandError("brand_new", "m")) == 500


@pytest.mark.parametrize("error", [RUN_ERROR, EMPTY_RESPONSE, RUN_INTERRUPTED, STREAM_GAP, INTERNAL_ERROR])
def test_the_error_of_a_failed_run_is_the_in_band_error_in_the_same_envelope(error):
    raised = failure_error(error)

    assert isinstance(raised, OpenAIError)
    assert raised.status_code == failure_status(error)
    assert raised.body() == error.body()
