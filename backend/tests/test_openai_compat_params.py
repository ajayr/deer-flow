"""The request-parameter policy of ``POST /chat/completions``: what is honored, what is ignored and reported, and what is refused.

The expected sets below are the contract of the design spec (section 5.4): a row added to or
removed from the policy table must change them too, so the reference table in the API docs cannot
drift unnoticed.
"""

from __future__ import annotations

import pytest

from app.gateway.openai_compat.errors import OpenAIError
from app.gateway.openai_compat.params import (
    HANDLED_FIELDS,
    MAX_IGNORED_NAMES,
    MESSAGE_REJECTED,
    POLICY,
    ignored_params_header,
    safe_name,
    validate_request,
)

MESSAGES = [{"role": "user", "content": "hi"}]

HONORED = {"model", "messages", "stream", "stream_options.include_usage", "reasoning_effort"}
IGNORED = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "user",
    "metadata",
    "store",
    "service_tier",
    "parallel_tool_calls",
}
REJECTED = {"tools", "functions", "tool_choice", "n", "response_format", "logprobs", "top_logprobs", "logit_bias", "modalities", "audio", "prediction", "web_search_options"}
MESSAGE_IGNORED = {"name", "reasoning_content", "reasoning", "reasoning_details", "refusal", "annotations"}
MESSAGE_REFUSED = {"tool_calls", "function_call", "tool_call_id", "audio"}

#: A value each rejected parameter refuses.
REFUSED = {
    "tools": [{"type": "function", "function": {"name": "lookup"}}],
    "functions": [{"name": "lookup"}],
    "tool_choice": "auto",
    "n": 2,
    "response_format": {"type": "json_object"},
    "logprobs": True,
    "top_logprobs": 3,
    "logit_bias": {"50256": -100},
    "modalities": ["text", "audio"],
    "audio": {"voice": "alloy", "format": "mp3"},
    "prediction": {"type": "content", "content": "x"},
    "web_search_options": {},
}
#: The harmless value some rejected parameters still accept.
HARMLESS = {
    "tools": [],
    "functions": [],
    "tool_choice": "none",
    "n": 1,
    "response_format": {"type": "text"},
    "logprobs": False,
    "logit_bias": {},
    "modalities": ["text"],
}
IGNORED_BY_NAME = sorted(rule.name for rule in POLICY if rule.tier == "ignored" and rule.name.isidentifier())


def _body(**fields):
    return {"model": "minimax", "messages": MESSAGES, **fields}


def _refusal(**fields) -> OpenAIError:
    with pytest.raises(OpenAIError) as caught:
        validate_request(_body(**fields))
    return caught.value


def _rule(name: str):
    return next(rule for rule in POLICY if rule.name == name)


# ── the table ────────────────────────────────────────────────────────────────


def test_every_row_of_the_policy_is_named_once_and_explained():
    names = [rule.name for rule in POLICY]

    assert len(names) == len(set(names))
    assert all(rule.note.strip() for rule in POLICY)


def test_the_policy_names_exactly_the_parameters_the_spec_lists():
    by_tier = {tier: {rule.name for rule in POLICY if rule.tier == tier} for tier in ("honored", "ignored", "rejected")}

    assert by_tier["honored"] == HONORED
    assert by_tier["ignored"] == IGNORED | {"stream_options.*", "*"} | {f"messages[].{key}" for key in MESSAGE_IGNORED}
    assert by_tier["rejected"] == REJECTED | {f"messages[].{key}" for key in MESSAGE_REFUSED}


def test_the_message_rows_that_refuse_a_value_are_the_ones_the_spec_lists():
    assert set(MESSAGE_REJECTED) == MESSAGE_REFUSED


def test_the_fields_the_router_handles_itself_are_the_honored_and_rejected_ones():
    assert HANDLED_FIELDS == HONORED - {"stream_options.include_usage"} | {"stream_options"} | REJECTED


def test_every_rejected_row_says_what_to_do_instead():
    assert all(rule.hint.strip() for rule in POLICY if rule.tier == "rejected")


def test_the_samples_in_this_file_cover_every_rejected_row():
    assert set(REFUSED) == REJECTED


# ── honored parameters ───────────────────────────────────────────────────────


def test_a_minimal_body_is_accepted_and_reports_nothing():
    params = validate_request(_body())

    assert (params.model, params.stream, params.include_usage, params.reasoning_effort, params.ignored) == ("minimax", False, False, None, ())


@pytest.mark.parametrize("model", [None, "", "   ", 3, ["minimax"]])
def test_the_model_is_a_required_non_empty_string(model):
    error = _refusal(model=model)

    assert (error.status_code, error.code, error.param) == (400, "invalid_request", "model")


def test_a_body_without_a_model_is_refused():
    with pytest.raises(OpenAIError) as caught:
        validate_request({"messages": MESSAGES})

    assert caught.value.param == "model"


def test_stream_and_include_usage_are_read():
    params = validate_request(_body(stream=True, stream_options={"include_usage": True}))

    assert params.stream is True and params.include_usage is True


@pytest.mark.parametrize(("fields", "param"), [({"stream": "true"}, "stream"), ({"stream": 1}, "stream"), ({"stream_options": "usage"}, "stream_options"), ({"stream_options": {"include_usage": "yes"}}, "stream_options.include_usage")])
def test_a_flag_of_the_wrong_type_is_refused(fields, param):
    error = _refusal(**fields)

    assert (error.status_code, error.code, error.param) == (400, "invalid_request", param)


def test_other_stream_options_are_ignored_and_reported():
    params = validate_request(_body(stream_options={"include_usage": True, "continuous_usage_stats": True, "empty": None}))

    assert params.include_usage is True
    assert params.ignored == ("stream_options.continuous_usage_stats",)


def test_a_stream_option_name_is_made_safe_for_a_header_too():
    assert validate_request(_body(stream_options={"x y\r\nInjected: 1": 1})).ignored == ("stream_options.x_y__Injected__1",)


def test_the_reasoning_effort_is_handed_on_as_sent_and_not_reported():
    params = validate_request(_body(reasoning_effort="high"))

    assert params.reasoning_effort == "high"
    assert params.ignored == ()


def test_a_reasoning_effort_that_is_not_a_string_is_refused():
    error = _refusal(reasoning_effort=5)

    assert (error.code, error.param) == ("invalid_request", "reasoning_effort")


# ── ignored parameters ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", IGNORED_BY_NAME)
def test_an_ignored_parameter_is_accepted_and_reported_by_name(name):
    assert validate_request(_body(**{name: 1})).ignored == (name,)


def test_the_policy_has_an_ignored_row_for_each_parameter_the_spec_lists():
    assert set(IGNORED_BY_NAME) == IGNORED


def test_an_unknown_field_is_ignored_and_reported():
    assert validate_request(_body(frobnicate=True)).ignored == ("frobnicate",)


def test_reported_names_are_made_safe_for_a_header():
    assert validate_request(_body(**{"x y\r\nInjected: 1": 1})).ignored == ("x_y__Injected__1",)


def test_names_are_reported_once_in_the_order_they_were_sent():
    params = validate_request(_body(top_p=1, seed=2, temperature=0.1, **{"top_p!": 1, "top_p?": 1}))

    assert params.ignored == ("top_p", "seed", "temperature", "top_p_")


def test_a_value_is_never_part_of_what_is_reported():
    params = validate_request(_body(temperature=0.123456, user="alice@example.com"))

    assert "0.123456" not in "".join(params.ignored) and "alice" not in "".join(params.ignored)


# ── rejected parameters ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_rejected_parameter_is_a_400_that_names_it(name):
    error = _refusal(**{name: REFUSED[name]})

    assert (error.status_code, error.code, error.param) == (400, _rule(name).code, name)
    assert f"'{name}'" in error.message


def test_the_code_tells_a_missing_feature_from_a_value_that_is_not_supported():
    assert _refusal(tools=REFUSED["tools"]).code == "unsupported_parameter"
    assert _refusal(logprobs=True).code == "unsupported_parameter"
    for name in ("tool_choice", "n", "response_format", "modalities"):
        assert _refusal(**{name: REFUSED[name]}).code == "unsupported_value"


def test_a_refused_tool_says_what_to_do_instead():
    assert "turn off client-side tools" in _refusal(tools=REFUSED["tools"]).message


@pytest.mark.parametrize("name", sorted(HARMLESS))
def test_the_harmless_value_of_a_rejected_parameter_is_accepted_and_not_reported(name):
    assert validate_request(_body(**{name: HARMLESS[name]})).ignored == ()


def test_a_rejected_parameter_is_refused_before_anything_else_is_reported():
    assert _refusal(temperature=1, tools=REFUSED["tools"]).param == "tools"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("tools", "lookup"),
        ("functions", {}),
        ("tool_choice", {"type": "function", "function": {"name": "lookup"}}),
        ("n", "1"),
        ("n", True),
        ("response_format", "text"),
        ("logprobs", 0),
        ("logit_bias", []),
        ("modalities", "text"),
    ],
)
def test_a_harmless_value_of_the_wrong_shape_is_refused_not_crashed_on(name, value):
    assert _refusal(**{name: value}).param == name


@pytest.mark.parametrize("key", ["messages[]", "messages[].tool_calls", "stream_options.include_usage"])
def test_a_key_that_only_looks_like_a_path_is_an_unknown_field(key):
    assert validate_request(_body(**{key: [1]})).ignored == (key,)


# ── null and absent ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted({*REFUSED, *IGNORED, "stream", "stream_options", "reasoning_effort", "unheard_of"}))
def test_null_counts_as_absent(name):
    params = validate_request(_body(**{name: None}))

    assert (params.stream, params.include_usage, params.reasoning_effort, params.ignored) == (False, False, None, ())


# ── the header ───────────────────────────────────────────────────────────────


def test_the_header_lists_each_name_once_in_order():
    assert ignored_params_header(["a", "b"], ("b", "c"), []) == "a, b, c"


def test_there_is_no_header_without_names():
    assert ignored_params_header() is None
    assert ignored_params_header([], ()) is None


def test_the_header_names_at_most_twenty_parameters():
    names = [f"p{index}" for index in range(30)]

    assert MAX_IGNORED_NAMES == 20
    assert ignored_params_header(names).split(", ") == names[:20]


def test_the_header_stops_reading_names_once_it_has_twenty():
    def names():
        yield from (f"p{index}" for index in range(MAX_IGNORED_NAMES))
        raise AssertionError("read past the twentieth name")

    assert ignored_params_header(names()).split(", ") == [f"p{index}" for index in range(MAX_IGNORED_NAMES)]


def test_names_are_cut_to_header_safe_characters_and_a_sane_length():
    assert safe_name("messages[].name") == "messages[].name"
    assert safe_name("a b,c\n") == "a_b_c_"
    assert safe_name("ünï") == "_n_"
    assert len(safe_name("x" * 100)) == 64
