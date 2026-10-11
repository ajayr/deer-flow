"""The request-parameter policy of ``POST /chat/completions``: one table that drives validation, the API reference and the contract tests.

Pure: no FastAPI, no I/O. Upstream treats a successful HTTP status as "accepted semantically"
(issue #4417), so no parameter is dropped silently. The line between the tiers is whether
ignoring a parameter changes what the caller gets in a way they would rely on:

* **honored**: the router acts on it.
* **ignored**: accepted, not acted on, and named in the ``X-DeerFlow-Ignored-Params`` response
  header (the name only, never the value).
* **rejected**: a 400 that names the parameter, because answering anyway would mislead.

Honored and rejected rows decide what happens. Every other field is ignored and reported, so the
ignored rows document what clients commonly send, and the tests pin them to that behavior.
A value of ``null`` counts as absent, as it does for the OpenAI API.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.gateway.openai_compat.errors import OpenAIError, invalid_request, unsupported_parameter, unsupported_value

Tier = Literal["honored", "ignored", "rejected"]

#: The header names at most this many ignored parameters.
MAX_IGNORED_NAMES = 20

_MAX_NAME_CHARS = 64
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.\[\]-]")

_NO_TOOLS_HINT = "DeerFlow runs its own tools on the server; turn off client-side tools in your client and retry."


@dataclass(frozen=True, slots=True)
class ParamRule:
    """One row of the policy."""

    name: str
    tier: Tier
    #: What the API reference says about this parameter.
    note: str
    #: Rejected rows: True for the values that are harmless defaults and are accepted.
    accepts: Callable[[Any], bool] | None = None
    #: Rejected rows: ``unsupported_value`` when the parameter exists but this value does not, else ``unsupported_parameter``.
    code: Literal["unsupported_parameter", "unsupported_value"] = "unsupported_parameter"
    #: Rejected rows: what the caller should do instead.
    hint: str = ""

    def rejection(self, param: str | None = None) -> OpenAIError:
        """The 400 for a value this rule refuses; ``param`` overrides the name for indexed paths."""
        param = param or self.name
        if self.code == "unsupported_value":
            return unsupported_value(param, f"Value for parameter '{param}' is not supported. {self.hint}")
        return unsupported_parameter(param, f"Parameter '{param}' is not supported. {self.hint}")


def _empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def _empty_object(value: Any) -> bool:
    return isinstance(value, dict) and not value


def _never(value: Any) -> bool:
    return False


_SAMPLING = "The model profile sets sampling."
_LENGTH = "The agent decides its own length; a run ends at the configured recursion limit."
_ECHO = "Clients echo these back from earlier replies; ignored."

#: The policy, in the order the API reference lists it.
POLICY: tuple[ParamRule, ...] = (
    # -- honored ---------------------------------------------------------------------------------
    ParamRule("model", "honored", "Required. An id from `GET /models`: `<model>` or `<model>:<mode>`."),
    ParamRule("messages", "honored", "Required. The conversation as user and assistant text messages; see Messages."),
    ParamRule("stream", "honored", "`true` streams `chat.completion.chunk` events; `false` or absent returns one JSON body."),
    ParamRule("stream_options.include_usage", "honored", "`true` adds a final chunk with the run's token totals, unless the model provider reported none."),
    ParamRule("reasoning_effort", "honored", "Overrides the variant's default effort. A value the model does not accept is a 400; a variant without thinking, or a model without an effort control, ignores and reports it."),
    # -- ignored: accepted, not acted on, reported -----------------------------------------------
    ParamRule("temperature", "ignored", _SAMPLING),
    ParamRule("top_p", "ignored", _SAMPLING),
    ParamRule("presence_penalty", "ignored", _SAMPLING),
    ParamRule("frequency_penalty", "ignored", _SAMPLING),
    ParamRule("seed", "ignored", _SAMPLING),
    ParamRule("max_tokens", "ignored", _LENGTH),
    ParamRule("max_completion_tokens", "ignored", _LENGTH),
    ParamRule("stop", "ignored", _LENGTH),
    ParamRule("user", "ignored", "The access token is the identity."),
    ParamRule("metadata", "ignored", "The thread is the conversation; nothing is stored for the client."),
    ParamRule("store", "ignored", "The thread is the conversation; nothing is stored for the client."),
    ParamRule("service_tier", "ignored", "Not applicable: the operator's model configuration picks the provider and its tier."),
    ParamRule("parallel_tool_calls", "ignored", "Not applicable: the agent runs its own tools."),
    ParamRule("stream_options.*", "ignored", "Any `stream_options` key other than `include_usage`."),
    ParamRule("messages[].name", "ignored", "Ignored on every message."),
    ParamRule("messages[].reasoning_content", "ignored", _ECHO),
    ParamRule("messages[].reasoning", "ignored", _ECHO),
    ParamRule("messages[].reasoning_details", "ignored", _ECHO),
    ParamRule("messages[].refusal", "ignored", _ECHO),
    ParamRule("messages[].annotations", "ignored", _ECHO),
    ParamRule("messages[].content[].*", "ignored", "Any key of a text part other than `type` and `text`, such as `cache_control`."),
    ParamRule("*", "ignored", "Any other field, at the top level or on a message."),
    # -- rejected: answering anyway would mislead ------------------------------------------------
    ParamRule("tools", "rejected", "Only an empty list is accepted.", _empty_list, hint=_NO_TOOLS_HINT),
    ParamRule("functions", "rejected", "Only an empty list is accepted.", _empty_list, hint=_NO_TOOLS_HINT),
    ParamRule("tool_choice", "rejected", "Only `none` is accepted.", lambda value: value == "none", "unsupported_value", "Only 'none' is accepted, because DeerFlow runs its own tools on the server."),
    ParamRule("n", "rejected", "Only `1` is accepted.", lambda value: value == 1 and not isinstance(value, bool), "unsupported_value", "Only n=1 is supported."),
    ParamRule("response_format", "rejected", 'Only `{"type": "text"}` is accepted.', lambda value: isinstance(value, dict) and value.get("type") == "text", "unsupported_value", "Only plain text answers are supported."),
    ParamRule("logprobs", "rejected", "Only `false` is accepted.", lambda value: value is False, hint="Log probabilities are not available."),
    ParamRule("top_logprobs", "rejected", "Not accepted.", _never, hint="Log probabilities are not available."),
    ParamRule("logit_bias", "rejected", "Only an empty object is accepted.", _empty_object, hint="Logit bias is not available."),
    ParamRule("modalities", "rejected", 'Only `["text"]` is accepted.', lambda value: value == ["text"], "unsupported_value", "Only text output is supported."),
    ParamRule("audio", "rejected", "Not accepted.", _never, hint="Audio output is not supported."),
    ParamRule("prediction", "rejected", "Not accepted.", _never, hint="Predicted outputs are not supported."),
    ParamRule("web_search_options", "rejected", "Not accepted.", _never, hint="DeerFlow runs its own web search tools on the server."),
    ParamRule("messages[].tool_calls", "rejected", "Only an empty list is accepted.", _empty_list, hint=_NO_TOOLS_HINT),
    ParamRule("messages[].function_call", "rejected", "Not accepted.", _never, hint=_NO_TOOLS_HINT),
    ParamRule("messages[].tool_call_id", "rejected", "Not accepted.", _never, hint=_NO_TOOLS_HINT),
    ParamRule("messages[].audio", "rejected", "Not accepted.", _never, hint="Audio is not supported."),
)

_MESSAGE_PREFIX = "messages[]."

#: The rules about the body itself, as opposed to the messages in it.
_REQUEST_RULES: tuple[ParamRule, ...] = tuple(rule for rule in POLICY if not rule.name.startswith(_MESSAGE_PREFIX))
_REJECTED_FIELDS: tuple[ParamRule, ...] = tuple(rule for rule in _REQUEST_RULES if rule.tier == "rejected")

#: Top-level fields the router reads or refuses itself; any other field is ignored and reported.
HANDLED_FIELDS: frozenset[str] = frozenset(rule.name.split(".")[0] for rule in _REQUEST_RULES if rule.tier != "ignored")

#: The rows that refuse a value on a message, keyed by the message key they refuse.
MESSAGE_REJECTED: dict[str, ParamRule] = {rule.name.removeprefix(_MESSAGE_PREFIX): rule for rule in POLICY if rule.tier == "rejected" and rule.name.startswith(_MESSAGE_PREFIX)}


def safe_name(name: str) -> str:
    """A parameter name that is safe to put in a response header."""
    return _UNSAFE_NAME_CHARS.sub("_", name[:_MAX_NAME_CHARS])


def ignored_params_header(*groups: Iterable[str]) -> str | None:
    """The value of ``X-DeerFlow-Ignored-Params``: the names in the groups, once each, in order; ``None`` when there are none."""
    names: list[str] = []
    for group in groups:
        for name in group:
            if name not in names:
                names.append(name)
                if len(names) == MAX_IGNORED_NAMES:
                    return ", ".join(names)
    return ", ".join(names) or None


@dataclass(frozen=True, slots=True)
class RequestParams:
    """What the router reads from the body's parameters, apart from the messages."""

    model: str
    stream: bool
    include_usage: bool
    #: The value the caller sent, if any. Whether the model accepts it is decided with the model (``modes.decide_effort``).
    reasoning_effort: str | None
    #: Names of the top-level parameters that were accepted and ignored.
    ignored: tuple[str, ...]


def _flag(source: Mapping[str, Any], key: str, param: str) -> bool:
    value = source.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise invalid_request(f"'{param}' must be true or false.", param=param)
    return value


def _ignored_stream_options(options: Mapping[str, Any]) -> list[str]:
    return [f"stream_options.{safe_name(option)}" for option, value in options.items() if option != "include_usage" and value is not None]


def validate_request(body: Mapping[str, Any]) -> RequestParams:
    """Check the body's parameters against the policy.

    Raises ``OpenAIError`` (a 400) for a malformed honored parameter or a rejected one. The
    messages are checked by ``parse_messages``, the model by ``modes.resolve_model``.
    """
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise invalid_request("'model' is required and must be a string.", param="model")
    stream = _flag(body, "stream", "stream")
    options = body.get("stream_options")
    include_usage = False
    if options is not None:
        if not isinstance(options, Mapping):
            raise invalid_request("'stream_options' must be an object.", param="stream_options")
        include_usage = _flag(options, "include_usage", "stream_options.include_usage")
    effort = body.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise invalid_request("'reasoning_effort' must be a string.", param="reasoning_effort")

    for rule in _REJECTED_FIELDS:
        value = body.get(rule.name)
        if value is not None and not (rule.accepts and rule.accepts(value)):
            raise rule.rejection()

    ignored = [safe_name(key) for key, value in body.items() if value is not None and key not in HANDLED_FIELDS]
    if options is not None:
        ignored += _ignored_stream_options(options)
    return RequestParams(model=model, stream=stream, include_usage=include_usage, reasoning_effort=effort, ignored=tuple(dict.fromkeys(ignored)))
