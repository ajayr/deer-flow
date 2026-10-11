"""The OpenAI-compatible API stays documented: config keys, the example block and the example response."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from app.gateway.openai_compat.completion import failure_status
from app.gateway.openai_compat.driver import FAILED_RUN_WAIT_SECONDS, KEEPALIVE_FRAME, KEEPALIVE_SECONDS
from app.gateway.openai_compat.errors import (
    EMPTY_RESPONSE,
    INTERNAL_ERROR,
    RUN_ERROR,
    RUN_INTERRUPTED,
    STREAM_GAP,
    context_length_exceeded,
    internal_error,
    invalid_message,
    invalid_request,
    model_not_found,
    unsupported_parameter,
    unsupported_value,
)
from app.gateway.openai_compat.ledger import CANDIDATE_LIMIT, CONV_KEY, TURNS_KEY
from app.gateway.openai_compat.messages import parse_messages
from app.gateway.openai_compat.modes import ADVERTISED_CONTEXT_LENGTH, MODE_FLAGS
from app.gateway.openai_compat.params import MAX_IGNORED_NAMES, MESSAGE_REJECTED, POLICY, validate_request
from app.gateway.openai_compat.schemas import ChatCompletion, ChatCompletionRequest, CompletionMessage, CompletionUsage, ErrorResponse, ModelList, ModelObject
from app.gateway.openai_compat.translator import SILENCE_NOTICE_SECONDS, Failed, to_sse
from app.gateway.routers.openai_compat import IGNORED_PARAMS_HEADER, _conversation_busy
from app.gateway.sse_headers import sse_response_headers
from deerflow.config.openai_compat_config import OPENAI_COMPAT_MODES, OpenAICompatConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

CHAT_HEADING = "### Create Chat Completion (OpenAI-style)"
CHAT_SCOPES = ("runs:create", "runs:read", "threads:read", "threads:write")


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _section(markdown: str, heading: str) -> str:
    """The text under ``heading`` (an exact heading line) up to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = markdown.splitlines()
    assert heading in lines, f"missing heading {heading!r}"
    body: list[str] = []
    in_fence = False
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(rf"#{{1,{level}}} ", line):
            break
        body.append(line)
    return "\n".join(body)


def _code_block(section: str, language: str) -> str:
    blocks = re.findall(rf"```{language}\n(.*?)\n```", section, flags=re.DOTALL)
    assert len(blocks) == 1, f"expected one {language} block, found {len(blocks)}"
    return blocks[0]


def test_configuration_guide_documents_every_key_and_mode():
    section = _section(_read("backend/docs/CONFIGURATION.md"), "### OpenAI-compatible API")

    for name in (*OpenAICompatConfig.model_fields, *OPENAI_COMPAT_MODES):
        assert f"`{name}`" in section, f"CONFIGURATION.md does not document `{name}`"


def test_configuration_guide_block_matches_the_example_config():
    section = _section(_read("backend/docs/CONFIGURATION.md"), "### OpenAI-compatible API")

    documented = yaml.safe_load(_code_block(section, "yaml"))["openai_compat"]
    example = yaml.safe_load(_read("config.example.yaml"))["openai_compat"]

    assert documented == example


def test_api_reference_example_response_matches_the_schema():
    section = _section(_read("backend/docs/API.md"), "### List Models (OpenAI-style)")
    example = json.loads(_code_block(section, "json"))

    listing = ModelList.model_validate(example)

    assert listing.data
    for raw, entry in zip(example["data"], listing.data, strict=True):
        assert set(raw) <= set(ModelObject.model_fields), f"{entry.id} documents a field the response does not have"
        assert entry.context_length == ADVERTISED_CONTEXT_LENGTH
        assert entry.is_multimodal is False


def test_api_reference_documents_the_route_and_its_credential():
    api = _read("backend/docs/API.md")
    section = _section(api, "## OpenAI-compatible API")

    assert "GET /api/openai/v1/models" in section
    assert "`runs:create`" in section
    assert "`GET /api/openai/v1/models`" in _section(api, "### PAT Constraints")


def test_readmes_list_the_api():
    readme = _read("README.md").splitlines()

    assert "#### OpenAI-compatible API" in readme
    assert any(line.strip() == "- [OpenAI-compatible API](#openai-compatible-api)" for line in readme)
    assert "`GET /api/openai/v1/models`" in _read("backend/README.md")
    assert "`POST /api/openai/v1/chat/completions`" in _read("backend/README.md")


# ── chat completions ─────────────────────────────────────────────────────────


def _chat_section() -> str:
    return _section(_read("backend/docs/API.md"), CHAT_HEADING)


def _blocks(section: str, language: str) -> list[str]:
    """The bodies of the fenced blocks tagged ``language`` (``""`` for an untagged fence), in order."""
    blocks: list[str] = []
    lines: list[str] = []
    opened: str | None = None
    for line in section.splitlines():
        if line.startswith("```"):
            if opened is None:
                opened = line[3:].strip()
            else:
                if opened == language:
                    blocks.append("\n".join(lines))
                opened, lines = None, []
        elif opened is not None:
            lines.append(line)
    return blocks


def _table_rows(section: str, header: str) -> list[list[str]]:
    """The cells of every row under the table whose header line is ``header``."""
    lines = section.splitlines()
    rows: list[list[str]] = []
    for line in lines[lines.index(header) + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip()[1:-1].split("|")])
    return rows


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_api_reference_parameter_table_is_the_policy():
    labels = {"honored": "Honored", "ignored": "Ignored, reported", "rejected": "Rejected (400)"}

    rows = _table_rows(_chat_section(), "| Parameter | Handling | Notes |")

    assert rows == [[f"`{rule.name}`", labels[rule.tier], rule.note] for rule in POLICY]


def test_api_reference_error_table_matches_the_errors_the_api_raises():
    raised = [
        invalid_request("x"),
        unsupported_parameter("p", "x"),
        unsupported_value("p", "x"),
        invalid_message("x"),
        context_length_exceeded("x"),
        model_not_found("x"),
        _conversation_busy(),
        internal_error(),
    ]

    rows = _table_rows(_chat_section(), "| Status | `code` | `param` | When |")

    assert [row[:2] for row in rows] == [[str(error.status_code), f"`{error.code}`"] for error in raised]


def test_api_reference_documents_every_in_band_error_with_its_status():
    section = _chat_section()
    listed = section.split("The in-band codes are", 1)[1].split("With `stream: false`", 1)[0]
    statuses = section.split("ordinary error response:", 1)[1].split("\n\n", 1)[0]

    for error in (RUN_ERROR, EMPTY_RESPONSE, RUN_INTERRUPTED, STREAM_GAP, INTERNAL_ERROR):
        assert f"`{error.code}`" in listed
        before = statuses[: statuses.index(f"`{error.code}`")]
        assert int(re.findall(r"`(\d{3})`", before)[-1]) == failure_status(error)


def test_api_reference_request_example_is_a_request_the_api_accepts():
    api = _read("backend/docs/API.md")
    request = json.loads(_blocks(_section(api, CHAT_HEADING), "json")[0])
    listed = {entry["id"] for entry in json.loads(_code_block(_section(api, "### List Models (OpenAI-style)"), "json"))["data"]}

    ChatCompletionRequest.model_validate(request)
    params = validate_request(request)
    messages = parse_messages(request["messages"])

    assert request["model"] in listed
    assert params.stream is True and params.include_usage is True and not params.ignored
    assert [message.role for message in messages.messages] == ["user", "assistant", "user"]


def test_api_reference_response_example_matches_the_schema():
    blocks = _blocks(_chat_section(), "json")
    assert len(blocks) == 2
    request, whole = (json.loads(block) for block in blocks)

    ChatCompletion.model_validate(whole)

    assert set(whole) <= set(ChatCompletion.model_fields)
    assert set(whole["choices"][0]["message"]) <= set(CompletionMessage.model_fields)
    assert whole["model"] == request["model"]
    assert whole["id"].startswith("chatcmpl-")


def test_api_reference_stream_example_is_the_wire_format():
    stream, failure = _blocks(_chat_section(), "")
    head, body = stream.split("\n\n", 1)
    status_line, *header_lines = head.splitlines()
    headers = {name.lower(): value for name, value in (line.split(": ", 1) for line in header_lines)}
    frames = body.split("\n\n")

    assert status_line == "HTTP/1.1 200 OK"
    assert headers["content-type"].startswith("text/event-stream")
    for name, value in sse_response_headers().items():
        assert headers.get(name.lower(), value) == value
    assert IGNORED_PARAMS_HEADER.lower() in headers
    assert frames.pop() == "data: [DONE]"
    chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames]
    for frame, chunk in zip(frames, chunks, strict=True):
        assert frame == "data: " + json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        assert chunk["object"] == "chat.completion.chunk"
    assert len({chunk["id"] for chunk in chunks}) == 1 and chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    for chunk in chunks[1:-2]:
        assert list(chunk["choices"][0]["delta"]) in (["reasoning_content"], ["content"])
    assert (chunks[-2]["choices"][0]["delta"], chunks[-2]["choices"][0]["finish_reason"]) == ({}, "stop")
    assert chunks[-1]["choices"] == [] and CompletionUsage.model_validate(chunks[-1]["usage"])

    assert "[DONE]" not in failure
    assert failure + "\n\n" == to_sse(Failed(RUN_ERROR))
    assert ErrorResponse.model_validate_json(failure.removeprefix("data: ")).error.code == RUN_ERROR.code


def test_api_reference_quotes_the_limits_and_names_the_code_uses():
    section = _chat_section()
    flat = _flat(section)
    efforts = [MODE_FLAGS[mode].default_effort for mode in ("thinking", "pro", "ultra")]

    for text in (
        IGNORED_PARAMS_HEADER,
        f"at most {MAX_IGNORED_NAMES}",
        KEEPALIVE_FRAME.strip(),
        f"every {KEEPALIVE_SECONDS:g} seconds",
        f"or after {FAILED_RUN_WAIT_SECONDS:g} seconds",
        f"{CANDIDATE_LIMIT} most recently updated",
        f"`{CONV_KEY}`",
        f"`{TURNS_KEY}`",
    ):
        assert text in flat
    assert SILENCE_NOTICE_SECONDS == 120 and "Still working... (N min)" in flat and "after every two minutes" in flat
    assert f"`{efforts[0]}` for `thinking`, `{efforts[1]}` for `pro` and `{efforts[2]}` for `ultra`" in flat
    assert MODE_FLAGS["flash"].default_effort is None
    assert OpenAICompatConfig().on_disconnect == "cancel" and "`cancel` (the default)" in flat
    for key in re.findall(r"`openai_compat\.(\w+)`", section):
        assert key in OpenAICompatConfig.model_fields, f"the chat section mentions openai_compat.{key}, which is not a config key"


def test_api_reference_names_every_message_field_the_policy_refuses():
    section = _flat(_chat_section())

    assert MESSAGE_REJECTED
    for key in MESSAGE_REJECTED:
        assert f"`{key}`" in section


def test_chat_credentials_are_documented_wherever_the_models_credential_is():
    api = _read("backend/docs/API.md")
    places = {
        "API.md chat section": _section(api, CHAT_HEADING),
        "API.md credentials": _section(api, "### Enabling and credentials"),
        "CONFIGURATION.md": _section(_read("backend/docs/CONFIGURATION.md"), "### OpenAI-compatible API"),
        "README.md": _section(_read("README.md"), "#### OpenAI-compatible API"),
    }

    for scope in CHAT_SCOPES:
        for place, text in places.items():
            assert f"`{scope}`" in text, f"{place} does not name the {scope} scope"
    assert "`POST /api/openai/v1/chat/completions`" in _section(api, "### PAT Constraints")
