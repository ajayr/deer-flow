"""The OpenAI-compatible API stays documented: config keys, the example block and the example response."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from app.gateway.openai_compat.modes import ADVERTISED_CONTEXT_LENGTH
from app.gateway.openai_compat.schemas import ModelList, ModelObject
from deerflow.config.openai_compat_config import OPENAI_COMPAT_MODES, OpenAICompatConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


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
