"""``config.example.yaml``, ``config_version`` and ``config-upgrade`` for the ``openai_compat`` section."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from support.shell import find_script_bash

from deerflow.config.app_config import AppConfig
from deerflow.config.openai_compat_config import OpenAICompatConfig

_SANDBOX = {"use": "deerflow.sandbox.local:LocalSandboxProvider"}
REPO_ROOT = Path(__file__).resolve().parents[2]
# The last config_version before the openai_compat section existed.
_VERSION_BEFORE_SECTION = 59
# The last config_version whose section had only enabled, default_mode and modes.
_VERSION_WITH_THREE_KEYS = 60
_DEFAULTS = OpenAICompatConfig().model_dump()
# Only the upgrade-script test shells out; it needs Git Bash on Windows.
SCRIPT_BASH = find_script_bash()


def _example() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.example.yaml").read_text(encoding="utf-8"))


def test_example_section_matches_the_model_defaults():
    """The documented block and the code defaults must not drift apart."""
    assert OpenAICompatConfig.model_validate(_example()["openai_compat"]) == OpenAICompatConfig()


def test_example_section_lists_every_key():
    """The defaults test cannot see a key missing from the example, because the model fills it in."""
    assert set(_example()["openai_compat"]) == set(OpenAICompatConfig.model_fields)


def test_example_config_version_covers_the_section():
    assert _example()["config_version"] > _VERSION_BEFORE_SECTION


def test_example_config_version_covers_the_run_and_ingest_keys():
    assert _example()["config_version"] > _VERSION_WITH_THREE_KEYS


def test_helm_chart_tracks_the_example_config_version():
    """scripts/check_config_version.sh only checks values.yaml in CI; the README example has no other guard."""
    version = _example()["config_version"]
    values = (REPO_ROOT / "deploy/helm/deer-flow/values.yaml").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "deploy/helm/deer-flow/README.md").read_text(encoding="utf-8")

    assert re.search(rf"^config: \|\n  config_version: {version}$", values, re.M)
    assert re.search(rf"^config: \|\n  config_version: {version}$", readme, re.M)


def test_a_config_from_before_the_section_is_reported_outdated(tmp_path, caplog):
    (tmp_path / "config.example.yaml").write_text((REPO_ROOT / "config.example.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"config_version": _VERSION_BEFORE_SECTION, "sandbox": _SANDBOX}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="deerflow.config.app_config"):
        AppConfig._check_config_version({"config_version": _VERSION_BEFORE_SECTION}, config_path)

    assert "outdated" in caplog.text
    assert f"(version {_VERSION_BEFORE_SECTION})" in caplog.text


@pytest.mark.skipif(SCRIPT_BASH is None, reason="repo shell-script tests need Git Bash on Windows")
@pytest.mark.parametrize(
    ("user_section", "expected"),
    [
        (None, _DEFAULTS),
        ({"enabled": True}, {**_DEFAULTS, "enabled": True}),
        ({"default_mode": "pro", "modes": ["pro"]}, {**_DEFAULTS, "default_mode": "pro", "modes": ["pro"]}),
    ],
    ids=["absent-section-gets-the-disabled-defaults", "enabled-is-kept", "narrowed-modes-are-kept"],
)
def test_config_upgrade_adds_the_section_and_keeps_user_values(tmp_path, user_section, expected):
    example_src = REPO_ROOT / "config.example.yaml"
    (tmp_path / "config.example.yaml").write_text(example_src.read_text(encoding="utf-8"), encoding="utf-8")
    user_config = {"config_version": _VERSION_BEFORE_SECTION, "sandbox": _SANDBOX, "agents_api": {"enabled": True}}
    if user_section is not None:
        user_config["openai_compat"] = user_section
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(user_config), encoding="utf-8")

    result = subprocess.run(
        [SCRIPT_BASH, str(REPO_ROOT / "scripts" / "config-upgrade.sh")],
        env={**os.environ, "DEER_FLOW_CONFIG_PATH": str(config_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    upgraded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert upgraded["config_version"] == _example()["config_version"]
    assert upgraded["openai_compat"] == expected
    assert upgraded["agents_api"] == {"enabled": True}
    # The upgraded file must load, i.e. the merged section passes OpenAICompatConfig validation.
    assert OpenAICompatConfig.model_validate(upgraded["openai_compat"]).model_dump() == expected


@pytest.mark.skipif(SCRIPT_BASH is None, reason="repo shell-script tests need Git Bash on Windows")
def test_config_upgrade_adds_the_run_and_ingest_keys_to_a_three_key_section_and_keeps_user_values(tmp_path):
    """A deployment that enabled the API when the section had three keys gets the other four with their defaults."""
    (tmp_path / "config.example.yaml").write_text((REPO_ROOT / "config.example.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    three_keys = {"enabled": True, "default_mode": "pro", "modes": ["thinking", "pro"]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"config_version": _VERSION_WITH_THREE_KEYS, "sandbox": _SANDBOX, "openai_compat": three_keys}), encoding="utf-8")

    result = subprocess.run(
        [SCRIPT_BASH, str(REPO_ROOT / "scripts" / "config-upgrade.sh")],
        env={**os.environ, "DEER_FLOW_CONFIG_PATH": str(config_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    upgraded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert upgraded["config_version"] == _example()["config_version"]
    assert upgraded["openai_compat"] == {**_DEFAULTS, **three_keys}
    assert OpenAICompatConfig.model_validate(upgraded["openai_compat"]).on_disconnect == "cancel"


def test_a_three_key_config_is_reported_outdated(tmp_path, caplog):
    (tmp_path / "config.example.yaml").write_text((REPO_ROOT / "config.example.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"config_version": _VERSION_WITH_THREE_KEYS, "sandbox": _SANDBOX}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="deerflow.config.app_config"):
        AppConfig._check_config_version({"config_version": _VERSION_WITH_THREE_KEYS}, config_path)

    assert "outdated" in caplog.text
    assert f"(version {_VERSION_WITH_THREE_KEYS})" in caplog.text
