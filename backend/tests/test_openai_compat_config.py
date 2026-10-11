"""``openai_compat`` config section: defaults, validation and AppConfig wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deerflow.config.app_config import AppConfig
from deerflow.config.openai_compat_config import OPENAI_COMPAT_MODES, OpenAICompatConfig
from deerflow.config.sandbox_config import SandboxConfig

_SANDBOX = {"use": "deerflow.sandbox.local:LocalSandboxProvider"}


def test_defaults_are_off_and_offer_every_mode():
    config = OpenAICompatConfig()

    assert config.enabled is False
    assert config.default_mode == "thinking"
    assert config.modes == ["flash", "thinking", "pro", "ultra"]
    assert tuple(config.modes) == OPENAI_COMPAT_MODES


def test_the_defaults_of_the_run_and_ingest_keys():
    config = OpenAICompatConfig()

    assert config.on_disconnect == "cancel"
    assert config.narration_window_chars == 200
    assert config.max_ingest_messages == 400
    assert config.max_ingest_chars == 500000


def test_a_subset_of_modes_in_any_order_is_accepted():
    config = OpenAICompatConfig(modes=["pro", "flash"], default_mode="flash")

    assert config.modes == ["pro", "flash"]
    assert config.default_mode == "flash"


@pytest.mark.parametrize(
    "modes",
    [[], ["flash", "turbo"], ["flash", "Pro"], ["flash", "flash"], "flash", None],
    ids=["empty", "unknown", "wrong-case", "duplicate", "not-a-list", "null"],
)
def test_invalid_modes_are_rejected(modes):
    with pytest.raises(ValidationError):
        OpenAICompatConfig(modes=modes, default_mode="flash")


def test_duplicate_modes_are_reported_by_name():
    with pytest.raises(ValidationError, match="modes must not contain duplicates"):
        OpenAICompatConfig(modes=["flash", "flash"], default_mode="flash")


def test_default_mode_rejects_an_unknown_value():
    with pytest.raises(ValidationError):
        OpenAICompatConfig(default_mode="turbo")


def test_default_mode_must_be_one_of_modes():
    with pytest.raises(ValidationError, match="default_mode 'pro' must be one of modes"):
        OpenAICompatConfig(default_mode="pro", modes=["flash", "thinking"])


def test_narrowing_modes_without_a_matching_default_mode_fails_loudly():
    """The built-in default_mode is "thinking": narrowing modes to ["flash"] alone must not boot a section that offers nothing usable."""
    with pytest.raises(ValidationError, match="default_mode 'thinking' must be one of modes"):
        OpenAICompatConfig(modes=["flash"])


@pytest.mark.parametrize("value", ["cancel", "continue"])
def test_on_disconnect_accepts_cancel_and_continue(value):
    assert OpenAICompatConfig(on_disconnect=value).on_disconnect == value


@pytest.mark.parametrize("value", ["Cancel", "interrupt", "", None, True, 1])
def test_on_disconnect_rejects_anything_else(value):
    with pytest.raises(ValidationError):
        OpenAICompatConfig(on_disconnect=value)


@pytest.mark.parametrize(("key", "value"), [("narration_window_chars", 0), ("narration_window_chars", 1), ("max_ingest_messages", 1), ("max_ingest_chars", 1)])
def test_the_smallest_allowed_value_of_each_limit_is_accepted(key, value):
    assert getattr(OpenAICompatConfig(**{key: value}), key) == value


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("narration_window_chars", -1),
        ("max_ingest_messages", 0),
        ("max_ingest_messages", -5),
        ("max_ingest_chars", 0),
        ("max_ingest_chars", -1),
    ],
)
def test_a_limit_below_its_minimum_is_rejected(key, value):
    with pytest.raises(ValidationError):
        OpenAICompatConfig(**{key: value})


@pytest.mark.parametrize("key", ["narration_window_chars", "max_ingest_messages", "max_ingest_chars"])
@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_is_not_a_limit(key, value):
    """Pydantic would turn true into 1 and false into 0, which silently changes what the API does."""
    with pytest.raises(ValidationError, match=f"{key} must be an integer, not a boolean"):
        OpenAICompatConfig(**{key: value})


@pytest.mark.parametrize("key", ["narration_window_chars", "max_ingest_messages", "max_ingest_chars"])
@pytest.mark.parametrize("value", ["many", 1.5, None, [1]])
def test_a_limit_must_be_an_integer(key, value):
    with pytest.raises(ValidationError):
        OpenAICompatConfig(**{key: value})


def _app_config(**sections) -> AppConfig:
    return AppConfig.model_validate({"sandbox": _SANDBOX, **sections})


def test_app_config_has_a_disabled_section_by_default():
    config = AppConfig(sandbox=SandboxConfig(use=_SANDBOX["use"]))

    assert config.openai_compat == OpenAICompatConfig()
    assert config.openai_compat.enabled is False


def test_app_config_reads_the_section_from_a_mapping():
    config = _app_config(openai_compat={"enabled": True, "default_mode": "pro", "modes": ["thinking", "pro"]})

    assert config.openai_compat.enabled is True
    assert config.openai_compat.default_mode == "pro"
    assert config.openai_compat.modes == ["thinking", "pro"]


def test_app_config_reads_the_run_and_ingest_keys():
    config = _app_config(openai_compat={"on_disconnect": "continue", "narration_window_chars": 0, "max_ingest_messages": 10, "max_ingest_chars": 2000})

    assert config.openai_compat.on_disconnect == "continue"
    assert config.openai_compat.narration_window_chars == 0
    assert config.openai_compat.max_ingest_messages == 10
    assert config.openai_compat.max_ingest_chars == 2000


def test_app_config_rejects_a_boolean_limit():
    with pytest.raises(ValidationError, match="max_ingest_chars must be an integer, not a boolean"):
        _app_config(openai_compat={"max_ingest_chars": True})


def test_app_config_rejects_an_invalid_section():
    with pytest.raises(ValidationError, match="default_mode"):
        _app_config(openai_compat={"modes": ["flash"], "default_mode": "pro"})
