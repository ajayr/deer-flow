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


def test_a_subset_of_modes_in_any_order_is_accepted():
    config = OpenAICompatConfig(modes=["pro", "flash"], default_mode="flash")

    assert config.modes == ["pro", "flash"]
    assert config.default_mode == "flash"


@pytest.mark.parametrize(
    "modes",
    [[], ["turbo"], ["Flash"], ["flash", "flash"], "flash", None],
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


def test_app_config_rejects_an_invalid_section():
    with pytest.raises(ValidationError, match="default_mode"):
        _app_config(openai_compat={"modes": ["flash"], "default_mode": "pro"})
