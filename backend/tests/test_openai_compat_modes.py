"""Model catalogue for the OpenAI-compatible API: which ids exist and what each one resolves to."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.gateway.openai_compat.errors import MAX_MESSAGE_CHARS, OpenAIError
from app.gateway.openai_compat.modes import (
    ADVERTISED_CONTEXT_LENGTH,
    MODE_FLAGS,
    EffortFacts,
    ModelFacts,
    bare_mode,
    build_catalogue,
    decide_effort,
    offered_modes,
    resolve_model,
    run_context,
)
from app.gateway.openai_compat.schemas import ModelList, ModelObject
from deerflow.config.openai_compat_config import OPENAI_COMPAT_MODES

ALL_MODES = ["flash", "thinking", "pro", "ultra"]
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "app" / "gateway" / "openai_compat"


def _model(name: str, thinking: str = "optional", *, display_name: str | None = None, description: str | None = None, effort: EffortFacts | None = None) -> ModelFacts:
    return ModelFacts(name=name, display_name=display_name, description=description, thinking=thinking, effort=effort)


def _catalogue(models, *, modes=ALL_MODES, default_mode="thinking"):
    return build_catalogue(models, configured_modes=modes, default_mode=default_mode)


def _ids(entries) -> list[str]:
    return [entry.id for entry in entries]


def test_advertised_context_length_is_large_enough_to_stop_client_side_compaction():
    # Conduit falls back to 4096 tokens when the field is missing and then trims the history it resends.
    assert ADVERTISED_CONTEXT_LENGTH == 1_048_576


# ── offered_modes ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        ("unsupported", ("flash",)),
        ("optional", ("flash", "thinking", "pro", "ultra")),
        ("required", ("thinking", "pro", "ultra")),
    ],
)
def test_offered_modes_follow_the_reasoning_contract(thinking, expected):
    assert offered_modes(thinking, ALL_MODES) == expected


def test_offered_modes_are_limited_to_the_configured_ones_in_canonical_order():
    assert offered_modes("optional", ["ultra", "flash"]) == ("flash", "ultra")
    assert offered_modes("required", ["flash", "pro"]) == ("pro",)
    assert offered_modes("unsupported", ["thinking", "pro"]) == ()


def test_an_unknown_reasoning_contract_offers_nothing():
    assert offered_modes("sometimes", ALL_MODES) == ()


# ── bare_mode ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("thinking", "configured", "default", "expected"),
    [
        ("optional", ALL_MODES, "thinking", "thinking"),
        ("optional", ALL_MODES, "ultra", "ultra"),
        ("required", ALL_MODES, "pro", "pro"),
        # The default is clamped to the contract, as the web UI does.
        ("unsupported", ALL_MODES, "ultra", "flash"),
        ("required", ALL_MODES, "flash", "thinking"),
        # The clamped mode is not one the operator offers: the first offered mode wins.
        ("required", ["flash", "pro"], "flash", "pro"),
        # Nothing offered at all.
        ("unsupported", ["thinking", "pro"], "thinking", None),
    ],
)
def test_bare_mode(thinking, configured, default, expected):
    assert bare_mode(thinking, configured, default) == expected


# ── build_catalogue ──────────────────────────────────────────────────────────


def test_catalogue_lists_the_bare_name_then_one_variant_per_offered_mode():
    entries = _catalogue([_model("alpha")])

    assert _ids(entries) == ["alpha", "alpha:flash", "alpha:thinking", "alpha:pro", "alpha:ultra"]
    assert [(entry.model_name, entry.mode) for entry in entries] == [
        ("alpha", "thinking"),
        ("alpha", "flash"),
        ("alpha", "thinking"),
        ("alpha", "pro"),
        ("alpha", "ultra"),
    ]


def test_each_model_offers_the_modes_its_reasoning_contract_allows():
    entries = _catalogue([_model("optional-m", "optional"), _model("fixed-m", "unsupported"), _model("always-m", "required")])

    assert _ids(entries) == [
        "optional-m",
        "optional-m:flash",
        "optional-m:thinking",
        "optional-m:pro",
        "optional-m:ultra",
        "fixed-m",
        "always-m",
        "always-m:thinking",
        "always-m:pro",
        "always-m:ultra",
    ]
    modes = {entry.id: entry.mode for entry in entries}
    assert modes["fixed-m"] == "flash"
    assert modes["always-m"] == "thinking"


def test_a_model_that_offers_a_single_mode_is_listed_by_its_bare_name_only():
    entries = _catalogue([_model("fixed-m", "unsupported")])

    assert _ids(entries) == ["fixed-m"]
    assert entries[0].mode == "flash"


def test_a_model_that_offers_no_mode_is_not_listed():
    # A model without thinking cannot serve thinking/pro/ultra, so with only those configured it has nothing to offer.
    entries = _catalogue([_model("fixed-m", "unsupported"), _model("alpha")], modes=["thinking", "pro"], default_mode="thinking")

    assert "fixed-m" not in _ids(entries)
    assert _ids(entries)[0] == "alpha"


def test_names_use_the_display_name_and_variants_carry_their_mode():
    entries = _catalogue([_model("alpha", display_name="Alpha 1", description="Fast and wide")])
    by_id = {entry.id: entry for entry in entries}

    assert by_id["alpha"].name == "Alpha 1"
    assert by_id["alpha:pro"].name == "Alpha 1 (pro)"
    assert by_id["alpha:pro"].description == "Fast and wide"


def test_a_model_without_a_display_name_is_labelled_by_its_name():
    entries = _catalogue([_model("alpha")])

    assert entries[0].name == "alpha"
    assert entries[1].name == "alpha (flash)"


@pytest.mark.parametrize("default_mode", ["flash", "thinking", "pro", "ultra"])
def test_the_bare_name_resolves_to_the_default_mode(default_mode):
    assert _catalogue([_model("alpha")], default_mode=default_mode)[0].mode == default_mode


def test_narrowing_modes_narrows_the_variants_and_the_bare_mode():
    entries = _catalogue([_model("alpha"), _model("always-m", "required")], modes=["flash", "pro"], default_mode="pro")

    assert {entry.id: entry.mode for entry in entries} == {
        "alpha": "pro",
        "alpha:flash": "flash",
        "alpha:pro": "pro",
        "always-m": "pro",
    }
    assert _ids(entries) == ["alpha", "alpha:flash", "alpha:pro", "always-m"]


def test_a_variant_id_equal_to_a_usable_model_name_is_skipped_and_the_model_wins():
    entries = _catalogue([_model("alpha"), _model("alpha:pro", "unsupported")])
    ids = _ids(entries)

    assert ids.count("alpha:pro") == 1
    owner = next(entry for entry in entries if entry.id == "alpha:pro")
    assert (owner.model_name, owner.mode) == ("alpha:pro", "flash")
    assert {"alpha:flash", "alpha:thinking", "alpha:ultra"} <= set(ids)


def test_an_unlisted_model_still_reserves_its_name():
    """A request for "a:pro" means the configured model of that name, even when that model is not listed."""
    entries = _catalogue([_model("a"), _model("a:pro", "unsupported")], modes=["thinking", "pro"], default_mode="thinking")

    assert _ids(entries) == ["a", "a:thinking"]


def test_a_name_that_is_not_in_the_input_reserves_nothing():
    """The router passes only the caller's models, so a model hidden from them cannot take a variant id away from a visible one."""
    assert "alpha:pro" in _ids(_catalogue([_model("alpha")]))


def test_a_repeated_model_name_keeps_its_first_entry():
    entries = _catalogue([_model("alpha", display_name="First"), _model("alpha", "unsupported", display_name="Second")])
    ids = _ids(entries)

    assert ids.count("alpha") == 1
    assert entries[0].name == "First"
    assert "alpha:pro" in ids


def test_a_repeated_model_name_whose_first_entry_offers_no_mode_is_not_listed():
    """The name resolves to its first entry, so a later entry must not advertise modes the first cannot serve."""
    entries = _catalogue([_model("x", "unsupported"), _model("x")], modes=["thinking", "pro"], default_mode="thinking")

    assert entries == []


def test_ids_are_unique_whatever_the_model_names():
    models = [_model("a"), _model("a:flash"), _model("a:flash:pro"), _model("b", "required"), _model("b:pro"), _model("a")]
    ids = _ids(_catalogue(models))

    assert len(ids) == len(set(ids))


def test_an_empty_model_list_gives_an_empty_catalogue():
    assert _catalogue([]) == []


# ── effort facts and mode flags ──────────────────────────────────────────────


def test_a_model_accepts_its_effort_values_and_then_its_alias_keys_once_each():
    assert EffortFacts.of(["low", "high"], {"max": "high", "low": "low"}).accepted == ("low", "high", "max")
    assert EffortFacts.of(["minimal", "low", "medium", "high"]).accepted == ("minimal", "low", "medium", "high")


def test_every_catalogue_entry_carries_the_effort_facts_of_its_model():
    facts = EffortFacts.of(["low", "high"])
    entries = _catalogue([_model("alpha", effort=facts), _model("beta")])

    assert {entry.id: entry.effort for entry in entries if entry.model_name == "alpha"} == dict.fromkeys(["alpha", "alpha:flash", "alpha:thinking", "alpha:pro", "alpha:ultra"], facts)
    assert all(entry.effort is None for entry in entries if entry.model_name == "beta")


def test_the_flags_cover_exactly_the_configurable_modes():
    assert set(MODE_FLAGS) == set(OPENAI_COMPAT_MODES)


@pytest.mark.parametrize(
    ("mode", "thinking", "plan", "subagents", "default_effort"),
    [
        ("flash", False, False, False, None),
        ("thinking", True, False, False, "low"),
        ("pro", True, True, False, "medium"),
        ("ultra", True, True, True, "high"),
    ],
)
def test_each_mode_switches_on_what_the_web_ui_does(mode, thinking, plan, subagents, default_effort):
    flags = MODE_FLAGS[mode]

    assert (flags.thinking, flags.plan, flags.subagents, flags.default_effort) == (thinking, plan, subagents, default_effort)


# ── run_context ──────────────────────────────────────────────────────────────


def _entry(mode: str = "pro", *, effort: EffortFacts | None = None):
    """The ``alpha:<mode>`` entry of a model that offers every mode."""
    return next(entry for entry in _catalogue([_model("alpha", effort=effort)]) if entry.id == f"alpha:{mode}")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("flash", {"thinking_enabled": False, "is_plan_mode": False, "subagent_enabled": False}),
        ("thinking", {"thinking_enabled": True, "is_plan_mode": False, "subagent_enabled": False}),
        ("pro", {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": False}),
        ("ultra", {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": True}),
    ],
)
def test_the_run_context_is_the_model_and_the_flags_of_the_mode(mode, expected):
    assert run_context(_entry(mode), None) == {"model_name": "alpha", **expected}


def test_the_run_context_names_the_configured_model_not_the_listed_id():
    entry = next(entry for entry in _catalogue([_model("alpha")]) if entry.id == "alpha:ultra")

    assert run_context(entry, None)["model_name"] == "alpha"


def test_the_run_context_carries_the_effort_only_when_there_is_one():
    assert "reasoning_effort" not in run_context(_entry("ultra"), None)
    assert run_context(_entry("ultra"), "high")["reasoning_effort"] == "high"


# ── decide_effort ────────────────────────────────────────────────────────────

GENERIC = EffortFacts.of(["minimal", "low", "medium", "high"])
DECLARED = EffortFacts.of(["low", "high"], {"max": "high"})


@pytest.mark.parametrize(("mode", "default"), [("flash", None), ("thinking", "low"), ("pro", "medium"), ("ultra", "high")])
def test_without_a_request_the_mode_supplies_its_default_effort(mode, default):
    decision = decide_effort(_entry(mode, effort=GENERIC), None)

    assert (decision.value, decision.ignored) == (default, False)


@pytest.mark.parametrize("mode", ["flash", "thinking", "pro", "ultra"])
def test_a_model_without_an_effort_control_is_sent_no_effort_by_default(mode):
    decision = decide_effort(_entry(mode), None)

    assert (decision.value, decision.ignored) == (None, False)


@pytest.mark.parametrize("requested", ["high", "bogus", ""])
def test_a_requested_effort_is_set_aside_and_reported_when_the_mode_has_thinking_off(requested):
    decision = decide_effort(_entry("flash", effort=GENERIC), requested)

    assert (decision.value, decision.ignored) == (None, True)


@pytest.mark.parametrize("requested", ["high", "bogus", ""])
@pytest.mark.parametrize("mode", ["flash", "thinking", "pro", "ultra"])
def test_a_requested_effort_is_set_aside_and_reported_when_the_model_has_no_effort_control(mode, requested):
    decision = decide_effort(_entry(mode), requested)

    assert (decision.value, decision.ignored) == (None, True)


@pytest.mark.parametrize("mode", ["thinking", "pro", "ultra"])
@pytest.mark.parametrize("requested", ["minimal", "low", "medium", "high"])
def test_a_value_the_model_accepts_replaces_the_default(mode, requested):
    decision = decide_effort(_entry(mode, effort=GENERIC), requested)

    assert (decision.value, decision.ignored) == (requested, False)


def test_an_alias_is_sent_as_it_came_and_the_gateway_maps_it():
    decision = decide_effort(_entry("pro", effort=DECLARED), "max")

    assert (decision.value, decision.ignored) == ("max", False)


@pytest.mark.parametrize("requested", ["xhigh", "none", "High", " high", ""])
def test_a_value_the_model_does_not_accept_is_a_400_that_lists_what_it_does(requested):
    with pytest.raises(OpenAIError) as caught:
        decide_effort(_entry("pro", effort=DECLARED), requested)

    error = caught.value
    assert (error.status_code, error.code, error.param) == (400, "unsupported_value", "reasoning_effort")
    assert f"Value '{requested}' is not supported for 'alpha:pro'." in error.message
    assert "Accepted: low, high, max." in error.message


def test_a_value_is_sent_exactly_as_the_model_lists_it():
    decision = decide_effort(_entry("pro", effort=EffortFacts.of(["Low", "xHigh"], {"Max": "xHigh"})), "xHigh")

    assert decision.value == "xHigh"
    assert decide_effort(_entry("pro", effort=EffortFacts.of(["Low", "xHigh"], {"Max": "xHigh"})), "Max").value == "Max"


def test_an_unaccepted_value_is_shortened_inside_the_message():
    with pytest.raises(OpenAIError) as caught:
        decide_effort(_entry("pro", effort=DECLARED), "x" * 500)

    assert len(caught.value.message) < MAX_MESSAGE_CHARS


# ── resolve_model ────────────────────────────────────────────────────────────


def test_a_listed_id_resolves_to_its_entry():
    entries = _catalogue([_model("alpha"), _model("beta", "required")])

    assert resolve_model("alpha", entries) is entries[0]
    assert resolve_model("alpha:ultra", entries).mode == "ultra"
    assert resolve_model("beta:pro", entries).model_name == "beta"


def test_resolving_the_bare_name_gives_the_entry_of_the_default_mode():
    assert resolve_model("alpha", _catalogue([_model("alpha")], default_mode="ultra")).mode == "ultra"


def test_a_model_whose_name_contains_a_colon_resolves_by_its_exact_id():
    entries = _catalogue([_model("llama3:8b"), _model("plain")])

    assert resolve_model("llama3:8b", entries).model_name == "llama3:8b"
    assert resolve_model("llama3:8b:pro", entries) is not None
    assert resolve_model("llama3:8b:pro", entries).model_name == "llama3:8b"


def test_a_name_that_is_also_a_variant_id_resolves_to_the_configured_model():
    entries = _catalogue([_model("alpha"), _model("alpha:pro", "unsupported")])

    assert resolve_model("alpha:pro", entries).model_name == "alpha:pro"


@pytest.mark.parametrize("model_id", ["", "gamma", "alpha:", "alpha:turbo", "ALPHA", "alpha:flash:pro", "alpha :pro", " alpha", "alpha ", "alpha\n"])
def test_an_id_the_list_does_not_contain_is_a_404_that_names_the_ids_that_exist(model_id):
    entries = _catalogue([_model("alpha"), _model("fixed", "unsupported")])

    with pytest.raises(OpenAIError) as caught:
        resolve_model(model_id, entries)

    error = caught.value
    assert (error.status_code, error.code, error.param) == (404, "model_not_found", "model")
    assert "does not exist or is not available to you" in error.message
    assert "Available: alpha, alpha:flash, alpha:thinking, alpha:pro, alpha:ultra, fixed." in error.message


def test_a_variant_the_list_omits_is_not_resolved_even_when_it_would_mean_the_listed_mode():
    """A model with one mode is listed by its bare name only; the list is the contract."""
    entries = _catalogue([_model("fixed", "unsupported")])

    with pytest.raises(OpenAIError) as caught:
        resolve_model("fixed:flash", entries)

    assert caught.value.status_code == 404


def test_a_model_the_caller_may_not_use_is_indistinguishable_from_one_that_does_not_exist():
    entries = _catalogue([_model("alpha")])

    with pytest.raises(OpenAIError) as hidden:
        resolve_model("secret-model", entries)
    with pytest.raises(OpenAIError) as missing:
        resolve_model("no-such-model", entries)

    assert hidden.value.message.replace("secret-model", "X") == missing.value.message.replace("no-such-model", "X")


def test_an_empty_catalogue_says_there_is_nothing_to_pick():
    with pytest.raises(OpenAIError) as caught:
        resolve_model("alpha", [])

    assert "Available: none." in caught.value.message


def test_the_list_in_the_message_is_cut_to_the_message_limit():
    entries = _catalogue([_model(f"model-number-{index:03d}") for index in range(40)])

    with pytest.raises(OpenAIError) as caught:
        resolve_model("missing", entries)

    assert len(caught.value.message) == MAX_MESSAGE_CHARS
    assert caught.value.message.endswith("...")
    assert "Available: model-number-000, model-number-000:flash" in caught.value.message


def test_a_client_supplied_id_is_shortened_and_made_printable_inside_the_message():
    with pytest.raises(OpenAIError) as caught:
        resolve_model("x\r\ny" * 40, _catalogue([_model("alpha")]))

    assert "\r" not in caught.value.message and "\n" not in caught.value.message
    assert len(caught.value.message) < MAX_MESSAGE_CHARS


# ── schemas ──────────────────────────────────────────────────────────────────


def test_model_object_serialises_the_openai_and_open_webui_fields():
    entry = ModelObject(id="alpha", created=1, name="Alpha", is_multimodal=False, context_length=ADVERTISED_CONTEXT_LENGTH)

    assert entry.model_dump(exclude_none=True) == {
        "id": "alpha",
        "object": "model",
        "created": 1,
        "owned_by": "deerflow",
        "name": "Alpha",
        "is_multimodal": False,
        "context_length": 1_048_576,
    }


def test_model_list_wraps_the_entries():
    assert ModelList(data=[]).model_dump() == {"object": "list", "data": []}


# ── layering ─────────────────────────────────────────────────────────────────


def test_the_package_modules_import_no_web_framework():
    """Only the router may import FastAPI, so the logic in this package tests without an app."""
    assert PACKAGE_DIR.is_dir()
    offenders = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}: {name}" for name in imported if name.split(".")[0] in {"fastapi", "starlette"}]
    assert offenders == []
