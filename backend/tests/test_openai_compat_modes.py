"""Model catalogue for the OpenAI-compatible API: which ids exist and what each one resolves to."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.gateway.openai_compat.modes import (
    ADVERTISED_CONTEXT_LENGTH,
    ModelFacts,
    bare_mode,
    build_catalogue,
    offered_modes,
)
from app.gateway.openai_compat.schemas import ModelList, ModelObject

ALL_MODES = ["flash", "thinking", "pro", "ultra"]
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "app" / "gateway" / "openai_compat"


def _model(name: str, thinking: str = "optional", *, display_name: str | None = None, description: str | None = None) -> ModelFacts:
    return ModelFacts(name=name, display_name=display_name, description=description, thinking=thinking)


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
