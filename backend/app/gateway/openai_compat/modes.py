"""Model catalogue for the OpenAI-compatible API: the ``<model>`` and ``<model>:<mode>`` ids a caller can pick.

Pure: no FastAPI, no I/O. The router passes in the models the caller may use
(already filtered by model authorization) and the ``openai_compat`` settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from deerflow.config.openai_compat_config import OPENAI_COMPAT_MODES, OpenAICompatMode

#: ``context_length`` advertised for every entry. Deliberately not a model's real
#: window: OpenAI-style clients such as Conduit fall back to 4096 tokens when the
#: field is missing and then compact or truncate their transcript, which would make
#: the resent history stop matching the stored thread. DeerFlow summarizes the
#: agent's context on the server, so the client's transcript size has no bearing on it.
ADVERTISED_CONTEXT_LENGTH = 1_048_576

Thinking = Literal["unsupported", "optional", "required"]

# Which modes a model's reasoning contract allows; mirrors getResolvedMode in the
# web UI (frontend/src/core/models/reasoning.ts).
_MODES_BY_THINKING: dict[str, tuple[OpenAICompatMode, ...]] = {
    "unsupported": ("flash",),
    "optional": ("flash", "thinking", "pro", "ultra"),
    "required": ("thinking", "pro", "ultra"),
}


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """What the catalogue needs to know about one model the caller may use."""

    name: str
    display_name: str | None
    description: str | None
    thinking: Thinking


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """One listed id and what it resolves to."""

    id: str
    name: str
    description: str | None
    model_name: str
    mode: OpenAICompatMode


def offered_modes(thinking: str, configured_modes: Sequence[OpenAICompatMode]) -> tuple[OpenAICompatMode, ...]:
    """Modes a model offers: those its reasoning contract allows and ``configured_modes`` lists, in canonical order."""
    supported = _MODES_BY_THINKING.get(thinking, ())
    return tuple(mode for mode in OPENAI_COMPAT_MODES if mode in supported and mode in configured_modes)


def _clamp_to_contract(thinking: str, mode: OpenAICompatMode) -> OpenAICompatMode:
    """Clamp a mode to what the model supports, as the web UI does."""
    if thinking == "unsupported":
        return "flash"
    if thinking == "required" and mode == "flash":
        return "thinking"
    return mode


def bare_mode(thinking: str, configured_modes: Sequence[OpenAICompatMode], default_mode: OpenAICompatMode) -> OpenAICompatMode | None:
    """Mode behind the bare model name, or ``None`` when the model offers no mode at all.

    ``default_mode`` clamped to the model's contract; when that mode is not
    offered (the operator left it out of ``modes``), the first offered mode.
    """
    offered = offered_modes(thinking, configured_modes)
    if not offered:
        return None
    clamped = _clamp_to_contract(thinking, default_mode)
    return clamped if clamped in offered else offered[0]


def build_catalogue(
    models: Sequence[ModelFacts],
    *,
    configured_modes: Sequence[OpenAICompatMode],
    default_mode: OpenAICompatMode,
) -> list[CatalogueEntry]:
    """List the ids a caller can pick, in model order: the bare name, then one variant per offered mode.

    * A model that offers no mode is not listed.
    * A model that offers exactly one mode is listed by its bare name only.
    * A variant id that equals the name of a model the caller can use is skipped,
      so ids stay unique and the configured model wins. Only the caller's own
      models count: a model hidden from them must not change what they see.
    * A model name that repeats keeps its first entry, as ``AppConfig.get_model_config`` does, even when that entry offers no mode.
    """
    reserved = {model.name for model in models}
    seen: set[str] = set()
    entries: list[CatalogueEntry] = []
    for model in models:
        if model.name in seen:
            continue
        seen.add(model.name)
        offered = offered_modes(model.thinking, configured_modes)
        bare = bare_mode(model.thinking, configured_modes, default_mode)
        if bare is None:
            continue
        label = model.display_name or model.name
        entries.append(CatalogueEntry(id=model.name, name=label, description=model.description, model_name=model.name, mode=bare))
        if len(offered) < 2:
            continue
        for mode in offered:
            variant_id = f"{model.name}:{mode}"
            if variant_id in reserved:
                continue
            entries.append(CatalogueEntry(id=variant_id, name=f"{label} ({mode})", description=model.description, model_name=model.name, mode=mode))
    return entries
