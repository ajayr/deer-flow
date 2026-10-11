"""Models and modes for the OpenAI-compatible API: the ``<model>`` and ``<model>:<mode>`` ids a caller can pick, and what a pick means for a run.

Pure: no FastAPI, no I/O. The router passes in the models the caller may use
(already filtered by model authorization) and the ``openai_compat`` settings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.gateway.openai_compat.errors import model_not_found, quote, unsupported_value
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
class EffortFacts:
    """A model's reasoning-effort control: the values a caller may send."""

    #: The contract's values, then the keys of its aliases.
    accepted: tuple[str, ...]

    @classmethod
    def of(cls, values: Iterable[str], aliases: Iterable[str] = ()) -> EffortFacts:
        return cls(accepted=tuple(dict.fromkeys([*values, *aliases])))


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """What the catalogue needs to know about one model the caller may use."""

    name: str
    display_name: str | None
    description: str | None
    thinking: Thinking
    #: ``None`` when the model has no effort control.
    effort: EffortFacts | None = None


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """One listed id and what it resolves to."""

    id: str
    name: str
    description: str | None
    model_name: str
    mode: OpenAICompatMode
    effort: EffortFacts | None = None


@dataclass(frozen=True, slots=True)
class ModeFlags:
    """What a mode switches on for a run, as the web UI does (``frontend/src/core/threads/hooks.ts``)."""

    thinking: bool
    plan: bool
    subagents: bool
    #: The ``reasoning_effort`` sent when the request names none; ``None`` sends nothing.
    default_effort: str | None


MODE_FLAGS: dict[OpenAICompatMode, ModeFlags] = {
    "flash": ModeFlags(thinking=False, plan=False, subagents=False, default_effort=None),
    "thinking": ModeFlags(thinking=True, plan=False, subagents=False, default_effort="low"),
    "pro": ModeFlags(thinking=True, plan=True, subagents=False, default_effort="medium"),
    "ultra": ModeFlags(thinking=True, plan=True, subagents=True, default_effort="high"),
}


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
        entries.append(CatalogueEntry(id=model.name, name=label, description=model.description, model_name=model.name, mode=bare, effort=model.effort))
        if len(offered) < 2:
            continue
        for mode in offered:
            variant_id = f"{model.name}:{mode}"
            if variant_id in reserved:
                continue
            entries.append(CatalogueEntry(id=variant_id, name=f"{label} ({mode})", description=model.description, model_name=model.name, mode=mode, effort=model.effort))
    return entries


def resolve_model(model_id: str, catalogue: Sequence[CatalogueEntry]) -> CatalogueEntry:
    """The entry whose id is ``model_id``.

    The list is the contract: only an id that ``GET /models`` lists for the caller resolves,
    so a model whose own name contains a colon works and a hidden model looks like a missing one.
    Anything else is a 404 that names the ids that do exist (cut to the message limit).
    """
    for entry in catalogue:
        if entry.id == model_id:
            return entry
    available = ", ".join(entry.id for entry in catalogue) or "none"
    raise model_not_found(f"The model {quote(model_id)} does not exist or is not available to you. Available: {available}.")


@dataclass(frozen=True, slots=True)
class EffortDecision:
    """The ``reasoning_effort`` a run gets and whether the request's own value was set aside."""

    value: str | None
    #: True when the request named an effort that had no effect: the variant has thinking off or the model has no effort control.
    ignored: bool = False


def decide_effort(entry: CatalogueEntry, requested: str | None) -> EffortDecision:
    """Apply the ``reasoning_effort`` rules of the design (section 5.4) to one request.

    No value requested: the mode's default, for a model that has an effort control. A value
    requested for a mode with thinking off, or for a model with no control, is set aside and
    reported. Any other value must be one the model accepts, or it is a 400. The Gateway's
    resolver then maps what is sent through the model's contract, as it does for the web UI.
    """
    effort = entry.effort
    flags = MODE_FLAGS[entry.mode]
    if requested is None:
        return EffortDecision(value=flags.default_effort if effort is not None else None)
    if effort is None or not flags.thinking:
        return EffortDecision(value=None, ignored=True)
    if requested not in effort.accepted:
        raise unsupported_value("reasoning_effort", f"Value {quote(requested)} is not supported for {quote(entry.id)}. Accepted: {', '.join(effort.accepted)}.")
    return EffortDecision(value=requested)


def run_context(entry: CatalogueEntry, reasoning_effort: str | None) -> dict[str, Any]:
    """The context of the run for one request: exactly the keys the web UI sets for a mode, and nothing from the client."""
    flags = MODE_FLAGS[entry.mode]
    context: dict[str, Any] = {
        "model_name": entry.model_name,
        "thinking_enabled": flags.thinking,
        "is_plan_mode": flags.plan,
        "subagent_enabled": flags.subagents,
    }
    if reasoning_effort is not None:
        context["reasoning_effort"] = reasoning_effort
    return context
