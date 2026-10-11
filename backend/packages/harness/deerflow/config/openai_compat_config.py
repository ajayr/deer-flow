"""Configuration for the OpenAI-compatible HTTP API (``/api/openai/v1``)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from deerflow.config.reload_boundary import format_field_description

OpenAICompatMode = Literal["flash", "thinking", "pro", "ultra"]

#: Every mode the API can offer, in the canonical order the model catalogue lists them.
OPENAI_COMPAT_MODES: tuple[OpenAICompatMode, ...] = ("flash", "thinking", "pro", "ultra")


class OpenAICompatConfig(BaseModel):
    """OpenAI-compatible API settings.

    The API serves OpenAI-style clients (for example Conduit's Direct mode) that
    authenticate with a personal access token. ``enabled`` is read once by
    ``create_app()``; ``default_mode`` and ``modes`` are read on every request.
    The order of ``modes`` does not matter: the model catalogue always lists
    modes in the canonical order of ``OPENAI_COMPAT_MODES``.
    """

    enabled: bool = Field(
        default=False,
        description=format_field_description(
            "openai_compat.enabled",
            field_doc="Mount the OpenAI-compatible API at /api/openai/v1. Off by default.",
        ),
    )
    default_mode: OpenAICompatMode = Field(
        default="thinking",
        description="Mode behind a bare model name: flash, thinking, pro or ultra. Must be one of `modes`.",
    )
    modes: list[OpenAICompatMode] = Field(
        default_factory=lambda: list(OPENAI_COMPAT_MODES),
        min_length=1,
        description="Modes offered as <model>:<mode> variants: a non-empty, duplicate-free subset of flash, thinking, pro and ultra.",
    )

    @field_validator("modes")
    @classmethod
    def _reject_duplicate_modes(cls, value: list[OpenAICompatMode]) -> list[OpenAICompatMode]:
        if len(set(value)) != len(value):
            raise ValueError("modes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _default_mode_must_be_offered(self) -> Self:
        if self.default_mode not in self.modes:
            raise ValueError(f"default_mode {self.default_mode!r} must be one of modes {self.modes}")
        return self
