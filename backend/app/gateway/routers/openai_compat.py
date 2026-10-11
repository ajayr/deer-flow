"""OpenAI-compatible API (``/api/openai/v1``) for OpenAI-style clients such as Conduit in Direct mode.

``create_app()`` mounts this router only when ``openai_compat.enabled`` is true.
Callers authenticate with a personal access token that carries ``runs:create``.
Everything that is not HTTP lives in ``app.gateway.openai_compat``; see its ``AGENTS.md``.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.gateway.authz import require_permission
from app.gateway.deps import get_config
from app.gateway.openai_compat.modes import ADVERTISED_CONTEXT_LENGTH, ModelFacts, build_catalogue
from app.gateway.openai_compat.schemas import ModelList, ModelObject
from app.gateway.routers.models import list_models
from deerflow.config.app_config import AppConfig

router = APIRouter(prefix="/api/openai/v1", tags=["openai-compat"])

# An OpenAI model object carries a creation time. The entries here come from
# configuration, so the moment this process started is the honest value.
_STARTED_AT = int(time.time())


@router.get(
    "/models",
    response_model=ModelList,
    response_model_exclude_none=True,
    summary="List Models (OpenAI-compatible)",
    description="List the ids an OpenAI-style client can send as `model`: each model the caller may use that offers a mode, as `<model>`, plus one `<model>:<mode>` variant per offered mode when it offers more than one.",
)
@require_permission("runs", "create")
async def list_openai_models(request: Request, config: AppConfig = Depends(get_config)) -> ModelList:
    """List the models the caller may use, in the OpenAI ``GET /models`` shape.

    The models come from the Gateway's own model listing, so role-based model
    authorization and its fail-closed / fail-open handling apply here exactly as
    they do for the web UI. Each model is listed by its bare name, which means
    ``openai_compat.default_mode`` clamped to the model's reasoning contract
    (or its first offered mode when the clamped one is not offered), and as one
    ``<model>:<mode>`` variant for each mode in ``openai_compat.modes`` that its
    reasoning contract allows. A model that offers only one mode is listed by
    its bare name alone, and a model that offers no mode is not listed.

    Every entry advertises the same large ``context_length`` and
    ``is_multimodal: false``: clients such as Conduit fall back to a 4096-token
    window when the field is missing, and treat a missing flag as multimodal.
    """
    visible = await list_models(request, config)
    models = [ModelFacts(name=model.name, display_name=model.display_name, description=model.description, thinking=model.reasoning.thinking) for model in visible.models]
    settings = config.openai_compat
    catalogue = build_catalogue(models, configured_modes=settings.modes, default_mode=settings.default_mode)
    return ModelList(
        data=[
            ModelObject(
                id=entry.id,
                created=_STARTED_AT,
                name=entry.name,
                description=entry.description,
                is_multimodal=False,
                context_length=ADVERTISED_CONTEXT_LENGTH,
            )
            for entry in catalogue
        ]
    )
