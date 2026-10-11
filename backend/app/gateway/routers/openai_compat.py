"""OpenAI-compatible API (``/api/openai/v1``) for OpenAI-style clients such as Conduit in Direct mode.

``create_app()`` mounts this router only when ``openai_compat.enabled`` is true.
Callers authenticate with a personal access token. ``GET /models`` needs ``runs:create``;
``POST /chat/completions`` also needs ``runs:read``, ``threads:read`` and ``threads:write``,
because it creates and updates the caller's threads and reads their runs.
Everything that is not HTTP lives in ``app.gateway.openai_compat``; see its ``AGENTS.md``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.gateway.authz import _is_internal_caller, authorize_model_use, require_permission
from app.gateway.deps import get_config, get_run_manager, get_stream_bridge, get_thread_store
from app.gateway.openai_compat.completion import completion_body, failure_error
from app.gateway.openai_compat.conversation import ThreadPlan, Turn, exchanges, imported_turns, plan_thread, seed_messages, seed_title, text_hash
from app.gateway.openai_compat.driver import RunStream, Settled, settle_run, to_wire
from app.gateway.openai_compat.errors import OpenAIError, internal_error, invalid_request
from app.gateway.openai_compat.ledger import append_turns, find_candidates, record_answer
from app.gateway.openai_compat.messages import parse_messages
from app.gateway.openai_compat.modes import ADVERTISED_CONTEXT_LENGTH, CatalogueEntry, EffortFacts, ModelFacts, build_catalogue, decide_effort, resolve_model, run_context
from app.gateway.openai_compat.params import ignored_params_header, validate_request
from app.gateway.openai_compat.schemas import ChatCompletion, ErrorResponse, ModelList, ModelObject
from app.gateway.openai_compat.translator import Done, StreamTranslator
from app.gateway.routers.models import list_models
from app.gateway.routers.threads import ThreadCreateRequest, ThreadStateUpdateRequest, create_thread, update_thread_state
from app.gateway.run_models import RunCreateRequest
from app.gateway.services import build_thread_checkpoint_state_accessor, start_run
from app.gateway.sse_headers import sse_response_headers
from app.gateway.utils import sanitize_log_param
from deerflow.config.app_config import AppConfig
from deerflow.runtime import DisconnectMode, RunManager, RunRecord, RunStatus, serialize_channel_values_for_api
from deerflow.utils.file_io import await_drained

if TYPE_CHECKING:
    from deerflow.persistence.thread_meta.base import ThreadMetaStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/openai/v1", tags=["openai-compat"])

# An OpenAI model object carries a creation time. The entries here come from
# configuration, so the moment this process started is the honest value.
_STARTED_AT = int(time.time())

#: A run streams the model's messages and the subagent events, the two the translator reads.
_STREAM_MODES = ["messages-tuple", "custom"]

IGNORED_PARAMS_HEADER = "X-DeerFlow-Ignored-Params"

_AUTH_ERRORS = {
    401: {"description": 'Not authenticated; in the Gateway\'s own `{"detail": ...}` shape.'},
    403: {"description": 'The token lacks a required scope, or the model is not available to the caller; in the Gateway\'s own `{"detail": ...}` shape.'},
}


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
            for entry in await _catalogue(request, config)
        ]
    )


@router.post(
    "/chat/completions",
    response_model=ChatCompletion,
    summary="Create Chat Completion (OpenAI-compatible)",
    description=(
        "Run the agent on the newest user message of a resent conversation and answer in the OpenAI `chat.completion` shape, "
        "or as server-sent `chat.completion.chunk` events with `stream: true`. The agent's progress arrives as `reasoning_content`; "
        "tool calls are never returned. The history is matched to a thread of the caller's, so a message already in a thread is not run twice."
    ),
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}, "description": "A `chat.completion`, or with `stream: true` a stream of `chat.completion.chunk` events ending with `data: [DONE]`."},
        400: {"model": ErrorResponse, "description": "A malformed request, an unsupported parameter or message, or a history too large to import."},
        404: {"model": ErrorResponse, "description": "The model is not one that `GET /models` lists for the caller."},
        409: {"model": ErrorResponse, "description": "A run is active in the conversation, or (with `stream: false`) the run was stopped."},
        500: {"model": ErrorResponse, "description": "The request failed before the agent started."},
        502: {"model": ErrorResponse, "description": "With `stream: false`: the model provider failed or the answer was lost. A stream reports these in-band."},
        **_AUTH_ERRORS,
    },
)
@require_permission("runs", "create")
@require_permission("runs", "read")
@require_permission("threads", "read")
@require_permission("threads", "write")
async def create_chat_completion(request: Request, config: AppConfig = Depends(get_config)) -> Response:
    """Answer a chat completion with the agent.

    The body is read as raw JSON, not as a model, so that every problem with it is
    answered in the OpenAI error envelope that OpenAI-style clients display.
    """
    try:
        return await _chat_completion(request, config)
    except OpenAIError as error:
        return _error_response(error)
    except HTTPException as error:
        # The auth layer's own errors keep their shape. Anything else from the Gateway's helpers is not
        # the caller's doing: the request was already checked, so it is the server's.
        if error.status_code in (401, 403):
            raise
        if error.status_code == 409:
            return _error_response(_conversation_busy())
        logger.error("A chat completion was refused by the Gateway with HTTP %s: %s", error.status_code, sanitize_log_param(str(error.detail)))
        return _error_response(internal_error())
    except Exception:
        logger.exception("A chat completion failed before it started")
        return _error_response(internal_error())


def _conversation_busy() -> OpenAIError:
    """The conversation has a run in flight; only this router words it, so it lives here."""
    return OpenAIError(409, "conversation_busy", "This conversation is still running. Wait for it to finish, or open the thread in the web UI to stop it.")


def _error_response(error: OpenAIError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.body())


async def _catalogue(request: Request, config: AppConfig) -> list[CatalogueEntry]:
    """The ids the caller can send as ``model``.

    Built from the Gateway's own model listing, never from ``config.models``, so a model hidden from the caller
    can neither be listed nor be run.
    """
    visible = await list_models(request, config)
    models = []
    for model in visible.models:
        effort = model.reasoning.effort
        models.append(
            ModelFacts(
                name=model.name,
                display_name=model.display_name,
                description=model.description,
                thinking=model.reasoning.thinking,
                effort=None if effort is None else EffortFacts.of(effort.values, effort.aliases),
            )
        )
    settings = config.openai_compat
    return build_catalogue(models, configured_modes=settings.modes, default_mode=settings.default_mode)


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise invalid_request("The request body must be valid JSON.") from None
    if not isinstance(body, dict):
        raise invalid_request("The request body must be a JSON object.")
    return body


class _Hangup:
    """Notices that the client went away.

    ``Request.is_disconnected`` cannot: behind the Gateway's ``BaseHTTPMiddleware`` layers it never reports a disconnect,
    so this waits on the ASGI receive channel instead, as ``skill_export`` does.
    """

    def __init__(self, request: Request) -> None:
        self._watch = asyncio.create_task(self._wait(request))

    @staticmethod
    async def _wait(request: Request) -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    async def occurred(self) -> bool:
        # A watch that failed says nothing about the client, so only a watch that ended cleanly counts.
        return self._watch.done() and not self._watch.cancelled() and self._watch.exception() is None

    def stop(self) -> None:
        self._watch.cancel()


@dataclass(frozen=True, slots=True)
class _LedgerTurn:
    """The ledger entry of this request's turn, so the answer can be recorded when the client has received it."""

    store: ThreadMetaStore
    thread_id: str
    index: int
    user_hash: str
    user_id: str

    async def answered(self, content: str) -> None:
        await record_answer(self.store, self.thread_id, self.index, user_hash=self.user_hash, answer_hash=text_hash(content), user_id=self.user_id)


@dataclass(frozen=True, slots=True)
class _Sent:
    """What the response needs once the run is started."""

    record: RunRecord
    translator: StreamTranslator
    stream: RunStream
    hangup: _Hangup
    turn: _LedgerTurn | None
    #: The caller, the model id with its mode and the names of the ignored parameters: the identifying fields of the log line, which never carries message content.
    user_id: str
    model: str
    ignored: str | None


async def _chat_completion(request: Request, config: AppConfig) -> Response:
    body = await _read_json_object(request)
    params = validate_request(body)
    parsed = parse_messages(body.get("messages"))
    entry = resolve_model(params.model, await _catalogue(request, config))
    effort = decide_effort(entry, params.reasoning_effort)
    user = request.state.auth.user
    authorize_model_use(user, entry.model_name, is_internal=_is_internal_caller(request, user), app_config=config)

    settings = config.openai_compat
    user_id = str(user.id)
    store = get_thread_store(request)
    history = exchanges(parsed.messages)
    conv = text_hash(history[0].user)
    plan = plan_thread(
        history,
        await find_candidates(store, user_id=user_id, conv=conv),
        max_messages=settings.max_ingest_messages,
        max_chars=settings.max_ingest_chars,
    )
    thread_id = plan.thread_id or await _create_thread(request, plan, history[0].user, config)

    user_message_id = str(uuid.uuid4())
    record = await start_run(
        RunCreateRequest(
            input={"messages": [{"role": "user", "content": plan.message, "id": user_message_id}]},
            context=run_context(entry, effort.value),
            stream_mode=_STREAM_MODES,
            multitask_strategy="reject",
            on_disconnect=settings.on_disconnect,
        ),
        thread_id,
        request,
    )

    # The turn is recorded only once the run exists: a request refused as busy leaves the ledger as it was.
    now = time.time()
    turns = [*imported_turns(plan.seed, now=now), Turn(u=text_hash(plan.message), a=None, t=now)]
    first = await append_turns(store, thread_id, turns, user_id=user_id, conv=conv)
    turn = None if first is None else _LedgerTurn(store, thread_id, first + len(turns) - 1, turns[-1].u, user_id)

    async def read_messages() -> Sequence[Mapping[str, Any]]:
        accessor, read_config = await build_thread_checkpoint_state_accessor(request, thread_id=thread_id)
        snapshot = await accessor.aget(read_config)
        return serialize_channel_values_for_api(snapshot.values).get("messages", [])

    async def settle() -> Settled:
        return await settle_run(record, user_message_id=user_message_id, read_messages=read_messages)

    translator = StreamTranslator(
        completion_id=f"chatcmpl-{record.run_id}",
        model=entry.id,
        created=int(now),
        narration_window_chars=settings.narration_window_chars,
        include_usage=params.include_usage,
    )
    hangup = _Hangup(request)
    stream = RunStream(translator, get_stream_bridge(request).subscribe(record.run_id), settle=settle, is_disconnected=hangup.occurred)
    # The header is capped, so the effort that was set aside goes first: a client that sends many unknown fields still learns that its effort had no effect.
    ignored = ignored_params_header(("reasoning_effort",) if effort.ignored else (), params.ignored, parsed.ignored)
    sent = _Sent(record, translator, stream, hangup, turn, user_id, f"{entry.id} ({entry.mode})", ignored)
    run_manager = get_run_manager(request)

    if params.stream:
        response: Response = StreamingResponse(_frames(sent, run_manager), media_type="text/event-stream", headers=sse_response_headers())
    else:
        response = await _answer(sent, run_manager, model=entry.id, created=int(now))
    if ignored is not None:
        response.headers[IGNORED_PARAMS_HEADER] = ignored
    return response


async def _create_thread(request: Request, plan: ThreadPlan, first_user_text: str, config: AppConfig) -> str:
    """A new thread for the caller, seeded with the exchanges the client resent before its newest message."""
    thread_id = str(uuid.uuid4())
    await create_thread(ThreadCreateRequest(thread_id=thread_id), request)
    if plan.seed:
        values = {"messages": seed_messages(plan.seed), "title": seed_title(first_user_text, max_chars=config.title.max_chars)}
        await update_thread_state(thread_id, ThreadStateUpdateRequest(values=values), request)
    return thread_id


async def _frames(sent: _Sent, run_manager: RunManager) -> AsyncIterator[str]:
    """The body of a streaming response."""
    try:
        async with aclosing(sent.stream.steps()) as steps:
            async for step in steps:
                if isinstance(step, Done) and sent.turn is not None:
                    # Before the client sees [DONE]: its next message may arrive at once.
                    await sent.turn.answered(sent.translator.content_text)
                yield to_wire(step)
    finally:
        await _finish(sent, run_manager)


async def _answer(sent: _Sent, run_manager: RunManager, *, model: str, created: int) -> Response:
    """Run the response to its end without sending anything on the way, then answer with the whole completion."""
    try:
        async for _ in sent.stream.steps():
            pass
        stream = sent.stream
        if stream.failure is not None:
            raise failure_error(stream.failure)
        if stream.ended != "done":
            return Response(status_code=499)
        if sent.turn is not None:
            await sent.turn.answered(sent.translator.content_text)
        return JSONResponse(
            completion_body(
                completion_id=f"chatcmpl-{sent.record.run_id}",
                model=model,
                created=created,
                content=sent.translator.content_text,
                reasoning=sent.translator.reasoning_text,
                usage=stream.settled.usage if stream.settled is not None else None,
            )
        )
    finally:
        await _finish(sent, run_manager)


async def _finish(sent: _Sent, run_manager: RunManager) -> None:
    """What ends every response: take the run down with a client that left, and log how it went."""
    sent.hangup.stop()
    stream, record = sent.stream, sent.record
    # A response its consumer closed or cancelled never saw its stream end: that is how a dropped connection looks.
    left = stream.ended in ("open", "disconnected")
    outcome = "done" if stream.ended == "done" else f"failed: {stream.failure.code}" if stream.failure else "client left"
    logger.info("Chat completion ended: user=%s thread=%s run=%s model=%s ignored=%s outcome=%s", sent.user_id, record.thread_id, record.run_id, sent.model, sent.ignored or "none", outcome)
    # Like the Gateway's own run stream: only the stream that created the run applies on_disconnect, only to a run that is still going.
    if left and not record.store_only and record.status in (RunStatus.pending, RunStatus.running) and record.on_disconnect == DisconnectMode.cancel:
        await await_drained(run_manager.cancel(record.run_id))
