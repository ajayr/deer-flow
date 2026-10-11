"""``POST /api/openai/v1/chat/completions`` end to end.

A real Gateway composition: the auth and CSRF middleware, personal access tokens, the router, ``start_run``, the run worker,
the stream bridge, the checkpointer and the thread store. Only the model is scripted: a one-node graph whose chat model
replies with the text a test queues, so a streamed answer is what the translator gets from LangGraph's own message stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import deerflow.persistence.models  # noqa: F401  (register every table)
from app.gateway.auth.config import AuthConfig, set_auth_config
from app.gateway.auth.pat import generate_pat_token, pat_token_digest
from app.gateway.auth_middleware import AuthMiddleware
from app.gateway.csrf_middleware import CSRFMiddleware
from app.gateway.extension_agent_runs import GatewayAgentRunsHost
from app.gateway.openai_compat.conversation import text_hash
from app.gateway.openai_compat.driver import KEEPALIVE, RunStream, Settled
from app.gateway.openai_compat.errors import RUN_ERROR, STREAM_GAP, OpenAIError
from app.gateway.openai_compat.params import MAX_IGNORED_NAMES
from app.gateway.openai_compat.translator import DONE, Chunk, Failed, Usage
from app.gateway.routers import openai_compat as openai_router
from deerflow.authz.provider import AuthzDecision, AuthzReason
from deerflow.config.app_config import AppConfig, reset_app_config, set_app_config
from deerflow.config.authorization_config import AuthorizationConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.openai_compat_config import OpenAICompatConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.title_config import TitleConfig
from deerflow.config.token_usage_config import TokenUsageConfig
from deerflow.persistence.base import Base
from deerflow.persistence.personal_access_tokens import PersonalAccessTokenRepository
from deerflow.persistence.run import RunRepository
from deerflow.persistence.thread_meta.memory import MemoryThreadMetaStore
from deerflow.runtime import DisconnectMode, RunManager, RunStatus
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.stream_bridge.memory import MemoryStreamBridge

pytestmark = pytest.mark.no_auto_user

URL = "/api/openai/v1/chat/completions"
JWT_SECRET = "test-openai-compat-jwt-secret-0123456789"
CHAT_SCOPES = ["runs:create", "runs:read", "threads:read", "threads:write"]
FIRST = "Capital of France?"
FOLLOW_UP = "How many people live there?"


# ── the Gateway ──────────────────────────────────────────────────────────────


def _user(user_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, email=f"{user_id}@example.com", system_role="user", needs_setup=False, token_version=0, oauth_provider=None, password_hash=None)


class _Users:
    """The Gateway's user lookup: resolves a user by id."""

    def __init__(self, *user_ids: str) -> None:
        self._users = {user_id: _user(user_id) for user_id in user_ids}

    async def get_user(self, user_id: str):
        return self._users.get(str(user_id))


@dataclass
class Reply:
    """One scripted answer: the text, the reasoning that comes before it and the token usage the provider reports."""

    text: str
    reasoning: str = ""
    usage: dict[str, int] | None = None
    #: The model asks for this tool after the text, which makes the text narration.
    tool: str | None = None
    #: A later node rewrites the stored message to this text, as middleware can; the rewrite is never streamed.
    rewritten: str | None = None
    #: Subagent tasks that start and finish while the run works.
    subagents: tuple[str, ...] = ()
    #: The reply is the error message that DeerFlow's model-call middleware returns when the provider fails, and it carries `deerflow_error_fallback`.
    fallback: bool = False


class _ScriptedModel(BaseChatModel):
    """A chat model that answers with a fixed reply, streamed word by word."""

    reply: Reply

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _message(self, text: str, usage: dict[str, int] | None, *, chunk: bool, first: bool = False) -> Any:
        extra = {"reasoning_content": self.reply.reasoning} if first and self.reply.reasoning else {}
        if self.reply.fallback:
            extra["deerflow_error_fallback"] = True
        factory = AIMessageChunk if chunk else AIMessage
        return factory(content=text, additional_kwargs=extra, usage_metadata=usage)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._message(self.reply.text, self.reply.usage, chunk=False, first=True)
        if self.reply.tool:
            message.tool_calls = [{"name": self.reply.tool, "args": {}, "id": "call-1"}]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        words = [word for word in re.split(r"(\s)", self.reply.text) if word]
        for index, word in enumerate(words):
            usage = self.reply.usage if index == len(words) - 1 else None
            yield ChatGenerationChunk(message=self._message(word, usage, chunk=True, first=index == 0))
        if self.reply.tool:
            call = {"name": self.reply.tool, "args": "{}", "id": "call-1", "index": 0}
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[call]))


@dataclass
class Script:
    """What the scripted agent says and what it was given.

    ``replies`` is the queue of what the next runs answer: a ``Reply`` or a string is streamed, an exception is raised, and a
    coroutine function is awaited first (it returns the reply), so a test can hold a run open.
    """

    replies: list[Any] = field(default_factory=list)
    #: The messages each run was given, as the model saw them.
    inputs: list[list[BaseMessage]] = field(default_factory=list)
    #: The ``configurable`` of each run: the context the router set for it.
    contexts: list[dict[str, Any]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    #: What the rewriting node of the current run puts in place of the stored answer, if anything.
    rewritten: str | None = None
    #: When set, the run stays open after its answer until the test sets this event.
    finishing: asyncio.Event | None = None

    def say(self, *replies: Any) -> None:
        self.replies.extend(replies)


class _State(MessagesState):
    """The lead agent's thread state, reduced to the channels these tests touch: it has a title channel too."""

    title: str | None


def _agent_factory(script: Script):
    async def answer(state: _State, config: RunnableConfig):
        script.inputs.append(list(state["messages"]))
        script.contexts.append(dict(config.get("configurable", {})))
        script.started.set()
        reply = script.replies.pop(0) if script.replies else "ok"
        try:
            if callable(reply):
                reply = await reply()
        except asyncio.CancelledError:
            script.cancelled.set()
            raise
        if isinstance(reply, BaseException):
            raise reply
        reply = Reply(reply) if isinstance(reply, str) else reply
        script.rewritten = reply.rewritten
        writer = get_stream_writer()
        for task in reply.subagents:
            writer({"type": "task_started", "task_id": task})
            writer({"type": "task_completed", "task_id": task})
        return {"messages": [await _ScriptedModel(reply=reply).ainvoke(state["messages"], config)]}

    async def rewrite(state: _State):
        if script.finishing is not None:
            await script.finishing.wait()
        if script.rewritten is None:
            return {"messages": []}
        return {"messages": [AIMessage(content=script.rewritten, id=state["messages"][-1].id)]}

    def factory(*, config):
        builder = StateGraph(_State)
        builder.add_node("answer", answer)
        builder.add_node("rewrite", rewrite)
        builder.set_entry_point("answer")
        builder.add_edge("answer", "rewrite")
        builder.set_finish_point("rewrite")
        return builder.compile()

    return factory


def _model(name: str, thinking: str = "optional", effort: list[str] | None = None, aliases: dict[str, str] | None = None, **extra: Any) -> ModelConfig:
    reasoning: dict[str, Any] = {"thinking": thinking}
    if effort:
        reasoning["effort"] = {"values": effort, "default": effort[0], **({"aliases": aliases} if aliases else {})}
    return ModelConfig(name=name, model=name, use="langchain_openai:ChatOpenAI", reasoning=reasoning, **extra)


@dataclass
class Gateway:
    app: FastAPI
    client: httpx.AsyncClient
    script: Script
    pats: PersonalAccessTokenRepository

    async def token(self, user_id: str = "user-1", scopes: list[str] | None = None) -> str:
        token = generate_pat_token()
        await self.pats.create(user_id=user_id, name="conduit", scopes=scopes or CHAT_SCOPES, token_digest=pat_token_digest(token))
        return token

    async def post(self, payload: Any, *, token: str | None = None, **headers: str) -> httpx.Response:
        content = payload if isinstance(payload, str) else json.dumps(payload)
        auth = {"Authorization": f"Bearer {token or await self.token()}"}
        return await self.client.post(URL, content=content, headers={"Content-Type": "application/json", **auth, **headers})

    async def chat(self, messages: list[dict[str, Any]], *, token: str | None = None, model: str = "alpha", **body: Any) -> httpx.Response:
        return await self.post({"model": model, "messages": messages, **body}, token=token)

    async def threads(self, user_id: str = "user-1") -> list[dict[str, Any]]:
        return await self.app.state.thread_store.search(user_id=user_id)

    async def only_thread(self, user_id: str = "user-1") -> dict[str, Any]:
        threads = await self.threads(user_id)
        assert len(threads) == 1, threads
        return threads[0]


@asynccontextmanager
async def running_gateway(
    monkeypatch,
    tmp_path,
    *,
    models: list[ModelConfig] | None = None,
    users: tuple[str, ...] = ("user-1", "user-2"),
    title_max_chars: int | None = None,
    **openai_compat: Any,
) -> AsyncIterator[Gateway]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'gateway.db'}", poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    config = AppConfig(
        models=models or [_model("alpha")],
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        token_usage=TokenUsageConfig(enabled=False),
        authorization=AuthorizationConfig(),
        openai_compat=OpenAICompatConfig(enabled=True, **openai_compat),
        **({} if title_max_chars is None else {"title": TitleConfig(max_chars=title_max_chars)}),
    )
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path))
    monkeypatch.setenv("DEER_FLOW_AUTH_DISABLED", "")
    set_auth_config(AuthConfig(jwt_secret=JWT_SECRET, token_expiry_days=7))
    set_app_config(config)

    script = Script()
    provider = _Users(*users)
    for target in ("app.gateway.deps.get_local_provider", "app.gateway.routers.auth.get_local_provider", "app.gateway.services.get_local_provider"):
        monkeypatch.setattr(target, lambda: provider, raising=False)
    monkeypatch.setattr("app.gateway.services.resolve_agent_factory", lambda _: _agent_factory(script))

    app = FastAPI()
    # Production order: AuthMiddleware is added first (inner), CSRF last (outer).
    app.add_middleware(AuthMiddleware)
    app.add_middleware(CSRFMiddleware)
    app.include_router(openai_router.router)
    state = app.state
    state.checkpointer = InMemorySaver()
    state.store = InMemoryStore()
    state.thread_store = MemoryThreadMetaStore(state.store)
    state.run_store = RunRepository(sessions)
    state.run_manager = RunManager(store=state.run_store)
    state.run_event_store = MemoryRunEventStore()
    state.stream_bridge = MemoryStreamBridge()
    state.pat_repo = PersonalAccessTokenRepository(sessions)
    host = GatewayAgentRunsHost(app, load_user=provider.get_user)
    state.agent_runs_host = host

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield Gateway(app=app, client=client, script=script, pats=state.pat_repo)
    finally:
        host.close()
        await state.run_manager.shutdown()
        reset_app_config()
        await engine.dispose()


@pytest_asyncio.fixture
async def gateway(monkeypatch, tmp_path) -> AsyncIterator[Gateway]:
    async with running_gateway(monkeypatch, tmp_path) as running:
        yield running


# ── reading responses ────────────────────────────────────────────────────────


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def events(body: str) -> list[Any]:
    """The ``data:`` payloads of a stream, in order; ``[DONE]`` stays a string and comments are skipped."""
    payloads: list[Any] = []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def streamed(body: str) -> tuple[str, str]:
    """The content and the reasoning a stream carried."""
    content, reasoning = [], []
    for event in events(body):
        if isinstance(event, dict) and "choices" in event:
            for choice in event["choices"]:
                content.append(choice["delta"].get("content", ""))
                reasoning.append(choice["delta"].get("reasoning_content", ""))
    return "".join(content), "".join(reasoning)


def error_of(response: httpx.Response) -> dict[str, Any]:
    return response.json()["error"]


def seen(messages: list[BaseMessage], kind: str) -> list[str]:
    """The texts of the messages of one type that the model was given, in order."""
    return [message.content for message in messages if message.type == kind]


async def until(condition, *, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not await condition():
            await asyncio.sleep(0.01)


# ── an answer ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_streamed_answer_is_openai_chunks_that_end_with_done(gateway):
    gateway.script.say("The capital of France is Paris.")

    before = int(time.time())
    response = await gateway.chat([user(FIRST)], stream=True)
    after = int(time.time())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    sent = events(response.text)
    assert sent[-1] == "[DONE]"
    chunks = sent[:-1]
    assert all(chunk["object"] == "chat.completion.chunk" and chunk["model"] == "alpha" for chunk in chunks)
    run = await only_run(gateway)
    assert {chunk["id"] for chunk in chunks} == {f"chatcmpl-{run.run_id}"}
    assert all(before <= chunk["created"] <= after for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert streamed(response.text) == ("The capital of France is Paris.", "")


@pytest.mark.asyncio
async def test_a_non_streaming_answer_is_a_chat_completion(gateway):
    gateway.script.say("Paris.")

    before = int(time.time())
    response = await gateway.chat([user(FIRST)])
    after = int(time.time())

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == "alpha"
    assert body["id"] == f"chatcmpl-{(await only_run(gateway)).run_id}"
    assert before <= body["created"] <= after
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}]
    # A provider that reports no tokens still gets the usage object clients expect.
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_the_reasoning_of_the_model_travels_as_reasoning_content(gateway):
    gateway.script.say(Reply("Paris.", reasoning="It is a well known fact."), Reply("Paris.", reasoning="It is a well known fact."))

    streaming = await gateway.chat([user(FIRST)], stream=True)
    whole = await gateway.chat([user("Another?")])

    assert streamed(streaming.text) == ("Paris.", "It is a well known fact.")
    assert whole.json()["choices"][0]["message"] == {"role": "assistant", "content": "Paris.", "reasoning_content": "It is a well known fact."}


@pytest.mark.asyncio
async def test_token_usage_is_reported_in_the_body_and_when_a_stream_asks_for_it(gateway):
    usage = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
    gateway.script.say(Reply("Paris.", usage=usage), Reply("Paris.", usage=usage), Reply("Paris.", usage=usage))

    whole = await gateway.chat([user("One?")])
    asked = await gateway.chat([user("Two?")], stream=True, stream_options={"include_usage": True})
    not_asked = await gateway.chat([user("Three?")], stream=True)

    assert whole.json()["usage"] == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    final = events(asked.text)[-2]
    assert final["choices"] == [] and final["usage"] == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    assert not any("usage" in event for event in events(not_asked.text) if isinstance(event, dict))


@pytest.mark.asyncio
async def test_a_message_rewritten_after_it_streamed_still_reaches_the_client(gateway):
    original, rewritten = "The task is finished.", "The task is finished. (The loop detector stopped the run.)"
    gateway.script.say(Reply(original, rewritten=rewritten), Reply(original, rewritten=rewritten))

    streaming = await gateway.chat([user(FIRST)], stream=True)
    whole = await gateway.chat([user("Another?")])

    assert streamed(streaming.text)[0] == rewritten
    assert whole.json()["choices"][0]["message"]["content"] == rewritten
    # What the client received is what the thread stored, and the ledger holds that answer.
    for thread in await gateway.threads():
        assert thread["metadata"]["openai_turns"][0]["a"] == text_hash(rewritten)


@pytest.mark.asyncio
async def test_text_before_a_tool_call_is_reasoning_and_the_window_is_the_configured_one(monkeypatch, tmp_path):
    narration = "Let me look that up."
    async with running_gateway(monkeypatch, tmp_path) as gateway:
        gateway.script.say(Reply(narration, tool="web_search"))
        held = streamed((await gateway.chat([user(FIRST)], stream=True)).text)
    (tmp_path / "unheld").mkdir()
    async with running_gateway(monkeypatch, tmp_path / "unheld", narration_window_chars=0) as gateway:
        gateway.script.say(Reply(narration, tool="web_search"))
        unheld = streamed((await gateway.chat([user(FIRST)], stream=True)).text)

    # A stream with no answer re-sends the narration as content, so only the reasoning tells the two apart.
    assert held[0] == unheld[0] == narration
    assert narration in held[1] and narration not in unheld[1]


@pytest.mark.asyncio
async def test_the_progress_of_subagents_reaches_the_client_as_reasoning(gateway):
    gateway.script.say(Reply("All done.", subagents=("task-1",)))

    content, reasoning = streamed((await gateway.chat([user(FIRST)], stream=True)).text)

    assert content == "All done."
    assert "Subagent 1: started" in reasoning and "Subagent 1: done" in reasoning


# ── conversations ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_next_message_continues_the_thread_and_only_it_is_run(gateway):
    gateway.script.say("Paris.", "About 2.1 million people.")

    first = await gateway.chat([user(FIRST)], stream=True)
    second = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)], stream=True)

    assert first.status_code == second.status_code == 200
    thread = await gateway.only_thread()
    # The agent saw the thread's own history, and the resent history was not added to it a second time.
    assert seen(gateway.script.inputs[1], "human") == [FIRST, FOLLOW_UP]
    assert seen(gateway.script.inputs[1], "ai") == ["Paris."]
    turns = thread["metadata"]["openai_turns"]
    assert [turn["u"] for turn in turns] == [text_hash(FIRST), text_hash(FOLLOW_UP)]
    assert [turn["a"] for turn in turns] == [text_hash("Paris."), text_hash("About 2.1 million people.")]
    assert thread["metadata"]["openai_conv"] == text_hash(FIRST)
    # Only a seeded thread is given a title; the web UI's title generation names the others.
    assert not thread["display_name"]


@pytest.mark.asyncio
async def test_a_history_no_thread_holds_starts_a_thread_seeded_with_it(gateway):
    gateway.script.say("About 2.1 million people.")

    before = time.time()
    response = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)])
    after = time.time()

    assert response.status_code == 200
    # The earlier exchange reached the agent as real human and ai messages, in order, before the new message.
    assert [(message.type, message.content) for message in gateway.script.inputs[0]] == [("human", FIRST), ("ai", "Paris."), ("human", FOLLOW_UP)]
    thread = await gateway.only_thread()
    assert thread["display_name"] == FIRST
    turns = thread["metadata"]["openai_turns"]
    assert [turn["u"] for turn in turns] == [text_hash(FIRST), text_hash(FOLLOW_UP)]
    # The imported exchange carries the answer the client held; the new turn carries the one it was sent.
    assert [turn["a"] for turn in turns] == [text_hash("Paris."), text_hash("About 2.1 million people.")]
    # Every entry is dated with the time of the request, the imported exchange as well as the new turn.
    assert all(before <= turn["t"] <= after for turn in turns)


@pytest.mark.asyncio
async def test_a_seeded_thread_is_continued_by_the_next_request(gateway):
    gateway.script.say("About 2.1 million people.", "Yes, by far.")
    history = [user(FIRST), assistant("Paris."), user(FOLLOW_UP)]

    await gateway.chat(history)
    await gateway.chat([*history, assistant("About 2.1 million people."), user("Is it the largest city?")])

    thread = await gateway.only_thread()
    assert len(thread["metadata"]["openai_turns"]) == 3
    assert seen(gateway.script.inputs[1], "human") == [FIRST, FOLLOW_UP, "Is it the largest city?"]


@pytest.mark.asyncio
async def test_the_title_of_a_seeded_thread_is_cut_to_the_configured_length(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, title_max_chars=20) as gateway:
        await gateway.chat([user("Tell me about the history of the city of Paris"), assistant("Gladly."), user("Start")])

        title = (await gateway.only_thread())["display_name"]

        assert len(title) == 20 and title == "Tell me about the..."


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_message_after_a_failed_answer_continues_the_same_thread(gateway, monkeypatch, stream):
    bridge = gateway.app.state.stream_bridge
    publish = bridge.publish

    async def finish_slowly(run_id, event, *args, **kwargs):
        await publish(run_id, event, *args, **kwargs)
        if event == "error":
            # The run's own wrap-up (checkpoint, journal, final status) is still ahead of it when its error frame is out.
            await asyncio.sleep(0.3)

    monkeypatch.setattr(bridge, "publish", finish_slowly)
    gateway.script.say(RuntimeError("the provider is down"), "Paris.")

    failed = await gateway.chat([user(FIRST)], stream=stream)
    retried = await gateway.chat([user(FIRST), user("Please try again")])

    # The failure is reported once the run is over, so the retry never meets the run it just saw fail.
    assert (failed.status_code, retried.status_code) == (200 if stream else 502, 200)
    assert (events(failed.text)[-1]["error"] if stream else error_of(failed))["code"] == "run_error"
    thread = await gateway.only_thread()
    assert [turn["a"] for turn in thread["metadata"]["openai_turns"]] == [None, text_hash("Paris.")]


@pytest.mark.asyncio
async def test_the_same_history_from_another_user_is_another_conversation(gateway):
    gateway.script.say("Paris.", "Madrid.")
    other = await gateway.token("user-2")

    await gateway.chat([user(FIRST)])
    await gateway.chat([user(FIRST), assistant("Paris."), user("And Spain?")], token=other)

    assert len(await gateway.threads("user-1")) == 1
    assert len(await gateway.threads("user-2")) == 1
    assert len((await gateway.only_thread("user-1"))["metadata"]["openai_turns"]) == 1


@pytest.mark.asyncio
async def test_an_edited_earlier_message_starts_a_new_thread(gateway):
    gateway.script.say("Paris.", "Madrid.")

    await gateway.chat([user(FIRST)])
    await gateway.chat([user("Capital of Spain?"), assistant("Paris."), user("Thanks")])

    assert len(await gateway.threads()) == 2


@pytest.mark.asyncio
async def test_sending_the_same_message_again_seeds_a_new_thread(gateway):
    gateway.script.say("Paris.", "Paris.")

    await gateway.chat([user(FIRST)])
    await gateway.chat([user(FIRST)])

    assert len(await gateway.threads()) == 2


@pytest.mark.asyncio
async def test_a_ledger_that_cannot_be_written_never_fails_the_request(gateway, monkeypatch):
    gateway.script.say("Paris.", "Paris again.")

    async def broken(*args, **kwargs):
        raise RuntimeError("the store is down")

    monkeypatch.setattr(gateway.app.state.thread_store, "update_metadata", broken)
    first = await gateway.chat([user(FIRST)])
    second = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)])

    assert first.status_code == second.status_code == 200
    # Nothing was recorded, so the second request could not be matched and started a thread of its own.
    assert len(await gateway.threads()) == 2


# ── limits ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_history_too_large_to_import_is_refused_and_a_continuing_one_is_not(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, max_ingest_messages=3) as gateway:
        gateway.script.say("One.", "Two.", "Three.")
        first = await gateway.chat([user("a")])
        second = await gateway.chat([user("a"), assistant("One."), user("b")])
        # Four messages are resent: they continue the stored conversation, so only the newest is ingested.
        third = await gateway.chat([user("a"), assistant("One."), user("b"), assistant("Two."), user("c")])
        cold = await gateway.chat([user("x"), assistant("1"), user("y"), assistant("2"), user("z")])

        assert first.status_code == second.status_code == third.status_code == 200
        assert cold.status_code == 400
        assert error_of(cold)["code"] == "context_length_exceeded" and error_of(cold)["param"] == "messages"
        assert len(await gateway.threads()) == 1


@pytest.mark.asyncio
async def test_a_history_with_too_much_text_to_import_is_refused(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, max_ingest_chars=20) as gateway:
        response = await gateway.chat([user("x" * 12), assistant("y" * 12), user("z")])

        assert response.status_code == 400 and error_of(response)["code"] == "context_length_exceeded"
        assert await gateway.threads() == []


# ── refused requests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "code", "param"),
    [
        ("{not json", "invalid_request", None),
        ("[1, 2]", "invalid_request", None),
        ({"messages": [user("hi")]}, "invalid_request", "model"),
        ({"model": "alpha"}, "invalid_request", "messages"),
        ({"model": "alpha", "messages": []}, "invalid_request", "messages"),
        ({"model": "alpha", "messages": [user("hi")], "tools": [{"type": "function"}]}, "unsupported_parameter", "tools"),
        ({"model": "alpha", "messages": [user("hi")], "n": 2}, "unsupported_value", "n"),
        ({"model": "alpha", "messages": [user("hi")], "stream": "yes"}, "invalid_request", "stream"),
        ({"model": "alpha", "messages": [{"role": "system", "content": "Be brief"}, user("hi")]}, "invalid_message", "messages[0].role"),
        ({"model": "alpha", "messages": [user("hi"), assistant("hello")]}, "invalid_message", "messages[1]"),
        ({"model": "alpha", "messages": [user("   ")]}, "invalid_message", "messages[0]"),
        ({"model": "alpha", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]}, "invalid_message", "messages[0].content[0].type"),
    ],
    ids=["not-json", "not-an-object", "no-model", "no-messages", "empty-messages", "tools", "n", "stream-type", "system-message", "last-is-assistant", "blank-user-message", "image"],
)
async def test_a_request_the_api_cannot_serve_is_a_400_in_the_openai_envelope(gateway, payload, code, param):
    response = await gateway.post(payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert error_of(response)["code"] == code and error_of(response)["param"] == param
    assert error_of(response)["type"] == "invalid_request_error"
    assert 0 < len(error_of(response)["message"]) <= 400
    if payload == "{not json":
        assert error_of(response)["message"] == "The request body must be valid JSON."
    if payload == "[1, 2]":
        assert error_of(response)["message"] == "The request body must be a JSON object."
    # A refused request leaves no trace behind.
    assert await gateway.threads() == []
    assert gateway.script.inputs == []


@pytest.mark.asyncio
async def test_a_model_the_listing_does_not_have_is_a_404_that_names_the_ones_it_has(gateway):
    for model in ("beta", "alpha:", "alpha:thinking:extra", "ALPHA"):
        response = await gateway.chat([user("hi")], model=model)

        assert response.status_code == 404
        assert error_of(response)["code"] == "model_not_found" and error_of(response)["param"] == "model"
        assert "alpha:ultra" in error_of(response)["message"]
    assert gateway.script.inputs == []


# ── a failure before the run starts ──────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status", "code", "logged", "traced"),
    [
        (RuntimeError("db unreachable at postgres://user:password@db"), 500, "internal_error", "A chat completion failed before it started", True),
        (HTTPException(status_code=503, detail="queue full: password\nFORGED log line"), 500, "internal_error", "refused by the Gateway with HTTP 503", False),
        # The caller names no thread, so a thread the Gateway cannot find is the server's own failure, not a 404 for the caller.
        (HTTPException(status_code=404, detail="Thread t-1 not found: password"), 500, "internal_error", "refused by the Gateway with HTTP 404", False),
        (HTTPException(status_code=409, detail="busy: password"), 409, "conversation_busy", None, False),
    ],
    ids=["unexpected", "refused-by-the-gateway", "not-found", "busy"],
)
async def test_a_failure_before_the_run_starts_is_answered_in_the_envelope_and_leaks_nothing(gateway, monkeypatch, caplog, failure, status, code, logged, traced):
    async def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(openai_router, "start_run", refuse)

    with caplog.at_level(logging.INFO, logger=openai_router.logger.name):
        response = await gateway.chat([user("hi")])

    assert response.status_code == status and error_of(response)["code"] == code
    assert "password" not in response.text and "postgres" not in response.text
    # The cause goes to the log, where an operator looks for it; a busy conversation is the caller's doing and is not an error.
    records = [record for record in caplog.records if record.levelno >= logging.ERROR and record.name == openai_router.logger.name]
    errors = [record.getMessage() for record in records]
    assert errors == ([] if logged is None else [errors[0]]) and (logged is None or logged in errors[0])
    assert all("\n" not in message for message in errors)
    # A bug is logged with its traceback; a refusal is logged by its status alone.
    assert all((record.exc_info is not None) == traced for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_a_refusal_by_the_auth_layer_keeps_its_own_shape(gateway, monkeypatch, status):
    async def refuse(*args, **kwargs):
        raise HTTPException(status_code=status, detail="refused by the gateway")

    monkeypatch.setattr(openai_router, "start_run", refuse)

    response = await gateway.chat([user("hi")])

    assert response.status_code == status and response.json() == {"detail": "refused by the gateway"}


# ── who may ask ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_token_the_gateway_does_not_know_is_a_401_in_the_gateway_shape(gateway):
    for token in (generate_pat_token(), "not-a-token"):
        response = await gateway.chat([user("hi")], token=token)

        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid token"}
    assert gateway.script.inputs == []


@pytest.mark.asyncio
async def test_a_request_with_no_credential_is_stopped_before_the_router(gateway):
    response = await gateway.client.post(URL, json={"model": "alpha", "messages": [user("hi")]})

    # The CSRF layer answers a credential-less POST before authentication does, for every Gateway route alike.
    assert response.status_code == 403 and "CSRF" in response.json()["detail"]
    assert gateway.script.inputs == [] and await gateway.threads() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", CHAT_SCOPES)
async def test_a_token_needs_every_scope_the_endpoint_uses(gateway, missing):
    token = await gateway.token(scopes=[scope for scope in CHAT_SCOPES if scope != missing])

    response = await gateway.chat([user("hi")], token=token)

    assert response.status_code == 403
    assert response.json() == {"detail": f"Permission denied: {missing}"}
    assert await gateway.threads() == []


@pytest.mark.asyncio
async def test_a_token_without_write_access_cannot_continue_a_thread(gateway):
    gateway.script.say("Paris.", "About 2.1 million people.")
    await gateway.chat([user(FIRST)])
    read_only = await gateway.token(scopes=[scope for scope in CHAT_SCOPES if scope != "threads:write"])

    response = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)], token=read_only)

    assert response.status_code == 403 and response.json() == {"detail": "Permission denied: threads:write"}
    # Nothing ran, and the thread's ledger is as the first request left it.
    assert len(gateway.script.inputs) == 1
    assert [turn["u"] for turn in (await gateway.only_thread())["metadata"]["openai_turns"]] == [text_hash(FIRST)]


class _ModelPolicy:
    """An authorization provider that hides some models from the listing and refuses others when they are used."""

    name = "model-policy"

    def __init__(self, *, hidden: set[str], refused: set[str]) -> None:
        self.hidden, self.refused = hidden, refused

    def authorize(self, request):
        allowed = not (request.resource == "model" and request.target in self.refused)
        return AuthzDecision(allow=allowed, reasons=[AuthzReason(code="authz.allowed" if allowed else "authz.denied")])

    async def aauthorize(self, request):
        return self.authorize(request)

    def filter_resources(self, principal, resource_type, candidates):
        return [candidate for candidate in candidates if candidate not in self.hidden]


@pytest.mark.asyncio
async def test_a_model_hidden_from_the_caller_cannot_be_run_and_one_refused_at_use_is_a_403(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, models=[_model("alpha"), _model("beta"), _model("gamma")]) as gateway:
        config = AuthorizationConfig(enabled=True, fail_closed=True, default_role="user")
        monkeypatch.setattr("app.gateway.authz._get_route_authorization_config", lambda: config)
        monkeypatch.setattr("app.gateway.authz._get_cached_route_provider", lambda _: _ModelPolicy(hidden={"beta"}, refused={"gamma"}))
        gateway.script.say("Paris.")

        hidden = await gateway.chat([user("hi")], model="beta")
        refused = await gateway.chat([user("hi")], model="gamma")
        refused_variant = await gateway.chat([user("hi")], model="gamma:pro")
        allowed = await gateway.chat([user("hi")], model="alpha")

        assert hidden.status_code == 404 and error_of(hidden)["code"] == "model_not_found"
        for response in (refused, refused_variant):
            assert response.status_code == 403 and response.json() == {"detail": "Model 'gamma' is not available for your role"}
        assert allowed.status_code == 200
        # Neither refusal started a run or left a thread behind.
        assert len(gateway.script.inputs) == 1 and len(await gateway.threads()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_a_model_is_authorized_for_the_kind_of_caller_the_gateway_says_it_is(gateway, monkeypatch, internal):
    seen: list[tuple[str, bool]] = []

    def authorize(user, model_name, *, is_internal, app_config):
        seen.append((model_name, is_internal))

    monkeypatch.setattr(openai_router, "authorize_model_use", authorize)
    monkeypatch.setattr(openai_router, "_is_internal_caller", lambda request, user: internal)
    gateway.script.say("Paris.")

    response = await gateway.chat([user("hi")], model="alpha:pro")

    assert response.status_code == 200
    assert seen == [("alpha", internal)]


@pytest.mark.asyncio
async def test_the_run_asks_for_exactly_the_streams_the_translator_reads(gateway, monkeypatch):
    requested = []
    real_start_run = openai_router.start_run

    async def spy(body, *args, **kwargs):
        requested.append(body)
        return await real_start_run(body, *args, **kwargs)

    monkeypatch.setattr(openai_router, "start_run", spy)
    gateway.script.say("Paris.")

    await gateway.chat([user("hi")])

    assert [sorted(body.stream_mode) for body in requested] == [["custom", "messages-tuple"]]


# ── what the response reports ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parameters_that_are_ignored_are_named_in_a_response_header(gateway):
    gateway.script.say("Paris.", "Paris.", "Paris.")

    ignored = await gateway.chat([{"role": "user", "content": FIRST, "name": "ajay"}], temperature=0.2, max_tokens=50, stream_options={"include_usage": True, "extra": 1})
    streaming = await gateway.chat([user("Two?")], stream=True, top_p=0.9)
    nothing = await gateway.chat([user("Three?")])

    assert ignored.headers["x-deerflow-ignored-params"] == "temperature, max_tokens, stream_options.extra, messages[].name"
    assert streaming.headers["x-deerflow-ignored-params"] == "top_p"
    assert "x-deerflow-ignored-params" not in nothing.headers


@pytest.mark.asyncio
async def test_a_client_cannot_steer_the_run_through_fields_the_api_ignores(gateway):
    gateway.script.say("Paris.")

    response = await gateway.chat([user(FIRST)], context={"model_name": "other"}, metadata={"deerflow_origin": {"kind": "schedule"}}, config={"recursion_limit": 1})

    assert response.status_code == 200
    assert response.headers["x-deerflow-ignored-params"] == "context, metadata, config"
    assert gateway.script.contexts[0]["model_name"] == "alpha"


# ── modes ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "thinking", "plan", "subagents"),
    [("alpha", True, False, False), ("alpha:flash", False, False, False), ("alpha:thinking", True, False, False), ("alpha:pro", True, True, False), ("alpha:ultra", True, True, True)],
)
async def test_the_model_id_chooses_what_the_run_switches_on(gateway, model, thinking, plan, subagents):
    gateway.script.say("Paris.")

    response = await gateway.chat([user(FIRST)], model=model)

    assert response.status_code == 200 and response.json()["model"] == model
    context = gateway.script.contexts[0]
    assert (context["model_name"], context["thinking_enabled"], context["is_plan_mode"], context["subagent_enabled"]) == ("alpha", thinking, plan, subagents)
    assert "reasoning_effort" not in context


@pytest.mark.asyncio
async def test_a_streamed_answer_names_the_model_id_it_was_asked_for_with_its_mode(gateway):
    gateway.script.say("Paris.")

    response = await gateway.chat([user(FIRST)], model="alpha:pro", stream=True)

    chunks = events(response.text)[:-1]
    assert chunks and {chunk["model"] for chunk in chunks} == {"alpha:pro"}


@pytest.mark.asyncio
async def test_the_bare_name_means_the_configured_default_mode(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, default_mode="pro") as gateway:
        gateway.script.say("Paris.")

        await gateway.chat([user(FIRST)])

        assert (gateway.script.contexts[0]["is_plan_mode"], gateway.script.contexts[0]["subagent_enabled"]) == (True, False)


@pytest.mark.asyncio
async def test_a_mode_the_operator_left_out_is_not_a_model(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, modes=["flash", "thinking"], default_mode="flash") as gateway:
        response = await gateway.chat([user(FIRST)], model="alpha:ultra")

        assert response.status_code == 404


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_a_model_that_has_the_control(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, models=[_model("glm", effort=["low", "medium", "high"])]) as gateway:
        gateway.script.say("A.", "B.", "C.", "D.")

        defaulted = await gateway.chat([user("1")], model="glm:pro")
        chosen = await gateway.chat([user("2")], model="glm:pro", reasoning_effort="high")
        thinking_off = await gateway.chat([user("3")], model="glm:flash", reasoning_effort="high")
        invalid = await gateway.chat([user("4")], model="glm:pro", reasoning_effort="extreme")

        assert [context.get("reasoning_effort") for context in gateway.script.contexts] == ["medium", "high", None]
        assert thinking_off.headers["x-deerflow-ignored-params"] == "reasoning_effort"
        assert "x-deerflow-ignored-params" not in defaulted.headers and "x-deerflow-ignored-params" not in chosen.headers
        assert invalid.status_code == 400
        assert error_of(invalid)["code"] == "unsupported_value" and error_of(invalid)["param"] == "reasoning_effort"
        assert "low, medium, high" in error_of(invalid)["message"]


@pytest.mark.asyncio
async def test_the_reasoning_contract_of_a_model_decides_which_modes_it_runs_in(monkeypatch, tmp_path):
    models = [_model("plain", thinking="unsupported"), _model("always", thinking="required")]
    async with running_gateway(monkeypatch, tmp_path, models=models) as gateway:
        gateway.script.say("A.", "B.")

        plain = await gateway.chat([user("1")], model="plain")
        always = await gateway.chat([user("2")], model="always")
        flash = await gateway.chat([user("3")], model="always:flash")

        assert plain.status_code == always.status_code == 200 and flash.status_code == 404
        assert [context["thinking_enabled"] for context in gateway.script.contexts] == [False, True]


@pytest.mark.asyncio
async def test_an_alias_of_the_model_is_a_valid_reasoning_effort(monkeypatch, tmp_path):
    async with running_gateway(monkeypatch, tmp_path, models=[_model("glm", effort=["low", "high"], aliases={"max": "high"})]) as gateway:
        gateway.script.say("A.")

        response = await gateway.chat([user("1")], model="glm:pro", reasoning_effort="max")

        assert response.status_code == 200 and gateway.script.contexts[0]["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_reasoning_effort_is_ignored_for_a_model_without_the_control(gateway):
    gateway.script.say("Paris.")

    response = await gateway.chat([user(FIRST)], model="alpha:pro", reasoning_effort="high")

    assert response.status_code == 200
    assert response.headers["x-deerflow-ignored-params"] == "reasoning_effort"
    assert "reasoning_effort" not in gateway.script.contexts[0]


@pytest.mark.asyncio
async def test_a_set_aside_reasoning_effort_is_named_first_so_the_cap_cannot_drop_it(gateway):
    gateway.script.say("Paris.")
    unknown = {f"extra_{index:02d}": index for index in range(MAX_IGNORED_NAMES + 10)}

    response = await gateway.chat([user(FIRST)], model="alpha:pro", reasoning_effort="high", **unknown)

    names = response.headers["x-deerflow-ignored-params"].split(", ")
    assert response.status_code == 200
    assert names[0] == "reasoning_effort" and len(names) == MAX_IGNORED_NAMES


# ── a busy conversation ──────────────────────────────────────────────────────


async def hold_the_first_run(gateway: Gateway, release: asyncio.Event, *, stream: bool) -> asyncio.Task:
    """Start a first message whose run stays open until ``release`` is set, and wait until its turn is recorded."""

    async def held():
        await release.wait()
        return "Paris."

    gateway.script.say(held)
    task = asyncio.create_task(gateway.chat([user(FIRST)], stream=stream))
    await asyncio.wait_for(gateway.script.started.wait(), timeout=5)

    async def recorded() -> bool:
        threads = await gateway.threads()
        return bool(threads) and bool(threads[0]["metadata"].get("openai_turns"))

    await until(recorded)
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_a_message_sent_while_the_conversation_runs_is_a_409_and_changes_nothing(gateway, stream):
    release = asyncio.Event()
    first = await hold_the_first_run(gateway, release, stream=stream)

    busy = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)], stream=stream)
    release.set()
    answered = await first

    assert busy.status_code == 409
    assert error_of(busy) == {
        "message": "This conversation is still running. Wait for it to finish, or open the thread in the web UI to stop it.",
        "type": "invalid_request_error",
        "param": None,
        "code": "conversation_busy",
    }
    assert answered.status_code == 200
    thread = await gateway.only_thread()
    assert [turn["u"] for turn in thread["metadata"]["openai_turns"]] == [text_hash(FIRST)]
    assert len(gateway.script.inputs) == 1


# ── a run that does not finish ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stream_whose_run_fails_ends_with_an_error_and_no_done(gateway):
    gateway.script.say(RuntimeError("secret provider detail https://provider.example/v1 key=sk-123"))

    response = await gateway.chat([user(FIRST)], stream=True)

    assert response.status_code == 200
    sent = events(response.text)
    assert "[DONE]" not in sent
    assert sent[-1]["error"]["code"] == "run_error" and sent[-1]["error"]["type"] == "server_error"
    assert "secret" not in response.text and "sk-123" not in response.text
    # The answer was never delivered, so the ledger does not hold one.
    assert [turn["a"] for turn in (await gateway.only_thread())["metadata"]["openai_turns"]] == [None]


@pytest.mark.asyncio
async def test_a_non_streaming_run_that_fails_is_a_502_with_the_same_code(gateway):
    gateway.script.say(RuntimeError("the provider is down"))

    response = await gateway.chat([user(FIRST)])

    assert response.status_code == 502
    assert error_of(response)["code"] == "run_error" and error_of(response)["type"] == "server_error"
    assert "provider is down" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_a_run_that_someone_else_stops_is_reported_as_interrupted(gateway, stream):
    async def never():
        await asyncio.Event().wait()

    gateway.script.say(never)
    request = asyncio.create_task(gateway.chat([user(FIRST)], stream=stream))
    await asyncio.wait_for(gateway.script.started.wait(), timeout=5)

    async def running() -> bool:
        return bool(await gateway.app.state.run_manager.list_by_thread((await gateway.only_thread())["thread_id"]))

    await until(running)
    run = (await gateway.app.state.run_manager.list_by_thread((await gateway.only_thread())["thread_id"]))[0]
    await gateway.app.state.run_manager.cancel(run.run_id, action="interrupt")
    response = await request

    assert gateway.script.cancelled.is_set()
    if stream:
        sent = events(response.text)
        assert "[DONE]" not in sent and sent[-1]["error"]["code"] == "run_interrupted"
    else:
        assert response.status_code == 409 and error_of(response)["code"] == "run_interrupted"


# ── when the client leaves ───────────────────────────────────────────────────


async def chat_and_hang_up(gateway: Gateway, messages: list[dict[str, Any]], *, stream: bool, when: Callable[[], Awaitable[bool]]) -> None:
    """Send a chat completion straight to the app and drop the connection once ``when`` is true, as a client that goes away does."""
    body = json.dumps({"model": "alpha", "messages": messages, "stream": stream}).encode()
    headers = [(b"host", b"test"), (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"authorization", f"Bearer {await gateway.token()}".encode())]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": URL,
        "raw_path": URL.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    gone, delivered = asyncio.Event(), False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    call = asyncio.create_task(gateway.app(scope, receive, send))
    await until(when)
    gone.set()
    await asyncio.wait_for(call, timeout=5)


async def only_run(gateway: Gateway):
    (run,) = await gateway.app.state.run_manager.list_by_thread((await gateway.only_thread())["thread_id"])
    return run


def held_until(release: asyncio.Event, text: str = "Paris."):
    async def held():
        await release.wait()
        return text

    return held


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_a_client_that_hangs_up_stops_the_run(gateway, monkeypatch, stream):
    monkeypatch.setattr(openai_router, "RunStream", partial(RunStream, keepalive_seconds=0.05))
    gateway.script.say(held_until(asyncio.Event()))

    async def working() -> bool:
        return gateway.script.started.is_set()

    await chat_and_hang_up(gateway, [user(FIRST)], stream=stream, when=working)

    await asyncio.wait_for(gateway.script.cancelled.wait(), timeout=5)

    async def stopped() -> bool:
        return (await only_run(gateway)).status == RunStatus.interrupted

    await until(stopped)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_a_client_that_hangs_up_leaves_the_run_going_when_the_operator_chose_that(monkeypatch, tmp_path, stream):
    async with running_gateway(monkeypatch, tmp_path, on_disconnect="continue") as gateway:
        monkeypatch.setattr(openai_router, "RunStream", partial(RunStream, keepalive_seconds=0.05))
        release = asyncio.Event()
        gateway.script.say(held_until(release))

        async def working() -> bool:
            return gateway.script.started.is_set()

        await chat_and_hang_up(gateway, [user(FIRST)], stream=stream, when=working)
        assert not gateway.script.cancelled.is_set()
        release.set()

        async def finished() -> bool:
            return (await only_run(gateway)).status == RunStatus.success

        await until(finished)
        assert not gateway.script.cancelled.is_set()


@pytest.mark.asyncio
async def test_a_client_that_hangs_up_while_a_failed_run_ends_does_not_stop_it(gateway, monkeypatch, caplog):
    holding = asyncio.Event()

    class HoldingRunStream(RunStream):
        async def _wait_for_end(self) -> None:
            holding.set()
            await super()._wait_for_end()

    monkeypatch.setattr(openai_router, "RunStream", HoldingRunStream)
    gateway.script.finishing = asyncio.Event()
    gateway.script.say(Reply("The model provider failed.", fallback=True))

    async def waiting() -> bool:
        return holding.is_set()

    with caplog.at_level(logging.INFO, logger=openai_router.logger.name):
        await chat_and_hang_up(gateway, [user(FIRST)], stream=True, when=waiting)
        gateway.script.finishing.set()

        async def over() -> bool:
            return (await only_run(gateway)).status not in (RunStatus.pending, RunStatus.running)

        await until(over)

    # The run had already failed when its client left, so it keeps its error instead of being stopped.
    assert (await only_run(gateway)).status is RunStatus.error
    assert any("outcome=failed: run_error" in record.getMessage() for record in caplog.records)


class _Manager:
    """The run manager as far as the router uses it: it records the runs it is asked to cancel."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True


class _Answers:
    """The ledger entry of a turn: it records the answers it is given."""

    def __init__(self) -> None:
        self.given: list[str] = []

    async def answered(self, content: str) -> None:
        self.given.append(content)


class _Stream:
    """A run stream that sends the steps it is given and then ends the way a test says; ``hang`` keeps it open instead."""

    def __init__(self, steps=(), *, ended="open", failure=None, settled=None, hang=False) -> None:
        self._steps, self._hang = list(steps), hang
        self.ended, self.failure, self.settled = ended, failure, settled
        self.closed = False

    async def steps(self):
        try:
            for step in self._steps:
                yield step
            if self._hang:
                await asyncio.Event().wait()
        finally:
            self.closed = True


class _Hangup:
    """The watcher for a client that went away, as far as the router uses it."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def sent_by(stream, *, status=RunStatus.running, on_disconnect=DisconnectMode.cancel, store_only=False, answers=None, hangup=None, ignored=None):
    record = SimpleNamespace(run_id="run-1", thread_id="thread-1", status=status, on_disconnect=on_disconnect, store_only=store_only)
    translator = SimpleNamespace(content_text="The answer", reasoning_text="Because.")
    return openai_router._Sent(record, translator, stream, hangup or _Hangup(), answers, "user-1", "alpha (thinking)", ignored)


def a_chunk(**delta: str) -> Chunk:
    return Chunk({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ended", "failure", "status", "on_disconnect", "store_only", "cancels", "outcome"),
    [
        pytest.param("open", None, RunStatus.running, DisconnectMode.cancel, False, True, "client left", id="closed-while-running"),
        pytest.param("disconnected", None, RunStatus.running, DisconnectMode.cancel, False, True, "client left", id="disconnected-while-running"),
        pytest.param("disconnected", None, RunStatus.pending, DisconnectMode.cancel, False, True, "client left", id="disconnected-before-it-started"),
        pytest.param("disconnected", None, RunStatus.running, DisconnectMode.continue_, False, False, "client left", id="the-run-is-to-continue"),
        pytest.param("disconnected", None, RunStatus.success, DisconnectMode.cancel, False, False, "client left", id="the-run-already-finished"),
        pytest.param("disconnected", None, RunStatus.interrupted, DisconnectMode.cancel, False, False, "client left", id="the-run-was-already-stopped"),
        pytest.param("disconnected", None, RunStatus.running, DisconnectMode.cancel, True, False, "client left", id="a-stored-only-run"),
        pytest.param("failed", STREAM_GAP, RunStatus.running, DisconnectMode.cancel, False, False, "failed: stream_gap", id="the-stream-was-lost-not-the-client"),
        pytest.param("done", None, RunStatus.running, DisconnectMode.cancel, False, False, "done", id="the-answer-was-delivered"),
    ],
)
async def test_a_run_is_cancelled_only_for_a_client_that_left_while_it_was_running(caplog, ended, failure, status, on_disconnect, store_only, cancels, outcome):
    manager, hangup = _Manager(), _Hangup()
    sent = sent_by(_Stream(ended=ended, failure=failure), status=status, on_disconnect=on_disconnect, store_only=store_only, hangup=hangup)

    with caplog.at_level(logging.INFO, logger=openai_router.logger.name):
        await openai_router._finish(sent, manager)

    assert manager.cancelled == (["run-1"] if cancels else [])
    assert [record.getMessage() for record in caplog.records] == [f"Chat completion ended: user=user-1 thread=thread-1 run=run-1 model=alpha (thinking) ignored=none outcome={outcome}"]
    # Whatever the verdict, nothing keeps waiting for the client once the response is over.
    assert hangup.stopped


@pytest.mark.asyncio
async def test_the_log_line_names_the_parameters_that_were_ignored(caplog):
    sent = sent_by(_Stream(ended="done"), ignored="temperature, top_p")

    with caplog.at_level(logging.INFO, logger=openai_router.logger.name):
        await openai_router._finish(sent, _Manager())

    assert [record.getMessage() for record in caplog.records] == ["Chat completion ended: user=user-1 thread=thread-1 run=run-1 model=alpha (thinking) ignored=temperature, top_p outcome=done"]


@pytest.mark.asyncio
async def test_a_consumer_that_closes_the_response_takes_the_run_down():
    manager, stream = _Manager(), _Stream([a_chunk(role="assistant")], hang=True)
    frames = openai_router._frames(sent_by(stream), manager)

    await anext(frames)
    await frames.aclose()

    assert manager.cancelled == ["run-1"]
    assert stream.closed


@pytest.mark.asyncio
async def test_a_response_whose_task_is_cancelled_still_takes_the_run_down():
    manager, stream = _Manager(), _Stream(hang=True)

    async def consume():
        async for _ in openai_router._frames(sent_by(stream), manager):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.cancelled == ["run-1"]
    assert stream.closed


@pytest.mark.asyncio
async def test_a_task_cancelled_again_while_the_run_is_being_cancelled_still_finishes_the_cancel():
    release, cancelling = asyncio.Event(), asyncio.Event()

    class SlowManager(_Manager):
        async def cancel(self, run_id: str) -> bool:
            cancelling.set()
            await release.wait()
            return await super().cancel(run_id)

    manager = SlowManager()

    async def consume():
        async for _ in openai_router._frames(sent_by(_Stream(hang=True)), manager):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(cancelling.wait(), timeout=5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.cancelled == ["run-1"]


# ── noticing a client that went away ─────────────────────────────────────────


class _Channel:
    """A request whose receive channel delivers the messages a test hands it, one at a time."""

    def __init__(self) -> None:
        self.messages: asyncio.Queue[Any] = asyncio.Queue()

    async def receive(self) -> dict[str, Any]:
        message = await self.messages.get()
        if isinstance(message, Exception):
            raise message
        return message


async def settle_loop() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_client_is_gone_once_the_receive_channel_says_so():
    channel = _Channel()
    hangup = openai_router._Hangup(channel)
    await settle_loop()
    assert not await hangup.occurred()

    channel.messages.put_nowait({"type": "http.request", "body": b"", "more_body": False})
    await settle_loop()
    assert not await hangup.occurred()

    channel.messages.put_nowait({"type": "http.disconnect"})
    await settle_loop()
    assert await hangup.occurred()


@pytest.mark.asyncio
async def test_a_watch_that_failed_does_not_claim_the_client_left():
    channel = _Channel()
    hangup = openai_router._Hangup(channel)
    channel.messages.put_nowait(RuntimeError("the channel broke"))
    await settle_loop()

    assert not await hangup.occurred()


@pytest.mark.asyncio
async def test_a_stopped_watch_reads_nothing_more():
    channel = _Channel()
    hangup = openai_router._Hangup(channel)
    await settle_loop()

    hangup.stop()
    channel.messages.put_nowait({"type": "http.disconnect"})
    await settle_loop()

    assert not await hangup.occurred() and channel.messages.qsize() == 1


# ── the body of a stream ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stream_records_its_answer_in_the_ledger_before_it_sends_done():
    answers, frames = _Answers(), []

    async for frame in openai_router._frames(sent_by(_Stream([KEEPALIVE, a_chunk(content="The answer"), DONE], ended="done"), answers=answers), _Manager()):
        frames.append(frame)
        if frame == "data: [DONE]\n\n":
            assert answers.given == ["The answer"]

    assert frames[0] == ": keepalive\n\n"
    assert frames[1].startswith("data: {") and frames[-1] == "data: [DONE]\n\n"
    assert answers.given == ["The answer"]


@pytest.mark.asyncio
async def test_a_stream_without_a_ledger_entry_still_ends_with_done():
    frames = [frame async for frame in openai_router._frames(sent_by(_Stream([a_chunk(content="Hi"), DONE], ended="done")), _Manager())]

    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_a_stream_that_fails_records_no_answer_and_sends_the_error():
    answers, manager = _Answers(), _Manager()
    stream = _Stream([Failed(RUN_ERROR)], ended="failed", failure=RUN_ERROR)

    frames = [frame async for frame in openai_router._frames(sent_by(stream, answers=answers), manager)]

    assert answers.given == [] and manager.cancelled == []
    assert json.loads(frames[-1].removeprefix("data: "))["error"]["code"] == "run_error"


# ── a whole answer ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_whole_answer_carries_the_text_the_reasoning_and_the_usage():
    answers = _Answers()
    stream = _Stream(ended="done", settled=Settled(usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)))

    response = await openai_router._answer(sent_by(stream, answers=answers), _Manager(), model="alpha", created=100)

    body = json.loads(response.body)
    assert response.status_code == 200
    assert (body["id"], body["created"], body["model"]) == ("chatcmpl-run-1", 100, "alpha")
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "The answer", "reasoning_content": "Because."}
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert answers.given == ["The answer"]


@pytest.mark.asyncio
async def test_a_whole_answer_whose_usage_is_unknown_reports_zeros():
    response = await openai_router._answer(sent_by(_Stream(ended="done")), _Manager(), model="alpha", created=1)

    assert json.loads(response.body)["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_a_whole_answer_to_a_failed_run_raises_the_error_for_its_code():
    answers, manager = _Answers(), _Manager()

    with pytest.raises(OpenAIError) as caught:
        await openai_router._answer(sent_by(_Stream(ended="failed", failure=RUN_ERROR), answers=answers), manager, model="alpha", created=1)

    assert (caught.value.status_code, caught.value.code) == (502, "run_error")
    assert answers.given == [] and manager.cancelled == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("on_disconnect", "cancels"), [(DisconnectMode.cancel, True), (DisconnectMode.continue_, False)])
async def test_a_whole_answer_nobody_waits_for_is_dropped_and_the_run_follows_the_policy(on_disconnect, cancels):
    answers, manager = _Answers(), _Manager()

    response = await openai_router._answer(sent_by(_Stream(ended="disconnected"), on_disconnect=on_disconnect, answers=answers), manager, model="alpha", created=1)

    assert response.status_code == 499 and answers.given == []
    assert manager.cancelled == (["run-1"] if cancels else [])


# ── logging ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_request_is_logged_by_its_ids_and_never_by_its_text(gateway, caplog):
    question, answer = "My password is hunter2-question", "The vault code is 0451-answer"
    gateway.script.say(answer, answer, RuntimeError(answer))

    with caplog.at_level(logging.DEBUG, logger="app.gateway"):
        await gateway.chat([user(question)], stream=True)
        await gateway.chat([user("Another question")], temperature=0.5)
        await gateway.chat([user("A third question")])

    lines = [record.getMessage() for record in caplog.records if record.name == openai_router.logger.name]
    assert [line.partition(" outcome=")[2] for line in lines] == ["done", "done", "failed: run_error"]
    assert [re.search(r" ignored=(.*) outcome=", line).group(1) for line in lines] == ["none", "temperature", "none"]
    assert all(line.startswith("Chat completion ended: user=user-1 thread=") and " model=alpha (thinking) " in line for line in lines)
    assert {re.search(r"thread=(\S+)", line).group(1) for line in lines} == {thread["thread_id"] for thread in await gateway.threads()}
    assert len({re.search(r"run=(\S+)", line).group(1) for line in lines}) == 3
    # Nothing the router or its package logged carries a word of the conversation.
    format_record = logging.Formatter().format
    written = [format_record(record) for record in caplog.records if record.name.startswith(("app.gateway.openai_compat", openai_router.logger.name))]
    for secret in (question, answer, "hunter2", "0451"):
        assert not any(secret in text for text in written)
