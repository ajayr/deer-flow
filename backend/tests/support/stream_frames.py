"""Record the frames a DeerFlow run publishes, as fixtures for stream-translation tests.

``record_scenario`` runs a real LangGraph agent (``langchain.agents.create_agent``, the factory
the lead agent is built with) against a scripted chat model, and pushes every stream item through
``deerflow.runtime.serialization.serialize``, which is what ``runtime/runs/worker.py`` does before
it publishes to the stream bridge. The result has the shape a bridge subscriber receives:
``{"event": "messages", "data": [message_dict, metadata]}``, ``{"event": "custom", ...}`` and the
``metadata`` and ``end`` frames around them, in publish order.

What is real and what is scripted. LangGraph and langchain-core produce the frame structure: the
``messages`` and ``custom`` framing, the per-chunk tool-call parsing, the thread's stored messages.
The scripted model supplies the tokens *and* their message ids, finish reasons, usage and
``chunk_position`` markers (see ``text_delta`` and ``last_chunk``). DeerFlow's own middleware runs
where a scenario names it: clarification, the error handling that retries a model call, and the
safety stop. The subagent task events are scripted too (the payloads copy ``tools/builtins/task_tool.py``).

What no recording covers: the lead agent's full middleware chain, the run worker's publish path
(``_LargeFileToolChunkBatcher`` holds back the first chunk of a ``write_file`` call, which delays
the point where a tool call is visible), and subagent execution (nested graphs never run, so the
recordings say nothing about nested frames). Those are settled by reading the code, and the unit
tests cover nested frames with hand-built ones.

Volatile values (uuids, library versions, the tracing settings LangSmith adds to frame metadata
when it is configured) are replaced or dropped so a recording is stable.
``DEERFLOW_WRITE_FIXTURES=1`` rewrites the committed copies after a library upgrade changes the frames.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import openai
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.config import get_stream_writer
from pydantic import PrivateAttr

from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware
from deerflow.config.app_config import AppConfig, LlmCallConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.runtime.serialization import serialize
from deerflow.tools.builtins.clarification_tool import ask_clarification_tool

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "openai_compat" / "streams"

RUN_ID = "run-1"
THREAD_ID = "thread-1"


class ScriptedChatModel(BaseChatModel):
    """A chat model that streams a scripted turn per call; a turn item that is an exception is raised."""

    script: list[list[Any]]
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        return self

    def _stream(self, messages: list[Any], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        turn = self.script[self._calls]
        self._calls += 1
        for item in turn:
            if isinstance(item, BaseException):
                raise item
            yield ChatGenerationChunk(message=item)

    def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        merged: Any = None
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            merged = chunk if merged is None else merged + chunk
        return ChatResult(generations=[ChatGeneration(message=merged.message)])


def text_delta(message_id: str, content: str, **fields: Any) -> AIMessageChunk:
    return AIMessageChunk(content=content, id=message_id, **fields)


def reasoning_delta(message_id: str, reasoning: str) -> AIMessageChunk:
    return AIMessageChunk(content="", id=message_id, additional_kwargs={"reasoning_content": reasoning})


def tool_call_start(message_id: str, name: str, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(content="", id=message_id, tool_call_chunks=[{"name": name, "args": "", "id": call_id, "index": 0}])


def tool_call_args(message_id: str, args: str) -> AIMessageChunk:
    return AIMessageChunk(content="", id=message_id, tool_call_chunks=[{"name": None, "args": args, "id": None, "index": 0}])


def last_chunk(message_id: str, finish_reason: str = "stop") -> AIMessageChunk:
    """The chunk that closes a streamed call: finish reason, token usage and ``chunk_position == "last"``."""
    return AIMessageChunk(
        content="",
        id=message_id,
        response_metadata={"finish_reason": finish_reason, "model_name": "scripted"},
        usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        chunk_position="last",
    )


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return "3 results"


@tool
async def task(description: str, prompt: str, subagent_type: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Delegate a piece of work to a subagent."""
    writer = get_stream_writer()
    model_name = "scripted"
    writer({"type": "task_started", "task_id": tool_call_id, "description": description, "model_name": model_name})
    step = AIMessage(content="Working on it.", id="sub-ai-1").model_dump()
    writer({"type": "task_running", "task_id": tool_call_id, "message": step, "message_index": 1, "total_messages": 1, "usage": None, "model_name": model_name})
    writer({"type": "task_completed", "task_id": tool_call_id, "result": "42", "usage": None, "model_name": model_name})
    return "Task Succeeded. Result: 42"


def _failing_connection() -> openai.APIConnectionError:
    return openai.APIConnectionError(message="Connection error.", request=httpx.Request("POST", "https://llm.invalid/v1/chat/completions"))


@dataclass(frozen=True)
class Scenario:
    prompt: str
    script: Callable[[], list[list[Any]]]
    tools: Callable[[], list[Any]] = lambda: []
    middleware: Callable[[], list[Any]] = lambda: []


def _error_handling_middleware() -> LLMErrorHandlingMiddleware:
    config = AppConfig(sandbox=SandboxConfig(use="test"), llm_call=LlmCallConfig(retry_max_attempts=2, retry_base_delay_ms=1, retry_cap_delay_ms=1))
    return LLMErrorHandlingMiddleware(app_config=config)


SCENARIOS: dict[str, Scenario] = {
    "plain_answer": Scenario(
        prompt="Say hello.",
        script=lambda: [[text_delta("ai-1", "Hello"), text_delta("ai-1", ", world."), last_chunk("ai-1")]],
    ),
    "tool_run_with_narration": Scenario(
        prompt="Find the DeerFlow docs.",
        script=lambda: [
            [
                text_delta("ai-1", "Let me look "),
                text_delta("ai-1", "that up."),
                tool_call_start("ai-1", "web_search", "call_1"),
                tool_call_args("ai-1", '{"query": "deer'),
                tool_call_args("ai-1", 'flow docs"}'),
                last_chunk("ai-1", "tool_calls"),
            ],
            [text_delta("ai-2", "The docs "), text_delta("ai-2", "are online."), last_chunk("ai-2")],
        ],
        tools=lambda: [web_search],
    ),
    "thinking": Scenario(
        prompt="What is 2 + 2?",
        script=lambda: [[reasoning_delta("ai-1", "Add the "), reasoning_delta("ai-1", "numbers."), text_delta("ai-1", "Four"), text_delta("ai-1", "."), last_chunk("ai-1")]],
    ),
    "subagent_run": Scenario(
        prompt="Ask a subagent for the answer.",
        script=lambda: [
            [
                text_delta("ai-1", "Delegating."),
                tool_call_start("ai-1", "task", "call_task"),
                tool_call_args("ai-1", '{"description": "find the answer", "prompt": "What is 6 x 7?", "subagent_type": "general-purpose"}'),
                last_chunk("ai-1", "tool_calls"),
            ],
            [text_delta("ai-2", "The subagent says 42."), last_chunk("ai-2")],
        ],
        tools=lambda: [task],
    ),
    "clarification": Scenario(
        prompt="Deploy it.",
        script=lambda: [
            [
                text_delta("ai-1", "I need to check something first."),
                tool_call_start("ai-1", "ask_clarification", "call_clar_1"),
                tool_call_args("ai-1", '{"question": "Which environment?", "clarification_type": "approach_choice", "options": ["staging", "production"]}'),
                last_chunk("ai-1", "tool_calls"),
            ]
        ],
        tools=lambda: [ask_clarification_tool],
        middleware=lambda: [ClarificationMiddleware()],
    ),
    "provider_failure": Scenario(
        prompt="Say hello.",
        script=lambda: [[_failing_connection()], [_failing_connection()]],
        middleware=lambda: [_error_handling_middleware()],
    ),
    "model_call_retried": Scenario(
        prompt="What is the capital of France?",
        script=lambda: [
            [text_delta("ai-1", "The capital of "), text_delta("ai-1", "France is"), _failing_connection()],
            [text_delta("ai-2", "The capital of "), text_delta("ai-2", "France is Paris."), last_chunk("ai-2")],
        ],
        middleware=lambda: [_error_handling_middleware()],
    ),
    "safety_stop_after_narration": Scenario(
        prompt="Write that file.",
        script=lambda: [[text_delta("ai-1", "Let me write that file."), tool_call_start("ai-1", "web_search", "call_1"), tool_call_args("ai-1", '{"query": "par'), last_chunk("ai-1", "content_filter")]],
        tools=lambda: [web_search],
        middleware=lambda: [SafetyFinishReasonMiddleware()],
    ),
    "safety_stop_without_narration": Scenario(
        prompt="Write that file.",
        script=lambda: [[tool_call_start("ai-1", "web_search", "call_1"), tool_call_args("ai-1", '{"query": "par'), last_chunk("ai-1", "content_filter")]],
        tools=lambda: [web_search],
        middleware=lambda: [SafetyFinishReasonMiddleware()],
    ),
}


@dataclass(frozen=True)
class Recording:
    """What a scenario published, and the messages the turn left in the thread."""

    prompt: str
    events: list[dict[str, Any]]
    turn_messages: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps({"prompt": self.prompt, "events": self.events, "turn_messages": self.turn_messages}, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

#: The metadata keys of a ``messages`` frame that a recording keeps. LangChain adds more when tracing
#: is configured (the ``LANGSMITH_*`` settings of the environment, a git revision), and a committed
#: recording must not carry the settings of whoever regenerated it.
_KEPT_METADATA_KEYS = re.compile(r"langgraph_.*|checkpoint_ns|thread_id|ls_integration|ls_model_type|ls_provider")


def _stabilize(value: Any, ids: dict[str, str]) -> Any:
    """Replace what changes between runs: uuids (numbered by first appearance) and library versions."""
    if isinstance(value, dict):
        return {key: _stabilize(item, ids) for key, item in value.items() if key != "lc_versions"}
    if isinstance(value, list):
        return [_stabilize(item, ids) for item in value]
    if isinstance(value, str):
        return _UUID.sub(lambda match: ids.setdefault(match.group(0), f"uuid-{len(ids) + 1}"), value)
    return value


async def record_scenario(name: str) -> Recording:
    scenario = SCENARIOS[name]
    model = ScriptedChatModel(script=scenario.script())
    graph = create_agent(model=model, tools=scenario.tools(), middleware=scenario.middleware())
    events: list[dict[str, Any]] = [{"event": "metadata", "data": {"run_id": RUN_ID, "thread_id": THREAD_ID}}]
    final: dict[str, Any] = {}
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 25}
    async for mode, chunk in graph.astream({"messages": [HumanMessage(content=scenario.prompt, id="human-1")]}, config=config, stream_mode=["messages", "custom", "values"]):
        if mode == "values":
            final = chunk
        elif mode == "messages":
            message, metadata = serialize(chunk, mode=mode)
            events.append({"event": mode, "data": [message, {key: value for key, value in metadata.items() if _KEPT_METADATA_KEYS.fullmatch(key)}]})
        else:
            events.append({"event": mode, "data": serialize(chunk, mode=mode)})
    events.append({"event": "end", "data": None})
    turn_messages = serialize(final, mode="values")["messages"][1:]
    ids: dict[str, str] = {}
    return Recording(prompt=scenario.prompt, events=_stabilize(events, ids), turn_messages=_stabilize(turn_messages, ids))


def load_recording(name: str) -> Recording:
    data = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return Recording(prompt=data["prompt"], events=data["events"], turn_messages=data["turn_messages"])
