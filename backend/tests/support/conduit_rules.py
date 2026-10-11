"""What an OpenAI-style streaming client needs from a stream, as assertions.

The rules are the ones the Conduit iOS app applies (design spec section 4) plus the OpenAI SDK's
own ``ChatCompletionChunk`` model, so a stream that passes here parses in both. ``shape`` and
``collect`` are the two views of a stream the tests compare against: the sequence of things a
client receives, and the text it assembles from them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openai.types.chat import ChatCompletionChunk

from app.gateway.openai_compat.translator import Chunk, Done, Failed, Output, to_sse


def shape(outputs: Sequence[Output]) -> list[tuple[str, Any]]:
    """One ``(kind, value)`` per output: ``role``, ``content``, ``reasoning``, ``finish``, ``usage``, ``error`` or ``done``."""
    result: list[tuple[str, Any]] = []
    for output in outputs:
        if isinstance(output, Done):
            result.append(("done", ""))
        elif isinstance(output, Failed):
            result.append(("error", output.error.code))
        elif not output.body["choices"]:
            result.append(("usage", output.body["usage"]))
        elif "role" in output.delta:
            result.append(("role", output.delta["role"]))
        elif "content" in output.delta:
            result.append(("content", output.delta["content"]))
        elif "reasoning_content" in output.delta:
            result.append(("reasoning", output.delta["reasoning_content"]))
        else:
            result.append(("finish", output.body["choices"][0]["finish_reason"]))
    return result


def collect(outputs: Sequence[Output]) -> tuple[str, str]:
    """The ``(content, reasoning_content)`` a client assembles from the stream."""
    content, reasoning = [], []
    for output in outputs:
        if isinstance(output, Chunk):
            content.append(output.delta.get("content", ""))
            reasoning.append(output.delta.get("reasoning_content", ""))
    return "".join(content), "".join(reasoning)


def assert_conduit_rules(outputs: Sequence[Output]) -> None:
    """Assert that ``outputs`` is a complete stream a Conduit client accepts."""
    assert outputs, "a stream sends at least its first chunk"
    first, last = outputs[0], outputs[-1]
    assert isinstance(first, Chunk) and first.delta == {"role": "assistant"}, "the first chunk carries the role"
    assert isinstance(last, (Done, Failed)), "a stream ends with [DONE] or an in-band error"
    assert not any(isinstance(output, (Done, Failed)) for output in outputs[:-1]), "nothing may follow [DONE] or an in-band error"

    for output in outputs:
        frame = to_sse(output)
        assert frame.startswith("data: ") and frame.endswith("\n\n") and "\n" not in frame[:-2], "one single-line data frame per output"
        payload = frame[len("data: ") : -2]
        assert payload, "an empty data frame throws in Conduit's parser"
        assert payload == "[DONE]" or isinstance(json.loads(payload), dict), "every payload is JSON or [DONE]"

    chunks = [output for output in outputs if isinstance(output, Chunk)]
    assert len({(chunk.body["id"], chunk.body["created"], chunk.body["model"]) for chunk in chunks}) == 1, "id, created and model stay constant"
    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk.body)
        assert chunk.body["object"] == "chat.completion.chunk"
        assert "tool_calls" not in chunk.delta and "function_call" not in chunk.delta, "Conduit turns streamed tool calls into an error"
        assert chunk.delta.get("content", "x") != "" and chunk.delta.get("reasoning_content", "x") != "", "no empty deltas"

    if isinstance(last, Done):
        assert any(chunk.delta.get("content") or chunk.delta.get("reasoning_content") for chunk in chunks), "[DONE] with no content and no reasoning is an error in Conduit"
        finishing = [chunk for chunk in chunks if chunk.body["choices"] and chunk.body["choices"][0]["finish_reason"] is not None]
        assert len(finishing) == 1, "exactly one chunk carries a finish_reason"
        assert finishing[0].body["choices"][0]["finish_reason"] == "stop" and finishing[0].delta == {}
        trailing = outputs[outputs.index(finishing[0]) + 1 : -1]
        assert len(trailing) <= 1 and all(isinstance(output, Chunk) and output.body["choices"] == [] and "usage" in output.body for output in trailing), "only the usage chunk sits between the finish chunk and [DONE]"
