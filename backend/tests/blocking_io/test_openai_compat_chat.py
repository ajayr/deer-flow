"""Regression anchors: the chat completions request path must not block the event loop.

The tests drive the Gateway composition of ``test_openai_compat_chat_router`` (auth middleware, personal access
tokens, the router, ``start_run``, the run worker, the stream bridge, the checkpointer and the thread store) with a
scripted model. Blocking work in the router, the driver or the Gateway helpers the router calls (the thread store,
the checkpoint reads, the ledger writes) therefore trips the strict Blockbuster gate here.

The Gateway modules that the composition patches by import path are imported at module scope: a first import reads
package metadata, which the gate counts as blocking when it happens on the loop under test.
"""

from __future__ import annotations

import pytest
from test_openai_compat_chat_router import FIRST, FOLLOW_UP, assistant, running_gateway, user

import app.gateway.deps  # noqa: F401
import app.gateway.routers.auth  # noqa: F401
import app.gateway.services  # noqa: F401

pytestmark = [pytest.mark.asyncio, pytest.mark.no_auto_user]


async def test_a_streamed_and_a_whole_answer_do_not_block_the_event_loop(monkeypatch, tmp_path) -> None:
    async with running_gateway(monkeypatch, tmp_path) as gateway:
        gateway.script.say("Paris.", "Berlin.")

        streamed = await gateway.chat([user(FIRST)], stream=True)
        whole = await gateway.chat([user("Capital of Germany?")])

    assert streamed.status_code == whole.status_code == 200
    assert streamed.text.rstrip().endswith("data: [DONE]")
    assert whole.json()["choices"][0]["message"]["content"] == "Berlin."


async def test_a_continued_conversation_and_a_seeded_history_do_not_block_the_event_loop(monkeypatch, tmp_path) -> None:
    async with running_gateway(monkeypatch, tmp_path) as gateway:
        gateway.script.say("Paris.", "About two million.", "The Tiber.")

        first = await gateway.chat([user(FIRST)])
        continued = await gateway.chat([user(FIRST), assistant("Paris."), user(FOLLOW_UP)])
        seeded = await gateway.chat([user("Capital of Italy?"), assistant("Rome."), user("Which river runs through it?")], stream=True)

        # The second request continued the first thread; the third had to start one and seed it.
        assert len(await gateway.threads()) == 2

    assert first.status_code == continued.status_code == seeded.status_code == 200
