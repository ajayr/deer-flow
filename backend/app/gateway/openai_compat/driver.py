"""The loop between a run's stream and one response: one translator, fed from one bridge subscription.

Pure asyncio: no FastAPI, no store. The router passes in the bridge's iterator and the two calls that
reach outside (the disconnect check and ``settle``), and time is the event loop's own, so scripted streams
on a virtual-time loop test every way a response can end. The translator decides what a client sees; this loop decides when the
translator is asked, what ends the response, and when the client is told that the connection is alive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.gateway.openai_compat.errors import INTERNAL_ERROR, RUN_ERROR, RUN_INTERRUPTED, STREAM_GAP, InBandError
from app.gateway.openai_compat.translator import Done, Failed, Output, StreamTranslator, Usage, stored_answer, to_sse
from deerflow.runtime import END_SENTINEL, RunRecord, RunStatus, StreamGap
from deerflow.runtime.messages_stream import stream_payload_type
from deerflow.utils.messages import INJECTED_USER_MESSAGE_ID_SUFFIX

logger = logging.getLogger(__name__)

#: A response that has sent nothing for this long sends an SSE comment. Five seconds stays well inside the idle
#: timeouts of common proxies, and below the ten at which one deployment's client path was seen to reap quiet streams.
KEEPALIVE_SECONDS = 5.0

KEEPALIVE_FRAME = ": keepalive\n\n"

#: A frame that fails the answer (the run's error event, or the model's error message) arrives while the run is still finishing.
#: The failure is reported only once the run has ended, or after this long, so that a client that sends again at once does not
#: find the run it just saw fail still holding its thread. It stays below ``KEEPALIVE_SECONDS``: the wait never needs a keepalive.
FAILED_RUN_WAIT_SECONDS = 3.0

# A run that did not finish says so in-band. A run that is still pending or running is not in the table: the worker
# sets the final status before it ends the stream, so the stream having ended means the run is over.
_FAILED_STATUSES = {RunStatus.error: RUN_ERROR, RunStatus.timeout: RUN_ERROR, RunStatus.interrupted: RUN_INTERRUPTED}


@dataclass(frozen=True, slots=True)
class Keepalive:
    """Nothing was sent for ``KEEPALIVE_SECONDS``: the response sends an SSE comment."""


KEEPALIVE = Keepalive()

Step = Output | Keepalive


def to_wire(step: Step) -> str:
    """The text one step puts on the wire."""
    return KEEPALIVE_FRAME if isinstance(step, Keepalive) else to_sse(step)


@dataclass(frozen=True, slots=True)
class Settled:
    """What the router learned about a run once its stream had ended."""

    usage: Usage | None = None
    #: Set when the run's final status says it did not finish.
    failure: InBandError | None = None
    #: The answer the thread stored for the turn, if it could be read.
    stored: str | None = None


def failure_of(status: RunStatus) -> InBandError | None:
    """How the run's final status ends its stream; ``None`` for a run that finished."""
    return _FAILED_STATUSES.get(status)


def usage_of(*, input_tokens: int, output_tokens: int, total_tokens: int) -> Usage | None:
    """The run's token totals, or ``None`` when the provider reported none."""
    if total_tokens <= 0:
        return None
    return Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens, total_tokens=total_tokens)


def turn_messages(messages: Sequence[Mapping[str, Any]], user_message_id: str) -> list[Mapping[str, Any]] | None:
    """The messages a thread holds after the turn's user message, or ``None`` when that message is not there.

    The first turn of a thread is the one the dynamic-context middleware re-keys: the reminder it adds takes the
    message's id and the user's own text moves to ``<id>__user``. So the user's message is the human one with
    either id, and a system reminder with the same id is not it.
    """
    wanted = {user_message_id, user_message_id + INJECTED_USER_MESSAGE_ID_SUFFIX}
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and stream_payload_type(message) == "human" and message.get("id") in wanted:
            return list(messages[index + 1 :])
    return None


async def settle_run(record: RunRecord, *, user_message_id: str, read_messages: Callable[[], Awaitable[Sequence[Mapping[str, Any]]]]) -> Settled:
    """What the router learns about a run whose stream has ended: how it ended, its token totals and the answer its thread kept.

    The worker finalizes the record before it ends the stream, so its status and totals are final here. A run that did not
    finish is settled by its status alone and its thread is not read. For one that did, a thread that cannot be read costs
    only the stored answer.
    """
    failure = failure_of(record.status)
    if failure is not None:
        return Settled(failure=failure)
    usage = usage_of(input_tokens=record.total_input_tokens, output_tokens=record.total_output_tokens, total_tokens=record.total_tokens)
    try:
        turn = turn_messages(await read_messages(), user_message_id)
    except Exception:
        logger.exception("Could not read the thread to settle a run")
        return Settled(usage=usage)
    return Settled(usage=usage, stored=None if turn is None else stored_answer(turn))


class RunStream:
    """The steps of one response: iterate ``steps()``; afterwards ``ended`` says how the response ended.

    ``items`` is the bridge's subscription to the run. The first step is the chunk that carries the role, so
    the response can start before the run has said anything. Every wait is bounded by whichever comes first of
    the keepalive and the translator's own deadline, so a held message is released and a quiet stream stays alive.
    The client's disconnect is polled between waits and ends the loop with the translator still open. After
    ``end`` the loop calls ``settle`` for the run's final facts and lets the translator finish with them. A frame
    that fails the answer is held back until the run has ended (at most ``FAILED_RUN_WAIT_SECONDS``), so the response
    never reports a failure while its run still owns the thread.

    ``ended`` is ``open`` until the loop knows, ``done`` after ``[DONE]``, ``failed`` after an in-band error
    (``failure`` says which) and ``disconnected`` when the client went away. It stays ``open`` if the consumer
    closes or cancels the generator, which is how a server reports a dropped connection too. It is set before the
    last step is handed over, so it is right even if the client leaves while that step is being sent.
    """

    def __init__(
        self,
        translator: StreamTranslator,
        items: AsyncIterator[Any],
        *,
        settle: Callable[[], Awaitable[Settled]],
        is_disconnected: Callable[[], Awaitable[bool]],
        keepalive_seconds: float = KEEPALIVE_SECONDS,
    ) -> None:
        self._translator = translator
        self._items = items
        self._settle = settle
        self._is_disconnected = is_disconnected
        self._keepalive_seconds = keepalive_seconds
        # The read of the next item is kept across waits: cancelling it would end the bridge's generator.
        self._pending: asyncio.Future[Any] | None = None
        self.ended: Literal["open", "done", "failed", "disconnected"] = "open"
        self.failure: InBandError | None = None
        self.settled: Settled | None = None

    async def steps(self) -> AsyncIterator[Step]:
        translator = self._translator
        try:
            for output in translator.open(now=self._now()):
                yield output
            sent_at = self._now()
            while True:
                if await self._is_disconnected():
                    self.ended = "disconnected"
                    return
                if self._pending is None:
                    self._pending = asyncio.ensure_future(self._items.__anext__())
                wake = sent_at + self._keepalive_seconds
                deadline = translator.next_deadline()
                if deadline is not None:
                    wake = min(wake, deadline)
                # A wake time that has already passed gives a negative timeout, which asyncio treats as zero.
                await asyncio.wait({self._pending}, timeout=wake - self._now())
                now = self._now()
                if self._pending.done():
                    read, self._pending = self._pending, None
                    outputs = await self._on_read(read, now)
                else:
                    outputs = translator.tick(now=now)
                self._mark_ended(outputs)
                for output in outputs:
                    yield output
                if translator.closed:
                    return
                if outputs:
                    sent_at = self._now()
                elif self._now() - sent_at >= self._keepalive_seconds:
                    yield KEEPALIVE
                    sent_at = self._now()
        except Exception:
            logger.exception("Streaming a run failed")
            self.ended, self.failure = "failed", INTERNAL_ERROR
            yield Failed(INTERNAL_ERROR)
        finally:
            self._discard_pending()

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    async def _on_read(self, read: asyncio.Future[Any], now: float) -> list[Output]:
        """Translate what a finished read of the subscription produced."""
        try:
            item = read.result()
        except StopAsyncIteration:
            # The subscription ended without `end`: the stream to the client is lost, not the run.
            return self._translator.finish(now=now, failure=STREAM_GAP)
        except Exception:
            logger.exception("The stream bridge failed while a run was being streamed")
            return self._translator.finish(now=now, failure=STREAM_GAP)
        if item is END_SENTINEL:
            return await self._finish(now)
        if isinstance(item, StreamGap):
            return self._translator.finish(now=now, failure=STREAM_GAP)
        # A heartbeat of the bridge is a frame the translator has nothing to say about.
        outputs = self._translator.feed(item.event, item.data, now=now)
        if self._translator.closed:
            # Known before the wait: a client that leaves while the run ends left a failed response, not a running one.
            self._mark_ended(outputs)
            await self._wait_for_end()
        return outputs

    async def _wait_for_end(self) -> None:
        """Let the run that a frame just failed finish: read and drop what it still sends until its end, or until the wait runs out."""
        try:
            async with asyncio.timeout(FAILED_RUN_WAIT_SECONDS):
                while True:
                    item = await anext(self._items)
                    if item is END_SENTINEL or isinstance(item, StreamGap):
                        return
        except (TimeoutError, StopAsyncIteration):
            return
        except Exception:
            logger.exception("The stream bridge failed while a failed run was ending")

    async def _finish(self, now: float) -> list[Output]:
        try:
            settled = await self._settle()
        except Exception:
            logger.exception("Could not settle a finished run")
            settled = Settled()
        self.settled = settled
        return self._translator.finish(now=now, usage=settled.usage, failure=settled.failure, stored=settled.stored)

    def _mark_ended(self, outputs: list[Output]) -> None:
        last = outputs[-1] if outputs else None
        if isinstance(last, Failed):
            self.ended, self.failure = "failed", last.error
        elif isinstance(last, Done):
            self.ended = "done"

    def _discard_pending(self) -> None:
        """Stop reading: the response is over, so the subscription is abandoned along with the read in flight."""
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.cancel()
