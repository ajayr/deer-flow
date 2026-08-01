"""Tests for the Buzz (Nostr) channel connector."""

import asyncio

import pytest

pytest.importorskip("coincurve")

from app.channels.base import Channel
from app.channels.buzz import BuzzChannel
from app.channels.manager import CHANNEL_CAPABILITIES
from app.channels.message_bus import MessageBus
from app.channels.run_policy import CHANNEL_RUN_POLICY
from app.channels.service import _CHANNEL_CREDENTIAL_KEYS, _CHANNEL_REGISTRY

SK3_HEX = "0000000000000000000000000000000000000000000000000000000000000003"
PK3_HEX = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"
OWNER = "dd" * 32
CHANNEL = "136852ee-63e1-49c2-8927-413b5ee8e5f7"


def _channel(**overrides) -> BuzzChannel:
    config = {"relay_url": "wss://buzz.example.com", "private_key": SK3_HEX, "allowed_users": [OWNER], **overrides}
    return BuzzChannel(bus=MessageBus(), config=config)


def test_registered_in_framework_registries():
    assert _CHANNEL_REGISTRY["buzz"] == "app.channels.buzz:BuzzChannel"
    assert _CHANNEL_CREDENTIAL_KEYS["buzz"] == ["private_key"]
    assert CHANNEL_CAPABILITIES["buzz"] == {"supports_streaming": True}
    policy = CHANNEL_RUN_POLICY["buzz"]
    assert policy.serialize_thread_runs is True and policy.requires_bound_identity is False


def test_is_a_channel_named_buzz_with_streaming():
    ch = _channel()
    assert isinstance(ch, Channel) and ch.name == "buzz" and ch.supports_streaming is True


def test_config_parsing_normalizes_allowlist_and_defaults():
    ch = _channel(allowed_users=[OWNER.upper()], mention_free_channels=[CHANNEL])
    assert ch._allowed_users == {OWNER}
    assert ch._require_mention is True and ch._mention_free == {CHANNEL}
    assert ch._relay_url == "wss://buzz.example.com"


def test_config_rejects_non_websocket_relay_url():
    with pytest.raises(ValueError):
        _channel(relay_url="https://buzz.example.com")


def test_start_and_stop_manage_outbound_subscription():
    async def run():
        ch = _channel()
        ch._spawn_connection = lambda: None  # skeleton: no real socket in tests
        await ch.start()
        assert ch.is_running and ch.bus._outbound_listeners == [ch._on_outbound]
        await ch.stop()
        assert not ch.is_running and ch.bus._outbound_listeners == []

    asyncio.run(run())


def test_start_is_idempotent_against_double_start():
    """A second start() while already running must not double-subscribe or re-spawn.

    Reproduces the review finding that calling start() twice appended
    `_on_outbound` to `bus._outbound_listeners` twice and spawned a second
    concurrent relay-loop task, since the original skeleton had no
    re-entrancy guard (unlike github.py / discord.py's `if self._running:
    return`).
    """

    async def run():
        ch = _channel()
        spawn_calls = 0

        def fake_spawn() -> None:
            nonlocal spawn_calls
            spawn_calls += 1

        ch._spawn_connection = fake_spawn

        await ch.start()
        await ch.start()  # must be a no-op: already running

        assert ch.bus._outbound_listeners == [ch._on_outbound]
        assert spawn_calls == 1

    asyncio.run(run())


def test_stop_is_safe_after_the_relay_task_already_crashed():
    """stop() must not re-raise a stored exception from an already-finished task.

    Reproduces the review finding that, after a real (non-monkeypatched)
    start(), the stubbed _run_loop() raises NotImplementedError almost
    immediately; a later stop() awaited the finished task and only caught
    CancelledError/TimeoutError, so the NotImplementedError propagated out
    of stop() — leaving `_task` non-None and skipping the "stopped" log.
    """

    async def run():
        ch = _channel()
        await ch.start()  # real _spawn_connection: creates a real asyncio task

        # Give the event loop a chance to run the stub _run_loop to completion
        # (it raises NotImplementedError on its very first statement, so one
        # or two scheduling turns are enough).
        for _ in range(10):
            if ch._task is not None and ch._task.done():
                break
            await asyncio.sleep(0)
        assert ch._task is not None and ch._task.done()

        await ch.stop()  # must not raise the task's stored NotImplementedError

        assert not ch.is_running
        assert ch._task is None

    asyncio.run(run())
