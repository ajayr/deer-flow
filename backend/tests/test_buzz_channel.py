"""Tests for the Buzz (Nostr) channel connector."""

import asyncio

import pytest

pytest.importorskip("coincurve")

from app.channels.base import Channel
from app.channels.buzz import BuzzChannel
from app.channels.manager import CHANNEL_CAPABILITIES
from app.channels.message_bus import InboundMessageType, MessageBus
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


def _event(*, pubkey=OWNER, kind=9, content="@DeerFlow hello", channel=CHANNEL, mentions=(PK3_HEX,), reply_to=None, eid="11" * 32, created_at=1700000100):
    tags = [["h", channel]]
    if reply_to:
        tags.append(["e", reply_to])
    tags.extend(["p", m] for m in mentions)
    return {"id": eid, "pubkey": pubkey, "created_at": created_at, "kind": kind, "tags": tags, "content": content}


def _started(**overrides):
    ch = _channel(**overrides)
    ch._keys = __import__("app.channels.buzz_nostr", fromlist=["x"]).parse_private_key(SK3_HEX)
    captured = []

    async def publish(msg):
        captured.append(msg)

    ch._publish = publish
    return ch, captured


def _dispatch(ch, ev):
    import json

    asyncio.run(ch.handle_relay_frame(json.dumps(["EVENT", "sub1", ev])))


def test_mentioned_allowed_author_is_published():
    ch, captured = _started()
    _dispatch(ch, _event())
    assert len(captured) == 1
    msg = captured[0]
    assert msg.channel_name == "buzz" and msg.chat_id == CHANNEL and msg.user_id == OWNER
    assert msg.text == "hello"  # own leading @mention stripped
    assert msg.metadata["event_id"] == "11" * 32 and msg.workspace_id == "buzz.example.com"
    assert msg.msg_type == InboundMessageType.CHAT


def test_disallowed_author_is_dropped():
    ch, captured = _started()
    _dispatch(ch, _event(pubkey="ee" * 32))
    assert captured == []


def test_own_events_are_ignored():
    ch, captured = _started()
    _dispatch(ch, _event(pubkey=PK3_HEX))
    assert captured == []


def test_unmentioned_channel_message_is_dropped_but_dm_passes():
    ch, captured = _started()
    _dispatch(ch, _event(content="no mention here", mentions=()))
    assert captured == []
    ch._handle_meta_event({"kind": 39000, "tags": [["d", CHANNEL], ["t", "dm"], ["name", "DM"]]})
    _dispatch(ch, _event(content="dm without mention", mentions=(), eid="22" * 32))
    assert len(captured) == 1 and captured[0].text == "dm without mention"


def test_mention_free_channel_and_thread_follow_pass_without_mention():
    ch, captured = _started(mention_free_channels=[CHANNEL])
    _dispatch(ch, _event(content="open channel", mentions=()))
    assert len(captured) == 1

    class FakeStore:
        def get_thread_id(self, channel_name, chat_id, topic_id=None):
            return "thread-1" if topic_id == "aa" * 32 else None

    ch2, captured2 = _started()
    ch2.config["channel_store"] = FakeStore()
    _dispatch(ch2, _event(content="follow-up", mentions=(), reply_to="aa" * 32, eid="33" * 32))
    assert len(captured2) == 1 and captured2[0].topic_id == "aa" * 32


def test_thread_replies_map_topic_and_requester_and_watermark():
    ch, captured = _started()
    _dispatch(ch, _event(reply_to="aa" * 32, created_at=1700000200))
    assert captured[0].topic_id == "aa" * 32 and captured[0].thread_ts == "aa" * 32
    assert ch._last_requester[(CHANNEL, "aa" * 32)] == OWNER
    assert ch._seen_created_at == 1700000200


def test_auth_frame_records_challenge_and_meta_defaults_fail_closed():
    ch, captured = _started()
    asyncio.run(ch.handle_relay_frame('["AUTH","challenge-xyz"]'))
    assert ch._pending_auth_challenge == "challenge-xyz"
    # unknown channel type == not a DM -> unmentioned message still dropped
    _dispatch(ch, _event(channel="99999999-9999-4999-8999-999999999999", content="x", mentions=()))
    assert captured == []
