"""Tests for the Buzz (Nostr) channel connector."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("coincurve")

from app.channels.base import Channel
from app.channels.buzz import BuzzChannel, _chunk_text
from app.channels.manager import CHANNEL_CAPABILITIES
from app.channels.message_bus import InboundMessageType, MessageBus, OutboundMessage
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

    Reproduces the review finding that a later stop() awaited the finished relay
    task and only caught CancelledError/TimeoutError, so any other stored
    exception propagated out of stop() -- leaving `_task` non-None and skipping
    the "stopped" log.

    Updated for Task 6: this originally reproduced the crash via the Task-3
    skeleton's `_run_loop` stub, which raised `NotImplementedError` on its very
    first statement. Task 6 replaces that stub with a real reconnect-forever loop
    that, by design, does NOT crash on an ordinary connection error -- it backs
    off and retries instead (see test_run_loop_reconnects_after_connection_failure).
    Real network I/O is also not an option here: with no `_connect` seam configured,
    the real `_run_loop` would call `websockets.connect()` against this test's fake
    `wss://buzz.example.com` relay URL, which is real (slow, sandboxing-dependent)
    network I/O that must never run inside a unit test. So the "already crashed"
    scenario is now reproduced by monkeypatching `_run_loop` itself to simulate a
    hypothetical future bug there, while still exercising the REAL
    (non-monkeypatched) `_spawn_connection` -> real asyncio task path this test
    is about.
    """

    async def run():
        ch = _channel()

        async def crashing_run_loop() -> None:
            raise RuntimeError("simulated _run_loop bug")

        ch._run_loop = crashing_run_loop
        await ch.start()  # real _spawn_connection: creates a real asyncio task

        # Give the event loop a chance to run the (monkeypatched) _run_loop to
        # completion (it raises on its very first statement, so one or two
        # scheduling turns are enough).
        for _ in range(10):
            if ch._task is not None and ch._task.done():
                break
            await asyncio.sleep(0)
        assert ch._task is not None and ch._task.done()

        await ch.stop()  # must not raise the task's stored RuntimeError

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


# -- Task 4 review fixes -----------------------------------------------------


class FakeConnectionRepo:
    """Minimal test double for ChannelConnectionRepository: consume + upsert only."""

    def __init__(self, *, states=None, raise_on_consume=False):
        self._states = dict(states or {})
        self._raise_on_consume = raise_on_consume
        self.upserts = []

    async def consume_oauth_state(self, *, provider, state, now=None):
        if self._raise_on_consume:
            raise RuntimeError("boom")
        owner_user_id = self._states.pop(state, None)
        if owner_user_id is None:
            return None
        return {"owner_user_id": owner_user_id, "provider": provider, "requested_scopes": [], "metadata": {}, "redirect_after": None}

    async def upsert_connection(self, **kwargs):
        self.upserts.append(kwargs)
        return {"id": "conn-1", **kwargs}


def test_connect_code_binds_and_never_publishes_even_for_unauthorized_author():
    """FINDING 1 (Critical): a valid /connect code from a non-allowlisted pubkey
    must bind via the connection repo and never reach _publish as a chat message."""
    repo = FakeConnectionRepo(states={"tok-1": "owner-xyz"})
    ch, captured = _started(connection_repo=repo)
    _dispatch(ch, _event(pubkey="ee" * 32, content="/connect tok-1", mentions=()))
    assert captured == []
    assert len(repo.upserts) == 1
    assert repo.upserts[0]["owner_user_id"] == "owner-xyz"
    assert repo.upserts[0]["external_account_id"] == "ee" * 32
    assert repo.upserts[0]["provider"] == "buzz"


def test_connect_code_invalid_never_publishes_and_does_not_bind():
    """An unrecognized/expired code must still never publish, and must not upsert."""
    repo = FakeConnectionRepo()
    ch, captured = _started(connection_repo=repo)
    _dispatch(ch, _event(pubkey="ee" * 32, content="/connect not-a-real-code", mentions=()))
    assert captured == []
    assert repo.upserts == []


def test_connect_code_repo_error_never_publishes_and_does_not_crash():
    """A connection-repo failure while binding must be swallowed, not crash the read loop,
    and must still never fall through to publish."""
    repo = FakeConnectionRepo(raise_on_consume=True)
    ch, captured = _started(connection_repo=repo)
    _dispatch(ch, _event(pubkey="ee" * 32, content="/connect tok-1", mentions=()))
    assert captured == []


def test_connect_code_never_publishes_even_for_already_allowed_and_mentioned_author():
    """Ruling: a /connect message must NEVER reach _publish, valid code or not --
    even when the author is already allowlisted and mentioned."""
    repo = FakeConnectionRepo(states={"tok-2": "owner-abc"})
    ch, captured = _started(connection_repo=repo)
    _dispatch(ch, _event(pubkey=OWNER, content="/connect tok-2", mentions=(PK3_HEX,)))
    assert captured == []
    assert len(repo.upserts) == 1 and repo.upserts[0]["external_account_id"] == OWNER


def test_known_command_classifies_as_command_but_plain_text_stays_chat():
    """FINDING 3: is_known_channel_command classification had no direct coverage."""
    ch, captured = _started()
    _dispatch(ch, _event(content="@DeerFlow /goal ship it", mentions=(PK3_HEX,), eid="44" * 32))
    assert len(captured) == 1
    assert captured[0].msg_type == InboundMessageType.COMMAND
    assert captured[0].text == "/goal ship it"

    _dispatch(ch, _event(content="@DeerFlow just chatting", mentions=(PK3_HEX,), eid="55" * 32))
    assert len(captured) == 2
    assert captured[1].msg_type == InboundMessageType.CHAT


def test_strip_own_mention_leaves_ambiguous_multi_mention_text_untouched():
    """FINDING 4: "@Alice, @DeerFlow help" must not have Alice's mention dropped
    just because our own mention also appears in the message."""
    ch, captured = _started()
    _dispatch(ch, _event(content="@Alice, @DeerFlow help", mentions=(PK3_HEX,)))
    assert len(captured) == 1
    assert captured[0].text == "@Alice, @DeerFlow help"


# -- Task 5: outbound — placeholder post, streaming edits, final, oversize split ---


class FakeTransport:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(json.loads(text))


def _outbound(ch, text, *, is_final, thread_ts=None):
    return OutboundMessage(channel_name="buzz", chat_id=CHANNEL, thread_id="t1", text=text, is_final=is_final, thread_ts=thread_ts)


def _events_of(transport):
    return [f[1] for f in transport.sent if f[0] == "EVENT"]


def test_streaming_posts_placeholder_then_edits_then_final():
    ch, _ = _started()
    transport = FakeTransport()
    ch._transport = transport
    ch._last_requester[(CHANNEL, None)] = OWNER
    asyncio.run(ch.send(_outbound(ch, "Working…", is_final=False)))
    asyncio.run(ch.send(_outbound(ch, "Working… more", is_final=False)))
    asyncio.run(ch.send(_outbound(ch, "Final answer", is_final=True)))
    events = _events_of(transport)
    assert [e["kind"] for e in events] == [9, 40003, 40003]
    placeholder = events[0]
    assert ["p", OWNER] in placeholder["tags"]  # requester notified on the initial post
    assert all(["e", placeholder["id"]] in e["tags"] for e in events[1:])
    assert events[-1]["content"] == "Final answer"
    assert (CHANNEL, None) not in ch._stream_targets  # final clears the target


def test_thread_reply_targets_thread_root():
    ch, _ = _started()
    transport = FakeTransport()
    ch._transport = transport
    root = "aa" * 32
    asyncio.run(ch.send(_outbound(ch, "reply", is_final=True, thread_ts=root)))
    (ev,) = _events_of(transport)
    assert ev["kind"] == 9 and ["e", root] in ev["tags"]


def test_oversized_final_splits_into_followup_posts():
    ch, _ = _started()
    transport = FakeTransport()
    ch._transport = transport
    big = "x" * 130_000  # > 2 * EDIT_MAX_BYTES
    asyncio.run(ch.send(_outbound(ch, big, is_final=True)))
    events = _events_of(transport)
    assert events[0]["kind"] == 9 and all(e["kind"] == 9 for e in events[1:])
    assert "".join(e["content"] for e in events) == big
    assert all(len(e["content"].encode()) <= 60_000 for e in events)


def test_send_without_transport_raises_for_retry(monkeypatch):
    # Correction vs. the brief's literal snippet: _send_with_retry sleeps 2**attempt
    # seconds between its 3 attempts on a real failure, which would otherwise burn
    # ~3s of wall-clock time here for no benefit (this is a pure failure-path test).
    # Patching the shared retry helper's sleep call keeps it instant without
    # touching BuzzChannel/_send_with_retry itself.
    monkeypatch.setattr("app.channels.base.asyncio.sleep", AsyncMock())
    ch, _ = _started()
    with pytest.raises(RuntimeError):
        asyncio.run(ch.send(_outbound(ch, "hi", is_final=True)))


def test_chunk_text_never_splits_a_multibyte_character_across_chunks():
    """Correctness requirement beyond the brief's (ASCII-only) oversize test:
    splitting is by ENCODED BYTE LENGTH and must never cut a multi-byte UTF-8
    character in half. Uses a 4-byte-wide character (an emoji outside the BMP)
    with a limit that is deliberately NOT a multiple of 4, so a naive
    text.encode()[:limit]-style byte slice would corrupt a character; the
    real chunker must not."""
    text = "\U0001f600" * 50  # grinning-face emoji: 4 bytes each in UTF-8
    limit = 61
    chunks = _chunk_text(text, limit=limit)
    assert "".join(chunks) == text
    assert all(len(c.encode()) <= limit for c in chunks)
    # A split-mid-character chunk could never have a byte length that is an
    # exact multiple of the (uniform) 4-byte character width.
    assert all(len(c.encode()) % 4 == 0 for c in chunks)
    assert chunks[0] == "\U0001f600" * 15  # 15*4=60 <= 61 bytes; a 16th char would make 64 > 61


def test_edit_failure_degrades_to_fresh_post_and_retargets_stream(monkeypatch):
    """Correctness requirement beyond the brief's literal tests: if an edit fails
    after retries, BuzzChannel must degrade by posting a fresh message rather than
    losing the content, and must retarget _stream_targets at the new message so
    later edits in the same conversation land on it instead of the abandoned one."""
    monkeypatch.setattr("app.channels.base.asyncio.sleep", AsyncMock())
    ch, _ = _started()

    class FailingEditTransport:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            frame = json.loads(text)
            if frame[0] == "EVENT" and frame[1]["kind"] == 40003:
                raise RuntimeError("relay rejected edit")
            self.sent.append(frame)

    transport = FailingEditTransport()
    ch._transport = transport

    asyncio.run(ch.send(_outbound(ch, "placeholder", is_final=False)))
    asyncio.run(ch.send(_outbound(ch, "update that fails to edit", is_final=False)))

    events = [f[1] for f in transport.sent if f[0] == "EVENT"]
    assert [e["kind"] for e in events] == [9, 9]  # placeholder + degraded fresh post, no successful edit
    assert events[1]["content"] == "update that fails to edit"
    assert ch._stream_targets[(CHANNEL, None)] == events[1]["id"]  # retargeted to the new message


# -- Review finding: _stream_targets must not leak when a FINAL send() raises -----


def test_stream_target_cleared_even_when_final_send_fails(monkeypatch):
    """FINDING (Important/spec): the edit->degrade path can raise (both the edit
    AND the degraded fresh-post attempts exhaust their retries), which used to
    propagate out of send() BEFORE reaching the `if msg.is_final: pop` at the end
    -- leaking a stale _stream_targets entry. This must hold regardless of
    success/failure: (a) send() must still raise (the framework's retry/error
    path in Channel._on_outbound must still see the failure), and (b) the stale
    target must not survive, so the NEXT send() for this conversation starts a
    fresh placeholder (kind 9) rather than editing the abandoned one (kind
    40003) once the relay recovers."""
    monkeypatch.setattr("app.channels.base.asyncio.sleep", AsyncMock())
    ch, _ = _started()

    class AlwaysFailingTransport:
        async def send(self, text):
            raise RuntimeError("relay down")

    ch._transport = AlwaysFailingTransport()

    # Seed an existing placeholder target, as a prior successful non-final
    # send() would have, so this final call takes the edit (not "first post")
    # branch -- the exact branch the finding calls out.
    key = (CHANNEL, None)
    ch._stream_targets[key] = "ff" * 32

    with pytest.raises(RuntimeError):
        asyncio.run(ch.send(_outbound(ch, "final answer", is_final=True)))

    assert key not in ch._stream_targets  # no stale/partial target survives a raised final send

    # Once the relay recovers, the next send() for the same conversation must
    # start a fresh placeholder, not an edit targeting the abandoned message.
    transport = FakeTransport()
    ch._transport = transport
    asyncio.run(ch.send(_outbound(ch, "retry after recovery", is_final=True)))
    (ev,) = _events_of(transport)
    assert ev["kind"] == 9


def test_stream_target_cleared_when_overflow_chunk_send_fails(monkeypatch):
    """Second FINDING regression case, pinning the other raise site named in the
    review: the overflow-chunk loop can also raise (a later chunk fails after
    retries even though the first chunk already succeeded). _stream_targets must
    still be cleared for this key -- not left pointing at the successfully-sent
    first chunk -- and the exception must still propagate."""
    monkeypatch.setattr("app.channels.base.asyncio.sleep", AsyncMock())
    ch, _ = _started()

    class FailsAfterFirstFrame:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            frame = json.loads(text)
            if self.sent:  # the first frame succeeds; every one after raises
                raise RuntimeError("relay dropped mid-stream")
            self.sent.append(frame)

    transport = FailsAfterFirstFrame()
    ch._transport = transport
    key = (CHANNEL, None)
    big = "x" * 130_000  # forces at least one follow-up overflow chunk

    with pytest.raises(RuntimeError):
        asyncio.run(ch.send(_outbound(ch, big, is_final=True)))

    assert key not in ch._stream_targets


# -- Task 4 carry-forward: /connect must reply, and a failed reply send must ------
# -- never be reported as a failed bind -------------------------------------------


def test_connect_success_sends_confirmation_reply():
    repo = FakeConnectionRepo(states={"tok-conf": "owner-conf"})
    ch, captured = _started(connection_repo=repo)
    transport = FakeTransport()
    ch._transport = transport
    _dispatch(ch, _event(pubkey="ff" * 32, content="/connect tok-conf", mentions=()))
    events = _events_of(transport)
    assert len(events) == 1
    assert events[0]["kind"] == 9
    assert events[0]["content"] == "Buzz connected to DeerFlow."
    assert ["p", "ff" * 32] in events[0]["tags"]
    assert captured == []  # still never published as a chat message


def test_connect_invalid_code_sends_error_reply():
    repo = FakeConnectionRepo()
    ch, captured = _started(connection_repo=repo)
    transport = FakeTransport()
    ch._transport = transport
    _dispatch(ch, _event(pubkey="ff" * 32, content="/connect not-a-real-code", mentions=()))
    (event,) = _events_of(transport)
    assert event["content"] == "Buzz connection code is invalid or expired."


def test_connect_repo_error_sends_error_reply_and_does_not_crash():
    repo = FakeConnectionRepo(raise_on_consume=True)
    ch, captured = _started(connection_repo=repo)
    transport = FakeTransport()
    ch._transport = transport
    _dispatch(ch, _event(pubkey="ff" * 32, content="/connect tok-1", mentions=()))
    (event,) = _events_of(transport)
    assert event["content"] == "Buzz connection could not be completed from this message."


def test_connect_success_survives_a_failed_confirmation_send(caplog):
    """CRITICAL (Task 4 review carry-forward pitfall): a failure to SEND the
    confirmation for an otherwise-successful bind must never be reported or
    logged as a bind failure. ch._transport is left unset (None) so the reply
    attempt raises, but the bind itself (consume_oauth_state + upsert_connection)
    must still go through, and only a distinctly-worded send-failure warning may
    be logged -- never "failed to bind"."""
    repo = FakeConnectionRepo(states={"tok-fail-send": "owner-fail-send"})
    ch, captured = _started(connection_repo=repo)
    with caplog.at_level(logging.INFO, logger="app.channels.buzz"):
        _dispatch(ch, _event(pubkey="ff" * 32, content="/connect tok-fail-send", mentions=()))

    assert len(repo.upserts) == 1  # the bind itself succeeded despite the failed reply
    assert repo.upserts[0]["owner_user_id"] == "owner-fail-send"
    assert captured == []
    assert "failed to bind" not in caplog.text
    assert "failed to send" in caplog.text


# -- Task 6: relay connection loop -- connect, NIP-42 auth, subscribe, reconnect --


class ScriptedWS:
    """Async-iterable fake websocket: yields scripted frames, records sends."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send(self, text):
        self.sent.append(json.loads(text))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_session_authenticates_then_subscribes_with_since_cursor():
    ch, _ = _started()
    ch._seen_created_at = 1700000500
    ws = ScriptedWS(['["AUTH","challenge-1"]'])

    asyncio.run(ch._session(ws))
    auth_frames = [f for f in ws.sent if f[0] == "AUTH"]
    req_frames = [f for f in ws.sent if f[0] == "REQ"]
    assert len(auth_frames) == 1 and auth_frames[0][1]["kind"] == 22242
    assert ["challenge", "challenge-1"] in auth_frames[0][1]["tags"]
    assert req_frames, "expected a REQ subscription"
    kinds = [f0 for f in req_frames for f0 in f[2:] if isinstance(f0, dict)]
    assert any(9 in f.get("kinds", []) and f.get("since") == 1700000500 for f in kinds)
    assert any(39000 in f.get("kinds", []) for f in kinds)


def test_session_routes_events_through_handle_relay_frame():
    ch, captured = _started()
    ws = ScriptedWS([json.dumps(["EVENT", "s", _event()])])
    asyncio.run(ch._session(ws))
    assert len(captured) == 1


def test_run_loop_reconnects_after_connection_failure():
    """After a connect failure, _run_loop must back off and retry rather than giving up.

    Correction vs. the brief's literal mock setup: patching `app.channels.buzz.asyncio.sleep`
    with a bare `unittest.mock.AsyncMock()` patches the *shared* `asyncio` module object (since
    `buzz.py` does `import asyncio`, not `from asyncio import sleep`), so it also silently
    replaces the `asyncio.sleep(0)` this test's own polling loop relies on to yield control
    back to the event loop. A bare AsyncMock's returned coroutine has no real suspension point,
    so awaiting it never actually hands control back to the scheduler -- verified empirically
    (see task report) two ways: (a) with the polling loop's own `asyncio.sleep(0)` calls also
    silently mocked out, the `_run_loop` task never gets scheduled even once, so `attempts`
    stays empty and the test fails outright; (b) if only the polling loop is protected (e.g. by
    capturing a `real_sleep` reference before patching) while `_run_loop`'s internal backoff
    `await asyncio.sleep(delay)` remains a non-yielding mock, `_run_loop` can retry in a genuine
    infinite tight loop with zero suspension points anywhere in its call chain (mocked connect,
    trivial ScriptedWS stubs, non-yielding sleep) -- this reproducibly hung the interpreter at
    100% CPU in manual verification and had to be killed. The fix keeps the mock's call-count
    bookkeeping (`slept.await_count`) but gives its `side_effect` a genuine zero-duration
    `asyncio.sleep(0)` (captured before patching, so it cannot recursively call itself),
    so every backoff still really yields to the loop -- never a real multi-second delay, but
    never a non-yielding busy spin either. `asyncio.wait_for(..., timeout=10)` is an outer,
    real-wall-clock safety bound so a future regression here fails fast instead of hanging CI.
    """
    ch, _ = _started()
    attempts = []

    def make_connect():
        async def connect():
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("boom")
            return ScriptedWS([json.dumps(["EVENT", "s", _event()])])

        return connect

    ch._connect = make_connect()

    async def run():
        import unittest.mock

        real_sleep = asyncio.sleep  # captured before patching: used by the mock's side_effect

        async def instant_yield(*_args, **_kwargs):
            await real_sleep(0)  # a genuine, zero-duration event-loop tick -- never real seconds

        with unittest.mock.patch("app.channels.buzz.asyncio.sleep", new=unittest.mock.AsyncMock(side_effect=instant_yield)) as slept:
            task = asyncio.get_running_loop().create_task(ch._run_loop())
            for _ in range(200):
                await asyncio.sleep(0)
                if len(attempts) >= 2 and task.done() is False and not ch._task:
                    break
                if len(attempts) >= 2:
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert slept.await_count >= 1  # backed off after the failure

    asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert len(attempts) >= 2


def test_spawn_connection_cannot_fail_start_even_if_connect_immediately_errors():
    """Carried-forward invariant (from an earlier review, resolved in this task): in start(),
    `_running = True` is set AFTER `subscribe_outbound()` and `_spawn_connection()`. That was
    only safe while `_spawn_connection` was a bare `create_task(...)` call that could not itself
    raise. Task 6 gives `_run_loop` real, potentially-failing connect logic, so this pins that
    the invariant still holds: `_spawn_connection` remains non-fallible because it still only
    calls `asyncio.create_task(self._run_loop(), ...)`, which schedules the coroutine and
    returns without running any of its body -- a connect failure happens later, inside the
    spawned task, never synchronously inside start(). So even a connect that fails on its very
    first attempt cannot leave start() partially applied (outbound listener subscribed but
    `_running` still False, which would make the guarded stop() silently no-op and leak both).
    """

    async def run():
        import unittest.mock

        ch = _channel()

        async def immediately_failing_connect():
            raise RuntimeError("boom-on-first-connect")

        ch._connect = immediately_failing_connect

        with unittest.mock.patch("app.channels.buzz.asyncio.sleep", new=unittest.mock.AsyncMock()):
            await ch.start()  # must fully commit even though the spawned relay loop will
            # immediately hit immediately_failing_connect the first time it gets scheduled
            assert ch.is_running
            assert ch.bus._outbound_listeners == [ch._on_outbound]
            assert ch._task is not None

            await ch.stop()  # must cleanly unwind: no leaked listener/task either

        assert not ch.is_running
        assert ch.bus._outbound_listeners == []
        assert ch._task is None

    asyncio.run(run())
