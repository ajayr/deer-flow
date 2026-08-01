"""Buzz (Nostr) channel: DeerFlow as a member of a Buzz workspace relay.

One NIP-42-authenticated WebSocket to ``relay_url``. Inbound kind-9 chat events are
gated (pubkey allowlist, then mention/DM/thread-follow) and published to the bus;
outbound replies post one kind-9 message and then stream via kind-40003 in-place edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any
from urllib.parse import urlparse

from app.channels import buzz_nostr
from app.channels.base import Channel
from app.channels.commands import is_known_channel_command
from app.channels.connection_identity import attach_connection_identity
from app.channels.message_bus import InboundMessage, InboundMessageType, MessageBus, OutboundMessage

logger = logging.getLogger(__name__)

# Headroom under the relay's 64KB edit-content cap (kind-40003 events).
EDIT_MAX_BYTES = 60_000

# Ceiling for how far ahead of our own clock a peer-supplied ``created_at`` may be
# and still advance the resubscribe watermark (see ``_advance_watermark``).
MAX_FUTURE_SKEW_SECONDS = 60

# Cap on the kind-39000 metadata cache. Any relay member can publish channel
# metadata, so this map is remote-fed and would otherwise grow without bound for
# the process lifetime; evicting the oldest entry only costs us the DM mention
# exemption for that channel until its metadata is seen again (``_is_dm`` fails
# closed on a cache miss).
MAX_CACHED_CHANNELS = 512

# Bound on how long ``stop()`` waits for the cancelled relay loop to finish.
STOP_TIMEOUT_SECONDS = 5.0

# Wording mirrors every sibling adapter's `/connect` reply style (discord.py's
# `_send_connection_reply`, slack.py's `_post_connection_reply`, and the same
# templates in wecom.py / feishu.py / dingtalk.py / wechat.py), substituting
# "Buzz" for the platform name.
_CONNECT_REPLY_TEXT = {
    "success": "Buzz connected to DeerFlow.",
    "invalid": "Buzz connection code is invalid or expired.",
    "error": "Buzz connection could not be completed from this message.",
}


def _chunk_text(text: str, limit: int = EDIT_MAX_BYTES) -> list[str]:
    """Split *text* into chunks whose UTF-8 ENCODED byte length never exceeds *limit*.

    Operates character-by-character (not on raw encoded bytes), so a multi-byte
    UTF-8 character is always appended to a chunk whole -- it is measured before
    being added and, if it would push the running byte total over *limit*, the
    chunk is flushed first and the character starts the next one. This makes a
    split-mid-character corruption structurally impossible, regardless of
    whether *limit* happens to be a multiple of any character's byte width.
    """
    chunks, current, size = [], [], 0
    for ch in text:
        b = len(ch.encode())
        if size + b > limit and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(ch)
        size += b
    if current:
        chunks.append("".join(current))
    return chunks or [""]


class BuzzChannel(Channel):
    _connect: Any = None  # test seam: async callable returning an async-context-manager transport

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="buzz", bus=bus, config=config)
        self._relay_url = str(config.get("relay_url", "")).strip()
        if not self._relay_url.startswith(("ws://", "wss://")):
            raise ValueError("channels.buzz.relay_url must be a ws:// or wss:// URL")
        # One community per relay URL (see the design's multi-community note), so the
        # relay host is this channel's workspace: it scopes inbound dedupe, the
        # persisted connection row written by `/connect`, and the lookup that resolves
        # that row back on the inbound path. Computed once here so those three uses
        # can never drift apart.
        self._workspace_id = urlparse(self._relay_url).netloc
        self._private_key_raw = str(config.get("private_key", ""))
        self._keys: buzz_nostr.NostrKeys | None = None  # parsed in start() so coincurve stays lazy
        self._allowed_users = {buzz_nostr.parse_pubkey(v) for v in config.get("allowed_users", []) or []}
        self._require_mention = bool(config.get("require_mention", True))
        self._mention_free = {str(c) for c in config.get("mention_free_channels", []) or []}
        self._channel_meta: dict[str, dict[str, Any]] = {}
        self._stream_targets: dict[tuple[str, str | None], str] = {}
        self._stream_tails: dict[tuple[str, str | None], list[str]] = {}  # overflow chunk ids beyond chunk 0, per conversation
        self._last_requester: dict[tuple[str, str | None], str] = {}
        self._pending_auth_challenge: str | None = None  # set from an AUTH relay frame; consumed by Task 6's NIP-42 flow
        self._seen_created_at: int = 0  # high-water mark of PROCESSED event created_at; Task 6 resubscribes with since=this
        self._transport: Any = None
        self._task: asyncio.Task | None = None
        self._publish = self.bus.publish_inbound  # test seam (discord.py idiom)

    @property
    def supports_streaming(self) -> bool:
        return True

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._keys = buzz_nostr.parse_private_key(self._private_key_raw)
        if not self._allowed_users:
            # Deny-by-default is deliberate (unlike siblings, where an empty
            # allowlist means "allow all"), so this cannot be a hard failure --
            # but a configured-and-connected channel that silently drops every
            # message looks exactly like a broken relay to the operator. Say so
            # once, loudly, at the only point where it is actionable.
            logger.warning("[buzz] channels.buzz.allowed_users is empty: EVERY inbound chat message will be dropped (Buzz denies by default). Add member pubkeys (hex or npub) to enable the channel.")
        self.bus.subscribe_outbound(self._on_outbound)
        self._spawn_connection()
        self._running = True
        logger.info("[buzz] channel started (relay=%s pubkey=%s allowed_users=%d)", self._relay_url, self._keys.pubkey_hex, len(self._allowed_users))

    def _spawn_connection(self) -> None:
        self._task = asyncio.create_task(self._run_loop(), name="buzz-relay-loop")

    async def stop(self) -> None:
        """Cancel the relay loop and drop every piece of per-connection state.

        Teardown is *bounded* and *coherent*, in that order:

        - Bounded: the cancelled task is awaited via ``asyncio.wait`` rather than
          ``asyncio.wait_for``. ``wait_for`` guarantees the awaited task is
          finished before it raises ``TimeoutError``, so a task that swallows
          ``CancelledError`` (a bug, but the exact case a timeout exists for)
          would hang ``stop()`` forever instead of the intended 5s.
          ``asyncio.wait`` returns after the timeout regardless and cancels
          nothing further, so this always returns.
        - Coherent: the reviewer's finding was that a timed-out ``_task`` was
          dropped while it might still own ``_transport``, leaving an abandoned
          task posting on a socket the channel believed it no longer held. The
          ``finally`` now clears ``_transport`` (so ``_post_event`` fails fast
          rather than writing through a socket nobody owns) alongside ``_task``,
          and the timeout is logged instead of passing silently.

        Per-connection bookkeeping (stream placeholders, overflow tails, last
        requester, half-consumed auth challenge, remote-fed channel metadata) is
        also cleared: those maps key on relay event ids from the session being
        torn down, so a later ``start()`` must not resume editing placeholders
        from a previous process lifetime.
        """
        if not self._running:
            return
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        if self._task is not None:
            task = self._task
            task.cancel()
            try:
                _, pending = await asyncio.wait({task}, timeout=STOP_TIMEOUT_SECONDS)
                if pending:
                    logger.warning("[buzz] relay loop did not finish within %ss of cancellation; abandoning it", STOP_TIMEOUT_SECONDS)
                elif not task.cancelled() and task.exception() is not None:
                    # _run_loop is designed to never end on an ordinary connection error (it
                    # backs off and retries instead -- see its docstring), but stop() must
                    # still complete cleanly rather than re-raise whatever a genuine bug in
                    # the relay loop task ended with.
                    logger.error("[buzz] relay loop task ended with an error during stop: %s", task.exception())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[buzz] failed to await the relay loop task during stop")
            finally:
                self._task = None
                self._transport = None
        self._stream_targets.clear()
        self._stream_tails.clear()
        self._last_requester.clear()
        self._channel_meta.clear()
        self._pending_auth_challenge = None
        logger.info("[buzz] channel stopped")

    def _subscription_filters(self) -> list[dict]:
        """NIP-01 REQ filters for our one subscription: member chat plus channel metadata.

        ``since`` rides on the chat filter only when we have a watermark
        (``_seen_created_at``, advanced by ``_advance_watermark``): on a fresh
        channel this is 0 (no watermark yet, so no ``since`` -- the relay's default
        backlog applies), and on every reconnect afterwards it is the newest
        ``created_at`` we have already processed, so re-subscribing after a drop
        does not replay the entire channel history nor silently skip events
        published while we were disconnected.

        "Processed" now means accepted-and-published (or a fully handled
        ``/connect``), never merely received. That is strictly safer for the
        skip direction: the cursor can only ever be lower than it would have been
        under the old advance-on-everything rule, so any event the old cursor
        would have covered is still covered. What it costs is replay -- events we
        already dropped come back after a reconnect and are dropped again by the
        same gates, which is idempotent -- plus one guaranteed redelivery of the
        boundary event itself, since NIP-01 ``since`` is inclusive; the manager's
        inbound dedupe (keyed on our ``event_id`` metadata within this workspace)
        is what keeps that from starting a second run.
        """
        chat_filter: dict = {"kinds": [buzz_nostr.KIND_CHAT]}
        if self._seen_created_at:
            chat_filter["since"] = self._seen_created_at
        return [chat_filter, {"kinds": [buzz_nostr.KIND_CHANNEL_META]}]

    async def _session(self, ws) -> None:
        """Run one relay connection's lifetime: subscribe, authenticate if challenged, read frames.

        ``self._transport`` is set for the duration of this connection so ``send()``
        / ``_post_event()`` can post outbound events on it, and is unconditionally
        cleared in ``finally`` on the way out -- including on error -- so a stale
        reference can never survive past this connection (``_post_event`` reads it
        fresh on every call and raises when it is ``None``; see its docstring).

        NIP-42 auth is opportunistic, not upfront: the initial REQ is sent
        immediately in case the relay allows unauthenticated reads, but
        ``handle_relay_frame`` records any ``AUTH`` challenge the relay sends onto
        ``self._pending_auth_challenge``, and this loop reacts to it the next time
        around by signing and sending the AUTH event and then RESUBSCRIBING --
        our relay is closed, so the pre-auth REQ may have been silently rejected,
        and without resubscribing we would end up authenticated but listening to
        nothing. A fresh challenge is per-connection (the relay mints a new one on
        every new socket), so this runs again from scratch on every reconnect.
        """
        self._transport = ws
        try:
            await ws.send(buzz_nostr.req_frame("buzz-main", *self._subscription_filters()))
            async for raw in ws:
                await self.handle_relay_frame(raw)
                if self._pending_auth_challenge is not None:
                    assert self._keys is not None
                    challenge = self._pending_auth_challenge
                    self._pending_auth_challenge = None
                    auth = buzz_nostr.build_auth_event(self._keys, self._relay_url, challenge, created_at=int(time.time()))
                    await ws.send(json.dumps(["AUTH", auth], separators=(",", ":")))
                    # Re-subscribe post-auth: the pre-auth REQ may have been rejected on closed relays.
                    await ws.send(buzz_nostr.req_frame("buzz-main", *self._subscription_filters()))
        finally:
            self._transport = None

    async def _run_loop(self) -> None:
        """Own the relay connection for the channel's lifetime: connect, run one session, retry forever.

        Runs as the asyncio task ``_spawn_connection`` starts; ``stop()`` cancels it.
        Every sibling connector (wecom.py, discord.py, feishu.py) delegates
        reconnection to its vendor SDK -- Buzz has no SDK, so this loop owns
        reconnect/backoff itself: exponential backoff capped at 60s with jitter
        (so a thundering herd of Buzz-connected agents restarting together does not
        all retry in lockstep), reset to 0 after any connection that actually got
        established (so a long stable run doesn't leave a stale attempt count that
        punishes the next transient blip with a large initial delay).

        A clean stream end -- ``_session`` returning normally because the relay
        closed the socket on us -- is treated the same as a connection error: both
        fall through to the backoff-and-retry below rather than ending the loop,
        because for a relay client "the peer hung up" is exactly the situation
        reconnection exists for.

        ``asyncio.CancelledError`` is re-raised immediately rather than reaching the
        generic ``except Exception`` below -- it isn't an ``Exception`` subclass in
        the supported Python version anyway, but the explicit clause documents the
        contract: this is how the guarded ``stop()`` (Task 3) actually stops this
        otherwise-infinite loop, and it must never be swallowed here.

        Untrusted relay input is guarded one layer down: ``handle_relay_frame``
        already logs-and-drops a non-JSON payload or a malformed ``EVENT`` instead
        of raising, so a single bad frame reaches neither this loop's
        ``except Exception`` (reconnect) nor kills the read loop -- it is simply
        skipped and iteration continues.
        """
        attempt = 0
        while True:
            try:
                if self._connect is not None:
                    ws = await self._connect()
                else:
                    import websockets

                    ws = await websockets.connect(self._relay_url)
                async with ws:
                    attempt = 0  # reset only once a connection is actually established
                    await self._session(ws)
                # Clean stream end (relay closed on us): reconnect with backoff too.
                attempt += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                logger.warning("[buzz] relay connection error (attempt %d): %s", attempt, exc)
            # Cap the counter itself, not just the delay: 2**10 already exceeds the
            # 60s ceiling below, so clamping here keeps the exponent computation
            # cheap indefinitely instead of growing the attempt count (and the
            # bigint math behind 2**attempt) without bound across a very long
            # outage.
            attempt = min(attempt, 10)
            delay = min(60, 2**attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)

    # -- inbound -------------------------------------------------------------

    async def handle_relay_frame(self, raw: str) -> None:
        """Route one raw relay WebSocket frame (Task 6 calls this per frame received).

        Relay input is untrusted: a non-JSON payload, an unexpected frame shape, or
        an ``EVENT`` payload whose tags/timestamp are malformed is logged and dropped
        rather than raised, so one bad frame can never crash the read loop.

        Every ``EVENT`` is signature-verified (``buzz_nostr.verify_event``) here,
        at the single entry point, BEFORE either handler runs. The relay operator
        is not necessarily the DeerFlow operator on a team-run Buzz relay, and
        ``ev["pubkey"]`` is just a field in a relay-supplied JSON object -- without
        this check a malicious or compromised relay could name any allowlisted
        author it liked and trigger tool-executing runs, or bind a victim's pubkey
        to an attacker's DeerFlow account through ``/connect``. Both handlers make
        authorization decisions from the event (the chat gate uses ``pubkey`` as
        the principal; kind-39000 metadata can relax the mention requirement), so
        verifying at the choke point rather than inside each one leaves no path
        where an unverified event reaches a decision.

        Chosen over verifying later, behind the cheap self/kind/channel gates, on
        purpose: a Schnorr verify is tens of microseconds against human-rate chat
        traffic, so the throughput saved by gating first is worth less than the
        guarantee that no future edit can reorder a gate ahead of the check.
        """
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[buzz] non-JSON relay frame ignored")
            return
        if not isinstance(frame, list) or not frame:
            return
        kind = frame[0]
        if kind == "AUTH" and len(frame) >= 2:
            self._pending_auth_challenge = str(frame[1])
        elif kind == "EVENT" and len(frame) >= 3 and isinstance(frame[2], dict):
            ev = frame[2]
            if not buzz_nostr.verify_event(ev):
                logger.warning("[buzz] dropped relay event with an invalid id/signature (claimed pubkey=%s kind=%s)", ev.get("pubkey"), ev.get("kind"))
                return
            try:
                if ev.get("kind") == buzz_nostr.KIND_CHANNEL_META:
                    self._handle_meta_event(ev)
                elif ev.get("kind") == buzz_nostr.KIND_CHAT:
                    await self._handle_chat_event(ev)
            except Exception:
                # Defense in depth against malformed tags/timestamps inside an
                # otherwise well-shaped EVENT frame (e.g. a non-integer created_at,
                # or a "tags" field that isn't a list of [name, value, ...] lists).
                logger.warning("[buzz] malformed relay event ignored", exc_info=True)
        # OK / EOSE / NOTICE frames need no action

    def _handle_meta_event(self, ev: dict) -> None:
        """Cache kind-39000 channel metadata: ``d`` = channel id, ``t`` = type, ``name``.

        Trust assumption (authorship, not authenticity): ``handle_relay_frame``
        has already proved this event really was signed by the pubkey it names,
        but this does not check WHICH pubkey that is. In the Buzz protocol,
        kind-39000 channel-discovery events are expected to be published by the
        relay's own keypair, not by ordinary members. So any relay member able to
        publish an event can still legitimately sign one and mark an arbitrary
        channel ``type: "dm"``, which relaxes the mention gate for that channel
        (see ``_is_dm``) — it does NOT bypass the independent pubkey allowlist
        gate in ``_handle_chat_event``. Closing this gap needs a trusted relay
        pubkey to check the author against, and nothing already configured
        identifies one: ``relay_url`` is a network address, not a signing key.
        Deliberately left as a follow-up rather than inventing a new required
        config key (e.g. ``relay_pubkey``) for it here.

        The cache is remote-fed, so it is capped (``MAX_CACHED_CHANNELS``) on a
        first-in-first-out basis: an unbounded map would let any member grow this
        process's memory one forged ``d`` tag at a time. Eviction is safe by
        construction — a channel we hold no metadata for is treated as a non-DM
        (``_is_dm`` fails closed), and the next kind-39000 event for it repopulates
        the entry.
        """
        d_values = buzz_nostr.tag_values(ev, "d")
        if not d_values:
            return
        names = buzz_nostr.tag_values(ev, "name")
        types = buzz_nostr.tag_values(ev, "t")
        channel_id = d_values[0]
        meta = {"type": types[0] if types else "stream", "name": names[0] if names else ""}
        self._channel_meta[channel_id] = meta
        while len(self._channel_meta) > MAX_CACHED_CHANNELS:
            self._channel_meta.pop(next(iter(self._channel_meta)))
        logger.debug("[buzz] channel metadata cached: %s (%s) type=%s", meta["name"] or "<unnamed>", channel_id, meta["type"])

    def _is_dm(self, channel_id: str) -> bool:
        """True only when metadata was positively cached as ``type == "dm"``.

        Fails closed: a channel we have no kind-39000 metadata for yet is never
        treated as a DM, even though ``dict.get(..., {})`` would otherwise make an
        absent entry look indistinguishable from an unset (non-DM) type.
        """
        return self._channel_meta.get(channel_id, {}).get("type") == "dm"

    def _thread_root(self, ev: dict) -> str | None:
        e_tags = buzz_nostr.tag_values(ev, "e")
        return e_tags[0] if e_tags else None

    def _strip_own_mention(self, text: str) -> str:
        """Strip a single, unambiguous leading ``@mention`` token.

        Nostr chat events carry no verified mapping from the free-text "@Name" a
        client rendered into the message body to the pubkeys in the event's ``p``
        tags — the caller only confirms *some* p-tagged mention exists (see
        ``mentioned`` in ``_handle_chat_event``), never that the specific leading
        token names *us*. When a second ``@token`` immediately follows the first
        (e.g. "@Alice, @DeerFlow help"), guessing that the first one is ours risks
        silently discarding a different member's mention while leaving ours
        untouched, so the conservative choice is to leave the text completely
        alone rather than guess. The common single-mention case ("@DeerFlow
        hello") remains unambiguous and is still stripped.
        """
        stripped = text.lstrip()
        if not stripped.startswith("@"):
            return text.strip()
        _, sep, rest = stripped.partition(" ")
        if not sep:
            return stripped  # "@DeerFlow" alone: nothing to strip without losing the whole message
        if rest.lstrip().startswith("@"):
            return text.strip()  # ambiguous multi-mention prefix: don't guess which one is ours
        return rest.strip() or stripped

    async def _bind_connection(self, code: str, author: str, channel_id: str) -> None:
        """Consume a ``/connect <code>`` bind code for *author* (a Nostr pubkey hex).

        Always fully handles the request — valid code, invalid/expired/already-used
        code, or a connection-repo error — so the caller (``_handle_chat_event``) can
        unconditionally return right after awaiting this without ever falling through
        to ``_make_inbound``/``_publish``. Mirrors discord.py's / slack.py's
        ``_bind_connection_from_connect_code``.

        Bind success/failure is fully determined and logged by the try/except below
        BEFORE ``_reply_to_connect`` is ever invoked, so a failure to *send* the
        confirmation/error reply can never be attributed back to (or logged as) a
        bind failure — see ``_reply_to_connect``. This ordering is deliberate: an
        earlier review of this method flagged that sending from inside the same
        try/except that decides bind success would let a relay-send hiccup on an
        otherwise-successful bind get reported as "failed to bind".
        """
        if self._connection_repo is None:
            return  # unreachable in practice: _pending_connect_code already requires this
        try:
            state = await self._connection_repo.consume_oauth_state(provider="buzz", state=code)
            if state is None:
                logger.info("[buzz] /connect code invalid, expired, or already used (pubkey=%s)", author)
                outcome = "invalid"
            else:
                await self._connection_repo.upsert_connection(
                    owner_user_id=state["owner_user_id"],
                    provider="buzz",
                    external_account_id=author,
                    workspace_id=self._workspace_id,
                    metadata={"pubkey": author},
                    status="connected",
                )
                logger.info("[buzz] connected pubkey=%s to owner_user_id=%s", author, state["owner_user_id"])
                outcome = "success"
        except Exception:
            # A repo/DB error binding the code must not propagate: handle_relay_frame's
            # outer guard would also catch it, but catching here keeps the log specific
            # to the bind failure instead of a generic "malformed relay event ignored".
            logger.exception("[buzz] failed to bind /connect code for pubkey=%s", author)
            outcome = "error"

        # Bind success/failure is already fully decided and logged above; sending
        # the reply is a separate, best-effort concern from here on.
        await self._reply_to_connect(channel_id, author, _CONNECT_REPLY_TEXT[outcome])

    async def _reply_to_connect(self, channel_id: str, author: str, text: str) -> None:
        """Best-effort confirmation/error reply for a ``/connect`` attempt.

        The caller (``_bind_connection``) has already fully decided and logged the
        bind outcome before this runs. A failure here is a relay-send problem, not
        a bind problem: mirroring discord.py's ``_send_connection_reply`` / slack.py's
        ``_post_connection_reply``, it never raises and logs its own,
        distinctly-worded warning, so it can never be mistaken for (or logged as) a
        failed bind. Uses a single attempt (no retry/backoff): this is a courtesy
        notification, not the delivery-critical agent-response path ``send()``
        serves, so a transient relay hiccup here should not add retry latency to
        the inbound relay read loop.
        """
        assert self._keys is not None
        try:
            event = buzz_nostr.build_chat_event(self._keys, channel_id, text, created_at=int(time.time()), mentions=(author,))
            await self._send_with_retry(lambda: self._post_event(event), max_retries=1, operation_name="connect-reply")
        except Exception:
            logger.warning("[buzz] failed to send /connect reply to pubkey=%s", author)

    def _advance_watermark(self, created_at: int) -> None:
        """Move the resubscribe cursor to *created_at*, refusing peer-supplied future timestamps.

        ``created_at`` is chosen by the event's author, and ``_subscription_filters``
        replays it as ``since`` on every reconnect. Accepting it unchecked was a
        remote denial of service: one event stamped year-5138 pinned the cursor
        there, so every subsequent REQ asked for events newer than that and the
        connector went permanently deaf with no log and no recovery short of a
        process restart.

        The cursor therefore never moves beyond ``now + MAX_FUTURE_SKEW_SECONDS``.
        An out-of-range timestamp is IGNORED rather than clamped down to the
        ceiling: clamping would still hand an attacker a blind window of exactly
        the slack, while ignoring leaves the cursor where the last plausible event
        put it. A legitimately fast-clocked member merely fails to advance the
        cursor, which costs replay (drops we re-apply) and never a miss.
        """
        if created_at <= 0:
            return
        ceiling = int(time.time()) + MAX_FUTURE_SKEW_SECONDS
        if created_at > ceiling:
            logger.debug("[buzz] ignoring future-dated created_at=%d for the resubscribe cursor (ceiling=%d)", created_at, ceiling)
            return
        self._seen_created_at = max(self._seen_created_at, created_at)

    async def _attach_connection_identity(self, inbound: InboundMessage) -> InboundMessage:
        """Resolve a persisted ``/connect`` binding for this pubkey, exactly as every sibling does.

        Without this the bind is write-only: ``connection_id`` / ``owner_user_id``
        stay ``None``, so ``ChannelManager`` runs the turn under a synthetic
        pubkey-derived user with its own memory and file buckets instead of the
        bound DeerFlow account, and revoking the connection has no runtime effect.

        ``fallback_without_workspace`` stays off (unlike discord/dingtalk/wecom,
        whose workspace is legitimately absent for DMs): ``_bind_connection``
        always writes ``workspace_id=<relay host>``, which ``__init__`` guarantees
        is non-empty, so every Buzz row is reachable by the primary lookup. Adding
        the ``None`` candidate could only ever match a row this connector did not
        write, and matching it would resolve a pubkey bound on some other relay —
        the one thing the workspace scoping exists to prevent.
        """
        return await attach_connection_identity(inbound, repo=self._connection_repo, provider="buzz", workspace_id=self._workspace_id)

    async def _handle_chat_event(self, ev: dict) -> None:
        assert self._keys is not None
        author = str(ev.get("pubkey", ""))
        channel_id_values = buzz_nostr.tag_values(ev, "h")
        if not channel_id_values or author == self._keys.pubkey_hex:
            return  # no channel tag, or our own event (no self-reply loops)
        channel_id = channel_id_values[0]
        created_at = int(ev.get("created_at", 0))
        text = str(ev.get("content", ""))

        # /connect <code> must be consulted before the allowlist gate (framework
        # ordering rule — see Channel._pending_connect_code) so a not-yet-bound
        # user can bootstrap a binding even though they aren't allowlisted yet.
        # Unlike every other gate below, a /connect message is never published:
        # _bind_connection fully handles it (valid, invalid, or erroring code)
        # and this always returns immediately after, matching every sibling
        # adapter's _bind_connection_from_connect_code (discord.py, slack.py,
        # wecom.py, dingtalk.py, wechat.py, feishu.py). Falling through to the
        # mention/allowlist gates and publishing this as an ordinary chat message
        # would let any pubkey trigger a real agent run just by prefixing a
        # message with "/connect" — Buzz's run policy sets
        # requires_bound_identity=False, so the manager has no independent
        # bound-identity check to catch that.
        code = self._pending_connect_code(text)
        if code is not None:
            await self._bind_connection(code, author, channel_id)
            # A bind attempt (valid or not) is fully processed, so it may advance the
            # cursor: leaving it behind would replay this /connect on every reconnect
            # and answer each replay with a spurious "code invalid or expired" reply.
            self._advance_watermark(created_at)
            return
        if author not in self._allowed_users:
            # Deny-by-default is intentional (see start()'s empty-allowlist warning),
            # but a silent drop is indistinguishable from a broken relay when an
            # operator forgets to allowlist someone. Debug level: an open relay can
            # carry plenty of chatter from members we never intend to serve.
            logger.debug("[buzz] dropped chat event from non-allowlisted pubkey=%s in channel %s (%s)", author, self._channel_meta.get(channel_id, {}).get("name") or "<unnamed>", channel_id)
            return

        # code is always None here: a non-None code was already fully handled
        # and returned above, so this gate only ever sees ordinary chat text.
        thread_root = self._thread_root(ev)
        mentioned = self._keys.pubkey_hex in buzz_nostr.tag_values(ev, "p")
        store = self.config.get("channel_store")
        engaged_thread = bool(thread_root and store is not None and store.get_thread_id(self.name, channel_id, topic_id=thread_root))
        allowed_without_mention = (not self._require_mention) or channel_id in self._mention_free or self._is_dm(channel_id) or engaged_thread
        if not mentioned and not allowed_without_mention:
            return

        if mentioned:
            text = self._strip_own_mention(text)

        msg_type = InboundMessageType.COMMAND if is_known_channel_command(text) else InboundMessageType.CHAT
        inbound: InboundMessage = self._make_inbound(chat_id=channel_id, user_id=author, text=text, msg_type=msg_type, thread_ts=thread_root, metadata={"event_id": str(ev.get("id", ""))})
        inbound.topic_id = thread_root
        inbound.workspace_id = self._workspace_id
        inbound = await self._attach_connection_identity(inbound)
        self._last_requester[(channel_id, thread_root)] = author
        await self._publish(inbound)
        # Only a fully accepted-and-published event advances the cursor, and only
        # after the publish actually succeeded. A dropped event must never move it
        # (that was the DoS), and a failed publish must leave it replayable.
        self._advance_watermark(created_at)

    # -- outbound --------------------------------------------------------------

    async def _post_event(self, event: dict) -> None:
        """Sign-and-post is already done by the caller; this only delivers the frame.

        Reads ``self._transport`` at call time (never cached): Task 6's relay loop
        sets it to a live WebSocket for the duration of a connection and back to
        ``None`` on disconnect, so a stale reference here would keep "succeeding"
        against a socket that is no longer attached to anything.
        """
        if self._transport is None:
            raise RuntimeError("[buzz] relay connection not established")
        await self._transport.send(buzz_nostr.event_frame(event))

    async def _edit_or_repost(self, msg: OutboundMessage, target: str, content: str, now: int, *, label: str) -> str:
        """Edit *target* in place; if that fails after retries, post a fresh message instead.

        Returns the event id later updates for this slot must target -- ``target``
        itself on success, the replacement's id after a degrade. Degrading rather
        than raising is the "never lose content" rule the placeholder path has
        always had; no mention rides the replacement because the requester was
        already notified by the original post and re-mentioning on every degraded
        edit would spam notifications.
        """
        assert self._keys is not None
        edit = buzz_nostr.build_edit_event(self._keys, msg.chat_id, target, content, created_at=now)
        try:
            await self._send_with_retry(lambda: self._post_event(edit), max_retries=3, operation_name=f"edit{label}")
            return target
        except Exception:
            fresh = buzz_nostr.build_chat_event(self._keys, msg.chat_id, content, created_at=now, reply_to=msg.thread_ts)
            await self._send_with_retry(lambda: self._post_event(fresh), max_retries=3, operation_name=f"post-degraded{label}")
            return fresh["id"]

    async def send(self, msg: OutboundMessage) -> None:
        """Post one placeholder chat message, then stream via in-place edits.

        The first message for a given ``(chat_id, thread_ts)`` is a kind-9 chat
        event (the only place a mention can ride, since kind-40003 edits carry
        only ``h``/``e`` tags — see ``buzz_nostr.build_edit_event``). Every
        subsequent update for the same key edits that placeholder in place via a
        kind-40003 event targeting its id. ``is_final`` clears the tracked target
        so the next run in this conversation starts a fresh placeholder instead of
        editing a stale, already-answered message.

        Oversize text is split by ``_chunk_text`` into <= ``EDIT_MAX_BYTES``-byte
        chunks: the first chunk rides the placeholder/edit, any remaining chunks
        ride follow-up kind-9 messages threaded to the same thread root, tracked
        per conversation in ``_stream_targets`` / ``_stream_tails`` respectively.

        Every chunk index is tracked, not just chunk 0, because the manager
        publishes CUMULATIVE text on each streaming update: an oversize reply
        therefore re-splits into >= 2 chunks on EVERY update, and posting
        ``chunks[1:]`` fresh each time flooded the channel with a near-duplicate
        tail per update (a ~100KB answer at ~1 update/sec produced dozens). A
        follow-up message is now posted only when the chunk count actually GROWS;
        an index we have already posted is edited in place, exactly like chunk 0.

        Raises whatever the underlying send raised (including ``_post_event``'s
        ``RuntimeError`` when ``_transport`` is ``None``) after retries are
        exhausted, so the framework's outer retry/error-logging path in
        ``Channel._on_outbound`` observes the failure instead of a reply being
        silently dropped. The ``is_final`` bookkeeping below runs in a
        ``finally`` block precisely so that a raised exception still propagates
        to the caller *and* still clears the stale target — see the ``finally``
        comment for why both matter.
        """
        assert self._keys is not None
        key = (msg.chat_id, msg.thread_ts)
        now = int(time.time())
        chunks = _chunk_text(msg.text)
        target = self._stream_targets.get(key)

        try:
            if target is None:
                requester = self._last_requester.get(key)
                mentions = (requester,) if requester else ()
                first = buzz_nostr.build_chat_event(self._keys, msg.chat_id, chunks[0], created_at=now, reply_to=msg.thread_ts, mentions=mentions)
                await self._send_with_retry(lambda: self._post_event(first), max_retries=3, operation_name="post")
                self._stream_targets[key] = first["id"]
            else:
                self._stream_targets[key] = await self._edit_or_repost(msg, target, chunks[0], now, label="")

            tails = self._stream_tails.get(key)
            for index, extra in enumerate(chunks[1:]):
                if tails is not None and index < len(tails):
                    tails[index] = await self._edit_or_repost(msg, tails[index], extra, now, label="-overflow")
                    continue
                follow = buzz_nostr.build_chat_event(self._keys, msg.chat_id, extra, created_at=now, reply_to=msg.thread_ts)
                await self._send_with_retry(lambda ev=follow: self._post_event(ev), max_retries=3, operation_name="post-overflow")
                if tails is None:
                    tails = self._stream_tails.setdefault(key, [])
                tails.append(follow["id"])
        finally:
            # Reviewer finding: the degrade-to-fresh-post branch above and the
            # overflow-chunk loop can both raise after their own retries are
            # exhausted, which used to propagate out of send() *before* reaching
            # an unconditional pop at the end of the function -- leaking a stale
            # (or half-updated) _stream_targets[key] entry whenever the failing
            # call had is_final=True. Once the relay recovered, the next send()
            # for this conversation would then EDIT that abandoned placeholder
            # instead of starting a fresh message, contradicting this method's
            # own contract. A `finally` clears the bookkeeping on every path --
            # success or failure -- while still letting the exception propagate
            # (a `finally` block never suppresses an in-flight exception unless
            # it itself returns/raises), so `Channel._on_outbound` still logs
            # the failure exactly as before. The overflow-tail map added for the
            # message-flood fix is cleared on exactly the same terms, so a final
            # send can never leave the next run editing this run's tail messages.
            if msg.is_final:
                self._stream_targets.pop(key, None)
                self._stream_tails.pop(key, None)
