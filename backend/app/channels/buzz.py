"""Buzz (Nostr) channel: DeerFlow as a member of a Buzz workspace relay.

One NIP-42-authenticated WebSocket to ``relay_url``. Inbound kind-9 chat events are
gated (pubkey allowlist, then mention/DM/thread-follow) and published to the bus;
outbound replies post one kind-9 message and then stream via kind-40003 in-place edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

from app.channels import buzz_nostr
from app.channels.base import Channel
from app.channels.commands import is_known_channel_command
from app.channels.message_bus import InboundMessage, InboundMessageType, MessageBus, OutboundMessage

logger = logging.getLogger(__name__)


class BuzzChannel(Channel):
    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="buzz", bus=bus, config=config)
        self._relay_url = str(config.get("relay_url", "")).strip()
        if not self._relay_url.startswith(("ws://", "wss://")):
            raise ValueError("channels.buzz.relay_url must be a ws:// or wss:// URL")
        self._private_key_raw = str(config.get("private_key", ""))
        self._keys: buzz_nostr.NostrKeys | None = None  # parsed in start() so coincurve stays lazy
        self._allowed_users = {buzz_nostr.parse_pubkey(v) for v in config.get("allowed_users", []) or []}
        self._require_mention = bool(config.get("require_mention", True))
        self._mention_free = {str(c) for c in config.get("mention_free_channels", []) or []}
        self._channel_meta: dict[str, dict[str, Any]] = {}
        self._stream_targets: dict[tuple[str, str | None], str] = {}
        self._last_requester: dict[tuple[str, str | None], str] = {}
        self._pending_auth_challenge: str | None = None  # set from an AUTH relay frame; consumed by Task 6's NIP-42 flow
        self._seen_created_at: int = 0  # high-water mark of accepted event created_at; Task 6 resubscribes with since=this
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
        self.bus.subscribe_outbound(self._on_outbound)
        self._spawn_connection()
        self._running = True
        logger.info("[buzz] channel started (relay=%s pubkey=%s)", self._relay_url, self._keys.pubkey_hex)

    def _spawn_connection(self) -> None:
        self._task = asyncio.create_task(self._run_loop(), name="buzz-relay-loop")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        if self._task is not None:
            task = self._task
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except Exception:
                # Task 6 replaces the stub _run_loop; until then (and even after, for a
                # genuine crash) stop() must still complete cleanly rather than re-raise
                # whatever the relay loop task ended with.
                logger.exception("[buzz] relay loop task ended with an error during stop")
            finally:
                self._task = None
        logger.info("[buzz] channel stopped")

    async def _run_loop(self) -> None:  # implemented in Task 6
        raise NotImplementedError

    # -- inbound -------------------------------------------------------------

    async def handle_relay_frame(self, raw: str) -> None:
        """Route one raw relay WebSocket frame (Task 6 calls this per frame received).

        Relay input is untrusted: a non-JSON payload, an unexpected frame shape, or
        an ``EVENT`` payload whose tags/timestamp are malformed is logged and dropped
        rather than raised, so one bad frame can never crash the read loop.
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

        Trust assumption (not currently verified): this does not check
        ``ev.get("pubkey")``. In the Buzz protocol, kind-39000 channel-discovery
        events are expected to be published by the relay's own keypair, not by
        ordinary members. As implemented, any relay member able to publish an
        event can forge one and mark an arbitrary channel ``type: "dm"``, which
        relaxes the mention gate for that channel (see ``_is_dm``) — it does NOT
        bypass the independent pubkey allowlist gate in ``_handle_chat_event``.
        Closing this gap needs a trusted relay pubkey to check the author
        against, and nothing already configured identifies one: ``relay_url`` is
        a network address, not a signing key. Deliberately left as a follow-up
        rather than inventing a new required config key (e.g. ``relay_pubkey``)
        for it here.
        """
        d_values = buzz_nostr.tag_values(ev, "d")
        if not d_values:
            return
        names = buzz_nostr.tag_values(ev, "name")
        types = buzz_nostr.tag_values(ev, "t")
        self._channel_meta[d_values[0]] = {"type": types[0] if types else "stream", "name": names[0] if names else ""}

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

    async def _bind_connection(self, code: str, author: str) -> None:
        """Consume a ``/connect <code>`` bind code for *author* (a Nostr pubkey hex).

        Always fully handles the request — valid code, invalid/expired/already-used
        code, or a connection-repo error — so the caller (``_handle_chat_event``) can
        unconditionally return right after awaiting this without ever falling through
        to ``_make_inbound``/``_publish``. Mirrors discord.py's
        ``_bind_connection_from_connect_code``, except Buzz has no outbound path yet
        (Task 5/6), so there is no confirmation reply sent back over the relay;
        success/failure is only logged.
        """
        if self._connection_repo is None:
            return  # unreachable in practice: _pending_connect_code already requires this
        try:
            state = await self._connection_repo.consume_oauth_state(provider="buzz", state=code)
            if state is None:
                logger.info("[buzz] /connect code invalid, expired, or already used (pubkey=%s)", author)
                return
            await self._connection_repo.upsert_connection(
                owner_user_id=state["owner_user_id"],
                provider="buzz",
                external_account_id=author,
                workspace_id=urlparse(self._relay_url).netloc,
                metadata={"pubkey": author},
                status="connected",
            )
            logger.info("[buzz] connected pubkey=%s to owner_user_id=%s", author, state["owner_user_id"])
        except Exception:
            # A repo/DB error binding the code must not propagate: handle_relay_frame's
            # outer guard would also catch it, but catching here keeps the log specific
            # to the bind failure instead of a generic "malformed relay event ignored".
            logger.exception("[buzz] failed to bind /connect code for pubkey=%s", author)

    async def _handle_chat_event(self, ev: dict) -> None:
        assert self._keys is not None
        author = str(ev.get("pubkey", ""))
        channel_id_values = buzz_nostr.tag_values(ev, "h")
        if not channel_id_values or author == self._keys.pubkey_hex:
            return  # no channel tag, or our own event (no self-reply loops)
        channel_id = channel_id_values[0]
        created_at = int(ev.get("created_at", 0))
        self._seen_created_at = max(self._seen_created_at, created_at)
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
            await self._bind_connection(code, author)
            return
        if author not in self._allowed_users:
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
        inbound.workspace_id = urlparse(self._relay_url).netloc
        self._last_requester[(channel_id, thread_root)] = author
        await self._publish(inbound)

    async def send(self, msg: OutboundMessage) -> None:  # implemented in Task 5
        raise NotImplementedError
