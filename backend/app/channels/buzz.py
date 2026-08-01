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
        """Cache kind-39000 channel metadata: ``d`` = channel id, ``t`` = type, ``name``."""
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
        stripped = text.lstrip()
        if stripped.startswith("@"):
            head, _, rest = stripped.partition(" ")
            return rest.strip() or stripped
        return text.strip()

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
        code = self._pending_connect_code(text)
        if code is None and author not in self._allowed_users:
            return

        thread_root = self._thread_root(ev)
        mentioned = self._keys.pubkey_hex in buzz_nostr.tag_values(ev, "p")
        store = self.config.get("channel_store")
        engaged_thread = bool(thread_root and store is not None and store.get_thread_id(self.name, channel_id, topic_id=thread_root))
        allowed_without_mention = (not self._require_mention) or channel_id in self._mention_free or self._is_dm(channel_id) or engaged_thread
        if code is None and not mentioned and not allowed_without_mention:
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
