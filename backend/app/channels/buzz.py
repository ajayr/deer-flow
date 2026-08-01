"""Buzz (Nostr) channel: DeerFlow as a member of a Buzz workspace relay.

One NIP-42-authenticated WebSocket to ``relay_url``. Inbound kind-9 chat events are
gated (pubkey allowlist, then mention/DM/thread-follow) and published to the bus;
outbound replies post one kind-9 message and then stream via kind-40003 in-place edits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.channels import buzz_nostr
from app.channels.base import Channel
from app.channels.message_bus import MessageBus, OutboundMessage

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

    async def send(self, msg: OutboundMessage) -> None:  # implemented in Task 5
        raise NotImplementedError
