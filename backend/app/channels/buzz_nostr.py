"""Pure Nostr (NIP-01) helpers for the Buzz channel connector.

No I/O, no wall-clock: callers supply ``created_at``. BIP-340 signing is done via
``coincurve``, which ships in the optional ``buzz`` dependency extra and is imported
lazily so the rest of the app never requires it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

COINCURVE_INSTALL_HINT = "The Buzz channel requires the 'buzz' extra: run `uv sync --extra buzz` (installs coincurve for BIP-340 signing)."


def _require_coincurve():
    try:
        import coincurve
    except ImportError as exc:  # pragma: no cover - exercised via BuzzChannel.start
        raise RuntimeError(COINCURVE_INSTALL_HINT) from exc
    return coincurve


@dataclass(frozen=True)
class NostrKeys:
    secret: bytes
    pubkey_hex: str


def _bech32_polymod(values: list[int]) -> int:
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if ((b >> i) & 1) else 0
    return chk


def _bech32_decode(expected_hrp: str, value: str) -> bytes:
    if "1" not in value:
        raise ValueError(f"not bech32: {value!r}")
    hrp, data_part = value.rsplit("1", 1)
    if hrp != expected_hrp:
        raise ValueError(f"expected {expected_hrp!r} bech32, got {hrp!r}")
    try:
        data = [_BECH32_CHARSET.index(c) for c in data_part]
    except ValueError as exc:
        raise ValueError(f"invalid bech32 character in {value!r}") from exc
    hrp_expanded = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    if _bech32_polymod(hrp_expanded + data) != 1:
        raise ValueError(f"bad bech32 checksum in {value!r}")
    acc = bits = 0
    out = bytearray()
    for v in data[:-6]:
        acc = (acc << 5) | v
        bits += 5
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    if len(out) != 32:
        raise ValueError(f"expected 32-byte payload in {value!r}")
    return bytes(out)


def _parse_32_bytes(value: str, bech_hrp: str) -> bytes:
    value = value.strip()
    if value.lower().startswith(f"{bech_hrp}1"):
        return _bech32_decode(bech_hrp, value.lower())
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"expected 64-hex or {bech_hrp}1... value") from exc
    if len(raw) != 32:
        raise ValueError("expected exactly 32 bytes")
    return raw


def parse_private_key(value: str) -> NostrKeys:
    secret = _parse_32_bytes(value, "nsec")
    coincurve = _require_coincurve()
    pubkey = coincurve.PrivateKey(secret).public_key.format(compressed=True)[1:]
    return NostrKeys(secret=secret, pubkey_hex=pubkey.hex())


def parse_pubkey(value: str) -> str:
    return _parse_32_bytes(value, "npub").hex()


def event_id(pubkey_hex: str, created_at: int, kind: int, tags: list[list[str]], content: str) -> str:
    payload = json.dumps([0, pubkey_hex, created_at, kind, tags, content], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sign_event(keys: NostrKeys, kind: int, tags: list[list[str]], content: str, created_at: int) -> dict:
    coincurve = _require_coincurve()
    eid = event_id(keys.pubkey_hex, created_at, kind, tags, content)
    sig = coincurve.PrivateKey(keys.secret).sign_schnorr(bytes.fromhex(eid))
    return {"id": eid, "pubkey": keys.pubkey_hex, "created_at": created_at, "kind": kind, "tags": tags, "content": content, "sig": sig.hex()}
