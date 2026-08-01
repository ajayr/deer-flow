"""Tests for the pure Nostr helpers behind the Buzz channel connector."""

import pytest

coincurve = pytest.importorskip("coincurve")

from app.channels import buzz_nostr  # noqa: E402

SK3_HEX = "0000000000000000000000000000000000000000000000000000000000000003"
SK3_NSEC = "nsec1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqps52s3re"
PK3_HEX = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"
PK3_NPUB = "npub1lycg5qvjtrp3qjf5f7zl382j9x6nrjz9sdhenvyxq8c3808qxmus6gq266"
CHANNEL = "136852ee-63e1-49c2-8927-413b5ee8e5f7"


def test_parse_private_key_hex_derives_xonly_pubkey():
    keys = buzz_nostr.parse_private_key(SK3_HEX)
    assert keys.secret == bytes.fromhex(SK3_HEX)
    assert keys.pubkey_hex == PK3_HEX


def test_parse_private_key_nsec_matches_hex():
    assert buzz_nostr.parse_private_key(SK3_NSEC) == buzz_nostr.parse_private_key(SK3_HEX)


@pytest.mark.parametrize("bad", ["", "zz" * 32, "nsec1invalid", "npub1lycg5qvjtrp3qjf5f7zl382j9x6nrjz9sdhenvyxq8c3808qxmus6gq266"])
def test_parse_private_key_rejects_garbage(bad):
    with pytest.raises(ValueError):
        buzz_nostr.parse_private_key(bad)


def test_parse_pubkey_accepts_hex_and_npub():
    assert buzz_nostr.parse_pubkey(PK3_HEX.upper()) == PK3_HEX
    assert buzz_nostr.parse_pubkey(PK3_NPUB) == PK3_HEX


def test_event_id_matches_nip01_reference_vector():
    eid = buzz_nostr.event_id(PK3_HEX, 1700000000, 9, [["h", CHANNEL]], "hello buzz")
    assert eid == "6aa2ef0a72e39e52ac7c3680e6a76ed75c90340e684148da6221086b443d2089"


def test_sign_event_produces_valid_schnorr_signature():
    keys = buzz_nostr.parse_private_key(SK3_HEX)
    ev = buzz_nostr.sign_event(keys, 9, [["h", CHANNEL]], "hello buzz", created_at=1700000000)
    assert ev["id"] == "6aa2ef0a72e39e52ac7c3680e6a76ed75c90340e684148da6221086b443d2089"
    assert ev["pubkey"] == PK3_HEX and ev["kind"] == 9 and ev["tags"] == [["h", CHANNEL]]
    xonly = coincurve.PublicKeyXOnly(bytes.fromhex(PK3_HEX))
    assert xonly.verify(bytes.fromhex(ev["sig"]), bytes.fromhex(ev["id"]))
