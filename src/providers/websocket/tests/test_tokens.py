from __future__ import annotations

import base64
import hashlib
import hmac

import pydantic
import pytest

from mosfet.providers import websocket

SECRET = pydantic.SecretStr("runtime-secret")


def test_a_token_is_the_unpadded_base64url_hmac_of_the_transport_domain_and_peer_id() -> None:
    digest = hmac.new(b"runtime-secret", b"mosfet.transport.v1:studio-laptop", hashlib.sha256).digest()

    token = websocket.HmacTokens(SECRET).issue("studio-laptop")

    assert token.get_secret_value() == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def test_verify_admits_only_the_peers_own_token() -> None:
    tokens = websocket.HmacTokens(SECRET)
    issued = tokens.issue("studio-laptop")

    assert tokens.verify("studio-laptop", issued)
    assert not tokens.verify("kiosk-1", issued)
    assert not tokens.verify("studio-laptop", pydantic.SecretStr(issued.get_secret_value()[:-1] + "A"))
    assert not tokens.verify("studio-laptop", pydantic.SecretStr(""))
    assert not websocket.HmacTokens(pydantic.SecretStr("other")).verify("studio-laptop", issued)


def test_a_token_never_shows_in_its_repr() -> None:
    token = websocket.HmacTokens(SECRET).issue("studio-laptop")

    assert token.get_secret_value() not in repr(token)


def test_an_empty_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="secret"):
        _ = websocket.HmacTokens(pydantic.SecretStr(""))


def test_hmac_tokens_is_a_token_verifier() -> None:
    verifier: websocket.TokenVerifier = websocket.HmacTokens(SECRET)

    assert verifier.verify("kiosk-1", websocket.HmacTokens(SECRET).issue("kiosk-1"))
