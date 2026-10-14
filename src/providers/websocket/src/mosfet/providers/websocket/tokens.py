"""Per-peer credentials for the WebSocket transport: who may speak for which peer.

A remote machine proves it is ``peer_id`` with a token in its first WebSocket frame. Tokens are derived
from one runtime secret, so the runtime keeps no per-peer table: the token for a peer is
``base64url(HMAC-SHA256(secret, b"mosfet.transport.v1:" + peer_id))`` without padding.
"""

import base64
import hashlib
import hmac
import typing

import pydantic

_DOMAIN = b"mosfet.transport.v1:"


class TokenVerifier(typing.Protocol):
    """Decides whether ``token`` admits a connection speaking for ``peer_id``."""

    def verify(self, peer_id: str, token: pydantic.SecretStr) -> bool:
        """True when ``token`` is ``peer_id``'s credential."""
        ...


class HmacTokens:
    """Peer tokens derived from one secret: ``issue`` mints them, ``verify`` checks them in constant time.

    Raises:
        ValueError: If ``secret`` is empty.
    """

    _secret: pydantic.SecretStr

    def __init__(self, secret: pydantic.SecretStr) -> None:
        if not secret.get_secret_value():
            raise ValueError("secret must not be empty.")
        self._secret = secret

    def issue(self, peer_id: str) -> pydantic.SecretStr:
        """The token that admits a connection for ``peer_id`` (for provisioning tools)."""

        digest = hmac.new(self._secret.get_secret_value().encode(), _DOMAIN + peer_id.encode(), hashlib.sha256).digest()
        return pydantic.SecretStr(base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii"))

    def verify(self, peer_id: str, token: pydantic.SecretStr) -> bool:
        """True when ``token`` is ``peer_id``'s token, compared in constant time."""

        expected = self.issue(peer_id).get_secret_value().encode()
        return hmac.compare_digest(expected, token.get_secret_value().encode())
