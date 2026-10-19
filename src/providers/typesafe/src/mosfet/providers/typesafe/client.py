"""Async System One client adapter: the transport boundary to TypeSafe AI.

One bot turn maps to one `system_one(state=..., questions=...)` call. The adapter
exists so the rest of the tree never imports `typesafe_sdk` directly: the SDK's own
exceptions surface as :class:`SystemOneError`, and answers stay in typed accessors
(`choices`/`nouls`/`scores` keyed by question name).

The server is hosted TypeSafe unless ``base_url`` names a local system_one server that keeps
to the same wire format. Hosted TypeSafe is named explicitly, so a ``TYPESAFE_BASE_URL`` in the
environment never reroutes a client nobody configured. The ``TYPESAFE_API_KEY`` environment
fallback is for hosted TypeSafe only: a client given a ``base_url`` and no ``api_key`` sends the
placeholder bearer ``local``, so the hosted key never reaches a local server. The placeholder is
never the hosted key, and it is not empty: typesafe-sdk always sends ``Bearer <key>``, and an empty
key is ``Bearer `` with a trailing space, which h11 refuses before the request leaves the client.
"""

from __future__ import annotations

import typesafe_sdk as typesafe_sdk_module
from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk import constants as typesafe_constants

import collections.abc
import typing

from . import _json as json_mapping

SystemOneResponse: typing.TypeAlias = typing.Any


class SystemOneError(RuntimeError):
    """TypeSafe provider failures surfaced as one bot-package error type."""


class AsyncSystemOneClient:
    """Async system_one transport. The SDK context lifecycle stays inside ``with``."""

    _client: AsyncTypeSafeClient | None
    _base_url: str
    _api_key: str | None
    _model: str | None
    _timeout: float | None

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Address hosted TypeSafe, or the local system_one server at ``base_url``.

        ``api_key=None`` lets the SDK read ``TYPESAFE_API_KEY`` for hosted TypeSafe; with a
        ``base_url`` it is the placeholder bearer ``local`` instead.
        """

        if base_url is None:
            self._base_url = typesafe_constants.DEFAULT_BASE_URL
            self._api_key = api_key
        else:
            self._base_url = base_url
            self._api_key = "local" if api_key is None else api_key
        self._model = model
        self._timeout = timeout
        self._client = None

    async def __aenter__(self) -> typing.Self:
        self._client = AsyncTypeSafeClient(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            timeout=self._timeout,
        )
        _ = await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: typing.Any,
    ) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        _ = await client.__aexit__(exc_type, exc_value, traceback)

    async def system_one(
        self,
        state: collections.abc.Mapping[str, typing.Any],
        questions: collections.abc.Mapping[str, typing.Any],
    ) -> SystemOneResponse:
        """One state document, one question batch."""

        client = self._client
        if client is None:
            raise RuntimeError("SystemOneClient is not open; use it as an async context manager.")
        try:
            return await client.system_one(
                state=typing.cast(typing.Any, json_mapping.jsonable(state)),
                questions=questions,
            )
        except typesafe_sdk_module.TypeSafeError as error:
            raise SystemOneError(str(error)) from error


__all__ = [
    "AsyncSystemOneClient",
    "SystemOneError",
]
