"""System One client contract tests: the SDK client is replaced by a recording fake, or reaches a loopback server."""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import pytest
import typesafe_sdk
from typesafe_sdk import constants as typesafe_constants

from mosfet.providers.typesafe import AsyncSystemOneClient, SystemOneError
from mosfet.providers.typesafe import client as client_module


class _RecordingSdkClient:
    """Stands in for `typesafe_sdk.AsyncTypeSafeClient`: records how it was built and what it was asked."""

    built: typing.ClassVar[list[dict[str, object]]] = []
    failure: typing.ClassVar[BaseException | None] = None

    def __init__(self, **kwargs: object) -> None:
        _RecordingSdkClient.built.append(kwargs)

    async def __aenter__(self) -> typing.Self:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def system_one(
        self,
        *,
        state: object,
        questions: collections.abc.Mapping[str, object],
    ) -> dict[str, object]:
        failure = _RecordingSdkClient.failure
        if failure is not None:
            raise failure
        return {"state": state, "questions": dict(questions)}


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingSdkClient]:
    _RecordingSdkClient.built = []
    _RecordingSdkClient.failure = None
    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", _RecordingSdkClient)
    return _RecordingSdkClient


def _ask(client: AsyncSystemOneClient) -> object:
    async def turn() -> object:
        async with client as opened:
            return await opened.system_one(state={"turn": "ring"}, questions={})

    return asyncio.run(turn())


def test_no_base_url_is_hosted_typesafe_with_the_sdk_key_fallback(sdk: type[_RecordingSdkClient]) -> None:
    """Hosted TypeSafe is named explicitly, so a stray `TYPESAFE_BASE_URL` never reroutes it (D-11-11)."""

    _ = _ask(AsyncSystemOneClient(model="system-one", timeout=2.0))

    assert sdk.built == [
        {
            "base_url": typesafe_constants.DEFAULT_BASE_URL,
            "api_key": None,
            "model": "system-one",
            "timeout": 2.0,
        }
    ]


def test_a_base_url_with_no_key_reaches_the_server_with_a_placeholder_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless local request leaves the client, and the hosted key never reaches it (D-R-23).

    This drives a real loopback server, so the SDK's headers pass h11's validation: an empty
    key would be ``Bearer `` with a trailing space, which h11 refuses before any byte is sent.
    """

    monkeypatch.setenv("TYPESAFE_API_KEY", "hosted-secret")
    received: list[str] = []
    answer = (
        b'{"model": "decide", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {"tone": '
        b'{"type": "choice", "choice": "calm", "confidence": 0.9, "probabilities": {"calm": 0.9, "angry": 0.1}}}}'
    )

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
        except asyncio.IncompleteReadError:
            writer.close()
            return
        headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in head[1:])}
        _ = await reader.readexactly(int(headers.get("content-length", "0")))
        received.append(headers.get("authorization", ""))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
            + f"Content-Length: {len(answer)}\r\n\r\n".encode()
            + answer
        )
        await writer.drain()
        writer.close()

    async def turn() -> str:
        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        # An AF_INET socket names itself as (host, port).
        address = typing.cast(tuple[str, int], server.sockets[0].getsockname())
        async with server, AsyncSystemOneClient(base_url=f"http://127.0.0.1:{address[1]}") as opened:
            # The adapter passes the SDK's SystemOneResponse through untouched.
            decided = typing.cast(
                typesafe_sdk.SystemOneResponse,
                await opened.system_one(
                    state={"turn": "ring"},
                    questions={
                        "tone": typesafe_sdk.Choice(
                            instructions="What is the tone?", criteria={"calm": None, "angry": None}
                        )
                    },
                ),
            )
        return decided.choices["tone"].choice

    assert asyncio.run(turn()) == "calm"
    assert received == ["Bearer local"]


def test_a_base_url_with_a_key_sends_both(sdk: type[_RecordingSdkClient]) -> None:
    _ = _ask(AsyncSystemOneClient(base_url="http://127.0.0.1:8102", api_key="local-key"))

    assert sdk.built == [{"base_url": "http://127.0.0.1:8102", "api_key": "local-key", "model": None, "timeout": None}]


@pytest.mark.usefixtures("sdk")
def test_the_state_reaches_the_sdk_as_json() -> None:
    answer = _ask(AsyncSystemOneClient(base_url="http://127.0.0.1:8102"))

    assert answer == {"state": {"turn": "ring"}, "questions": {}}


def test_an_sdk_failure_surfaces_as_a_system_one_error(sdk: type[_RecordingSdkClient]) -> None:
    sdk.failure = typesafe_sdk.TypeSafeError("Invalid response data at 'model'")

    with pytest.raises(SystemOneError, match="Invalid response data"):
        _ = _ask(AsyncSystemOneClient(base_url="http://127.0.0.1:8102"))


def test_asking_before_opening_is_a_runtime_error() -> None:
    async def turn() -> object:
        return await AsyncSystemOneClient().system_one(state={}, questions={})

    with pytest.raises(RuntimeError, match="not open"):
        _ = asyncio.run(turn())
