import asyncio

import hsm

from bot.protocols.yamux.client import Client
from bot.protocols.yamux.events import OpenStreamEvent, OpenStreamData
from bot.protocols.yamux.session import Session


def test_client_uses_session_state_model() -> None:
    assert issubclass(Client, Session)
    assert "model" not in Client.__dict__
    assert Client.model is Session.model


def test_client_constructs_client_role_session() -> None:
    async def run() -> None:
        client = Client()
        _ = await hsm.started(None, client, client.model)

        assert client.state() == "/Session/connected/open"

        await client.dispatch(client.context(), OpenStreamEvent.with_data(OpenStreamData()))

        assert 1 in client.take_snapshot().streams

    asyncio.run(run())
