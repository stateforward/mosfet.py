import asyncio

import bot

from bot.protocols.yamux.events import OpenStreamEvent, OpenStreamData
from bot.protocols.yamux.server import Server
from bot.protocols.yamux.session import Session


def test_server_uses_session_state_model() -> None:
    assert issubclass(Server, Session)
    assert "model" not in Server.__dict__
    assert Server.model is Session.model


def test_server_constructs_server_role_session() -> None:
    async def run() -> None:
        server = Server()
        _ = await bot.started(None, server, server.model)

        assert server.state() == "/Session/connected/open"

        await server.dispatch(server.context(), OpenStreamEvent.with_data(OpenStreamData()))

        assert 2 in server.take_snapshot().streams

    asyncio.run(run())
