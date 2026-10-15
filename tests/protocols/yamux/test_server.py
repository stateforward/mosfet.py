import asyncio

from mosfet.protocols import yamux
from tests.protocols.yamux import wire


def test_server_uses_session_state_model() -> None:
    assert issubclass(yamux.Server, yamux.Session)
    assert "model" not in yamux.Server.__dict__
    assert yamux.Server.model is yamux.Session.model


def test_server_opens_even_stream_ids_for_its_connection() -> None:
    async def run() -> None:
        connection = await wire.start(wire.Recorder(), "connection")
        owner = await wire.start(wire.Recorder(), "owner")
        server = await wire.start(yamux.Server(connection=connection), "server")

        assert server.state() == "/Session/connected/open"

        _ = await server.dispatch(
            server.context(),
            yamux.OpenStreamEvent.with_data(yamux.OpenStreamData(owner=owner, codec=wire.codec(), channel="notes")),
        )

        assert 2 in server.take_snapshot().streams
        assert connection.frames() == [yamux.FrameData.window_update(stream_id=2, delta=0, flags=yamux.Flag.SYN)]

    asyncio.run(run())
