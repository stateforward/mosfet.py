import asyncio

from mosfet.protocols import yamux
from tests.protocols.yamux import wire


def test_client_uses_session_state_model() -> None:
    assert issubclass(yamux.Client, yamux.Session)
    assert "model" not in yamux.Client.__dict__
    assert yamux.Client.model is yamux.Session.model


def test_client_opens_odd_stream_ids_for_its_connection() -> None:
    async def run() -> None:
        connection = await wire.start(wire.Recorder(), "connection")
        owner = await wire.start(wire.Recorder(), "owner")
        client = await wire.start(yamux.Client(connection=connection), "client")

        assert client.state() == "/Session/connected/open"

        _ = await client.dispatch(
            client.context(),
            yamux.OpenStreamEvent.with_data(yamux.OpenStreamData(owner=owner, codec=wire.codec(), channel="notes")),
        )

        assert 1 in client.take_snapshot().streams
        assert connection.frames() == [yamux.FrameData.window_update(stream_id=1, delta=0, flags=yamux.Flag.SYN)]

    asyncio.run(run())
