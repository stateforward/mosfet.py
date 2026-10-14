from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import pathlib
import socket as sockets
import typing

import hsm
import mosfet.lifecycle
import pydantic
import pytest
import websockets.asyncio.client
import websockets.exceptions

from mosfet.devices import computer, hid
from mosfet.environment import Environment
from mosfet.protocols import attachment, transform, transport, yamux
from mosfet.providers import websocket
from mosfet.start import started
from tests import metric_reader
from tests.devices.computer.remote import DISPLAY_CODEC, Controller, Holder, json_message, settle, start, wait_until
from tests.websocket_fakes import Browser, FakeServe, stream_open

PEER_ID = "studio-laptop"
PATH = f"/v1/{PEER_ID}"
TOKENS = websocket.HmacTokens(pydantic.SecretStr("runtime-secret"))
TOKEN = TOKENS.issue(PEER_ID).get_secret_value()
LISTENING = "/Server/listening"
FAILED = "/Server/failed"
TIMEOUT = datetime.timedelta(seconds=2)


def make_server(
    serve: FakeServe,
    *,
    port: int = 0,
    start_timeout: datetime.timedelta = TIMEOUT,
    auth_timeout: datetime.timedelta = TIMEOUT,
    max_message_bytes: int = 1 << 20,
    max_channels: int = 8,
) -> websocket.Server:
    return websocket.Server(
        host="127.0.0.1",
        port=port,
        verifier=TOKENS,
        start_timeout=start_timeout,
        auth_timeout=auth_timeout,
        max_message_bytes=max_message_bytes,
        max_channels=max_channels,
        serve=serve,
    )


@dataclasses.dataclass
class Rig:
    environment: Environment
    server: websocket.Server
    serve: FakeServe
    endpoint: Controller


async def listening(serve: FakeServe | None = None) -> Rig:
    environment = Environment()
    fake = serve if serve is not None else FakeServe()
    server = make_server(fake)
    endpoint = Controller()
    await start(environment, endpoint, "endpoint")
    await start(environment, server, "server")
    await wait_until(lambda: server.state() == LISTENING)
    return Rig(environment=environment, server=server, serve=fake, endpoint=endpoint)


def register_event(endpoint: hsm.Instance, peer_id: str = PEER_ID) -> hsm.Event[typing.Any]:
    return attachment.AttachEvent.with_data(transport.PeerAttachData(actor=endpoint, peer_id=peer_id))


def unregister_event(endpoint: hsm.Instance) -> hsm.Event[typing.Any]:
    return attachment.DetachEvent.with_data(attachment.DetachData(actor=endpoint))


async def register(state: Rig, endpoint: hsm.Instance, peer_id: str = PEER_ID) -> None:
    _ = await hsm.dispatch(None, state.server, register_event(endpoint, peer_id))


def offers(endpoint: Controller) -> list[transport.PeerConnectedData]:
    return [event.data for event in endpoint.events if isinstance(event.data, transport.PeerConnectedData)]


def answers(endpoint: Controller) -> list[tuple[str, object]]:
    """Each registration answer the endpoint was told: created/existing, failed kind, or removed/absent."""

    told: list[tuple[str, object]] = []
    for event in endpoint.events:
        data = event.data
        if isinstance(data, attachment.AttachCompleteData):
            told.append(("complete", data.created))
        elif isinstance(data, attachment.FailedData):
            told.append(("failed", data.kind))
        elif isinstance(data, attachment.DetachedData):
            told.append(("detached", data.removed))
    return told


def test_model_starts_listens_or_fails() -> None:
    members = websocket.Server.model.members

    for state in ("starting", "listening", "failed"):
        assert f"/Server/{state}" in members
    for event in (attachment.AttachEvent, attachment.DetachEvent, websocket.AuthenticatedEvent):
        assert event.name in websocket.Server.model.events


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: make_server(FakeServe(), port=70000), "port"),
        (lambda: make_server(FakeServe(), start_timeout=datetime.timedelta(0)), "start_timeout"),
        (lambda: make_server(FakeServe(), auth_timeout=datetime.timedelta(0)), "auth_timeout"),
        (lambda: make_server(FakeServe(), max_message_bytes=0), "max_message_bytes"),
        (lambda: make_server(FakeServe(), max_channels=0), "max_channels"),
    ],
)
def test_construction_rejects_invalid_settings(build: typing.Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _ = build()


def test_a_registered_peers_connection_is_bound_to_its_endpoint() -> None:
    async def run() -> tuple[list[tuple[str, object]], list[transport.PeerConnectedData], str]:
        state = await listening()
        await register(state, state.endpoint)
        browser = Browser(state.serve.connect(PATH))
        await browser.connect(TOKEN)
        await wait_until(lambda: bool(offers(state.endpoint)))
        (offered,) = offers(state.endpoint)
        observed = offered.connection.state()
        await browser.close()
        await hsm.stop(state.server)
        return answers(state.endpoint), offers(state.endpoint), observed

    told, offered, observed = asyncio.run(run())

    assert told == [("complete", True)]
    (connected,) = offered
    assert (connected.peer_id, connected.transport) == (PEER_ID, "websocket")
    assert isinstance(connected.connection, websocket.Connection)
    assert observed == "/Connection/bound/offered"


def test_registration_is_answered_once_per_peer_and_endpoint() -> None:
    async def run() -> tuple[list[tuple[str, object]], list[tuple[str, object]]]:
        state = await listening()
        other = Controller()
        await start(state.environment, other, "other")
        await register(state, state.endpoint)
        await register(state, state.endpoint)
        await register(state, other)
        _ = await hsm.dispatch(None, state.server, attachment.AttachEvent.with_data(attachment.AttachData(actor=other)))
        await settle()
        await hsm.stop(state.server)
        return answers(state.endpoint), answers(other)

    mine, theirs = asyncio.run(run())

    assert mine == [("complete", True), ("complete", False)]
    assert theirs == [("failed", attachment.FailureKind.CONFLICT), ("failed", attachment.FailureKind.CONFLICT)]


def test_an_unregistered_peer_is_refused_4404(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> list[int]:
        state = await listening()
        socket = state.serve.connect(PATH)
        socket.deliver(json.dumps({"token": TOKEN}))
        await wait_until(socket.is_closed)
        await hsm.stop(state.server)
        return socket.close_codes

    def unknown(metrics: metric_reader.Reader) -> float:
        return sum(
            point.value
            for point in metrics.points("bot.transport.rejected.count")
            if point.attributes.get("transport") == "websocket" and point.attributes.get("reason") == "unknown_peer"
        )

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = unknown(metrics)
        codes = asyncio.run(run())
        after = unknown(metrics)

    assert codes == [4404]
    assert after - before == 1


@pytest.mark.parametrize("path", ["/v1/Not A Peer", "/v2/studio-laptop", "/computer/v1/studio-laptop", "/", ""])
def test_a_socket_on_any_other_path_is_closed_4404_without_a_connection(path: str) -> None:
    async def run() -> list[int]:
        state = await listening()
        await register(state, state.endpoint)
        socket = state.serve.connect(path)
        await wait_until(socket.is_closed)
        await hsm.stop(state.server)
        return socket.close_codes

    assert asyncio.run(run()) == [4404]


def test_only_the_registered_endpoint_can_withdraw_its_route() -> None:
    async def run() -> tuple[list[tuple[str, object]], list[tuple[str, object]], int, list[int]]:
        state = await listening()
        other = Controller()
        await start(state.environment, other, "other")
        await register(state, state.endpoint)
        _ = await hsm.dispatch(None, state.server, unregister_event(other))
        kept = Browser(state.serve.connect(PATH))
        await kept.connect(TOKEN)
        await wait_until(lambda: bool(offers(state.endpoint)))
        await kept.close()
        _ = await hsm.dispatch(None, state.server, unregister_event(state.endpoint))
        refused = state.serve.connect(PATH)
        refused.deliver(json.dumps({"token": TOKEN}))
        await wait_until(refused.is_closed)
        await hsm.stop(state.server)
        return answers(state.endpoint), answers(other), len(offers(state.endpoint)), refused.close_codes

    mine, theirs, offered, codes = asyncio.run(run())

    assert mine == [("complete", True), ("detached", True)]
    assert theirs == [("detached", False)]
    assert (offered, codes) == (1, [4404])


def test_a_registration_before_the_server_listens_is_routed_once_it_does() -> None:
    async def run() -> tuple[list[tuple[str, object]], int]:
        gate = asyncio.Event()
        environment = Environment()
        serve = FakeServe(gate=gate)
        server = make_server(serve)
        endpoint = Controller()
        await start(environment, endpoint, "endpoint")
        await start(environment, server, "server")
        _ = await hsm.dispatch(None, server, register_event(endpoint))
        await settle()
        early = answers(endpoint)
        gate.set()
        await wait_until(lambda: server.state() == LISTENING)
        browser = Browser(serve.connect(PATH))
        await browser.connect(TOKEN)
        await wait_until(lambda: bool(offers(endpoint)))
        await browser.close()
        await hsm.stop(server)
        assert early == []
        return answers(endpoint), len(offers(endpoint))

    assert asyncio.run(run()) == ([("complete", True)], 1)


@pytest.mark.parametrize(
    "serve",
    [lambda: FakeServe(error=OSError("address in use")), lambda: FakeServe(gate=asyncio.Event())],
    ids=["cannot-listen", "not-in-time"],
)
def test_a_server_that_does_not_listen_fails_and_refuses_registrations(serve: typing.Callable[[], FakeServe]) -> None:
    async def run() -> tuple[str, list[tuple[str, object]]]:
        environment = Environment()
        server = make_server(serve(), start_timeout=datetime.timedelta(milliseconds=50))
        endpoint = Controller()
        await start(environment, endpoint, "endpoint")
        await start(environment, server, "server")
        await wait_until(lambda: server.state() == FAILED)
        _ = await hsm.dispatch(None, server, register_event(endpoint))
        await settle()
        observed = server.state()
        await hsm.stop(server)
        return observed, answers(endpoint)

    assert asyncio.run(run()) == (FAILED, [("failed", attachment.FailureKind.CONFLICT)])


def test_a_closed_socket_stops_its_connection() -> None:
    async def run() -> bool:
        state = await listening()
        await register(state, state.endpoint)
        browser = Browser(state.serve.connect(PATH))
        await browser.connect(TOKEN)
        await wait_until(lambda: bool(offers(state.endpoint)))
        (offered,) = offers(state.endpoint)
        browser.socket.hang_up()
        await wait_until(lambda: not mosfet.lifecycle.is_started(offered.connection))
        await browser.close()
        await hsm.stop(state.server)
        return mosfet.lifecycle.is_started(offered.connection)

    assert asyncio.run(run()) is False


def test_stopping_the_server_closes_every_socket_and_stops_every_connection() -> None:
    """Round-3 note: hsm.stop alone would leave connections started under a cancelled context."""

    async def run() -> tuple[bool, list[int], bool]:
        state = await listening()
        await register(state, state.endpoint)
        browser = Browser(state.serve.connect(PATH))
        await browser.connect(TOKEN)
        await wait_until(lambda: bool(offers(state.endpoint)))
        (offered,) = offers(state.endpoint)
        await hsm.stop(state.server)
        await settle()
        result = (
            state.serve.listener.closed,
            browser.socket.close_codes,
            mosfet.lifecycle.is_started(offered.connection),
        )
        await browser.close()
        return result

    assert asyncio.run(run()) == (True, [1001], False)


def test_a_computer_typed_into_through_the_server_tells_its_holder_and_shows_on_the_remote_display() -> None:
    """End to end: a real Computer registers with a real Server; the remote peer opens its streams."""

    async def run() -> tuple[list[tuple[str, object]], list[bytes]]:
        environment = Environment()
        serve = FakeServe()
        server = make_server(serve)
        await start(environment, server, "server")
        await wait_until(lambda: server.state() == LISTENING)
        device = computer.Computer(device_id=PEER_ID, transport=server, person_id="p-1")
        _ = await started(environment, device, typing.cast(hsm.Model, device.model))
        holder = Holder()
        _ = await started(environment, holder, holder.model)
        _ = await hsm.dispatch(
            environment, device, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder))
        )
        await wait_until(lambda: device.state() == "/Device/attached")
        browser = Browser(serve.connect(PATH))
        await browser.connect(TOKEN)
        _ = await browser.open("display")
        _ = await browser.open("keyboard")
        show = computer.ShowEvent.with_data(
            computer.ShowData(elements=(computer.FieldElement(id="name", focused=True),))
        )
        _ = await device.dispatch(device.context(), show)
        await wait_until(lambda: bool(browser.received.get("display")))
        for key, text in (("h", "h"), ("i", "i"), ("Enter", None)):
            typed = hid.KeyEvent.with_data(hid.KeyData(action="down", key=key, code="", text=text))
            await browser.send("keyboard", json_message(typed))

        def entered() -> list[tuple[str, object]]:
            return [seen for seen in holder.observations() if seen[0] == computer.TextEnteredEvent.name]

        await wait_until(lambda: bool(entered()))
        observations = entered()
        shown = browser.received["display"]
        await browser.close()
        await hsm.stop(device)
        await hsm.stop(server)
        return observations, shown

    observations, shown = asyncio.run(run())

    assert observations == [
        (
            computer.TextEnteredEvent.name,
            computer.TextEnteredData(device_id=PEER_ID, person_id="p-1", field_id="name", text="hi"),
        )
    ]
    content = DISPLAY_CODEC.decode(shown[-1])
    assert isinstance(content.data, computer.ContentData)
    assert [element.id for element in content.data.elements] == ["name"]


def free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


@pytest.mark.integration
def test_a_remote_peer_connects_over_a_real_websocket_on_loopback() -> None:
    """The default ``websockets`` listener: path, token frame, then a yamux stream named by its first message."""

    async def run() -> tuple[int, list[str], int | None]:
        environment = Environment()
        port = free_port()
        server = websocket.Server(host="127.0.0.1", port=port, verifier=TOKENS)
        endpoint, keyboard = Controller(), Controller()
        await start(environment, endpoint, "endpoint")
        await start(environment, keyboard, "keyboard")
        await start(environment, server, "server")
        await wait_until(lambda: server.state() == LISTENING)
        codec = transform.JsonCodec(hid.KeyEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)
        channel = transport.ChannelData(name="keyboard", owner=keyboard, codec=codec, kind="data")
        _ = await hsm.dispatch(
            None,
            server,
            attachment.AttachEvent.with_data(
                transport.PeerAttachData(actor=endpoint, peer_id=PEER_ID, channels=(channel,))
            ),
        )
        typed = hid.KeyEvent.with_data(hid.KeyData(action="down", key="a", code="KeyA", text="a"))
        async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}{PATH}") as remote:
            await remote.send(json.dumps({"token": TOKEN}))
            await wait_until(lambda: bool(offers(endpoint)))
            (offered,) = offers(endpoint)
            accepted = transport.AcceptPeerData(peer_id=PEER_ID, connection_id=offered.connection_id)
            _ = await hsm.dispatch(None, offered.connection, transport.AcceptPeerEvent.with_data(accepted))
            payload = b"".join(
                len(message).to_bytes(4, "big") + message for message in (stream_open("keyboard"), codec.encode(typed))
            )
            await remote.send(yamux.FrameData.data(stream_id=1, payload=payload, flags=yamux.Flag.SYN).to_bytes())
            await wait_until(lambda: bool(keyboard.named(hid.KeyEvent)))
            ack = yamux.decode_frame(typing.cast(bytes, await asyncio.wait_for(remote.recv(), 2)))[0]
        async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}/v1/kiosk-1") as stranger:
            await stranger.send(json.dumps({"token": TOKENS.issue("kiosk-1").get_secret_value()}))
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                _ = await asyncio.wait_for(stranger.recv(), 2)
            refused = stranger.close_code
        await hsm.stop(server)
        keys = [event.data.key for event in keyboard.events if isinstance(event.data, hid.KeyData)]
        return ack.stream_id, keys, refused

    stream_id, keys, refused = asyncio.run(run())

    assert stream_id == 1
    assert keys == ["a"]
    assert refused == 4404
