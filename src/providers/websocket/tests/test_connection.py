from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import pathlib
import typing

import hsm
import mosfet.lifecycle
import pydantic
import pytest

from mosfet.devices import computer
from mosfet.environment import Environment
from mosfet.protocols import transform, transport, yamux
from mosfet.providers import websocket
from mosfet.start import started
from tests import metric_reader
from tests.devices.computer.remote import Controller, settle, start, wait_until
from tests.websocket_fakes import Browser, FakeSocket

PEER_ID = "studio-laptop"
TOKENS = websocket.HmacTokens(pydantic.SecretStr("runtime-secret"))
TOKEN = TOKENS.issue(PEER_ID).get_secret_value()
SHORT = datetime.timedelta(milliseconds=200)
LONG = datetime.timedelta(seconds=2)
MAX_MESSAGE = 4096
CLOSED = "/Connection/closed"
OFFERED = "/Connection/bound/offered"
ACCEPTED = "/Connection/bound/accepted"


class NoteData(pydantic.BaseModel):
    """A short note a port and the remote peer exchange on a stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str


NoteEvent = hsm.Event[NoteData](name="test.websocket.note", schema=NoteData)
CODEC = transform.JsonCodec(NoteEvent, max_message_bytes=1024)


def note(text: str) -> hsm.Event[NoteData]:
    return NoteEvent.with_data(NoteData(text=text))


@dataclasses.dataclass
class Rig:
    socket: FakeSocket
    connection: websocket.Connection
    server: Controller
    endpoint: Controller
    keyboard: Controller
    display: Controller
    camera: Controller
    browser: Browser


async def rig(
    *,
    auth_timeout: datetime.timedelta = LONG,
    max_channels: int = 8,
    socket: FakeSocket | None = None,
) -> Rig:
    environment = Environment()
    server, endpoint, keyboard, display, camera = Controller(), Controller(), Controller(), Controller(), Controller()
    for machine, name in (
        (server, "server"),
        (endpoint, "endpoint"),
        (keyboard, "keyboard"),
        (display, "display"),
        (camera, "camera"),
    ):
        await start(environment, machine, name)
    resolved = socket if socket is not None else FakeSocket()
    connection = websocket.Connection(
        socket=resolved,
        peer_id=PEER_ID,
        server=server,
        verifier=TOKENS,
        auth_timeout=auth_timeout,
        max_message_bytes=MAX_MESSAGE,
        max_channels=max_channels,
    )
    _ = await started(None, connection, websocket.Connection.model)
    return Rig(resolved, connection, server, endpoint, keyboard, display, camera, Browser(resolved))


def channels(state: Rig) -> tuple[transport.ChannelData, ...]:
    return (
        transport.ChannelData(name="keyboard", owner=state.keyboard, codec=CODEC, kind="data"),
        transport.ChannelData(name="display", owner=state.display, codec=CODEC, kind="data"),
        transport.ChannelData(name="camera", owner=state.camera, codec=computer.StillCodec(), kind="data"),
    )


async def bind(state: Rig) -> transport.PeerConnectedData:
    """The remote peer authenticates and the server binds the connection to the endpoint."""

    await state.browser.connect(TOKEN)
    await wait_until(lambda: bool(state.server.named(websocket.AuthenticatedEvent)))
    bound = websocket.BoundEvent.with_data(websocket.BoundData(endpoint=state.endpoint, channels=channels(state)))
    _ = await hsm.dispatch(None, state.connection, bound)
    await wait_until(lambda: bool(state.endpoint.named(transport.PeerConnectedEvent)))
    (offered,) = state.endpoint.named(transport.PeerConnectedEvent)
    assert isinstance(offered.data, transport.PeerConnectedData)
    return offered.data


async def accept(state: Rig, offered: transport.PeerConnectedData) -> None:
    accepted = transport.AcceptPeerData(peer_id=offered.peer_id, connection_id=offered.connection_id)
    _ = await hsm.dispatch(None, state.connection, transport.AcceptPeerEvent.with_data(accepted))
    await wait_until(lambda: state.connection.state() == ACCEPTED)


async def close_peer(
    state: Rig, offered: transport.PeerConnectedData, reason: typing.Literal["superseded", "released"]
) -> None:
    closed = transport.ClosePeerData(peer_id=offered.peer_id, connection_id=offered.connection_id, reason=reason)
    _ = await hsm.dispatch(None, state.connection, transport.ClosePeerEvent.with_data(closed))


async def teardown(state: Rig) -> None:
    await state.browser.close()
    if mosfet.lifecycle.is_started(state.connection):
        await hsm.stop(state.connection)


def notes(port: Controller) -> list[str]:
    return [event.data.text for event in port.events if isinstance(event.data, NoteData)]


def disconnects(endpoint: Controller) -> list[str]:
    return [event.data.reason for event in endpoint.events if isinstance(event.data, transport.PeerDisconnectedData)]


def counted(metrics: metric_reader.Reader, name: str, key: str) -> dict[object, float]:
    return {
        point.attributes.get(key): point.value
        for point in metrics.points(name)
        if point.attributes.get("transport", "websocket") == "websocket"
    }


def test_model_authenticates_routes_offers_then_accepts() -> None:
    members = websocket.Connection.model.members

    for state in ("authenticating", "awaiting_route", "bound", "bound/offered", "bound/accepted", "closing", "closed"):
        assert f"/Connection/{state}" in members
    for event in (websocket.BoundEvent, transport.AcceptPeerEvent, transport.ClosePeerEvent, yamux.TransmitEvent):
        assert event.name in websocket.Connection.model.events


def _connection(
    *,
    peer_id: str = PEER_ID,
    auth_timeout: datetime.timedelta = SHORT,
    max_message_bytes: int = MAX_MESSAGE,
    max_channels: int = 8,
) -> websocket.Connection:
    return websocket.Connection(
        socket=FakeSocket(),
        peer_id=peer_id,
        server=Controller(),
        verifier=TOKENS,
        auth_timeout=auth_timeout,
        max_message_bytes=max_message_bytes,
        max_channels=max_channels,
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: _connection(auth_timeout=datetime.timedelta(0)), "auth_timeout"),
        (lambda: _connection(max_message_bytes=0), "max_message_bytes"),
        (lambda: _connection(max_channels=0), "max_channels"),
        (lambda: _connection(peer_id="not a peer/"), "peer"),
    ],
)
def test_construction_rejects_invalid_settings(build: typing.Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _ = build()


def test_a_verified_token_announces_the_connection_to_the_server() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], websocket.Connection, str]:
        state = await rig()
        state.socket.deliver(json.dumps({"token": TOKEN}))
        await wait_until(lambda: bool(state.server.named(websocket.AuthenticatedEvent)))
        await settle()
        observed = state.connection.state()
        await teardown(state)
        return state.server.named(websocket.AuthenticatedEvent), state.connection, observed

    announced, connection, observed = asyncio.run(run())

    (event,) = announced
    assert isinstance(event.data, websocket.AuthenticatedData)
    assert (event.data.peer_id, event.data.connection) == (PEER_ID, connection)
    assert observed == "/Connection/awaiting_route"


@pytest.mark.parametrize(
    "first",
    [
        json.dumps({"token": "not-the-token"}),
        json.dumps({"token": TOKENS.issue("kiosk-1").get_secret_value()}),
        json.dumps({"token": TOKEN, "peer": "kiosk-1"}),
        "not json",
        TOKEN.encode(),
    ],
)
def test_a_wrong_or_malformed_first_frame_closes_4401(
    first: str | bytes, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[int], int]:
        state = await rig()
        state.socket.deliver(first)
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes, len(state.server.named(websocket.AuthenticatedEvent))

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.rejected.count", "reason").get("unauthenticated", 0.0)
        codes, announced = asyncio.run(run())
        after = counted(metrics, "bot.transport.rejected.count", "reason").get("unauthenticated", 0.0)

    assert (codes, announced) == ([4401], 0)
    assert after - before == 1


def test_no_token_within_the_auth_timeout_closes_4408(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> list[int]:
        state = await rig(auth_timeout=SHORT)
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.connect.count", "outcome").get("timed_out", 0.0)
        codes = asyncio.run(run())
        after = counted(metrics, "bot.transport.connect.count", "outcome").get("timed_out", 0.0)

    assert codes == [4408]
    assert after - before == 1


def test_no_route_within_the_auth_timeout_closes_4408() -> None:
    async def run() -> list[int]:
        state = await rig(auth_timeout=SHORT)
        state.socket.deliver(json.dumps({"token": TOKEN}))
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes

    assert asyncio.run(run()) == [4408]


def test_a_refusal_closes_4404() -> None:
    async def run() -> list[int]:
        state = await rig()
        state.socket.deliver(json.dumps({"token": TOKEN}))
        await wait_until(lambda: bool(state.server.named(websocket.AuthenticatedEvent)))
        _ = await hsm.dispatch(None, state.connection, websocket.RefusedEvent.with_data(websocket.RefusedData()))
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes

    assert asyncio.run(run()) == [4404]


def test_a_remote_hang_up_before_the_token_just_ends_the_connection() -> None:
    async def run() -> list[int]:
        state = await rig()
        state.socket.hang_up()
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes

    assert asyncio.run(run()) == []


def test_bound_it_offers_itself_to_the_endpoint_as_a_websocket_connection(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[transport.PeerConnectedData, websocket.Connection, str, str]:
        state = await rig()
        offered = await bind(state)
        (event,) = state.endpoint.named(transport.PeerConnectedEvent)
        observed = state.connection.state()
        source = event.source
        connection_id = hsm.id(state.connection)
        await teardown(state)
        assert source == connection_id
        return offered, state.connection, observed, connection_id

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.connect.count", "outcome").get("connected", 0.0)
        offered, connection, observed, _ = asyncio.run(run())
        after = counted(metrics, "bot.transport.connect.count", "outcome").get("connected", 0.0)

    assert (offered.peer_id, offered.transport, offered.connection) == (PEER_ID, "websocket", connection)
    assert len(offered.connection_id) == 32
    assert observed == OFFERED
    assert after - before == 1


def test_a_stream_opened_before_acceptance_is_held_unread_then_reaches_its_port_on_accept() -> None:
    async def run() -> tuple[list[str], list[str], str, str, int]:
        state = await rig()
        offered = await bind(state)
        _ = await state.browser.open("keyboard")
        await state.browser.send("keyboard", CODEC.encode(note("early")))
        await settle(40)
        before = notes(state.keyboard)
        await accept(state, offered)
        await wait_until(lambda: bool(notes(state.keyboard)))
        await state.browser.send("keyboard", CODEC.encode(note("later")))
        await wait_until(lambda: len(notes(state.keyboard)) == 2)
        (first,) = [event for event in state.keyboard.events if isinstance(event.data, NoteData)][:1]
        told = [event.name for event in state.keyboard.events]
        await teardown(state)
        return before, notes(state.keyboard), first.target, first.source, told.count(transform.OwnedEvent.name)

    before, after, target, source, owned = asyncio.run(run())

    assert before == []
    assert after == ["early", "later"]
    assert target == "keyboard"
    assert source not in ("", "keyboard")
    # The keyboard's codec encodes, so it is told which stream it owns, once.
    assert owned == 1


def test_a_duplex_port_is_told_its_stream_and_what_it_sends_reaches_the_remote_peer() -> None:
    async def run() -> tuple[str, list[bytes]]:
        state = await rig()
        await accept(state, await bind(state))
        _ = await state.browser.open("display")
        await wait_until(lambda: bool(state.display.named(transform.OwnedEvent)))
        (owned,) = state.display.named(transform.OwnedEvent)
        assert isinstance(owned.data, transform.OwnedData)
        _ = await hsm.dispatch(None, owned.data.stream, note("shown"))
        await wait_until(lambda: bool(state.browser.received.get("display")))
        channel = owned.data.channel
        await teardown(state)
        return channel, state.browser.received["display"]

    channel, received = asyncio.run(run())

    assert channel == "display"
    assert received == [CODEC.encode(note("shown"))]


def test_a_still_larger_than_the_initial_stream_window_reaches_the_camera_port() -> None:
    content = b"\x89PNG\r\n\x1a\n" + b"\x00" * yamux.INITIAL_STREAM_WINDOW
    message = bytes([len(b"image/png")]) + b"image/png" + content
    framed = len(message).to_bytes(4, "big") + message

    def snapshot(stream: hsm.Instance) -> yamux.StreamSnapshot:
        taken = hsm.take_snapshot(None, stream)
        assert isinstance(taken, yamux.StreamSnapshot)
        return taken

    async def run() -> list[hsm.Event[typing.Any]]:
        state = await rig()
        await accept(state, await bind(state))
        stream = await state.browser.open("camera")
        # The peer sends only what the window allows, so the stream must be granted room for the whole still.
        await wait_until(lambda: snapshot(stream).send_window >= len(framed))
        # Written as raw frames: the fake browser's own stream queues at most one initial window to send.
        chunk = MAX_MESSAGE - yamux.HEADER_LENGTH
        for offset in range(0, len(framed), chunk):
            frame = yamux.FrameData.data(stream_id=snapshot(stream).stream_id, payload=framed[offset : offset + chunk])
            state.socket.deliver(frame.to_bytes())
        await wait_until(lambda: bool(state.camera.named(computer.StillEvent)))
        await teardown(state)
        return state.camera.named(computer.StillEvent)

    (still,) = asyncio.run(run())

    assert still.data == computer.StillCodec().decode(message).data


def test_a_channel_the_endpoint_did_not_register_is_reset_and_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[str], int]:
        state = await rig()
        await accept(state, await bind(state))
        _ = await state.browser.open("printer")
        await wait_until(lambda: bool(state.browser.ended.get("printer")))
        await teardown(state)
        told = [event.name for port in (state.keyboard, state.display) for event in port.events]
        return state.browser.ended["printer"], told.count(transform.OwnedEvent.name)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.rejected.count", "reason").get("refused_channel", 0.0)
        ended, told = asyncio.run(run())
        after = counted(metrics, "bot.transport.rejected.count", "reason").get("refused_channel", 0.0)

    assert (ended, told) == (["reset"], 0)
    assert after - before == 1


@pytest.mark.parametrize(
    ("first", "reason"),
    [(b"not json", "malformed"), (b'{"name":"keyboard","v":1}', "schema_invalid"), (b"x" * 1025, "oversize")],
)
def test_a_stream_whose_first_message_is_not_a_stream_open_is_reset_and_counted(
    first: bytes, reason: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[str]:
        state = await rig()
        await accept(state, await bind(state))
        _ = await state.browser.open("keyboard", first=first)
        await wait_until(lambda: bool(state.browser.ended.get("keyboard")))
        await teardown(state)
        return state.browser.ended["keyboard"]

    def failed(metrics: metric_reader.Reader) -> float:
        return sum(
            point.value
            for point in metrics.points("bot.transform.decode.failed.count")
            if point.attributes.get("codec") == "stream_open" and point.attributes.get("reason") == reason
        )

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = failed(metrics)
        ended = asyncio.run(run())
        after = failed(metrics)

    assert ended == ["reset"]
    assert after - before == 1


def test_streams_past_max_channels_are_reset_and_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[str], list[str], list[str]]:
        state = await rig(max_channels=1)
        await accept(state, await bind(state))
        _ = await state.browser.open("keyboard")
        _ = await state.browser.open("display")
        await wait_until(lambda: bool(state.browser.ended.get("display")))
        await state.browser.send("keyboard", CODEC.encode(note("kept")))
        await wait_until(lambda: bool(notes(state.keyboard)))
        await teardown(state)
        return state.browser.ended["display"], notes(state.keyboard), notes(state.display)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.rejected.count", "reason").get("too_many_channels", 0.0)
        ended, kept, refused = asyncio.run(run())
        after = counted(metrics, "bot.transport.rejected.count", "reason").get("too_many_channels", 0.0)

    assert (ended, kept, refused) == (["reset"], ["kept"], [])
    assert after - before == 1


@pytest.mark.parametrize(("reason", "code"), [("superseded", 4409), ("released", 1000)])
def test_the_endpoint_closes_the_connection_with_its_reason(
    reason: typing.Literal["superseded", "released"], code: int
) -> None:
    async def run() -> tuple[list[int], list[str]]:
        state = await rig()
        offered = await bind(state)
        await close_peer(state, offered, reason)
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return state.socket.close_codes, disconnects(state.endpoint)

    assert asyncio.run(run()) == ([code], ["closed"])


def test_an_accept_or_close_for_another_connection_is_ignored() -> None:
    async def run() -> tuple[str, list[int]]:
        state = await rig()
        offered = await bind(state)
        stale = offered.model_copy(update={"connection_id": "0" * 32})
        _ = await hsm.dispatch(
            None,
            state.connection,
            transport.AcceptPeerEvent.with_data(transport.AcceptPeerData(peer_id=PEER_ID, connection_id="0" * 32)),
        )
        await close_peer(state, stale, "superseded")
        await settle(40)
        observed = state.connection.state()
        await teardown(state)
        return observed, state.socket.close_codes

    observed, codes = asyncio.run(run())

    assert observed == OFFERED
    assert 4409 not in codes


def test_a_remote_hang_up_tells_the_endpoint_the_peer_left() -> None:
    async def run() -> tuple[list[str], list[int]]:
        state = await rig()
        _ = await bind(state)
        state.socket.hang_up()
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return disconnects(state.endpoint), state.socket.close_codes

    assert asyncio.run(run()) == (["left"], [])


def test_a_data_frame_over_the_limit_closes_4413(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[str], list[int]]:
        state = await rig()
        _ = await bind(state)
        oversize = yamux.FrameData.data(stream_id=1, payload=b"x" * (MAX_MESSAGE + 1), flags=yamux.Flag.SYN)
        # Only the header arrives: the frame is refused on what it announces, before its payload.
        state.socket.deliver(oversize.to_bytes()[: yamux.HEADER_LENGTH])
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return disconnects(state.endpoint), state.socket.close_codes

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.rejected.count", "reason").get("oversize", 0.0)
        told, codes = asyncio.run(run())
        after = counted(metrics, "bot.transport.rejected.count", "reason").get("oversize", 0.0)

    assert (told, codes) == (["transport_lost"], [4413])
    assert after - before == 1


@pytest.mark.parametrize(
    "message",
    [
        "text after the token",
        b"\x07" + b"\x00" * 11,
        # Data on a stream the remote peer never opened: the session fails and ends.
        yamux.FrameData.data(stream_id=9, payload=b"stray").to_bytes(),
    ],
    ids=["text", "not-yamux", "session-protocol-error"],
)
def test_a_message_that_breaks_yamux_closes_1002(message: str | bytes) -> None:
    async def run() -> tuple[list[str], list[int]]:
        state = await rig()
        _ = await bind(state)
        state.socket.deliver(message)
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return disconnects(state.endpoint), state.socket.close_codes

    assert asyncio.run(run()) == (["transport_lost"], [1002])


def test_a_go_away_with_nothing_open_closes_1000_and_the_peer_left() -> None:
    async def run() -> tuple[list[str], list[int]]:
        state = await rig()
        _ = await bind(state)
        session = state.browser.session
        _ = await hsm.dispatch(session.context(), session, yamux.GoAwayEvent.with_data(yamux.GoAwayData(code=0)))
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return disconnects(state.endpoint), state.socket.close_codes

    assert asyncio.run(run()) == (["left"], [1000])


def test_a_socket_that_cannot_keep_up_closes_1011(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[str], list[int]]:
        gate = asyncio.Event()
        state = await rig(socket=FakeSocket(send_gate=gate))
        await accept(state, await bind(state))
        _ = await state.browser.open("display")
        await wait_until(lambda: bool(state.display.named(transform.OwnedEvent)))
        (owned,) = state.display.named(transform.OwnedEvent)
        assert isinstance(owned.data, transform.OwnedData)
        # The socket takes nothing: queued frames pass four times the largest message.
        for index in range(24):
            _ = await hsm.dispatch(None, owned.data.stream, note(f"{index:04d}" + "x" * 900))
        await wait_until(lambda: state.connection.state() == CLOSED)
        await teardown(state)
        return disconnects(state.endpoint), state.socket.close_codes

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counted(metrics, "bot.transport.send.dropped.count", "reason").get("backlog", 0.0)
        told, codes = asyncio.run(run())
        after = counted(metrics, "bot.transport.send.dropped.count", "reason").get("backlog", 0.0)

    assert (told, codes) == (["transport_lost"], [1011])
    assert after - before == 1


def test_stopping_a_bound_connection_tells_the_endpoint_and_ends_every_stream(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[str], list[str]]:
        state = await rig()
        await accept(state, await bind(state))
        _ = await state.browser.open("display")
        await wait_until(lambda: bool(state.display.named(transform.OwnedEvent)))
        await hsm.stop(state.connection)
        await wait_until(lambda: bool(state.display.named(transform.EndedEvent)))
        ended = [event.data.reason for event in state.display.events if isinstance(event.data, transform.EndedData)]
        await teardown(state)
        return disconnects(state.endpoint), ended

    def active(metrics: metric_reader.Reader) -> float:
        return sum(point.value for point in metrics.points("bot.transport.peer.active"))

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = active(metrics)
        told, ended = asyncio.run(run())
        after = active(metrics)

    assert (told, ended) == (["closed"], ["closed"])
    assert after - before == 0
