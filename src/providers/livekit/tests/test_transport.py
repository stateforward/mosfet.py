from __future__ import annotations

from mosfet.devices import audio, computer, hid

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing
import uuid

import hsm
import pydantic
import pytest

from mosfet.abilities import ability
from mosfet.environment import Environment
from mosfet.protocols import attachment, transform, transport
from mosfet.providers import livekit
from tests import metric_reader
from tests.devices.computer.remote import Controller, Holder, settle, start, wait_until
from tests.livekit_room_fakes import (
    FakeAudioFrame,
    FakeAudioSource,
    FakeByteStreamInfo,
    FakeByteStreamReader,
    FakeDataPacket,
    FakeRemoteParticipant,
    FakeRemoteTrack,
    FakeRoom,
    FakeTrackPublication,
    fake_audio_bridge,
    fake_local_track_factory,
)

PEER = "kiosk-1"
"""The remote peer's participant identity in the room."""
URL = "wss://livekit.example.com"
TOKEN = pydantic.SecretStr("bot-jwt")
REQUEST_TIMEOUT = datetime.timedelta(seconds=2)
SHORT = datetime.timedelta(seconds=2)
FAST_RETRY = datetime.timedelta(milliseconds=10)
OFFERED = "/RoomTransport/connected/present/offered"
ACCEPTED = "/RoomTransport/connected/present/accepted"
AWAITING_PEER = "/RoomTransport/connected/awaiting_peer"
DETACHED = "/RoomTransport/detached"


class NoteData(pydantic.BaseModel):
    """A test payload: one line of text."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str


NoteEvent = hsm.Event[NoteData](name="test.note", schema=NoteData)
NOTES = transform.JsonCodec(NoteEvent, max_message_bytes=128 * 1024)


class InboundOnly:
    """A codec that decodes notes and encodes nothing, like a sensor that only reports."""

    name: str
    max_message_bytes: int
    encodes_any: bool

    def __init__(self) -> None:
        self.name = "inbound"
        self.max_message_bytes = NOTES.max_message_bytes
        self.encodes_any = False

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        return NOTES.decode(message)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        del event
        return False

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        raise transform.EncodeError("schema_invalid", f"{event.name!r} is never sent.")


def note(text: str) -> bytes:
    return NOTES.encode(NoteEvent.with_data(NoteData(text=text)))


@dataclasses.dataclass(frozen=True)
class RoomSetup:
    """How one attempt's fake room behaves: a connect that blocks or fails, a disconnect that fails."""

    connect_gate: asyncio.Event | None = None
    connect_error: BaseException | None = None
    disconnect_error: BaseException | None = None


class Rooms:
    """Builds one fake room per attempt and keeps every one, so a test can see each attempt's room."""

    built: list[FakeRoom]
    _setups: list[RoomSetup]
    _peer_present: bool

    def __init__(self, *setups: RoomSetup, peer_present: bool = True) -> None:
        self.built = []
        self._setups = list(setups)
        self._peer_present = peer_present

    def __call__(self) -> livekit.DataRoom:
        setup = self._setups.pop(0) if self._setups else RoomSetup()
        room = FakeRoom(
            connect_gate=setup.connect_gate,
            connect_error=setup.connect_error,
            disconnect_error=setup.disconnect_error,
        )
        if self._peer_present:
            room.remote_participants[PEER] = FakeRemoteParticipant(identity=PEER)
        self.built.append(room)
        # A fake stands in for the SDK room; its plain fields are the room's read-only surface.
        return typing.cast(livekit.DataRoom, typing.cast(object, room))

    def last(self) -> FakeRoom:
        return self.built[-1]


def frame_stream(frame: FakeAudioFrame) -> collections.abc.Callable[[object], collections.abc.AsyncIterable[object]]:
    async def stream(track: object) -> collections.abc.AsyncIterable[object]:
        del track
        yield frame

    return stream


class FailingStream:
    """A remote audio track whose first read fails, as a track torn down under the reader does."""

    def __aiter__(self) -> "FailingStream":
        return self

    async def __anext__(self) -> object:
        raise RuntimeError("remote audio track failed")


def failing_stream(track: object) -> collections.abc.AsyncIterable[object]:
    del track
    return FailingStream()


def make_transport(
    rooms: Rooms,
    *,
    source: FakeAudioSource | None = None,
    stream_factory: collections.abc.Callable[[object], collections.abc.AsyncIterable[object]] | None = None,
    connect_timeout: datetime.timedelta = SHORT,
    release_timeout: datetime.timedelta = SHORT,
    reconnect_initial: datetime.timedelta = FAST_RETRY,
    max_message_bytes: int = 256 * 1024,
) -> livekit.RoomTransport:
    speaker = source if source is not None else FakeAudioSource()
    return livekit.RoomTransport(
        url=URL,
        token=TOKEN,
        connect_timeout=connect_timeout,
        release_timeout=release_timeout,
        reconnect_initial=reconnect_initial,
        reconnect_max=max(reconnect_initial, datetime.timedelta(milliseconds=100)),
        max_message_bytes=max_message_bytes,
        room_factory=rooms,
        bridge_factory=lambda: fake_audio_bridge(speaker),
        stream_factory=stream_factory,
        local_track_factory=fake_local_track_factory,
    )


async def request(
    actor: hsm.Instance,
    build: collections.abc.Callable[[hsm.Instance], hsm.Event[typing.Any]],
    terminals: tuple[hsm.Event[typing.Any], ...],
) -> hsm.Event[typing.Any]:
    return await ability.run_terminal_operation(
        hsm.Context(),
        child=actor,
        request=lambda reply: dataclasses.replace(build(reply), id=uuid.uuid4().hex),
        terminals=terminals,
        timeout=REQUEST_TIMEOUT,
    )


class Owners(typing.NamedTuple):
    """One stand-in owner per channel: notes (duplex), sensor (inbound only), voice and playback (audio)."""

    notes: Controller
    sensor: Controller
    voice: Controller
    playback: Controller


def channels(owners: Owners) -> tuple[transport.ChannelData, ...]:
    pcm = audio.PcmCodec()
    return (
        transport.ChannelData(name="notes", owner=owners.notes, codec=NOTES, kind="data"),
        transport.ChannelData(name="sensor", owner=owners.sensor, codec=InboundOnly(), kind="data"),
        transport.ChannelData(name="voice", owner=owners.voice, codec=pcm, kind="audio"),
        transport.ChannelData(name="playback", owner=owners.playback, codec=pcm, kind="audio"),
    )


async def register(
    actor: hsm.Instance, endpoint: hsm.Instance, owners: Owners, *, peer_id: str = PEER
) -> hsm.Event[typing.Any]:
    return await request(
        actor,
        lambda reply: attachment.AttachEvent.with_data(
            transport.PeerAttachData(actor=endpoint, peer_id=peer_id, channels=channels(owners), reply_to=reply)
        ),
        (attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
    )


async def unregister(actor: hsm.Instance, endpoint: hsm.Instance) -> hsm.Event[typing.Any]:
    return await request(
        actor,
        lambda reply: attachment.DetachEvent.with_data(attachment.DetachData(actor=endpoint, reply_to=reply)),
        (attachment.DetachedEvent, attachment.DetachFailedEvent),
    )


@dataclasses.dataclass
class Registered:
    environment: Environment
    transport: livekit.RoomTransport
    endpoint: Controller
    owners: Owners


async def registered(room_transport: livekit.RoomTransport, *, owners: Owners | None = None) -> Registered:
    """Start ``room_transport``, a stand-in endpoint and channel owners, and register the peer."""

    environment = Environment()
    await start(environment, room_transport, "transport")
    endpoint = Controller()
    await start(environment, endpoint, "endpoint")
    resolved = owners if owners is not None else Owners(Controller(), Controller(), Controller(), Controller())
    for name, owner in zip(Owners._fields, resolved, strict=True):
        await start(environment, owner, name)
    terminal = await register(room_transport, endpoint, resolved)
    assert terminal.name == attachment.AttachCompleteEvent.name
    return Registered(environment=environment, transport=room_transport, endpoint=endpoint, owners=resolved)


def peer_events(endpoint: Controller) -> list[transport.PeerConnectedData | transport.PeerDisconnectedData]:
    return [
        event.data
        for event in endpoint.events
        if isinstance(event.data, transport.PeerConnectedData | transport.PeerDisconnectedData)
    ]


def offers(endpoint: Controller) -> list[transport.PeerConnectedData]:
    return [data for data in peer_events(endpoint) if isinstance(data, transport.PeerConnectedData)]


async def accept(state: Registered, *, count: int = 1) -> transport.PeerConnectedData:
    """Accept the ``count``-th connection the transport offered, as the endpoint would."""

    await wait_until(lambda: len(offers(state.endpoint)) >= count)
    offered = offers(state.endpoint)[count - 1]
    _ = await hsm.dispatch(
        hsm.Context(),
        state.transport,
        transport.AcceptPeerEvent.with_data(
            transport.AcceptPeerData(peer_id=offered.peer_id, connection_id=offered.connection_id)
        ),
    )
    await wait_until(lambda: state.transport.state() == ACCEPTED)
    return offered


async def accepted(room_transport: livekit.RoomTransport, *, owners: Owners | None = None) -> Registered:
    state = await registered(room_transport, owners=owners)
    _ = await accept(state)
    return state


def unbound(room: FakeRoom) -> bool:
    """Every callback the transport put on this room has been taken off it."""

    names = ("disconnected", "participant_connected", "participant_disconnected", "data_received", "track_subscribed")
    return all(not room.callbacks.get(name) for name in names) and not room.byte_stream_handlers


def packet(payload: bytes, *, sender: str = PEER, topic: str | None = "notes") -> FakeDataPacket:
    return FakeDataPacket(data=payload, participant=FakeRemoteParticipant(identity=sender), topic=topic)


def stream(payload: bytes, *, size: int | None = None) -> FakeByteStreamReader:
    return FakeByteStreamReader(
        info=FakeByteStreamInfo(size=len(payload) if size is None else size, mime_type="application/octet-stream"),
        chunks=(payload[:5], payload[5:]),
    )


def ownership(owner: Controller) -> list[transform.OwnedData | transform.EndedData]:
    return [event.data for event in owner.events if isinstance(event.data, transform.OwnedData | transform.EndedData)]


def notes(owner: Controller) -> list[hsm.Event[typing.Any]]:
    return owner.named(NoteEvent)


def counts(metrics: metric_reader.Reader, name: str, key: str) -> dict[object, float]:
    return {
        point.attributes.get(key): point.value
        for point in metrics.points(name)
        if point.attributes.get("transport", "livekit") == "livekit"
    }


def delta(before: dict[object, float], after: dict[object, float]) -> dict[object, float]:
    return {key: after[key] - before.get(key, 0) for key in after if after[key] - before.get(key, 0)}


def transform_counts(metrics: metric_reader.Reader, name: str) -> dict[object, float]:
    return {
        (point.attributes.get("codec"), point.attributes.get("reason")): point.value for point in metrics.points(name)
    }


def test_model_has_one_attempt_lifecycle() -> None:
    members = livekit.RoomTransport.model.members

    for state in (
        DETACHED,
        "/RoomTransport/starting_path",
        AWAITING_PEER,
        OFFERED,
        ACCEPTED,
        "/RoomTransport/releasing/disconnecting_path",
        "/RoomTransport/releasing/stopping_path",
        "/RoomTransport/releasing/released",
        "/RoomTransport/reconnect_wait",
    ):
        assert state in members
    for event in (attachment.AttachEvent, transport.AcceptPeerEvent, transport.ClosePeerEvent):
        assert event.name in livekit.RoomTransport.model.events


def _transport(
    *,
    url: str = URL,
    connect_timeout: datetime.timedelta = SHORT,
    release_timeout: datetime.timedelta = SHORT,
    reconnect_initial: datetime.timedelta = FAST_RETRY,
    max_message_bytes: int = 1024,
) -> livekit.RoomTransport:
    return livekit.RoomTransport(
        url=url,
        token=TOKEN,
        connect_timeout=connect_timeout,
        release_timeout=release_timeout,
        reconnect_initial=reconnect_initial,
        max_message_bytes=max_message_bytes,
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: _transport(url=""), "url"),
        (lambda: _transport(connect_timeout=datetime.timedelta(0)), "connect_timeout"),
        (lambda: _transport(release_timeout=datetime.timedelta(0)), "release_timeout"),
        (lambda: _transport(reconnect_initial=datetime.timedelta(seconds=40)), "reconnect_initial"),
        (lambda: _transport(max_message_bytes=0), "max_message_bytes"),
    ],
)
def test_transport_rejects_invalid_construction(build: collections.abc.Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _ = build()


def test_registration_joins_the_room_and_offers_the_peer_unaccepted() -> None:
    async def run() -> tuple[Rooms, Registered]:
        rooms = Rooms()
        state = await registered(make_transport(rooms))
        await wait_until(lambda: state.transport.state() == OFFERED)
        await settle()
        return rooms, state

    rooms, state = asyncio.run(run())

    (room,) = rooms.built
    assert room.connected == [(URL, "bot-jwt")]
    (offered,) = peer_events(state.endpoint)
    assert isinstance(offered, transport.PeerConnectedData)
    assert (offered.peer_id, offered.connection_id, offered.transport) == (PEER, "1.1", "livekit")
    assert offered.connection is state.transport
    # Nothing is handed to an owner before the endpoint accepts the connection.
    assert all(ownership(owner) == [] and notes(owner) == [] for owner in state.owners)
    (connected,) = state.endpoint.named(transport.PeerConnectedEvent)
    assert connected.source == "transport" and connected.target == "endpoint"


def test_registration_refuses_a_plain_request_and_a_second_endpoint() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        room_transport = make_transport(Rooms())
        environment = Environment()
        first, second = Controller(), Controller()
        owners = Owners(Controller(), Controller(), Controller(), Controller())
        await start(environment, room_transport, "transport")
        await start(environment, first, "first")
        await start(environment, second, "second")
        plain = await request(
            room_transport,
            lambda reply: attachment.AttachEvent.with_data(attachment.AttachData(actor=first, reply_to=reply)),
            (attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
        )
        accepted_one = await register(room_transport, first, owners)
        conflict = await register(room_transport, second, owners, peer_id="kiosk-2")
        return plain, accepted_one, conflict

    plain, accepted_one, conflict = asyncio.run(run())

    assert plain.name == attachment.AttachFailedEvent.name
    assert accepted_one.name == attachment.AttachCompleteEvent.name
    assert isinstance(conflict.data, attachment.FailedData)
    assert conflict.data.kind == attachment.FailureKind.CONFLICT


def test_accepting_tells_each_owner_that_can_send_which_stream_it_owns() -> None:
    async def run() -> Registered:
        state = await accepted(make_transport(Rooms()))
        await wait_until(lambda: bool(ownership(state.owners.notes)))
        await settle()
        return state

    state = asyncio.run(run())

    for owner, channel in (
        (state.owners.notes, "notes"),
        (state.owners.voice, "voice"),
        (state.owners.playback, "playback"),
    ):
        (owned,) = ownership(owner)
        assert isinstance(owned, transform.OwnedData)
        assert (owned.stream, owned.channel) == (state.transport, channel)
    # A sensor's codec sends nothing, so it is never told where to send.
    assert ownership(state.owners.sensor) == []


def test_messages_decode_to_their_channel_owner_stamped_by_the_transport(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[Registered, list[hsm.Event[typing.Any]]]:
        rooms = Rooms()
        state = await registered(make_transport(rooms))
        await wait_until(lambda: state.transport.state() == OFFERED)
        rooms.last().emit("data_received", packet(note("too early")))
        await settle(30)
        early = notes(state.owners.notes)
        _ = await accept(state)
        room = rooms.last()
        room.emit("data_received", packet(note("hello")))
        room.stream("sensor", stream(note("from a stream")), PEER)
        await wait_until(lambda: bool(notes(state.owners.notes)) and bool(notes(state.owners.sensor)))
        return state, early

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = transform_counts(metrics, "bot.transform.dropped.count")
        state, early = asyncio.run(run())
        after = transform_counts(metrics, "bot.transform.dropped.count")

    assert early == []
    assert delta(before, after) == {("json", "unowned"): 1}
    (hello,) = notes(state.owners.notes)
    assert hello.data == NoteData(text="hello")
    assert (hello.source, hello.target) == ("transport", "notes")
    (streamed,) = notes(state.owners.sensor)
    assert streamed.data == NoteData(text="from a stream")
    assert (streamed.source, streamed.target) == ("transport", "sensor")


def test_inbound_messages_are_admitted_at_the_sdk_edge(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> Registered:
        rooms = Rooms()
        state = await accepted(make_transport(rooms, max_message_bytes=1024))
        room = rooms.last()
        room.emit("data_received", packet(note("hi"), sender="stranger"))
        room.emit("data_received", packet(note("hi"), topic="lk.chat"))
        room.emit("data_received", packet(note("hi"), topic="voice"))
        room.emit("data_received", packet(note("hi"), topic=None))
        room.emit("data_received", packet(note("x" * 2048)))
        room.emit("data_received", packet(b"{not json"))
        room.stream("notes", stream(note("hi")), "stranger")
        room.stream("notes", stream(note("hi"), size=4096), PEER)
        unsized = FakeByteStreamReader(info=FakeByteStreamInfo(size=None, mime_type="text/plain"), chunks=(b"{}",))
        room.stream("notes", unsized, PEER)
        await settle(30)
        return state

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        rejected_before = counts(metrics, "bot.transport.rejected.count", "reason")
        failed_before = transform_counts(metrics, "bot.transform.decode.failed.count")
        state = asyncio.run(run())
        rejected_after = counts(metrics, "bot.transport.rejected.count", "reason")
        failed_after = transform_counts(metrics, "bot.transform.decode.failed.count")

    assert notes(state.owners.notes) == []
    assert delta(rejected_before, rejected_after) == {"wrong_peer": 2, "refused_channel": 3, "oversize": 3}
    assert delta(failed_before, failed_after) == {("json", "malformed"): 1}


@pytest.mark.parametrize(
    ("track_name", "channel"), [("voice", "voice"), ("playback", "playback"), ("notes", None), ("", None)]
)
def test_the_peers_audio_track_reaches_the_audio_channel_it_is_named_after(
    track_name: str, channel: str | None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = FakeAudioFrame(data=b"\x01\x00\x02\x00", sample_rate=48_000, num_channels=1, samples_per_channel=2)

    async def run() -> Registered:
        rooms = Rooms()
        state = await accepted(make_transport(rooms, stream_factory=frame_stream(frame)))
        rooms.last().emit(
            "track_subscribed",
            FakeRemoteTrack(),
            FakeTrackPublication(sid="TR_peer", name=track_name),
            FakeRemoteParticipant(identity=PEER),
        )
        owner = getattr(state.owners, channel) if channel is not None else None
        if isinstance(owner, Controller):
            await wait_until(lambda: bool(owner.named(audio.InputEvent)), timeout=3.0)
        else:
            await asyncio.sleep(0.4)
        return state

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counts(metrics, "bot.transport.rejected.count", "reason")
        state = asyncio.run(run())
        after = counts(metrics, "bot.transport.rejected.count", "reason")

    heard = {name: owner.named(audio.InputEvent) for name, owner in zip(Owners._fields, state.owners, strict=True)}
    if channel is None:
        assert all(events == [] for events in heard.values())
        assert delta(before, after).get("refused_channel", 0) >= 1
        return
    (chunk,) = heard.pop(channel)
    assert chunk.data == audio.InputData(
        audio=b"\x01\x00\x02\x00", media_type="audio/pcm", sample_rate_hz=48_000, channels=1
    )
    assert (chunk.source, chunk.target) == ("transport", channel)
    assert all(events == [] for events in heard.values())


def test_owners_send_on_the_channel_they_own(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    large = NoteEvent.with_data(NoteData(text="x" * (15 * 1024)))

    async def run() -> tuple[FakeRoom, FakeAudioSource]:
        rooms = Rooms()
        source = FakeAudioSource()
        state = await accepted(make_transport(rooms, source=source))
        owners = state.owners

        async def send(owner: Controller | None, event: hsm.Event[typing.Any]) -> None:
            _ = await hsm.dispatch(
                hsm.Context(), state.transport, dataclasses.replace(event, source=hsm.id(owner) if owner else "")
            )

        await send(owners.notes, NoteEvent.with_data(NoteData(text="small")))
        await send(owners.notes, large)
        output = audio.OutputData(audio=b"\x03\x00\x04\x00", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        await send(owners.playback, audio.OutputEvent.with_data(output))
        # Not sent: the sensor's codec encodes nothing, nobody owns a channel for a sourceless event,
        # and audio that is not PCM cannot be encoded.
        await send(owners.sensor, NoteEvent.with_data(NoteData(text="never")))
        await send(None, NoteEvent.with_data(NoteData(text="never")))
        await send(owners.playback, audio.OutputEvent.with_data(audio.OutputData(audio=b"opus")))
        participant = rooms.last().local_participant
        await wait_until(
            lambda: bool(participant.published_data)
            and bool(participant.byte_streams)
            and participant.byte_streams[0].closed
            and bool(source.captured_frames)
        )
        await settle(30)
        return rooms.last(), source

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = transform_counts(metrics, "bot.transform.dropped.count")
        room, source = asyncio.run(run())
        after = transform_counts(metrics, "bot.transform.dropped.count")

    (sent,) = room.local_participant.published_data
    assert (sent.topic, sent.reliable, sent.destination_identities) == ("notes", True, (PEER,))
    assert sent.payload == note("small")
    (streamed,) = room.local_participant.byte_streams
    assert (streamed.topic, streamed.destination_identities) == ("notes", (PEER,))
    assert bytes(streamed.written) == NOTES.encode(large)
    (captured,) = source.captured_frames
    assert captured.data == b"\x03\x00\x04\x00"
    assert delta(before, after) == {("pcm", "encode_failed"): 1}


class Holding(Controller):
    """An owner slow to process: every event it is handed stays in flight until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[asyncio.Future[bool]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        del ctx
        self.events.append(event)
        held: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.pending.append(held)
        return held

    def release(self) -> None:
        for held in self.pending:
            if not held.done():
                held.set_result(True)


def test_an_owner_slow_to_process_holds_at_most_256_kib_in_flight(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sixty_kib = note("x" * (60 * 1024))

    async def run() -> tuple[int, int]:
        rooms = Rooms()
        slow = Holding()
        state = await accepted(make_transport(rooms), owners=Owners(slow, Controller(), Controller(), Controller()))
        room = rooms.last()
        for _ in range(5):
            room.emit("data_received", packet(sixty_kib))
        await settle(30)
        held = len(notes(slow))
        slow.release()
        await settle(30)
        room.emit("data_received", packet(sixty_kib))
        await wait_until(lambda: len(notes(slow)) == held + 1)
        del state
        return held, len(notes(slow))

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = transform_counts(metrics, "bot.transform.dropped.count")
        held, total = asyncio.run(run())
        after = transform_counts(metrics, "bot.transform.dropped.count")

    assert (held, total) == (4, 5)
    assert delta(before, after) == {("json", "flow_control"): 1}


def test_the_peer_leaving_ends_the_connection_and_rejoining_offers_a_new_one() -> None:
    async def run() -> tuple[Rooms, Registered]:
        rooms = Rooms()
        state = await accepted(make_transport(rooms))
        rooms.last().leave(PEER)
        await wait_until(lambda: state.transport.state() == AWAITING_PEER)
        rooms.last().join(FakeRemoteParticipant(identity=PEER))
        await wait_until(lambda: state.transport.state() == OFFERED)
        await settle()
        return rooms, state

    rooms, state = asyncio.run(run())

    assert len(rooms.built) == 1 and rooms.built[0].disconnect_calls == 0
    assert [(type(data).__name__, data.connection_id) for data in peer_events(state.endpoint)] == [
        ("PeerConnectedData", "1.1"),
        ("PeerDisconnectedData", "1.1"),
        ("PeerConnectedData", "1.2"),
    ]
    (_, left, _) = peer_events(state.endpoint)
    assert isinstance(left, transport.PeerDisconnectedData) and left.reason == "left"
    ended = [data for data in ownership(state.owners.notes) if isinstance(data, transform.EndedData)]
    assert [data.reason for data in ended] == ["finished"]


def test_closing_the_connection_leaves_the_room_and_offers_the_peer_again() -> None:
    async def run() -> tuple[Rooms, Registered]:
        rooms = Rooms()
        state = await accepted(make_transport(rooms))
        offered = offers(state.endpoint)[0]
        _ = await hsm.dispatch(
            hsm.Context(),
            state.transport,
            transport.ClosePeerEvent.with_data(
                transport.ClosePeerData(peer_id=PEER, connection_id=offered.connection_id, reason="released")
            ),
        )
        await wait_until(lambda: len(rooms.built) == 2 and state.transport.state() == OFFERED)
        await settle()
        return rooms, state

    rooms, state = asyncio.run(run())

    first, _ = rooms.built
    assert first.disconnect_calls == 1 and unbound(first)
    told = peer_events(state.endpoint)
    assert [(type(data).__name__, data.connection_id) for data in told] == [
        ("PeerConnectedData", "1.1"),
        ("PeerDisconnectedData", "1.1"),
        ("PeerConnectedData", "2.1"),
    ]
    assert isinstance(told[1], transport.PeerDisconnectedData) and told[1].reason == "closed"
    ended = [data for data in ownership(state.owners.notes) if isinstance(data, transform.EndedData)]
    assert [data.reason for data in ended] == ["closed"]


def test_a_stale_accept_or_close_is_ignored() -> None:
    async def run() -> tuple[str, list[object]]:
        rooms = Rooms()
        state = await registered(make_transport(rooms))
        await wait_until(lambda: state.transport.state() == OFFERED)
        for stale in (
            transport.AcceptPeerEvent.with_data(transport.AcceptPeerData(peer_id=PEER, connection_id="9.9")),
            transport.AcceptPeerEvent.with_data(transport.AcceptPeerData(peer_id="kiosk-2", connection_id="1.1")),
            transport.ClosePeerEvent.with_data(
                transport.ClosePeerData(peer_id=PEER, connection_id="9.9", reason="superseded")
            ),
        ):
            _ = await hsm.dispatch(hsm.Context(), state.transport, stale)
        await settle(30)
        return state.transport.state(), list(peer_events(state.endpoint))

    state, told = asyncio.run(run())

    assert state == OFFERED
    assert len(told) == 1


def test_a_lost_room_releases_the_attempt_and_offers_the_peer_on_a_new_room_and_path() -> None:
    async def run() -> tuple[Rooms, Registered]:
        rooms = Rooms()
        state = await accepted(make_transport(rooms))
        rooms.last().emit("disconnected", "SERVER_SHUTDOWN")
        await wait_until(lambda: len(rooms.built) == 2 and state.transport.state() == OFFERED)
        await settle()
        return rooms, state

    rooms, state = asyncio.run(run())

    first, second = rooms.built
    # Each attempt's room was joined once and left at most once, both by that attempt's path.
    assert (len(first.connected), first.disconnect_calls) == (1, 1)
    assert (len(second.connected), second.disconnect_calls) == (1, 0)
    assert unbound(first)
    told = peer_events(state.endpoint)
    assert [type(data).__name__ for data in told] == ["PeerConnectedData", "PeerDisconnectedData", "PeerConnectedData"]
    assert isinstance(told[1], transport.PeerDisconnectedData) and told[1].reason == "transport_lost"
    ended = [data for data in ownership(state.owners.notes) if isinstance(data, transform.EndedData)]
    assert [data.reason for data in ended] == ["reset"]


def test_a_path_that_tears_itself_down_is_stopped_at_once_without_a_release_wait() -> None:
    """The path reports itself disconnected; the transport skips asking it to leave and offers the peer again."""

    async def run() -> tuple[Rooms, Registered]:
        rooms = Rooms()
        room_transport = make_transport(
            rooms, stream_factory=failing_stream, release_timeout=datetime.timedelta(seconds=30)
        )
        state = await registered(room_transport)
        await wait_until(lambda: room_transport.state() == OFFERED)
        rooms.last().emit(
            "track_subscribed",
            FakeRemoteTrack(),
            FakeTrackPublication(sid="TR_peer", name="voice"),
            FakeRemoteParticipant(identity=PEER),
        )
        await wait_until(lambda: len(rooms.built) == 2 and room_transport.state() == OFFERED, timeout=2.0)
        return rooms, state

    rooms, state = asyncio.run(run())

    assert rooms.built[0].disconnect_calls == 1
    told = peer_events(state.endpoint)
    assert isinstance(told[1], transport.PeerDisconnectedData) and told[1].reason == "transport_lost"


def test_a_path_whose_disconnect_fails_is_discarded_after_the_release_timeout() -> None:
    """The path ends in its dead `failed` leaf; the transport stops it and joins a fresh room and path."""

    async def run() -> Rooms:
        rooms = Rooms(RoomSetup(disconnect_error=RuntimeError("room disconnect failed")))
        room_transport = make_transport(rooms, release_timeout=datetime.timedelta(milliseconds=50))
        _ = await registered(room_transport)
        await wait_until(lambda: room_transport.state() == OFFERED)
        rooms.last().emit("disconnected", "SIGNAL_CLOSE")
        await wait_until(lambda: len(rooms.built) == 2 and room_transport.state() == OFFERED)
        return rooms

    rooms = asyncio.run(run())

    first, second = rooms.built
    assert first.disconnect_calls == 1
    assert len(second.connected) == 1


def test_a_connect_that_fails_retries_on_a_fresh_room(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> Rooms:
        rooms = Rooms(RoomSetup(connect_error=RuntimeError("join refused")))
        state = await registered(make_transport(rooms))
        await wait_until(lambda: len(rooms.built) == 2 and state.transport.state() == OFFERED)
        return rooms

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counts(metrics, "bot.transport.connect.count", "outcome")
        rooms = asyncio.run(run())
        after = counts(metrics, "bot.transport.connect.count", "outcome")

    first, second = rooms.built
    assert len(first.connected) == 1 and len(second.connected) == 1
    assert unbound(first)
    assert delta(before, after) == {"failed": 1, "connected": 1}


def test_a_connect_that_never_finishes_times_out_and_retries(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> Rooms:
        rooms = Rooms(RoomSetup(connect_gate=asyncio.Event()))
        room_transport = make_transport(rooms, connect_timeout=datetime.timedelta(milliseconds=50))
        _ = await registered(room_transport)
        await wait_until(lambda: len(rooms.built) == 2 and room_transport.state() == OFFERED)
        return rooms

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = counts(metrics, "bot.transport.connect.count", "outcome")
        rooms = asyncio.run(run())
        after = counts(metrics, "bot.transport.connect.count", "outcome")

    assert len(rooms.built) == 2
    assert delta(before, after).get("timed_out") == 1


def test_each_failed_attempt_doubles_the_wait_before_the_next() -> None:
    """Backoff grows from reconnect_initial: 50, 100, 200, 400 ms, so 0.7 s holds a handful of attempts, not a dozen."""

    async def run() -> int:
        refused = RoomSetup(connect_error=RuntimeError("join refused"))
        rooms = Rooms(*(refused for _ in range(50)))
        room_transport = livekit.RoomTransport(
            url=URL,
            token=TOKEN,
            reconnect_initial=datetime.timedelta(milliseconds=50),
            reconnect_max=datetime.timedelta(seconds=10),
            room_factory=rooms,
            bridge_factory=lambda: fake_audio_bridge(FakeAudioSource()),
            local_track_factory=fake_local_track_factory,
        )
        _ = await registered(room_transport)
        await asyncio.sleep(0.7)
        return len(rooms.built)

    attempts = asyncio.run(run())

    assert 3 <= attempts <= 6


def test_late_callbacks_from_a_discarded_attempt_are_ignored() -> None:
    async def run() -> tuple[list[str], str]:
        rooms = Rooms()
        state = await registered(make_transport(rooms))
        await wait_until(lambda: state.transport.state() == OFFERED)
        stale = rooms.last()
        # Keep the first attempt's SDK callbacks the way a late SDK thread might still hold them.
        stale_callbacks = {name: list(stale.callbacks.get(name, ())) for name in stale.callbacks}
        stale.emit("disconnected", "SERVER_SHUTDOWN")
        await wait_until(lambda: len(rooms.built) == 2 and state.transport.state() == OFFERED)
        for callback in stale_callbacks["participant_disconnected"]:
            _ = callback(FakeRemoteParticipant(identity=PEER))
        for callback in stale_callbacks["disconnected"]:
            _ = callback("SERVER_SHUTDOWN")
        for callback in stale_callbacks["data_received"]:
            _ = callback(packet(note("stale")))
        await settle(30)
        return [type(data).__name__ for data in peer_events(state.endpoint)], state.transport.state()

    told, state = asyncio.run(run())

    assert state == OFFERED
    assert told == ["PeerConnectedData", "PeerDisconnectedData", "PeerConnectedData"]


DetachCase = typing.Literal["starting_path", "awaiting_peer", "offered", "accepted", "releasing", "reconnect_wait"]


@pytest.mark.parametrize("case", typing.get_args(DetachCase))
def test_unregistering_from_every_state_answers_once_and_leaves_no_attempt_behind(case: DetachCase) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], Rooms, str, Registered]:
        if case == "starting_path":
            rooms = Rooms(RoomSetup(connect_gate=asyncio.Event()))
        elif case == "awaiting_peer":
            rooms = Rooms(peer_present=False)
        elif case == "releasing":
            # The path's disconnect fails, so the transport stays in disconnecting_path for its release timeout.
            rooms = Rooms(RoomSetup(disconnect_error=RuntimeError("stuck")))
        else:
            rooms = Rooms()
        room_transport = make_transport(
            rooms,
            release_timeout=datetime.timedelta(milliseconds=200 if case == "releasing" else 50),
            reconnect_initial=datetime.timedelta(seconds=10 if case == "reconnect_wait" else 0.01),
        )
        state = await registered(room_transport)
        expected = {
            "starting_path": "/RoomTransport/starting_path",
            "awaiting_peer": AWAITING_PEER,
            "offered": OFFERED,
            "accepted": ACCEPTED,
            "releasing": "/RoomTransport/releasing/disconnecting_path",
            "reconnect_wait": "/RoomTransport/reconnect_wait",
        }[case]
        if case == "accepted":
            _ = await accept(state)
        if case in ("releasing", "reconnect_wait"):
            await wait_until(lambda: room_transport.state() == OFFERED)
            rooms.last().emit("disconnected", "SERVER_SHUTDOWN")
        await wait_until(lambda: room_transport.state() == expected)
        replies = Controller()
        await start(state.environment, replies, "replies")
        _ = await hsm.dispatch(
            hsm.Context(),
            room_transport,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=state.endpoint, reply_to=replies)),
        )
        await wait_until(lambda: room_transport.state() == DETACHED, timeout=3.0)
        await settle(30)
        return replies.named(attachment.DetachedEvent), rooms, room_transport.state(), state

    answers, rooms, state, registration = asyncio.run(run())

    (answer,) = answers
    assert isinstance(answer.data, attachment.DetachedData) and answer.data.removed is True
    assert state == DETACHED
    assert all(unbound(room) for room in rooms.built)
    told = peer_events(registration.endpoint)
    # Every connection offered was ended exactly once.
    assert len([data for data in told if isinstance(data, transport.PeerConnectedData)]) == len(
        [data for data in told if isinstance(data, transport.PeerDisconnectedData)]
    )
    if case in ("offered", "accepted"):
        assert isinstance(told[-1], transport.PeerDisconnectedData) and told[-1].reason == "closed"


def test_unregistering_what_is_not_registered_answers_absent() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        room_transport = make_transport(Rooms())
        environment = Environment()
        endpoint, stranger = Controller(), Controller()
        owners = Owners(Controller(), Controller(), Controller(), Controller())
        await start(environment, room_transport, "transport")
        await start(environment, endpoint, "endpoint")
        await start(environment, stranger, "stranger")
        while_detached = await unregister(room_transport, endpoint)
        _ = await register(room_transport, endpoint, owners)
        await wait_until(lambda: room_transport.state() == OFFERED)
        while_offered = await unregister(room_transport, stranger)
        return while_detached, while_offered

    while_detached, while_offered = asyncio.run(run())

    for answer in (while_detached, while_offered):
        assert isinstance(answer.data, attachment.DetachedData) and answer.data.removed is False


def test_a_registration_during_a_release_waits_for_it_then_is_answered() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str, Rooms]:
        rooms = Rooms(RoomSetup(disconnect_error=RuntimeError("stuck")))
        room_transport = make_transport(rooms, release_timeout=datetime.timedelta(milliseconds=100))
        state = await registered(room_transport)
        await wait_until(lambda: room_transport.state() == OFFERED)
        _ = await unregister(room_transport, state.endpoint)
        assert room_transport.state().startswith("/RoomTransport/releasing")
        successor = Controller()
        await start(state.environment, successor, "successor")
        answered = await register(room_transport, successor, state.owners)
        state_when_answered = room_transport.state()
        await wait_until(lambda: room_transport.state() == OFFERED)
        return answered, state_when_answered, rooms

    answered, state_when_answered, rooms = asyncio.run(run())

    assert answered.name == attachment.AttachCompleteEvent.name
    assert not state_when_answered.startswith("/RoomTransport/releasing")
    assert len(rooms.built) == 2


def test_stopping_ends_the_connection_and_its_attempt() -> None:
    async def run() -> Registered:
        state = await accepted(make_transport(Rooms()))
        await hsm.stop(state.transport)
        await settle(30)
        return state

    state = asyncio.run(run())

    told = peer_events(state.endpoint)
    assert isinstance(told[-1], transport.PeerDisconnectedData) and told[-1].reason == "transport_lost"
    ended = [data for data in ownership(state.owners.notes) if isinstance(data, transform.EndedData)]
    assert [data.reason for data in ended] == ["closed"]


KEY_CODEC = transform.JsonCodec(hid.KeyEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)
DISPLAY_CODEC = transform.JsonCodec(
    computer.ContentEvent, computer.LayoutEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES
)
FIELD = computer.ContentData(elements=(computer.FieldElement(id="name", focused=True),))


def test_a_computer_typed_into_through_the_room_tells_its_holder() -> None:
    """A field shown on the remote display, typed into and entered remotely, reaches the bot holding the computer."""

    def shown(room: FakeRoom) -> list[object]:
        return [
            DISPLAY_CODEC.decode(sent.payload).data
            for sent in room.local_participant.published_data
            if sent.topic == "display" and isinstance(sent.payload, bytes)
        ]

    async def run() -> tuple[list[tuple[str, object]], FakeRoom, bool, str]:
        environment = Environment()
        rooms = Rooms()
        room_transport = make_transport(rooms)
        await start(environment, room_transport, "transport")
        device = computer.Computer(device_id="studio-laptop", transport=room_transport, peer_id=PEER, person_id="p-1")
        await start(environment, device, "computer")
        await wait_until(lambda: device.state() == "/Device/detached" and room_transport.state() == ACCEPTED)
        holder = Holder()
        await start(environment, holder, "holder")
        _ = await hsm.dispatch(
            environment, device, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder))
        )
        await wait_until(lambda: device.state() == "/Device/attached")
        show = computer.ShowEvent.with_data(computer.ShowData(elements=FIELD.elements))
        _ = await device.dispatch(device.context(), show)
        room = rooms.last()
        await wait_until(lambda: FIELD in shown(room))
        for key, text in (("h", "h"), ("i", "i"), ("Enter", None)):
            typed = hid.KeyEvent.with_data(hid.KeyData(action="down", key=key, code="", text=text))
            room.emit("data_received", packet(KEY_CODEC.encode(typed), topic="keyboard"))
        await wait_until(lambda: any(name == computer.TextEnteredEvent.name for name, _ in holder.observations()))
        await hsm.stop(device)
        await wait_until(lambda: room_transport.state() == DETACHED)
        return holder.observations(), room, unbound(room), room_transport.state()

    observations, room, released, state = asyncio.run(run())

    assert (
        computer.TextEnteredEvent.name,
        computer.TextEnteredData(device_id="studio-laptop", person_id="p-1", field_id="name", text="hi"),
    ) in observations
    assert FIELD in shown(room)
    assert released and state == DETACHED
