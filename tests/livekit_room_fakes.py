"""Shared LiveKit room/audio fakes for provider tests.

These stand in for the LiveKit SDK so phone and room-audio paths can be exercised
without a real SFU: connect emits remote tracks, stream factories yield PCM frames.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import json
import typing

from livekit import rtc

from mosfet.providers.livekit.audio import AudioBridge, AudioFrameEncoder


@dataclasses.dataclass(frozen=True)
class FakeAudioFrame:
    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int


@dataclasses.dataclass
class FakeAudioSource:
    captured_frames: list[FakeAudioFrame] = dataclasses.field(default_factory=list)

    async def capture_frame(self, frame: FakeAudioFrame) -> None:
        self.captured_frames.append(frame)


@dataclasses.dataclass(frozen=True)
class FakeTrackPublication:
    sid: str
    name: str = ""


@dataclasses.dataclass(frozen=True)
class FakeRemoteTrack:
    kind: int = rtc.TrackKind.KIND_AUDIO


@dataclasses.dataclass(frozen=True)
class FakeRemoteParticipant:
    """Minimal remote participant for presence-departure tests."""

    identity: str = "human"
    sid: str = "PA_remote"


@dataclasses.dataclass(frozen=True)
class FakeRpcCall:
    """One addressed call-setup message this participant sent."""

    destination_identity: str
    method: str
    payload: str


@dataclasses.dataclass(frozen=True)
class FakeDataPacket:
    """One data packet as the room's ``data_received`` callback hands it over."""

    data: bytes
    participant: FakeRemoteParticipant | None
    topic: str | None = None


@dataclasses.dataclass(frozen=True)
class FakePublishedData:
    """One data packet this participant sent."""

    payload: bytes | str
    topic: str
    reliable: bool
    destination_identities: tuple[str, ...]


@dataclasses.dataclass
class FakeByteStreamWriter:
    """A byte stream this participant opened: what was written, and whether it was closed."""

    name: str
    topic: str
    mime_type: str
    total_size: int | None
    destination_identities: tuple[str, ...]
    written: bytearray = dataclasses.field(default_factory=bytearray)
    closed: bool = False

    async def write(self, data: bytes) -> None:
        self.written += data

    async def aclose(self) -> None:
        self.closed = True


@dataclasses.dataclass(frozen=True)
class FakeByteStreamInfo:
    size: int | None
    mime_type: str


@dataclasses.dataclass
class FakeByteStreamReader:
    """An incoming byte stream: its header, then its chunks."""

    info: FakeByteStreamInfo
    chunks: tuple[bytes, ...] = ()

    def __aiter__(self) -> collections.abc.AsyncIterator[bytes]:
        return self._read()

    async def _read(self) -> collections.abc.AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@dataclasses.dataclass
class FakeSfu:
    """The exchange addressed RPC crosses: it delivers a message to the endpoint it names.

    A real ``perform_rpc`` reaches the far participant's registered handler and returns what that
    handler returned. Echoing the payload back to the sender instead would make every call-setup
    test a monologue — two phones could never be wired into one loop, which is the only way to
    find out whether dialling one of them makes the other ring.
    """

    endpoints: dict[str, "FakeLocalParticipant"] = dataclasses.field(default_factory=dict)
    delivered: int = 0

    def deliver(self, *, caller_identity: str, destination_identity: str, method: str, payload: str) -> str:
        endpoint = self.endpoints.get(destination_identity)
        if endpoint is None:
            raise rtc.RpcError(
                rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND,
                f"No participant named {destination_identity!r} is in the room.",
            )
        handler = endpoint.rpc_handlers.get(method)
        if handler is None:
            raise rtc.RpcError(
                rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD,
                f"{destination_identity!r} does not answer {method!r}.",
            )
        self.delivered += 1
        return handler(
            rtc.RpcInvocationData(
                request_id=f"RQ_{self.delivered}",
                caller_identity=caller_identity,
                payload=payload,
                response_timeout=5.0,
            )
        )


@dataclasses.dataclass
class FakeLocalParticipant:
    """Local participant with a publish path and a call-setup RPC channel.

    Participants that share an ``sfu`` can reach each other by identity, the way they do in one
    real room. A participant given no ``sfu`` gets a private one holding only itself, so its
    outbound calls reach nobody — which is what being alone in a room means.
    """

    identity: str = "local"
    sfu: FakeSfu = dataclasses.field(default_factory=FakeSfu)
    publications: list[object] = dataclasses.field(default_factory=list)
    unpublished: list[str] = dataclasses.field(default_factory=list)
    unpublish_error: BaseException | None = None
    rpc_calls: list[FakeRpcCall] = dataclasses.field(default_factory=list)
    rpc_handlers: dict[str, collections.abc.Callable[[object], str]] = dataclasses.field(default_factory=dict)
    published_data: list[FakePublishedData] = dataclasses.field(default_factory=list)
    byte_streams: list[FakeByteStreamWriter] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.sfu.endpoints[self.identity] = self

    async def publish_track(self, track: object, options: object | None = None) -> FakeTrackPublication:
        del options
        self.publications.append(track)
        return FakeTrackPublication(sid="TR_local")

    async def unpublish_track(self, track_sid: str) -> None:
        if self.unpublish_error is not None:
            raise self.unpublish_error
        self.unpublished.append(track_sid)

    async def publish_data(
        self,
        payload: bytes | str,
        *,
        reliable: bool = True,
        destination_identities: list[str] | None = None,
        topic: str = "",
    ) -> None:
        self.published_data.append(
            FakePublishedData(
                payload=payload,
                topic=topic,
                reliable=reliable,
                destination_identities=tuple(destination_identities or ()),
            )
        )

    async def stream_bytes(
        self,
        name: str,
        *,
        total_size: int | None = None,
        mime_type: str = "application/octet-stream",
        attributes: dict[str, str] | None = None,
        stream_id: str | None = None,
        destination_identities: list[str] | None = None,
        topic: str = "",
    ) -> FakeByteStreamWriter:
        del attributes, stream_id
        writer = FakeByteStreamWriter(
            name=name,
            topic=topic,
            mime_type=mime_type,
            total_size=total_size,
            destination_identities=tuple(destination_identities or ()),
        )
        self.byte_streams.append(writer)
        return writer

    async def perform_rpc(
        self,
        *,
        destination_identity: str,
        method: str,
        payload: str,
        response_timeout: float | None = None,
    ) -> str:
        del response_timeout
        self.rpc_calls.append(FakeRpcCall(destination_identity=destination_identity, method=method, payload=payload))
        return self.sfu.deliver(
            caller_identity=self.identity,
            destination_identity=destination_identity,
            method=method,
            payload=payload,
        )

    def register_rpc_method(
        self,
        method_name: str,
        handler: collections.abc.Callable[[object], str],
    ) -> object:
        self.rpc_handlers[method_name] = handler
        return handler

    def unregister_rpc_method(self, method: str) -> None:
        _ = self.rpc_handlers.pop(method, None)

    def invoke(self, method: str, *, caller_identity: str, call_id: str) -> str:
        """Deliver one call-setup message to this participant, from an endpoint tests do not model."""

        return self.sfu.deliver(
            caller_identity=caller_identity,
            destination_identity=self.identity,
            method=method,
            payload=json.dumps({"call_id": call_id}),
        )


@dataclasses.dataclass
class FakeRoom:
    """Minimal RoomHandle: connect can emit remote track_subscribed events."""

    local_participant: FakeLocalParticipant = dataclasses.field(default_factory=FakeLocalParticipant)
    connected: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    connect_gate: asyncio.Event | None = None
    disconnected: bool = False
    callbacks: dict[str, list[collections.abc.Callable[..., object]]] = dataclasses.field(default_factory=dict)
    off_calls: int = 0
    tracks_to_emit_on_connect: list[FakeRemoteTrack] = dataclasses.field(default_factory=list)
    remote_participants: dict[str, FakeRemoteParticipant] = dataclasses.field(default_factory=dict)
    byte_stream_handlers: dict[str, collections.abc.Callable[[typing.Any, str], None]] = dataclasses.field(
        default_factory=dict
    )
    connect_error: BaseException | None = None
    disconnect_error: BaseException | None = None
    disconnect_calls: int = 0

    async def connect(self, url: str, token: str) -> None:
        self.connected.append((url, token))
        if self.connect_gate is not None:
            _ = await self.connect_gate.wait()
        if self.connect_error is not None:
            raise self.connect_error
        for track in self.tracks_to_emit_on_connect:
            self.emit("track_subscribed", track, FakeTrackPublication(sid="TR_remote"), object())

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.disconnected = True

    def register_byte_stream_handler(
        self, topic: str, handler: collections.abc.Callable[[typing.Any, str], None]
    ) -> None:
        if topic in self.byte_stream_handlers:
            raise ValueError(f"byte stream handler for topic {topic!r} already set")
        self.byte_stream_handlers[topic] = handler

    def unregister_byte_stream_handler(self, topic: str) -> None:
        _ = self.byte_stream_handlers.pop(topic, None)

    def join(self, participant: FakeRemoteParticipant) -> None:
        """A remote participant joins the room."""

        self.remote_participants[participant.identity] = participant
        self.emit("participant_connected", participant)

    def leave(self, identity: str) -> None:
        """A remote participant leaves the room."""

        participant = self.remote_participants.pop(identity)
        self.emit("participant_disconnected", participant)

    def stream(self, topic: str, reader: FakeByteStreamReader, participant_identity: str) -> None:
        """A remote participant opens a byte stream on ``topic``."""

        self.byte_stream_handlers[topic](reader, participant_identity)

    def on(
        self,
        event: str,
        callback: collections.abc.Callable[..., object] | None = None,
    ) -> collections.abc.Callable[..., object]:
        assert callback is not None
        self.callbacks.setdefault(event, []).append(callback)
        return callback

    def off(self, event: str, callback: collections.abc.Callable[..., object]) -> None:
        self.off_calls += 1
        if event in self.callbacks:
            self.callbacks[event].remove(callback)

    def emit(self, event: str, *args: object) -> None:
        for callback in tuple(self.callbacks.get(event, ())):
            _ = callback(*args)


def fake_audio_frame_factory(
    *,
    data: bytes,
    sample_rate: int,
    num_channels: int,
    samples_per_channel: int,
) -> FakeAudioFrame:
    return FakeAudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def fake_audio_bridge(source: FakeAudioSource) -> AudioBridge[rtc.AudioFrame]:
    bridge: AudioBridge[FakeAudioFrame] = AudioBridge[FakeAudioFrame].from_audio_source(
        source=source,
        encoder=AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory),
    )
    return typing.cast(AudioBridge[rtc.AudioFrame], bridge)


def fake_local_track_factory(name: str, source: object) -> object:
    del source
    return {"track_name": name}


def fake_pcm_stream(
    frame: FakeAudioFrame | rtc.AudioFrame,
) -> collections.abc.Callable[[object], collections.abc.AsyncIterable[object]]:
    async def stream(track: object) -> collections.abc.AsyncIterable[object]:
        del track
        yield frame

    return typing.cast(collections.abc.Callable[[object], collections.abc.AsyncIterable[object]], stream)


def rtc_pcm_frame(payload: bytes = b"\x01\x00\x02\x00", *, sample_rate: int = 48_000) -> rtc.AudioFrame:
    """Build a real LiveKit AudioFrame for use with Service's default audio bridge."""

    return rtc.AudioFrame(
        data=payload,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(payload) // 2,
    )
