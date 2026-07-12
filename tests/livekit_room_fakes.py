"""Shared LiveKit room/audio fakes for provider tests.

These stand in for the LiveKit SDK so phone and room-audio paths can be exercised
without a real SFU: connect emits remote tracks, stream factories yield PCM frames.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

from livekit import rtc

from bot.providers.livekit.audio import AudioBridge, AudioFrameEncoder


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


@dataclasses.dataclass(frozen=True)
class FakeRemoteTrack:
    kind: int = rtc.TrackKind.KIND_AUDIO


@dataclasses.dataclass(frozen=True)
class FakeRemoteParticipant:
    """Minimal remote participant for presence → incoming_call tests."""

    identity: str = "human"
    sid: str = "PA_remote"


@dataclasses.dataclass
class FakeLocalParticipant:
    publications: list[object] = dataclasses.field(default_factory=list)
    unpublished: list[str] = dataclasses.field(default_factory=list)
    unpublish_error: BaseException | None = None

    async def publish_track(self, track: object, options: object | None = None) -> FakeTrackPublication:
        del options
        self.publications.append(track)
        return FakeTrackPublication(sid="TR_local")

    async def unpublish_track(self, track_sid: str) -> None:
        if self.unpublish_error is not None:
            raise self.unpublish_error
        self.unpublished.append(track_sid)


@dataclasses.dataclass
class FakeRoom:
    """Minimal RoomHandle: connect can emit remote track_subscribed / participant events."""

    local_participant: FakeLocalParticipant = dataclasses.field(default_factory=FakeLocalParticipant)
    connected: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    connect_gate: asyncio.Event | None = None
    disconnected: bool = False
    callbacks: dict[str, list[collections.abc.Callable[..., object]]] = dataclasses.field(default_factory=dict)
    off_calls: int = 0
    tracks_to_emit_on_connect: list[FakeRemoteTrack] = dataclasses.field(default_factory=list)
    participants_to_emit_on_connect: list[FakeRemoteParticipant] = dataclasses.field(default_factory=list)
    remote_participants: dict[str, FakeRemoteParticipant] = dataclasses.field(default_factory=dict)

    async def connect(self, url: str, token: str) -> None:
        self.connected.append((url, token))
        if self.connect_gate is not None:
            _ = await self.connect_gate.wait()
        for participant in self.participants_to_emit_on_connect:
            self.remote_participants[participant.sid] = participant
            self.emit("participant_connected", participant)
        for track in self.tracks_to_emit_on_connect:
            self.emit("track_subscribed", track, FakeTrackPublication(sid="TR_remote"), object())

    async def disconnect(self) -> None:
        self.disconnected = True
        self.remote_participants.clear()

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
    """Build a real LiveKit AudioFrame for use with PhoneService's default audio bridge."""

    return rtc.AudioFrame(
        data=payload,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(payload) // 2,
    )
