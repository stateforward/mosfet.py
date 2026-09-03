from __future__ import annotations

from bot.devices import audio as audio_device

import asyncio
import collections.abc

import hsm

import bot.providers.livekit.room_audio as room_audio_module

from bot.providers.livekit.room_audio import (
    RoomAudioConnectEvent,
    RoomAudioDisconnectEvent,
    RoomAudioConnectData,
    RoomAudioTrackPath,
)
from tests.livekit_room_fakes import (
    FakeAudioFrame,
    FakeAudioSource,
    FakeRemoteTrack,
    FakeRoom,
    fake_audio_bridge,
    fake_pcm_stream,
)


def test_livekit_room_audio_uses_instance_owned_state() -> None:
    assert not hasattr(room_audio_module, "_ROOM_AUDIO_TRACK_PATH_STATES")
    assert not hasattr(room_audio_module, "_ROOM_AUDIO_ENDPOINT_STATES")


async def _wait_until(predicate: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Timed out waiting for LiveKit room audio path condition.")


def _fake_stream(frame: FakeAudioFrame) -> collections.abc.Callable[[object], collections.abc.AsyncIterable[object]]:
    return fake_pcm_stream(frame)


def test_livekit_room_audio_track_path_model_exposes_room_lifecycle_events() -> None:
    model = RoomAudioTrackPath.model

    assert model.qualified_name == "/RoomAudioTrackPath"
    assert model.initial == "/RoomAudioTrackPath/.initial"
    assert "/RoomAudioTrackPath/disconnected" in model.members
    assert "/RoomAudioTrackPath/connecting" in model.members
    assert "/RoomAudioTrackPath/connected" in model.members
    assert "/RoomAudioTrackPath/receiving_remote_audio" in model.members
    assert "/RoomAudioTrackPath/disconnecting" in model.members
    assert "/RoomAudioTrackPath/failed" in model.members
    assert RoomAudioConnectEvent.name in model.events
    assert RoomAudioDisconnectEvent.name in model.events


def test_livekit_room_audio_track_path_models_disconnect_cleanup_preconditions() -> None:
    model = RoomAudioTrackPath.model

    assert "/RoomAudioTrackPath/disconnecting/routing_disconnect_cleanup" in model.members
    assert "/RoomAudioTrackPath/disconnecting/unpublishing_local_track" in model.members
    assert "/RoomAudioTrackPath/disconnecting/disconnecting_room" in model.members
    assert "/RoomAudioTrackPath/disconnecting/routing_listener_cleanup" in model.members
    assert "/RoomAudioTrackPath/disconnecting/unregistering_listener" in model.members
    assert "/RoomAudioTrackPath/disconnecting/disconnecting_room_after_failure" in model.members
    assert "/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure" in model.members
    assert "/RoomAudioTrackPath/disconnecting/unregistering_listener_after_failure" in model.members


def test_livekit_room_audio_track_path_publishes_local_source_and_subscribes_remote_frames() -> None:
    async def run() -> None:
        source = FakeAudioSource()
        bridge = fake_audio_bridge(source)
        room = FakeRoom(tracks_to_emit_on_connect=[FakeRemoteTrack()])
        local_track_sources: list[tuple[str, object]] = []
        received_audio: list[audio_device.InputData] = []
        connected_events: list[room_audio_module.RoomAudioConnectedData] = []

        async def remote_audio_sink(audio: audio_device.InputData) -> None:
            received_audio.append(audio)

        def local_track_factory(name: str, source: object) -> object:
            local_track_sources.append((name, source))
            return {"track_name": name}

        path = RoomAudioTrackPath(
            bridge=bridge,
            room=room,
            stream_factory=_fake_stream(
                FakeAudioFrame(
                    data=b"\x01\x00\x02\x00",
                    sample_rate=48_000,
                    num_channels=1,
                    samples_per_channel=2,
                )
            ),
            local_track_factory=local_track_factory,
            remote_audio_sink=remote_audio_sink,
            connection_sink=connected_events.append,
        )
        _ = await hsm.started(None, path, path.model)

        await path.connect_room(
            path.context(),
            RoomAudioConnectData(url="wss://livekit.example.com", token="token", track_name="alice-audio"),
        )
        await _wait_until(
            lambda: connected_events == [room_audio_module.RoomAudioConnectedData(local_track_sid="TR_local")]
        )
        await bridge.publish_audio(
            audio_device.OutputData(
                audio=b"\x03\x00\x04\x00", media_type="audio/pcm", sample_rate_hz=48_000, channels=1
            )
        )
        await _wait_until(lambda: len(received_audio) == 1)

        assert room.connected == [("wss://livekit.example.com", "token")]
        assert local_track_sources == [("alice-audio", source)]
        assert room.local_participant.publications == [{"track_name": "alice-audio"}]
        assert source.captured_frames == [
            FakeAudioFrame(
                data=b"\x03\x00\x04\x00",
                sample_rate=48_000,
                num_channels=1,
                samples_per_channel=2,
            )
        ]
        assert received_audio == [
            audio_device.InputData(audio=b"\x01\x00\x02\x00", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        ]

        await path.disconnect_room(path.context())
        await _wait_until(lambda: path.state() == "/RoomAudioTrackPath/disconnected" and room.disconnected)

        assert room.local_participant.unpublished == ["TR_local"]
        assert room.callbacks["track_subscribed"] == []

    asyncio.run(run())


def test_livekit_room_audio_track_path_disconnects_cleanly_while_connect_is_blocked() -> None:
    async def run() -> None:
        source = FakeAudioSource()
        bridge = fake_audio_bridge(source)
        room = FakeRoom(connect_gate=asyncio.Event())

        async def remote_audio_sink(audio: audio_device.InputData) -> None:
            del audio

        path = RoomAudioTrackPath(
            bridge=bridge,
            room=room,
            stream_factory=_fake_stream(
                FakeAudioFrame(
                    data=b"\x01\x00\x02\x00",
                    sample_rate=48_000,
                    num_channels=1,
                    samples_per_channel=2,
                )
            ),
            remote_audio_sink=remote_audio_sink,
        )
        _ = await hsm.started(None, path, path.model)
        connect_task = path.connect_room(
            path.context(),
            RoomAudioConnectData(
                url="wss://livekit.example.com",
                token="token",
                track_name="alice-audio",
            ),
        )
        await _wait_until(lambda: room.connected == [("wss://livekit.example.com", "token")])

        await path.disconnect_room(path.context())
        await _wait_until(lambda: path.state() == "/RoomAudioTrackPath/disconnected")
        await connect_task

        assert room.disconnected is True
        assert room.off_calls == 1
        assert room.callbacks["track_subscribed"] == []
        assert room.local_participant.publications == []

    asyncio.run(run())
