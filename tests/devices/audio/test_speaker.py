from bot.devices import audio

import asyncio
import collections.abc
import typing

import hsm

from bot.device import Device
from bot.world import SoundData, SoundEvent, World
from tests.hsm_instance_state import device_peripherals

class RecordingDevice(Device):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[hsm.Event[typing.Any]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        del ctx
        self.events.append(event)

        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        return done

def test_speaker_is_generic_audio_output_peripheral() -> None:
    speaker = audio.Speaker()

    assert isinstance(speaker, Device)
    assert device_peripherals(speaker) == ()
    assert audio.Speaker.required_bot_abilities == ()

def test_speaker_dispatches_audio_output_to_target_device() -> None:
    async def run() -> None:
        speaker = audio.Speaker()
        target = RecordingDevice()
        _ = await hsm.started(None, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(None, target, target.model, hsm.Config(id="physical-speaker"))
        target.events.clear()
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

        await speaker.dispatch_audio_output(
            hsm.Context(),
            target,
            data,
            metadata={"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"},
        )

        assert len(target.events) == 1
        event = target.events[0]
        assert event.name == audio.OutputEvent.name
        assert event.data == data
        assert event.source == "phone-speaker"
        assert event.target == "physical-speaker"
        assert event.metadata == {"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"}

    asyncio.run(run())

def test_speaker_dispatches_audio_output_to_world_scope() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        world = World()
        speaker = audio.Speaker()
        inside = RecordingDevice()
        outside = RecordingDevice()
        _ = await hsm.started(world, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside-speaker"))
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside-speaker"))
        inside.events.clear()
        outside.events.clear()
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

        await speaker.dispatch_audio_output_to_world(
            world,
            data,
            metadata={"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"},
        )

        return inside.events, outside.events, hsm.id(speaker)

    inside_events, outside_events, speaker_id = asyncio.run(run())

    assert len(inside_events) == 1
    event = inside_events[0]
    assert event.name == SoundEvent.name
    assert event.data == SoundData(
        audio=b"playback-audio",
        media_type="audio/pcm",
        sample_rate_hz=44_100,
        channels=2,
    )
    assert event.source == speaker_id
    assert event.target == "inside-speaker"
    assert event.metadata == {"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"}
    assert outside_events == []
