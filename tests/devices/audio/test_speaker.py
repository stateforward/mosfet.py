from bot.devices import audio

import asyncio
import collections.abc
import dataclasses
import typing

import hsm

from bot.device import Device
from bot.protocols import attachment
from bot.environment import SoundData, SoundEvent, Environment
from tests.hsm_instance_state import device_peripherals


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)

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

def test_speaker_transduces_attached_controller_signal_into_environment_sound() -> None:
    """A speaker converts signal into acoustic energy for the environment, the mirror of a microphone.

    ``devices.audio.output`` from the controller that attached becomes ``environment.sound`` sourced at
    the speaker, so bots hear playout as environment stimulus rather than as raw device product.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        environment = Environment()
        speaker = audio.Speaker()
        controller = RecordingDevice()
        inside = RecordingDevice()
        outside = RecordingDevice()
        _ = await hsm.started(environment, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(environment, controller, controller.model, hsm.Config(id="controller"))
        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside-speaker"))
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside-speaker"))
        await speaker.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda: speaker.state() == "/Device/attached")
        inside.events.clear()
        outside.events.clear()
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

        await speaker.dispatch(
            environment,
            dataclasses.replace(
                audio.OutputEvent.with_data(data),
                metadata={"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"},
            ),
        )
        await wait_until(lambda: bool([event for event in inside.events if event.name == SoundEvent.name]))

        return inside.events, outside.events, hsm.id(speaker)

    inside_events, outside_events, speaker_id = asyncio.run(run())

    sounds = [event for event in inside_events if event.name == SoundEvent.name]
    assert len(sounds) == 1
    event = sounds[0]
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


def test_unattached_speaker_transduces_nothing() -> None:
    """An unwired speaker is silent, exactly as its physical counterpart is."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        speaker = audio.Speaker()
        inside = RecordingDevice()
        _ = await hsm.started(environment, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside-speaker"))
        inside.events.clear()
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

        await speaker.dispatch(environment, audio.OutputEvent.with_data(data))
        await asyncio.sleep(0)

        return [event for event in inside.events if event.name == SoundEvent.name]

    assert asyncio.run(run()) == []


def test_speaker_does_not_override_firmware_model() -> None:
    """Playout lives on the shell, so a speaker declares no firmware model of its own.

    It still *has* firmware — the inherited default really starts. What is pinned here is only
    that this class does not declare its own, which is what keeps transduction on the shell.
    """

    assert audio.Speaker.firmware_model is Device.firmware_model
