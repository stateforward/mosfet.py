from bot.devices import audio

import asyncio
import collections.abc
import typing

import hsm

from bot.device import Device
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

def test_microphone_is_generic_audio_input_peripheral() -> None:
    microphone = audio.Microphone()

    assert isinstance(microphone, Device)
    assert device_peripherals(microphone) == ()
    assert audio.Microphone.required_bot_abilities == ()

def test_microphone_dispatches_audio_input_to_target_device() -> None:
    async def run() -> None:
        microphone = audio.Microphone()
        target = RecordingDevice()
        _ = await hsm.started(None, microphone, microphone.model, hsm.Config(id="livekit-microphone"))
        _ = await hsm.started(None, target, target.model, hsm.Config(id="phone-audio"))
        target.events.clear()
        data = audio.AudioInputData(audio=b"captured-audio", media_type="audio/opus", sample_rate_hz=48_000, channels=1)

        await microphone.dispatch_audio_input(
            hsm.Context(),
            target,
            data,
            metadata={"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"},
        )

        assert len(target.events) == 1
        event = target.events[0]
        assert event.name == audio.InputEvent.name
        assert event.data == data
        assert event.source == "livekit-microphone"
        assert event.target == "phone-audio"
        assert event.metadata == {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}

    asyncio.run(run())
