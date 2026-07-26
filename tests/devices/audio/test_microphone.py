from bot.devices import audio

import asyncio
import collections.abc
import typing

import hsm

from bot.device import Device
from bot.world import SoundData, SoundEvent, World
from tests.hsm_instance_state import device_firmware, device_peripherals


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


def test_microphone_emits_one_audio_input_per_world_sound() -> None:
    """One ``world.sound`` broadcast reaches microphone firmware once, so it captures once.

    Presence is the device shell; firmware stays addressable in the world instance map and
    hears the stimulus only through its shell's forward, never as a second broadcast recipient.

    The recorder counts *emissions*, not deliveries: it overrides ``dispatch`` and does not
    forward to its own firmware. How many times a captured ``devices.audio.input`` is delivered
    downstream is an addressing-plane question this test says nothing about.
    """

    async def run() -> list[hsm.Event[typing.Any]]:
        world = World()
        microphone = audio.Microphone()
        listener = RecordingDevice()

        _ = await hsm.started(world, microphone, microphone.model, hsm.Config(id="microphone"))
        _ = await hsm.started(world, listener, listener.model, hsm.Config(id="listener"))
        await wait_until(lambda: device_firmware(microphone) is not None)
        listener.events.clear()

        def audio_inputs() -> list[hsm.Event[typing.Any]]:
            return [event for event in listener.events if event.name == audio.InputEvent.name]

        # The capture runs in a firmware effect, so awaiting the broadcast does not guarantee it
        # ran: HSM.dispatch can return the processing wait with the event still queued. Wait for
        # the capture, then settle one turn so a second one would be counted rather than missed.
        await world.broadcast(
            SoundEvent.with_data(
                SoundData(audio=b"heard-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            )
        )
        await wait_until(lambda: len(audio_inputs()) >= 1)
        await asyncio.sleep(0)

        return audio_inputs()

    captured = asyncio.run(run())

    assert len(captured) == 1
    assert captured[0].data == audio.AudioInputData(
        audio=b"heard-audio",
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
    )
