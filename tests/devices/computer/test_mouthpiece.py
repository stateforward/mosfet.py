from mosfet.devices import computer

import asyncio
import typing

import hsm

from mosfet.environment import Environment, SoundData, SoundEvent
from tests.devices.computer.remote import Controller, settle, start, wait_until, wire


def test_mouthpiece_is_a_transducer_without_firmware_or_observation() -> None:
    assert computer.Mouthpiece.observation_name is None


def test_attached_mouthpiece_hands_each_sound_it_hears_to_every_controller_as_received() -> None:
    """One sound in the room reaches each controller once, the same object, from the port."""

    async def run() -> tuple[SoundData, list[tuple[str, list[hsm.Event[typing.Any]]]]]:
        environment = Environment()
        mouthpiece = computer.Mouthpiece()
        first, second = Controller(), Controller()
        await start(environment, mouthpiece, "mouthpiece")
        await start(environment, first, "first")
        await start(environment, second, "second")
        for controller in (first, second):
            await wire(environment, mouthpiece, controller)
        await wait_until(lambda: first.attached() and second.attached())

        sound = SoundData(audio=b"bot speech", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
        await environment.broadcast(SoundEvent.with_data(sound))
        await wait_until(lambda: bool(first.named(computer.HeardEvent)) and bool(second.named(computer.HeardEvent)))
        return sound, [("first", first.named(computer.HeardEvent)), ("second", second.named(computer.HeardEvent))]

    sound, received = asyncio.run(run())

    for controller_id, heard in received:
        (event,) = heard
        assert isinstance(event.data, computer.HeardData)
        assert event.data.sound is sound
        assert (event.source, event.target) == ("mouthpiece", controller_id)


def test_mouthpiece_does_not_filter_what_it_hears() -> None:
    """The remote microphone's own audio is forwarded too, still its own type: the decision is the app's."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        mouthpiece, controller = computer.Mouthpiece(), Controller()
        await start(environment, mouthpiece, "mouthpiece")
        await start(environment, controller, "controller")
        await wire(environment, mouthpiece, controller)
        await wait_until(controller.attached)

        await environment.broadcast(SoundEvent.with_data(computer.MicrophoneSoundData(audio=b"remote voice")))
        await wait_until(lambda: bool(controller.named(computer.HeardEvent)))
        return controller.named(computer.HeardEvent)

    (heard,) = asyncio.run(run())

    assert isinstance(heard.data, computer.HeardData)
    assert isinstance(heard.data.sound, computer.MicrophoneSoundData)


def test_unattached_mouthpiece_hears_nothing() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        mouthpiece, controller = computer.Mouthpiece(), Controller()
        await start(environment, mouthpiece, "mouthpiece")
        await start(environment, controller, "controller")

        await environment.broadcast(SoundEvent.with_data(SoundData(audio=b"bot speech")))
        await settle()
        return controller.named(computer.HeardEvent)

    assert asyncio.run(run()) == []
