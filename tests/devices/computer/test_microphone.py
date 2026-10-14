from mosfet.devices import audio, computer

import asyncio
import dataclasses
import typing

import hsm

from mosfet.device import Device
from mosfet.environment import Environment, SoundData, SoundEvent
from tests.devices.computer.remote import Controller, settle, start, wait_until, wire

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}
FRAME = audio.InputData(audio=b"remote voice", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)


def frame() -> hsm.Event[typing.Any]:
    """A captured frame as the link delivers it to the microphone port."""

    return dataclasses.replace(
        audio.InputEvent.with_data(FRAME), id="frame-1", source="link", target="microphone", metadata=dict(_TRACE)
    )


def test_microphone_is_a_transducer_without_firmware_or_observation() -> None:
    assert computer.Microphone.firmware_model is Device.firmware_model
    assert computer.Microphone.observation_name is None


def test_attached_microphone_hands_each_frame_to_its_controller() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        microphone, controller = computer.Microphone(), Controller()
        await start(environment, microphone, "microphone")
        await start(environment, controller, "controller")
        await wire(environment, microphone, controller)
        await wait_until(controller.attached)

        _ = await microphone.dispatch(hsm.Context(), frame())
        await wait_until(lambda: bool(controller.named(audio.InputEvent)))
        return controller.named(audio.InputEvent)

    (forwarded,) = asyncio.run(run())

    assert forwarded.data is FRAME
    assert (forwarded.id, forwarded.source, forwarded.target) == ("frame-1", "microphone", "controller")
    assert forwarded.metadata == _TRACE


def test_unattached_microphone_forwards_nothing() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        microphone, controller = computer.Microphone(), Controller()
        await start(environment, microphone, "microphone")
        await start(environment, controller, "controller")

        _ = await microphone.dispatch(hsm.Context(), frame())
        await settle()
        return controller.named(audio.InputEvent)

    assert asyncio.run(run()) == []


def test_microphone_does_not_hear_the_room() -> None:
    """The loop regression: sound in the computer's own room is never captured as the remote person."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        microphone, controller = computer.Microphone(), Controller()
        await start(environment, microphone, "microphone")
        await start(environment, controller, "controller")
        await wire(environment, microphone, controller)
        await wait_until(controller.attached)
        controller.events.clear()

        await environment.broadcast(SoundEvent.with_data(SoundData(audio=b"room", media_type="audio/pcm")))
        await settle()
        return [event for event in controller.events if event.source == "microphone"]

    assert asyncio.run(run()) == []
