from mosfet.devices import computer

import asyncio
import dataclasses
import typing

import hsm

from mosfet import media
from mosfet.device import Device
from mosfet.environment import Environment, VisualData, VisualEvent
from tests.devices.computer.remote import Controller, settle, start, wait_until, wire

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}
PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, media_type="image/png")


def still() -> hsm.Event[typing.Any]:
    """A still as the link delivers it to the camera port."""

    return dataclasses.replace(
        computer.StillEvent.with_data(computer.StillData(image=PNG)),
        id="still-1",
        source="link",
        target="camera",
        metadata=dict(_TRACE),
    )


def test_camera_is_a_transducer_without_firmware_or_observation() -> None:
    assert computer.Camera.firmware_model is Device.firmware_model
    assert computer.Camera.observation_name is None


def test_attached_camera_hands_each_still_to_its_controller() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        camera, controller = computer.Camera(), Controller()
        await start(environment, camera, "camera")
        await start(environment, controller, "controller")
        await wire(environment, camera, controller)
        await wait_until(controller.attached)

        _ = await camera.dispatch(hsm.Context(), still())
        await wait_until(lambda: bool(controller.named(computer.StillEvent)))
        return controller.named(computer.StillEvent)

    (forwarded,) = asyncio.run(run())

    assert isinstance(forwarded.data, computer.StillData) and forwarded.data.image is PNG
    assert (forwarded.id, forwarded.source, forwarded.target) == ("still-1", "camera", "controller")
    assert forwarded.metadata == _TRACE


def test_unattached_camera_forwards_nothing() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        camera, controller = computer.Camera(), Controller()
        await start(environment, camera, "camera")
        await start(environment, controller, "controller")

        _ = await camera.dispatch(hsm.Context(), still())
        await settle()
        return controller.named(computer.StillEvent)

    assert asyncio.run(run()) == []


def test_camera_does_not_see_the_room() -> None:
    """Light in the computer's room is not what the remote camera sees: a visual broadcast goes nowhere."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        camera, controller = computer.Camera(), Controller()
        await start(environment, camera, "camera")
        await start(environment, controller, "controller")
        await wire(environment, camera, controller)
        await wait_until(controller.attached)
        controller.events.clear()

        await environment.broadcast(VisualEvent.with_data(VisualData(image=PNG, kind="camera")))
        await settle()
        return [event for event in controller.events if event.source == "camera"]

    assert asyncio.run(run()) == []
