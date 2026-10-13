from mosfet.devices import hid

import asyncio
import collections.abc
import dataclasses
import typing

import hsm
import mosfet

from mosfet.device import Device
from mosfet.environment import Environment, SoundData, SoundEvent
from mosfet.protocols import attachment

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}


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
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        del ctx
        self.events.append(event)

        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        done.set_result(True)
        return done

    def pointers(self) -> list[hsm.Event[typing.Any]]:
        return [event for event in self.events if event.name == hid.PointerEvent.name]

    def attached(self) -> bool:
        return any(event.name == attachment.AttachCompleteEvent.name for event in self.events)


def pointer_event(data: object) -> hsm.Event[typing.Any]:
    """A pointer transition as the link delivers it to the pointer port: addressed from the link to the port."""

    return dataclasses.replace(
        hid.PointerEvent, data=data, id="pointer-1", source="link", target="pointer", metadata=dict(_TRACE)
    )


async def start(environment: Environment, device: Device, device_id: str) -> None:
    _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model), hsm.Config(id=device_id))


def test_pointer_is_a_transducer_without_firmware_or_observation() -> None:
    pointer = hid.Pointer()

    assert isinstance(pointer, Device)
    assert hid.Pointer.firmware_model is Device.firmware_model
    assert hid.Pointer.observation_name is None
    assert hid.Pointer.required_bot_abilities == ()


def test_attached_pointer_readdresses_each_pointer_transition_to_every_attached_controller() -> None:
    """One pointer transition from the link reaches each attached controller once, restamped from the port."""

    data = hid.PointerData(action="down", pointer_type="touch", pointer_id=1, x=0.25, y=0.5, pressure=0.4)

    async def run() -> tuple[RecordingDevice, RecordingDevice, RecordingDevice]:
        environment = Environment()
        pointer = hid.Pointer()
        first, second, bystander = RecordingDevice(), RecordingDevice(), RecordingDevice()
        await start(environment, pointer, "pointer")
        await start(environment, first, "first")
        await start(environment, second, "second")
        await start(environment, bystander, "bystander")
        for controller in (first, second):
            _ = await pointer.attach(
                environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller))
            )
        await wait_until(lambda: first.attached() and second.attached())

        _ = await pointer.dispatch(hsm.Context(), pointer_event(data))
        await wait_until(lambda: len(first.pointers()) >= 1 and len(second.pointers()) >= 1)
        await asyncio.sleep(0)
        return first, second, bystander

    first, second, bystander = asyncio.run(run())

    for controller, controller_id in ((first, "first"), (second, "second")):
        (forwarded,) = controller.pointers()
        assert forwarded.data is data
        assert forwarded.id == "pointer-1"
        assert forwarded.source == "pointer"
        assert forwarded.target == controller_id
        assert forwarded.metadata == _TRACE
    assert bystander.pointers() == []


def test_unattached_pointer_forwards_nothing() -> None:
    """An unplugged pointer reaches nobody, exactly as its physical counterpart does not."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        pointer = hid.Pointer()
        listener = RecordingDevice()
        await start(environment, pointer, "pointer")
        await start(environment, listener, "listener")

        _ = await pointer.dispatch(
            hsm.Context(),
            pointer_event(hid.PointerData(action="down", pointer_type="mouse", pointer_id=0, x=0.5, y=0.5)),
        )
        await asyncio.sleep(0)
        return listener.pointers()

    assert asyncio.run(run()) == []


class HalfAttachedPointer(hid.Pointer):
    """Pointer whose attach handshake never completes, holding it in ``attaching``."""

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == attachment.AttachCompleteEvent.name:
            done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            done.set_result(False)
            return done
        return super().dispatch(ctx, event)


def test_pointer_mid_attach_forwards_nothing() -> None:
    """Pointer transitions must not flow before the attach handshake completes, though the actor is already recorded."""

    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        pointer = HalfAttachedPointer()
        controller = RecordingDevice()
        await start(environment, pointer, "pointer")
        await start(environment, controller, "controller")
        _ = await pointer.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda: pointer.state() == "/Device/attaching")

        _ = await pointer.dispatch(
            hsm.Context(),
            pointer_event(hid.PointerData(action="down", pointer_type="mouse", pointer_id=0, x=0.5, y=0.5)),
        )
        await asyncio.sleep(0)
        return pointer.state(), controller.pointers()

    state, forwarded = asyncio.run(run())

    assert state == "/Device/attaching"
    assert forwarded == []


def test_pointer_does_not_forward_an_unvalidated_payload() -> None:
    """The link validates wire data into ``PointerData``; a raw mapping is not a pointer transition and goes nowhere."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        pointer = hid.Pointer()
        controller = RecordingDevice()
        await start(environment, pointer, "pointer")
        await start(environment, controller, "controller")
        _ = await pointer.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(controller.attached)

        _ = await pointer.dispatch(
            hsm.Context(),
            pointer_event(
                {"kind": "hid.pointer", "action": "down", "pointer_type": "mouse", "pointer_id": 0, "x": 0.5, "y": 0.5}
            ),
        )
        await asyncio.sleep(0)
        return controller.pointers()

    assert asyncio.run(run()) == []


def test_pointer_does_not_handle_environment_broadcasts() -> None:
    """A pointer is a remote-side port: sound in the room is not a pointer transition, so nothing is forwarded."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        pointer = hid.Pointer()
        controller = RecordingDevice()
        await start(environment, pointer, "pointer")
        await start(environment, controller, "controller")
        _ = await pointer.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(controller.attached)
        controller.events.clear()

        await environment.broadcast(
            SoundEvent.with_data(SoundData(audio=b"room", media_type="audio/pcm", sample_rate_hz=16_000, channels=1))
        )
        await asyncio.sleep(0)
        return [event for event in controller.events if event.source == "pointer"]

    assert asyncio.run(run()) == []
