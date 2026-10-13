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

    def keys(self) -> list[hsm.Event[typing.Any]]:
        return [event for event in self.events if event.name == hid.KeyEvent.name]

    def attached(self) -> bool:
        return any(event.name == attachment.AttachCompleteEvent.name for event in self.events)


def key_event(data: object) -> hsm.Event[typing.Any]:
    """A key as the link delivers it to the keyboard port: addressed from the link to the port."""

    return dataclasses.replace(
        hid.KeyEvent, data=data, id="key-1", source="link", target="keyboard", metadata=dict(_TRACE)
    )


async def start(environment: Environment, device: Device, device_id: str) -> None:
    _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model), hsm.Config(id=device_id))


def test_keyboard_is_a_transducer_without_firmware_or_observation() -> None:
    keyboard = hid.Keyboard()

    assert isinstance(keyboard, Device)
    assert hid.Keyboard.firmware_model is Device.firmware_model
    assert hid.Keyboard.observation_name is None
    assert hid.Keyboard.required_bot_abilities == ()


def test_attached_keyboard_readdresses_each_key_to_every_attached_controller() -> None:
    """One key from the link reaches each attached controller once, restamped from the port."""

    data = hid.KeyData(action="down", key="a", code="KeyA", text="a")

    async def run() -> tuple[RecordingDevice, RecordingDevice, RecordingDevice]:
        environment = Environment()
        keyboard = hid.Keyboard()
        first, second, bystander = RecordingDevice(), RecordingDevice(), RecordingDevice()
        await start(environment, keyboard, "keyboard")
        await start(environment, first, "first")
        await start(environment, second, "second")
        await start(environment, bystander, "bystander")
        for controller in (first, second):
            _ = await keyboard.attach(
                environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller))
            )
        await wait_until(lambda: first.attached() and second.attached())

        _ = await keyboard.dispatch(hsm.Context(), key_event(data))
        await wait_until(lambda: len(first.keys()) >= 1 and len(second.keys()) >= 1)
        await asyncio.sleep(0)
        return first, second, bystander

    first, second, bystander = asyncio.run(run())

    for controller, controller_id in ((first, "first"), (second, "second")):
        (forwarded,) = controller.keys()
        assert forwarded.data is data
        assert forwarded.id == "key-1"
        assert forwarded.source == "keyboard"
        assert forwarded.target == controller_id
        assert forwarded.metadata == _TRACE
    assert bystander.keys() == []


def test_unattached_keyboard_forwards_nothing() -> None:
    """An unplugged keyboard reaches nobody, exactly as its physical counterpart does not."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        keyboard = hid.Keyboard()
        listener = RecordingDevice()
        await start(environment, keyboard, "keyboard")
        await start(environment, listener, "listener")

        _ = await keyboard.dispatch(hsm.Context(), key_event(hid.KeyData(action="down", key="a", code="KeyA")))
        await asyncio.sleep(0)
        return listener.keys()

    assert asyncio.run(run()) == []


class HalfAttachedKeyboard(hid.Keyboard):
    """Keyboard whose attach handshake never completes, holding it in ``attaching``."""

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == attachment.AttachCompleteEvent.name:
            done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            done.set_result(False)
            return done
        return super().dispatch(ctx, event)


def test_keyboard_mid_attach_forwards_nothing() -> None:
    """Keys must not flow before the attach handshake completes, though the actor is already recorded."""

    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        keyboard = HalfAttachedKeyboard()
        controller = RecordingDevice()
        await start(environment, keyboard, "keyboard")
        await start(environment, controller, "controller")
        _ = await keyboard.attach(
            environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller))
        )
        await wait_until(lambda: keyboard.state() == "/Device/attaching")

        _ = await keyboard.dispatch(hsm.Context(), key_event(hid.KeyData(action="down", key="a", code="KeyA")))
        await asyncio.sleep(0)
        return keyboard.state(), controller.keys()

    state, forwarded = asyncio.run(run())

    assert state == "/Device/attaching"
    assert forwarded == []


def test_keyboard_does_not_forward_an_unvalidated_payload() -> None:
    """The link validates wire data into ``KeyData``; a raw mapping is not a key and goes nowhere."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        keyboard = hid.Keyboard()
        controller = RecordingDevice()
        await start(environment, keyboard, "keyboard")
        await start(environment, controller, "controller")
        _ = await keyboard.attach(
            environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller))
        )
        await wait_until(controller.attached)

        _ = await keyboard.dispatch(
            hsm.Context(), key_event({"kind": "hid.key", "action": "down", "key": "a", "code": "KeyA"})
        )
        await asyncio.sleep(0)
        return controller.keys()

    assert asyncio.run(run()) == []


def test_keyboard_does_not_handle_environment_broadcasts() -> None:
    """A keyboard is a remote-side port: sound in the room is not a keystroke, so nothing is forwarded."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        keyboard = hid.Keyboard()
        controller = RecordingDevice()
        await start(environment, keyboard, "keyboard")
        await start(environment, controller, "controller")
        _ = await keyboard.attach(
            environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller))
        )
        await wait_until(controller.attached)
        controller.events.clear()

        await environment.broadcast(
            SoundEvent.with_data(SoundData(audio=b"room", media_type="audio/pcm", sample_rate_hz=16_000, channels=1))
        )
        await asyncio.sleep(0)
        return [event for event in controller.events if event.source == "keyboard"]

    assert asyncio.run(run()) == []
