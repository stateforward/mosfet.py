from bot.devices import audio

import asyncio
import collections.abc
import typing

import hsm
import bot

from bot.device import Device
from bot.protocols import attachment
from bot.environment import SoundData, SoundEvent, Environment
from tests.hsm_instance_state import device_bots, device_peripherals


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
        _ = await bot.started(
            None, microphone, typing.cast(hsm.Model, microphone.model), hsm.Config(id="livekit-microphone")
        )
        _ = await bot.started(None, target, typing.cast(hsm.Model, target.model), hsm.Config(id="phone-audio"))
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


def test_microphone_transduces_one_capture_per_environment_sound_to_each_attached_controller() -> None:
    """A microphone converts what it hears once, and hands it to whatever attached to it.

    The capture runs on the shell, which is the environment's only presence for this device, so one
    ``environment.sound`` yields exactly one ``devices.audio.input`` per attached controller. Delivery
    is addressed, not broadcast: nothing that did not attach hears it.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        microphone = audio.Microphone()
        controller = RecordingDevice()
        bystander = RecordingDevice()

        _ = await bot.started(
            environment, microphone, typing.cast(hsm.Model, microphone.model), hsm.Config(id="microphone")
        )
        _ = await bot.started(
            environment, controller, typing.cast(hsm.Model, controller.model), hsm.Config(id="controller")
        )
        _ = await bot.started(
            environment, bystander, typing.cast(hsm.Model, bystander.model), hsm.Config(id="bystander")
        )
        await microphone.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda: microphone.state() == "/Device/attached")
        controller.events.clear()
        bystander.events.clear()

        def captures(device: RecordingDevice) -> list[hsm.Event[typing.Any]]:
            return [event for event in device.events if event.name == audio.InputEvent.name]

        await environment.broadcast(
            SoundEvent.with_data(
                SoundData(audio=b"heard-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            )
        )
        await wait_until(lambda: len(captures(controller)) >= 1)
        await asyncio.sleep(0)

        return captures(controller), captures(bystander)

    captured, overheard = asyncio.run(run())

    assert len(captured) == 1
    assert captured[0].data == audio.AudioInputData(
        audio=b"heard-audio",
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
    )
    assert captured[0].source == "microphone"
    assert captured[0].target == "controller"
    # Nothing attached, nothing heard: a transducer feeds its controllers, not the environment.
    assert overheard == []


def test_unattached_microphone_transduces_nothing() -> None:
    """An unwired microphone produces no signal, exactly as its physical counterpart does not."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        microphone = audio.Microphone()
        listener = RecordingDevice()

        _ = await bot.started(
            environment, microphone, typing.cast(hsm.Model, microphone.model), hsm.Config(id="microphone")
        )
        _ = await bot.started(environment, listener, typing.cast(hsm.Model, listener.model), hsm.Config(id="listener"))
        listener.events.clear()

        await environment.broadcast(
            SoundEvent.with_data(
                SoundData(audio=b"heard-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            )
        )
        await asyncio.sleep(0)

        return [event for event in listener.events if event.name == audio.InputEvent.name]

    assert asyncio.run(run()) == []


def test_microphone_does_not_override_firmware_model() -> None:
    """Capture lives on the shell, so a microphone declares no firmware model of its own.

    It still *has* firmware — the inherited default really starts, so a lone microphone is two
    machines in the addressing map. What is pinned here is only that this class does not declare
    its own, which is what keeps transduction on the shell.
    """

    assert audio.Microphone.firmware_model is Device.firmware_model


class HalfAttachedMicrophone(audio.Microphone):
    """Microphone whose attach handshake never completes, holding it in ``attaching``."""

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        if event.name == attachment.AttachCompleteEvent.name:
            done = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        return super().dispatch(ctx, event)


def test_microphone_mid_attach_transduces_nothing() -> None:
    """Signal must not flow before the attach handshake completes.

    ``Attachment._attach`` records the actor on the edge *into* ``attaching``, so the attachment
    list is already populated there and only the state gate holds transduction back. Without it,
    audio goes up the wire during a window that lasts until the attach timeout.
    """

    async def run() -> tuple[str, list[hsm.Event[typing.Any]], tuple[hsm.Instance, ...]]:
        environment = Environment()
        microphone = HalfAttachedMicrophone()
        controller = RecordingDevice()

        _ = await bot.started(
            environment, microphone, typing.cast(hsm.Model, microphone.model), hsm.Config(id="microphone")
        )
        _ = await bot.started(
            environment, controller, typing.cast(hsm.Model, controller.model), hsm.Config(id="controller")
        )
        await microphone.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda: microphone.state() == "/Device/attaching")
        controller.events.clear()

        await environment.broadcast(
            SoundEvent.with_data(
                SoundData(audio=b"heard-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            )
        )
        await asyncio.sleep(0)

        return (
            microphone.state(),
            [event for event in controller.events if event.name == audio.InputEvent.name],
            device_bots(microphone),
        )

    state, captured, attached = asyncio.run(run())

    assert state == "/Device/attaching"
    # The attachment is already recorded, so the state gate is the only thing holding capture back.
    assert len(attached) == 1
    assert captured == []
