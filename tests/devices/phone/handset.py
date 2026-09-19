"""Handset test fixtures shared by every phone device's tests: a hand to hold it and a room around it."""

from mosfet.devices import audio as audio_device

import asyncio
import collections.abc
import datetime
import typing

import hsm

import mosfet
from mosfet.devices import phone as phone_device
from mosfet.environment import Environment, SoundEvent
from mosfet.protocols import attachment
from tests.hsm_instance_state import device_bots, phone_firmware


def firmware_of(phone: phone_device.Phone) -> phone_device.Firmware:
    firmware = phone_firmware(phone)
    assert isinstance(firmware, phone_device.Firmware)
    return firmware


async def wait_until(predicate: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Timed out waiting for phone firmware condition.")


def event_names(recorder: phone_device.EventRecorder) -> list[str]:
    return [event.name for event in recorder.events]


def _record_phone_observation(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event) -> None:
    del ctx, instance, event


class PhoneObservationRecorder(hsm.Instance):
    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name in {audio_device.OutputEvent.name, SoundEvent.name, phone_device.RingingEvent.name}:
            self.events.append(event)
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "PhoneObservationRecorder",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(
                hsm.on(SoundEvent, phone_device.RingingEvent),
                hsm.effect(_record_phone_observation),
            ),
        ),
    )


async def emit_service_event(phone: phone_device.Phone, event: hsm.Event[typing.Any]) -> None:
    await firmware_of(phone).event_recorder().receive(phone.context(), event)


async def answer_call(phone: phone_device.Phone, call_id: str) -> None:
    await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
    await emit_service_event(
        phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id=call_id))
    )


class PhoneHolder(hsm.Instance):
    """Stands in for the bot holding the phone, recording every occasion the handset gives it.

    Attaches the way a bot attaches, so what it receives is exactly what a bot receives.
    """

    occasions: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.occasions = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == mosfet.InputEvent.name:
            self.occasions.append(event)
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "PhoneHolder",
        hsm.initial(hsm.target("holding")),
        hsm.state("holding"),
    )


class RoomOccupant(hsm.Instance):
    """An environment citizen that records everything the room delivers to it.

    Deliberately unfiltered, unlike the recorders above. Selecting by event name makes a
    recorder blind to exactly the regression this exists to catch: a device-plane payload put
    back on the environment bus would simply not be recorded, and the test would pass while the
    room quietly carried a private fact again.
    """

    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        self.received.append(event)
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "RoomOccupant",
        hsm.initial(hsm.target("present")),
        hsm.state("present"),
    )


async def phone_in_a_hand(
    environment: Environment,
    phone: phone_device.Phone | None = None,
) -> tuple[phone_device.Phone, phone_device.Firmware, PhoneHolder, RoomOccupant]:
    """A started phone attached to something holding it, plus somebody standing in the room.

    Any phone: a basic one by default, or the handset passed in (a smart phone is a phone).
    """

    if phone is None:
        phone = phone_device.Phone(answer_timeout=datetime.timedelta(milliseconds=1))
    holder = PhoneHolder()
    bystander = RoomOccupant()
    _ = await mosfet.started(environment, phone, typing.cast(hsm.Model, phone.model))
    _ = await mosfet.started(environment, holder, holder.model, hsm.Config(id="holder"))
    _ = await mosfet.started(environment, bystander, bystander.model, hsm.Config(id="bystander"))
    environment.join(bystander)
    await wait_until(lambda: phone.state() == "/Device/detached")
    await phone.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder)))
    await wait_until(lambda: holder in device_bots(phone))
    return phone, firmware_of(phone), holder, bystander


def occasion_sources(holder: PhoneHolder) -> list[str | None]:
    return [
        occasion.data.observation.event
        for occasion in holder.occasions
        if isinstance(occasion.data, mosfet.InputEventData) and occasion.data.observation is not None
    ]
