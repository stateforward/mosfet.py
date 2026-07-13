from __future__ import annotations

import collections.abc
import asyncio
import typing

import hsm

from bot import abilities
from bot import habit
from bot.abilities import cognition

from bot.devices import audio as audio_device
from bot.devices import phone as phone_device

from bot.device import Device

def _record_ability_terminal_mirror_event(
    ctx: hsm.Context,
    instance: "_AbilityTerminalMirror",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    instance.record(event)

class _AbilityTerminalMirror(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "AbilityTerminalMirror",
        hsm.initial(hsm.target("/AbilityTerminalMirror/recording")),
        hsm.state(
            "recording",
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.effect(_record_ability_terminal_mirror_event),
            ),
        ),
    )
    ability: abilities.Ability[typing.Any, typing.Any]
    results: dict[str, asyncio.Future[object]]

    def __init__(self, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
        super().__init__()
        self.ability = ability
        self.results = {}

    def result_for(self, operation_id: str) -> asyncio.Future[object]:
        result = self.results.get(operation_id)
        if result is None:
            result = asyncio.get_running_loop().create_future()
            self.results[operation_id] = result
        return result

    def record(self, event: hsm.Event[typing.Any]) -> None:
        operation_id = event.id if event.id else None
        result = self.results.get(operation_id) if operation_id is not None else None
        product_event = event
        product_data = event.data
        if event.name == cognition.InputEvent.name and isinstance(event.data, cognition.InputData):
            if self.ability.output_event.name == cognition.InputEvent.name:
                _append_if_list(self.ability, "outputs", event.data)
                _append_if_list(self.ability, "handoffs", event.data)
                if result is not None and not result.done():
                    result.set_result(event.data)
                return
            product_event = event.data.stimulus
            product_data = product_event.data if isinstance(product_event, hsm.Event) else product_event
        if isinstance(product_event, hsm.Event) and product_event.name == self.ability.output_event.name:
            _append_if_list(self.ability, "outputs", product_data)
            if result is not None and not result.done():
                result.set_result(product_data)
        if event.name == self.ability.failed_event.name:
            _append_if_list(self.ability, "failures", event.data)
            if result is not None and not result.done():
                failure = event.data
                message = failure.message if isinstance(failure, abilities.FailureData) else str(failure)
                result.set_exception(RuntimeError(message))

def _append_if_list(ability: abilities.Ability[typing.Any, typing.Any], attribute: str, data: object) -> None:
    values = getattr(ability, attribute, None)
    if isinstance(values, list):
        values = typing.cast(list[object], values)
        values.append(data)

TResult = typing.TypeVar("TResult")

async def await_result(awaitable: collections.abc.Awaitable[TResult]) -> TResult:
    return await awaitable

async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    context = hsm.Context() if ctx is None else ctx
    owner = _AbilityTerminalMirror(ability)
    _ = await hsm.started(context, owner, typing.cast(hsm.Model, owner.model))
    _ = await ability.attach(owner=owner, ctx=context)

def bot_has_focus(bot: hsm.Instance) -> bool:
    return bot.state() == "/Bot/active/focused"

def device_bots(device: Device) -> tuple[hsm.Instance, ...]:
    return tuple(typing.cast(collections.abc.Iterable[hsm.Instance], vars(device)["_bots"]))

def device_firmware(device: Device) -> hsm.Instance | None:
    return typing.cast(hsm.Instance | None, object.__getattribute__(device, "_firmware"))

def device_peripherals(device: Device) -> tuple[Device, ...]:
    return typing.cast(tuple[Device, ...], object.__getattribute__(device, "_peripherals"))

def phone_firmware(phone: phone_device.Phone) -> phone_device.PhoneFirmware:
    return typing.cast(phone_device.PhoneFirmware, object.__getattribute__(phone, "_firmware_instance"))

def phone_microphone(phone: phone_device.Phone) -> audio_device.Microphone:
    return typing.cast(audio_device.Microphone, object.__getattribute__(phone, "_microphone"))

def phone_speaker(phone: phone_device.Phone) -> audio_device.Speaker:
    return typing.cast(audio_device.Speaker, object.__getattribute__(phone, "_speaker"))

def phone_current_call_id(firmware: phone_device.PhoneFirmware) -> str | None:
    return typing.cast(str | None, object.__getattribute__(firmware, "_current_call_id"))

def phone_current_transfer_id(firmware: phone_device.PhoneFirmware) -> str | None:
    return typing.cast(str | None, object.__getattribute__(firmware, "_current_transfer_id"))

def phone_current_transfer_target(firmware: phone_device.PhoneFirmware) -> phone_device.TransferTarget | None:
    return typing.cast(phone_device.TransferTarget | None, object.__getattribute__(firmware, "_current_transfer_target"))

def phone_closed_call_ids(firmware: phone_device.PhoneFirmware) -> frozenset[str]:
    return typing.cast(frozenset[str], object.__getattribute__(firmware, "_closed_call_ids"))

def phone_service(firmware: phone_device.PhoneFirmware) -> phone_device.PhoneService:
    return typing.cast(phone_device.PhoneService, object.__getattribute__(firmware, "_service"))

def habit_spec(instance: object) -> habit.source.Source:
    return typing.cast(habit.source.Source, object.__getattribute__(instance, "_spec"))
