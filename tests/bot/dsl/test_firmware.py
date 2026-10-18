"""The bot's one firmware, running: ``bot.Firmware``, ``bot.send`` and ``bot.report`` (DSL-14, DSL-16..DSL-18, DSL-20).

Pinned here: ``bot.start`` gives a model with firmware a ``bot.Firmware`` of its own, which the body
shows and offers; the firmware acts on its own initiative, sending typed events to the bot's devices
and abilities by id, stamped and in its trigger's trace; an unknown id raises and a peer that is not
running drops the send, counted; a report reaches the body as an ability product and wakes cognition,
whose choice among the firmware's commands is its own.
"""

import asyncio
import collections.abc
import datetime
import pathlib
import typing

import hsm
import pydantic
import pytest

import mosfet
from mosfet import abilities, bot, event, lifecycle
from mosfet.abilities import cognition
from mosfet.bot import testing
from mosfet.device import Device
from mosfet.environment import Environment
from tests import metric_reader


class LightData(pydantic.BaseModel):
    """Turn the light on."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


LightEvent = hsm.Event[LightData](name="tests.firmware.light", schema=LightData)


class RangData(pydantic.BaseModel):
    """The kitchen timer rang."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    minutes: int = pydantic.Field(description="Minutes it ran for.", examples=[10])


RangEvent = hsm.Event[RangData](name="tests.firmware.rang", schema=RangData)


class SetData(pydantic.BaseModel):
    """Set the kitchen timer again."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    minutes: int = pydantic.Field(description="Minutes until it rings.", examples=[10])


SetEvent = hsm.Event[SetData](name="tests.firmware.set", kind=event.EventKind, schema=SetData)


class Lamp(Device):
    """A device that records every light event it is sent."""

    received: list[hsm.Event[typing.Any]]

    def __init__(self, *, id: str = "lamp") -> None:
        super().__init__(id=id)
        self.received = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == LightEvent.name:
            self.received.append(event)
        return super().dispatch(ctx, event)


class Bell(abilities.Ability[pydantic.BaseModel, typing.Any]):
    """An ability that records every light event it is sent."""

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Bell", hsm.initial(hsm.target("quiet")), hsm.state("quiet")
    )
    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__(id="bell")
        self.received = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == LightEvent.name:
            self.received.append(event)
        return super().dispatch(ctx, event)


_SOON = datetime.timedelta(hours=1)


def _soon(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> datetime.timedelta:
    del ctx, instance, event
    return _SOON


def _light_everything(ctx: hsm.Context, instance: bot.Firmware, event: hsm.Event[typing.Any]) -> None:
    bot.send(ctx, instance, "lamp", LightEvent.with_data(LightData()), trigger=event)
    bot.send(ctx, instance, "bell", LightEvent.with_data(LightData()), trigger=event)


def _report_rang(ctx: hsm.Context, instance: bot.Firmware, event: hsm.Event[typing.Any]) -> None:
    bot.report(ctx, instance, RangEvent.with_data(RangData(minutes=60)), trigger=event)


def _set_again(ctx: hsm.Context, instance: bot.Firmware, event: hsm.Event[typing.Any]) -> None:
    del ctx
    data = event.data
    assert isinstance(data, SetData)
    _ = instance.set("minutes", data.minutes)


async def _at_once(duration: datetime.timedelta) -> datetime.datetime:
    """A test clock: the firmware's hour passes at once; every other wait is real."""

    if duration != _SOON:
        await asyncio.sleep(duration.total_seconds())
    return datetime.datetime.now(datetime.UTC)


async def _until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(2000):
        if condition():
            return
        await asyncio.sleep(0)
    raise TimeoutError("the condition never held")


def test_a_firmware_sends_on_its_own_timer_to_a_device_and_an_ability_by_id() -> None:
    """DSL-18 (W7-22, W7-23): nothing cognition chose; the owner's machinery acts by typed events."""

    lamp, bell = Lamp(), Bell()
    model = bot.define(
        "kitchen",
        bot.cognition(testing.Scripted()),
        bot.attach(lamp),
        bot.attach(bell),
        bot.allow("ability_without_tools"),
        bot.initial(bot.target("waiting")),
        bot.state("waiting", bot.transition(bot.after(_soon), bot.effect(_light_everything), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str, str, str]:
        async with testing.running(model, clock=hsm.Clock(after=_at_once)) as running:
            await _until(lambda: bool(lamp.received) and bool(bell.received))
            attributes = running.bot.take_snapshot().Attributes or {}
            return (
                list(lamp.received),
                list(bell.received),
                typing.cast(str, attributes["/kitchen/firmware"]),
                hsm.id(lamp),
                hsm.id(bell),
            )

    lit, rung, firmware_id, lamp_id, bell_id = asyncio.run(run())

    assert [(sent.source, sent.target) for sent in lit] == [(firmware_id, lamp_id)]
    assert [(sent.source, sent.target) for sent in rung] == [(firmware_id, bell_id)]


def test_each_start_of_a_model_with_firmware_runs_a_firmware_of_its_own_shown_like_a_device() -> None:
    """DSL-14, DSL-20 (W7-26, OQ-59): one ``<firmware ref="firmware">`` in the world block, fresh each start."""

    model = bot.define(
        "kitchen",
        bot.cognition(testing.Scripted()),
        bot.attach(Lamp()),
        bot.attribute("minutes", int, 10),
        bot.initial(bot.target("idle")),
        bot.state("idle"),
    )

    async def run() -> tuple[str | None, str, str]:
        async with testing.running(model) as running:
            world = running.environment.model_snapshot(running.bot)
            first = typing.cast(str, (running.bot.take_snapshot().Attributes or {})["/kitchen/firmware"])
        async with testing.running(model) as running:
            second = typing.cast(str, (running.bot.take_snapshot().Attributes or {})["/kitchen/firmware"])
        return world, first, second

    world, first, second = asyncio.run(run())

    assert world is not None
    assert f'<firmware ref="firmware" id="{first}" state="/kitchen_firmware/idle">' in world
    assert "<minutes>10</minutes>" in world
    assert "accepts" not in world
    assert first != second


def test_a_report_wakes_cognition_and_its_choice_among_the_firmwares_commands_reaches_the_firmware() -> None:
    """DSL-16, DSL-17 (OQ-49): a report is an ability product; what cognition does about it is cognition's."""

    processor = testing.Scripted(testing.Choice(stimulus=RangEvent.name, event=SetEvent.name, data={"minutes": 5}))
    model = bot.define(
        "kitchen",
        bot.cognition(processor),
        bot.attach(Lamp()),
        bot.attribute("minutes", int, 0),
        bot.initial(bot.target("running")),
        bot.state("running", bot.transition(bot.after(_soon), bot.effect(_report_rang), bot.target("../rang"))),
        bot.state("rang", bot.transition(bot.on(SetEvent), bot.effect(_set_again), bot.target("../set"))),
        bot.state("set"),
    )

    async def run() -> str:
        async with testing.running(model, clock=hsm.Clock(after=_at_once)) as running:
            await _until(lambda: "/kitchen_firmware/set" in (running.environment.model_snapshot(running.bot) or ""))
            return running.environment.model_snapshot(running.bot) or ""

    world = asyncio.run(run())

    (woken, *_) = processor.woken_by(RangEvent.name)
    assert {schema.name for schema in woken.schemas} >= {SetEvent.name}
    assert woken.actor_events[SetEvent.name] == ("firmware",)
    assert "<minutes>5</minutes>" in world


# bot.send and bot.report, on one firmware


_IDLE = mosfet.define("Idle", hsm.initial(hsm.target("idle")), hsm.state("idle"))


class _Owner(hsm.Instance):
    """Stands in for the body that starts a firmware: records the products it is handed."""

    received: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(ctx: hsm.Context, instance: "_Owner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.received.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Owner",
        hsm.initial(hsm.target("on")),
        hsm.state("on", hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.received = []


_TRIGGER = hsm.Event(name="tests.firmware.tick", id="tick-1", metadata={"traceparent": "00-trace-span-01"})


def _sends(points: tuple[metric_reader.Point, ...], outcome: str) -> float:
    return sum(point.value for point in points if point.attributes == {"bot.outcome": outcome})


def test_send_stamps_the_event_as_the_firmwares_in_its_triggers_trace_and_counts_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lamp, unplugged = Lamp(), Bell()

    async def run() -> tuple[str, str]:
        environment = Environment()
        assert Lamp.model is not None
        _ = await mosfet.started(environment, lamp, Lamp.model)
        firmware = bot.Firmware(model=_IDLE, peers={"lamp": lamp, "spare": unplugged})
        _ = await mosfet.started(environment, firmware, firmware.model)
        try:
            bot.send(environment, firmware, "lamp", LightEvent.with_data(LightData()), trigger=_TRIGGER)
            bot.send(environment, firmware, "spare", LightEvent.with_data(LightData()), trigger=_TRIGGER)
            await _until(lambda: bool(lamp.received))
            return hsm.id(firmware), hsm.id(lamp)
        finally:
            await hsm.stop(firmware, environment)
            await lamp.stop(environment)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.firmware.send.count")
        firmware_id, lamp_id = asyncio.run(run())
        after = reader.points("bot.firmware.send.count")

    (sent,) = lamp.received
    assert (sent.id, sent.source, sent.target) == ("tick-1", firmware_id, lamp_id)
    assert sent.metadata == _TRIGGER.metadata
    assert unplugged.received == []
    assert _sends(after, "delivered") - _sends(before, "delivered") == 1
    assert _sends(after, "dropped") - _sends(before, "dropped") == 1
    assert {key for point in after for key in point.attributes} == {"bot.outcome"}


def test_a_send_to_an_id_the_bot_does_not_have_raises_naming_the_ids_it_has() -> None:
    lamp = Lamp()

    async def run() -> None:
        environment = Environment()
        firmware = bot.Firmware(model=_IDLE, peers={"lamp": lamp})
        _ = await mosfet.started(environment, firmware, firmware.model)
        try:
            bot.send(environment, firmware, "oven", LightEvent.with_data(LightData()), trigger=_TRIGGER)
        finally:
            await hsm.stop(firmware, environment)

    with pytest.raises(LookupError, match="the bot has no device or ability `oven`; its ids are `lamp`"):
        asyncio.run(run())


def test_report_hands_the_body_an_ability_product_stamped_as_the_firmwares_in_its_triggers_trace() -> None:
    owner = _Owner()

    async def run() -> tuple[str, str]:
        environment = Environment()
        _ = await mosfet.started(environment, owner, owner.model)
        firmware = bot.Firmware(model=_IDLE, peers={})
        _ = await mosfet.started(owner.context(), firmware, firmware.model)
        try:
            bot.report(owner.context(), firmware, RangEvent.with_data(RangData(minutes=10)), trigger=_TRIGGER)
            await _until(lambda: bool(owner.received))
            return hsm.id(firmware), hsm.id(owner)
        finally:
            await hsm.stop(firmware, environment)
            await hsm.stop(owner, environment)

    firmware_id, owner_id = asyncio.run(run())

    (product,) = owner.received
    assert isinstance(product.data, cognition.InputData)
    stimulus = typing.cast(hsm.Event[typing.Any], product.data.stimulus)
    assert (stimulus.name, stimulus.data) == (RangEvent.name, RangData(minutes=10))
    assert (stimulus.id, stimulus.source) == ("tick-1", firmware_id)
    assert (product.source, product.target) == (firmware_id, owner_id)
    assert product.metadata == stimulus.metadata == _TRIGGER.metadata


def test_a_firmware_started_by_no_machine_drops_its_report() -> None:
    async def run() -> bool:
        environment = Environment()
        firmware = bot.Firmware(model=_IDLE, peers={})
        _ = await mosfet.started(environment, firmware, firmware.model)
        try:
            bot.report(environment, firmware, RangEvent.with_data(RangData(minutes=10)), trigger=_TRIGGER)
            return lifecycle.is_started(firmware)
        finally:
            await hsm.stop(firmware, environment)

    assert asyncio.run(run())


def test_send_and_report_are_the_firmwares_own() -> None:
    assert bot.send is bot.Firmware.send
    assert bot.report is bot.Firmware.report
