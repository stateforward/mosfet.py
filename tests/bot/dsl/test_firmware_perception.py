"""What the bot's firmware perceives: what is attached to the bot (DSL-31; W7-21, W7-31 OQ-58).

Pinned here: the firmware is a second holder of every device the bot holds, through the attachment
protocol, and lets go of them before it stops; a device's ``bot.input`` reaches it, through the
device's own fan-out, as the device's own typed event, with its id, source and trace; an event its
model does not name is ignored; cognition perceives the same devices as before; and a device that
allows one holder fails the firmware's start with the typed attach failure.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import pydantic
import pytest

import mosfet
from mosfet import bot, lifecycle
from mosfet.bot import testing
from mosfet.device import Device
from mosfet.environment import Environment
from mosfet.protocols import attachment
from tests.hsm_instance_state import device_bots


class SwitchedData(pydantic.BaseModel):
    """The lamp was switched."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    on: bool = pydantic.Field(description="Whether the lamp is now on.", examples=[True])


SwitchedEvent = hsm.Event[SwitchedData](name="tests.lamp.switched", schema=SwitchedData)
FlickeredEvent = hsm.Event[SwitchedData](name="tests.lamp.flickered", schema=SwitchedData)

_TRACE = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}


class Lamp(Device):
    """A device a test can switch: it tells every holder, the way a device tells what happened to it."""

    def __init__(self, *, id: str = "lamp") -> None:
        super().__init__(id=id)

    def switch(self, happened: hsm.Event[SwitchedData], *, id: str) -> None:
        self._tell_holders(self.context(), dataclasses.replace(happened, id=id, source=hsm.id(self), metadata=_TRACE))


class OneHolderLamp(Lamp):
    """A lamp only one machine may hold."""

    _attachment_limit: typing.ClassVar[int | None] = 1


class DeafLamp(Lamp):
    """A lamp that never answers a request to hold it."""

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        del ctx, event
        return True


_seen: list[hsm.Event[typing.Any]] = []


def _record(ctx: hsm.Context, instance: bot.Firmware, event: hsm.Event[typing.Any]) -> None:
    del ctx, instance
    _seen.append(event)


@pytest.fixture(autouse=True)
def forget_what_the_firmware_saw() -> None:
    _seen.clear()


def _firmware_id(running: testing.Run) -> str:
    return typing.cast(str, (running.bot.take_snapshot().Attributes or {})["/kitchen/firmware"])


async def _until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(2000):
        if condition():
            return
        await asyncio.sleep(0)
    raise TimeoutError("the condition never held")


def test_the_firmware_is_a_second_holder_of_every_device_the_bot_holds() -> None:
    lamp, spare = Lamp(), Lamp(id="spare")

    kitchen = bot.define(
        "kitchen",
        bot.cognition(testing.Scripted(), intuition=testing.Decider()),
        bot.attach(lamp),
        bot.attach(spare),
        bot.initial(bot.target("dark")),
        bot.state("dark", bot.transition(bot.on(SwitchedEvent), bot.effect(_record), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> tuple[list[list[str]], str, str]:
        async with testing.running(kitchen) as running:
            holders = [[hsm.id(holder) for holder in device_bots(device)] for device in (lamp, spare)]
            return holders, hsm.id(running.bot), _firmware_id(running)

    holders, body_id, firmware_id = asyncio.run(run())

    assert holders == [[body_id, firmware_id], [body_id, firmware_id]]


def test_the_firmware_lets_go_of_the_devices_before_it_stops() -> None:
    lamp = Lamp()

    kitchen = bot.define(
        "kitchen",
        bot.cognition(testing.Scripted(), intuition=testing.Decider()),
        bot.attach(lamp),
        bot.initial(bot.target("dark")),
        bot.state("dark", bot.transition(bot.on(SwitchedEvent), bot.effect(_record), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> tuple[list[hsm.Instance], hsm.Instance]:
        async with testing.running(kitchen) as running:
            body = running.bot
        return list(device_bots(lamp)), body

    holders, body = asyncio.run(run())

    # The firmware stops first and lets go; the body, stopped after it, is the last holder there was.
    assert all(holder is body for holder in holders)


def test_a_device_event_reaches_the_firmware_as_the_devices_own_typed_event_in_its_trace() -> None:
    lamp = Lamp()
    processor = testing.Scripted()

    kitchen = bot.define(
        "kitchen",
        bot.cognition(processor, intuition=testing.Decider()),
        bot.attach(lamp),
        bot.initial(bot.target("dark")),
        bot.state("dark", bot.transition(bot.on(SwitchedEvent), bot.effect(_record), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> tuple[str, str]:
        async with testing.running(kitchen) as running:
            lamp.switch(SwitchedEvent.with_data(SwitchedData(on=True)), id="switch-1")
            await _until(lambda: bool(_seen) and bool(processor.woken_by(SwitchedEvent.name)))
            world = running.environment.model_snapshot(running.bot) or ""
            return hsm.id(lamp), world

    lamp_id, world = asyncio.run(run())

    (seen,) = _seen
    assert seen.name == SwitchedEvent.name
    assert seen.data == SwitchedData(on=True)
    assert (seen.id, seen.source) == ("switch-1", lamp_id)
    assert seen.metadata == _TRACE
    assert 'state="/kitchen_firmware/lit"' in world
    # Cognition perceives the very same device event: the body is a holder too.
    assert processor.woken_by(SwitchedEvent.name)


def test_an_event_the_firmwares_model_does_not_name_is_ignored() -> None:
    lamp = Lamp()
    processor = testing.Scripted()

    kitchen = bot.define(
        "kitchen",
        bot.cognition(processor, intuition=testing.Decider()),
        bot.attach(lamp),
        bot.initial(bot.target("dark")),
        bot.state("dark", bot.transition(bot.on(SwitchedEvent), bot.effect(_record), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> str:
        async with testing.running(kitchen) as running:
            lamp.switch(FlickeredEvent.with_data(SwitchedData(on=False)), id="flicker-1")
            await _until(lambda: bool(processor.woken_by(FlickeredEvent.name)))
            return running.environment.model_snapshot(running.bot) or ""

    world = asyncio.run(run())

    assert _seen == []
    assert 'state="/kitchen_firmware/dark"' in world


def test_a_bot_without_firmware_is_its_devices_one_holder() -> None:
    lamp = Lamp()
    model = bot.define("kitchen", bot.cognition(testing.Scripted(), intuition=testing.Decider()), bot.attach(lamp))

    async def run() -> tuple[list[hsm.Instance], hsm.Instance]:
        async with testing.running(model) as running:
            return list(device_bots(lamp)), running.bot

    holders, body = asyncio.run(run())

    assert holders == [body]


def test_a_device_that_allows_one_holder_fails_the_firmwares_start_and_so_the_activation() -> None:
    lamp = OneHolderLamp()

    kitchen = bot.define(
        "kitchen",
        bot.cognition(testing.Scripted(), intuition=testing.Decider()),
        bot.attach(lamp),
        bot.initial(bot.target("dark")),
        bot.state("dark", bot.transition(bot.on(SwitchedEvent), bot.effect(_record), bot.target("../lit"))),
        bot.state("lit"),
    )

    async def run() -> str:
        environment = Environment()
        started = await bot.start(environment, kitchen)
        try:
            return started.state()
        finally:
            await bot.stop(started)

    state = asyncio.run(run())

    # bot.start returns the body as its activation failure leaves it: rolling back, or rolled back.
    assert state in {"/kitchen/activation_cleanup", "/kitchen/inactive"}


def test_a_firmware_whose_devices_do_not_answer_fails_its_start_within_its_bound_and_stops_again() -> None:
    lamp = DeafLamp()
    model = mosfet.define("Quiet", hsm.initial(hsm.target("on")), hsm.state("on"))
    firmware = bot.Firmware(model=model, peers={"lamp": lamp}, attachment_timeout=datetime.timedelta(milliseconds=50))

    async def run() -> float:
        environment = Environment()
        assert Lamp.model is not None
        _ = await mosfet.started(environment, lamp, Lamp.model)
        began = asyncio.get_running_loop().time()
        try:
            with pytest.raises((RuntimeError, TimeoutError)):
                _ = await mosfet.started(environment, firmware, model)
            return asyncio.get_running_loop().time() - began
        finally:
            await lamp.stop(environment)

    elapsed = asyncio.run(run())

    assert elapsed < 2.0
    assert not lifecycle.is_started(firmware)
