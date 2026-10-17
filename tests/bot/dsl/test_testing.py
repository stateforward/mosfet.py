"""``mosfet.bot.testing``: the scripted model, a running bot for a test, and its snapshot (DSL-10)."""

import asyncio
import typing

import hsm
import pydantic
import pytest

import mosfet
from mosfet import bot, lifecycle
from mosfet.abilities import cognition, processing
from mosfet.bot import testing
from mosfet.devices import phone
from mosfet.environment import Environment
from tests.bot.dsl import support


class _SetData(pydantic.BaseModel):
    """Set a timer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)
    minutes: int = pydantic.Field(description="Minutes until it rings.", examples=[10])


_SetEvent = hsm.Event[_SetData](name="timer.set", schema=_SetData)
_CancelEvent = hsm.Event[_SetData](name="timer.cancel", schema=_SetData)


def _turn(stimulus: str, *offered: hsm.Event[typing.Any]) -> processing.InputData:
    """A model call on a turn woken by a device observation named ``stimulus``, offering ``offered`` on ``desk``."""

    return processing.InputData(
        input=mosfet.InputEventData(observation=mosfet.StimulusData[object](event=stimulus, data={})),
        schemas=offered,
        actor_events={event.name: ("desk",) for event in offered},
    )


def _woken_by_an_event(event: hsm.Event[typing.Any], *offered: hsm.Event[typing.Any]) -> processing.InputData:
    """A model call on a turn woken by an ability's own observation event."""

    return processing.InputData(
        input=event, schemas=offered, actor_events={schema.name: ("bench",) for schema in offered}
    )


# Scripted


def test_a_scripted_model_makes_its_choice_when_its_event_is_offered_on_a_turn_woken_by_its_stimulus() -> None:
    model = testing.Scripted(testing.Choice(stimulus="device.phone.ringing", event="timer.set", data={"minutes": 3}))

    selected = asyncio.run(model.process(_turn("device.phone.ringing", _SetEvent, _CancelEvent)))

    assert selected == (processing.SelectedEvent(event="timer.set", target="desk", data={"minutes": 3}),)


def test_a_choice_is_made_once() -> None:
    model = testing.Scripted(testing.Choice(stimulus="device.phone.ringing", event="timer.set", data={"minutes": 3}))

    async def run() -> tuple[processing.Events, processing.Events]:
        first = await model.process(_turn("device.phone.ringing", _SetEvent))
        return first, await model.process(_turn("device.phone.ringing", _SetEvent))

    first, second = asyncio.run(run())

    assert [selected.event for selected in first] == ["timer.set"]
    assert second == ()


@pytest.mark.parametrize(
    "call",
    [
        _turn("device.phone.ringing", _CancelEvent),
        _turn("device.phone.hung_up", _SetEvent),
        _turn("device.phone.ringing"),
    ],
    ids=["not offered", "another stimulus", "nothing offered"],
)
def test_otherwise_a_scripted_model_selects_nothing(call: processing.InputData) -> None:
    model = testing.Scripted(testing.Choice(stimulus="device.phone.ringing", event="timer.set"))

    assert asyncio.run(model.process(call)) == ()


def test_a_scripted_model_records_every_call_what_was_offered_and_what_woke_it() -> None:
    model = testing.Scripted()
    ringing = _turn("device.phone.ringing", _SetEvent)
    ran = _woken_by_an_event(_CancelEvent.with_data(_SetData(minutes=1)), _CancelEvent)

    async def run() -> None:
        _ = await model.process(ringing)
        _ = await model.process(ran)

    asyncio.run(run())

    assert model.calls() == (ringing, ran)
    assert model.offered_names() == frozenset({"timer.set", "timer.cancel"})
    assert model.woken_by("device.phone.ringing") == (ringing,)
    assert model.woken_by("timer.cancel") == (ran,)
    assert model.woken_by("device.phone.hung_up") == ()
    assert not model.reflected()


def test_a_scripted_model_knows_when_the_bot_reflected() -> None:
    model = testing.Scripted()
    reflecting = processing.InputData(input=cognition.reflection.SelectInput.model_construct(), schemas=())

    _ = asyncio.run(model.process(reflecting))

    assert model.reflected()


def test_a_choice_reaches_through_reasoning_to_the_turn_it_reasons_about() -> None:
    model = testing.Scripted(testing.Choice(stimulus="device.phone.ringing", event="timer.set", data={"minutes": 3}))
    turn = _turn("device.phone.ringing", _SetEvent)
    reasoning = processing.InputData(input=cognition.reasoning.ProcessorInput(host_input=turn), schemas=(_SetEvent,))

    selected = asyncio.run(model.process(reasoning))

    assert [(chosen.event, chosen.target) for chosen in selected] == [("timer.set", "desk")]
    assert model.woken_by("device.phone.ringing") == (reasoning,)


# running and snapshot


class _Shared(hsm.Instance):
    """A shared actor a bot's parts may need running, such as a server."""

    model: typing.ClassVar[hsm.Model] = mosfet.define("Shared", hsm.initial(hsm.target("on")), hsm.state("on"))


def _greeter() -> bot.Model:
    return bot.define("greeter", bot.cognition(testing.Scripted()), bot.attach(phone.Phone(id="desk")))


def test_running_yields_the_started_bot_and_its_environment_then_stops_both() -> None:
    model = _greeter()
    shared = _Shared()

    async def run() -> tuple[str, bool, bool, mosfet.Bot, bool]:
        async with testing.running(model, actors=(shared,)) as running:
            state = running.bot.state()
            in_environment = running.environment.contains(running.bot)
            actor_started = lifecycle.is_started(shared)
        return state, in_environment, actor_started, running.bot, lifecycle.is_started(shared)

    state, in_environment, actor_started, started, actor_after = asyncio.run(run())

    assert state.startswith("/Bot/active")
    assert in_environment
    assert actor_started
    assert not lifecycle.is_started(started)
    assert not actor_after


def test_running_stops_the_bot_and_its_actors_when_the_block_raises() -> None:
    model = _greeter()
    shared = _Shared()
    seen: list[mosfet.Bot] = []

    async def run() -> None:
        async with testing.running(model, actors=(shared,)) as running:
            seen.append(running.bot)
            raise RuntimeError("the test failed")

    with pytest.raises(RuntimeError, match="the test failed"):
        asyncio.run(run())

    assert not lifecycle.is_started(seen[0])
    assert not lifecycle.is_started(shared)


def test_running_stops_the_actors_it_started_when_the_bot_does_not_start() -> None:
    model = _greeter()
    shared = _Shared()

    async def run() -> None:
        async with testing.running(model, actors=(shared,)):
            pass

    async def twice() -> None:
        async with testing.running(model):
            await run()

    with pytest.raises(bot.StartError):
        asyncio.run(twice())

    assert not lifecycle.is_started(shared)


def test_running_takes_actors_that_declare_their_model() -> None:
    async def run() -> None:
        async with testing.running(_greeter(), actors=(hsm.Instance(),)):
            pass

    with pytest.raises(TypeError, match="declares no model"):
        asyncio.run(run())


def test_snapshot_is_the_bodys_own_snapshot_with_its_devices() -> None:
    async def run() -> tuple[hsm.Snapshot, hsm.Snapshot]:
        async with testing.running(_greeter()) as running:
            return testing.snapshot(running.bot), running.bot.take_snapshot()

    taken, own = asyncio.run(run())

    assert taken.State == own.State
    assert taken.Attributes == own.Attributes
    assert list(typing.cast(dict[str, str], (taken.Attributes or {})["/Bot/owned_devices"])) == ["desk"]


def test_a_run_is_the_bot_and_its_environment() -> None:
    environment = Environment()
    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    async def run() -> testing.Run:
        started = await bot.start(environment, model)
        try:
            return testing.Run(bot=started, environment=environment)
        finally:
            await bot.stop(started)

    made = asyncio.run(run())

    assert made.environment is environment
