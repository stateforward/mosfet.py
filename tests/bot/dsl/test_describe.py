"""``bot.describe``: the JSON view of defined bots, with no value and nothing constructed (DSL-8)."""

import datetime
import json
import typing

import hsm
import pydantic
import pytest

from mosfet import bot, event
from mosfet.abilities import communication, memory, speaking
from mosfet.devices import audio, phone
from tests.bot.dsl import support

_SECRET = "SENTINEL-not-for-describe"


class SetData(pydantic.BaseModel):
    """Start the kitchen timer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    minutes: int = pydantic.Field(ge=1, le=240, description="Minutes until the timer rings.", examples=[10])


class CancelData(pydantic.BaseModel):
    """Stop the running timer without it ringing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


SetEvent = hsm.Event[SetData](name="timer.set", kind=event.EventKind, schema=SetData)
CancelEvent = hsm.Event[CancelData](name="timer.cancel", kind=event.EventKind, schema=CancelData)
RangEvent = hsm.Event[None](name="timer.rang")


def _clock() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


_SENSES = memory.Memory(id="senses", database=f"/tmp/{_SECRET}.sqlite3")
GREETER = bot.define(
    "greeter",
    bot.traits("curious", "patient"),
    bot.cognition(
        support.Silent(),
        memory=memory.Memory(database=f"/tmp/{_SECRET}-mind.sqlite3"),
        associative=True,
        routines=_clock,
    ),
    bot.attach(phone.Phone(id="desk")),
    bot.attach(_SENSES),
    bot.input(_SENSES),
    bot.output(memory.Memory(id="voice")),
    bot.attach(audio.Speaker()),
    bot.register(memory.StmMemory(database=f"/tmp/{_SECRET}-register.sqlite3")),
    bot.timeouts(processing=datetime.timedelta(seconds=17)),
    bot.allow("ability_without_tools"),
)


def test_describe_shows_each_bot_its_traits_faculties_and_parts_in_order() -> None:
    description = bot.describe(GREETER)

    assert json.loads(description.model_dump_json()) == json.loads(support.golden("describe"))


def test_a_bot_without_traits_memory_register_timeouts_or_allow_says_so() -> None:
    model = bot.define("plain", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    (outline,) = bot.describe(model).bots

    assert outline == bot.Outline(
        name="plain",
        traits=(),
        faculties=("intuition", "reasoning", "reflection"),
        seeds=(),
        memory=None,
        parts=(bot.Part(id="phone", kind="device", roles=("attach",), constructor="Phone"),),
        sensory_register=None,
        timeouts=(),
        allowed=(),
        firmware=None,
    )


def test_describe_lists_the_voice_among_the_faculties_and_the_seeds_by_name() -> None:
    """W8-10: a bot born with a voice has the voice faculty, and its seeds are named, never their machines."""

    model = bot.define(
        "greeter",
        bot.traits("warm"),
        bot.voice(support.Shelf()),
        bot.cognition(support.Silent(), autonomy=True, seeds=(communication.speech_heard_seed(),)),
        bot.output(speaking.Speaking(encoder=support.Voices())),
        bot.attach(phone.Phone()),
    )

    (outline,) = bot.describe(model).bots

    assert outline.faculties == ("intuition", "reasoning", "reflection", "autonomy", "voice")
    assert outline.seeds == ("SpeechHeard",)
    assert [(part.id, part.roles) for part in outline.parts] == [("speaking", ("output",)), ("phone", ("attach",))]


def _minutes(ctx: hsm.Context, instance: bot.Firmware, event: hsm.Event[typing.Any]) -> datetime.timedelta:
    del ctx, instance, event
    return datetime.timedelta(minutes=1)


TIMER = bot.define(
    "timer",
    bot.cognition(support.Silent()),
    bot.attach(phone.Phone()),
    bot.attribute("minutes", int, 17),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../running"))),
    bot.state(
        "running",
        bot.initial(bot.target("counting")),
        bot.state("counting"),
        bot.transition(bot.on(CancelEvent), bot.target("../idle")),
        bot.transition(bot.on(RangEvent), bot.target("../idle")),
        bot.transition(bot.after(_minutes), bot.target("../done")),
    ),
    bot.final("done"),
)


def test_describe_shows_the_firmware_its_commands_and_its_states() -> None:
    (outline,) = bot.describe(TIMER).bots

    assert outline.firmware == bot.FirmwareOutline(
        commands=("timer.set", "timer.cancel"),
        states=(
            "/timer_firmware/idle",
            "/timer_firmware/running",
            "/timer_firmware/running/counting",
            "/timer_firmware/done",
        ),
    )


def test_a_firmware_description_round_trips_and_carries_no_value() -> None:
    description = bot.describe(TIMER)
    text = description.model_dump_json()

    assert bot.Description.model_validate_json(text) == description
    assert "17" not in text
    assert "Minutes until the timer rings" not in text


def test_describe_takes_many_models_in_order_and_none() -> None:
    first = bot.define("first", bot.cognition(support.Silent()), bot.attach(phone.Phone()))
    second = bot.define("second", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    assert [outline.name for outline in bot.describe(first, second).bots] == ["first", "second"]
    assert bot.describe().bots == ()


def test_a_description_round_trips_through_json() -> None:
    description = bot.describe(GREETER)

    assert bot.Description.model_validate_json(description.model_dump_json()) == description


def test_no_value_given_to_a_part_is_described() -> None:
    text = bot.describe(GREETER).model_dump_json()

    assert _SECRET not in text
    assert "17" not in text


def test_describe_constructs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = bot.define("plain", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    def refuse(self: object, **given: object) -> None:
        del self, given
        raise AssertionError("describe constructed a part")

    monkeypatch.setattr(memory.Memory, "__init__", refuse)

    assert bot.describe(model).bots[0].memory is None


def test_a_description_is_frozen_and_every_field_is_described() -> None:
    description = bot.describe(GREETER)
    with pytest.raises(pydantic.ValidationError):
        description.bots = ()

    for model in (bot.Description, bot.Outline, bot.Part, bot.FirmwareOutline):
        schema = model.model_json_schema()
        properties = _properties(schema)
        assert properties
        assert all("description" in field for field in properties.values())


def _properties(schema: dict[str, typing.Any]) -> dict[str, dict[str, object]]:
    properties = typing.cast(dict[str, dict[str, object]], schema["properties"])
    assert isinstance(properties, dict)
    return properties
