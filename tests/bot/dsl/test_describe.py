"""``bot.describe``: the JSON view of defined bots, with no value and nothing constructed (DSL-8)."""

import datetime
import json
import typing

import pydantic
import pytest

from mosfet import bot
from mosfet.abilities import memory
from mosfet.devices import audio, phone
from tests.bot.dsl import support

_SECRET = "SENTINEL-not-for-describe"


def _clock() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _greeter() -> bot.Model:
    senses = memory.Memory(id="senses", database=f"/tmp/{_SECRET}.sqlite3")
    return bot.define(
        "greeter",
        bot.traits("curious", "patient"),
        bot.cognition(
            support.Silent(),
            memory=memory.Memory(database=f"/tmp/{_SECRET}-mind.sqlite3"),
            associative=True,
            routines=_clock,
        ),
        bot.attach(phone.Phone(id="desk")),
        bot.attach(senses),
        bot.input(senses),
        bot.output(memory.Memory(id="voice")),
        bot.attach(audio.Speaker()),
        bot.register(memory.StmMemory(database=f"/tmp/{_SECRET}-register.sqlite3")),
        bot.timeouts(processing=datetime.timedelta(seconds=17)),
        bot.allow("ability_without_tools"),
    )


def test_describe_shows_each_bot_its_traits_faculties_and_parts_in_order() -> None:
    description = bot.describe(_greeter())

    assert json.loads(description.model_dump_json()) == json.loads(support.golden("describe"))


def test_a_bot_without_traits_memory_register_timeouts_or_allow_says_so() -> None:
    model = bot.define("plain", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    (outline,) = bot.describe(model).bots

    assert outline == bot.Outline(
        name="plain",
        traits=(),
        faculties=("intuition", "reasoning", "reflection"),
        memory=None,
        parts=(bot.Part(id="phone", kind="device", roles=("attach",), constructor="Phone"),),
        sensory_register=None,
        timeouts=(),
        allowed=(),
    )


def test_describe_takes_many_models_in_order_and_none() -> None:
    first = bot.define("first", bot.cognition(support.Silent()), bot.attach(phone.Phone()))
    second = bot.define("second", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    assert [outline.name for outline in bot.describe(first, second).bots] == ["first", "second"]
    assert bot.describe().bots == ()


def test_a_description_round_trips_through_json() -> None:
    description = bot.describe(_greeter())

    assert bot.Description.model_validate_json(description.model_dump_json()) == description


def test_no_value_given_to_a_part_is_described() -> None:
    text = bot.describe(_greeter()).model_dump_json()

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
    description = bot.describe(_greeter())
    with pytest.raises(pydantic.ValidationError):
        description.bots = ()

    for model in (bot.Description, bot.Outline, bot.Part):
        schema = model.model_json_schema()
        properties = _properties(schema)
        assert properties
        assert all("description" in field for field in properties.values())


def _properties(schema: dict[str, typing.Any]) -> dict[str, dict[str, object]]:
    properties = typing.cast(dict[str, dict[str, object]], schema["properties"])
    assert isinstance(properties, dict)
    return properties
