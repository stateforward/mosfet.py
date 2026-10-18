"""The bot elements: what each registers in the model, and the diagnostic for each misuse (DSL-3..DSL-6, DSL-22..24)."""

import collections.abc
import datetime
import linecache
import pathlib
import typing

import hsm
import pytest

from mosfet import bot, environment
from mosfet.abilities import memory
from mosfet.bot import elements
from mosfet.devices import audio, phone
from tests.bot.dsl import support


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _written(location: hsm.Location) -> str:
    assert isinstance(location, bot.Location)
    return linecache.getline(location.filename, location.lineno).encode()[location.col : location.end_col].decode()


@pytest.mark.parametrize(
    "element",
    [
        bot.attach(phone.Phone()),
        bot.input(memory.StmMemory()),
        bot.output(memory.StmMemory()),
        bot.register(memory.StmMemory()),
        bot.traits("curious"),
        bot.cognition(support.Silent()),
        bot.timeouts(),
    ],
)
def test_every_element_is_an_hsm_redefinable_element(element: bot.Element[hsm.Element]) -> None:
    assert isinstance(element, hsm.RedefinableElement)
    assert isinstance(element.location, bot.Location)


def test_an_element_remembers_the_call_that_wrote_it() -> None:
    element = bot.attach(phone.Phone(id="desk"))

    assert _written(element.location) == 'bot.attach(phone.Phone(id="desk"))'


def test_an_element_function_starts_nothing() -> None:
    handset = phone.Phone()

    _ = bot.attach(handset)

    assert handset.take_snapshot().ID == ""


def test_a_device_is_held_under_its_id() -> None:
    handset = phone.Phone(id="desk")

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(handset))

    member = model.members["/greeter/desk"]
    assert isinstance(member, elements.AttachmentElement)
    assert member.instance is handset
    assert member.roles == ["attach"]


def test_an_ability_is_keyed_by_its_id_in_every_role() -> None:
    sense, voice = memory.Memory(id="recall"), memory.Memory(id="voice")

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.input(sense), bot.output(voice))

    assert _instance(model, "/greeter/recall") is sense
    assert _instance(model, "/greeter/voice") is voice


def _instance(model: bot.Model, key: str) -> object:
    member = model.members[key]
    assert isinstance(member, elements.AttachmentElement)
    return member.instance


def test_one_ability_given_as_input_and_acquired_is_one_member_with_both_roles() -> None:
    sense = memory.Memory()

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.input(sense), bot.attach(sense))

    member = model.members["/greeter/memory"]
    assert isinstance(member, elements.AttachmentElement)
    assert member.roles == ["input", "attach"]


def test_an_instance_passed_to_another_part_and_attached_is_the_same_object() -> None:
    speaker = audio.Speaker()
    handset = phone.Phone(speaker=speaker)

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(handset), bot.attach(speaker))

    assert _instance(model, "/greeter/speaker") is speaker
    assert _instance(model, "/greeter/phone") is handset


def test_the_singular_elements_register_at_their_keys() -> None:
    store, register = memory.Memory(), memory.StmMemory()
    model = bot.define(
        "greeter",
        bot.traits("curious", "patient"),
        bot.cognition(support.Silent(), memory=store, associative=True, autonomy=True),
        bot.timeouts(processing=datetime.timedelta(seconds=5)),
        bot.register(register),
        bot.attach(phone.Phone()),
    )

    traits = model.members["/greeter/.traits"]
    cognition = model.members["/greeter/.cognition"]
    timeouts = model.members["/greeter/.timeouts"]
    sensory = model.members["/greeter/.register"]
    assert isinstance(traits, elements.TraitsElement) and traits.traits == ("curious", "patient")
    assert isinstance(cognition, elements.CognitionElement)
    assert (cognition.memory, cognition.associative, cognition.consolidation, cognition.autonomy) == (
        store,
        True,
        False,
        True,
    )
    assert isinstance(timeouts, elements.TimeoutsElement) and timeouts.processing == datetime.timedelta(seconds=5)
    assert isinstance(sensory, elements.RegisterElement) and sensory.store is register


def test_hsm_define_rejects_a_bot_element_as_hsm_rejects_a_misplaced_attribute() -> None:
    with pytest.raises(hsm.ErrorValidatingModel, match="bot.attach must be called within bot.define"):
        _ = hsm.define("machine", bot.attach(phone.Phone()))


@pytest.mark.parametrize(
    "code", ["B0003", "B0004", "B0006", "B0007", "B0008", "B0009", "B0015", "B0016", "B0017", "B0019"]
)
def test_each_misuse_renders_its_golden(code: str, here: pathlib.Path) -> None:
    path = support.copy(code, here)

    error = support.failure(path)

    assert [diagnostic.code for diagnostic in error.diagnostics] == [code]
    assert support.rendered(error) == support.golden(code)


@pytest.mark.parametrize("element", [bot.attach, bot.input, bot.output])
def test_an_object_is_placed_not_given_to_the_bot(
    element: collections.abc.Callable[[typing.Any], bot.Element[hsm.Element]],
) -> None:
    """W8-2: an object does nothing, so no element takes it; a part sits on it through its placement."""

    desk = environment.Object(id="desk")
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(support.Silent()),
            bot.attach(phone.Phone(placement=desk.surface())),
            element(desk),
        )

    (diagnostic,) = caught.value.diagnostics
    assert diagnostic.code == "B0019"
    assert diagnostic.help is not None
    assert "placement=desk.surface()" in diagnostic.help.message
    assert diagnostic.help.suggestion is not None
    assert (diagnostic.help.suggestion.replacement, diagnostic.help.suggestion.applicability) == (
        "",
        "machine-applicable",
    )


def test_a_part_on_an_object_is_held_with_its_placement() -> None:
    desk = environment.Object(id="desk")
    handset = phone.Phone(placement=desk.surface())

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(handset))

    assert _instance(model, "/greeter/phone") is handset


def test_two_devices_of_one_kind_get_a_numbered_id_and_a_reserved_id_its_kind(here: pathlib.Path) -> None:
    error = support.failure(support.copy("unnamed", here))

    assert support.rendered(error) == support.golden("unnamed")


def test_a_duplicate_names_both_places(here: pathlib.Path) -> None:
    error = support.failure(support.copy("B0003", here))

    (duplicate,) = error.diagnostics
    assert (duplicate.primary.lineno, duplicate.primary.label) == (12, "used again here")
    assert [(span.lineno, span.label) for span in duplicate.secondary] == [(11, "first used here")]


def test_one_instance_given_twice_is_a_duplicate() -> None:
    handset = phone.Phone()

    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(handset), bot.attach(handset))

    (duplicate,) = caught.value.diagnostics
    assert (duplicate.code, duplicate.message) == ("B0003", "`phone` is given to `bot.attach` twice")


def test_a_device_and_an_ability_share_one_namespace_of_ids() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(support.Silent()),
            bot.attach(phone.Phone(id="desk")),
            bot.input(memory.Memory(id="desk")),
        )

    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["B0003"]


@pytest.mark.parametrize(
    "first",
    [bot.attach(phone.Phone(id="ear")), bot.attach(memory.Memory(id="ear")), bot.input(memory.Memory(id="ear"))],
)
def test_an_output_ability_shares_the_namespace_of_ids(first: bot.Element[hsm.Element]) -> None:
    """DSL-18, DSL-32: the firmware sends to the bot's devices and abilities by id, its output abilities too."""

    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define("greeter", bot.cognition(support.Silent()), first, bot.output(memory.Memory(id="ear")))

    (duplicate,) = caught.value.diagnostics
    assert (duplicate.code, duplicate.message) == ("B0003", "ability id `ear` is used twice")


def test_an_output_ability_may_not_take_a_reserved_id() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define("greeter", bot.cognition(support.Silent()), bot.output(memory.Memory(id="firmware")))

    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["B0007"]


def test_one_ability_given_as_output_and_input_is_one_member_with_both_roles() -> None:
    voice = memory.Memory(id="voice")

    model = bot.define("greeter", bot.cognition(support.Silent()), bot.output(voice), bot.input(voice))

    member = model.members["/greeter/voice"]
    assert isinstance(member, elements.AttachmentElement)
    assert member.roles == ["output", "input"]


_PREAMBLE = """\
import datetime

from mosfet import bot
from mosfet.abilities import memory
from mosfet.devices import phone

from tests.bot.dsl import support

processor = support.Silent()
"""


@pytest.mark.parametrize(
    ("statement", "code", "message", "replacement"),
    [
        (
            'bot.define("greeter", bot.attach(processor), bot.attach(phone.Phone()))',
            "B0006",
            "`bot.attach` does not take a processor",
            "bot.cognition(processor)",
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.input(memory.StmMemory()))',
            "B0006",
            "`bot.input` does not take a sensory register",
            "bot.register(memory.StmMemory())",
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.register(memory.Memory()))',
            "B0006",
            "`bot.register` does not take a memory",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.register(memory.StmMemory))',
            "B0004",
            "`bot.register` takes a constructed sensory register, not a class",
            "memory.StmMemory()",
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach("desk"))',
            "B0006",
            "`bot.attach` does not take a string",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), processor)',
            "B0017",
            "`bot.define` does not take a processor",
            "bot.cognition(processor)",
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.traits)',
            "B0017",
            "`bot.traits` is given, not called",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor), (bot.attach(phone.Phone()),))',
            "B0017",
            "`bot.define` takes elements one by one, not a `tuple` of them",
            "*(bot.attach(phone.Phone()),)",
        ),
        (
            'bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), datetime.date(2026, 1, 1))',
            "B0017",
            "`bot.define` takes bot elements, not a `date`",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor, "associative"), bot.attach(phone.Phone()))',
            "B0017",
            "`bot.cognition` takes a processor, then faculty elements; this is a `str`",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(phone.Phone()), bot.attach(phone.Phone(id="desk")))',
            "B0006",
            "`bot.cognition` does not take a device",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor, memory=memory.StmMemory()), bot.attach(phone.Phone()))',
            "B0006",
            "`bot.cognition(memory=)` does not take a sensory register",
            None,
        ),
        (
            'bot.define("greeter", bot.cognition(processor, routines=3), bot.attach(phone.Phone()))',
            "B0006",
            "`bot.cognition(routines=)` takes a wall clock, not `int`",
            None,
        ),
    ],
)
def test_a_part_given_to_the_wrong_place_names_the_element_that_takes_it(
    statement: str, code: str, message: str, replacement: str | None, here: pathlib.Path
) -> None:
    path = here / "greeter.py"
    _ = path.write_text(f"{_PREAMBLE}\nmodel = {statement}\n", encoding="utf-8")

    error = support.failure(path)

    (diagnostic,) = [diagnostic for diagnostic in error.diagnostics if diagnostic.code == code]
    assert diagnostic.message == message
    suggestion = diagnostic.help.suggestion if diagnostic.help is not None else None
    assert (suggestion.replacement if suggestion is not None else None) == replacement
