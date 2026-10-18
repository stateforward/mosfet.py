"""The bot's one firmware at define: hsm partials at ``bot.define``'s top level compile into it (DSL-12, DSL-13).

Pinned here: the aliases *are* hsm's functions; partials mix with ``bot.*`` elements; no partials is
no firmware; ``bot.redefine`` recompiles one firmware from every partial; ``mosfet.Bot.model`` is
untouched; and the firmware's errors B0012..B0014, each rendered as its golden.
"""

import datetime
import pathlib
import textwrap
import typing

import hsm
import pydantic
import pytest

import mosfet
from mosfet import bot, event
from mosfet.bot import elements, firmware
from mosfet.devices import phone
from tests.bot.dsl import support

ALIASES = (
    "state",
    "initial",
    "transition",
    "on",
    "entry",
    "exit",
    "activity",
    "effect",
    "guard",
    "after",
    "every",
    "at",
    "when",
    "choice",
    "final",
    "defer",
    "source",
    "target",
    "attribute",
    "operation",
    "on_call",
    "on_set",
    "shallow_history",
    "deep_history",
    "entry_point",
    "exit_point",
    "submachine_state",
)


class SetData(pydantic.BaseModel):
    """Start the kitchen timer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    minutes: int = pydantic.Field(ge=1, le=240, description="Minutes until the timer rings.", examples=[10])


SetEvent = hsm.Event[SetData](name="timer.set", kind=event.EventKind, schema=SetData)


def _minutes(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> datetime.timedelta:
    del ctx, instance, event
    return datetime.timedelta(minutes=1)


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _firmware(model: bot.Model) -> firmware.FirmwareElement:
    found = [member for member in model.members.values() if isinstance(member, firmware.FirmwareElement)]
    assert len(found) == 1
    return found[0]


def _timer() -> bot.Model:
    return bot.define(
        "timer",
        bot.traits("attentive"),
        bot.cognition(support.Silent()),
        bot.attribute("minutes", int, 0),
        bot.initial(bot.target("idle")),
        bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../running"))),
        bot.state("running", bot.transition(bot.after(_minutes), bot.target("../idle"))),
    )


@pytest.mark.parametrize("name", ALIASES)
def test_each_alias_is_hsms_own_function(name: str) -> None:
    assert getattr(bot, name) is getattr(hsm, name)
    assert name in bot.__all__


def test_hsms_define_and_runtime_functions_are_not_aliased() -> None:
    assert bot.define is not hsm.define
    assert bot.redefine is not hsm.redefine
    for name in ("validator", "finalizer", "new", "started", "dispatch", "take_snapshot"):
        assert not hasattr(bot, name)
    assert bot.start is not hsm.start
    assert bot.stop is not hsm.stop


def test_partials_mixed_with_bot_elements_compile_into_one_firmware() -> None:
    model = _timer()

    compiled = _firmware(model).model
    assert _firmware(model).qualified_name == "/timer/.firmware"
    assert compiled.qualified_name == "/timer_firmware"
    assert {"/timer_firmware/idle", "/timer_firmware/running", "/timer_firmware/minutes"} <= set(compiled.members)
    assert compiled.initial == "/timer_firmware/.initial"
    assert "/timer/.cognition" in model.members
    assert "/timer/.traits" in model.members
    assert not any(isinstance(member, hsm.StateElement) for member in model.members.values() if member is not model)


def test_the_firmware_offers_only_its_tool_events_as_commands() -> None:
    commands = _firmware(_timer()).commands()

    assert [command.name for command in commands] == ["timer.set"]
    assert commands[0] is SetEvent


def test_a_bot_with_no_partials_has_no_firmware() -> None:
    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(phone.Phone()))

    assert not any(isinstance(member, firmware.FirmwareElement) for member in model.members.values())


def test_redefine_with_more_partials_recompiles_one_firmware_and_keeps_the_old() -> None:
    model = _timer()

    more = bot.redefine(model, bot.state("ringing"), bot.allow("unreachable_state"))

    assert set(_firmware(more).model.members) > set(_firmware(model).model.members)
    assert "/timer_firmware/ringing" in _firmware(more).model.members
    assert "/timer_firmware/ringing" not in _firmware(model).model.members


def test_redefine_of_a_state_already_defined_is_hsms_duplicate() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.redefine(_timer(), bot.state("idle"))

    (found,) = caught.value.diagnostics
    assert found.code == "B0014"
    assert "state '/timer_firmware/idle' already defined" in found.message


def test_the_bodys_own_model_is_never_touched() -> None:
    before = dict(mosfet.Bot.model.members)

    _ = _timer()

    assert mosfet.Bot.model.members == before


def test_the_firmware_is_a_finalized_hsm_model_of_its_own() -> None:
    compiled = _firmware(_timer()).model

    assert isinstance(compiled, hsm.FinalizedModel)
    assert compiled.qualified_name == "/timer_firmware"


def test_partials_are_kept_as_elements_so_the_model_redefines() -> None:
    model = _timer()

    partials = [element for element in model.owned_elements if isinstance(element, firmware.Partial)]
    assert [type(partial.partial).__name__ for partial in partials] == [
        "RedefinableAttribute",
        "RedefinableInitial",
        "RedefinableState",
        "RedefinableState",
    ]
    assert all(isinstance(element, elements.Element | firmware.Partial) for element in model.owned_elements)


def test_compiling_leaves_the_partials_where_hsm_put_them() -> None:
    idle = bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../idle")))
    before = idle.location

    _ = bot.define("timer", bot.cognition(support.Silent()), bot.initial(bot.target("idle")), idle)

    assert idle.location is before


def test_a_firmware_state_carries_where_it_was_written() -> None:
    model = _timer()

    running = _firmware(model).model.members["/timer_firmware/running"]
    assert running.location.filename == __file__
    assert running.location.lineno == _timer.__code__.co_firstlineno + 8


@pytest.mark.parametrize("code", ["B0012", "B0013", "B0014"])
def test_each_firmware_error_renders_its_golden(code: str, here: pathlib.Path) -> None:
    error = support.failure(support.copy(code, here))

    assert [diagnostic.code for diagnostic in error.diagnostics] == [code]
    assert support.rendered(error, name=_bot_name(code)) == support.golden(code)


def _bot_name(code: str) -> str:
    return {"B0012": "timer", "B0013": "dialer", "B0014": "ticker"}[code]


def test_a_firmware_error_is_one_diagnostic_among_the_others(here: pathlib.Path) -> None:
    path = here / "greeter.py"
    _ = path.write_text(
        textwrap.dedent(
            """\
            from mosfet import bot
            from mosfet.devices import phone
            from tests.bot.dsl import support

            model = bot.define(
                "greeter",
                bot.cognition(support.Silent()),
                bot.attach(phone.Phone(id="desk")),
                bot.attach(phone.Phone(id="desk")),
                bot.state("idle"),
            )
            """
        ),
        encoding="utf-8",
    )

    error = support.failure(path)

    assert [diagnostic.code for diagnostic in error.diagnostics] == ["B0003", "B0014"]
    b0014 = error.diagnostics[1]
    assert b0014.primary.lineno == 10
    assert b0014.message == "hsm rejects the bot's firmware: initial state is required for state machine"


def test_hsms_error_keeps_hsms_message_and_points_at_the_written_partial(here: pathlib.Path) -> None:
    path = here / "greeter.py"
    _ = path.write_text(
        textwrap.dedent(
            """\
            from mosfet import bot
            from tests.bot.dsl import support

            model = bot.define(
                "greeter",
                bot.cognition(support.Silent()),
                bot.initial(bot.target("idle")),
                bot.state("idle", bot.transition(bot.on("go"), bot.target("../nowhere"))),
            )
            """
        ),
        encoding="utf-8",
    )

    (found,) = support.failure(path).diagnostics

    assert found.code == "B0014"
    assert "target Vertex '/greeter_firmware/nowhere' not found" in found.message
    assert (found.primary.lineno, found.primary.col) == (8, 4)
    assert found.help is not None
    assert "../" in found.help.message


def test_a_field_without_a_description_points_at_the_field(here: pathlib.Path) -> None:
    (found,) = support.failure(support.copy("B0012", here)).diagnostics

    assert (found.primary.filename, found.primary.lineno) == (str(here / "greeter.py"), 14)
    assert found.secondary[0].lineno == 23


def test_a_command_named_like_a_devices_event_names_the_device(here: pathlib.Path) -> None:
    (found,) = support.failure(support.copy("B0013", here)).diagnostics

    assert "`phone.dial`" in found.message
    assert [span.lineno for span in found.secondary] == [21]
