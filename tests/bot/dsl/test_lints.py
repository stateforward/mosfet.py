"""Clippy-style lints: each lint's golden, ``bot.allow``, ``bot.check`` and ``LintWarning`` (DSL-33)."""

import datetime
import pathlib
import subprocess
import sys
import warnings

import hsm
import pydantic
import pytest

from mosfet import bot, event
from mosfet.abilities import memory, people
from mosfet.bot import diagnostics, lints
from mosfet.devices import phone
from tests.bot.dsl import support

LINTED = [
    "no_devices",
    "ability_without_tools",
    "instruction_like_traits",
    "empty_timeouts",
    "unused_register",
    "unreachable_state",
    "command_without_examples",
    "people_without_associative",
]


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _warned(path: pathlib.Path) -> list[bot.LintWarning]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = support.run(path)
    return [record.message for record in caught if isinstance(record.message, bot.LintWarning)]


def test_the_lints_are_the_initial_set_each_explained() -> None:
    assert sorted(lints.LINTS) == sorted(LINTED)
    for name in lints.LINTS:
        assert bot.explain(name).startswith(f"warning[{name}]: ")


@pytest.mark.parametrize("name", LINTED)
def test_each_lint_renders_its_golden(name: str, here: pathlib.Path) -> None:
    warned = _warned(support.copy(name, here))

    assert {warning.diagnostic.code for warning in warned} == {name}
    assert "\n\n".join(str(warning) for warning in warned) == support.golden(name)


def test_a_lint_warns_at_the_bot_files_own_line(here: pathlib.Path) -> None:
    path = support.copy("empty_timeouts", here)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = support.run(path)

    (record,) = [record for record in caught if issubclass(record.category, bot.LintWarning)]
    assert (record.filename, record.lineno) == (str(path), 11)


def test_a_lint_is_a_user_warning_and_the_model_keeps_it(here: pathlib.Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = support.run(support.copy("no_devices", here))

    model = getattr(module, "model", None)
    assert issubclass(bot.LintWarning, UserWarning)
    assert isinstance(model, bot.Model)
    assert [found.code for found in model.findings] == ["no_devices"]


def test_allow_silences_a_lint_for_that_bot_only() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = bot.define(
            "quiet", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.allow("no_devices")
        )
        loud = bot.define("loud", bot.cognition(support.Silent(), intuition=support.Undecided()))

    assert [record.message.diagnostic.code for record in caught if isinstance(record.message, bot.LintWarning)] == [
        "no_devices"
    ]
    assert bot.check(quiet) == ()
    assert [found.code for found in bot.check(loud)] == ["no_devices"]


def test_check_returns_what_define_found() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bot.define(
            "greeter",
            bot.traits("never stop talking"),
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.timeouts(),
            bot.attach(phone.Phone()),
        )

    assert bot.check(model) == tuple(model.findings)
    assert [found.code for found in bot.check(model)] == ["instruction_like_traits", "empty_timeouts"]


def test_timeouts_that_set_only_the_step_timeout_set_a_timeout() -> None:
    model = bot.define(
        "greeter",
        bot.traits("curious"),
        bot.cognition(support.Silent(), intuition=support.Undecided()),
        bot.timeouts(step=datetime.timedelta(minutes=3)),
        bot.attach(phone.Phone()),
    )

    assert bot.check(model) == ()


def test_one_pass_reports_errors_and_lints_but_no_lint_about_a_part_that_failed(here: pathlib.Path) -> None:
    """DSL-27: every error and warning from one pass; a lint about what the bot lacks waits for its parts."""

    path = here / "greeter.py"
    _ = path.write_text(
        "from mosfet import bot\nfrom mosfet.devices import phone\nfrom tests.bot.dsl import support\n\n"
        + "model = bot.define('greeter', bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone), bot.timeouts())\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        error = support.failure(path)

    assert [(found.code, found.level) for found in error.diagnostics] == [
        ("B0004", "error"),
        ("empty_timeouts", "warning"),
    ]
    assert str(error).endswith("error: could not define bot `greeter`: 1 error, 1 warning")
    assert [record for record in caught if isinstance(record.message, bot.LintWarning)] == []


_LINTED_TEST = """\
from mosfet import bot
from mosfet.abilities import processing


class Silent(processing.Processor):
    async def process(self, input, *, drafts=None):
        return ()


class Undecided(processing.DecisionProcessor):
    async def process(self, input, *, drafts=None):
        return ()


def test_a_bot_that_perceives_nothing():
    bot.define("greeter", bot.cognition(Silent(), intuition=Undecided()))
"""


def test_pytests_warning_option_makes_a_lint_fail(tmp_path: pathlib.Path) -> None:
    _ = (tmp_path / "test_greeter.py").write_text(_LINTED_TEST, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tmp_path),
        str(tmp_path),
    ]

    warned = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=120)
    failed = subprocess.run(
        [*command, "-W", "error::mosfet.bot.LintWarning"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert warned.returncode == 0
    assert "warning[no_devices]" in warned.stdout
    assert failed.returncode == 1
    assert "warning[no_devices]" in failed.stdout


@pytest.mark.parametrize(
    "trait",
    [
        "always be kind",
        "Never interrupt",
        "you must listen",
        "reply briefly",
        "respond in kind",
        "do not lie",
        "your friend",
    ],
)
def test_instruction_like_traits_catches_instructions(trait: str) -> None:
    with pytest.warns(bot.LintWarning, match="instruction_like_traits"):
        _ = bot.define(
            "greeter",
            bot.traits(trait),
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.attach(phone.Phone()),
        )


@pytest.mark.parametrize("trait", ["patient", "curious about people", "warm", "speaks French", "loves a quiet morning"])
def test_instruction_like_traits_lets_a_trait_be(trait: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = bot.define(
            "greeter",
            bot.traits(trait),
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.attach(phone.Phone()),
        )


def test_a_trait_longer_than_twelve_words_is_an_instruction() -> None:
    long = "a bot that is " + " ".join(["very"] * 10) + " kind"

    with pytest.warns(bot.LintWarning, match="more than 12 words"):
        _ = bot.define(
            "greeter",
            bot.traits(long),
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.attach(phone.Phone()),
        )


def test_an_input_ability_is_a_sense_for_no_devices_and_unused_register() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = bot.define(
            "greeter",
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.input(memory.Memory(id="senses")),
            bot.register(memory.StmMemory()),
        )


def test_people_on_a_cognition_with_associations_is_not_linted() -> None:
    """W8-9: with `associative=True` every turn's associations tell the bot who is speaking and who is present."""

    mind = memory.Memory()
    model = bot.define(
        "house",
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=mind, associative=False),
        bot.attach(phone.Phone()),
        bot.attach(people.People(memory=mind)),
        bot.allow("people_without_associative"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        heard = bot.define(
            "house",
            bot.cognition(support.Silent(), intuition=support.Undecided(), memory=mind, associative=True),
            bot.attach(phone.Phone()),
            bot.attach(people.People(memory=mind)),
        )

    assert bot.check(model) == ()
    assert bot.check(heard) == ()


def test_people_without_associative_rewrites_a_written_flag() -> None:
    mind = memory.Memory()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bot.define(
            "house",
            bot.cognition(support.Silent(), intuition=support.Undecided(), memory=mind, associative=False),
            bot.attach(phone.Phone()),
            bot.attach(people.People(memory=mind)),
        )

    (found,) = bot.check(model)
    assert found.code == "people_without_associative"
    assert found.help is not None
    assert found.help.suggestion is not None
    assert (found.help.suggestion.replacement, found.help.suggestion.applicability) == ("True", "machine-applicable")
    assert diagnostics.text(found.help.suggestion.span) == "False"


def test_an_ability_with_tools_is_not_linted() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = bot.define(
            "greeter",
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.attach(phone.Phone()),
            bot.attach(memory.Memory()),
        )


def test_an_unknown_lint_is_b0018_with_the_nearest_name(here: pathlib.Path) -> None:
    error = support.failure(support.copy("B0018", here))

    assert support.rendered(error) == support.golden("B0018")


def test_allow_is_given_once() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.allow("no_devices"),
            bot.allow("empty_timeouts"),
        )

    assert [found.code for found in caught.value.diagnostics] == ["B0009"]


def _codes(model: bot.Model) -> list[str]:
    return [found.code for found in bot.check(model)]


def test_a_nested_state_its_parent_enters_is_reached_and_one_only_an_unreached_state_leaves_to_is_not() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bot.define(
            "ticker",
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.initial(bot.target("outer")),
            bot.state("outer", bot.initial(bot.target("inner")), bot.state("inner")),
            bot.state("orphan", bot.transition(bot.on("go"), bot.target("../stranded"))),
            bot.state("stranded"),
        )

    found = [diagnostic for diagnostic in bot.check(model) if diagnostic.code == "unreachable_state"]
    assert [diagnostic.message for diagnostic in found] == [
        "firmware state `orphan` is never entered",
        "firmware state `stranded` is never entered",
    ]


def test_a_firmware_is_something_the_bot_perceives_and_a_command_with_examples_is_not_linted() -> None:
    class SetData(pydantic.BaseModel):
        minutes: int = pydantic.Field(description="Minutes until the timer rings.", examples=[10])

    command = hsm.Event[SetData](name="timer.set", kind=event.EventKind, schema=SetData)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = bot.define(
            "timer",
            bot.cognition(support.Silent(), intuition=support.Undecided()),
            bot.initial(bot.target("idle")),
            bot.state("idle", bot.transition(bot.on(command), bot.target("../idle"))),
        )

    assert _codes(model) == []
