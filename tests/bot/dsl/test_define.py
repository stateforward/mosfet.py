"""``bot.define`` and ``bot.redefine``: the walk, the validator, one pass, and the errors (DSL-3, DSL-27)."""

import ast
import asyncio
import datetime
import pathlib
import types
import typing

import hsm
import pytest

import mosfet
from mosfet import bot
from mosfet.bot import diagnostics, elements
from mosfet.devices import audio, phone
from mosfet.environment import Environment
from tests.bot.dsl import support


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _greeter() -> bot.Model:
    return bot.define("greeter", bot.cognition(support.Silent()), bot.attach(phone.Phone(id="desk")))


def test_define_returns_the_bot_model_with_its_members_in_order() -> None:
    model = bot.define(
        "greeter",
        bot.traits("curious"),
        bot.cognition(support.Silent()),
        bot.attach(phone.Phone(id="desk")),
        bot.attach(audio.Speaker()),
    )

    assert isinstance(model, bot.Model)
    assert isinstance(model, hsm.NamespaceElement)
    assert model.qualified_name == "/greeter"
    assert list(model.members) == [
        "/greeter",
        "/greeter/.traits",
        "/greeter/.cognition",
        "/greeter/desk",
        "/greeter/speaker",
    ]
    assert model.findings == []


def test_define_is_the_redefinable_bot_walk_with_its_validator_and_finalizer() -> None:
    elements_ = [bot.cognition(support.Silent()), bot.attach(phone.Phone(id="desk"))]

    model = bot.RedefinableBot(qualified_name="/greeter", owned_elements=list(elements_)).redefine(
        bot.Model(qualified_name=""), []
    )

    assert list(model.members) == ["/greeter", "/greeter/.cognition", "/greeter/desk"]
    assert issubclass(bot.Validator, hsm.ModelValidator)
    assert bot.Finalizer().finalize(model) is model


def test_redefine_makes_a_new_model_and_leaves_the_old_one_alone() -> None:
    model = _greeter()

    more = bot.redefine(model, bot.attach(audio.Speaker()), bot.traits("patient"))

    assert list(model.members) == ["/greeter", "/greeter/.cognition", "/greeter/desk"]
    assert list(more.members) == [
        "/greeter",
        "/greeter/.cognition",
        "/greeter/desk",
        "/greeter/speaker",
        "/greeter/.traits",
    ]
    assert _member(more, "/greeter/desk") is _member(model, "/greeter/desk")


def _member(model: bot.Model, key: str) -> object:
    member = model.members[key]
    assert isinstance(member, elements.AttachmentElement)
    return member.instance


def test_an_id_given_again_by_redefine_is_a_duplicate_with_both_places() -> None:
    model = _greeter()

    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.redefine(model, bot.attach(audio.Speaker(id="desk")))

    (duplicate,) = caught.value.diagnostics
    assert duplicate.code == "B0003"
    assert duplicate.primary.filename == __file__
    assert [span.label for span in duplicate.secondary] == ["first used here"]


def test_a_definition_error_is_hsms_validation_error_and_the_bots() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define("greeter", bot.attach(phone.Phone()))

    error = caught.value
    assert isinstance(error, hsm.ErrorValidatingModel)
    assert isinstance(error, bot.Error)
    first = error.diagnostics[0].primary
    assert str(error).startswith(f"{first.filename}:{first.lineno}: error[B0002]:")
    assert str(error).endswith("error: could not define bot `greeter`: 1 error")


@pytest.mark.parametrize("code", ["B0001", "B0002", "B0010", "B0011"])
def test_each_validator_error_renders_its_golden(code: str, here: pathlib.Path) -> None:
    path = support.copy(code, here)

    error = support.failure(path)

    assert [diagnostic.code for diagnostic in error.diagnostics] == [code]
    name = "Greeter Bot" if code == "B0001" else "greeter"
    assert support.rendered(error, name) == support.golden(code)
    first = error.diagnostics[0].primary
    assert str(error) == f"{path}:{first.lineno}: {support.golden(code)}"


def test_an_already_started_instance_renders_its_golden(here: pathlib.Path) -> None:
    module = support.run(support.copy("B0005", here))
    greeter = typing.cast(typing.Callable[[phone.Phone], bot.Model], module.greeter)

    async def run() -> bot.DefinitionError:
        environment = Environment()
        handset = phone.Phone()
        _ = await mosfet.started(environment, handset, typing.cast(hsm.Model, handset.model))
        try:
            with pytest.raises(bot.DefinitionError) as caught:
                _ = greeter(handset)
        finally:
            await hsm.stop(handset, environment)
        return caught.value

    assert support.rendered(asyncio.run(run())) == support.golden("B0005")


def test_a_part_started_and_stopped_again_is_unstarted_and_defines() -> None:
    handset = phone.Phone()

    async def run() -> None:
        environment = Environment()
        _ = await mosfet.started(environment, handset, typing.cast(hsm.Model, handset.model))
        await hsm.stop(handset, environment)

    asyncio.run(run())
    model = bot.define("greeter", bot.cognition(support.Silent()), bot.attach(handset))

    assert list(model.members) == ["/greeter", "/greeter/.cognition", "/greeter/phone"]
    assert model.findings == []


def test_a_missing_name_is_b0001_and_the_elements_are_still_walked(here: pathlib.Path) -> None:
    path = here / "greeter.py"
    _ = path.write_text(
        "from mosfet import bot\nfrom mosfet.devices import phone\nfrom tests.bot.dsl import support\n\n"
        + "model = bot.define(bot.cognition(support.Silent()), bot.input(phone.Phone()))\n",
        encoding="utf-8",
    )

    error = support.failure(path)

    assert [diagnostic.code for diagnostic in error.diagnostics] == ["B0001", "B0006"]
    assert error.diagnostics[0].message == "`bot.define` takes the bot's name first"


def test_one_pass_reports_every_mistake_in_source_order(here: pathlib.Path) -> None:
    error = support.failure(support.copy("mixed", here))

    assert [diagnostic.code for diagnostic in error.diagnostics] == ["B0001", "B0003", "B0004", "B0006", "B0017"]
    assert [diagnostic.primary.lineno for diagnostic in error.diagnostics] == [11, 14, 15, 16, 17]
    assert support.rendered(error, "Greeter Bot") == support.golden("mixed")
    assert str(error).endswith("error: could not define bot `Greeter Bot`: 5 errors")


def test_no_value_from_the_bot_file_is_rendered(here: pathlib.Path) -> None:
    error = support.failure(support.copy("mixed", here))

    assert "SENTINEL" not in str(error)
    assert "SENTINEL" not in "".join(diagnostic.model_dump_json() for diagnostic in error.diagnostics)


_NAMESPACE_MODULES = {"bot": bot, "phone": phone, "hsm": hsm, "mosfet": mosfet, "datetime": datetime}


async def _execute(source: str) -> None:
    """Run an explanation's example (top-level ``await`` allowed), then stop whatever it started."""

    environment = Environment()
    namespace: dict[str, object] = {**_NAMESPACE_MODULES, "processor": support.Silent(), "environment": environment}
    code = typing.cast(types.CodeType, compile(source, "<explanation>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
    try:
        # The examples are this repository's own code, documented in docs/bot_diagnostics.md.
        result = typing.cast(object, eval(code, namespace))
        if asyncio.iscoroutine(result):
            await result
    finally:
        for value in namespace.values():
            if isinstance(value, hsm.Instance) and value.take_snapshot().ID:
                await hsm.stop(value, environment)


@pytest.mark.parametrize(
    "explanation", diagnostics.EXPLANATIONS, ids=[explanation.code for explanation in diagnostics.EXPLANATIONS]
)
def test_each_explanation_shows_its_mistake_and_the_fix(explanation: diagnostics.Explanation) -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        asyncio.run(_execute(explanation.wrong))
    assert explanation.code in [diagnostic.code for diagnostic in caught.value.diagnostics]

    asyncio.run(_execute(explanation.right))


@pytest.mark.parametrize(
    "code",
    [
        "B0001",
        "B0002",
        "B0003",
        "B0004",
        "B0005",
        "B0006",
        "B0007",
        "B0008",
        "B0009",
        "B0010",
        "B0011",
        "B0015",
        "B0016",
        "B0017",
    ],
)
def test_every_define_code_has_an_explanation(code: str) -> None:
    assert bot.explain(code).startswith(f"error[{code}]: ")
