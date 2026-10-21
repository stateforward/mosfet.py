"""``bot.define`` and ``bot.redefine``: the walk, the validator, one pass, and the errors (DSL-3, DSL-27)."""

import asyncio
import datetime
import importlib.util
import pathlib
import typing
import warnings

import hsm
import pytest

import mosfet
from mosfet import bot
from mosfet.abilities import identification, memory, processing, speaking
from mosfet.bot import diagnostics, elements
from mosfet.devices import audio, phone
from mosfet.environment import Environment
from tests.bot.dsl import support


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_define_returns_the_bot_model_with_its_members_in_order() -> None:
    model = bot.define(
        "greeter",
        bot.traits("curious"),
        bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
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
    elements_ = [
        bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
        bot.attach(phone.Phone(id="desk")),
    ]

    model = bot.RedefinableBot(qualified_name="/greeter", owned_elements=list(elements_)).redefine(
        bot.Model(qualified_name=""), []
    )

    assert list(model.members) == ["/greeter", "/greeter/.cognition", "/greeter/desk"]
    assert issubclass(bot.Validator, hsm.ModelValidator)
    assert bot.Finalizer().finalize(model) is model


def test_redefine_makes_a_new_model_and_leaves_the_old_one_alone() -> None:
    model = bot.define(
        "greeter",
        bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
        bot.attach(phone.Phone(id="desk")),
    )

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
    model = bot.define(
        "greeter",
        bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
        bot.attach(phone.Phone(id="desk")),
    )

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


@pytest.mark.parametrize("code", ["B0001", "B0002", "B0010", "B0011", "B0021", "B0022", "B0023", "B0024", "B0025"])
def test_each_validator_error_renders_its_golden(code: str, here: pathlib.Path) -> None:
    path = support.copy(code, here)

    error = support.failure(path)

    assert [diagnostic.code for diagnostic in error.diagnostics] == [code]
    name = "Greeter Bot" if code == "B0001" else "greeter"
    assert support.rendered(error, name) == support.golden(code)
    first = error.diagnostics[0].primary
    assert str(error) == f"{path}:{first.lineno}: {support.golden(code)}"


def _voices(kind: identification.Kind = "voice") -> identification.Registry:
    return identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=4),
            policy=identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
        ),
        kind=kind,
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=4,
    )


def test_a_cognition_knows_people_through_the_registries_it_is_given() -> None:
    """R-12, D-R-41: knowing who someone is is memory's; the cognition is given the prints' registries."""

    model = bot.define(
        "house",
        bot.cognition(
            reasoning=support.Silent(),
            reflex=support.Undecided(),
            memory=memory.Memory(),
            associative=True,
            registries=(_voices(), _voices("face")),
        ),
        bot.attach(phone.Phone()),
    )

    assert model.findings == []
    # R-12: there is no people ability beside memory any more.
    assert importlib.util.find_spec("mosfet.abilities.people") is None


def test_a_cognition_registry_that_is_no_registry_or_a_second_of_one_kind_is_b0006() -> None:
    # A bot file's mistake, which the annotation would refuse: what is given is checked, not promised.
    stray = typing.cast(identification.Registry, typing.cast(object, "voices"))

    for registries in ((stray,), (_voices(), _voices())):
        with pytest.raises(bot.DefinitionError) as caught:
            _ = bot.define(
                "house",
                bot.cognition(
                    reasoning=support.Silent(), reflex=support.Undecided(), associative=True, registries=registries
                ),
                bot.attach(phone.Phone()),
            )

        (diagnostic,) = caught.value.diagnostics
        assert diagnostic.code == "B0006"
        assert "registries" in diagnostics.text(diagnostic.primary)


def test_a_voice_with_two_speakings_names_each_of_them(here: pathlib.Path) -> None:
    """W8-10: the voice inception gives is the one mouth's, so two Speakings given to bot.output are B0023."""

    error = support.failure(support.copy("two_mouths", here))

    (diagnostic,) = error.diagnostics
    assert diagnostic.code == "B0023"
    assert [span.label for span in diagnostic.secondary] == ["a `speaking.Speaking`", "a `speaking.Speaking`"]
    assert support.rendered(error) == support.golden("two_mouths")


def test_a_speaking_only_acquired_is_not_the_voices_mouth() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.traits("warm"),
            bot.voice(support.Shelf()),
            bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
            bot.attach(speaking.Speaking(encoder=support.Voices())),
            bot.attach(phone.Phone()),
        )

    errors = [diagnostic.code for diagnostic in caught.value.diagnostics if diagnostic.level == "error"]
    assert errors == ["B0023"]


def test_one_speaking_given_to_bot_output_twice_is_still_one_mouth() -> None:
    mouth = speaking.Speaking(encoder=support.Voices())

    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.traits("warm"),
            bot.voice(support.Shelf()),
            bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
            bot.output(mouth),
            bot.output(mouth),
            bot.attach(phone.Phone()),
        )

    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["B0003"]


_PROCESSORS = (
    "`processing.Processor`",
    "`from mosfet.abilities import processing`",
    "`testing.Scripted()`",
    "`from mosfet.bot import testing`",
)


_DECISIONS = (
    "`processing.DecisionProcessor`",
    "`decision.Processor(...)`",
    "`from mosfet.providers import decision`",
    "`testing.Decider()`",
    "`from mosfet.bot import testing`",
)


def test_a_bot_without_cognition_is_told_where_a_reasoning_processor_and_the_offline_one_come_from() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define("greeter", bot.attach(phone.Phone()))

    (missing,) = caught.value.diagnostics
    assert missing.code == "B0002"
    assert missing.help is not None
    for named in _PROCESSORS:
        assert named in missing.help.message
    assert missing.help.suggestion is not None
    assert missing.help.suggestion.replacement == ", bot.cognition(reasoning=processor)"


def test_a_cognition_without_reasoning_is_b0024_told_where_a_processor_comes_from() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(reasoning=typing.cast(processing.Processor, typing.cast(object, None))),
            bot.attach(phone.Phone()),
        )

    (missing,) = caught.value.diagnostics
    assert (missing.code, missing.message) == ("B0024", "`bot.cognition` has no reasoning")
    assert missing.help is not None
    for named in _PROCESSORS:
        assert named in missing.help.message


def test_a_cognition_without_reflex_defines_and_hands_what_behaviors_leave_to_reasoning() -> None:
    """D-R-52: reflex's decision layer is optional; a cognition is only required to reason."""

    model = bot.define("greeter", bot.cognition(reasoning=support.Silent()), bot.attach(phone.Phone()))

    assert model.findings == []


def test_a_chat_processor_given_as_reflex_is_b0025_named_by_its_type() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(
                reasoning=support.Silent(),
                reflex=typing.cast(processing.DecisionProcessor, typing.cast(object, support.Silent())),
            ),
            bot.attach(phone.Phone()),
        )

    (chatting,) = caught.value.diagnostics
    assert (chatting.code, chatting.message) == (
        "B0025",
        "`bot.cognition(reflex=)` is given a chat processor, not a decision processor",
    )
    assert chatting.primary.label == "this is a `Silent`"
    assert chatting.help is not None
    for named in _DECISIONS:
        assert named in chatting.help.message


def test_a_reflex_that_is_no_processor_is_b0006_taking_a_decision_processor() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(reasoning=support.Silent(), reflex=typing.cast(processing.DecisionProcessor, len)),
            bot.attach(phone.Phone()),
        )

    (wrong,) = caught.value.diagnostics
    assert (wrong.code, wrong.message) == (
        "B0006",
        "`bot.cognition(reflex=)` takes a decision processor (`processing.DecisionProcessor`), "
        + "not `builtin_function_or_method`",
    )
    assert wrong.help is not None
    for named in _DECISIONS:
        assert named in wrong.help.message


def test_a_cognition_given_no_processor_is_told_where_a_processor_and_the_offline_one_come_from() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(reasoning=typing.cast(processing.Processor, len), reflex=support.Undecided()),
            bot.attach(phone.Phone()),
        )

    (wrong,) = caught.value.diagnostics
    assert (wrong.code, wrong.message) == (
        "B0006",
        "`bot.cognition(reasoning=)` takes a `processing.Processor`, not `builtin_function_or_method`",
    )
    assert wrong.help is not None
    for named in _PROCESSORS:
        assert named in wrong.help.message


def test_an_already_started_instance_renders_its_golden(here: pathlib.Path) -> None:
    path = support.copy("B0005", here)

    async def run() -> bot.DefinitionError:
        environment = Environment()
        handset = phone.Phone()
        _ = await mosfet.started(environment, handset, typing.cast(hsm.Model, handset.model))
        try:
            with pytest.raises(bot.DefinitionError) as caught:
                _ = support.run(path, handset=handset)
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
    model = bot.define(
        "greeter", bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()), bot.attach(handset)
    )

    assert list(model.members) == ["/greeter", "/greeter/.cognition", "/greeter/phone"]
    assert model.findings == []


def test_a_missing_name_is_b0001_and_the_elements_are_still_walked(here: pathlib.Path) -> None:
    path = here / "greeter.py"
    _ = path.write_text(
        "from mosfet import bot\nfrom mosfet.devices import phone\nfrom tests.bot.dsl import support\n\n"
        + "model = bot.define(bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()), bot.input(phone.Phone()))\n",
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


def test_the_ids_one_pass_suggests_are_unique_among_themselves_and_the_ids_written(here: pathlib.Path) -> None:
    error = support.failure(support.copy("conflicting_ids", here))

    assert [diagnostic.code for diagnostic in error.diagnostics] == ["B0003", "B0004"]
    assert support.rendered(error) == support.golden("conflicting_ids")
    # The duplicate is renamed `computer`, so the class is constructed as `computer-3`: `computer-2` is written.
    assert [
        diagnostic.help.suggestion.replacement
        for diagnostic in error.diagnostics
        if diagnostic.help and diagnostic.help.suggestion
    ] == [
        '"computer"',
        'computer.Computer(id="computer-3")',
    ]


def test_a_misplaced_parts_rewrite_never_reuses_an_id(here: pathlib.Path) -> None:
    """D-W7-10: a stray part keeps its id while it is free, and the class, the second stray and the
    misplaced part beside an attached `desk` each get an id no other part has or is given."""

    error = support.failure(support.copy("misplaced_ids", here))

    assert [
        (diagnostic.code, diagnostic.help.suggestion.replacement, diagnostic.help.suggestion.applicability)
        for diagnostic in error.diagnostics
        if diagnostic.help and diagnostic.help.suggestion
    ] == [
        ("B0017", "bot.attach(computer.Computer())", "machine-applicable"),
        ("B0004", 'computer.Computer(id="computer-2")', "machine-applicable"),
        ("B0006", 'bot.attach(computer.Computer(id="computer-3"))', "machine-applicable"),
        ("B0017", 'bot.attach(computer.Computer(id="computer-4"))', "machine-applicable"),
    ]


def test_a_misplaced_parts_rewrite_into_an_element_the_bot_has_is_not_certain(here: pathlib.Path) -> None:
    """D-W7-10: a second `bot.cognition` or `bot.register` would be B0009, so that rewrite is maybe-incorrect."""

    error = support.failure(support.copy("misplaced_singletons", here))

    assert [
        (diagnostic.code, diagnostic.help.suggestion.applicability)
        for diagnostic in error.diagnostics
        if diagnostic.help and diagnostic.help.suggestion
    ] == [("B0006", "maybe-incorrect"), ("B0017", "maybe-incorrect")]


def test_a_step_timeout_that_is_not_positive_is_b0011() -> None:
    with pytest.raises(bot.DefinitionError) as caught:
        _ = bot.define(
            "greeter",
            bot.cognition(reasoning=support.Silent(), reflex=support.Undecided()),
            bot.attach(phone.Phone()),
            bot.timeouts(step=datetime.timedelta(0)),
        )

    (found,) = caught.value.diagnostics
    assert (found.code, found.message) == ("B0011", "the step timeout is not a positive `datetime.timedelta`")


def test_no_value_from_the_bot_file_is_rendered(here: pathlib.Path) -> None:
    error = support.failure(support.copy("mixed", here))

    assert "SENTINEL" not in str(error)
    assert "SENTINEL" not in "".join(diagnostic.model_dump_json() for diagnostic in error.diagnostics)


_DEFINE_EXPLANATIONS = [
    explanation for explanation in diagnostics.EXPLANATIONS if not explanation.code.startswith("B01")
]
"""Every explanation but ``bot.start``'s (``B01xx``), whose examples ``test_start`` runs."""


@pytest.mark.parametrize(
    "explanation", _DEFINE_EXPLANATIONS, ids=[explanation.code for explanation in _DEFINE_EXPLANATIONS]
)
def test_each_explanation_shows_its_mistake_and_the_fix(explanation: diagnostics.Explanation) -> None:
    if explanation.level == "error":
        with pytest.raises(bot.DefinitionError) as caught:
            asyncio.run(support.execute(explanation.wrong))
        found = [diagnostic.code for diagnostic in caught.value.diagnostics]
    else:
        with pytest.warns(bot.LintWarning) as warned:
            asyncio.run(support.execute(explanation.wrong))
        found = [record.message.diagnostic.code for record in warned if isinstance(record.message, bot.LintWarning)]
    assert explanation.code in found

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(support.execute(explanation.right))


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
        "B0012",
        "B0013",
        "B0014",
        "B0015",
        "B0016",
        "B0017",
        "B0018",
        "B0019",
        "B0021",
        "B0022",
        "B0023",
        "B0024",
        "B0025",
    ],
)
def test_every_define_code_has_an_explanation(code: str) -> None:
    assert bot.explain(code).startswith(f"error[{code}]: ")
