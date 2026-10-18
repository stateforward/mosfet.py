"""The builder-loop proof, core (W7-28): a wrong bot file is fixed from its diagnostics alone.

Each case is a deliberately wrong bot file under ``wrong/``. The test imports it, keeps **only** what
the ``DefinitionError`` says (its rendered text and each diagnostic's suggestion), applies every
machine-applicable replacement to the source back to front, and imports the result again. A model
reading the plain text has the same fix: every replacement appears verbatim in the rendered text.
"""

import collections
import pathlib
import warnings

import pytest

from mosfet import bot
from mosfet.bot import diagnostics
from tests.bot.dsl import support

FIXABLE = [
    "B0001",
    "B0003",
    "B0004",
    "B0006",
    "B0007",
    "B0014",
    "B0017",
    "B0018",
    "B0019",
    "B0020",
    "B0021",
    "conflicting_ids",
    "empty_timeouts",
    "misplaced_ids",
    "mixed",
    "people_without_associative",
    "unnamed",
]
"""The wrong files whose every diagnostic carries a machine-applicable rewrite."""

ALL = [
    *FIXABLE,
    "B0002",
    "B0008",
    "B0009",
    "B0010",
    "B0011",
    "B0012",
    "B0013",
    "B0015",
    "B0016",
    "B0022",
    "B0023",
    "ability_without_tools",
    "command_without_examples",
    "instruction_like_traits",
    "misplaced_singletons",
    "no_devices",
    "two_mouths",
    "unreachable_state",
    "unused_register",
]


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _diagnose(path: pathlib.Path) -> tuple[list[diagnostics.Diagnostic], str]:
    """What running the bot file at ``path`` says: its errors, else its lint warnings, and their text."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _ = support.run(path)
        except bot.DefinitionError as error:
            return list(error.diagnostics), str(error)
    warned = [record.message for record in caught if isinstance(record.message, bot.LintWarning)]
    return [warning.diagnostic for warning in warned], "\n\n".join(str(warning) for warning in warned)


def _machine_applicable(found: list[diagnostics.Diagnostic]) -> list[diagnostics.Suggestion]:
    return [
        diagnostic.help.suggestion
        for diagnostic in found
        if diagnostic.help is not None
        and diagnostic.help.suggestion is not None
        and diagnostic.help.suggestion.applicability == "machine-applicable"
    ]


@pytest.mark.parametrize("name", FIXABLE)
def test_a_wrong_bot_file_defines_clean_with_only_its_diagnostics_applied(name: str, here: pathlib.Path) -> None:
    path = support.copy(name, here)
    found, text = _diagnose(path)
    suggestions = _machine_applicable(found)
    assert found
    assert len(suggestions) == len(found)
    for suggestion in suggestions:
        assert suggestion.replacement in text

    _ = path.write_text(support.apply(path.read_text(encoding="utf-8"), suggestions), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module = support.run(path)

    model = getattr(module, "model", None)
    assert isinstance(model, bot.Model)
    assert model.findings == []


@pytest.mark.parametrize("name", ALL)
def test_a_machine_applicable_suggestion_always_clears_its_diagnostic(name: str, here: pathlib.Path) -> None:
    path = support.copy(name, here)
    found, _ = _diagnose(path)
    suggestions = _machine_applicable(found)
    expected = collections.Counter(
        diagnostic.code
        for diagnostic in found
        if diagnostic.help is None or diagnostic.help.suggestion not in suggestions
    )

    _ = path.write_text(support.apply(path.read_text(encoding="utf-8"), suggestions), encoding="utf-8")
    again, _ = _diagnose(path)

    assert collections.Counter(diagnostic.code for diagnostic in again) == expected
