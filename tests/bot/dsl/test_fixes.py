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

FIXABLE = ["B0001", "B0003", "B0004", "B0006", "B0007", "B0017", "mixed", "unnamed"]
"""The wrong files whose every diagnostic carries a machine-applicable rewrite."""

ALL = [*FIXABLE, "B0002", "B0008", "B0009", "B0010", "B0011", "B0015", "B0016"]


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _machine_applicable(error: bot.DefinitionError) -> list[diagnostics.Suggestion]:
    return [
        diagnostic.help.suggestion
        for diagnostic in error.diagnostics
        if diagnostic.help is not None
        and diagnostic.help.suggestion is not None
        and diagnostic.help.suggestion.applicability == "machine-applicable"
    ]


@pytest.mark.parametrize("name", FIXABLE)
def test_a_wrong_bot_file_defines_clean_with_only_its_diagnostics_applied(name: str, here: pathlib.Path) -> None:
    path = support.copy(name, here)
    error = support.failure(path)
    text = str(error)
    suggestions = _machine_applicable(error)
    assert len(suggestions) == len(error.diagnostics)
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
    error = support.failure(path)
    suggestions = _machine_applicable(error)
    fixed = {
        id(diagnostic)
        for diagnostic in error.diagnostics
        if diagnostic.help and diagnostic.help.suggestion in suggestions
    }
    expected = collections.Counter(diagnostic.code for diagnostic in error.diagnostics if id(diagnostic) not in fixed)

    _ = path.write_text(support.apply(path.read_text(encoding="utf-8"), suggestions), encoding="utf-8")
    try:
        _ = support.run(path)
        remaining: collections.Counter[str] = collections.Counter()
    except bot.DefinitionError as again:
        remaining = collections.Counter(diagnostic.code for diagnostic in again.diagnostics)

    assert remaining == expected
