"""``docs/bot.md`` and the module docstrings of ``mosfet.bot`` run as doctests (DSL-11, W7-28).

The DSL is outside every writer's training data, so its documentation is held to what the code does:
each example on the page and in each module's docstring runs, and a failure prints the example.
"""

import doctest
import importlib
import pathlib
import pkgutil

import pytest

import mosfet.bot

_GUIDE = pathlib.Path(__file__).resolve().parents[3] / "docs" / "bot.md"
_MODULES = sorted(module.name for module in pkgutil.iter_modules(mosfet.bot.__path__, "mosfet.bot."))
_DOCUMENTED = ("mosfet.bot", "mosfet.bot.define", "mosfet.bot.describe", "mosfet.bot.start", "mosfet.bot.testing")


def test_every_example_in_docs_bot_md_runs() -> None:
    failed, attempted = doctest.testfile(str(_GUIDE), module_relative=False, encoding="utf-8")

    assert attempted > 0
    assert failed == 0


@pytest.mark.parametrize("name", ["mosfet.bot", *_MODULES])
def test_every_example_in_the_modules_docstrings_runs(name: str) -> None:
    failed, _ = doctest.testmod(importlib.import_module(name))

    assert failed == 0


@pytest.mark.parametrize("name", _DOCUMENTED)
def test_the_dsls_own_modules_carry_runnable_examples(name: str) -> None:
    _, attempted = doctest.testmod(importlib.import_module(name))

    assert attempted > 0
