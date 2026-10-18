"""``docs/bot.md`` and the module docstrings of ``mosfet.bot`` run as doctests (DSL-11, W7-28).

The DSL is outside every writer's training data, so its documentation is held to what the code does:
each example on the page and in each module's docstring runs, and a failure prints the example.
"""

import ast
import doctest
import importlib
import pathlib
import pkgutil
import re

import pytest

import mosfet.bot

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_GUIDE = _ROOT / "docs" / "bot.md"
_FENCE = re.compile(r"^```(?:python|pycon)\n(.*?)^```", re.MULTILINE | re.DOTALL)
# Everywhere a bot is written: the docs, the DSL's own sources (their docstrings and explanations), the bot
# examples and every test, including the wrong bot files the diagnostics tests define.
_WRITTEN = (
    _ROOT / "README.md",
    *sorted((_ROOT / "docs").glob("*.md")),
    *sorted((_ROOT / "examples").glob("*/README.md")),
    *sorted((_ROOT / "src" / "mosfet" / "bot").glob("*.py")),
    *sorted((_ROOT / "examples" / "builder_bot").glob("[st]*/**/*.py")),
    *sorted((_ROOT / "examples" / "minimal_bot").glob("[st]*/**/*.py")),
    *sorted((_ROOT / "tests").glob("**/*.py")),
    *sorted((_ROOT / "tests").glob("**/*.py.txt")),
)


def _sources(text: str) -> list[str]:
    """The Python a string holds: its fenced Markdown code blocks, its doctest examples, or itself."""

    if "```" in text:
        return [source for block in _FENCE.finditer(text) for source in _sources(block.group(1))]
    if ">>>" in text:
        return ["".join(example.source for example in doctest.DocTestParser().get_examples(text))]
    return [text]


def _defines(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func).endswith("bot.define")


def _wrappers(source: str, where: str) -> list[str]:
    """Every function in ``source``, and in the code its strings hold, that returns a bot's model.

    A function returns a model when its annotation says so, or when it returns a ``bot.define(...)`` call or a
    name it bound to one.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "bot.define(" in node.value:
            found.extend(wrapper for held in _sources(node.value) for wrapper in _wrappers(held, where))
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        bound = {
            target.id
            for assign in ast.walk(node)
            if isinstance(assign, ast.Assign) and _defines(assign.value)
            for target in assign.targets
            if isinstance(target, ast.Name)
        }
        returned = [given.value for given in ast.walk(node) if isinstance(given, ast.Return)]
        annotated = node.returns is not None and ast.unparse(node.returns) in {"bot.Model", "mosfet.bot.Model"}
        if annotated or any(_defines(value) or isinstance(value, ast.Name) and value.id in bound for value in returned):
            found.append(f"{where}: def {node.name}")
    return found


def test_bot_define_is_the_only_function_that_defines_a_bot() -> None:
    wrappers = [
        found
        for path in _WRITTEN
        for source in (
            _sources(path.read_text(encoding="utf-8")) if path.suffix == ".md" else [path.read_text(encoding="utf-8")]
        )
        for found in _wrappers(source, str(path.relative_to(_ROOT)))
    ]

    assert len(_WRITTEN) > 50
    assert wrappers == []


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
