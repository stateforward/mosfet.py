"""Shared helpers for the bot DSL tests: a processor that selects nothing, and the wrong bot files.

Each file under ``wrong/`` is a deliberately wrong bot file, kept as ``.py.txt`` so the type checker
does not reject the mistakes on purpose. :func:`load` copies one into a directory as a ``.py`` file
and imports it there, the way the builder's bench runs what it wrote.
"""

import collections.abc
import pathlib
import shutil
import types
import typing

from mosfet import bot
from mosfet.abilities import processing
from mosfet.bot import diagnostics

WRONG = pathlib.Path(__file__).parent / "wrong"
GOLDEN = pathlib.Path(__file__).parent / "golden"


class Silent(processing.Processor):
    """A processor that selects nothing: enough to define a bot, never asked to think in these tests."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        return ()


def copy(name: str, directory: pathlib.Path) -> pathlib.Path:
    """``wrong/<name>.py.txt`` copied to ``directory/greeter.py``."""

    path = directory / "greeter.py"
    _ = shutil.copyfile(WRONG / f"{name}.py.txt", path)
    return path


def run(path: pathlib.Path) -> types.ModuleType:
    """Run the bot file at ``path`` as a fresh module, as importing it would.

    It is compiled from its source every time: a rewrite within one second of the same size would
    otherwise reuse the bytecode cached for the version before it.
    """

    module = types.ModuleType("greeter")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def failure(path: pathlib.Path) -> bot.DefinitionError:
    """The ``DefinitionError`` importing the bot file at ``path`` raises."""

    try:
        _ = run(path)
    except bot.DefinitionError as error:
        return error
    raise AssertionError(f"{path.name} defined without an error")


def rendered(error: bot.DefinitionError, name: str = "greeter") -> str:
    """Every diagnostic of ``error`` and its summary, as ``str(error)`` shows them after hsm's ``file:line:``."""

    return diagnostics.report(error.diagnostics, action="define", name=name)


def golden(name: str) -> str:
    return (GOLDEN / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


def apply(source: str, suggestions: collections.abc.Iterable[diagnostics.Suggestion]) -> str:
    """``source`` with every suggestion's replacement applied, back to front so earlier spans stay put."""

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = sorted(
        (
            (offsets[s.span.lineno - 1] + s.span.col, offsets[s.span.end_lineno - 1] + s.span.end_col, s.replacement)
            for s in suggestions
        ),
        reverse=True,
    )
    for start, end, replacement in edits:
        source = source[:start] + replacement + source[end:]
    return source
