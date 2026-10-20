"""Shared helpers for the bot DSL tests: a processor and a decision processor that select nothing, a voice source and a mouth
for a bot born with a voice, the wrong bot files, and the runner for the explanations' examples.

Each file under ``wrong/`` is a deliberately wrong bot file, kept as ``.py.txt`` so the type checker
does not reject the mistakes on purpose. :func:`copy` puts one into a directory as a ``.py`` file and
:func:`run` runs it there, the way the builder's bench runs what it wrote.
"""

import ast
import asyncio
import collections.abc
import datetime
import pathlib
import shutil
import types
import typing

import hsm
import pydantic

import mosfet
from mosfet import bot
from mosfet.abilities import communication, memory, processing, speaking, vocal
from mosfet.bot import diagnostics
from mosfet.devices import phone
from mosfet.environment import Environment

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


class Undecided(processing.DecisionProcessor):
    """A decision processor that selects nothing: the intuition a bot is defined with, never asked to decide here."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        return ()


WARM = vocal.voice.Entry(
    ref=vocal.voice.Ref(source="lab", key="warm"),
    name="warm",
    description="A warm, unhurried voice.",
    languages=("en",),
)
"""The one voice :class:`Shelf` holds."""


class Shelf(vocal.voice.Catalog):
    """A voice source that holds one voice, :data:`WARM`, and remembers every time it is asked for its voices."""

    def __init__(self) -> None:
        self.source: str = "lab"
        self.asked: list[tuple[vocal.speech.Locale, ...]] = []

    @typing.override
    async def voices(
        self, *, languages: tuple[vocal.speech.Locale, ...] = (), descriptors: vocal.voice.Descriptors | None = None
    ) -> tuple[vocal.voice.Entry, ...]:
        del descriptors
        self.asked.append(languages)
        return (WARM,)


class Studio(vocal.voice.Designer):
    """A designer at :class:`Shelf`'s source: enough to define a bot that designs its voice, never asked to make one."""

    def __init__(self) -> None:
        self.source: str = "lab"
        self.max_description: int = 300

    @typing.override
    async def create(self, request: vocal.voice.DesignRequest) -> vocal.voice.Designed:
        del request
        raise vocal.voice.Error("unavailable", "this studio makes no voices")


class Mute(vocal.speech.SpeechEncoder):
    """A speech encoder that reads everything it is given and says it as a moment of silence."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=24_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        words = "".join([piece.words async for piece in input])
        yield vocal.speech.Audio(pcm=b"\x00\x00", text=words)


class Voices:
    """Speaking's encoder factory: a :class:`Mute` in whichever voice it is given, every call remembered."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, key: str, description: str | None) -> vocal.speech.SpeechEncoder:
        self.calls.append((key, description))
        return Mute()


class Conceiving(processing.Processor):
    """A model that conceives the bot's voice as :data:`WARM` when it is asked to, and otherwise selects nothing.

    It stands in for the makers' inception model, the one place a test may write what is chosen: being given
    a voice at birth is the bot's making, not its choice. ``conceived`` counts the conceptions it made.
    """

    def __init__(self) -> None:
        self.conceived: int = 0

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del drafts
        if vocal.voice.ConceiveEvent.name not in {schema.name for schema in input.schemas}:
            return ()
        self.conceived += 1
        shortlisted = {"conception": {"kind": "shortlisted", "ref": WARM.ref.model_dump()}}
        return (processing.SelectedEvent(event=vocal.voice.ConceiveEvent.name, data=shortlisted),)


def copy(name: str, directory: pathlib.Path) -> pathlib.Path:
    """``wrong/<name>.py.txt`` copied to ``directory/greeter.py``."""

    path = directory / "greeter.py"
    _ = shutil.copyfile(WRONG / f"{name}.py.txt", path)
    return path


def run(path: pathlib.Path, **given: object) -> types.ModuleType:
    """Run the bot file at ``path`` as a fresh module, as importing it would, with the names ``given``.

    It is compiled from its source every time: a rewrite within one second of the same size would
    otherwise reuse the bytecode cached for the version before it. ``given`` are globals the file
    finds already bound, such as an instance the test started for it.
    """

    module = types.ModuleType("greeter")
    module.__file__ = str(path)
    module.__dict__.update(given)
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


_NAMESPACE_MODULES = {
    "bot": bot,
    "phone": phone,
    "memory": memory,
    "communication": communication,
    "speaking": speaking,
    "hsm": hsm,
    "mosfet": mosfet,
    "datetime": datetime,
    "pydantic": pydantic,
}


async def execute(source: str) -> None:
    """Run an explanation's example (top-level ``await`` allowed), then stop whatever it started."""

    environment = Environment()
    namespace: dict[str, object] = {
        **_NAMESPACE_MODULES,
        "processor": Silent(),
        "decider": Undecided(),
        "environment": environment,
        "catalog": Shelf(),
        "voices": Voices(),
    }
    code = typing.cast(types.CodeType, compile(source, "<explanation>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
    try:
        # The examples are this repository's own code, documented in docs/bot_diagnostics.md.
        result = typing.cast(object, eval(code, namespace))
        if asyncio.iscoroutine(result):
            await result
    finally:
        for value in namespace.values():
            if isinstance(value, mosfet.Bot):
                await bot.stop(value)
            elif isinstance(value, hsm.Instance) and value.take_snapshot().ID:
                await hsm.stop(value, environment)
