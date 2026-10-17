"""Diagnostics for bot files: rustc-style errors, clippy-style warnings, and what each one means.

A bot file is written in a DSL that its writer, a person or a model, may never have seen. So every
mistake comes back as plain text that says *what* is wrong, *where* (the span of the writer's own
call, with its source line and a caret under it), *why*, and *exactly how to rewrite it*::

    error[B0003]: device id `desk` is used twice
      --> greeter.py:15:5
       |
    14 |     bot.attach(computer.Computer(id="desk")),
       |     ---------------------------------------- first used here
    15 |     bot.attach(phone.Phone(id="desk")),
       |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ used again here
       |
       = note: a bot keys its devices by `id`
    help: give the second device its own id
       |
    15 |     bot.attach(phone.Phone(id="phone")),
       |                               ~~~~~~~
       = explain: bot.explain("B0003")

The pieces are hsm's own: :class:`Location` is ``hsm.Location`` with columns, and the validator that
finds the mistakes is an ``hsm.ModelValidator``. A :class:`Diagnostic` is also typed data, so a reader
that prefers structure takes ``Diagnostic.model_dump_json()``.

**No values.** A diagnostic names ids, classes, elements and fields. The source lines it echoes keep
the diagnosed span as written, and every string and number literal outside a span is masked as
``"…"`` or ``…``, so a credential written on a nearby line or argument never prints. A cause raised by
the writer's code renders as ``<Type>: <message>``, and a ``pydantic.ValidationError`` as ``loc: msg``
without its input.

Every code and lint has an :class:`Explanation`: :func:`explain` returns it, like ``rustc --explain``,
and ``docs/bot_diagnostics.md`` is generated from the same table by :func:`document`::

    python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end="")' > docs/bot_diagnostics.md
"""

import ast
import collections.abc
import dataclasses
import difflib
import inspect
import io
import linecache
import os
import pathlib
import textwrap
import tokenize
import typing

import hsm
import pydantic

# Frames inside this package and inside hsm are the DSL's machinery, never the writer's call.
_SKIPPED_DIRECTORIES = (
    os.path.dirname(os.path.abspath(__file__)) + os.sep,
    os.path.dirname(os.path.abspath(hsm.__file__)) + os.sep,
)

Level = typing.Literal["error", "warning"]
"""How bad a diagnostic is: an ``error`` stops ``bot.define``; a ``warning`` is a lint that lets it finish."""

Applicability = typing.Literal["machine-applicable", "maybe-incorrect"]
"""Whether applying a suggestion as written is known to clear its diagnostic (rustc's applicability)."""


def _characters(line: str, offset: int) -> int:
    """The character column of UTF-8 byte ``offset`` in ``line``: frame positions count bytes."""

    return len(line.encode("utf-8")[:offset].decode("utf-8", errors="ignore"))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Location(hsm.Location):
    """Where a bot element was written: hsm's file and line, plus the columns of the call expression.

    ``col`` and ``end_col`` are 0-based UTF-8 byte offsets, ``end_col`` exclusive, on ``lineno`` and
    ``end_lineno``: Python's own positions, so capturing one reads no source. :meth:`Span.of` turns
    them into characters. A location with ``end_lineno == 0`` has no columns.
    """

    col: int = 0
    end_lineno: int = 0
    end_col: int = 0

    @classmethod
    @typing.override
    def capture(cls) -> "Location":
        """The call expression in the first frame outside ``mosfet.bot`` and ``hsm``: the writer's own call."""

        frame = inspect.currentframe()
        try:
            caller = frame
            while frame is not None:
                caller = frame
                if not os.path.abspath(frame.f_code.co_filename).startswith(_SKIPPED_DIRECTORIES):
                    break
                frame = frame.f_back
            if caller is None:
                return cls(filename="<unknown>", lineno=0)
            filename = caller.f_code.co_filename
            positions = inspect.getframeinfo(caller, context=0).positions
            lineno = caller.f_lineno
            if positions is None or None in (
                positions.lineno,
                positions.end_lineno,
                positions.col_offset,
                positions.end_col_offset,
            ):
                return cls(filename=filename, lineno=lineno)
            start = positions.lineno or lineno
            end = positions.end_lineno or start
            return cls(
                filename=filename,
                lineno=start,
                col=positions.col_offset or 0,
                end_lineno=end,
                end_col=positions.end_col_offset or 0,
            )
        finally:
            del frame


class Span(pydantic.BaseModel):
    """A stretch of a bot file's source, with what a diagnostic says about it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    filename: str = pydantic.Field(description="The source file, as Python names it.", examples=["greeter.py"])
    lineno: int = pydantic.Field(ge=0, description="The 1-based line the span starts on.", examples=[15])
    col: int = pydantic.Field(default=0, ge=0, description="The 0-based character column it starts at.", examples=[4])
    end_lineno: int = pydantic.Field(
        default=0,
        ge=0,
        description="The 1-based line it ends on; 0 when the span has no columns and covers its whole line.",
        examples=[15],
    )
    end_col: int = pydantic.Field(
        default=0, ge=0, description="The 0-based character column it ends before (exclusive).", examples=[38]
    )
    label: str = pydantic.Field(
        default="", description="What the diagnostic says about this span.", examples=["used again here"]
    )

    @classmethod
    def of(cls, location: hsm.Location, label: str = "") -> "Span":
        """The span of ``location``, labelled ``label``: its columns in characters when it has them.

        This reads the location's source line, so it belongs where a diagnostic is made, never on a
        path that defines a bot cleanly.
        """

        if isinstance(location, Location) and location.end_lineno >= location.lineno > 0:
            lines = _source(location.filename)
            first = lines[location.lineno - 1] if location.lineno <= len(lines) else ""
            last = lines[location.end_lineno - 1] if location.end_lineno <= len(lines) else ""
            return cls(
                filename=location.filename,
                lineno=location.lineno,
                col=_characters(first, location.col),
                end_lineno=location.end_lineno,
                end_col=_characters(last, location.end_col),
                label=label,
            )
        return cls(filename=location.filename, lineno=location.lineno, label=label)


class Suggestion(pydantic.BaseModel):
    """A rewrite: replace the source under ``span`` with ``replacement``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    span: Span = pydantic.Field(description="The source to replace; an empty span inserts at its start.")
    replacement: str = pydantic.Field(
        description="The text to put there, in the DSL's real names.", examples=['phone.Phone(id="phone")']
    )
    applicability: Applicability = pydantic.Field(
        description=(
            "`machine-applicable` when applying the replacement as written yields a bot that defines "
            "without this diagnostic; `maybe-incorrect` when it needs the writer's judgment."
        ),
        examples=["machine-applicable"],
    )


class Help(pydantic.BaseModel):
    """How to fix a diagnostic: what to do, and when it can be written down, the rewrite itself."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    message: str = pydantic.Field(description="What to do.", examples=["give the second device its own id"])
    suggestion: Suggestion | None = pydantic.Field(default=None, description="The rewrite, when there is one.")


class Diagnostic(pydantic.BaseModel):
    """One thing wrong with a bot file: what, where, why, and how to rewrite it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    code: str = pydantic.Field(
        pattern=r"^(B[0-9]{4}|[a-z][a-z0-9_]*)$",
        description=(
            "A stable code: an error is `B` and four digits (`B00xx` from `bot.define`, `B01xx` from "
            "`bot.start` and `bot.run`); a lint is its snake_case name. A code's meaning never changes, "
            'and `bot.explain("<code>")` says what it means.'
        ),
        examples=["B0003", "no_devices"],
    )
    level: Level = pydantic.Field(description="`error` or `warning`.", examples=["error"])
    message: str = pydantic.Field(
        description="What is wrong, in one line.", examples=["device id `desk` is used twice"]
    )
    primary: Span = pydantic.Field(description="Where it is wrong: the writer's own call.")
    secondary: tuple[Span, ...] = pydantic.Field(
        default=(), description="Other places that explain it, such as where an id was first used."
    )
    notes: tuple[str, ...] = pydantic.Field(
        default=(), description="Why it is wrong.", examples=[["a bot keys its devices by `id`"]]
    )
    help: Help | None = pydantic.Field(default=None, description="How to fix it.")


class Explanation(pydantic.BaseModel):
    """What one code or lint means, why the DSL has it, and a wrong bot file beside a corrected one."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    code: str = pydantic.Field(pattern=r"^(B[0-9]{4}|[a-z][a-z0-9_]*)$", examples=["B0003"])
    level: Level = pydantic.Field(examples=["error"])
    title: str = pydantic.Field(
        description="The mistake, in one line.", examples=["a device or ability id is used twice"]
    )
    meaning: str = pydantic.Field(description="What the code means.")
    why: str = pydantic.Field(description="Why the DSL requires it.")
    wrong: str = pydantic.Field(description="Bot file code that has the mistake.")
    right: str = pydantic.Field(description="The same code, corrected.")


EXPLANATIONS: tuple[Explanation, ...] = (
    Explanation(
        code="B0001",
        level="error",
        title="the bot's name is missing or not lower snake case",
        meaning=(
            "`bot.define` takes the bot's name first: lower snake case (a letter, then letters, digits and "
            "underscores), at most 64 characters."
        ),
        why=(
            "The name is the bot's topology name and a telemetry attribute, so it is bounded and spelled one "
            "way. The suggestion is the name in snake case."
        ),
        wrong='model = bot.define("Greeter Bot", bot.cognition(processor), bot.attach(phone.Phone()))',
        right='model = bot.define("greeter_bot", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0002",
        level="error",
        title="the bot has no `bot.cognition`",
        meaning="Every bot has exactly one `bot.cognition`, the mind that perceives and decides.",
        why=(
            "A body with no cognition can hold devices but can never choose anything to do with them. "
            "`bot.cognition(processor)` thinks with a `processing.Processor`."
        ),
        wrong='model = bot.define("greeter", bot.attach(phone.Phone()))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0003",
        level="error",
        title="a device or ability id is used twice",
        meaning=(
            "A bot keys its devices and the abilities it can address by `id`, so no two of them share one. A "
            "device or ability built without an `id` is known by its kind (`phone.Phone()` is `phone`), so "
            "two of one kind need ids of their own. The same instance given twice is also this error."
        ),
        why=(
            "The id is how the bot addresses a part: on its tool menu, as its focus, and in what it owns. Two "
            "parts with one id would be one name for two things."
        ),
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.attach(phone.Phone()),\n)"
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            '    bot.attach(phone.Phone(id="handset")),\n)'
        ),
    ),
    Explanation(
        code="B0004",
        level="error",
        title="an element is given a class instead of an instance",
        meaning=(
            "`bot.attach`, `bot.input`, `bot.output` and `bot.register` take a constructed instance: "
            '`bot.attach(phone.Phone(id="phone"))`, not `bot.attach(phone.Phone)`.'
        ),
        why=(
            "The bot file builds its parts and passes in what they need; the bot holds the very instances it "
            "is given, so one may also be handed to another part's constructor. Constructing a part does "
            "nothing until the bot starts."
        ),
        wrong='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="phone")))',
    ),
    Explanation(
        code="B0005",
        level="error",
        title="an element is given an instance that is already started",
        meaning="A part given to a bot is unstarted: the bot starts it, once, as part of its body.",
        why=(
            "A started instance already belongs to whatever started it. Construct a new one for this bot; "
            "construction has no side effects, so that costs nothing."
        ),
        wrong=(
            "handset = await mosfet.started(environment, phone.Phone(), phone.Phone.model)\n"
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(handset))'
        ),
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0006",
        level="error",
        title="an element is given something it does not take",
        meaning=(
            "`bot.attach` takes a device or an ability; `bot.input` and `bot.output` take an ability; "
            "`bot.register` takes a `memory.StmMemory`; `bot.cognition` takes a `processing.Processor` first, "
            "and `memory=` takes a `memory.Memory`. The help names the element that takes what was given."
        ),
        why=(
            "Each element is one part of the body: a device is held, a sense organ is an input, a processor is "
            "cognition's. Giving a part to the wrong element would wire it where it cannot work."
        ),
        wrong='model = bot.define("greeter", bot.cognition(processor), bot.input(phone.Phone()))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0007",
        level="error",
        title="a device or ability uses a reserved id",
        meaning="The ids `bot`, `cognition` and `firmware` are the body's own.",
        why="Cognition addresses the body, its cognition and its firmware by those keys, so no part may take one.",
        wrong='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="bot")))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="phone")))',
    ),
    Explanation(
        code="B0008",
        level="error",
        title="an element is written inside another element",
        meaning=(
            "Every part of a bot is an argument of `bot.define` itself. `bot.cognition` takes a processor and "
            "keyword flags, not `bot.attach` or another element."
        ),
        why="A bot is one flat list of parts; nesting would suggest an ownership the body does not have.",
        wrong='model = bot.define("greeter", bot.cognition(processor, bot.attach(phone.Phone())))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0009",
        level="error",
        title="an element a bot has once is given twice",
        meaning=(
            "A bot has at most one `bot.traits`, `bot.cognition`, `bot.timeouts`, `bot.register` and `bot.allow`."
        ),
        why="Each is one thing about the bot. Merge the two into one, for example one `bot.traits` with every trait.",
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.traits("curious"),\n    bot.traits("patient"),\n'
            "    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n)"
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.traits("curious", "patient"),\n'
            "    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n)"
        ),
    ),
    Explanation(
        code="B0010",
        level="error",
        title="the traits are not one to 24 short traits",
        meaning="`bot.traits` takes one to 24 traits, each a word or a short phrase of at most 200 characters.",
        why=(
            "Traits are the owner's fixed seed of who the bot is. The bot grows its own description on top of "
            "them, so they stay few and short."
        ),
        wrong='model = bot.define("greeter", bot.traits(), bot.cognition(processor), bot.attach(phone.Phone()))',
        right=(
            'model = bot.define("greeter", bot.traits("curious"), bot.cognition(processor), bot.attach(phone.Phone()))'
        ),
    ),
    Explanation(
        code="B0011",
        level="error",
        title="a timeout is not a positive `datetime.timedelta`",
        meaning="Each of `bot.timeouts(processing=, deactivation=, attachment=)` is a positive `datetime.timedelta`.",
        why="A timeout bounds how long the body waits for something; zero or less would never wait at all.",
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.timeouts(processing=datetime.timedelta(0)),\n)"
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.timeouts(processing=datetime.timedelta(seconds=30)),\n)"
        ),
    ),
    Explanation(
        code="B0015",
        level="error",
        title="an observer is given to `bot.define`",
        meaning="An observer (`hsm.observe(...)`) is not part of a bot's definition.",
        why=(
            "An observer watches a bot while it runs, for logs and records, and never acts on it. It is given "
            "where the bot is started, `bot.start(..., observe=(...))`."
        ),
        wrong=(
            "def log(ctx, instance, event):\n    print(event.name)\n\n\n"
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), hsm.observe(log))'
        ),
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0016",
        level="error",
        title="a held device has no id",
        meaning=(
            "Every device has an id, set by `mosfet.device.Device.__init__`. A device class whose `__init__` "
            "never calls its base's has none."
        ),
        why="The id is how the bot addresses the device; a device without one cannot be held.",
        wrong=(
            "class Lamp(mosfet.device.Device):\n    def __init__(self):\n        pass\n\n\n"
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(Lamp()))'
        ),
        right=(
            "class Lamp(mosfet.device.Device):\n    def __init__(self, *, id=None):\n"
            "        super().__init__(id=id)\n\n\n"
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(Lamp()))'
        ),
    ),
    Explanation(
        code="B0017",
        level="error",
        title="`bot.define` is given something that is not a bot element",
        meaning=(
            "`bot.define` takes the bot's name, then elements: `bot.attach`, `bot.input`, `bot.output`, "
            "`bot.register`, `bot.traits`, `bot.cognition` and `bot.timeouts`. A device, a processor or a "
            "tuple of elements given directly is this error, and the help says which element takes it."
        ),
        why="Each element says what a part is to the body; a bare part has no place in it until it is given to one.",
        wrong='model = bot.define("greeter", bot.cognition(processor), phone.Phone())',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="B0018",
        level="error",
        title="`bot.allow` names a lint that does not exist",
        meaning="Each name given to `bot.allow` is a lint's. The help names the nearest one.",
        why="Allowing a lint that does not exist allows nothing, so the warning it was meant for still shows.",
        wrong='model = bot.define("greeter", bot.cognition(processor), bot.allow("no_device"))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.allow("no_devices"))',
    ),
    Explanation(
        code="B0101",
        level="error",
        title="a model is started again while its parts run",
        meaning=(
            "`bot.start` builds one bot around the very parts a model holds. Starting a model again, or another "
            "model that shares one of its parts, while any of those parts runs is an error; the labels name the "
            "parts that are running."
        ),
        why=(
            "A part is one real thing: a phone is held by one body, and a memory remembers for one mind. Two bots "
            "cannot hold it at once, so each bot is defined from parts of its own, and a model runs one bot at a "
            "time: `bot.stop` stops its parts, and then it starts again."
        ),
        wrong=(
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))\n'
            "greeter = await bot.start(environment, model)\n"
            "again = await bot.start(environment, model)"
        ),
        right=(
            'greeter = await bot.start(environment, bot.define("greeter", bot.cognition(processor), '
            "bot.attach(phone.Phone())))\n"
            'helper = await bot.start(environment, bot.define("helper", bot.cognition(processor), '
            "bot.attach(phone.Phone())))"
        ),
    ),
    Explanation(
        code="B0102",
        level="error",
        title="a part raised while `bot.start` built the bot",
        meaning=(
            "`bot.start` builds the bot's cognition, its faculties and its persona from `bot.cognition` and "
            "`bot.traits`, then its body around every part. One of them raised: the diagnostic points at the "
            "element that declared it, and its note is the cause, as `<Type>: <message>`."
        ),
        why=(
            "A mistake is reported where it was written. A part that only fails once it is built is reported at "
            "the element that declared it, with its cause, so it is fixed there."
        ),
        wrong=(
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))\n'
            "greeter = await bot.start(environment, model)  # building its cognition raised"
        ),
        right=(
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))\n'
            "greeter = await bot.start(environment, model)  # once what the cause names is fixed"
        ),
    ),
    Explanation(
        code="B0103",
        level="error",
        title="`bot.start` is given something that is not an `Environment`",
        meaning="`bot.start(ctx, model)` starts the bot in `ctx`, an `Environment`: the world it perceives and acts in.",
        why=(
            "An environment is where a bot's stimuli come from and the scope its parts are addressed in; a plain "
            "`hsm.Context` is neither."
        ),
        wrong=(
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))\n'
            "greeter = await bot.start(hsm.Context(), model)"
        ),
        right=(
            'model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))\n'
            "greeter = await bot.start(environment, model)"
        ),
    ),
    Explanation(
        code="ability_without_tools",
        level="warning",
        title="an acquired ability offers cognition nothing to select",
        meaning=(
            "An ability given to `bot.attach` is acquired: cognition uses it through the events it offers. One "
            "that offers none can never be used. A sense is given to `bot.input`, an effector to `bot.output`, "
            "and the sensory register to `bot.register`."
        ),
        why="An acquired ability with no tools is almost always a sense or an effector given to the wrong element.",
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.attach(memory.StmMemory()),\n)"
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.register(memory.StmMemory()),\n)"
        ),
    ),
    Explanation(
        code="empty_timeouts",
        level="warning",
        title="`bot.timeouts()` sets no timeout",
        meaning="With no values, `bot.timeouts()` keeps every default, so it changes nothing.",
        why="An element that changes nothing reads as if it did something. Remove it, or set the timeout meant.",
        wrong=('model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.timeouts())'),
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="instruction_like_traits",
        level="warning",
        title="a trait reads as an instruction",
        meaning=(
            "A trait opens like an instruction (`always`, `never`, `you must`, `reply`, `respond`, `do not`), "
            "speaks to the bot as `you`, or runs past 12 words."
        ),
        why=(
            "Traits are who the bot is: the owner's fixed seed of its nature. How it acts is the bot's own to "
            "develop; an instruction in a trait would script a choice that is the bot's to make."
        ),
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.traits("always reply in French"),\n'
            "    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n)"
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.traits("curious", "speaks French"),\n'
            "    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n)"
        ),
    ),
    Explanation(
        code="no_devices",
        level="warning",
        title="the bot can perceive nothing",
        meaning=(
            "The bot has a cognition but no device and no input ability, so no stimulus ever reaches it and "
            "nothing can wake it."
        ),
        why=(
            "A bot that perceives nothing is allowed, and doing nothing is a choice it may make; but with nothing "
            "to perceive it was never given the choice. Attach what it should perceive, or allow the lint."
        ),
        wrong='model = bot.define("greeter", bot.cognition(processor))',
        right='model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))',
    ),
    Explanation(
        code="unused_register",
        level="warning",
        title="a sensory register on a bot that perceives nothing",
        meaning=(
            "`bot.register` records every stimulus the body admits, and only devices and input abilities bring "
            "stimuli. With neither, it records nothing."
        ),
        why="The register is perception machinery; without a sense it is machinery for nothing.",
        wrong=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.register(memory.StmMemory()),\n'
            '    bot.allow("no_devices"),\n)'
        ),
        right=(
            'model = bot.define(\n    "greeter",\n    bot.cognition(processor),\n    bot.attach(phone.Phone()),\n'
            "    bot.register(memory.StmMemory()),\n)"
        ),
    ),
)
"""Every code and lint, in code order. A retired code keeps its entry and is never reused.

The examples assume ``from mosfet import bot``, ``from mosfet.devices import phone``,
``from mosfet.abilities import memory``, ``import hsm``, ``import mosfet``, ``import datetime``, a
``processor`` (a ``processing.Processor``) and an ``environment`` (an ``Environment``); a test runs
every one of them.
"""


def explain(code: str) -> str:
    """What ``code`` means, why, and a wrong bot file beside a corrected one, like ``rustc --explain``.

    Raises:
        ValueError: If ``code`` is no diagnostic's code; the message names the nearest ones.
    """

    for explanation in EXPLANATIONS:
        if explanation.code == code:
            return "\n".join(
                (
                    f"{explanation.level}[{explanation.code}]: {explanation.title}",
                    "",
                    textwrap.fill(explanation.meaning, width=100),
                    "",
                    textwrap.fill(f"Why: {explanation.why}", width=100),
                    "",
                    "Wrong:",
                    "",
                    _indent(explanation.wrong),
                    "",
                    "Right:",
                    "",
                    _indent(explanation.right),
                )
            )
    nearest = difflib.get_close_matches(code, [explanation.code for explanation in EXPLANATIONS], n=3)
    hint = f"; did you mean {', '.join(f'`{near}`' for near in nearest)}?" if nearest else ""
    raise ValueError(f"unknown diagnostic code `{code}`{hint}")


def document() -> str:
    """``docs/bot_diagnostics.md``: the index and every explanation, generated from :data:`EXPLANATIONS`."""

    lines = [
        "# Bot diagnostics",
        "",
        "<!-- Generated from mosfet.bot.diagnostics.EXPLANATIONS. Do not edit by hand; regenerate with",
        "python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end=\"\")' > docs/bot_diagnostics.md -->",
        "",
        "`bot.define` reports every mistake in a bot file at once, the way rustc does: each with a stable",
        "code, the source span, notes on why, and a `help:` that shows the rewrite. Errors (`B00xx` from",
        "`bot.define`, `B01xx` from `bot.start` and `bot.run`) stop the bot; lints (snake_case names) are",
        'warnings. `bot.explain("<code>")` prints any entry below.',
        "",
        "The examples assume `from mosfet import bot`, `from mosfet.devices import phone`,",
        "`from mosfet.abilities import memory`, `import hsm`, `import mosfet`, `import datetime`, a `processor`",
        "(a `processing.Processor`) and an `environment`.",
        "",
    ]
    lines.extend(
        f"- [`{explanation.code}`](#{explanation.code.lower()}) ({explanation.level}): {explanation.title}"
        for explanation in EXPLANATIONS
    )
    for explanation in EXPLANATIONS:
        lines.extend(
            (
                "",
                f"## {explanation.code}",
                "",
                f"**{explanation.level}:** {explanation.title}",
                "",
                explanation.meaning,
                "",
                f"Why: {explanation.why}",
                "",
                "Wrong:",
                "",
                "```python",
                explanation.wrong,
                "```",
                "",
                "Right:",
                "",
                "```python",
                explanation.right,
                "```",
            )
        )
    return "\n".join(lines) + "\n"


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())


def cause(error: BaseException) -> str:
    """``error`` as ``<Type>: <message>``; a ``pydantic.ValidationError`` as ``loc: msg``, never its input."""

    if isinstance(error, pydantic.ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc']) or '<value>'}: {detail['msg']}"
            for detail in error.errors(include_url=False, include_context=False, include_input=False)
        )
        return f"ValidationError: {details}"
    return f"{type(error).__name__}: {error}"


def call(location: Span) -> ast.Call | None:
    """The call expression under ``location``, parsed, with character positions in the file's coordinates.

    ``None`` when the source is not available or the span is not a call (a span without columns).
    """

    if location.end_lineno < location.lineno or location.lineno < 1:
        return None
    lines = _source(location.filename)
    if len(lines) < location.end_lineno:
        return None
    segment = [line.rstrip("\r\n") for line in lines[location.lineno - 1 : location.end_lineno]]
    segment[-1] = segment[-1][: location.end_col]
    segment[0] = segment[0][location.col :]
    try:
        expression = ast.parse("(" + "\n".join(segment) + ")", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call):
        return None
    for node in ast.walk(expression):
        if not isinstance(node, ast.expr | ast.keyword):
            continue
        for attribute in ("lineno", "end_lineno"):
            line = typing.cast(int | None, getattr(node, attribute, None))
            if line is not None:
                setattr(node, attribute, line + location.lineno - 1)
        for attribute, line_attribute in (("col_offset", "lineno"), ("end_col_offset", "end_lineno")):
            column = typing.cast(int | None, getattr(node, attribute, None))
            line = typing.cast(int | None, getattr(node, line_attribute, None))
            if column is None or line is None:
                continue
            text = segment[line - location.lineno] if line - location.lineno < len(segment) else ""
            # ast columns count UTF-8 bytes; the first line also carries the "(" added above.
            prefix = "(" + text if line == location.lineno else text
            characters = _characters(prefix, column)
            setattr(node, attribute, characters - 1 + location.col if line == location.lineno else characters)
    return expression


def node_span(location: Span, node: ast.expr | ast.keyword, label: str = "") -> Span:
    """The span of ``node``, an expression from :func:`call` under ``location``, labelled ``label``."""

    return Span(
        filename=location.filename,
        lineno=node.lineno,
        col=node.col_offset,
        end_lineno=node.end_lineno or node.lineno,
        end_col=node.end_col_offset or node.col_offset,
        label=label,
    )


def argument(span: Span, key: int | str) -> tuple[Span, ast.expr] | None:
    """Positional argument ``key`` (or keyword argument ``key``) of the call under ``span``, and its span.

    ``None`` when the call does not parse, has no such argument, or unpacks ``*args`` before it.
    """

    node = call(span)
    if node is None:
        return None
    if isinstance(key, int):
        if key >= len(node.args) or any(isinstance(item, ast.Starred) for item in node.args[: key + 1]):
            return None
        expression = node.args[key]
    else:
        expression = next((keyword.value for keyword in node.keywords if keyword.arg == key), None)
        if expression is None:
            return None
    return node_span(span, expression), expression


def head(span: Span, label: str) -> Span:
    """The callee of the call under ``span`` (``bot.define``), labelled ``label``; the call when it does not parse."""

    node = call(span)
    if node is None:
        return span.model_copy(update={"label": label})
    return node_span(span, node.func, label)


def append(span: Span, element: str) -> Suggestion | None:
    """A ``maybe-incorrect`` insertion of ``element`` after the last positional argument of the call under ``span``."""

    node = call(span)
    if node is None or not node.args:
        return None
    end = node_span(span, node.args[-1])
    return Suggestion(
        span=end.model_copy(update={"lineno": end.end_lineno, "col": end.end_col}),
        replacement=f", {element}",
        applicability="maybe-incorrect",
    )


def removal(span: Span, argument: Span) -> Span | None:
    """The source to delete to remove ``argument`` from the call under ``span``, with its comma.

    An argument alone on its line takes the whole line; otherwise it takes the comma and space after
    it, or before it when it is the last. ``None`` when ``argument`` is not one of the call's arguments.
    """

    node = call(span)
    if node is None:
        return None
    arguments = [node_span(span, item) for item in [*node.args, *node.keywords]]
    arguments.sort(key=lambda item: (item.lineno, item.col))
    index = next(
        (
            position
            for position, item in enumerate(arguments)
            if (item.lineno, item.col, item.end_lineno, item.end_col)
            == (argument.lineno, argument.col, argument.end_lineno, argument.end_col)
        ),
        None,
    )
    if index is None:
        return None
    lines = _source(span.filename)
    line = lines[argument.lineno - 1].rstrip("\r\n")
    after = line[argument.end_col :].strip()
    if argument.lineno == argument.end_lineno and not line[: argument.col].strip() and after in ("", ","):
        return argument.model_copy(update={"col": 0, "end_lineno": argument.lineno + 1, "end_col": 0})
    if index + 1 < len(arguments):
        following = arguments[index + 1]
        return argument.model_copy(update={"end_lineno": following.lineno, "end_col": following.col})
    if index > 0:
        previous = arguments[index - 1]
        return argument.model_copy(update={"lineno": previous.end_lineno, "col": previous.end_col})
    return argument


def text(span: Span) -> str:
    """The source under ``span`` as written (empty when the source is not available)."""

    lines = _source(span.filename)
    if span.end_lineno < span.lineno or len(lines) < span.end_lineno or span.lineno < 1:
        return ""
    chunk = [line.rstrip("\r\n") for line in lines[span.lineno - 1 : span.end_lineno]]
    chunk[-1] = chunk[-1][: span.end_col]
    chunk[0] = chunk[0][span.col :]
    return "\n".join(chunk)


def _source(filename: str) -> list[str]:
    linecache.checkcache(filename)
    return linecache.getlines(filename)


# A literal's extent in its file: (start line, start column, end line, end column), columns in characters.
_Literal = tuple[int, int, int, int]


def _literals(filename: str) -> tuple[_Literal, ...] | None:
    """Every string and number literal in ``filename``; ``None`` when it does not tokenize."""

    literals: list[_Literal] = []
    fstring: tuple[int, int] | None = None
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO("".join(_source(filename))).readline):
            if token.type == tokenize.FSTRING_START:
                if depth == 0:
                    fstring = token.start
                depth += 1
            elif token.type == tokenize.FSTRING_END:
                depth -= 1
                if depth == 0 and fstring is not None:
                    literals.append((*fstring, *token.end))
                    fstring = None
            elif depth == 0 and token.type in (tokenize.STRING, tokenize.NUMBER):
                literals.append((*token.start, *token.end))
    except (tokenize.TokenError, SyntaxError):
        return None
    return tuple(literals)


def _mask(
    line: str, lineno: int, literals: tuple[_Literal, ...] | None, revealed: list[tuple[int, int]]
) -> tuple[str, typing.Callable[[int], int]]:
    """``line`` with every literal outside ``revealed`` masked, and the map from old columns to new.

    Without ``literals`` (the file does not tokenize) everything outside ``revealed`` is masked.
    """

    pieces: list[tuple[int, int, str]] = []
    if literals is None:
        cursor = len(line) - len(line.lstrip())
        for low, high in [*sorted(revealed), (len(line), len(line))]:
            if line[cursor:low].strip():
                pieces.append((cursor, low, "…"))
            cursor = max(cursor, high)
    for start_line, start_col, end_line, end_col in literals or ():
        if not start_line <= lineno <= end_line:
            continue
        start = start_col if start_line == lineno else 0
        end = end_col if end_line == lineno else len(line)
        if any(low <= start and end <= high for low, high in revealed):
            continue
        is_string = line[start:end][-1:] in ("'", '"') or start_line != end_line
        pieces.append((start, end, '"…"' if is_string and start_line == lineno else "…"))
    pieces.sort()
    masked: list[str] = []
    cursor = 0
    for start, end, replacement in pieces:
        masked.append(line[cursor:start])
        masked.append(replacement)
        cursor = end
    masked.append(line[cursor:])

    def column(old: int) -> int:
        shift = 0
        for start, end, replacement in pieces:
            if end <= old:
                shift += len(replacement) - (end - start)
        return old + shift

    return "".join(masked), column


def _display(filename: str, relative_to: pathlib.Path | None) -> str:
    root = pathlib.Path.cwd() if relative_to is None else relative_to
    path = pathlib.Path(filename)
    if not path.is_absolute():
        return filename
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return filename


def _extent(span: Span, line: str) -> tuple[int, int]:
    """The columns ``span`` covers on its first line: to its end, or to the end of the line it runs past."""

    if span.end_lineno == 0:
        stripped = line.rstrip()
        return len(stripped) - len(stripped.lstrip()), len(stripped)
    end = span.end_col if span.end_lineno == span.lineno else len(line.rstrip())
    return span.col, max(end, span.col + 1)


def render(diagnostic: Diagnostic, *, relative_to: pathlib.Path | None = None) -> str:
    """``diagnostic`` as rustc prints one: header, ``-->``, source with carets, notes, help, explain.

    File names under ``relative_to`` (the current directory when ``None``) print relative to it.
    """

    marked = [(diagnostic.primary, "^"), *((span, "-") for span in diagnostic.secondary)]
    files = [diagnostic.primary.filename]
    files.extend(span.filename for span, _ in marked if span.filename not in files)
    suggestion = diagnostic.help.suggestion if diagnostic.help is not None else None
    numbers = [span.lineno for span, _ in marked]
    if suggestion is not None:
        numbers.append(suggestion.span.lineno)
    width = len(str(max(numbers)))
    gutter = " " * width
    out = [f"{diagnostic.level}[{diagnostic.code}]: {diagnostic.message}"]
    for filename in files:
        spans = sorted((item for item in marked if item[0].filename == filename), key=lambda item: item[0].lineno)
        first = spans[0][0] if filename != diagnostic.primary.filename else diagnostic.primary
        arrow = "-->" if filename == diagnostic.primary.filename else ":::"
        out.append(f"{gutter}{arrow} {_display(filename, relative_to)}:{first.lineno}:{first.col + 1}")
        lines = _source(filename)
        literals = _literals(filename)
        shown = sorted({span.lineno for span, _ in spans if 0 < span.lineno <= len(lines)})
        if not shown:
            continue
        out.append(f"{gutter} |")
        previous = 0
        for lineno in shown:
            if previous and lineno > previous + 1:
                out.append("...")
            previous = lineno
            line = lines[lineno - 1].rstrip("\r\n")
            here = [(span, mark) for span, mark in spans if span.lineno == lineno]
            extents = [_extent(span, line) for span, _ in here]
            masked, column = _mask(line, lineno, literals, extents)
            out.append(f"{lineno:>{width}} | {masked}".rstrip())
            for (span, mark), (start, end) in sorted(zip(here, extents, strict=True), key=lambda item: item[1]):
                new_start, new_end = column(start), column(end)
                out.append(f"{gutter} | {' ' * new_start}{mark * (new_end - new_start)} {span.label}".rstrip())
    out.append(f"{gutter} |")
    out.extend(f"{gutter} = note: {note}" for note in diagnostic.notes)
    if diagnostic.help is not None:
        out.append(f"help: {diagnostic.help.message}")
        if suggestion is not None:
            out.extend(_suggested(suggestion, diagnostic.primary, width))
    out.append(f'{gutter} = explain: bot.explain("{diagnostic.code}")')
    return "\n".join(out)


def _suggested(suggestion: Suggestion, primary: Span, width: int) -> list[str]:
    """The suggestion's line with the replacement applied and ``~`` under it, literals masked as elsewhere."""

    span = suggestion.span
    lines = _source(span.filename)
    if not 0 < span.lineno <= len(lines):
        return []
    gutter = " " * width
    if not suggestion.replacement:
        # A deletion shows the line as written, with ``-`` under what goes (as rustc does).
        line = lines[span.lineno - 1].rstrip("\r\n")
        whole = span.end_lineno > span.lineno and span.col == 0
        start = len(line) - len(line.lstrip()) if whole else span.col
        end = span.end_col if span.end_lineno == span.lineno else len(line)
        masked, column = _mask(line, span.lineno, _literals(span.filename), [(start, end)])
        return [
            f"{gutter} |",
            f"{span.lineno:>{width}} | {masked}".rstrip(),
            f"{gutter} | {' ' * column(start)}{'-' * max(column(end) - column(start), 1)}",
        ]
    if span.end_lineno != span.lineno or "\n" in suggestion.replacement:
        return []
    line = lines[span.lineno - 1].rstrip("\r\n")
    start, end = span.col, max(span.end_col, span.col)
    delta = len(suggestion.replacement) - (end - start)
    edited = line[:start] + suggestion.replacement + line[end:]

    def moved(column: int) -> int:
        return column if column <= start else (start if column < end else column + delta)

    found = _literals(span.filename)
    literals = (
        None
        if found is None
        else tuple(
            (lineno, moved(col), end_line, moved(end_col) if end_line == span.lineno else end_col)
            for lineno, col, end_line, end_col in found
            if not (lineno == end_line == span.lineno and start <= col and end_col <= end)
        )
    )
    revealed = [(start, start + len(suggestion.replacement))]
    if primary.filename == span.filename and primary.lineno == span.lineno:
        low, high = _extent(primary, line)
        revealed.append((moved(low), high + delta if high >= end else moved(high)))
    masked, column = _mask(edited, span.lineno, literals, revealed)
    marker_start, marker_end = column(start), column(start + len(suggestion.replacement))
    return [
        f"{gutter} |",
        f"{span.lineno:>{width}} | {masked}".rstrip(),
        f"{gutter} | {' ' * marker_start}{'~' * (marker_end - marker_start)}".rstrip(),
    ]


def ordered(found: collections.abc.Iterable[Diagnostic], home: str) -> list[Diagnostic]:
    """``found`` in source order, the order rustc reports in: ``home``'s file first, then by file, line, column."""

    return sorted(
        found,
        key=lambda item: (item.primary.filename != home, item.primary.filename, item.primary.lineno, item.primary.col),
    )


def summary(diagnostics: collections.abc.Iterable[Diagnostic], *, action: str, name: str) -> str:
    """The closing line: ``error: could not <action> bot `<name>`: 2 errors, 1 warning``."""

    levels = [diagnostic.level for diagnostic in diagnostics]
    errors, warnings = levels.count("error"), levels.count("warning")
    counts = [
        f"{count} {noun}{'' if count == 1 else 's'}"
        for count, noun in ((errors, "error"), (warnings, "warning"))
        if count
    ]
    if errors:
        return f"error: could not {action} bot `{name}`: {', '.join(counts)}"
    return f"warning: bot `{name}` has {', '.join(counts) or 'no warnings'}"


def report(
    diagnostics: collections.abc.Sequence[Diagnostic],
    *,
    action: str,
    name: str,
    relative_to: pathlib.Path | None = None,
) -> str:
    """Every diagnostic rendered, in order, then the :func:`summary` line."""

    rendered = [render(diagnostic, relative_to=relative_to) for diagnostic in diagnostics]
    return "\n\n".join([*rendered, summary(diagnostics, action=action, name=name)])


__all__ = [
    "EXPLANATIONS",
    "Applicability",
    "Diagnostic",
    "Explanation",
    "Help",
    "Level",
    "Location",
    "Span",
    "Suggestion",
    "append",
    "argument",
    "call",
    "cause",
    "document",
    "explain",
    "head",
    "node_span",
    "ordered",
    "removal",
    "render",
    "report",
    "summary",
    "text",
]
