"""The bot package: the body every bot runs on, and the names a bot file is written with.

The body is ``mosfet.Bot``, defined in ``mosfet.bot.body``. This package's own namespace is the bot
definition DSL (``from mosfet import bot``), so it re-exports nothing from the body::

    model = bot.define(
        "greeter",
        bot.traits("curious"),
        bot.cognition(processor),
        bot.attach(phone.Phone()),
    )

:func:`describe` is the JSON view of defined bots. :func:`start` builds the body around the model's
parts and brings it to life in an environment, and :func:`stop` deactivates and stops it::

    greeter = await bot.start(Environment(), model)
    await bot.stop(greeter)

A bot file runs its bots as a process from its ``__main__`` with :func:`run`, reading the process's
options once with :func:`process`::

    if __name__ == "__main__":
        bot.run(model, process=bot.process())

A bot file's mistakes come back as rustc-style :class:`Diagnostic`\\ s (``mosfet.bot.diagnostics``),
all of them in one :class:`DefinitionError`; clippy-style lints are :class:`LintWarning`\\ s that
:func:`allow` allows per bot and :func:`check` returns; :func:`start`'s are :class:`StartError`\\ s; and
:func:`explain` says what any code means. ``mosfet.bot.testing`` is what a bot's tests need: a scripted
model, a bot running for an ``async with`` block, and its snapshot.
"""

from .define import DefinitionError, Error, Finalizer, RedefinableBot, Validator, define, redefine
from .describe import Description, Outline, Part, describe
from .diagnostics import Diagnostic, Help, Location, Span, Suggestion, explain
from .elements import Element, Model, allow, attach, cognition, input, output, register, timeouts, traits
from .lints import LintWarning, check
from .process import Process, process
from .run import run
from .start import StartError, start, stop

__all__ = [
    "DefinitionError",
    "Description",
    "Diagnostic",
    "Element",
    "Error",
    "Finalizer",
    "Help",
    "LintWarning",
    "Location",
    "Model",
    "Outline",
    "Part",
    "Process",
    "RedefinableBot",
    "Span",
    "StartError",
    "Suggestion",
    "Validator",
    "allow",
    "attach",
    "check",
    "cognition",
    "define",
    "describe",
    "explain",
    "input",
    "output",
    "process",
    "redefine",
    "register",
    "run",
    "start",
    "stop",
    "timeouts",
    "traits",
]
