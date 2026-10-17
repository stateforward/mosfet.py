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

The hsm element functions are re-exported as the very same objects (``bot.state is hsm.state``):
given at the top level of :func:`define`, mixed with the ``bot.*`` elements, they are the bot's one
firmware (``mosfet.bot.firmware``), the machinery its owner wrote for it::

    model = bot.define(
        "timer",
        bot.cognition(processor),
        bot.initial(bot.target("idle")),
        bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../running"))),
        bot.state("running", bot.transition(bot.after(duration), bot.target("../idle"))),
    )

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
from .firmware import (
    activity,
    after,
    at,
    attribute,
    choice,
    deep_history,
    defer,
    effect,
    entry,
    entry_point,
    every,
    exit,
    exit_point,
    final,
    guard,
    initial,
    on,
    on_call,
    on_set,
    operation,
    shallow_history,
    source,
    state,
    submachine_state,
    target,
    transition,
    when,
)
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
    "activity",
    "after",
    "allow",
    "at",
    "attach",
    "attribute",
    "check",
    "choice",
    "cognition",
    "deep_history",
    "defer",
    "define",
    "describe",
    "effect",
    "entry",
    "entry_point",
    "every",
    "exit",
    "exit_point",
    "explain",
    "final",
    "guard",
    "initial",
    "input",
    "on",
    "on_call",
    "on_set",
    "operation",
    "output",
    "process",
    "redefine",
    "register",
    "run",
    "shallow_history",
    "source",
    "start",
    "state",
    "stop",
    "submachine_state",
    "target",
    "timeouts",
    "traits",
    "transition",
    "when",
]
