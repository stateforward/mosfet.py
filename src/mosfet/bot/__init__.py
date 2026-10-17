"""The bot package: the body every bot runs on, and the names a bot file is written with.

The body is ``mosfet.Bot``, defined in ``mosfet.bot.body``. This package's own namespace is the bot
definition DSL (``from mosfet import bot``), so it re-exports nothing from the body::

    model = bot.define(
        "greeter",
        bot.traits("curious"),
        bot.cognition(processor),
        bot.attach(phone.Phone()),
    )

A bot file's mistakes come back as rustc-style :class:`Diagnostic`\\ s (``mosfet.bot.diagnostics``),
all of them in one :class:`DefinitionError`, and :func:`explain` says what any code means.
"""

from .define import DefinitionError, Error, Finalizer, RedefinableBot, Validator, define, redefine
from .diagnostics import Diagnostic, Help, Location, Span, Suggestion, explain
from .elements import Element, Model, attach, cognition, input, output, register, timeouts, traits

__all__ = [
    "DefinitionError",
    "Diagnostic",
    "Element",
    "Error",
    "Finalizer",
    "Help",
    "Location",
    "Model",
    "RedefinableBot",
    "Span",
    "Suggestion",
    "Validator",
    "attach",
    "cognition",
    "define",
    "explain",
    "input",
    "output",
    "redefine",
    "register",
    "timeouts",
    "traits",
]
