"""The bot package: the body every bot runs on, and the names a bot file is written with.

The body is ``mosfet.Bot``, defined in ``mosfet.bot.body``. This package's own namespace is kept for
the bot-definition names (``from mosfet import bot``), so it re-exports nothing from the body.

A bot file's mistakes come back as rustc-style :class:`Diagnostic`\\ s (``mosfet.bot.diagnostics``),
and :func:`explain` says what any code means.
"""

from .diagnostics import Diagnostic, Help, Location, Span, Suggestion, explain

__all__ = ["Diagnostic", "Help", "Location", "Span", "Suggestion", "explain"]
