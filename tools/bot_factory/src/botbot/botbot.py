"""botbot: the bot that builds the bots.

The CLI entry runs the Textual chat TUI by default. TODO: the effort-gated
cognition host wiring (typesafe intuition + luna reasoning) composes here when
the text-path abilities declare effort on their contract.
"""

from __future__ import annotations

import asyncio
import typing

_MAX_SESSIONS = 4


def main() -> int:
    """Run botbot TUI by default."""
    from .chat_tui import BotBotApp

    BotBotApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
