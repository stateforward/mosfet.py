"""The smallest bot: no devices, and a cognition with memory that recalls and consolidates.

    python -m minimal_bot_example

With no devices nothing reaches the bot, so it activates, stays ready, and stops cleanly; add a device or
an acquired ability to give it something to perceive. What it does with what it perceives is its
cognition's to decide.
"""

from mosfet import bot
from mosfet.abilities import memory, processing


def minimal(*, processor: processing.Processor, store: memory.Memory) -> bot.Model:
    """The minimal bot, thinking with ``processor`` and remembering in ``store``.

    It has nothing to perceive yet, which ``bot.allow("no_devices")`` says is deliberate.
    """

    return bot.define(
        "minimal",
        bot.cognition(processor, memory=store, associative=True, consolidation=True),
        bot.allow("no_devices"),
    )


__all__ = ["minimal"]
