"""stateforward.bot — software-robot runtime package.

Import body primitives from the package root and abilities under ``bot.abilities``:

    import bot
    from bot.abilities import cognition
    from bot.devices import phone

    bot.Bot
    bot.InputEvent            # body control (bot.input)
    cognition.InputEvent      # ability (bot.ability.cognition.input)
    phone.RingingEvent

Do not import a bare ``events`` package as a domain namespace. Ability/device events live on
their domain package; body control events are re-exported here from ``bot.events``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Body control events first so ability packages can `import bot` without cycles.
from bot.events import (
    ActivateEvent,
    ActivateEventData,
    ActivatingDoneEvent,
    ActivatingDoneEventData,
    ActivatingFailedEvent,
    ActivatingFailedEventData,
    BotInputData,
    ClearFocusEvent,
    ClearFocusEventData,
    DeactivateEvent,
    DeactivateEventData,
    DeactivatingDoneEvent,
    DeactivatingDoneEventData,
    FocusDeviceEvent,
    FocusDeviceEventData,
    InputEvent,
    InputEventData,
    ProcessingCompletedEvent,
    ProcessingCompletedEventData,
    ProcessingFailedEvent,
    ProcessingFailedEventData,
)
from bot import abilities, habit, skills

if TYPE_CHECKING:
    from bot.bot import Bot


def __getattr__(name: str) -> object:
    if name == "Bot":
        from bot.bot import Bot

        return Bot
    raise AttributeError(name)


__all__ = [
    "Bot",
    "ActivateEvent",
    "ActivateEventData",
    "ActivatingDoneEvent",
    "ActivatingDoneEventData",
    "ActivatingFailedEvent",
    "ActivatingFailedEventData",
    "BotInputData",
    "ClearFocusEvent",
    "ClearFocusEventData",
    "DeactivateEvent",
    "DeactivateEventData",
    "DeactivatingDoneEvent",
    "DeactivatingDoneEventData",
    "FocusDeviceEvent",
    "FocusDeviceEventData",
    "InputEvent",
    "InputEventData",
    "ProcessingCompletedEvent",
    "ProcessingCompletedEventData",
    "ProcessingFailedEvent",
    "ProcessingFailedEventData",
    "abilities",
    "habit",
    "skills",
    "__version__",
]
