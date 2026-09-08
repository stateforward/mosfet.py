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
their domain package; body control events are re-exported directly here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Body control events first so ability packages can `import bot` without cycles.
from .events import (
    ActivateEvent,
    ActivateEventData,
    ActivatingDoneEvent,
    ActivatingDoneEventData,
    ActivatingFailedEvent,
    ActivatingFailedEventData,
    InputData,
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
    RebootEvent,
    RebootEventData,
    RebootReason,
    StimulusData,
)

# Construction API before abilities so class-body `bot.define` sees the hook.
from bot.define import (
    activity,
    after,
    choice,
    defer,
    define,
    effect,
    entry,
    exit,
    final,
    guard,
    initial,
    observe,
    on,
    source,
    state,
    target,
    transition,
)
from bot.start import register, start, started
from bot import abilities, behavior, skills, address

if TYPE_CHECKING:
    from bot.bot import Bot


def __getattr__(name: str) -> object:
    if name == "Bot":
        from bot.bot import Bot

        return Bot
    raise AttributeError(name)


__all__ = [
    "Bot",
    "activity",
    "after",
    "choice",
    "defer",
    "define",
    "effect",
    "entry",
    "exit",
    "final",
    "guard",
    "initial",
    "observe",
    "on",
    "source",
    "start",
    "started",
    "register",
    "address",
    "state",
    "target",
    "transition",
    "ActivateEvent",
    "ActivateEventData",
    "ActivatingDoneEvent",
    "ActivatingDoneEventData",
    "ActivatingFailedEvent",
    "ActivatingFailedEventData",
    "InputData",
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
    "RebootEvent",
    "RebootEventData",
    "RebootReason",
    "StimulusData",
    "abilities",
    "behavior",
    "skills",
    "__version__",
]
