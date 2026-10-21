"""Cognitive system: host ability plus autonomy, intuition, reasoning, reflection, and routines.

Same-named module ``cognition.cognition`` is re-exported at package level.
Leaf deliberative processing lives in ``mosfet.abilities.processing``.
"""

from __future__ import annotations

import importlib
import typing

from . import cognition
from . import autonomy, event, input, intuition, reasoning, reflection, routines, types

if typing.TYPE_CHECKING:
    from .autonomy import Autonomy
    from .cognition import (
        CancelData,
        CancelEvent,
        CancelledData,
        CancelledEvent,
        Cognition,
        InputData,
        InputEvent,
        OutputData,
        OutputEvent,
        StepFailedData,
        StepFailedEvent,
        StepTimedOutData,
        StepTimedOutEvent,
    )
    from .input import SelfData, is_input, stimulus_name
    from .intuition import Intuition
    from .reasoning import Reasoning
    from .reflection import Reflection
    from .routines import Routines
    from .types import EventData, IgnoreData, IgnoreEvent, RecalledData, is_ignore_event, is_output

_LAZY_EXPORT_MODULES = {
    "InputEvent": ".cognition",
    "OutputEvent": ".cognition",
    "CancelData": ".cognition",
    "CancelEvent": ".cognition",
    "CancelledData": ".cognition",
    "CancelledEvent": ".cognition",
    "Cognition": ".cognition",
    "InputData": ".cognition",
    "OutputData": ".cognition",
    "StepFailedData": ".cognition",
    "StepFailedEvent": ".cognition",
    "StepTimedOutData": ".cognition",
    "StepTimedOutEvent": ".cognition",
    "SelfData": ".input",
    "is_input": ".input",
    "stimulus_name": ".input",
    "Autonomy": ".autonomy",
    "Intuition": ".intuition",
    "Reasoning": ".reasoning",
    "Reflection": ".reflection",
    "Routines": ".routines",
    "EventData": ".types",
    "IgnoreData": ".types",
    "IgnoreEvent": ".types",
    "RecalledData": ".types",
    "is_ignore_event": ".types",
    "is_output": ".types",
}

__all__ = [
    "InputEvent",
    "OutputEvent",
    "CancelData",
    "CancelEvent",
    "CancelledData",
    "CancelledEvent",
    "Cognition",
    "InputData",
    "OutputData",
    "StepFailedData",
    "StepFailedEvent",
    "StepTimedOutData",
    "StepTimedOutEvent",
    "SelfData",
    "is_input",
    "stimulus_name",
    "Autonomy",
    "Intuition",
    "Reasoning",
    "Reflection",
    "Routines",
    "EventData",
    "IgnoreData",
    "IgnoreEvent",
    "RecalledData",
    "is_ignore_event",
    "is_output",
    "autonomy",
    "cognition",
    "event",
    "input",
    "intuition",
    "reasoning",
    "reflection",
    "routines",
    "types",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = typing.cast(object, getattr(module, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
