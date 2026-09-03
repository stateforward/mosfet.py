"""Bot behavior primitives for stateforward.bot.

An executable behavior is an HSM program under cognition. Trusted seeded programs may
be native Python HSMs; learned programs remain Starlark source in the behavior inventory.
Autonomy runs both through the same candidate lifecycle before deliberation.

Inventory events (first-class package exports):

- ``CreateData`` / ``CreateEvent`` — ``bot.behavior.create``
- ``ChangeData`` / ``ChangeEvent`` — ``bot.behavior.change``
- ``BreakData`` / ``BreakEvent`` — ``bot.behavior.break``

Heavy modules (behavior, runtime, compiler) load lazily to avoid import cycles
with abilities/cognition.
"""

from __future__ import annotations

import importlib
import typing

from .events import (
    BreakData,
    BreakEvent,
    ChangeData,
    ChangeEvent,
    CreateData,
    CreateEvent,
    data_from_event,
    event_for_data,
    is_inventory_event,
)

if typing.TYPE_CHECKING:
    from .behavior import Behavior, define_model
    from .compiler import build
    from .diagnostic import Checked, Diagnostic, Level, Report, Stage
    from .instance import Instance, check, start
    from .seed import Seed
    from .schema import JsonSchema
    from .source import (
        Source,
        EventContract,
        SourceError,
        STARLARK_API,
        parse_source,
    )

_LAZY = {
    "Instance": ".instance",
    "start": ".instance",
    "check": ".instance",
    "verify_apply": ".verify",
    "Behavior": ".behavior",
    "define_model": ".behavior",
    "build": ".compiler",
    "Checked": ".diagnostic",
    "Diagnostic": ".diagnostic",
    "Level": ".diagnostic",
    "Report": ".diagnostic",
    "Stage": ".diagnostic",
    "JsonSchema": ".schema",
    "Source": ".source",
    "EventContract": ".source",
    "SourceError": ".source",
    "STARLARK_API": ".source",
    "parse_source": ".source",
    "behavior": ".behavior",
    "compiler": ".compiler",
    "diagnostic": ".diagnostic",
    "instance": ".instance",
    "Seed": ".seed",
    "seed": ".seed",
    "runtime": ".runtime",
    "schema": ".schema",
    "storage": ".storage",
    "source": ".source",
    "behavior_table": ".storage",
    "behavior_trigger_table": ".storage",
}

__all__ = [
    "BreakData",
    "BreakEvent",
    "ChangeData",
    "ChangeEvent",
    "CreateData",
    "CreateEvent",
    "Instance",
    "Seed",
    "Behavior",
    "Source",
    "EventContract",
    "SourceError",
    "JsonSchema",
    "STARLARK_API",
    "Checked",
    "Diagnostic",
    "Level",
    "Report",
    "Stage",
    "build",
    "check",
    "data_from_event",
    "define_model",
    "event_for_data",
    "start",
    "verify_apply",
    "is_inventory_event",
    "parse_source",
    "behavior",
    "compiler",
    "diagnostic",
    "instance",
    "seed",
    "runtime",
    "schema",
    "storage",
    "behavior_table",
    "behavior_trigger_table",
    "source",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    if name in {
        "behavior",
        "compiler",
        "diagnostic",
        "instance",
        "runtime",
        "schema",
        "source",
        "seed",
    }:
        value: object = module
    else:
        value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
