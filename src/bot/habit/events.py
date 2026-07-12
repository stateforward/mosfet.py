"""Habit inventory HSM events: create, change, break.

First-class habit-package contracts. Reflection applies these (build Instance,
validate Starlark source, store); reasoning may record them with episodes.
The event is the inventory operation; create/change install executable behavior.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

CREATE_EVENT_NAME = "bot.habit.create"
CHANGE_EVENT_NAME = "bot.habit.change"
BREAK_EVENT_NAME = "bot.habit.break"


class CreateData(pydantic.BaseModel):
    """Payload for ``bot.habit.create``: install a new executable habit."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Create a new executable habit. Select may omit source (intent only); the create write "
                "step must author full Starlark HSM source from the habit Starlark API and the observed "
                "pattern — not by copying canned examples."
            ),
            "examples": [
                {
                    "event": CREATE_EVENT_NAME,
                    "name": "AnswerIncomingRing",
                    "triggers": ["world.sound"],
                    "description": "Answer when a labeled ring sound arrives while phone is a focus candidate.",
                    "reason": "Same ring→answer pattern across recent episodes improves call handling.",
                }
            ],
        },
    )

    event: typing.Literal["bot.habit.create"] = pydantic.Field(
        default="bot.habit.create",
        description="Habit create event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable PascalCase habit / HSM model name to install.")
    triggers: tuple[str, ...] = pydantic.Field(default=(), description="Optional trigger names.")
    description: str | None = pydantic.Field(default=None, min_length=1)
    reason: str | None = pydantic.Field(default=None, min_length=1)
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Starlark HSM habit source authored for this install. Required on the create write step. "
            "Must declare input_event, output_event, and habit = hsm.define(Name, ...) (or habit_behavior(...)) "
            "with string-named callbacks. Invent names, schemas, guards, and effects from the observed "
            "pattern using the Starlark habit API. Select intent may omit this field."
        ),
    )


class ChangeData(pydantic.BaseModel):
    """Payload for ``bot.habit.change``: revise an existing executable habit."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Change an existing executable habit. Select may omit source (intent only); the change "
                "write step must author full revised Starlark HSM source from the habit Starlark API, "
                "the existing habit, and the observed pattern."
            ),
            "examples": [
                {
                    "event": CHANGE_EVENT_NAME,
                    "name": "AnswerIncomingRing",
                    "reason": "Also clear browser focus before answering.",
                }
            ],
        },
    )

    event: typing.Literal["bot.habit.change"] = pydantic.Field(
        default="bot.habit.change",
        description="Habit change event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable name of the habit to revise.")
    triggers: tuple[str, ...] = pydantic.Field(default=(), description="Optional replacement triggers.")
    description: str | None = pydantic.Field(default=None, min_length=1)
    reason: str | None = pydantic.Field(default=None, min_length=1)
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Revised Starlark HSM habit source. Required on the change write step. Must remain a valid "
            "executable habit declaration whose model name matches this habit name. Author revisions from "
            "the Starlark habit API and existing_habit.source — do not paste unrelated canned source."
        ),
    )


class BreakData(pydantic.BaseModel):
    """Payload for ``bot.habit.break``: set an existing habit status=BROKEN (do not delete)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Break a habit that no longer improves (or harms) robot behavior. "
                "Sets inventory status=BROKEN so Autonomy will not run it; the row remains "
                "for a later bot.habit.change repair. Optional reason becomes status_reason."
            ),
            "examples": [
                {
                    "event": BREAK_EVENT_NAME,
                    "name": "AnswerIncomingRing",
                    "reason": "Answered while the user was already on another call.",
                }
            ],
        },
    )

    event: typing.Literal["bot.habit.break"] = pydantic.Field(
        default="bot.habit.break",
        description="Habit break event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable name of the habit to set status=BROKEN.")
    reason: str | None = pydantic.Field(default=None, min_length=1)


Data: typing.TypeAlias = typing.Annotated[
    CreateData | ChangeData | BreakData,
    pydantic.Field(discriminator="event"),
]

CreateEvent = hsm.Event[CreateData](
    name=CREATE_EVENT_NAME,
    kind=hsm.CallEventKind,
    schema=CreateData,
)
ChangeEvent = hsm.Event[ChangeData](
    name=CHANGE_EVENT_NAME,
    kind=hsm.CallEventKind,
    schema=ChangeData,
)
BreakEvent = hsm.Event[BreakData](
    name=BREAK_EVENT_NAME,
    kind=hsm.CallEventKind,
    schema=BreakData,
)

Event: typing.TypeAlias = hsm.Event[CreateData] | hsm.Event[ChangeData] | hsm.Event[BreakData]


def event_for_data(data: CreateData | ChangeData | BreakData) -> Event:
    """Build the HSM event for a habit create/change/break payload."""

    if isinstance(data, CreateData):
        return CreateEvent.with_data(data)
    if isinstance(data, ChangeData):
        return ChangeEvent.with_data(data)
    if isinstance(data, BreakData):
        return BreakEvent.with_data(data)
    raise TypeError(f"Unsupported habit event payload: {type(data)!r}")


def data_from_event(event: hsm.Event[typing.Any]) -> CreateData | ChangeData | BreakData | None:
    """Extract create/change/break payload from a habit inventory event; else None."""

    if event.name == CreateEvent.name and isinstance(event.data, CreateData):
        return event.data
    if event.name == ChangeEvent.name and isinstance(event.data, ChangeData):
        return event.data
    if event.name == BreakEvent.name and isinstance(event.data, BreakData):
        return event.data
    return None


def is_inventory_event(event: hsm.Event[typing.Any]) -> bool:
    return event.name in {CreateEvent.name, ChangeEvent.name, BreakEvent.name}


__all__ = [
    "BREAK_EVENT_NAME",
    "BreakData",
    "BreakEvent",
    "CHANGE_EVENT_NAME",
    "CREATE_EVENT_NAME",
    "ChangeData",
    "ChangeEvent",
    "CreateData",
    "CreateEvent",
    "Data",
    "Event",
    "data_from_event",
    "event_for_data",
    "is_inventory_event",
]
