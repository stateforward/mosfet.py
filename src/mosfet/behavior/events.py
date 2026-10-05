"""Behavior HSM events: inventory create, change, break, and the routine tick.

First-class behavior-package contracts. Reflection applies the inventory events (build
Instance, validate Starlark source, store); reasoning may record them with episodes.
The event is the inventory operation; create/change install executable behavior.

``TickEvent`` is what a running persistent routine emits when one of its ``hsm.every`` /
``hsm.at`` schedules fires and its callbacks produce output.
"""

from __future__ import annotations

import typing

import hsm
from mosfet import event
import pydantic


class CreateData(pydantic.BaseModel):
    """Payload for ``bot.behavior.create``: install a new executable behavior."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Create a new executable behavior. Select may omit source (intent only); the create write "
                "step must author full Starlark HSM source from the behavior Starlark API and the observed "
                "pattern — not by copying canned examples."
            ),
            "examples": [
                {
                    "event": "bot.behavior.create",
                    "name": "GreetOnKnock",
                    "triggers": ["environment.sound"],
                    "description": "Say hello when a knock sound arrives.",
                    "reason": "Same knock→greeting pattern across recent episodes.",
                }
            ],
        },
    )

    event: typing.Literal["bot.behavior.create"] = pydantic.Field(
        default="bot.behavior.create",
        description="Behavior create event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable PascalCase behavior / HSM model name to install.")
    triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "External environment event names that may propose this behavior to fast intuition. "
            "Installed on the inventory instance alongside the Starlark program's own triggers."
        ),
        examples=[["environment.sound"]],
    )
    description: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Human-readable summary of what the behavior does, shown in inventory and reflection. "
            "The install keeps the caller-supplied description when present."
        ),
        examples=["Say hello when a knock sound arrives."],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Why this behavior is being installed now: the observed pattern across recent episodes "
            "that justifies it. Recorded for reflection, never dispatched."
        ),
        examples=["Same knock→greeting pattern across recent episodes."],
    )
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Starlark HSM behavior source authored for this install. Required on the create write step. "
            "Must declare input_event, output_event, and behavior = hsm.define(Name, ...) (or behavior_program(...)) "
            "with string-named callbacks. Invent names, schemas, guards, and effects from the observed "
            "pattern using the Starlark behavior API. Select intent may omit this field."
        ),
    )


class ChangeData(pydantic.BaseModel):
    """Payload for ``bot.behavior.change``: revise an existing executable behavior."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Change an existing executable behavior. Select may omit source (intent only); the change "
                "write step must author full revised Starlark HSM source from the behavior Starlark API, "
                "the existing behavior, and the observed pattern."
            ),
            "examples": [
                {
                    "event": "bot.behavior.change",
                    "name": "GreetOnKnock",
                    "reason": "Also clear focus before greeting.",
                }
            ],
        },
    )

    event: typing.Literal["bot.behavior.change"] = pydantic.Field(
        default="bot.behavior.change",
        description="Behavior change event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable name of the behavior to revise.")
    triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "Replacement external event names that may propose this behavior to fast intuition. "
            "Installed on the inventory instance alongside the revised Starlark program."
        ),
        examples=[["environment.sound", "conversation.greeting.recognized"]],
    )
    description: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Revised human-readable summary of what the behavior does, shown in inventory and reflection. "
            "The install keeps the caller-supplied description when present."
        ),
        examples=["Say hello when a knock sound arrives, after clearing focus."],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Why this behavior is being revised now: the observed gap or regression the revision fixes. "
            "Recorded for reflection, never dispatched."
        ),
        examples=["Also clear focus before greeting."],
    )
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Revised Starlark HSM behavior source. Required on the change write step. Must remain a valid "
            "executable behavior declaration whose model name matches this behavior name. Author revisions from "
            "the Starlark behavior API and existing_behavior.source — do not paste unrelated canned source."
        ),
    )


class BreakData(pydantic.BaseModel):
    """Payload for ``bot.behavior.break``: set an existing behavior status=BROKEN (do not delete)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Break a behavior that no longer improves (or harms) robot behavior. "
                "Sets inventory status=BROKEN so Autonomy will not run it; the row remains "
                "for a later bot.behavior.change repair. Optional reason becomes status_reason."
            ),
            "examples": [
                {
                    "event": "bot.behavior.break",
                    "name": "GreetOnKnock",
                    "reason": "Greeted while the bot was already mid-conversation.",
                }
            ],
        },
    )

    event: typing.Literal["bot.behavior.break"] = pydantic.Field(
        default="bot.behavior.break",
        description="Behavior break event name (discriminator).",
    )
    name: str = pydantic.Field(min_length=1, description="Stable name of the behavior to set status=BROKEN.")
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Why this behavior is being retired: the observed harm or staleness. Becomes the inventory "
            "status_reason so a later change can repair it."
        ),
        examples=["Greeted while the bot was already mid-conversation."],
    )


class TickData(pydantic.BaseModel):
    """Payload for ``bot.behavior.tick``: one fired schedule of a persistent routine and what it emitted."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A persistent routine's schedule fired. name is the routine, trigger is the schedule kind "
                "(every: fixed interval; at: wall-clock time of day), scheduled_at is the occurrence the "
                "schedule was due, fired_at is when it actually fired, and output carries the cognition "
                "event selections the routine emitted for this tick. The tick is an observation: it begins "
                "a new turn, and the bot decides what (if anything) to do with it."
            ),
            "examples": [
                {
                    "name": "MorningBriefing",
                    "trigger": "at",
                    "scheduled_at": "2026-10-05T08:00:00-04:00",
                    "fired_at": "2026-10-05T08:00:00.012000-04:00",
                    "output": [
                        {
                            "event": "bot.speaking.say",
                            "target": "speaker",
                            "data": {"text": "Good morning. It is eight o'clock."},
                            "reason": "Weekday morning briefing routine.",
                        }
                    ],
                }
            ],
        },
    )

    name: str = pydantic.Field(
        min_length=1,
        description="PascalCase name of the persistent routine (its behavior inventory name) that ticked.",
        examples=["MorningBriefing"],
    )
    trigger: typing.Literal["every", "at"] = pydantic.Field(
        description=(
            "Schedule kind that fired: every (hsm.every fixed interval) or at (hsm.at wall-clock time "
            "of day on chosen weekdays in a named time zone)."
        ),
        examples=["at", "every"],
    )
    scheduled_at: pydantic.AwareDatetime = pydantic.Field(
        description=(
            "Timezone-aware ISO-8601 occurrence this tick was due. For at() it is the wall-clock "
            "occurrence in the routine's time zone; for every() it is the interval deadline, which is "
            "when the timer fired."
        ),
        examples=["2026-10-05T08:00:00-04:00"],
    )
    fired_at: pydantic.AwareDatetime = pydantic.Field(
        description="Timezone-aware ISO-8601 time the schedule actually fired, read from the injected clock.",
        examples=["2026-10-05T08:00:00.012000-04:00"],
    )
    output: tuple[dict[str, object], ...] = pydantic.Field(
        default=(),
        description=(
            "Cognition event selections the routine emitted for this tick, each an object with event "
            "(required), target, data, and reason. Empty on the routine's own tick event, which its "
            "callbacks see before they emit."
        ),
        examples=[
            [
                {
                    "event": "bot.speaking.say",
                    "target": "speaker",
                    "data": {"text": "Good morning."},
                    "reason": "Weekday morning briefing routine.",
                }
            ]
        ],
    )

    @pydantic.field_validator("output")
    @classmethod
    def validate_selections(cls, value: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
        """Each selection names the event it selects."""

        for index, selection in enumerate(value):
            event_name = selection.get("event")
            if not isinstance(event_name, str) or not event_name:
                raise ValueError(f"output[{index}] must name a non-empty event.")
        return value


CreateEvent = hsm.Event[CreateData](
    name="bot.behavior.create",
    kind=event.EventKind,
    schema=CreateData,
)
ChangeEvent = hsm.Event[ChangeData](
    name="bot.behavior.change",
    kind=event.EventKind,
    schema=ChangeData,
)
BreakEvent = hsm.Event[BreakData](
    name="bot.behavior.break",
    kind=event.EventKind,
    schema=BreakData,
)


TickEvent = hsm.Event[TickData](
    name="bot.behavior.tick",
    schema=TickData,
)


def event_for_data(
    data: CreateData | ChangeData | BreakData,
) -> hsm.Event[CreateData] | hsm.Event[ChangeData] | hsm.Event[BreakData]:
    """Build the HSM event for a behavior create/change/break payload."""

    event_type = type(data)
    if event_type is CreateData:
        return CreateEvent.with_data(typing.cast(CreateData, data))
    if event_type is ChangeData:
        return ChangeEvent.with_data(typing.cast(ChangeData, data))
    if event_type is BreakData:
        return BreakEvent.with_data(typing.cast(BreakData, data))
    raise TypeError(f"Unsupported behavior event payload: {type(data)!r}")


def data_from_event(event: hsm.Event[typing.Any]) -> CreateData | ChangeData | BreakData | None:
    """Extract create/change/break payload from a behavior inventory event; else None."""

    data = event.data
    payload = typing.cast(object, data)
    event_type = type(payload)
    if event_type is CreateData:
        return typing.cast(CreateData, payload)
    if event_type is ChangeData:
        return typing.cast(ChangeData, payload)
    if event_type is BreakData:
        return typing.cast(BreakData, payload)
    return None


def is_inventory_event(event: hsm.Event[typing.Any]) -> bool:
    return isinstance(event.data, (CreateData, ChangeData, BreakData))


__all__ = [
    "BreakData",
    "BreakEvent",
    "ChangeData",
    "ChangeEvent",
    "CreateData",
    "CreateEvent",
    "TickData",
    "TickEvent",
    "data_from_event",
    "event_for_data",
    "is_inventory_event",
]
