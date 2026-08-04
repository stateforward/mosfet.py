from .. import processing
from .. import ability
from . import input as cognition_input

import collections.abc
import typing

import hsm
from bot import event_schema
import pydantic

# Model-facing deliberate (System-2) handoff event name. Reasoning owns the event; intuition
# detects cascade handoff via this constant and/or ``processing.is_deliberative_handoff_schema``
# so stages never import each other for the check.
DELIBERATIVE_HANDOFF_EVENT_NAME: typing.Final[str] = "bot.ability.reasoning.input"


def is_deliberative_handoff_event_name(event_name: str) -> bool:
    """True when ``event_name`` is the shared deliberate handoff event."""

    return event_name == DELIBERATIVE_HANDOFF_EVENT_NAME


_Reference = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Stable reference used by a cognition host to resolve a target or modeled event.",
        examples=["runtime"],
    ),
]

EventPayload = dict[str, object]


class EventData(pydantic.BaseModel):
    """One selected HSM event to dispatch."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "One selected modeled HSM event. Processing dispatches it to the resolved target "
                "fire-and-forget; hosts do not re-apply OutputData. Put selection rationale on "
                "reason here — not on the event payload."
            ),
            "examples": [
                {
                    "target": "bot",
                    "event": "bot.focus_device",
                    "data": {"device": "device-a"},
                    "reason": "Attention should move to that device for this turn.",
                }
            ],
        },
    )

    target: _Reference | None = pydantic.Field(
        default=None,
        description=(
            "Stable reference for the host-local receiver of the modeled event. Hosts may omit this when the "
            "event resolves relative to focus or a single actor."
        ),
        examples=["bot", "device-a"],
    )
    event: _Reference = pydantic.Field(
        description="Canonical modeled HSM event name selected by cognition.",
        examples=["bot.focus_device"],
    )
    data: EventPayload | None = pydantic.Field(
        default=None,
        description=(
            "Optional JSON-serializable event data for the selected event. Do not include raw audio, message text, "
            "credentials, provider-specific objects, or high-cardinality diagnostics. Do not put selection "
            "rationale here — use reason on this envelope."
        ),
        examples=[{"device": "device-a"}],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional reason this event was selected.",
        examples=["Attention should move to that device for this turn."],
    )


# Always a tuple of selected events. Empty tuple = no dispatch.
OutputData: typing.TypeAlias = tuple[EventData, ...]


class IgnoreData(pydantic.BaseModel):
    """Explicit decision that this cognition turn should run no environment or body actions.

    Prefer selecting this event over an empty ``events`` array so models have a named
    branch under required tool-calling. Host treats ignore-only as handled (no cascade
    to deliberation solely because nothing else was selected) and does not dispatch it
    to devices or the bot body. Selection rationale belongs on the selection envelope
    (``EventData.reason`` / dispatch item ``reason``), not on this payload.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Deliberately ignore this stimulus: no device command, no focus change, no "
                "speech. Required when no other offered event should run. Do not use a device "
                "command, focus, or clear_focus as a stand-in for ignore. Put why on the "
                "selection envelope reason field, not on this empty payload."
            ),
            "examples": [{}],
        },
    )


IgnoreEvent = hsm.Event[IgnoreData](
    name="bot.ability.cognition.ignore",
    kind=event_schema.EventKind,
    schema=IgnoreData,
)


def is_ignore_event(event_name: str) -> bool:
    """True when ``event_name`` is the cognition ignore selection."""

    return event_name == IgnoreEvent.name


def without_ignore_selections(selections: processing.Events) -> processing.Events:
    """Drop cognition ignore selections (they are cognition only, not dispatch targets)."""

    return tuple(item for item in selections if not is_ignore_event(item.event))


class TurnData(pydantic.BaseModel):
    """Typed correlation and body input for one active cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    input: cognition_input.InputData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(
        min_length=1,
        description="Live operation-actor identifier that proves this turn is still current.",
    )


class CompletionData(pydantic.BaseModel):
    """Typed terminal from one cognition stage back to the Cognition host."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: TurnData
    output: OutputData | None = None


class FailureData(ability.FailureData):
    """Typed cognition-stage failure correlated to its originating turn."""

    turn: TurnData


OUTPUT_SCHEMA_CONTRACT = typing.cast(
    pydantic.TypeAdapter[OutputData],
    pydantic.TypeAdapter(OutputData),
)
OPTIONAL_OUTPUT_SCHEMA_CONTRACT = typing.cast(
    pydantic.TypeAdapter[OutputData | None],
    pydantic.TypeAdapter(OutputData | None),
)
OUTPUT_SCHEMA = typing.cast(dict[str, object], OUTPUT_SCHEMA_CONTRACT.json_schema())
OPTIONAL_OUTPUT_SCHEMA = typing.cast(
    dict[str, object],
    OPTIONAL_OUTPUT_SCHEMA_CONTRACT.json_schema(),
)


def is_output(value: object) -> typing.TypeGuard[OutputData]:
    """Return whether a value is a tuple of event selections."""

    if not isinstance(value, tuple):
        return False
    return all(isinstance(item, EventData) for item in value)


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: processing.InputData,
    selections: processing.Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    focus_candidates: tuple[str, ...],
    focused_device: str | None = None,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate body-action constraints, then dispatch selected modeled events.

    Cognition ignore is offered from the host model-offerable snapshot and may appear in
    selections as a handled terminal product, but it is not a body/device dispatch target
    (cognition only). Strip it before actor delivery.

    Body attention policy (focus/clear legality) is owned by ``bot.Bot``; cognition only
    fails closed on that policy before dispatch so illegal selections are not silent drops.
    ``focused_device`` is the body-stamped current focus for this turn (not a private field read).
    """

    import bot
    from bot import device

    event_metadata = dict(metadata or {})
    configured_device_names = frozenset(
        name for name, actor in input.actors.items() if name != "bot" and isinstance(actor, device.Device)
    )
    allowed = set(focus_candidates)
    candidates = tuple(item for item in configured_device_names if item in allowed)
    bot_actor = input.actors.get("bot")
    # Enforce candidates when devices are configured or the bot actor is a real Bot.
    enforce_candidates = bool(configured_device_names) or isinstance(bot_actor, bot.Bot)
    to_dispatch = without_ignore_selections(selections)
    if not to_dispatch:
        return

    current_focus = focused_device if focused_device else None
    for selection in to_dispatch:
        error = bot.Bot.attention_selection_error(
            selection,
            focus_candidates=candidates,
            configured_device_names=configured_device_names,
            enforce_candidates=enforce_candidates,
            focused_device=current_focus,
        )
        if error is not None:
            raise RuntimeError(error)

    await processing.dispatch_selected_events(
        ctx,
        input,
        to_dispatch,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
    )


__all__ = [
    "DELIBERATIVE_HANDOFF_EVENT_NAME",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_CONTRACT",
    "CompletionData",
    "EventData",
    "EventPayload",
    "FailureData",
    "IgnoreData",
    "IgnoreEvent",
    "OutputData",
    "TurnData",
    "OPTIONAL_OUTPUT_SCHEMA",
    "OPTIONAL_OUTPUT_SCHEMA_CONTRACT",
    "dispatch_selected_events",
    "is_deliberative_handoff_event_name",
    "is_ignore_event",
    "is_output",
    "without_ignore_selections",
]
