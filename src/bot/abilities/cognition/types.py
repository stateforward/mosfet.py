from .. import processing
from .. import ability
from . import input as cognition_input

import collections.abc
import typing

import hsm
import pydantic

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
                "fire-and-forget; hosts do not re-apply OutputData."
            ),
            "examples": [
                {
                    "target": "phone",
                    "event": "phone.answer_call",
                    "data": {"call_id": "call-123"},
                    "reason": "Incoming call should be answered.",
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
        examples=["phone"],
    )
    event: _Reference = pydantic.Field(
        description="Canonical modeled HSM event name selected by cognition.",
        examples=["phone.answer_call"],
    )
    data: EventPayload | None = pydantic.Field(
        default=None,
        description=(
            "Optional JSON-serializable event data for the selected event. Do not include raw audio, message text, "
            "credentials, provider-specific objects, or high-cardinality diagnostics."
        ),
        examples=[{"call_id": "incoming-call"}],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional reason this event was selected.",
        examples=["The incoming call should be answered."],
    )


# Always a tuple of selected events. Empty tuple = no dispatch.
OutputData: typing.TypeAlias = tuple[EventData, ...]


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
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate body-action constraints, then dispatch selected modeled events."""

    import bot
    from bot import device

    event_metadata = dict(metadata or {})
    configured_candidates = tuple(
        name for name, actor in input.actors.items() if name != "bot" and isinstance(actor, device.Device)
    )
    allowed = set(focus_candidates)
    candidates = tuple(item for item in configured_candidates if item in allowed)

    for selection in selections:
        if selection.event == bot.FocusDeviceEvent.name:
            if selection.target is not None and selection.target != "bot":
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
            data = bot.FocusDeviceEventData.model_validate(selection.data or {})
            bot_actor = input.actors.get("bot")
            if data.device not in candidates and (configured_candidates or isinstance(bot_actor, bot.Bot)):
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
        elif selection.event == bot.ClearFocusEvent.name and selection.target not in (None, "bot"):
            raise RuntimeError("Processing selected clear_focus for a non-bot target.")

    await processing.dispatch_selected_events(
        ctx,
        input,
        selections,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
    )


__all__ = [
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_CONTRACT",
    "CompletionData",
    "EventData",
    "EventPayload",
    "FailureData",
    "OutputData",
    "TurnData",
    "OPTIONAL_OUTPUT_SCHEMA",
    "OPTIONAL_OUTPUT_SCHEMA_CONTRACT",
    "dispatch_selected_events",
    "is_output",
]
