import typing

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


__all__ = [
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_CONTRACT",
    "EventData",
    "EventPayload",
    "OutputData",
    "OPTIONAL_OUTPUT_SCHEMA",
    "OPTIONAL_OUTPUT_SCHEMA_CONTRACT",
    "is_output",
]
