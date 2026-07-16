import collections.abc
import typing

import pydantic
import hsm
from pydantic.json_schema import SkipJsonSchema
from pydantic import PlainSerializer

BotOperationReason = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Concise reason the ability selected this output or operation.",
        examples=["Incoming notification is lower priority than the active call."],
    ),
]
DeviceReference = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Stable reference for a device. Only devices may become focused.",
        examples=["phone"],
    ),
]


class ActivateEventData(pydantic.BaseModel):
    """No-payload command that activates a bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class DeactivateEventData(pydantic.BaseModel):
    """No-payload command that deactivates a bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


RebootReason = typing.Literal[
    "cognition_child_teardown_failed",
    "cognition_cancel_teardown_failed",
    "cognition_detach_rollback_failed",
]


class RebootEventData(pydantic.BaseModel):
    """Request a full Bot-owned teardown and activation cycle."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    reason: RebootReason = pydantic.Field(
        description="Stable unrecoverable failure category that requires a clean robot lifecycle restart.",
        examples=["cognition_child_teardown_failed"],
    )


class InputEventData(pydantic.BaseModel):
    """Interrupt signal observed by an active bot."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "target_device": "phone",
                    "priority": 0,
                    "source_event": "phone.incoming_call",
                    "payload": {"call_id": "call-123"},
                }
            ],
        },
    )

    target_device: DeviceReference = pydantic.Field(
        description="Stable reference for the device that produced or owns the input.",
        examples=["phone"],
    )
    priority: int = pydantic.Field(
        ge=0,
        le=10,
        description="Priority of the interrupt, where 0 is the highest priority and 10 is the lowest priority.",
        examples=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    source_event: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Modeled event name that caused this input, when the runtime knows it. Cognition may use this as "
            "decision context, but the source device still owns the event semantics and lifecycle."
        ),
        examples=["phone.incoming_call"],
    )
    payload: dict[str, object] | None = pydantic.Field(
        default=None,
        description=(
            "JSON-serializable payload from the source event, when needed for cognition to choose a typed output. "
            "Do not include raw audio, text transcripts, credentials, provider-specific blobs, or high-cardinality "
            "diagnostic data."
        ),
        examples=[{"call_id": "call-123"}],
    )


def _event_data_contains_binary(value: object) -> bool:
    if isinstance(value, bytes | bytearray | memoryview):
        return True
    if isinstance(value, pydantic.BaseModel):
        return _event_data_contains_binary(value.model_dump(mode="python"))
    if isinstance(value, collections.abc.Mapping):
        return any(_event_data_contains_binary(item) for item in value.values())
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        return any(_event_data_contains_binary(item) for item in value)
    return False


def _jsonable_event_data(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes"}
    if isinstance(value, memoryview):
        return {"type": "bytes"}
    if isinstance(value, pydantic.BaseModel):
        if _event_data_contains_binary(value):
            return {"type": type(value).__qualname__}
        return typing.cast(object, value.model_dump(mode="json"))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): _jsonable_event_data(item) for key, item in mapping.items()}
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, bytes | bytearray | str):
        return [_jsonable_event_data(item) for item in value]
    return {"type": type(value).__qualname__}


def _jsonable_bot_observed_event(event: hsm.Event[typing.Any]) -> dict[str, object]:
    return {
        "event": event.name,
        "data": _jsonable_event_data(event.data),
    }


ObservedBotEvent: typing.TypeAlias = SkipJsonSchema[
    typing.Annotated[
        hsm.Event[typing.Any],
        PlainSerializer(_jsonable_bot_observed_event, return_type=dict[str, object], when_used="json"),
    ]
]
BotInputData: typing.TypeAlias = InputEventData | ObservedBotEvent


class FocusDeviceEventData(pydantic.BaseModel):
    """Command payload that requests moving bot focus to one configured device."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Move body attention (focus) to one configured device. device is required and must "
                "be a live focus candidate. Prefer this only when a device should become the active "
                "attention target — not as a substitute for speaking or reasoning."
            ),
            "examples": [{"device": "phone", "reason": "Incoming call should become the active device."}],
        },
    )

    device: DeviceReference = pydantic.Field(
        description="Required stable configured device reference that should become focused.",
        examples=["phone"],
    )
    reason: BotOperationReason | None = pydantic.Field(
        default=None,
        description="Optional reason the focus operation was selected.",
    )


class ClearFocusEventData(pydantic.BaseModel):
    """Command payload that requests leaving the bot without a focused device."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Clear body focus so no device is the active attention target. Use only when focus "
                "should end (for example the focused device is gone). Do not use this as a default "
                "response to user speech — prefer speaking.input and/or reasoning.input."
            ),
            "examples": [{"reason": "The focused device has no remaining available transition."}],
        },
    )

    reason: BotOperationReason | None = pydantic.Field(
        default=None,
        description="Optional reason the clear-focus operation was selected.",
    )


class ProcessingCompletedEventData(pydantic.BaseModel):
    """Completion signal that the primary ability finished a processing turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "output": [],
                    "focus_candidates": ["phone"],
                }
            ]
        },
    )

    output: object = pydantic.Field(
        description=(
            "Terminal payload from the primary ability. When the primary ability is Cognition, this is typically a "
            "cognitive output; the body does not interpret or apply it."
        ),
        examples=[[]],
    )
    focus_candidates: tuple[DeviceReference, ...] = pydantic.Field(
        description=(
            "Stable device references associated with this processing turn for focus bookkeeping on the body. "
            "Empty when the turn had no device context (e.g. no configured devices)."
        ),
        examples=[["phone"], ["phone", "browser"], []],
    )


class ProcessingFailedEventData(pydantic.BaseModel):
    """FailureData signal produced when the bot ability cannot produce a cognitive output."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"message": "ability timed out"}],
        },
    )

    message: str = pydantic.Field(
        min_length=1,
        description="Human-readable failure message for the bot processing stage.",
        examples=["ability timed out"],
    )


class ActivatingDoneEventData(pydantic.BaseModel):
    """No-payload completion signal that bot startup finished after required devices attached."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class ActivatingFailedEventData(pydantic.BaseModel):
    """No-payload error signal that bot activation failed during device or ability startup."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class DeactivatingDoneEventData(pydantic.BaseModel):
    """No-payload completion signal that bot abilities stopped during deactivation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


InputEventKind = hsm.EventKind

ActivateEvent = hsm.Event[ActivateEventData](
    name="bot.activate",
    schema=ActivateEventData,
)
DeactivateEvent = hsm.Event[DeactivateEventData](
    name="bot.deactivate",
    schema=DeactivateEventData,
)
RebootEvent = hsm.Event[RebootEventData](
    name="bot.reboot",
    schema=RebootEventData,
)
InputEvent = hsm.Event[InputEventData](
    name="bot.input",
    kind=InputEventKind,
    schema=InputEventData,
)
FocusDeviceEvent = hsm.Event[FocusDeviceEventData](
    name="bot.focus_device",
    kind=hsm.CallEventKind,
    schema=FocusDeviceEventData,
)
ClearFocusEvent = hsm.Event[ClearFocusEventData](
    name="bot.clear_focus",
    kind=hsm.CallEventKind,
    schema=ClearFocusEventData,
)
ProcessingCompletedEvent = hsm.Event[ProcessingCompletedEventData](
    name="bot.processing.completed",
    kind=hsm.CompletionEventKind,
    schema=ProcessingCompletedEventData,
)
ProcessingFailedEvent = hsm.Event[ProcessingFailedEventData](
    name="bot.processing.failed",
    kind=hsm.ErrorEventKind,
    schema=ProcessingFailedEventData,
)
ActivatingDoneEvent = hsm.Event[ActivatingDoneEventData](
    name="bot.activated",
    kind=hsm.CompletionEventKind,
    schema=ActivatingDoneEventData,
)
ActivatingFailedEvent = hsm.Event[ActivatingFailedEventData](
    name="bot.activating.failed",
    kind=hsm.ErrorEventKind,
    schema=ActivatingFailedEventData,
)
DeactivatingDoneEvent = hsm.Event[DeactivatingDoneEventData](
    name="bot.deactivated",
    kind=hsm.CompletionEventKind,
    schema=DeactivatingDoneEventData,
)
