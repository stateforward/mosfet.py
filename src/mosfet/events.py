import typing

import pydantic
import hsm
from mosfet import event
from pydantic.json_schema import SkipJsonSchema
from pydantic import PlainSerializer

T = typing.TypeVar("T", default=typing.Any, covariant=True)


class StimulusData(pydantic.BaseModel, typing.Generic[T]):
    """Typed causal parent for a product emitted from another HSM event.

    ``data`` is the complete upstream payload and the envelope fields preserve the upstream event's
    identity without using HSM metadata or looking up an actor graph. Products may nest another
    ``StimulusData`` in ``data`` through their own typed ``parent`` field.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    __model_facing_event_data__: typing.ClassVar[bool] = True

    event: str = pydantic.Field(
        min_length=1,
        description="Canonical HSM event name that emitted this upstream payload.",
        examples=["environment.sound", "bot.ability.listening.speech.output"],
    )
    data: T = pydantic.Field(description="Complete typed payload emitted by the upstream event.")
    id: str | None = pydantic.Field(
        default=None,
        description="Upstream HSM event correlation id, when the emitter stamped one.",
        examples=["turn-123"],
    )
    source: str | None = pydantic.Field(
        default=None,
        description="Upstream HSM event source identity, when stamped by its producer.",
        examples=["phone-1"],
    )
    target: str | None = pydantic.Field(
        default=None,
        description="Upstream HSM event target identity, when stamped for delivery.",
        examples=["listening-1"],
    )

    @classmethod
    def from_event(cls, event: hsm.Event[T]) -> "StimulusData[T]":
        """Capture an event's typed envelope and payload at emission time."""

        data = event.data
        if data is None:
            raise ValueError(f"Cannot capture event {event.name!r} without typed payload data.")
        return cls(
            event=event.name,
            data=data,
            id=event.id,
            source=event.source,
            target=event.target,
        )


DeviceReference = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Stable reference for a device. Only devices may become focused.",
        examples=["device-a"],
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
    "learning_child_teardown_failed",
    "learning_detach_rollback_failed",
]


class RebootEventData(pydantic.BaseModel):
    """Request a full Bot-owned teardown and activation cycle."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    reason: RebootReason = pydantic.Field(
        description="Stable unrecoverable failure category that requires a clean robot lifecycle restart.",
        examples=["cognition_child_teardown_failed"],
    )


class InputEventData(pydantic.BaseModel):
    """Interrupt signal observed by an active bot.

    This is how something that happens to a bot becomes an occasion for it. A device reports
    its own state change here; the body grants the turn on device identity alone and cognition
    decides what, if anything, the change is worth doing about.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "target_device": "device-a",
                    "priority": 0,
                    "observation": {
                        "event": "device.phone.ringing",
                        "data": {"call_id": "call-123"},
                        "id": "call-123",
                        "source": "phone-1",
                    },
                }
            ],
        },
    )

    target_device: DeviceReference | None = pydantic.Field(
        default=None,
        description=(
            "Stable reference for the device that produced or owns the input. A device does not "
            "know what its bot files it under, so it leaves this unset and the body resolves the "
            "device from the envelope source — the nerve the signal arrived on."
        ),
        examples=["device-a"],
    )
    priority: int = pydantic.Field(
        default=5,
        ge=0,
        le=10,
        description=(
            "How insistent this signal is, where 0 is the highest priority and 10 is the lowest. "
            "Like loudness, this belongs to the signal rather than to whatever produced it; a "
            "source that does not distinguish leaves it at the ordinary default."
        ),
        examples=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    observation: StimulusData[object] | None = pydantic.Field(
        default=None,
        description=(
            "Typed product that caused this input. It preserves the producing event's concrete payload and envelope "
            "provenance without reducing the observation to an event-name string and untyped JSON bag."
        ),
        examples=[
            {
                "event": "device.phone.ringing",
                "data": {"call_id": "call-123"},
                "id": "call-123",
                "source": "phone-1",
            }
        ],
    )


def _jsonable_bot_observed_event(observed_event: hsm.Event[typing.Any]) -> dict[str, object]:
    return {
        "event": observed_event.name,
        "data": event.event_json_value(observed_event.data),
    }


ObservedBotEvent: typing.TypeAlias = SkipJsonSchema[
    typing.Annotated[
        hsm.Event[typing.Any],
        PlainSerializer(_jsonable_bot_observed_event, return_type=dict[str, object], when_used="json"),
    ]
]
InputData: typing.TypeAlias = InputEventData | ObservedBotEvent


class FocusDeviceEventData(pydantic.BaseModel):
    """Command payload that requests moving bot focus to one configured device.

    Selection rationale belongs on the cognition selection envelope (``EventData.reason`` /
    dispatch item ``reason``), not on this payload.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Move body attention (focus) to one configured device. device is required and must "
                "be a live focus candidate. Prefer this only when a device should become the active "
                "attention target — not as a substitute for speaking or reasoning. Why this was "
                "selected is selection-envelope reason, not a field here."
            ),
            "examples": [{"device": "device-a"}],
        },
    )

    device: DeviceReference = pydantic.Field(
        description="Required stable configured device reference that should become focused.",
        examples=["device-a"],
    )


class ClearFocusEventData(pydantic.BaseModel):
    """Command payload that requests leaving the bot without a focused device.

    Selection rationale belongs on the cognition selection envelope (``EventData.reason`` /
    dispatch item ``reason``), not on this payload.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Clear body focus so no device is the active attention target. Use only when focus "
                "should end (for example the focused device is gone). Do not use this as a default "
                "response to user speech — prefer a behavior/topology route or reasoning input. Why this "
                "was selected is selection-envelope reason, not a field here."
            ),
            "examples": [{}],
        },
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
                    "focus_candidates": ["device-a"],
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
        examples=[["device-a"], ["device-a", "device-b"], []],
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
# Body ingress, never a model tool: an occasion is something that happens to a bot, so it keeps
# the default event kind rather than the tool-offerable event.EventKind. A bot cannot
# select having a moment.
InputEvent = hsm.Event[InputEventData](
    name="bot.input",
    kind=InputEventKind,
    schema=InputEventData,
)
FocusDeviceEvent = hsm.Event[FocusDeviceEventData](
    name="bot.focus_device",
    kind=event.EventKind,
    schema=FocusDeviceEventData,
)
ClearFocusEvent = hsm.Event[ClearFocusEventData](
    name="bot.clear_focus",
    kind=event.EventKind,
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
