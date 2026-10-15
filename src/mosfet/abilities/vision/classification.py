from .. import ability
from .. import classifying

import abc
import dataclasses
import typing

import hsm
import mosfet
import mosfet.media
import pydantic


VisualClassificationKind: typing.TypeAlias = typing.Literal["image", "text", "unreadable"]


class InputData(pydantic.BaseModel):
    """InputData for classifying a readable visual or text source before reading."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Readable source candidate that should be classified before the reading pipeline routes it.",
            "examples": [
                {"content": "Read this note."},
                {"content": {"content": "iVBORw0KGgo=", "media_type": "image/png"}},
            ],
        },
    )

    content: typing.Annotated[str, pydantic.Field(min_length=1, strict=True)] | mosfet.media.Image = pydantic.Field(
        description=(
            "Readable source payload: non-empty text, or one typed image (`mosfet.media.Image`) whose encoded "
            "bytes travel as base64 in JSON alongside its media type."
        ),
        examples=["Read this note.", {"content": "iVBORw0KGgo=", "media_type": "image/png"}],
    )


class OutputData(pydantic.BaseModel):
    """Provider-neutral classification used to route reading input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Routing classification that says whether reading should decode text, decode an image, or publish no readable content.",
            "examples": [
                {"kind": "text", "confidence": 0.99},
                {"kind": "image", "confidence": 0.87},
                {"kind": "unreadable", "confidence": 0.76},
            ],
        },
    )

    kind: VisualClassificationKind = pydantic.Field(
        description="Reading route selected for the classified source.",
        examples=["image"],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for the routing classification, normalized from 0.0 to 1.0.",
        examples=[0.87],
    )


class VisualClassifier(classifying.Classifier[InputData, OutputData], abc.ABC):
    """Classifier that routes readable visual or text input before decoding."""


_VisualClassificationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.vision.classification.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_VisualClassificationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.vision.classification.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_visual_classification_input(
    ctx: hsm.Context,
    instance: "VisualClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_visual_classification_output(
    ctx: hsm.Context,
    instance: "VisualClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_visual_classification_output(
    ctx: hsm.Context,
    instance: "VisualClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_visual_classification_failure(
    ctx: hsm.Context,
    instance: "VisualClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class VisualClassification(classifying.Classifying[InputData, OutputData]):
    """Ability to classify reading input before text or image decoding."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.vision.classification.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.vision.classification.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VisualClassificationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VisualClassificationApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_visual_classification_output_failure(
        ctx: hsm.Context,
        instance: "VisualClassification",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(
            message="Visual classification produced output that does not match its output schema."
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "VisualClassification",
        hsm.initial(hsm.target("/VisualClassification/Unclassified")),
        hsm.state(
            "Unclassified",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_visual_classification_input),
                hsm.target("/VisualClassification/Classifying"),
            ),
        ),
        hsm.state(
            "Classifying",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_VisualClassificationApplyCompletedEvent),
                hsm.guard(_has_visual_classification_output),
                hsm.effect(classifying.Classifying._dispatch_classifying_output),
                hsm.target("/VisualClassification/Unclassified"),
            ),
            hsm.transition(
                hsm.on(_VisualClassificationApplyCompletedEvent),
                hsm.guard(_has_invalid_visual_classification_output),
                hsm.effect(_dispatch_invalid_visual_classification_output_failure),
                hsm.target("/VisualClassification/Unclassified"),
            ),
            hsm.transition(
                hsm.on(_VisualClassificationApplyFailedEvent),
                hsm.guard(_has_visual_classification_failure),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("/VisualClassification/Unclassified"),
            ),
        ),
    )


__all__ = [
    "VisualClassification",
    "InputData",
    "VisualClassificationKind",
    "OutputData",
    "VisualClassifier",
]
