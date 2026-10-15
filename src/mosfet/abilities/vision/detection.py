"""Object detection: what things are in an image and where, through one injected detector.

``ObjectDetector`` is the plain, non-HSM capacity a provider implements (a classic CV network, an
open-vocabulary model, a remote VLM). ``ObjectDetection`` is the HSM wrapper an owning ability
attaches and addresses with typed events. Labels are model output: never a metric attribute, a
routing key or anything the bot branches on.
"""

import abc
import dataclasses
import typing

import hsm
import mosfet
import pydantic

from mosfet import media

from .. import ability
from .. import classifying
from . import geometry

MAX_OBJECTS: typing.Final = 32
"""The most objects one detection reports."""


class InputData(pydantic.BaseModel):
    """One image to find objects in."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One encoded image to find objects in.",
            "examples": [{"image": {"content": "/9j/4AAQ", "media_type": "image/jpeg"}}],
        },
    )

    image: media.Image = pydantic.Field(description="The image, exactly as its source encoded it.")


class DetectedObject(pydantic.BaseModel):
    """One thing a detector found: what it called it, how sure it was, and where it is."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One object found in the image: its label as the detector produced it, and where it is.",
            "examples": [
                {"label": "cup", "confidence": 0.87, "box": {"x": 0.4, "y": 0.5, "width": 0.1, "height": 0.15}}
            ],
        },
    )

    label: str = pydantic.Field(
        min_length=1,
        max_length=100,
        description="What the detector called the object, exactly as it produced it.",
        examples=["cup", "person"],
    )
    confidence: float | None = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="How sure the detector was, 0..1; null when it does not rate its findings.",
        examples=[0.87, None],
    )
    box: geometry.Box = pydantic.Field(description="Where the object is, in normalized image coordinates.")


class OutputData(pydantic.BaseModel):
    """Everything a detector found in one image, most confident first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The objects found in one image, by descending confidence (unrated last); empty when the "
                "detector found nothing."
            ),
            "examples": [
                {
                    "objects": [
                        {"label": "cup", "confidence": 0.87, "box": {"x": 0.4, "y": 0.5, "width": 0.1, "height": 0.15}}
                    ]
                },
                {"objects": []},
            ],
        },
    )

    objects: tuple[DetectedObject, ...] = pydantic.Field(
        max_length=MAX_OBJECTS,
        description="The objects found, by descending confidence with unrated objects last.",
    )

    @pydantic.field_validator("objects")
    @classmethod
    def _most_confident_first(cls, objects: tuple[DetectedObject, ...]) -> tuple[DetectedObject, ...]:
        # Stable: equally rated objects keep the detector's order.
        return tuple(sorted(objects, key=lambda found: (found.confidence is None, -(found.confidence or 0.0))))


class ObjectDetector(classifying.Classifier[InputData, OutputData], abc.ABC):
    """A provider-agnostic, non-HSM capacity to find objects in an image.

    ``classify`` raises on failure. A call that cannot be cancelled (a worker thread, a blocking
    SDK) keeps running after its owner gives up on it, so a provider bounds its own work.
    """


_ApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.vision.detection.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_ApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.vision.detection.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_input(ctx: hsm.Context, instance: "ObjectDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_output(ctx: hsm.Context, instance: "ObjectDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_output(ctx: hsm.Context, instance: "ObjectDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_failure(ctx: hsm.Context, instance: "ObjectDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class ObjectDetection(classifying.Classifying[InputData, OutputData]):
    """Ability that finds objects in one image at a time through its injected ``ObjectDetector``."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.vision.detection.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.vision.detection.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _ApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _ApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_output_failure(
        ctx: hsm.Context,
        instance: "ObjectDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="Object detection produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "ObjectDetection",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(hsm.on(input_event), hsm.guard(_has_input), hsm.target("../applying")),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_ApplyCompletedEvent),
                hsm.guard(_has_output),
                hsm.effect(classifying.Classifying._dispatch_classifying_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyCompletedEvent),
                hsm.guard(_has_invalid_output),
                hsm.effect(_dispatch_invalid_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("../idle"),
            ),
        ),
    )


__all__ = [
    "MAX_OBJECTS",
    "DetectedObject",
    "InputData",
    "ObjectDetection",
    "ObjectDetector",
    "OutputData",
]
