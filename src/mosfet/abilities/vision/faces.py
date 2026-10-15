"""Face detection: where faces are in an image and, when the detector can, each face's print.

``FaceDetector`` is the plain, non-HSM capacity a provider implements (a classic CV network, an
omni-modal model). ``FaceDetection`` is the HSM wrapper an owning ability attaches and addresses
with typed events. A face print is the shared biometric ``identification.Print``: full capture
renders it ``<biometric:Print>``, and it never leaves the face stage, the identify request and the
identity registry.
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
from .. import identification

MAX_FACES: typing.Final = 16
"""The most faces one detection reports."""


class InputData(pydantic.BaseModel):
    """One image to find faces in."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One encoded image to find faces in.",
            "examples": [{"image": {"content": "/9j/4AAQ", "media_type": "image/jpeg"}}],
        },
    )

    image: media.Image = pydantic.Field(description="The image, exactly as its source encoded it.")


class DetectedFace(pydantic.BaseModel):
    """One face a detector found: where it is, how sure the detector was, and its print if it took one."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One face found in the image, with its biometric print when the detector takes one.",
            "examples": [
                {
                    "box": {"x": 0.42, "y": 0.18, "width": 0.12, "height": 0.2},
                    "confidence": 0.93,
                    "print": {"vector": [0.6, 0.8], "model": "opencv-sface-2021dec"},
                }
            ],
        },
    )

    box: geometry.Box = pydantic.Field(description="Where the face is, in normalized image coordinates.")
    confidence: float | None = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="How sure the detector was that this is a face, 0..1; null when it does not rate faces.",
        examples=[0.93, None],
    )
    print: identification.Print | None = pydantic.Field(
        description="The face's biometric print in the shared print type; null when the detector takes none.",
    )


class OutputData(pydantic.BaseModel):
    """Every face a detector found in one image, largest first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "The faces found in one image, by descending box area; empty when there were none.",
            "examples": [
                {
                    "faces": [
                        {"box": {"x": 0.42, "y": 0.18, "width": 0.12, "height": 0.2}, "confidence": 0.93, "print": None}
                    ]
                },
                {"faces": []},
            ],
        },
    )

    faces: tuple[DetectedFace, ...] = pydantic.Field(
        max_length=MAX_FACES,
        description="The faces found, by descending box area.",
    )

    @pydantic.field_validator("faces")
    @classmethod
    def _largest_first(cls, faces: tuple[DetectedFace, ...]) -> tuple[DetectedFace, ...]:
        # Stable: faces of equal area keep the detector's order.
        return tuple(sorted(faces, key=lambda face: -(face.box.width * face.box.height)))


class FaceDetector(classifying.Classifier[InputData, OutputData], abc.ABC):
    """A provider-agnostic, non-HSM capacity to find faces in an image and, optionally, print them.

    ``classify`` raises on failure. A call that cannot be cancelled (a worker thread, a blocking
    SDK) keeps running after its owner gives up on it, so a provider bounds its own work.
    """


_ApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.vision.faces.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_ApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.vision.faces.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_input(ctx: hsm.Context, instance: "FaceDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_output(ctx: hsm.Context, instance: "FaceDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_output(ctx: hsm.Context, instance: "FaceDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_failure(ctx: hsm.Context, instance: "FaceDetection", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class FaceDetection(classifying.Classifying[InputData, OutputData]):
    """Ability that finds faces in one image at a time through its injected ``FaceDetector``."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.vision.faces.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.vision.faces.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _ApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _ApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_output_failure(
        ctx: hsm.Context,
        instance: "FaceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="Face detection produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "FaceDetection",
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
    "MAX_FACES",
    "DetectedFace",
    "FaceDetection",
    "FaceDetector",
    "InputData",
    "OutputData",
]
