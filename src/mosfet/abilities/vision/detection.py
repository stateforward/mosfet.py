"""Object detection: what things are in an image and where, through one injected detector.

``ObjectDetector`` is the plain, non-HSM capacity a provider implements (a classic CV network, an
open-vocabulary model, a remote VLM). ``GenerativeDetector`` serves it over any image-capable
``text.TextGenerator``. ``ObjectDetection`` is the HSM wrapper an owning ability attaches and
addresses with typed events. Labels are model output: never a metric attribute, a routing key or
anything the bot branches on.

Telemetry (scope ``bot.abilities.vision``): ``bot.vision.detection.invalid`` counts the reply items
a ``GenerativeDetector`` skipped because they were not a valid detected object; no attributes.
"""

import abc
import dataclasses
import typing

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import media

from .. import ability
from .. import classifying
from ..language import text
from . import geometry

MAX_OBJECTS: typing.Final = 32
"""The most objects one detection reports."""

DETECT_INSTRUCTIONS: typing.Final = (
    "List the distinct objects you can see in this image. Answer with only a JSON array and no other text. "
    'Each item is {"label": a short name for the object, "confidence": how sure you are from 0 to 1, or null, '
    '"box": {"x": ..., "y": ..., "width": ..., "height": ...}}, where the box is normalized to the image: '
    "0 to 1, measured from the top-left corner. Answer [] when you see no objects."
)
"""The default request ``GenerativeDetector`` sends with the image; a composition may override it."""

_DEFAULT_MAX_REPLY_BYTES: typing.Final = 65_536
_REPLY_ITEMS: typing.Final = pydantic.TypeAdapter(list[pydantic.JsonValue])

_METER = metrics.get_meter("bot.abilities.vision")
_INVALID_ITEMS = _METER.create_counter(
    "bot.vision.detection.invalid",
    unit="{item}",
    description="Reply items a generative detector skipped because they were not a valid detected object.",
)


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


class Error(RuntimeError):
    """A detector could not find the objects in an image."""


class GenerativeDetector(ObjectDetector):
    """Finds objects by asking any image-capable text generator for a JSON array, once per image.

    Not an actor: a plain collaborator over a ``text.TextGenerator``, called directly. Each item of
    the reply is validated on its own; an item that is not a valid ``DetectedObject`` is skipped and
    counted, and the first ``max_objects`` valid items, in reply order, are kept.
    """

    _generator: text.TextGenerator
    _instructions: str
    _max_objects: int
    _max_reply_bytes: int

    def __init__(
        self,
        *,
        generator: text.TextGenerator,
        instructions: str = DETECT_INSTRUCTIONS,
        max_objects: int = MAX_OBJECTS,
        max_reply_bytes: int = _DEFAULT_MAX_REPLY_BYTES,
    ) -> None:
        """Detect through ``generator``, asking with ``instructions`` placed before the image.

        Raises:
            ValueError: If ``instructions`` is blank, ``max_objects`` is outside 1..``MAX_OBJECTS``
                or ``max_reply_bytes`` is not positive.
        """

        if not instructions.strip():
            raise ValueError("GenerativeDetector instructions must not be blank.")
        if not 1 <= max_objects <= MAX_OBJECTS:
            raise ValueError(f"GenerativeDetector max_objects must be in 1..{MAX_OBJECTS}.")
        if max_reply_bytes < 1:
            raise ValueError("GenerativeDetector max_reply_bytes must be positive.")
        self._generator = generator
        self._instructions = instructions
        self._max_objects = max_objects
        self._max_reply_bytes = max_reply_bytes

    @typing.override
    async def classify(self, input: InputData) -> OutputData:
        """Ask the generator what objects ``input.image`` holds; return the valid ones it named.

        Raises:
            Error: If the generator's model cannot take the image (checked before asking), the reply
                is larger than ``max_reply_bytes``, or it is not a JSON array.
        """

        if self._generator.image_capacity.admit((input.image,)) != (None,):
            raise Error("the generator's model cannot take this image")
        reply = await self._generator.generate(
            text.InputData(
                messages=(
                    text.TextMessage(
                        role=text.TextRole.USER,
                        content=(text.TextPart(text=self._instructions), text.ImagePart(image=input.image)),
                    ),
                ),
                tool_selection=text.ToolSelectionPolicy.NONE,
            )
        )
        size = len(reply.content.encode())
        if size > self._max_reply_bytes:
            raise Error(f"the reply of {size} bytes exceeds max_reply_bytes ({self._max_reply_bytes})")
        answer = reply.content.strip()
        if answer.startswith("```"):
            # A model may fence its JSON as markdown: read what is inside the fence.
            answer = answer.partition("\n")[2].removesuffix("```").strip()
        try:
            items = _REPLY_ITEMS.validate_json(answer)
        except pydantic.ValidationError as error:
            # The reply is model output: it never becomes the failure message.
            raise Error("the reply is not a JSON array of objects") from error
        found: list[DetectedObject] = []
        skipped = 0
        for item in items:
            try:
                found.append(DetectedObject.model_validate(item))
            except pydantic.ValidationError:
                skipped += 1
        if skipped:
            _INVALID_ITEMS.add(skipped)
        return OutputData(objects=tuple(found[: self._max_objects]))


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
    "DETECT_INSTRUCTIONS",
    "MAX_OBJECTS",
    "DetectedObject",
    "Error",
    "GenerativeDetector",
    "InputData",
    "ObjectDetection",
    "ObjectDetector",
    "OutputData",
]
