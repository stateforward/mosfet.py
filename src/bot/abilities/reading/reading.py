from .. import ability
from .. import decoding
from .. import encoding
from .. import vision

import base64
import binascii
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import bot

from bot.protocols import attachment
import pydantic

from bot.telemetry import observer

InputKind: typing.TypeAlias = typing.Literal["image", "text"]
ReadingOutputKind: typing.TypeAlias = typing.Literal["image", "text", "unreadable"]
ReadingStage: typing.TypeAlias = typing.Literal[
    "classification",
    "text_decoding",
    "image_decoding",
    "output_encoding",
]

# Minimal correlation id only (HSM-CORRELATION-001); not a source_event bag.
_READING_ACTIVE_OPERATION_ID_ATTRIBUTE = "reading_active_operation_id"
_STAGE_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)


class InputData(pydantic.BaseModel):
    """InputData for focusing the reading ability on text or image content."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Readable source candidate that focuses the reading ability.",
            "examples": [
                {"kind": "text", "content": "Read this note."},
                {"kind": "image", "content": "aW1hZ2UgYnl0ZXM="},
            ],
        },
    )

    kind: InputKind = pydantic.Field(
        description="InputData carrier that determines the first concrete reading route after classification.",
        examples=["text", "image"],
    )
    content: str | bytes = pydantic.Field(
        description=(
            "Readable source payload. Text inputs carry a string, while image inputs carry raw image bytes encoded "
            "as base64 in JSON."
        ),
        examples=["Read this note.", "aW1hZ2UgYnl0ZXM="],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def decode_image_content(cls, data: object) -> object:
        """DecodeData JSON image content before the kind/content invariant is checked."""

        if not isinstance(data, dict):
            return data
        source = typing.cast(dict[str, object], data)
        if source.get("kind") != "image":
            return source
        content = source.get("content")
        if not isinstance(content, str):
            return source
        try:
            decoded = base64.b64decode(content, altchars=b"-_", validate=True)
        except binascii.Error as error:
            raise ValueError("image reading input requires base64 content.") from error
        decoded_data: dict[str, object] = dict(source)
        decoded_data["content"] = decoded
        return decoded_data

    @pydantic.model_validator(mode="after")
    def validate_content_kind(self) -> typing.Self:
        """Keep the declared input carrier aligned with the payload type."""

        if self.kind == "text" and not isinstance(self.content, str):
            raise ValueError("text reading input requires string content.")
        if self.kind == "image" and not isinstance(self.content, bytes):
            raise ValueError("image reading input requires bytes content.")
        return self


class OutputData(pydantic.BaseModel):
    """Normalized readable content emitted by the reading ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Conversation-ready readable content produced after classifying, decoding, and encoding a reading input.",
            "examples": [
                {"text": "Read this note.", "source_kind": "text", "confidence": 0.99},
                {"text": "Total due: $14.21", "source_kind": "image", "confidence": 0.87},
                {"text": "", "source_kind": "unreadable", "confidence": 0.72},
            ],
        },
    )

    text: str = pydantic.Field(
        description="Normalized readable text. Empty text represents classified input with no readable content.",
        examples=["Total due: $14.21"],
    )
    source_kind: ReadingOutputKind = pydantic.Field(
        description="Classified source route that produced this reading output.",
        examples=["image"],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence carried from the classification or decoding route, normalized from 0.0 to 1.0.",
        examples=[0.87],
    )


class FailedEventData(pydantic.BaseModel):
    """FailureData signal produced when a reading stage cannot complete."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when a reading stage cannot complete.",
            "examples": [{"stage": "image_decoding", "message": "image decoder unavailable"}],
        },
    )

    stage: ReadingStage = pydantic.Field(
        description="Reading pipeline stage that failed.",
        examples=["image_decoding"],
    )
    message: str = pydantic.Field(
        description="Human-readable failure message for the reading stage.",
        examples=["image decoder unavailable"],
    )

    @classmethod
    def from_ability_failure(cls, *, stage: ReadingStage, failure: ability.FailureData) -> typing.Self:
        """Adapt an ability-protocol failure into a reading stage failure."""

        return cls(stage=stage, message=failure.message)


class _ReadingClassifiedEventData(pydantic.BaseModel):
    """Private completion payload for reading classification."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    input: InputData
    classification: vision.classification.OutputData


class _ReadingOutputCandidateEventData(pydantic.BaseModel):
    """Private completion payload for conversation-ready reading output candidates."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str
    source_kind: ReadingOutputKind
    confidence: float | None = pydantic.Field(default=None, ge=0.0, le=1.0)


ReadingInputEvent = hsm.Event[InputData](
    name="bot.ability.reading.input",
    schema=InputData,
)
ReadingOutputEvent = hsm.Event[OutputData](
    name="bot.ability.reading.output",
    schema=OutputData,
)
ReadingFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.reading.failed",
    schema=FailedEventData,
)
_ReadingClassificationCompletedEvent = hsm.Event[_ReadingClassifiedEventData](
    name="bot.ability.reading.classification.completed",
    kind=hsm.CompletionEventKind,
    schema=_ReadingClassifiedEventData,
)
_ReadingStageFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.reading.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=FailedEventData,
)
_ReadingTextDecodedEvent = hsm.Event[_ReadingOutputCandidateEventData](
    name="bot.ability.reading.text.decoded",
    kind=hsm.CompletionEventKind,
    schema=_ReadingOutputCandidateEventData,
)
_ReadingImageDecodedEvent = hsm.Event[_ReadingOutputCandidateEventData](
    name="bot.ability.reading.image.decoded",
    kind=hsm.CompletionEventKind,
    schema=_ReadingOutputCandidateEventData,
)
_ReadingUnreadableOutputReadyEvent = hsm.Event[_ReadingOutputCandidateEventData](
    name="bot.ability.reading.unreadable.output.ready",
    kind=hsm.CompletionEventKind,
    schema=_ReadingOutputCandidateEventData,
)
_ReadingOutputEncodedEvent = hsm.Event[OutputData](
    name="bot.ability.reading.output.encoded",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_ReadingBehavior = collections.abc.Callable[
    [hsm.Context, "Reading", hsm.Event[typing.Any]],
    collections.abc.Coroutine[None, None, None] | None,
]
_ReadingGuard = collections.abc.Callable[[hsm.Context, "Reading", hsm.Event[typing.Any]], bool]


def _reading_active_operation_id(instance: "Reading") -> str | None:
    value, ok = typing.cast(tuple[object, bool], instance.get(_READING_ACTIVE_OPERATION_ID_ATTRIBUTE))
    if ok and isinstance(value, str) and value:
        return value
    return None


def _set_reading_active_operation_id(instance: "Reading", operation_id: str | None) -> None:
    _ = instance.set(_READING_ACTIVE_OPERATION_ID_ATTRIBUTE, operation_id)


def _start_reading_operation(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> None:
    """Record only the in-flight operation id for stale-result correlation."""

    del ctx
    _set_reading_active_operation_id(instance, event.id if event.id else None)


def _clear_reading_operation(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _set_reading_active_operation_id(instance, None)


def _matches_active_reading_operation(instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    active_operation_id = _reading_active_operation_id(instance)
    if active_operation_id is None:
        return True
    return event.id == active_operation_id


def _reading_event_with_context(
    event: hsm.Event[typing.Any],
    source: hsm.Event[typing.Any],
    *,
    public_metadata: bool = False,
) -> hsm.Event[typing.Any]:
    del public_metadata
    operation_id = source.id if source.id else None
    if operation_id is not None:
        event = event.with_data_and_id(event.data, operation_id)
    # Telemetry only; stage payloads ride private completions / activity locals.
    return dataclasses.replace(
        event,
        source=source.source,
        target=source.target,
        metadata=dict(source.metadata),
    )


def _dispatch_reading_terminal_output(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
    output: OutputData,
) -> None:
    terminal = _reading_event_with_context(instance.output_event.with_data(output), event, public_metadata=True)
    terminal = dataclasses.replace(
        terminal,
        source=hsm.id(instance),
        target=(
            event.source
            if event.target == hsm.id(instance) and event.source and event.source != hsm.id(instance)
            else None
        ),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_reading_terminal_failure(
    ctx: hsm.Context,
    instance: "Reading",
    source: hsm.Event[typing.Any],
    failure: FailedEventData,
) -> None:
    terminal = _reading_event_with_context(instance.failed_event.with_data(failure), source, public_metadata=True)
    terminal = dataclasses.replace(
        terminal,
        source=hsm.id(instance),
        target=(
            source.source
            if source.target == hsm.id(instance) and source.source and source.source != hsm.id(instance)
            else None
        ),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _reading_visual_classifier(instance: "Reading") -> vision.classification.VisualClassification:
    return Reading.visual_classifier(instance)


def _reading_text_decoder(instance: "Reading") -> decoding.Decoding[str, str]:
    return Reading.text_decoder(instance)


def _reading_image_decoder(instance: "Reading") -> decoding.Decoding[bytes, str]:
    return Reading.image_decoder(instance)


def _reading_output_encoder(instance: "Reading") -> encoding.Encoding[typing.Any, typing.Any]:
    return Reading.output_encoder(instance)


def _stage_failure(stage: ReadingStage, error: BaseException) -> FailedEventData:
    return FailedEventData(stage=stage, message=str(error))


async def _run_reading_child(
    instance: "Reading",
    event: hsm.Event[typing.Any],
    *,
    child: ability.Ability[typing.Any, typing.Any],
    child_input: object,
    parent_operation_id: str,
    stage: ReadingStage,
) -> hsm.Event[typing.Any]:
    child_operation_id = f"{parent_operation_id}:{stage}:{uuid.uuid4().hex}"
    return await ability.run_terminal_operation(
        instance.context(),
        child=child,
        request=dataclasses.replace(
            child.input_event.with_data_and_id(child_input, child_operation_id),
            metadata=dict(event.metadata),
        ),
        terminals=(child.output_event, child.failed_event),
        timeout=_STAGE_OPERATION_TIMEOUT,
    )


async def _run_reading_classification(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    """Classification activity: InputData is activity-local; completion carries both input+label."""

    input = event.data
    assert isinstance(input, InputData)
    operation_id = _reading_active_operation_id(instance) or (event.id if event.id else "")
    if not operation_id:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="classification", message="Reading refused apply without operation id.")
                ),
                event,
            ),
        )
        return
    try:
        classification_input = vision.classification.InputData(kind=input.kind, content=input.content)
    except Exception as error:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(_stage_failure("classification", error)),
                event,
            ),
        )
        return
    child = _reading_visual_classifier(instance)
    terminal = await _run_reading_child(
        instance,
        event,
        child=typing.cast(ability.Ability[typing.Any, typing.Any], child),
        child_input=classification_input,
        parent_operation_id=operation_id,
        stage="classification",
    )
    if terminal.name == child.failed_event.name:
        message = getattr(terminal.data, "message", "Reading classification child failed.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(FailedEventData(stage="classification", message=str(message))),
                event,
            ),
        )
        return
    classification = terminal.data
    if not isinstance(classification, vision.classification.OutputData):
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="classification", message="Reading classification produced invalid output.")
                ),
                event,
            ),
        )
        return
    completion = _ReadingClassifiedEventData(input=input, classification=classification)
    _ = hsm.dispatch(
        ctx,
        instance,
        _reading_event_with_context(_ReadingClassificationCompletedEvent.with_data(completion), event),
    )


async def _run_text_decoding(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    """Text decode activity: classified payload is the entry event (completion), not a module bag."""

    classified = event.data
    assert isinstance(classified, _ReadingClassifiedEventData)
    operation_id = _reading_active_operation_id(instance) or (event.id if event.id else "")
    if not operation_id:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="text_decoding", message="Reading refused decode without operation id.")
                ),
                event,
            ),
        )
        return
    try:
        content = classified.input.content
        assert isinstance(content, str)
    except Exception as error:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(_stage_failure("text_decoding", error)),
                event,
            ),
        )
        return
    child = _reading_text_decoder(instance)
    terminal = await _run_reading_child(
        instance,
        event,
        child=typing.cast(ability.Ability[typing.Any, typing.Any], child),
        child_input=content,
        parent_operation_id=operation_id,
        stage="text_decoding",
    )
    if terminal.name == child.failed_event.name:
        message = getattr(terminal.data, "message", "Reading text decoder failed.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(FailedEventData(stage="text_decoding", message=str(message))),
                event,
            ),
        )
        return
    text = terminal.data
    if not isinstance(text, str):
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="text_decoding", message="Reading text decoder produced non-text output.")
                ),
                event,
            ),
        )
        return
    completion = _ReadingOutputCandidateEventData(
        text=text,
        source_kind="text",
        confidence=classified.classification.confidence,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _reading_event_with_context(_ReadingTextDecodedEvent.with_data(completion), event),
    )


async def _run_image_decoding(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    """Image decode activity: classified payload is the entry event (completion), not a module bag."""

    classified = event.data
    assert isinstance(classified, _ReadingClassifiedEventData)
    operation_id = _reading_active_operation_id(instance) or (event.id if event.id else "")
    if not operation_id:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="image_decoding", message="Reading refused decode without operation id.")
                ),
                event,
            ),
        )
        return
    try:
        content = classified.input.content
        assert isinstance(content, bytes)
    except Exception as error:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(_stage_failure("image_decoding", error)),
                event,
            ),
        )
        return
    child = _reading_image_decoder(instance)
    terminal = await _run_reading_child(
        instance,
        event,
        child=typing.cast(ability.Ability[typing.Any, typing.Any], child),
        child_input=content,
        parent_operation_id=operation_id,
        stage="image_decoding",
    )
    if terminal.name == child.failed_event.name:
        message = getattr(terminal.data, "message", "Reading image decoder failed.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(FailedEventData(stage="image_decoding", message=str(message))),
                event,
            ),
        )
        return
    text = terminal.data
    if not isinstance(text, str):
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="image_decoding", message="Reading image decoder produced non-text output.")
                ),
                event,
            ),
        )
        return
    completion = _ReadingOutputCandidateEventData(
        text=text,
        source_kind="image",
        confidence=classified.classification.confidence,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _reading_event_with_context(_ReadingImageDecodedEvent.with_data(completion), event),
    )


async def _run_unreadable_output_preparation(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    classified = event.data
    assert isinstance(classified, _ReadingClassifiedEventData)
    completion = _ReadingOutputCandidateEventData(
        text="",
        source_kind="unreadable",
        confidence=classified.classification.confidence,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _reading_event_with_context(_ReadingUnreadableOutputReadyEvent.with_data(completion), event),
    )


async def _run_output_encoding(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    """Encode activity: candidate is entry-event local (HSM-COMPLETION-001)."""

    candidate = event.data
    assert isinstance(candidate, _ReadingOutputCandidateEventData)
    operation_id = _reading_active_operation_id(instance) or (event.id if event.id else "")
    if not operation_id:
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(stage="output_encoding", message="Reading refused encode without operation id.")
                ),
                event,
            ),
        )
        return
    output_candidate = OutputData(
        text=candidate.text,
        source_kind=candidate.source_kind,
        confidence=candidate.confidence,
    )
    child = typing.cast(ability.Ability[typing.Any, typing.Any], _reading_output_encoder(instance))
    terminal = await _run_reading_child(
        instance,
        event,
        child=child,
        child_input=output_candidate,
        parent_operation_id=operation_id,
        stage="output_encoding",
    )
    if terminal.name == child.failed_event.name:
        message = getattr(terminal.data, "message", "Reading encoder failed.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(FailedEventData(stage="output_encoding", message=str(message))),
                event,
            ),
        )
        return
    output = terminal.data
    if not isinstance(output, OutputData):
        _ = hsm.dispatch(
            ctx,
            instance,
            _reading_event_with_context(
                _ReadingStageFailedEvent.with_data(
                    FailedEventData(
                        stage="output_encoding",
                        message="Reading output encoder produced output that does not match its output schema.",
                    )
                ),
                event,
            ),
        )
        return
    _ = hsm.dispatch(
        ctx,
        instance,
        _reading_event_with_context(_ReadingOutputEncodedEvent.with_data(output), event),
    )


def _dispatch_reading_output(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> None:
    output = event.data
    assert isinstance(output, OutputData)
    _dispatch_reading_terminal_output(ctx, instance, event, output)


def _dispatch_reading_failure(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> None:
    failure = event.data
    assert isinstance(failure, FailedEventData)
    _dispatch_reading_terminal_failure(ctx, instance, event, failure)


def _has_reading_input(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_reading_classification(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, _ReadingClassifiedEventData) and _matches_active_reading_operation(instance, event)


def _classified_as_text(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingClassifiedEventData)
        and data.classification.kind == "text"
        and isinstance(data.input.content, str)
        and _matches_active_reading_operation(instance, event)
    )


def _classified_as_image(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingClassifiedEventData)
        and data.classification.kind == "image"
        and isinstance(data.input.content, bytes)
        and _matches_active_reading_operation(instance, event)
    )


def _classified_as_unreadable(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingClassifiedEventData)
        and data.classification.kind == "unreadable"
        and _matches_active_reading_operation(instance, event)
    )


def _has_text_output_candidate(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingOutputCandidateEventData)
        and data.source_kind == "text"
        and _matches_active_reading_operation(instance, event)
    )


def _has_image_output_candidate(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingOutputCandidateEventData)
        and data.source_kind == "image"
        and _matches_active_reading_operation(instance, event)
    )


def _has_unreadable_output_candidate(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, _ReadingOutputCandidateEventData)
        and data.source_kind == "unreadable"
        and _matches_active_reading_operation(instance, event)
    )


def _has_reading_output(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, OutputData) and _matches_active_reading_operation(instance, event)


def _has_reading_stage_failure(ctx: hsm.Context, instance: "Reading", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, FailedEventData) and _matches_active_reading_operation(instance, event)


def _dispatch_invalid_classification_route_failure(
    ctx: hsm.Context,
    instance: "Reading",
    event: hsm.Event[typing.Any],
) -> None:
    failure = FailedEventData(
        stage="classification",
        message="Reading classification route did not match the input payload.",
    )
    _dispatch_reading_terminal_failure(ctx, instance, event, failure)


class Reading(ability.Ability[InputData, OutputData]):
    """Composite ability that focuses on readable text or image input and emits normalized reading output."""

    input_event: typing.ClassVar[hsm.Event[InputData]] = ReadingInputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ReadingOutputEvent
    failed_event: typing.ClassVar[hsm.Event[FailedEventData]] = ReadingFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _visual_classifier: vision.classification.VisualClassification
    _text_decoder: decoding.Decoding[str, str]
    _image_decoder: decoding.Decoding[bytes, str]
    _output_encoder: encoding.Encoding[typing.Any, typing.Any]

    @staticmethod
    def visual_classifier(instance: "Reading") -> vision.classification.VisualClassification:
        return instance._visual_classifier

    @staticmethod
    def text_decoder(instance: "Reading") -> decoding.Decoding[str, str]:
        return instance._text_decoder

    @staticmethod
    def image_decoder(instance: "Reading") -> decoding.Decoding[bytes, str]:
        return instance._image_decoder

    @staticmethod
    def output_encoder(instance: "Reading") -> encoding.Encoding[typing.Any, typing.Any]:
        return instance._output_encoder

    def __init__(
        self,
        *,
        visual_classifier: vision.classification.VisualClassifier,
        text_decoder: decoding.Decoder[str, str],
        image_decoder: decoding.Decoder[bytes, str],
        output_encoder: encoding.Encoder[OutputData, OutputData],
    ) -> None:
        super().__init__()
        self._visual_classifier = vision.classification.VisualClassification(classifier=visual_classifier)
        self._text_decoder = decoding.Decoding(decoder=text_decoder)
        self._image_decoder = decoding.Decoding(decoder=image_decoder)
        self._output_encoder = encoding.Encoding(encoder=output_encoder)
        self._attachment_group = attachment.Group(
            self._visual_classifier,
            self._text_decoder,
            self._image_decoder,
            self._output_encoder,
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Reading",
        hsm.attribute(_READING_ACTIVE_OPERATION_ID_ATTRIBUTE),
        hsm.initial(hsm.target("/Reading/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reading/Unfocused"),
            ),
        ),
        hsm.state(
            "Unfocused",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_reading_input),
                hsm.effect(_start_reading_operation),
                hsm.target("/Reading/Focused/Classifying/Applying"),
            ),
        ),
        hsm.state(
            "Focused",
            hsm.initial(hsm.target("/Reading/Focused/Classifying/Applying")),
            hsm.transition(
                hsm.on(_ReadingStageFailedEvent),
                hsm.guard(_has_reading_stage_failure),
                hsm.effect(_dispatch_reading_failure, _clear_reading_operation),
                hsm.target("/Reading/Unfocused"),
            ),
            hsm.state(
                "Classifying",
                hsm.initial(hsm.target("/Reading/Focused/Classifying/Applying")),
                hsm.state(
                    "Applying",
                    hsm.defer(input_event),
                    hsm.activity(_run_reading_classification),
                    hsm.transition(
                        hsm.on(_ReadingClassificationCompletedEvent),
                        hsm.guard(_has_reading_classification),
                        hsm.target("/Reading/Focused/Classifying/Classified"),
                    ),
                ),
                hsm.choice(
                    "Classified",
                    hsm.transition(
                        hsm.guard(_classified_as_text),
                        hsm.target("/Reading/Focused/DecodingText"),
                    ),
                    hsm.transition(
                        hsm.guard(_classified_as_image),
                        hsm.target("/Reading/Focused/DecodingImage"),
                    ),
                    hsm.transition(
                        hsm.guard(_classified_as_unreadable),
                        hsm.target("/Reading/Focused/PreparingUnreadableOutput"),
                    ),
                    hsm.transition(
                        hsm.effect(_dispatch_invalid_classification_route_failure, _clear_reading_operation),
                        hsm.target("/Reading/Unfocused"),
                    ),
                ),
            ),
            hsm.state(
                "DecodingText",
                hsm.defer(input_event),
                hsm.activity(_run_text_decoding),
                hsm.transition(
                    hsm.on(_ReadingTextDecodedEvent),
                    hsm.guard(_has_text_output_candidate),
                    hsm.target("/Reading/Focused/EncodingOutput"),
                ),
            ),
            hsm.state(
                "DecodingImage",
                hsm.defer(input_event),
                hsm.activity(_run_image_decoding),
                hsm.transition(
                    hsm.on(_ReadingImageDecodedEvent),
                    hsm.guard(_has_image_output_candidate),
                    hsm.target("/Reading/Focused/EncodingOutput"),
                ),
            ),
            hsm.state(
                "PreparingUnreadableOutput",
                hsm.defer(input_event),
                hsm.activity(_run_unreadable_output_preparation),
                hsm.transition(
                    hsm.on(_ReadingUnreadableOutputReadyEvent),
                    hsm.guard(_has_unreadable_output_candidate),
                    hsm.target("/Reading/Focused/EncodingOutput"),
                ),
            ),
            hsm.state(
                "EncodingOutput",
                hsm.defer(input_event),
                hsm.activity(_run_output_encoding),
                hsm.transition(
                    hsm.on(_ReadingOutputEncodedEvent),
                    hsm.guard(_has_reading_output),
                    hsm.effect(_dispatch_reading_output, _clear_reading_operation),
                    hsm.target("/Reading/Unfocused"),
                ),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.entry(_clear_reading_operation),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reading/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reading/Unfocused"),
            ),
        ),
        hsm.state("degraded"),
        hsm.observe(observer),
    )


__all__ = [
    "ReadingFailedEvent",
    "ReadingInputEvent",
    "ReadingOutputEvent",
    "FailedEventData",
    "InputData",
    "InputKind",
    "OutputData",
    "ReadingOutputKind",
    "Reading",
    "ReadingStage",
]
