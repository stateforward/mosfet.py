from .. import ability
from .. import classifying

import abc
import dataclasses
import enum
import typing

import hsm
import bot
import pydantic

from bot.telemetry import observer


class MemoryClassificationRetention(enum.StrEnum):
    """Storage policy recommended for a candidate memory."""

    RETAIN = "retain"
    DISCARD = "discard"
    REVIEW = "review"


class MemoryClassificationKind(enum.StrEnum):
    """Provider-neutral category assigned to a candidate memory."""

    PREFERENCE = "preference"
    FACT = "fact"
    INSTRUCTION = "instruction"
    TASK = "task"
    SUBJECT = "subject"
    SUMMARY = "summary"
    OTHER = "other"


class MemoryClassificationSensitivity(enum.StrEnum):
    """Sensitivity level assigned before a memory is retained or used for retrieval."""

    STANDARD = "standard"
    PRIVATE = "private"
    SENSITIVE = "sensitive"


MemoryEncodedValue: typing.TypeAlias = str | bytes | tuple[float, ...]


class GeneratedMemory(pydantic.BaseModel):
    """Provider-neutral memory candidate ready for encoding and policy review."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "content": "The operator prefers terse handoff notes.",
                    "kind": "preference",
                    "subject_ref": "operator",
                }
            ],
        },
    )

    content: str = pydantic.Field(
        min_length=1,
        description=(
            "Memory candidate content that can be retained by a memory owner after policy checks. This should be "
            "concise and self-contained when it is derived memory; raw turn memory may preserve the decoded source."
        ),
        examples=["The operator prefers terse handoff notes."],
    )
    kind: MemoryClassificationKind | None = pydantic.Field(
        default=None,
        description=(
            "Optional provider-neutral memory kind when the generator can identify it. A separate classification "
            "ability may still review or override this suggestion."
        ),
        examples=[MemoryClassificationKind.PREFERENCE],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional subject reference the generated memory is about or came from. This preserves domain "
            "identity without forcing provider-specific message roles into memory."
        ),
        examples=["operator"],
    )


class EncodedMemory(pydantic.BaseModel):
    """Encoded memory payload ready for a caller-owned memory store or index."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {"format": "text/plain", "value": "The operator prefers terse handoff notes."},
                {"format": "embedding/f32", "value": [0.12, 0.4, -0.08]},
            ],
        },
    )

    format: str = pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral format label for the encoded memory payload, such as text/plain, application/json, "
            "or an embedding format."
        ),
        examples=["text/plain", "embedding/f32"],
    )
    value: MemoryEncodedValue = pydantic.Field(
        description=(
            "Encoded memory payload produced by the injected encoder. Text payloads use strings, binary payloads use "
            "base64 in JSON, and vector payloads use a numeric array."
        ),
        examples=["The operator prefers terse handoff notes.", [0.12, 0.4, -0.08]],
    )


class InputData(pydantic.BaseModel):
    """CandidateData memory content and optional local context to classify."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "candidate": "The operator prefers terse handoff notes.",
                    "context": "Classify whether this source observation should be retained for future collaboration.",
                    "subject_ref": "operator",
                }
            ],
        },
    )

    candidate: str = pydantic.Field(
        min_length=1,
        description=(
            "CandidateData memory content to classify. This should be the smallest useful content span, not an entire "
            "source transcript unless the transcript itself is the memory candidate."
        ),
        examples=["The operator prefers terse handoff notes."],
    )
    context: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional local task, source, or retrieval context that helps judge the candidate memory. "
            "Callers should keep this focused on the classification decision."
        ),
        examples=["Classify whether this source observation should be retained for future collaboration."],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional caller-supplied subject reference for the candidate memory. This is domain identity context, "
            "not a provider role such as assistant or user."
        ),
        examples=["operator"],
    )


class OutputData(pydantic.BaseModel):
    """Provider-neutral classification result for a candidate memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "retention": "retain",
                    "kind": "preference",
                    "sensitivity": "standard",
                    "confidence": 0.93,
                    "rationale": "The candidate describes a durable collaboration preference.",
                }
            ],
        },
    )

    retention: MemoryClassificationRetention = pydantic.Field(
        description=(
            "Recommended storage policy for the candidate memory. RetainData means the content is useful durable context, "
            "discard means it should not be stored, and review means a policy owner should decide before storage."
        ),
        examples=[MemoryClassificationRetention.RETAIN],
    )
    kind: MemoryClassificationKind = pydantic.Field(
        description="Provider-neutral category that describes the main kind of memory being classified.",
        examples=[MemoryClassificationKind.PREFERENCE],
    )
    sensitivity: MemoryClassificationSensitivity = pydantic.Field(
        description=(
            "Sensitivity level for the candidate before retention or retrieval. This supports policy decisions without "
            "making the classification ability own memory lifecycle."
        ),
        examples=[MemoryClassificationSensitivity.STANDARD],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional provider confidence for the classification decision, normalized from 0.0 to 1.0 when available."
        ),
        examples=[0.93],
    )
    rationale: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional concise explanation of the classification result. This is an output payload field for callers, "
            "not telemetry metadata."
        ),
        examples=["The candidate describes a durable collaboration preference."],
    )


class MemoryClassifier(classifying.Classifier[InputData, OutputData], abc.ABC):
    """Classifier that decides memory retention, category, and sensitivity."""


_MemoryClassificationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.memory.classification.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_MemoryClassificationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.classification.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _dispatch_memory_classification_output(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> None:
    output = event.data
    assert isinstance(output, OutputData)
    terminal = dataclasses.replace(
        instance.output_event.with_data(output),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_memory_classification_failure(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> None:
    data = event.data
    assert isinstance(data, ability.FailureData)
    terminal = dataclasses.replace(
        instance.failed_event.with_data(data),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _has_memory_classification_input(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_memory_classification_output(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    """Single in-flight apply (inputs deferred); completion already carries op id."""

    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_memory_classification_output(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_memory_classification_failure(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


def _dispatch_invalid_memory_classification_output_failure(
    ctx: hsm.Context,
    instance: "MemoryClassification",
    event: hsm.Event[typing.Any],
) -> None:
    failure = ability.FailureData(message="MemoryClassification produced output that does not match its output schema.")
    terminal = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class MemoryClassification(classifying.Classifying[InputData, OutputData]):
    """Ability to classify candidate memory content before storage or retrieval."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.memory.classification.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.memory.classification.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _MemoryClassificationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _MemoryClassificationApplyFailedEvent

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "MemoryClassification",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_memory_classification_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_MemoryClassificationApplyCompletedEvent),
                hsm.guard(_has_memory_classification_output),
                hsm.effect(_dispatch_memory_classification_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryClassificationApplyCompletedEvent),
                hsm.guard(_has_invalid_memory_classification_output),
                hsm.effect(_dispatch_invalid_memory_classification_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryClassificationApplyFailedEvent),
                hsm.guard(_has_memory_classification_failure),
                hsm.effect(_dispatch_memory_classification_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )
