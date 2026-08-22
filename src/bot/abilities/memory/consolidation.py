from . import store
from . import classification
from .. import ability
from .. import decoding
from .. import encoding
from .. import generative

import abc
import dataclasses
import typing

import hsm
import bot
import pydantic

from bot.telemetry import observer


class MemoryConsolidationSource(pydantic.BaseModel):
    """A retained memory record and the decoded memory content used during consolidation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "record": {
                        "memory_id": "mem-1",
                        "scope": "short_term",
                        "context_ref": "active-task",
                        "subject_ref": "operator",
                        "content": "Gabe prefers terse handoff notes.",
                        "kind": "preference",
                    },
                    "decoded": {
                        "content": "Gabe prefers terse handoff notes.",
                        "kind": "preference",
                        "subject_ref": "operator",
                    },
                }
            ],
        },
    )

    record: store.MemoryRecord = pydantic.Field(
        description="Original memory record selected as source material for consolidation.",
    )
    decoded: classification.GeneratedMemory = pydantic.Field(
        description=(
            "DecodedData memory content produced from the record's encoded payload. The consolidation generator consumes "
            "decoded memories rather than raw store bytes or embeddings."
        ),
    )


class InputData(pydantic.BaseModel):
    """Retained memory records and local context to consolidate into a stable memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {
                            "memory_id": "mem-1",
                            "scope": "short_term",
                            "context_ref": "active-task",
                            "subject_ref": "operator",
                            "content": "Gabe prefers terse handoff notes.",
                            "kind": "preference",
                        },
                        {
                            "memory_id": "mem-2",
                            "scope": "short_term",
                            "context_ref": "active-task",
                            "subject_ref": "operator",
                            "content": "Gabe asked for concise final reports.",
                            "kind": "preference",
                        },
                    ],
                    "context": "Consolidate repeated collaboration preferences after an interaction.",
                    "context_ref": "active-task",
                    "subject_ref": "operator",
                }
            ],
        },
    )

    records: tuple[store.MemoryRecord, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Retained memory records to consolidate. These are usually recently recalled or short-term records that "
            "should be integrated into a smaller stable memory, similar to how repeated human observations become "
            "durable context."
        ),
    )
    context: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional local task, source, or retrieval context that helps the generator decide what stable memory "
            "should survive consolidation."
        ),
        examples=["Consolidate repeated collaboration preferences after an interaction."],
    )
    context_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional context identity shared by the source memories being consolidated.",
        examples=["active-task"],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional subject identity for the consolidated memory.",
        examples=["operator"],
    )


class SourceData(pydantic.BaseModel):
    """DecodedData source memories prepared for a consolidation generator."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "sources": [
                        {
                            "record": {
                                "memory_id": "mem-src-1",
                                "scope": "short_term",
                                "context_ref": "active-task",
                                "subject_ref": "operator",
                                "content": "Gabe prefers terse handoff notes.",
                                "kind": "preference",
                            },
                            "decoded": {
                                "content": "Gabe prefers terse handoff notes.",
                                "kind": "preference",
                                "subject_ref": "operator",
                            },
                        }
                    ],
                    "context": "Consolidate repeated collaboration preferences after an interaction.",
                    "context_ref": "active-task",
                    "subject_ref": "operator",
                }
            ],
        },
    )

    sources: tuple[MemoryConsolidationSource, ...] = pydantic.Field(
        min_length=1,
        description=(
            "DecodedData source memories available to the generator. The generator should integrate repeated, related, "
            "or salient records into one stable candidate rather than copying every source verbatim."
        ),
    )
    context: str | None = pydantic.Field(
        default=None,
        description="Optional local task, source, or retrieval context carried from the consolidation request.",
        examples=["Consolidate repeated collaboration preferences after an interaction."],
    )
    context_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional context identity carried from the consolidation request.",
        examples=["active-task"],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional subject identity carried from the consolidation request.",
        examples=["operator"],
    )


class OutputData(pydantic.BaseModel):
    """Consolidated memory candidate, encoded representation, and decoded source evidence."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "memory": {
                        "content": "The operator prefers concise handoff and final notes.",
                        "kind": "preference",
                        "subject_ref": "operator",
                    },
                    "encoded": {
                        "format": "text/plain",
                        "value": "The operator prefers concise handoff and final notes.",
                    },
                    "sources": [
                        {
                            "record": {
                                "memory_id": "mem-src-2",
                                "scope": "short_term",
                                "context_ref": "active-task",
                                "subject_ref": "operator",
                                "content": "Gabe prefers terse handoff notes.",
                                "kind": "preference",
                            },
                            "decoded": {
                                "content": "Gabe prefers terse handoff notes.",
                                "kind": "preference",
                                "subject_ref": "operator",
                            },
                        }
                    ],
                }
            ],
        },
    )

    memory: classification.GeneratedMemory = pydantic.Field(
        description="Stable consolidated memory candidate produced from the decoded source records.",
    )
    encoded: classification.EncodedMemory = pydantic.Field(
        description=(
            "Encoded representation of the consolidated memory. Retention and promotion remain owned by the caller's "
            "memory lifecycle, not by this ability."
        ),
    )
    sources: tuple[MemoryConsolidationSource, ...] = pydantic.Field(
        min_length=1,
        description="DecodedData source records that shaped the consolidated memory.",
    )


class MemoryConsolidationGenerator(generative.Generator[SourceData, classification.GeneratedMemory], abc.ABC):
    """Generator that integrates decoded memory sources into one stable memory."""


_MemoryConsolidationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.memory.consolidation.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_MemoryConsolidationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.consolidation.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _dispatch_memory_consolidation_output(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
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


def _dispatch_memory_consolidation_failure(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
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


def _has_memory_consolidation_input(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_memory_consolidation_output(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
    event: hsm.Event[typing.Any],
) -> bool:
    """Single in-flight apply (inputs deferred); completion already carries op id."""

    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_memory_consolidation_output(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_memory_consolidation_failure(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


def _dispatch_invalid_memory_consolidation_output_failure(
    ctx: hsm.Context,
    instance: "MemoryConsolidation",
    event: hsm.Event[typing.Any],
) -> None:
    failure = ability.FailureData(message="MemoryConsolidation produced output that does not match its output schema.")
    terminal = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class MemoryConsolidation(ability.Ability[InputData, OutputData]):
    """Ability to consolidate recalled memories into a stable encoded memory candidate."""

    generator: MemoryConsolidationGenerator
    encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory]
    decoder: decoding.Decoder[classification.EncodedMemory, classification.GeneratedMemory]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.memory.consolidation.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.memory.consolidation.output",
        schema=OutputData,
    )

    def __init__(
        self,
        *,
        generator: MemoryConsolidationGenerator,
        encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory],
        decoder: decoding.Decoder[classification.EncodedMemory, classification.GeneratedMemory],
    ) -> None:
        super().__init__()
        self.generator = generator
        self.encoder = encoder
        self.decoder = decoder

    async def _apply(self, ctx: hsm.Context, input: InputData) -> OutputData:
        del ctx
        source_list: list[MemoryConsolidationSource] = []
        for record in input.records:
            source_list.append(
                MemoryConsolidationSource(
                    record=record,
                    decoded=classification.GeneratedMemory(
                        content=record.content,
                        kind=(
                            classification.MemoryClassificationKind(record.kind) if record.kind is not None else None
                        ),
                        subject_ref=record.subject_ref,
                    ),
                )
            )
        sources = tuple(source_list)
        memory = await self.generator.generate(
            SourceData(
                sources=sources,
                context=input.context,
                context_ref=input.context_ref,
                subject_ref=input.subject_ref,
            )
        )
        encoded = await self.encoder.encode(memory)
        return OutputData(memory=memory, encoded=encoded, sources=sources)

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _MemoryConsolidationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _MemoryConsolidationApplyFailedEvent

    @staticmethod
    async def _run_behavior_activity(
        ctx: hsm.Context,
        instance: "MemoryConsolidation",
        event: hsm.Event[InputData],
    ) -> None:
        input = typing.cast(InputData, event.data)
        try:
            output = await instance._apply(ctx, input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                instance._apply_completed_event.with_data(output),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "MemoryConsolidation",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_memory_consolidation_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_behavior_activity),
            hsm.transition(
                hsm.on(_MemoryConsolidationApplyCompletedEvent),
                hsm.guard(_has_memory_consolidation_output),
                hsm.effect(_dispatch_memory_consolidation_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryConsolidationApplyCompletedEvent),
                hsm.guard(_has_invalid_memory_consolidation_output),
                hsm.effect(_dispatch_invalid_memory_consolidation_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryConsolidationApplyFailedEvent),
                hsm.guard(_has_memory_consolidation_failure),
                hsm.effect(_dispatch_memory_consolidation_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "MemoryConsolidation",
    "SourceData",
    "MemoryConsolidationGenerator",
    "InputData",
    "OutputData",
    "MemoryConsolidationSource",
]
