from . import store
from . import consolidation
from . import classification
from .. import ability
from .. import decoding
from .. import encoding
from .. import processing

import asyncio
import dataclasses
import datetime
import typing
import uuid

import hsm
import mosfet

from mosfet.protocols import attachment
import pydantic


_STAGE_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_AssociativeMemoryChildrenAttachedEvent = hsm.Event[object](
    name="bot.ability.memory.associative.children.attached",
    kind=hsm.CompletionEventKind,
    schema=object,
)


class Link(pydantic.BaseModel):
    """Graph-ready association between two memory subjects, concepts, or facts."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "source_ref": "operator",
                    "target_ref": "preference:concise-handoff",
                    "relation": "prefers",
                    "weight": 0.94,
                    "evidence_refs": ["active-task"],
                }
            ],
        },
    )

    source_ref: str = pydantic.Field(
        min_length=1,
        description=(
            "Stable graph node reference for the source side of the association. This is a knowledge-store identity, "
            "not a live object reference."
        ),
        examples=["operator"],
    )
    target_ref: str = pydantic.Field(
        min_length=1,
        description=(
            "Stable graph node reference for the target side of the association. Knowledge stores may create or "
            "merge the target node later."
        ),
        examples=["preference:concise-handoff"],
    )
    relation: str = pydantic.Field(
        min_length=1,
        description="Provider-neutral relationship label suitable for a knowledge graph edge.",
        examples=["prefers"],
    )
    weight: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional normalized association strength. Omit this when the processor cannot provide a meaningful "
            "confidence or salience score."
        ),
        examples=[0.94],
    )
    evidence_refs: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "Stable references to source records, contexts, or retrieval windows that support this association. "
            "These are evidence pointers for the knowledge-store owner, not raw memory contents."
        ),
        examples=[["active-task"]],
    )


class InputData(pydantic.BaseModel):
    """Retained memory records and graph-store context to turn into associations."""

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
                        }
                    ],
                    "context": "Build graph associations for a knowledge store.",
                    "context_ref": "active-task",
                    "subject_ref": "operator",
                    "store_ref": "knowledge-store",
                    "graph_ref": "operator-preferences",
                }
            ],
        },
    )

    records: tuple[store.MemoryRecord, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Retained memory records to associate. These are decoded before processing so graph association decisions "
            "are based on memory content, not raw store bytes or embeddings."
        ),
    )
    context: str | None = pydantic.Field(
        default=None,
        description="Optional local task, source, or retrieval context for the association processor.",
        examples=["Build graph associations for a knowledge store."],
    )
    context_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional context identity shared by the source memories being associated.",
        examples=["active-task"],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional subject identity for the primary memory association target.",
        examples=["operator"],
    )
    store_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional stable reference for the knowledge store that may later retain the association output. The "
            "ability does not open or mutate that store."
        ),
        examples=["knowledge-store"],
    )
    graph_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=("Optional stable graph, namespace, or tenant reference for downstream graph-database retention."),
        examples=["operator-preferences"],
    )


class AssociationData(pydantic.BaseModel):
    """DecodedData memory sources prepared for graph-oriented processing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "sources": [
                        {
                            "record": {
                                "memory_id": "mem-2",
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
                    "context": "Build graph associations for a knowledge store.",
                    "context_ref": "active-task",
                    "subject_ref": "operator",
                    "store_ref": "knowledge-store",
                    "graph_ref": "operator-preferences",
                }
            ],
        },
    )

    sources: tuple[consolidation.MemoryConsolidationSource, ...] = pydantic.Field(
        min_length=1,
        description=(
            "DecodedData source records available to the processing ability. The processor should derive graph-ready "
            "associations from these sources without retaining or mutating any store."
        ),
    )
    context: str | None = pydantic.Field(
        default=None,
        description="Optional local task, source, or retrieval context carried from the association request.",
        examples=["Build graph associations for a knowledge store."],
    )
    context_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional context identity carried from the association request.",
        examples=["active-task"],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional subject identity carried from the association request.",
        examples=["operator"],
    )
    store_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional stable reference for the downstream knowledge store.",
        examples=["knowledge-store"],
    )
    graph_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional stable graph, namespace, or tenant reference for downstream graph-database retention.",
        examples=["operator-preferences"],
    )


class LinkedData(pydantic.BaseModel):
    """Memory candidate and graph-ready association links selected by a processing ability."""

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
                    "links": [
                        {
                            "source_ref": "operator",
                            "target_ref": "preference:concise-handoff",
                            "relation": "prefers",
                            "weight": 0.94,
                            "evidence_refs": ["active-task"],
                        }
                    ],
                }
            ],
        },
    )

    memory: classification.GeneratedMemory = pydantic.Field(
        description=(
            "Stable memory candidate selected by the processing ability before encoding. This candidate may become a "
            "node payload in a knowledge store, but this ability does not retain it."
        ),
    )
    links: tuple[Link, ...] = pydantic.Field(
        default=(),
        description=(
            "Graph-ready association links derived from the decoded sources. Retention, node creation, and graph "
            "mutation remain downstream responsibilities."
        ),
    )


class OutputData(pydantic.BaseModel):
    """Encoded memory candidate and graph-ready association links derived from source memories."""

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
                                "memory_id": "mem-3",
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
                    "links": [
                        {
                            "source_ref": "operator",
                            "target_ref": "preference:concise-handoff",
                            "relation": "prefers",
                            "weight": 0.94,
                            "evidence_refs": ["active-task"],
                        }
                    ],
                }
            ],
        },
    )

    memory: classification.GeneratedMemory = pydantic.Field(
        description="Stable memory candidate selected from the decoded sources.",
    )
    encoded: classification.EncodedMemory = pydantic.Field(
        description=(
            "Encoded representation of the associative memory candidate. Knowledge-store mutation remains owned by "
            "the caller or memory lifecycle."
        ),
    )
    sources: tuple[consolidation.MemoryConsolidationSource, ...] = pydantic.Field(
        min_length=1,
        description="DecodedData source records that shaped the associative memory output.",
    )
    links: tuple[Link, ...] = pydantic.Field(
        default=(),
        description="Graph-ready links derived from the decoded source records.",
    )


class _AssociativeMemoryDecodedEventData(pydantic.BaseModel):
    """Private phase payload carrying decoded source memories."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: InputData
    sources: tuple[consolidation.MemoryConsolidationSource, ...]


class _AssociativeMemoryProcessingEventData(pydantic.BaseModel):
    """Private phase payload carrying processor output with decoded sources."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: InputData
    sources: tuple[consolidation.MemoryConsolidationSource, ...]
    processing: LinkedData


_AssociativeMemorySourcesDecodedEvent = hsm.Event[_AssociativeMemoryDecodedEventData](
    name="bot.ability.memory.associative.sources.decoded",
    kind=hsm.CompletionEventKind,
    schema=_AssociativeMemoryDecodedEventData,
)
_AssociativeMemoryProcessingCompletedEvent = hsm.Event[_AssociativeMemoryProcessingEventData](
    name="bot.ability.memory.associative.processing.completed",
    kind=hsm.CompletionEventKind,
    schema=_AssociativeMemoryProcessingEventData,
)
_AssociativeMemoryApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.memory.associative.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_AssociativeMemoryApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.associative.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _associative_memory_event_with_context(
    event: hsm.Event[typing.Any],
    source: hsm.Event[typing.Any],
) -> hsm.Event[typing.Any]:
    return dataclasses.replace(
        event,
        id=source.id or None,
        source=source.source,
        target=source.target,
        metadata=dict(source.metadata),
    )


def _public_associative_memory_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Pass-through telemetry metadata (telemetry only; phase payloads ride completions)."""

    return dict(metadata)


def _associative_turn_id(event: hsm.Event[typing.Any]) -> str | None:
    return event.id if event.id else None


def _dispatch_associative_memory_output(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    output = event.data
    assert isinstance(output, OutputData)
    terminal = dataclasses.replace(
        instance.output_event.with_data(output),
        id=event.id or None,
        metadata=_public_associative_memory_metadata(dict(event.metadata)),
        source=hsm.id(instance),
        target=(
            event.source
            if event.target == hsm.id(instance) and event.source and event.source != hsm.id(instance)
            else None
        ),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_associative_memory_failure(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    data = event.data
    assert isinstance(data, ability.FailureData)
    terminal = dataclasses.replace(
        instance.failed_event.with_data(data),
        id=event.id or None,
        metadata=_public_associative_memory_metadata(dict(event.metadata)),
        source=hsm.id(instance),
        target=(
            event.source
            if event.target == hsm.id(instance) and event.source and event.source != hsm.id(instance)
            else None
        ),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _has_associative_memory_input(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_associative_memory_output(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    """Single in-flight turn (inputs deferred); completion already carries op id."""

    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_associative_memory_sources(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, _AssociativeMemoryDecodedEventData)


def _has_associative_memory_processing(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, _AssociativeMemoryProcessingEventData)


def _has_invalid_associative_memory_output(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_associative_memory_failure(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


def _dispatch_invalid_associative_memory_output_failure(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    failure = ability.FailureData(message="AssociativeMemory produced output that does not match its output schema.")
    terminal = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=_public_associative_memory_metadata(dict(event.metadata)),
        source=hsm.id(instance),
        target=(
            event.source
            if event.target == hsm.id(instance) and event.source and event.source != hsm.id(instance)
            else None
        ),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


async def _attach_associative_memory_children_activity(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    await AssociativeMemory.attach_subordinate_abilities(ctx, instance, event)


def _detach_associative_memory_children_on_detach(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    AssociativeMemory.detach_subordinate_abilities_on_detach(instance, event)


async def _decode_associative_memory_sources(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    input = event.data
    assert isinstance(input, InputData)
    try:
        source_list = [
            consolidation.MemoryConsolidationSource(
                record=record,
                decoded=classification.GeneratedMemory(
                    content=record.content,
                    kind=(classification.MemoryClassificationKind(record.kind) if record.kind is not None else None),
                    subject_ref=record.subject_ref,
                ),
            )
            for record in input.records
        ]
    except Exception as error:
        _ = hsm.dispatch(
            ctx,
            instance,
            _associative_memory_event_with_context(
                _AssociativeMemoryApplyFailedEvent.with_data(ability.FailureData(message=str(error))),
                event,
            ),
        )
        return
    decoded = _AssociativeMemoryDecodedEventData(input=input, sources=tuple(source_list))
    _ = hsm.dispatch(
        ctx,
        instance,
        _associative_memory_event_with_context(_AssociativeMemorySourcesDecodedEvent.with_data(decoded), event),
    )


async def _run_associative_memory_processor(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    """Processor activity: decoded payload is entry-event local (HSM-COMPLETION-001)."""

    decoded = event.data
    assert isinstance(decoded, _AssociativeMemoryDecodedEventData)
    operation_id = _associative_turn_id(event)
    public_metadata = _public_associative_memory_metadata(dict(event.metadata))
    if operation_id is None:
        _ = hsm.dispatch(
            ctx,
            instance,
            _associative_memory_event_with_context(
                _AssociativeMemoryApplyFailedEvent.with_data(
                    ability.FailureData(message="AssociativeMemory refused processing without operation id.")
                ),
                event,
            ),
        )
        return
    child = instance.processor
    child_operation_id = f"{operation_id}:processing:{uuid.uuid4().hex}"
    child_request = dataclasses.replace(
        child.input_event.with_data_and_id(
            processing.InputData(
                input=AssociationData(
                    sources=decoded.sources,
                    context=decoded.input.context,
                    context_ref=decoded.input.context_ref,
                    subject_ref=decoded.input.subject_ref,
                    store_ref=decoded.input.store_ref,
                    graph_ref=decoded.input.graph_ref,
                )
            ),
            child_operation_id,
        ),
        metadata=public_metadata,
    )
    try:
        terminal = await ability.run_terminal_operation(
            instance.context(),
            child=child,
            request=child_request,
            terminals=(child.output_event, child.failed_event),
            timeout=_STAGE_OPERATION_TIMEOUT,
        )
    except asyncio.CancelledError:
        cancel_token = uuid.uuid4().hex
        await asyncio.shield(
            hsm.dispatch(
                instance.context(),
                child,
                dataclasses.replace(
                    processing.CancelEvent.with_data(
                        processing.CancelData(
                            operation_id=child_operation_id,
                            token=cancel_token,
                            parent_operation_id=operation_id,
                        )
                    ),
                    id=child_operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(child),
                    metadata=public_metadata,
                ),
            )
        )
        raise
    if terminal.name == child.failed_event.name:
        message = getattr(terminal.data, "message", "AssociativeMemory processor failed.")
        _ = hsm.dispatch(
            ctx,
            instance,
            _associative_memory_event_with_context(
                _AssociativeMemoryApplyFailedEvent.with_data(ability.FailureData(message=str(message))),
                event,
            ),
        )
        return
    linked = terminal.data
    if not isinstance(linked, LinkedData):
        _ = hsm.dispatch(
            ctx,
            instance,
            _associative_memory_event_with_context(
                _AssociativeMemoryApplyFailedEvent.with_data(
                    ability.FailureData(
                        message="AssociativeMemory processor produced output that does not match its output schema."
                    )
                ),
                event,
            ),
        )
        return
    data = _AssociativeMemoryProcessingEventData(
        input=decoded.input,
        sources=decoded.sources,
        processing=linked,
    )
    _ = hsm.dispatch(
        ctx,
        instance,
        _associative_memory_event_with_context(
            _AssociativeMemoryProcessingCompletedEvent.with_data(data),
            event,
        ),
    )


async def _encode_associative_memory_output(
    ctx: hsm.Context,
    instance: "AssociativeMemory",
    event: hsm.Event[typing.Any],
) -> None:
    data = event.data
    assert isinstance(data, _AssociativeMemoryProcessingEventData)
    try:
        encoded = await instance.encoder.encode(data.processing.memory)
        output = OutputData(
            memory=data.processing.memory,
            encoded=encoded,
            sources=data.sources,
            links=data.processing.links,
        )
    except Exception as error:
        _ = hsm.dispatch(
            ctx,
            instance,
            _associative_memory_event_with_context(
                _AssociativeMemoryApplyFailedEvent.with_data(ability.FailureData(message=str(error))),
                event,
            ),
        )
        return
    _ = hsm.dispatch(
        ctx,
        instance,
        _associative_memory_event_with_context(_AssociativeMemoryApplyCompletedEvent.with_data(output), event),
    )


class AssociativeMemory(ability.Ability[InputData, OutputData]):
    """Ability to derive encoded memory and graph-ready links from retained memories."""

    processor: processing.Processing
    encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory]
    decoder: decoding.Decoder[classification.EncodedMemory, classification.GeneratedMemory]
    _subordinate_abilities: tuple[ability.Ability[typing.Any, typing.Any], ...]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.memory.associative.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.memory.associative.output",
        schema=OutputData,
    )

    @staticmethod
    async def attach_subordinate_abilities(
        ctx: hsm.Context,
        instance: "AssociativeMemory",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        for child in instance._subordinate_abilities:
            _ = await child.attach(
                instance.context(),
                dataclasses.replace(
                    attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                    source=hsm.id(instance),
                ),
            )
        _ = hsm.dispatch(ctx, instance, _AssociativeMemoryChildrenAttachedEvent.with_data(None))

    @staticmethod
    def detach_subordinate_abilities_on_detach(instance: "AssociativeMemory", event: hsm.Event[typing.Any]) -> None:
        # Shared exit fires on every exit; cascade only for typed DetachEvent payloads.
        if not isinstance(event.data, attachment.DetachData):
            return
        for child in instance._subordinate_abilities:
            _ = child.detach(
                instance.context(),
                dataclasses.replace(
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                    source=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    def __init__(
        self,
        *,
        processor: processing.Processing,
        encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory],
        decoder: decoding.Decoder[classification.EncodedMemory, classification.GeneratedMemory],
    ) -> None:
        super().__init__()
        self.processor = processor
        self.encoder = encoder
        self.decoder = decoder
        self._subordinate_abilities = (typing.cast(ability.Ability[typing.Any, typing.Any], self.processor),)

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _AssociativeMemoryApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _AssociativeMemoryApplyFailedEvent

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "AssociativeMemory",
        hsm.initial(hsm.target("initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_attach_associative_memory_children_activity),
            hsm.exit(_detach_associative_memory_children_on_detach),
            hsm.transition(
                hsm.on(_AssociativeMemoryChildrenAttachedEvent),
                hsm.target("../idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.exit(_detach_associative_memory_children_on_detach),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_associative_memory_input),
                hsm.target("../decoding"),
            ),
        ),
        hsm.state(
            "decoding",
            hsm.defer(input_event),
            hsm.activity(_decode_associative_memory_sources),
            hsm.exit(_detach_associative_memory_children_on_detach),
            hsm.transition(
                hsm.on(_AssociativeMemorySourcesDecodedEvent),
                hsm.guard(_has_associative_memory_sources),
                hsm.target("../processing"),
            ),
            hsm.transition(
                hsm.on(_AssociativeMemoryApplyFailedEvent),
                hsm.guard(_has_associative_memory_failure),
                hsm.effect(_dispatch_associative_memory_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.state(
            "processing",
            hsm.defer(input_event),
            hsm.activity(_run_associative_memory_processor),
            hsm.exit(_detach_associative_memory_children_on_detach),
            hsm.transition(
                hsm.on(_AssociativeMemoryProcessingCompletedEvent),
                hsm.guard(_has_associative_memory_processing),
                hsm.target("../encoding"),
            ),
            hsm.transition(
                hsm.on(_AssociativeMemoryApplyFailedEvent),
                hsm.guard(_has_associative_memory_failure),
                hsm.effect(_dispatch_associative_memory_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.state(
            "encoding",
            hsm.defer(input_event),
            hsm.activity(_encode_associative_memory_output),
            hsm.exit(_detach_associative_memory_children_on_detach),
            hsm.transition(
                hsm.on(_AssociativeMemoryApplyCompletedEvent),
                hsm.guard(_has_associative_memory_output),
                hsm.effect(_dispatch_associative_memory_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_AssociativeMemoryApplyCompletedEvent),
                hsm.guard(_has_invalid_associative_memory_output),
                hsm.effect(_dispatch_invalid_associative_memory_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_AssociativeMemoryApplyFailedEvent),
                hsm.guard(_has_associative_memory_failure),
                hsm.effect(_dispatch_associative_memory_failure),
                hsm.target("../idle"),
            ),
        ),
    )


__all__ = [
    "AssociativeMemory",
    "InputData",
    "Link",
    "OutputData",
    "AssociationData",
    "LinkedData",
]
