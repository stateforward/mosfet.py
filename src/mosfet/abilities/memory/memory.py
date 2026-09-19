"""Memory ability: SQL-transaction store apply, plus optional MemoryGeneration.

Memory apply is one transaction (query, store, or both). Generation is a separate
ability that produces content for INSERT parameters—not an encode/decode store pipeline.
"""

from . import classification
from . import store
from .. import ability
from .. import generative
from .. import encoding

import abc
import dataclasses
import typing

import hsm
import mosfet
import pydantic


# Re-export store transaction contract as the Memory ability surface.
Statement = store.Statement
InputData = store.InputData
OutputData = store.OutputData
Row = store.Row
StatementResult = store.StatementResult
MemoryRecord = store.MemoryRecord
MEMORY_TABLE = store.MEMORY_TABLE


class SourceData(pydantic.BaseModel):
    """Decoded source content and optional context used to generate a memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Input for MemoryGeneration: decoded turn text plus optional context.",
            "examples": [
                {
                    "decoded": "Gabe prefers terse handoff notes.",
                    "context": "Generate a durable collaboration memory from the decoded turn.",
                    "subject_ref": "operator",
                }
            ],
        },
    )

    decoded: str = pydantic.Field(
        min_length=1,
        description="Decoded input content to turn into a candidate memory.",
        examples=["Gabe prefers terse handoff notes."],
    )
    context: str | None = pydantic.Field(
        default=None,
        description="Optional task or retrieval context for the generator.",
        examples=["Generate a durable collaboration memory from the decoded turn."],
    )
    subject_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional subject identity associated with the decoded input.",
        examples=["operator"],
    )


class CandidateData(pydantic.BaseModel):
    """Structured memory candidate produced by generation for callers to INSERT via Memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Generation output. Callers map memory.content (and optional encoded packaging) into "
                "SQL INSERT parameters on Memory; generation does not commit storage."
            ),
            "examples": [
                {
                    "memory": {
                        "content": "The operator prefers terse handoff notes.",
                        "kind": "preference",
                        "subject_ref": "operator",
                    },
                    "encoded": {
                        "format": "text/plain",
                        "value": "The operator prefers terse handoff notes.",
                    },
                }
            ],
        },
    )

    memory: classification.GeneratedMemory = pydantic.Field(
        description="Structured memory candidate available for inspection and SQL INSERT.",
    )
    encoded: classification.EncodedMemory | None = pydantic.Field(
        default=None,
        description=(
            "Optional encoded packaging of the candidate when a generator encoder is configured. "
            "Storage is always via Memory SQL apply, not via this field alone."
        ),
    )


class MemoryGenerator(generative.Generator[SourceData, classification.GeneratedMemory], abc.ABC):
    """Generator that creates a structured memory from decoded input."""


_MemoryGenerationApplyCompletedEvent = hsm.Event[CandidateData](
    name="bot.ability.memory.generation.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=CandidateData,
)
_MemoryGenerationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.generation.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class MemoryGeneration(ability.Ability[SourceData, CandidateData]):
    """Generate a structured memory candidate for callers to store via Memory SQL apply."""

    generator: MemoryGenerator
    encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory] | None
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = SourceData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = CandidateData
    input_event: typing.ClassVar[hsm.Event[SourceData]] = hsm.Event[SourceData](
        name="bot.ability.memory.generation.input",
        schema=SourceData,
    )
    output_event: typing.ClassVar[hsm.Event[CandidateData]] = hsm.Event[CandidateData](
        name="bot.ability.memory.generation.output",
        schema=CandidateData,
    )
    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _MemoryGenerationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _MemoryGenerationApplyFailedEvent

    def __init__(
        self,
        *,
        generator: MemoryGenerator,
        encoder: encoding.Encoder[classification.GeneratedMemory, classification.EncodedMemory] | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.encoder = encoder

    async def _apply(self, ctx: hsm.Context, input: SourceData) -> CandidateData:
        del ctx
        memory = await self.generator.generate(input)
        encoded = await self.encoder.encode(memory) if self.encoder is not None else None
        return CandidateData(memory=memory, encoded=encoded)

    @staticmethod
    async def _run_behavior_activity(
        ctx: hsm.Context,
        instance: "MemoryGeneration",
        event: hsm.Event[SourceData],
    ) -> None:
        input = typing.cast(SourceData, event.data)
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

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, SourceData)

    @staticmethod
    def _has_output(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, CandidateData)

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, CandidateData)
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _dispatch_invalid_output(ctx: hsm.Context, instance: "MemoryGeneration", event: hsm.Event[typing.Any]) -> None:
        failure = ability.FailureData(message="MemoryGeneration produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "MemoryGeneration",
        hsm.initial(hsm.target("/MemoryGeneration/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.target("/MemoryGeneration/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_behavior_activity),
            hsm.transition(
                hsm.on(_MemoryGenerationApplyCompletedEvent),
                hsm.guard(_has_output),
                hsm.effect(_dispatch_output),
                hsm.target("/MemoryGeneration/idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryGenerationApplyCompletedEvent),
                hsm.effect(_dispatch_invalid_output),
                hsm.target("/MemoryGeneration/idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryGenerationApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/MemoryGeneration/idle"),
            ),
        ),
    )


def memory_model(*, name: str, input_event: hsm.Event[InputData]) -> hsm.Model:
    """SQL-transaction apply topology for Memory and short/long-term variants."""

    root = f"/{name}"
    completed = hsm.Event[OutputData](
        name=f"bot.ability.memory.{name.lower()}.apply.completed",
        kind=hsm.CompletionEventKind,
        schema=OutputData,
    )
    failed = hsm.Event[ability.FailureData](
        name=f"bot.ability.memory.{name.lower()}.apply.failed",
        kind=hsm.ErrorEventKind,
        schema=ability.FailureData,
    )

    def has_input(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    def has_output(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, OutputData)

    def has_failure(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    def dispatch_output(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, OutputData)
        requester = event.source if event.source and event.source != hsm.id(instance) else ""
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=requester,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    def dispatch_failure(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        requester = event.source if event.source and event.source != hsm.id(instance) else ""
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=requester,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    async def run_transaction(
        ctx: hsm.Context,
        instance: "Memory",
        event: hsm.Event[InputData],
    ) -> None:
        data = event.data
        assert isinstance(data, InputData)
        try:
            output = instance.execute(data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    failed.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    source=event.source,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                completed.with_data(output),
                id=event.id or None,
                source=event.source,
                metadata=dict(event.metadata),
            ),
        )

    return mosfet.define(
        name,
        hsm.initial(hsm.target(f"{root}/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(has_input),
                hsm.target(f"{root}/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(run_transaction),
            hsm.transition(
                hsm.on(completed),
                hsm.guard(has_output),
                hsm.effect(dispatch_output),
                hsm.target(f"{root}/idle"),
            ),
            hsm.transition(
                hsm.on(failed),
                hsm.guard(has_failure),
                hsm.effect(dispatch_failure),
                hsm.target(f"{root}/idle"),
            ),
        ),
    )


class Memory(store.MemoryStore):
    """Relational memory ability: one apply = one transaction of crude Statements.

    Public surface is HSM events only (``input_event`` / ``output_event`` / ``failed_event``)
    plus Ability lifecycle. Callers supply parameterized ``Statement`` rows; providers
    execute them against SQLite, Postgres, or another relational store. The connection
    stays private to the ability implementation.
    """

    default_scope: typing.ClassVar[str] = "memory"
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.memory.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.memory.output",
        schema=OutputData,
    )
    submodel: typing.ClassVar[hsm.Model | None] = memory_model(name="Memory", input_event=input_event)

    def __init__(
        self,
        *,
        connection: typing.Any | None = None,
        database: str = ":memory:",
    ) -> None:
        super().__init__(connection=connection, database=database)


__all__ = [
    "MEMORY_TABLE",
    "CandidateData",
    "InputData",
    "Memory",
    "MemoryGeneration",
    "MemoryGenerator",
    "MemoryRecord",
    "OutputData",
    "Row",
    "SourceData",
    "Statement",
    "StatementResult",
    "memory_model",
]
