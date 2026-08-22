from bot import abilities
from bot.abilities import generative
from bot.abilities import memory
from bot.abilities import processing

import asyncio
import collections.abc
import datetime
import typing

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test
from tests.bot.abilities.memory.memory_fixtures import (
    require_model,
    start_ability_tree,
    stop_ability_tree,
    wait_until,
)
from tests.type_helpers import invalid_value, model_view, object_dict


class GeneratedMemoryDecoder(
    abilities.Decoder[memory.classification.EncodedMemory, memory.classification.GeneratedMemory]
):
    decoded_values: list[memory.classification.EncodedMemory]

    def __init__(self) -> None:
        self.decoded_values = []

    @typing.override
    async def decode(self, input: memory.classification.EncodedMemory) -> memory.classification.GeneratedMemory:
        self.decoded_values.append(input)
        assert isinstance(input.value, str)
        return memory.classification.GeneratedMemory(
            content=input.value, kind=memory.classification.MemoryClassificationKind.FACT
        )


class PreferenceAssociationGenerator(generative.Generator[processing.InputData, memory.associative.LinkedData]):
    """Domain generator that yields LinkedData for AssociativeMemory (not event selections)."""

    inputs: list[processing.InputData]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def generate(self, input: processing.InputData) -> memory.associative.LinkedData:
        self.inputs.append(input)
        frame = input.input
        assert isinstance(frame, memory.associative.AssociationData)
        assert [source.decoded.content for source in frame.sources] == [
            "Gabe prefers terse handoff notes.",
            "Gabe asked for concise final reports.",
        ]
        return memory.associative.LinkedData(
            memory=memory.classification.GeneratedMemory(
                content="The operator prefers concise handoff and final notes.",
                kind=memory.classification.MemoryClassificationKind.PREFERENCE,
                subject_ref=frame.subject_ref,
            ),
            links=(
                memory.associative.Link(
                    source_ref="operator",
                    target_ref="preference:concise-handoff",
                    relation="prefers",
                    weight=0.94,
                    evidence_refs=("active-task",),
                ),
            ),
        )


class PreferenceAssociationProcessor(generative.Generative[processing.InputData, memory.associative.LinkedData]):
    """AssociativeMemory's ``processor`` slot is typed as Processing but product is LinkedData."""

    inputs: list[processing.InputData]

    def __init__(self) -> None:
        generator = PreferenceAssociationGenerator()
        super().__init__(generator=generator)
        self._generator = generator
        self.inputs = generator.inputs


class GeneratedMemoryEncoder(
    abilities.Encoder[memory.classification.GeneratedMemory, memory.classification.EncodedMemory]
):
    encoded_values: list[memory.classification.GeneratedMemory]

    def __init__(self) -> None:
        self.encoded_values = []

    @typing.override
    async def encode(self, input: memory.classification.GeneratedMemory) -> memory.classification.EncodedMemory:
        self.encoded_values.append(input)
        return memory.classification.EncodedMemory(format="text/plain", value=input.content)


class WrongGeneratedMemoryDecoder(
    abilities.Decoder[memory.classification.EncodedMemory, memory.classification.GeneratedMemory]
):
    @typing.override
    async def decode(self, input: memory.classification.EncodedMemory) -> memory.classification.GeneratedMemory:
        del input
        return invalid_value(memory.classification.GeneratedMemory, "not generated memory")


class WrongAssociationGenerator(generative.Generator[processing.InputData, memory.associative.LinkedData]):
    @typing.override
    async def generate(self, input: processing.InputData) -> memory.associative.LinkedData:
        del input
        return invalid_value(memory.associative.LinkedData, "not associative memory processing output")


class WrongAssociationProcessor(generative.Generative[processing.InputData, memory.associative.LinkedData]):
    def __init__(self) -> None:
        super().__init__(generator=WrongAssociationGenerator())


class WrongGeneratedMemoryEncoder(
    abilities.Encoder[memory.classification.GeneratedMemory, memory.classification.EncodedMemory]
):
    @typing.override
    async def encode(self, input: memory.classification.GeneratedMemory) -> memory.classification.EncodedMemory:
        del input
        return invalid_value(memory.classification.EncodedMemory, "not encoded memory")


class RecordingAssociativeMemory(memory.associative.AssociativeMemory):
    outputs: list[memory.associative.OutputData]
    failures: list[object]

    def __init__(
        self,
        *,
        processor: processing.Processing | generative.Generative[typing.Any, typing.Any],
        encoder: abilities.Encoder[memory.classification.GeneratedMemory, memory.classification.EncodedMemory],
        decoder: abilities.Decoder[memory.classification.EncodedMemory, memory.classification.GeneratedMemory],
    ) -> None:
        super().__init__(
            processor=typing.cast(processing.Processing, processor),
            encoder=encoder,
            decoder=decoder,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, memory.associative.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            self.failures.append(event.data)
        return super().dispatch(ctx, event)


def source_records() -> tuple[memory.store.MemoryRecord, ...]:
    return (
        memory.store.MemoryRecord(
            memory_id=__import__("uuid").uuid4().hex,
            scope="short_term",
            context_ref="active-task",
            subject_ref="operator",
            content="Gabe prefers terse handoff notes.",
            kind="preference",
        ),
        memory.store.MemoryRecord(
            memory_id=__import__("uuid").uuid4().hex,
            scope="short_term",
            context_ref="active-task",
            subject_ref="operator",
            content="Gabe asked for concise final reports.",
            kind="preference",
        ),
    )


def expected_links() -> tuple[memory.associative.Link, ...]:
    return (
        memory.associative.Link(
            source_ref="operator",
            target_ref="preference:concise-handoff",
            relation="prefers",
            weight=0.94,
            evidence_refs=("active-task",),
        ),
    )


def expected_memory() -> memory.classification.GeneratedMemory:
    return memory.classification.GeneratedMemory(
        content="The operator prefers concise handoff and final notes.",
        kind=memory.classification.MemoryClassificationKind.PREFERENCE,
        subject_ref="operator",
    )


def test_associative_memory_contract_lives_in_associative_module() -> None:
    assert memory.associative.AssociativeMemory.__module__ == "bot.abilities.memory.associative"
    assert memory.associative.InputData.__module__ == "bot.abilities.memory.associative"
    assert memory.associative.OutputData.__module__ == "bot.abilities.memory.associative"
    assert memory.associative.AssociationData.__module__ == "bot.abilities.memory.associative"
    assert memory.associative.LinkedData.__module__ == "bot.abilities.memory.associative"
    assert memory.associative.Link.__module__ == "bot.abilities.memory.associative"


def test_associative_memory_requires_source_records() -> None:
    try:
        _ = memory.associative.InputData(records=())
    except ValueError as error:
        assert "at least 1" in str(error)
    else:
        raise AssertionError("InputData accepted no records")


def test_associative_memory_uses_concrete_event_schemas() -> None:
    input_schema = object_dict(memory.associative.AssociativeMemory.input_event.schema)
    output_schema = object_dict(memory.associative.AssociativeMemory.output_event.schema)

    assert memory.associative.AssociativeMemory.input_event.name == "bot.ability.memory.associative.input"
    assert memory.associative.AssociativeMemory.output_event.name == "bot.ability.memory.associative.output"
    assert input_schema == memory.associative.InputData.model_json_schema()
    assert output_schema == memory.associative.OutputData.model_json_schema()
    assert input_schema["description"]
    assert output_schema["description"]


def test_associative_memory_schema_examples_validate_against_models() -> None:
    for model in (
        memory.associative.Link,
        memory.associative.InputData,
        memory.associative.AssociationData,
        memory.associative.LinkedData,
        memory.associative.OutputData,
    ):
        schema = object_dict(model.model_json_schema())
        examples = typing.cast(list[object], schema["examples"])
        assert examples
        for example in examples:
            _ = model.model_validate(example)


def test_associative_memory_model_tracks_operation_lifecycle() -> None:
    model = model_view(require_model(memory.associative.AssociativeMemory.model))

    assert model.qualified_name == "/AssociativeMemoryLifecycle"
    assert model.initial == "/AssociativeMemoryLifecycle/.initial"
    assert "/AssociativeMemoryLifecycle/detached" in model.members
    assert "/AssociativeMemoryLifecycle/attaching" in model.members
    assert "/AssociativeMemoryLifecycle/attached" in model.members
    assert "/AssociativeMemoryLifecycle/attached/behavior/initializing" in model.members
    assert "/AssociativeMemoryLifecycle/attached/behavior/idle" in model.members
    assert "/AssociativeMemoryLifecycle/attached/behavior/decoding" in model.members
    assert "/AssociativeMemoryLifecycle/attached/behavior/processing" in model.members
    assert "/AssociativeMemoryLifecycle/attached/behavior/encoding" in model.members
    assert (
        "bot.ability.memory.associative.input"
        in model.transition_map["/AssociativeMemoryLifecycle/attached/behavior/idle"]
    )
    assert (
        "bot.ability.memory.associative.sources.decoded"
        in model.transition_map["/AssociativeMemoryLifecycle/attached/behavior/decoding"]
    )
    assert (
        "bot.ability.memory.associative.processing.completed"
        in model.transition_map["/AssociativeMemoryLifecycle/attached/behavior/processing"]
    )
    assert (
        "bot.ability.memory.associative.apply.completed"
        in model.transition_map["/AssociativeMemoryLifecycle/attached/behavior/encoding"]
    )


def test_associative_memory_decodes_sources_then_processes_graph_associations() -> None:
    async def run() -> tuple[
        list[memory.associative.OutputData],
        list[processing.InputData],
        list[memory.classification.GeneratedMemory],
    ]:
        decoder = GeneratedMemoryDecoder()
        encoder = GeneratedMemoryEncoder()
        association = PreferenceAssociationProcessor()
        ability = RecordingAssociativeMemory(
            processor=association,
            encoder=encoder,
            decoder=decoder,
        )
        await start_ability_tree(ability)

        _ = await ability.apply(
            memory.associative.InputData(
                records=source_records(),
                context="Build graph associations for a knowledge store.",
                context_ref="active-task",
                subject_ref="operator",
                store_ref="knowledge-store",
                graph_ref="operator-preferences",
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs, association.inputs, encoder.encoded_values

    outputs, processor_inputs, encoded_values = asyncio.run(run())

    assert len(processor_inputs) == 1
    processor_frame = processor_inputs[0].input
    assert isinstance(processor_frame, memory.associative.AssociationData)
    assert processor_frame.context == "Build graph associations for a knowledge store."
    assert processor_frame.context_ref == "active-task"
    assert processor_frame.subject_ref == "operator"
    assert processor_frame.store_ref == "knowledge-store"
    assert processor_frame.graph_ref == "operator-preferences"
    assert [source.record.content for source in processor_frame.sources] == [
        "Gabe prefers terse handoff notes.",
        "Gabe asked for concise final reports.",
    ]
    assert encoded_values == [expected_memory()]
    assert len(outputs) == 1
    assert outputs[0].memory.content
    assert len(outputs[0].sources) == 2
    assert outputs[0].sources[0].record.content == "Gabe prefers terse handoff notes."
    assert outputs[0].sources[1].record.content == "Gabe asked for concise final reports."


def test_associative_memory_directed_operation_preserves_requester_through_processing() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        association = RecordingAssociativeMemory(
            processor=PreferenceAssociationProcessor(),
            encoder=GeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(association)
        terminal = await abilities.run_terminal_operation(
            association.context(),
            child=association,
            request=association.input_event.with_data_and_id(
                memory.associative.InputData(
                    records=source_records(),
                    context="Build graph associations for a knowledge store.",
                    context_ref="active-task",
                    subject_ref="operator",
                ),
                "associative:directed",
            ),
            terminals=(association.output_event, association.failed_event),
            timeout=datetime.timedelta.max,
        )
        return terminal, hsm.id(association)

    terminal, association_id = asyncio.run(run())

    assert terminal.id == "associative:directed"
    assert terminal.source == association_id
    assert terminal.target and terminal.target != association_id
    assert isinstance(terminal.data, memory.associative.OutputData)


def test_associative_memory_routes_wrong_association_to_failure() -> None:
    async def run() -> tuple[str, list[object]]:
        ability = RecordingAssociativeMemory(
            processor=WrongAssociationProcessor(),
            encoder=GeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(ability)

        with pytest.raises(RuntimeError, match="output schema"):
            _ = await dispatch_ability_for_test(
                ability, hsm.Context(), memory.associative.InputData(records=source_records())
            )
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state.endswith("/attached/behavior/idle")
    assert len(failures) == 1


def test_associative_memory_routes_wrong_encoded_memory_to_failure() -> None:
    async def run() -> tuple[str, list[object]]:
        ability = RecordingAssociativeMemory(
            processor=PreferenceAssociationProcessor(),
            encoder=WrongGeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(ability)

        with pytest.raises(RuntimeError, match="EncodedMemory"):
            _ = await dispatch_ability_for_test(
                ability, hsm.Context(), memory.associative.InputData(records=source_records())
            )
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state.endswith("/attached/behavior/idle")
    assert len(failures) == 1


def test_associative_memory_completion_correlates_via_event_id_without_instance_stash() -> None:
    """HSM-COMPLETION-001: associative turn rides completion events, not processing state bags."""

    async def run() -> tuple[memory.associative.OutputData, object]:
        ability = RecordingAssociativeMemory(
            processor=PreferenceAssociationProcessor(),
            encoder=GeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(ability)
        output = await dispatch_ability_for_test(
            ability,
            hsm.Context(),
            memory.associative.InputData(
                records=source_records(),
                context="Build graph associations for a knowledge store.",
                context_ref="active-task",
                subject_ref="operator",
            ),
        )
        proc_state, ok = typing.cast(
            tuple[object, bool],
            ability.get("associative_memory_processing_state"),
        )
        await stop_ability_tree(ability)
        return output, proc_state if ok else None

    output, proc_state = asyncio.run(run())

    assert output.memory == expected_memory()
    assert output.links == expected_links()
    assert proc_state is None
