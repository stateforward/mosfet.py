from bot import abilities
from bot.abilities import memory

import asyncio
import collections.abc
import typing

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test
from tests.bot.abilities.memory.memory_fixtures import (
    require_model,
    start_ability_tree,
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


class PreferenceConsolidator(memory.consolidation.MemoryConsolidationGenerator):
    inputs: list[memory.consolidation.SourceData]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def generate(self, input: memory.consolidation.SourceData) -> memory.classification.GeneratedMemory:
        self.inputs.append(input)
        assert [source.decoded.content for source in input.sources] == [
            "Gabe prefers terse handoff notes.",
            "Gabe asked for concise final reports.",
        ]
        return memory.classification.GeneratedMemory(
            content="The operator prefers concise handoff and final notes.",
            kind=memory.classification.MemoryClassificationKind.PREFERENCE,
            subject_ref=input.subject_ref,
        )


class WrongGeneratedMemoryDecoder(
    abilities.Decoder[memory.classification.EncodedMemory, memory.classification.GeneratedMemory]
):
    @typing.override
    async def decode(self, input: memory.classification.EncodedMemory) -> memory.classification.GeneratedMemory:
        del input
        return invalid_value(memory.classification.GeneratedMemory, "not generated memory")


class WrongGeneratedMemoryEncoder(
    abilities.Encoder[memory.classification.GeneratedMemory, memory.classification.EncodedMemory]
):
    @typing.override
    async def encode(self, input: memory.classification.GeneratedMemory) -> memory.classification.EncodedMemory:
        del input
        return invalid_value(memory.classification.EncodedMemory, "not encoded memory")


class RecordingMemoryConsolidation(memory.consolidation.MemoryConsolidation):
    outputs: list[memory.consolidation.OutputData]
    failures: list[object]

    def __init__(
        self,
        *,
        generator: memory.consolidation.MemoryConsolidationGenerator,
        encoder: abilities.Encoder[memory.classification.GeneratedMemory, memory.classification.EncodedMemory],
        decoder: abilities.Decoder[memory.classification.EncodedMemory, memory.classification.GeneratedMemory],
    ) -> None:
        super().__init__(generator=generator, encoder=encoder, decoder=decoder)
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, memory.consolidation.OutputData)
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


def test_memory_consolidation_contract_lives_in_consolidation_module() -> None:
    assert memory.consolidation.MemoryConsolidation.__module__ == "bot.abilities.memory.consolidation"
    assert memory.consolidation.InputData.__module__ == "bot.abilities.memory.consolidation"
    assert memory.consolidation.OutputData.__module__ == "bot.abilities.memory.consolidation"
    assert memory.consolidation.MemoryConsolidationSource.__module__ == "bot.abilities.memory.consolidation"


def test_memory_consolidation_requires_source_records() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ = memory.consolidation.InputData(records=())


def test_memory_consolidation_uses_concrete_event_schemas() -> None:
    input_schema = object_dict(memory.consolidation.MemoryConsolidation.input_event.schema)
    output_schema = object_dict(memory.consolidation.MemoryConsolidation.output_event.schema)

    assert memory.consolidation.MemoryConsolidation.input_event.name == "bot.ability.memory.consolidation.input"
    assert memory.consolidation.MemoryConsolidation.output_event.name == "bot.ability.memory.consolidation.output"
    assert input_schema == memory.consolidation.InputData.model_json_schema()
    assert output_schema == memory.consolidation.OutputData.model_json_schema()
    assert input_schema["description"]
    assert output_schema["description"]


def test_memory_consolidation_schema_examples_validate_against_models() -> None:
    for model in (
        memory.consolidation.MemoryConsolidationSource,
        memory.consolidation.InputData,
        memory.consolidation.SourceData,
        memory.consolidation.OutputData,
    ):
        schema = object_dict(model.model_json_schema())
        examples = typing.cast(list[object], schema["examples"])
        assert examples
        for example in examples:
            _ = model.model_validate(example)


def test_memory_consolidation_model_tracks_operation_lifecycle() -> None:
    model = model_view(require_model(memory.consolidation.MemoryConsolidation.model))

    assert model.qualified_name == "/MemoryConsolidationLifecycle"
    assert model.initial == "/MemoryConsolidationLifecycle/.initial"
    assert "/MemoryConsolidationLifecycle/detached" in model.members
    assert "/MemoryConsolidationLifecycle/attaching" in model.members
    assert "/MemoryConsolidationLifecycle/attached" in model.members
    assert "/MemoryConsolidationLifecycle/attached/behavior/idle" in model.members
    assert "/MemoryConsolidationLifecycle/attached/behavior/applying" in model.members
    assert (
        "bot.ability.memory.consolidation.input"
        in model.transition_map["/MemoryConsolidationLifecycle/attached/behavior/idle"]
    )
    assert (
        "bot.ability.memory.consolidation.apply.completed"
        in model.transition_map["/MemoryConsolidationLifecycle/attached/behavior/applying"]
    )
    assert (
        "bot.ability.memory.consolidation.apply.failed"
        in model.transition_map["/MemoryConsolidationLifecycle/attached/behavior/applying"]
    )


def test_memory_consolidation_decodes_sources_then_generates_and_encodes_consolidated_memory() -> None:
    async def run() -> tuple[
        list[memory.consolidation.OutputData],
        list[memory.consolidation.SourceData],
        list[memory.classification.GeneratedMemory],
    ]:
        decoder = GeneratedMemoryDecoder()
        encoder = GeneratedMemoryEncoder()
        generator = PreferenceConsolidator()
        ability = RecordingMemoryConsolidation(generator=generator, encoder=encoder, decoder=decoder)
        await start_ability_tree(ability)

        _ = await ability.apply(
            memory.consolidation.InputData(
                records=source_records(),
                context="Consolidate repeated collaboration preferences after an interaction.",
                context_ref="active-task",
                subject_ref="operator",
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs, generator.inputs, encoder.encoded_values

    outputs, generator_inputs, encoded_values = asyncio.run(run())

    assert len(generator_inputs) == 1
    assert [source.record.content for source in generator_inputs[0].sources] == [
        "Gabe prefers terse handoff notes.",
        "Gabe asked for concise final reports.",
    ]
    assert encoded_values == [
        memory.classification.GeneratedMemory(
            content="The operator prefers concise handoff and final notes.",
            kind=memory.classification.MemoryClassificationKind.PREFERENCE,
            subject_ref="operator",
        )
    ]
    assert len(outputs) == 1
    assert outputs[0].memory.content
    assert len(outputs[0].sources) == 2
    assert outputs[0].sources[0].record.content == "Gabe prefers terse handoff notes."
    assert outputs[0].sources[1].record.content == "Gabe asked for concise final reports."


def test_memory_consolidation_routes_wrong_encoded_memory_to_failure() -> None:
    async def run() -> tuple[str, list[object]]:
        ability = RecordingMemoryConsolidation(
            generator=PreferenceConsolidator(),
            encoder=WrongGeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(ability)

        with pytest.raises(RuntimeError, match="EncodedMemory"):
            _ = await dispatch_ability_for_test(
                ability, hsm.Context(), memory.consolidation.InputData(records=source_records())
            )
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state.endswith("/attached/behavior/idle")
    assert len(failures) == 1


def test_memory_consolidation_completion_correlates_via_event_id_without_instance_stash() -> None:
    """HSM-COMPLETION-001: consolidation apply correlation rides event id/metadata, not instance stash."""

    async def run() -> tuple[memory.consolidation.OutputData, object]:
        ability = RecordingMemoryConsolidation(
            generator=PreferenceConsolidator(),
            encoder=GeneratedMemoryEncoder(),
            decoder=GeneratedMemoryDecoder(),
        )
        await start_ability_tree(ability)
        output = await dispatch_ability_for_test(
            ability,
            hsm.Context(),
            memory.consolidation.InputData(records=source_records()),
        )
        op_state, ok = typing.cast(
            tuple[object, bool],
            ability.get("memory_consolidation_operation_state"),
        )
        await ability.stop(ability.context())
        return output, op_state if ok else None

    output, op_state = asyncio.run(run())

    assert output.memory.content
    assert op_state is None
