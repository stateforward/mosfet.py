from mosfet.abilities import memory

import asyncio

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test
from tests.bot.abilities.memory.memory_fixtures import (
    RecordingMemoryClassification,
    RetainPreferenceClassifier,
    WrongMemoryClassifier,
    require_model,
    start_ability_tree,
    wait_until,
)
from tests.type_helpers import model_view, object_dict


def test_memory_classification_contract_lives_in_classification_module() -> None:
    assert memory.classification.MemoryClassification.__module__ == "mosfet.abilities.memory.classification"
    assert memory.classification.InputData.__module__ == "mosfet.abilities.memory.classification"
    assert memory.classification.OutputData.__module__ == "mosfet.abilities.memory.classification"


def test_generated_memory_content_is_required() -> None:
    with pytest.raises(ValueError):
        _ = memory.classification.GeneratedMemory(content="")


def test_memory_classification_uses_concrete_event_schemas() -> None:
    input_schema = object_dict(memory.classification.MemoryClassification.input_event.schema)
    output_schema = object_dict(memory.classification.MemoryClassification.output_event.schema)

    assert memory.classification.MemoryClassification.input_event.name == "bot.ability.memory.classification.input"
    assert memory.classification.MemoryClassification.output_event.name == "bot.ability.memory.classification.output"
    assert input_schema == memory.classification.InputData.model_json_schema()
    assert output_schema == memory.classification.OutputData.model_json_schema()
    assert input_schema["description"]
    assert output_schema["description"]


def test_memory_classification_model_tracks_operation_lifecycle() -> None:
    model = model_view(require_model(memory.classification.MemoryClassification.model))

    assert model.qualified_name == "/MemoryClassificationLifecycle"
    assert model.initial == "/MemoryClassificationLifecycle/.initial"
    assert "/MemoryClassificationLifecycle/attached/behavior/idle" in model.members
    assert "/MemoryClassificationLifecycle/attached/behavior/applying" in model.members
    assert (
        "bot.ability.memory.classification.input"
        in model.transition_map["/MemoryClassificationLifecycle/attached/behavior/idle"]
    )
    assert (
        "bot.ability.memory.classification.apply.completed"
        in model.transition_map["/MemoryClassificationLifecycle/attached/behavior/applying"]
    )
    assert (
        "bot.ability.memory.classification.apply.failed"
        in model.transition_map["/MemoryClassificationLifecycle/attached/behavior/applying"]
    )


def test_memory_classification_apply_dispatches_classifier_output() -> None:
    async def run() -> list[memory.classification.OutputData]:
        ability = RecordingMemoryClassification(classifier=RetainPreferenceClassifier())
        await start_ability_tree(ability)

        _ = await ability.apply(memory.classification.InputData(candidate="The operator prefers terse handoff notes."))
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [
        memory.classification.OutputData(
            retention=memory.classification.MemoryClassificationRetention.RETAIN,
            kind=memory.classification.MemoryClassificationKind.PREFERENCE,
            sensitivity=memory.classification.MemoryClassificationSensitivity.STANDARD,
            confidence=0.93,
        )
    ]


def test_memory_classification_routes_wrong_output_type_to_failure() -> None:
    async def run() -> tuple[str, list[object]]:
        ability = RecordingMemoryClassification(classifier=WrongMemoryClassifier())
        await start_ability_tree(ability)

        with pytest.raises(RuntimeError, match="classifier failed|output schema"):
            _ = await dispatch_ability_for_test(
                ability,
                hsm.Context(),
                memory.classification.InputData(candidate="The operator prefers terse handoff notes."),
            )
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingMemoryClassificationLifecycle/attached/behavior/idle"
    assert len(failures) == 1


def test_memory_classification_confidence_is_normalized() -> None:
    with pytest.raises(ValueError):
        _ = memory.classification.OutputData(
            retention=memory.classification.MemoryClassificationRetention.RETAIN,
            kind=memory.classification.MemoryClassificationKind.PREFERENCE,
            sensitivity=memory.classification.MemoryClassificationSensitivity.STANDARD,
            confidence=1.1,
        )
