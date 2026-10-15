from mosfet import abilities
from mosfet import media
from mosfet.abilities import vision

import asyncio
import collections.abc
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, start_abilities_for_test
from tests.type_helpers import model_view, object_dict


PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"image bytes", media_type="image/png")


class FixedVisualClassifier(vision.VisualClassifier):
    @override
    async def classify(self, input: vision.InputData) -> vision.OutputData:
        del input
        return vision.OutputData(kind="image", confidence=0.87)


class RecordingVisualClassification(vision.VisualClassification):
    outputs: list[vision.OutputData]
    failures: list[abilities.FailureData]

    def __init__(self, *, classifier: vision.VisualClassifier) -> None:
        super().__init__(classifier=classifier)
        self.outputs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, vision.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    await start_abilities_for_test(hsm.Context() if ctx is None else ctx, ability)


def test_visual_classification_input_carries_text_or_a_typed_image() -> None:
    text_input = vision.InputData(content="read this")
    image_input = vision.InputData(content=PNG)
    image_json_input = vision.InputData.model_validate_json(
        '{"content":{"content":"iVBORw0KGgo=","media_type":"image/png"}}'
    )
    round_tripped_image_input = vision.InputData.model_validate_json(image_input.model_dump_json())

    assert text_input.content == "read this"
    assert image_input.content == PNG
    assert image_json_input.content == media.Image(content=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    assert round_tripped_image_input == image_input


@pytest.mark.parametrize(
    "data",
    [
        {"content": b"raw image bytes"},
        {"content": ""},
        {"kind": "image", "content": PNG},
        {"kind": "text", "content": "read this"},
    ],
    ids=["bytes", "empty-text", "kind-image", "kind-text"],
)
def test_visual_classification_input_rejects_bytes_empty_text_and_kind(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ = vision.InputData.model_validate(data)


def test_visual_classification_input_no_longer_exports_an_input_kind() -> None:
    assert not hasattr(vision, "VisualClassificationInputKind")
    assert not hasattr(vision.classification, "VisualClassificationInputKind")


def test_visual_classification_output_records_kind_and_confidence() -> None:
    classification = vision.OutputData(kind="image", confidence=0.87)

    assert classification.kind == "image"
    assert classification.confidence == 0.87

    with pytest.raises(ValueError):
        _ = vision.OutputData(kind="text", confidence=1.1)


def test_visual_classification_uses_injected_classifier() -> None:
    async def run() -> list[vision.OutputData]:
        ability = RecordingVisualClassification(classifier=FixedVisualClassifier())
        await start_ability_tree(None, ability)

        _ = await dispatch_ability_for_test(ability, hsm.Context(), vision.InputData(content=PNG))
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [vision.OutputData(kind="image", confidence=0.87)]


def test_visual_classification_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(vision.VisualClassification.input_event.schema)
    output_schema = object_dict(vision.VisualClassification.output_event.schema)

    assert vision.VisualClassification.input_event.name == "bot.ability.vision.classification.input"
    assert input_schema == vision.InputData.model_json_schema()

    assert vision.VisualClassification.output_event.name == "bot.ability.vision.classification.output"
    assert output_schema == vision.OutputData.model_json_schema()


def test_visual_classification_model_tracks_classification_lifecycle() -> None:
    model = model_view(require_model(vision.VisualClassification.model))

    assert model.qualified_name == "/VisualClassificationLifecycle"
    assert model.initial == "/VisualClassificationLifecycle/.initial"
    assert "/VisualClassificationLifecycle/attached/behavior/Unclassified" in model.members
    assert "/VisualClassificationLifecycle/attached/behavior/Classifying" in model.members
    assert (
        "bot.ability.vision.classification.input"
        in model.transition_map["/VisualClassificationLifecycle/attached/behavior/Unclassified"]
    )
    assert (
        "bot.ability.vision.classification.apply.completed"
        in model.transition_map["/VisualClassificationLifecycle/attached/behavior/Classifying"]
    )
    assert (
        "bot.ability.vision.classification.apply.failed"
        in model.transition_map["/VisualClassificationLifecycle/attached/behavior/Classifying"]
    )


def test_visual_classification_is_concrete_ability() -> None:
    ability = vision.VisualClassification(classifier=FixedVisualClassifier())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(vision.VisualClassifier, abilities.Classifier)
    assert vision.VisualClassification.output_data_type is vision.OutputData
