from bot import abilities
from bot.abilities import reading
from bot.abilities import vision

import asyncio
import collections.abc
import contextlib
import dataclasses
import typing

import hsm
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

import bot.abilities.reading.reading as reading_module

from tests.hsm_instance_state import start_ability_tree
from tests.type_helpers import model_view, object_dict

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

class StubVisualClassifier(vision.VisualClassifier):
    outputs: list[vision.classification.OutputData]
    calls: list[vision.classification.InputData]

    def __init__(self, *outputs: vision.classification.OutputData) -> None:
        self.outputs = list(outputs) or [vision.classification.OutputData(kind="text")]
        self.calls = []

    @typing.override
    async def classify(self, input: vision.classification.InputData) -> vision.classification.OutputData:
        self.calls.append(input)
        return self.outputs.pop(0)

class HangingVisualClassifier(vision.VisualClassifier):
    cancelled: bool

    def __init__(self) -> None:
        self.cancelled = False

    @typing.override
    async def classify(self, input: vision.classification.InputData) -> vision.classification.OutputData:
        del input
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")

class StubTextDecoder(abilities.Decoder[str, str]):
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: str) -> str:
        self.calls.append(input)
        return f"text:{input}"

class StubImageDecoder(abilities.Decoder[bytes, str]):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: bytes) -> str:
        self.calls.append(input)
        return "image text"

class StubOutputEncoder(abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData]):
    calls: list[reading.reading.OutputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def encode(self, input: reading.reading.OutputData) -> reading.reading.OutputData:
        self.calls.append(input)
        return input

class HangingOutputEncoder(abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData]):
    cancelled: bool

    def __init__(self) -> None:
        self.cancelled = False

    @typing.override
    async def encode(self, input: reading.reading.OutputData) -> reading.reading.OutputData:
        del input
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")

class RecordingReading(reading.Reading):
    outputs: list[reading.reading.OutputData]
    failures: list[reading.FailedEventData]

    def __init__(
        self,
        *,
        visual_classifier: vision.VisualClassifier,
        text_decoder: abilities.Decoder[str, str],
        image_decoder: abilities.Decoder[bytes, str],
        output_encoder: abilities.Encoder[reading.reading.OutputData, reading.reading.OutputData],
    ) -> None:
        super().__init__(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=output_encoder,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, reading.reading.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, reading.FailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)

async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)

def stub_reading(
    *,
    visual_classifier: StubVisualClassifier | None = None,
    text_decoder: StubTextDecoder | None = None,
    image_decoder: StubImageDecoder | None = None,
    output_encoder: StubOutputEncoder | None = None,
) -> reading.Reading:
    return reading.Reading(
        visual_classifier=visual_classifier or StubVisualClassifier(),
        text_decoder=text_decoder or StubTextDecoder(),
        image_decoder=image_decoder or StubImageDecoder(),
        output_encoder=output_encoder or StubOutputEncoder(),
    )

def test_reading_input_separates_text_and_image_payloads() -> None:
    text_input = reading.reading.InputData(kind="text", content="read this")
    image_input = reading.reading.InputData(kind="image", content=b"image bytes")
    image_json_input = reading.reading.InputData.model_validate_json('{"kind":"image","content":"aW1hZ2UgYnl0ZXM="}')
    urlsafe_image_input = reading.reading.InputData(kind="image", content=b"\xfb\xff")
    round_tripped_image_input = reading.reading.InputData.model_validate_json(urlsafe_image_input.model_dump_json())

    assert text_input.content == "read this"
    assert image_input.content == b"image bytes"
    assert image_json_input.content == b"image bytes"
    assert round_tripped_image_input.content == b"\xfb\xff"

    with pytest.raises(ValueError):
        _ = reading.reading.InputData(kind="text", content=b"not text")

    with pytest.raises(ValueError):
        _ = reading.reading.InputData(kind="image", content="not image")

def test_reading_output_records_normalized_text_and_source_kind() -> None:
    output = reading.reading.OutputData(text="normalized", source_kind="text")

    assert output.text == "normalized"
    assert output.source_kind == "text"
    assert output.confidence is None

    with pytest.raises(ValueError):
        _ = reading.reading.OutputData(text="", source_kind="image", confidence=1.1)

def test_reading_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(reading.Reading.input_event.schema)
    output_schema = object_dict(reading.Reading.output_event.schema)
    failed_schema = object_dict(reading.Reading.failed_event.schema)

    assert reading.Reading.input_event.name == "bot.ability.reading.input"
    assert input_schema == reading.reading.InputData.model_json_schema()

    assert reading.Reading.output_event.name == "bot.ability.reading.output"
    assert output_schema == reading.reading.OutputData.model_json_schema()

    assert reading.Reading.failed_event.name == "bot.ability.reading.failed"
    assert failed_schema == reading.FailedEventData.model_json_schema()

def test_reading_records_injected_classification_decoding_and_encoding_abilities() -> None:
    visual_classifier = StubVisualClassifier()
    text_decoder = StubTextDecoder()
    image_decoder = StubImageDecoder()
    output_encoder = StubOutputEncoder()
    reading_ability = stub_reading(
        visual_classifier=visual_classifier,
        text_decoder=text_decoder,
        image_decoder=image_decoder,
        output_encoder=output_encoder,
    )
    instance_state = vars(reading_ability)
    stored_visual_classifier = typing.cast(vision.VisualClassification, instance_state["_visual_classifier"])
    stored_text_decoder = typing.cast(abilities.Decoding[str, str], instance_state["_text_decoder"])
    stored_image_decoder = typing.cast(abilities.Decoding[bytes, str], instance_state["_image_decoder"])
    stored_output_encoder = typing.cast(abilities.Encoding[reading.OutputData, reading.OutputData], instance_state["_output_encoder"])
    subordinate_abilities = typing.cast(tuple[object, ...], instance_state["_subordinate_abilities"])

    assert stored_visual_classifier.classifier is visual_classifier
    assert stored_text_decoder.decoder is text_decoder
    assert stored_image_decoder.decoder is image_decoder
    assert stored_output_encoder.encoder is output_encoder
    assert subordinate_abilities == (
        stored_visual_classifier,
        stored_text_decoder,
        stored_image_decoder,
        stored_output_encoder,
    )

def test_reading_apply_bridge_keeps_operation_state_out_of_instance() -> None:
    reading_ability = stub_reading()

    assert "_pending_apply_results" not in vars(reading_ability)
    assert "_active_apply_operation_id" not in vars(reading_ability)

def test_reading_apply_runs_text_route_to_encoded_output() -> None:
    async def run() -> tuple[reading.OutputData, list[vision.classification.InputData], list[str], list[reading.OutputData]]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99))
        text_decoder = StubTextDecoder()
        output_encoder = StubOutputEncoder()
        reading_ability = stub_reading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            output_encoder=output_encoder,
        )
        await start_ability_tree(None, reading_ability)

        output = await dispatch_ability_for_test(reading_ability, hsm.Context(), reading.InputData(kind="text", content="hello"))
        return output, visual_classifier.calls, text_decoder.calls, output_encoder.calls

    output, classification_calls, text_calls, encoder_calls = asyncio.run(run())

    assert output == reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)
    assert classification_calls == [vision.classification.InputData(kind="text", content="hello")]
    assert text_calls == ["hello"]
    assert encoder_calls == [reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)]

def test_reading_ignores_stale_terminal_event_for_previous_apply_operation() -> None:
    async def run() -> tuple[list[reading.OutputData], str, bool]:
        async def run_apply(reading_ability: reading.Reading) -> reading.OutputData:
            return await dispatch_ability_for_test(reading_ability, hsm.Context(), reading.InputData(kind="text", content="current"))

        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99)),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=HangingOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)
        task = asyncio.create_task(run_apply(reading_ability))
        await wait_until(
            lambda: reading_ability.state() == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
        )
        output_event = typing.cast(hsm.Event[reading.OutputData], getattr(reading_module, "_ReadingOutputEncodedEvent"))
        stale_event = dataclasses.replace(
            output_event.with_data(reading.OutputData(text="stale", source_kind="text", confidence=0.1)),
            id="previous",
        )
        await hsm.dispatch(None, reading_ability, stale_event)
        await asyncio.sleep(0.01)
        outputs = list(reading_ability.outputs)
        state = reading_ability.state()
        task_done = task.done()
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await reading_ability.stop(reading_ability.context())
        return outputs, state, task_done

    outputs, state, task_done = asyncio.run(run())

    assert outputs == []
    assert state == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
    assert not task_done

def test_reading_detach_releases_owned_subabilities_while_focused() -> None:
    async def run() -> tuple[tuple[hsm.Instance | None, ...], str]:
        ctx = hsm.Context()
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99)),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=HangingOutputEncoder(),
        )
        await start_ability_tree(ctx, reading_ability)
        _ = await reading_ability.apply(reading.InputData(kind="text", content="current"), ctx=ctx)
        await wait_until(
            lambda: reading_ability.state() == "/RecordingReadingLifecycle/attached/behavior/Focused/EncodingOutput"
        )
        subabilities = typing.cast(
            tuple[abilities.Ability[typing.Any, typing.Any], ...],
            vars(reading_ability)["_subordinate_abilities"],
        )

        _ = await reading_ability.detach(ctx=ctx)
        await wait_until(lambda: all(abilities.Ability.current_owner(ability) is None for ability in subabilities))
        owners = tuple(abilities.Ability.current_owner(ability) for ability in subabilities)
        state = reading_ability.state()
        await reading_ability.stop(ctx)
        return owners, state

    owners, state = asyncio.run(run())

    assert owners == (None, None, None, None)
    assert state == "/RecordingReadingLifecycle/detached"

def test_reading_model_tracks_focus_classification_decoding_and_encoding() -> None:
    model = model_view(require_model(reading.Reading.model))

    assert model.qualified_name == "/ReadingLifecycle"
    assert model.initial == "/ReadingLifecycle/.initial"
    assert "/ReadingLifecycle/detached" in model.members
    assert "/ReadingLifecycle/attaching" in model.members
    assert "/ReadingLifecycle/attached" in model.members
    assert "/ReadingLifecycle/attached/behavior/initializing" in model.members
    assert "/ReadingLifecycle/attached/behavior/Unfocused" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying/Applying" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/Classifying/Classified" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/DecodingText" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/DecodingImage" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/PreparingUnreadableOutput" in model.members
    assert "/ReadingLifecycle/attached/behavior/Focused/EncodingOutput" in model.members
    assert "bot.ability.reading.input" in model.transition_map["/ReadingLifecycle/attached/behavior/Unfocused"]
    assert (
        "bot.ability.reading.classification.completed"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/Classifying/Applying"]
    )
    assert (
        "bot.ability.reading.text.decoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/DecodingText"]
    )
    assert (
        "bot.ability.reading.image.decoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/DecodingImage"]
    )
    assert (
        "bot.ability.reading.unreadable.output.ready"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/PreparingUnreadableOutput"]
    )
    assert (
        "bot.ability.reading.output.encoded"
        in model.transition_map["/ReadingLifecycle/attached/behavior/Focused/EncodingOutput"]
    )

def test_reading_runs_text_route_to_encoded_output() -> None:
    async def run() -> tuple[list[reading.OutputData], list[vision.classification.InputData], list[str], list[bytes], str]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.99))
        text_decoder = StubTextDecoder()
        image_decoder = StubImageDecoder()
        reading_ability = RecordingReading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="text", content="hello"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return reading_ability.outputs, visual_classifier.calls, text_decoder.calls, image_decoder.calls, reading_ability.state()

    outputs, classification_calls, text_calls, image_calls, active_state = asyncio.run(run())

    assert outputs == [reading.OutputData(text="text:hello", source_kind="text", confidence=0.99)]
    assert classification_calls == [vision.classification.InputData(kind="text", content="hello")]
    assert text_calls == ["hello"]
    assert image_calls == []
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"

def test_reading_runs_image_route_to_encoded_output() -> None:
    async def run() -> tuple[list[reading.OutputData], list[vision.classification.InputData], list[str], list[bytes], str]:
        visual_classifier = StubVisualClassifier(vision.classification.OutputData(kind="image", confidence=0.87))
        text_decoder = StubTextDecoder()
        image_decoder = StubImageDecoder()
        reading_ability = RecordingReading(
            visual_classifier=visual_classifier,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="image", content=b"image bytes"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return reading_ability.outputs, visual_classifier.calls, text_decoder.calls, image_decoder.calls, reading_ability.state()

    outputs, classification_calls, text_calls, image_calls, active_state = asyncio.run(run())

    assert outputs == [reading.OutputData(text="image text", source_kind="image", confidence=0.87)]
    assert classification_calls == [vision.classification.InputData(kind="image", content=b"image bytes")]
    assert text_calls == []
    assert image_calls == [b"image bytes"]
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"

def test_reading_runs_unreadable_route_to_encoded_output() -> None:
    async def run() -> list[reading.OutputData]:
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="unreadable", confidence=0.76)),
            text_decoder=StubTextDecoder(),
            image_decoder=StubImageDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        _ = await reading_ability.apply(reading.InputData(kind="image", content=b"blank"))
        await wait_until(lambda: bool(reading_ability.outputs))

        return reading_ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [reading.reading.OutputData(text="", source_kind="unreadable", confidence=0.76)]

def test_reading_rejects_classification_that_does_not_match_input_payload() -> None:
    async def run() -> tuple[list[reading.OutputData], list[reading.FailedEventData], list[str], str]:
        text_decoder = StubTextDecoder()
        reading_ability = RecordingReading(
            visual_classifier=StubVisualClassifier(vision.classification.OutputData(kind="text", confidence=0.87)),
            text_decoder=text_decoder,
            image_decoder=StubImageDecoder(),
            output_encoder=StubOutputEncoder(),
        )
        await start_ability_tree(None, reading_ability)

        with pytest.raises(RuntimeError, match="classification route"):
            _ = await dispatch_ability_for_test(
                reading_ability, hsm.Context(), reading.InputData(kind="image", content=b"image bytes")
            )
        await wait_until(lambda: bool(reading_ability.failures))

        return reading_ability.outputs, reading_ability.failures, text_decoder.calls, reading_ability.state()

    outputs, failures, text_calls, active_state = asyncio.run(run())

    assert outputs == []
    assert failures == [
        reading.FailedEventData(
            stage="classification",
            message="Reading classification route did not match the input payload.",
        )
    ]
    assert text_calls == []
    assert active_state == "/RecordingReadingLifecycle/attached/behavior/Unfocused"
