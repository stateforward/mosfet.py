from __future__ import annotations

from mosfet import abilities, media
from mosfet.abilities import decoding
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.protocols import attachment
from mosfet.telemetry import span

import asyncio
import collections.abc
import datetime
import typing
import uuid

import hsm
import pydantic
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree

TURN_REF = "turn-1"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _image(index: int = 0) -> media.Image:
    return media.Image(content=_PNG_SIGNATURE + f"png-{index}".encode(), media_type="image/png")


class RecordingDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
    calls: list[turn_detector.ParticipationStimulus]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
        self.calls.append(input)
        return "decoded"


class RecordingTurnDetector(turn_detector.TurnDetector):
    outputs: list[turn_detector.TurnCompleteData]
    failures: list[turn_detector.FailedEventData]

    def __init__(
        self,
        *,
        participant_ref: str = "bot",
        conversation_ref: str = "conversation",
        end_of_turn_silence_seconds: float = 0.5,
        decoder: decoding.Decoder[turn_detector.ParticipationStimulus, str] | None = None,
    ) -> None:
        super().__init__(
            participant_ref=participant_ref,
            conversation_ref=conversation_ref,
            end_of_turn_silence_seconds=end_of_turn_silence_seconds,
            decoder=decoder,
        )
        self.outputs = []
        self.failures = []


async def wait_until(condition: object, *, timeout: float = 1.0) -> None:
    predicate = condition
    assert callable(predicate)
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met")


def _start(*, text: str, participant_ref: str = "bot", conversation_ref: str = "conversation") -> hsm.Event:
    return turn_detector.TurnStartEvent.with_data_and_id(
        turn_detector.TurnStartData(
            conversation_ref=conversation_ref,
            turn_ref=TURN_REF,
            self_participant_ref=participant_ref,
            source_participant_ref="caller",
            content=turn_detector.TextStimulus(source_participant_ref="caller", content=text),
        ),
        uuid.uuid4().hex,
    )


def test_turn_detector_owns_typed_turn_lifecycle_and_exports_no_perception_pipeline() -> None:
    assert turn_detector.TurnDetector.input_event is turn_detector.TurnStartEvent
    assert turn_detector.TurnDetector.output_event is turn_detector.TurnCompleteEvent
    assert turn_detector.TurnDetector.failed_event is turn_detector.TurnDetectorFailedEvent
    assert "InputData" not in turn_detector.__all__
    assert "OutputData" not in turn_detector.__all__


def test_turn_detector_returns_ready_to_the_terminal_operation() -> None:
    async def run() -> hsm.Event[object]:
        detector = RecordingTurnDetector(participant_ref="bot-a", conversation_ref="conversation")
        context = hsm.Context()
        await start_ability_tree(context, detector)
        operation_id = uuid.uuid4().hex
        return await abilities.run_terminal_operation(
            context,
            child=detector,
            request=turn_detector.TurnDetectorReadyRequestEvent.with_data_and_id(
                turn_detector.TurnDetectorReadyRequestData(
                    participant_ref="bot-a",
                    conversation_ref="conversation",
                ),
                operation_id,
            ),
            terminals=(turn_detector.TurnDetectorReadyEvent, detector.failed_event),
            timeout=datetime.timedelta(seconds=1),
        )

    terminal = asyncio.run(run())
    assert terminal.name == turn_detector.TurnDetectorReadyEvent.name
    assert terminal.data == turn_detector.TurnDetectorReadyData(
        participant_ref="bot-a",
        conversation_ref="conversation",
    )


def test_turn_detector_start_update_end_emits_participant_scoped_complete() -> None:
    async def run() -> turn_detector.TurnCompleteData:
        ability = RecordingTurnDetector(participant_ref="bot-a", end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        operation_id = uuid.uuid4().hex
        await hsm.dispatch(context, ability, _start(text="hello", participant_ref="bot-a"))
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/open"))
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnUpdateEvent.with_data_and_id(
                turn_detector.TurnUpdateData(
                    turn_ref=TURN_REF,
                    source_participant_ref="caller",
                    content=turn_detector.TextStimulus(source_participant_ref="caller", content="world"),
                ),
                operation_id,
            ),
        )
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnEndEvent.with_data_and_id(
                turn_detector.TurnEndData(turn_ref=TURN_REF, source_participant_ref="caller"),
                operation_id,
            ),
        )
        await wait_until(lambda: bool(getattr(ability, "outputs", [])))
        output = ability.outputs[-1]
        assert isinstance(output, turn_detector.TurnCompleteData)
        return output

    output = asyncio.run(run())
    assert output.participant_ref == "bot-a"
    assert output.self_participant_ref == "bot-a"
    assert output.turn_ref == TURN_REF
    assert output.text == "hello world"
    assert output.source_participant_ref == "caller"


def test_turn_detector_ignores_turn_start_for_another_participant() -> None:
    async def run() -> str:
        ability = RecordingTurnDetector(participant_ref="bot-a", end_of_turn_silence_seconds=0.03)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnStartEvent.with_data_and_id(
                turn_detector.TurnStartData(
                    turn_ref=TURN_REF,
                    self_participant_ref="bot-b",
                    source_participant_ref="caller",
                    content=turn_detector.TextStimulus(source_participant_ref="caller", content="not mine"),
                ),
                "wrong-participant",
            ),
        )
        await asyncio.sleep(0.05)
        return ability.state()

    assert asyncio.run(run()).endswith("/behavior/idle")


def test_turn_detector_pause_and_resume_keeps_one_turn() -> None:
    async def run() -> turn_detector.TurnCompleteData:
        ability = RecordingTurnDetector(end_of_turn_silence_seconds=0.2)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        operation_id = uuid.uuid4().hex
        await hsm.dispatch(context, ability, _start(text="before"))
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/open"))
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnPauseEvent.with_data_and_id(
                turn_detector.TurnPauseData(turn_ref=TURN_REF, duration_seconds=0.1),
                operation_id,
            ),
        )
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/paused"))
        assert getattr(ability, "outputs", []) == []
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnUpdateEvent.with_data_and_id(
                turn_detector.TurnUpdateData(
                    turn_ref=TURN_REF,
                    source_participant_ref="caller",
                    content=turn_detector.TextStimulus(source_participant_ref="caller", content="after"),
                ),
                operation_id,
            ),
        )
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/open"))
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnEndEvent.with_data_and_id(
                turn_detector.TurnEndData(turn_ref=TURN_REF, source_participant_ref="caller"),
                operation_id,
            ),
        )
        await wait_until(lambda: len(getattr(ability, "outputs", [])) == 1)
        return ability.outputs[0]

    output = asyncio.run(run())
    assert output.text == "before after"


def test_turn_detector_silence_timeout_completes_open_turn() -> None:
    async def run() -> turn_detector.TurnCompleteData:
        ability = RecordingTurnDetector(end_of_turn_silence_seconds=0.03)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        await hsm.dispatch(context, ability, _start(text="timed"))
        await wait_until(lambda: len(getattr(ability, "outputs", [])) == 1, timeout=1.0)
        return ability.outputs[0]

    output = asyncio.run(run())
    assert output.text == "timed"


def test_turn_detector_explicit_end_without_content_emits_failure() -> None:
    async def run() -> turn_detector.FailedEventData:
        ability = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        operation_id = uuid.uuid4().hex
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnStartEvent.with_data_and_id(
                turn_detector.TurnStartData(turn_ref=TURN_REF, source_participant_ref="caller"),
                uuid.uuid4().hex,
            ),
        )
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/open"))
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnEndEvent.with_data_and_id(
                turn_detector.TurnEndData(turn_ref=TURN_REF, source_participant_ref="caller", content=None),
                operation_id,
            ),
        )
        await wait_until(lambda: bool(getattr(ability, "failures", [])))
        failure = ability.failures[-1]
        assert isinstance(failure, turn_detector.FailedEventData)
        return failure

    failure = asyncio.run(run())
    assert failure.stage == "turn"
    assert "neither text, audio, modality-neutral content, nor images" in failure.message


def test_turn_detector_rejects_stale_turn_and_source_boundaries() -> None:
    async def run() -> turn_detector.TurnCompleteData:
        ability = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        await hsm.dispatch(context, ability, _start(text="before"))
        await wait_until(lambda: (ability.state() or "").endswith("/behavior/open"))
        for data in (
            turn_detector.TurnUpdateData(
                turn_ref="stale-turn",
                source_participant_ref="caller",
                content=turn_detector.TextStimulus(source_participant_ref="caller", content="wrong-turn"),
            ),
            turn_detector.TurnUpdateData(
                turn_ref=TURN_REF,
                conversation_ref="other-conversation",
                source_participant_ref="caller",
                content=turn_detector.TextStimulus(source_participant_ref="caller", content="wrong-conversation"),
            ),
            turn_detector.TurnUpdateData(
                turn_ref=TURN_REF,
                source_participant_ref="other-caller",
                content=turn_detector.TextStimulus(source_participant_ref="other-caller", content="wrong-source"),
            ),
        ):
            await hsm.dispatch(context, ability, turn_detector.TurnUpdateEvent.with_data(data))
        await hsm.dispatch(
            context,
            ability,
            turn_detector.TurnEndEvent.with_data(turn_detector.TurnEndData(turn_ref=TURN_REF)),
        )
        await wait_until(lambda: len(ability.outputs) == 1)
        return ability.outputs[0]

    output = asyncio.run(run())
    assert output.text == "before"


def test_turn_complete_rejects_mixed_text_and_audio() -> None:
    with pytest.raises(ValueError, match="Turn product cannot contain both text and audio"):
        _ = turn_detector.TurnCompleteData(turn_ref=TURN_REF, text="hello", audio=b"audio")


def test_turn_complete_rejects_empty_product() -> None:
    with pytest.raises(ValueError, match="Turn product requires text, audio, modality-neutral content, or images"):
        _ = turn_detector.TurnCompleteData(turn_ref=TURN_REF, text="  ", audio=b"")


def test_turn_complete_accepts_images_alone_and_with_a_caption() -> None:
    image = _image()

    image_only = turn_detector.TurnCompleteData(turn_ref=TURN_REF, images=(image,))
    captioned = turn_detector.TurnCompleteData(turn_ref=TURN_REF, text="look", images=(image,))

    assert image_only.images == (image,)
    assert (image_only.text, image_only.audio, image_only.content) == ("", b"", None)
    assert captioned.text == "look"
    assert captioned.images == (image,)


def test_turn_complete_bounds_images_per_turn() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = turn_detector.TurnCompleteData(
            turn_ref=TURN_REF,
            images=tuple(_image(index) for index in range(media.MAX_TURN_IMAGES + 1)),
        )


def test_turn_detector_emits_structured_content_without_text_or_audio() -> None:
    data = turn_detector.TurnCompleteData(
        turn_ref=TURN_REF,
        source_participant_ref="caller",
        content={"gesture": "wave"},
        content_type="application/x-sign-language",
    )

    assert data.text == ""
    assert data.audio == b""
    assert data.content == {"gesture": "wave"}
    assert data.content_type == "application/x-sign-language"


def test_turn_detector_boundaries_canonicalize_participant_references_and_refuse_vectors() -> None:
    data = turn_detector.TurnStartData(
        turn_ref=TURN_REF,
        self_participant_ref=" bot ",
        source_participant_ref="voice:3f2a9c",
        content=turn_detector.TextStimulus(source_participant_ref=" voice:3f2a9c ", content="hello"),
    )

    assert data.self_participant_ref == "bot"
    assert isinstance(data.content, turn_detector.TextStimulus)
    assert data.content.source_participant_ref == "voice:3f2a9c"
    with pytest.raises(pydantic.ValidationError):
        _ = turn_detector.TextStimulus(source_participant_ref=typing.cast(typing.Any, (0.1, -0.2)), content="hello")


@pytest.mark.parametrize(
    "data_factory",
    [
        lambda: turn_detector.TurnStartData(
            turn_ref=TURN_REF,
            source_participant_ref="caller",
            content=turn_detector.TextStimulus(source_participant_ref="other-caller", content="hello"),
        ),
        lambda: turn_detector.TurnUpdateData(
            turn_ref=TURN_REF,
            source_participant_ref="caller",
            content=turn_detector.AudioStimulus(source_participant_ref="other-caller", content=b"audio"),
        ),
        lambda: turn_detector.TurnEndData(
            turn_ref=TURN_REF,
            source_participant_ref="caller",
            content=turn_detector.TextStimulus(source_participant_ref="other-caller", content="goodbye"),
        ),
    ],
)
def test_turn_boundaries_reject_mismatched_nested_source(
    data_factory: collections.abc.Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="content source_participant_ref must match boundary"):
        _ = data_factory()


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Route this module's spans to an in-memory exporter for the duration of one test."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def _spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [recorded for recorded in exporter.get_finished_spans() if recorded.name == name]


def _image_chunk(image: media.Image, *, turn_ref: str = TURN_REF) -> hsm.Event[turn_detector.TurnUpdateData]:
    return turn_detector.TurnUpdateEvent.with_data_and_id(
        turn_detector.TurnUpdateData(
            turn_ref=turn_ref,
            source_participant_ref="caller",
            content=turn_detector.ImageStimulus(source_participant_ref="caller", content=image),
        ),
        uuid.uuid4().hex,
    )


async def _run_turn(
    detector: RecordingTurnDetector,
    context: hsm.Context,
    *,
    first: turn_detector.ParticipationStimulus | None,
    updates: collections.abc.Sequence[media.Image] = (),
    last: turn_detector.ParticipationStimulus | None = None,
    turn_ref: str = TURN_REF,
) -> None:
    _ = await hsm.dispatch(
        context,
        detector,
        turn_detector.TurnStartEvent.with_data_and_id(
            turn_detector.TurnStartData(turn_ref=turn_ref, source_participant_ref="caller", content=first),
            uuid.uuid4().hex,
        ),
    )
    for image in updates:
        _ = await hsm.dispatch(context, detector, _image_chunk(image, turn_ref=turn_ref))
    _ = await hsm.dispatch(
        context,
        detector,
        turn_detector.TurnEndEvent.with_data_and_id(
            turn_detector.TurnEndData(turn_ref=turn_ref, source_participant_ref="caller", content=last),
            uuid.uuid4().hex,
        ),
    )


def _image_stimulus(image: media.Image) -> turn_detector.ImageStimulus:
    return turn_detector.ImageStimulus(source_participant_ref="caller", content=image)


def test_turn_detector_preserves_a_typed_image_and_its_media_type() -> None:
    image = media.Image(content=_PNG_SIGNATURE + b"photo", media_type="image/png", width_px=4, height_px=3)

    async def run() -> turn_detector.TurnCompleteData:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(image))
        await wait_until(lambda: len(detector.outputs) == 1)
        return detector.outputs[0]

    output = asyncio.run(run())
    assert output.images == (image,)
    assert output.images[0].media_type == "image/png"
    assert (output.images[0].width_px, output.images[0].height_px) == (4, 3)
    assert (output.text, output.audio, output.content, output.content_type) == ("", b"", None, None)


def test_turn_detector_assembles_a_caption_with_its_images() -> None:
    images = (_image(1), _image(2))

    async def run() -> turn_detector.TurnCompleteData:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(
            detector,
            context,
            first=turn_detector.TextStimulus(source_participant_ref="caller", content="look at these"),
            updates=images,
        )
        await wait_until(lambda: len(detector.outputs) == 1)
        return detector.outputs[0]

    output = asyncio.run(run())
    assert output.text == "look at these"
    assert output.images == images


def test_turn_detector_with_decoder_completes_an_image_only_turn_without_decoding(
    finished_spans: InMemorySpanExporter,
) -> None:
    image = _image()
    decoder = RecordingDecoder()

    async def run() -> tuple[list[turn_detector.TurnCompleteData], list[turn_detector.FailedEventData]]:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0, decoder=decoder)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(image))
        await wait_until(lambda: bool(detector.outputs or detector.failures))
        return detector.outputs, detector.failures

    outputs, failures = asyncio.run(run())
    assert failures == []
    (output,) = outputs
    assert output.images == (image,)
    assert output.text == ""
    assert decoder.calls == []
    (normalize,) = _spans_named(finished_spans, "bot.turn.normalize")
    attributes = dict(normalize.attributes or {})
    assert attributes["bot.turn.decode.skipped"] is True
    assert attributes["bot.turn.decoder.present"] is True


def test_turn_detector_with_decoder_decodes_the_caption_and_keeps_its_images() -> None:
    image = _image()
    decoder = RecordingDecoder()

    async def run() -> turn_detector.TurnCompleteData:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0, decoder=decoder)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(
            detector,
            context,
            first=turn_detector.TextStimulus(source_participant_ref="caller", content="what is this"),
            updates=(image,),
        )
        await wait_until(lambda: len(detector.outputs) == 1)
        return detector.outputs[0]

    output = asyncio.run(run())
    assert decoder.calls == [turn_detector.TextStimulus(source_participant_ref="caller", content="what is this")]
    assert output.text == "decoded"
    assert output.images == (image,)


def test_turn_detector_with_decoder_decodes_audio_and_keeps_its_images() -> None:
    image = _image()
    decoder = RecordingDecoder()

    async def run() -> turn_detector.TurnCompleteData:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0, decoder=decoder)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(
            detector,
            context,
            first=turn_detector.AudioStimulus(source_participant_ref="caller", content=b"pcm", sample_rate_hz=16_000),
            updates=(image,),
        )
        await wait_until(lambda: len(detector.outputs) == 1)
        return detector.outputs[0]

    output = asyncio.run(run())
    (call,) = decoder.calls
    assert isinstance(call, turn_detector.AudioStimulus)
    assert call.content == b"pcm"
    assert output.text == "decoded"
    assert output.audio == b""
    assert output.images == (image,)


@pytest.mark.parametrize("overflow_on_end", [False, True], ids=["update", "end"])
def test_turn_detector_fails_a_turn_with_more_images_than_a_turn_carries(
    overflow_on_end: bool,
    finished_spans: InMemorySpanExporter,
) -> None:
    images = tuple(_image(index) for index in range(media.MAX_TURN_IMAGES + 1))
    updates = images[1:-1] if overflow_on_end else images[1:]
    last = _image_stimulus(images[-1]) if overflow_on_end else None

    async def run() -> tuple[list[turn_detector.TurnCompleteData], list[turn_detector.FailedEventData]]:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(images[0]), updates=updates, last=last)
        await wait_until(lambda: bool(detector.outputs or detector.failures))
        return detector.outputs, detector.failures

    outputs, failures = asyncio.run(run())
    assert outputs == []
    (failure,) = failures
    assert failure.stage == "turn"
    assert failure.message == f"Turn exceeds {media.MAX_TURN_IMAGES} images."
    (end,) = _spans_named(finished_spans, "bot.turn.end")
    assert dict(end.attributes or {})["bot.failure.kind"] == "turn_image_overflow"


def test_turn_detector_reattached_after_an_image_overflow_completes_a_single_image_turn() -> None:
    # A close failure leaves a standalone detector in normalizing until detach (a pre-existing
    # defect tracked outside this change). Reattaching the same instance is the one path where the
    # failed turn's state can leak into the next turn, so it proves the failure path clears it.
    images = tuple(_image(index) for index in range(media.MAX_TURN_IMAGES + 1))
    single = _image(100)

    async def run() -> tuple[list[turn_detector.FailedEventData], list[turn_detector.TurnCompleteData]]:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(images[0]), updates=images[1:])
        await wait_until(lambda: bool(detector.failures))
        owner = ability_terminal_owner(detector)
        assert owner is not None
        _ = await detector.detach(context, attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)))
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(single), turn_ref="turn-after-overflow")
        await wait_until(lambda: bool(detector.outputs) or len(detector.failures) > 1)
        return detector.failures, detector.outputs

    failures, outputs = asyncio.run(run())
    (failure,) = failures
    assert failure.message == f"Turn exceeds {media.MAX_TURN_IMAGES} images."
    (output,) = outputs
    assert output.turn_ref == "turn-after-overflow"
    assert output.images == (single,)


def test_turn_detector_clears_images_between_turns() -> None:
    first_images = (_image(1), _image(2))
    second_image = _image(3)

    async def run() -> list[turn_detector.TurnCompleteData]:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(first_images[0]), updates=first_images[1:])
        await wait_until(lambda: len(detector.outputs) == 1)
        await _run_turn(detector, context, first=_image_stimulus(second_image), turn_ref="turn-2")
        await wait_until(lambda: len(detector.outputs) == 2)
        return detector.outputs

    first, second = asyncio.run(run())
    assert first.images == first_images
    assert second.turn_ref == "turn-2"
    assert second.images == (second_image,)


def test_turn_end_span_counts_the_turn_images(finished_spans: InMemorySpanExporter) -> None:
    async def run() -> None:
        detector = RecordingTurnDetector(end_of_turn_silence_seconds=30.0)
        context = hsm.Context()
        await start_ability_tree(context, detector)
        await _run_turn(detector, context, first=_image_stimulus(_image(1)), updates=(_image(2),))
        await wait_until(lambda: len(detector.outputs) == 1)

    asyncio.run(run())
    (end,) = _spans_named(finished_spans, "bot.turn.end")
    assert dict(end.attributes or {})["bot.turn.image.count"] == 2
