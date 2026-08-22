from __future__ import annotations

from bot import abilities
from bot.abilities.communication.conversation import turn_detector

import asyncio
import collections.abc
import datetime
import uuid

import hsm
import pytest

from tests.hsm_instance_state import start_ability_tree

TURN_REF = "turn-1"


class RecordingTurnDetector(turn_detector.TurnDetector):
    outputs: list[turn_detector.TurnCompleteData]
    failures: list[turn_detector.FailedEventData]

    def __init__(
        self,
        *,
        participant_ref: str = "bot",
        conversation_ref: str = "conversation",
        end_of_turn_silence_seconds: float = 0.5,
    ) -> None:
        super().__init__(
            participant_ref=participant_ref,
            conversation_ref=conversation_ref,
            end_of_turn_silence_seconds=end_of_turn_silence_seconds,
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
    assert "neither text, audio, nor modality-neutral content" in failure.message


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
    with pytest.raises(ValueError, match="Turn product requires text, audio, or modality-neutral content"):
        _ = turn_detector.TurnCompleteData(turn_ref=TURN_REF, text="  ", audio=b"")


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


def test_turn_detector_boundaries_canonicalize_embedding_participant_references() -> None:
    embedding = (0.1, -0.2)
    data = turn_detector.TurnStartData(
        turn_ref=TURN_REF,
        self_participant_ref=embedding,
        source_participant_ref=embedding,
        content=turn_detector.TextStimulus(source_participant_ref=embedding, content="hello"),
    )

    assert data.self_participant_ref == (0.1, -0.2)
    assert data.source_participant_ref == (0.1, -0.2)
    assert isinstance(data.content, turn_detector.TextStimulus)
    assert data.content.source_participant_ref == (0.1, -0.2)
    assert data.model_dump_json().count("[0.1,-0.2]") == 3


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
