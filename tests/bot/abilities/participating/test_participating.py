from bot import abilities
from bot.abilities import participating
from bot.abilities import reading
from bot.abilities.hearing import voice

import asyncio
import collections.abc
import typing

import hsm
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

from tests.hsm_instance_state import start_ability_tree
from tests.type_helpers import model_view, object_dict

class RecordingParticipating(participating.Participating):
    outputs: list[participating.participating.OutputData]
    failures: list[participating.participating.FailedEventData]

    def __init__(
        self,
        *,
        listening: participating.AudioPerception | None = None,
        reading: participating.ReadablePerception | None = None,
    ) -> None:
        super().__init__(listening=listening, reading=reading)
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, participating.participating.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, participating.participating.FailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)

class TextReading(participating.ReadablePerception):
    @typing.override
    async def _apply(self, ctx: hsm.Context, input: reading.reading.InputData) -> reading.reading.OutputData:
        del ctx
        assert input.kind == "text"
        assert isinstance(input.content, str)
        return reading.reading.OutputData(text=input.content.upper(), source_kind="text", confidence=0.99)

class HangingTextReading(participating.ReadablePerception):
    @typing.override
    async def _apply(self, ctx: hsm.Context, input: reading.reading.InputData) -> reading.reading.OutputData:
        del ctx, input
        _ = await asyncio.Event().wait()
        raise AssertionError("unreachable")

class ImageReading(participating.ReadablePerception):
    @typing.override
    async def _apply(self, ctx: hsm.Context, input: reading.reading.InputData) -> reading.reading.OutputData:
        del ctx
        assert input.kind == "image"
        assert input.content == b"image bytes"
        return reading.reading.OutputData(text="diagram label", source_kind="image", confidence=0.87)

class VoiceListening(participating.AudioPerception):
    @typing.override
    async def _apply(self, ctx: hsm.Context, input: bytes) -> participating.AudioPerceptionData:
        del ctx
        assert input == b"audio bytes"
        return participating.AudioPerceptionData(
            voice_detection=voice.detection.OutputData(is_voice=True, confidence=0.91),
            speech=b"decoded speech",
        )

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)

def participant() -> participating.ParticipantSnapshot:
    return participating.ParticipantSnapshot(
        ref="bot",
        kind="bot",
        state=participating.ParticipantStateSnapshot(
            presence="present",
            attention="available",
            turn="listening",
        ),
    )

def text_input(content: str, *, conversation_ref: str = "conversation") -> participating.participating.InputData:
    return participating.participating.InputData(
        conversation_ref=conversation_ref,
        self_participant_ref="bot",
        participants=(participant(),),
        stimulus=participating.TextStimulus(source_participant_ref="human", content=content),
    )

def test_participating_defines_multimodal_operation_contract() -> None:
    ability = participating.Participating(reading=TextReading())
    input_schema = object_dict(participating.ParticipatingInputEvent.schema)
    output_schema = object_dict(participating.ParticipatingOutputEvent.schema)
    failed_schema = object_dict(participating.ParticipatingFailedEvent.schema)

    assert isinstance(ability, abilities.Ability)
    assert participating.Participating.input_event is participating.ParticipatingInputEvent
    assert participating.Participating.output_event is participating.ParticipatingOutputEvent
    assert participating.Participating.failed_event is participating.ParticipatingFailedEvent
    assert input_schema == participating.participating.InputData.model_json_schema()
    assert output_schema == participating.participating.OutputData.model_json_schema()
    assert failed_schema == participating.participating.FailedEventData.model_json_schema()
    assert input_schema["description"]
    assert output_schema["description"]
    assert failed_schema["description"]

def test_participating_requires_operation_backed_perception_lanes() -> None:
    with pytest.raises(TypeError, match="listening must be an AudioPerception ability"):
        _ = participating.Participating(listening=typing.cast(participating.AudioPerception, abilities.Ability[bytes, participating.AudioPerceptionData]()))

    with pytest.raises(TypeError, match="reading must be a ReadablePerception ability"):
        _ = participating.Participating(
            reading=typing.cast(
                participating.ReadablePerception, abilities.Ability[reading.reading.InputData, reading.reading.OutputData]()
            )
        )

def test_participating_apply_bridge_keeps_operation_state_out_of_instance() -> None:
    ability = participating.Participating(reading=TextReading())

    assert "_pending_apply_results" not in vars(ability)
    assert "_active_apply_operation_id" not in vars(ability)

def test_participating_keeps_perception_lanes_on_instance() -> None:
    listening = VoiceListening()
    reading = TextReading()
    ability = participating.Participating(listening=listening, reading=reading)

    state = vars(ability)

    assert state["_listening"] is listening
    assert state["_reading"] is reading

def test_participating_model_tracks_perception_and_contribution_lifecycle() -> None:
    model = model_view(require_model(participating.Participating.model))

    assert model.qualified_name == "/ParticipatingLifecycle"
    assert model.initial == "/ParticipatingLifecycle/.initial"
    assert "/ParticipatingLifecycle/detached" in model.members
    assert "/ParticipatingLifecycle/attaching" in model.members
    assert "/ParticipatingLifecycle/attached" in model.members
    assert "/ParticipatingLifecycle/attached/behavior/initializing" in model.members
    assert "/ParticipatingLifecycle/attached/behavior/idle" in model.members
    assert "/ParticipatingLifecycle/attached/behavior/perceiving" in model.members
    assert "/ParticipatingLifecycle/attached/behavior/contributing" in model.members
    assert "bot.ability.participating.input" in model.transition_map["/ParticipatingLifecycle/attached/behavior/idle"]
    assert (
        "bot.ability.participating.perception.completed"
        in model.transition_map["/ParticipatingLifecycle/attached/behavior/perceiving"]
    )
    assert (
        "bot.ability.participating.contribution.completed"
        in model.transition_map["/ParticipatingLifecycle/attached/behavior/contributing"]
    )

def test_participating_apply_reads_text_stimulus_into_contribution() -> None:
    async def run() -> participating.participating.OutputData:
        ability = participating.Participating(reading=TextReading())
        await start_ability_tree(None, ability)

        return await dispatch_ability_for_test(ability, hsm.Context(), text_input("hello"))

    output = asyncio.run(run())

    assert output == participating.participating.OutputData(
        participant_ref="bot",
        contribution=participating.ParticipantContribution(
            conversation_ref="conversation",
            participant_ref="human",
            perception=participating.Perception(
                source_participant_ref="human",
                modality="text",
                readable="HELLO",
                confidence=0.99,
            ),
        ),
    )

def test_participating_apply_contributes_device_event_without_perception_lanes() -> None:
    async def run() -> participating.participating.OutputData:
        ability = participating.Participating()
        await start_ability_tree(None, ability)
        input = participating.participating.InputData(
            conversation_ref="conversation",
            self_participant_ref="bot",
            participants=(participant(),),
            stimulus=participating.EventStimulus(
                source_participant_ref="phone",
                event="conversation.reading",
                payload={"text": "hello"},
            ),
        )

        return await dispatch_ability_for_test(ability, hsm.Context(), input)

    output = asyncio.run(run())

    assert output == participating.participating.OutputData(
        participant_ref="bot",
        contribution=participating.ParticipantContribution(
            conversation_ref="conversation",
            participant_ref="phone",
            perception=participating.Perception(
                source_participant_ref="phone",
                modality="event",
                structured={"event": "conversation.reading", "payload": {"text": "hello"}},
            ),
        ),
    )

def test_participating_reads_text_stimulus_into_contribution() -> None:
    async def run() -> list[participating.participating.OutputData]:
        ability = RecordingParticipating(reading=TextReading())
        await start_ability_tree(None, ability)

        _ = await ability.apply(text_input("hello"))
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [
        participating.participating.OutputData(
            participant_ref="bot",
            contribution=participating.ParticipantContribution(
                conversation_ref="conversation",
                participant_ref="human",
                perception=participating.Perception(
                    source_participant_ref="human",
                    modality="text",
                    readable="HELLO",
                    confidence=0.99,
                ),
            ),
        )
    ]

def test_participating_detach_releases_owned_perception_while_reading() -> None:
    async def run() -> tuple[str, str]:
        ctx = hsm.Context()
        reading = HangingTextReading()
        ability = RecordingParticipating(reading=reading)
        await start_ability_tree(ctx, ability)
        _ = await ability.apply(text_input("hello"), ctx=ctx)
        await wait_until(
            lambda: ability.state() == "/RecordingParticipatingLifecycle/attached/behavior/perceiving/reading_text"
        )
        assert ability.state() == "/RecordingParticipatingLifecycle/attached/behavior/perceiving/reading_text"

        _ = await ability.detach(ctx=ctx)
        await wait_until(lambda: reading.state().endswith("/detached"))
        reading_state = reading.state()
        state = ability.state()
        await ability.stop(ctx)
        return reading_state, state

    reading_state, state = asyncio.run(run())

    assert reading_state.endswith("/detached")
    assert state == "/RecordingParticipatingLifecycle/detached"

def test_participating_reads_image_stimulus_into_contribution() -> None:
    async def run() -> list[participating.participating.OutputData]:
        ability = RecordingParticipating(reading=ImageReading())
        await start_ability_tree(None, ability)

        _ = await ability.apply(
            participating.participating.InputData(
                conversation_ref="conversation",
                self_participant_ref="bot",
                participants=(participant(),),
                stimulus=participating.ImageStimulus(source_participant_ref="human", content=b"image bytes"),
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].contribution == participating.ParticipantContribution(
        conversation_ref="conversation",
        participant_ref="human",
        perception=participating.Perception(
            source_participant_ref="human",
            modality="image",
            readable="diagram label",
            confidence=0.87,
        ),
    )

def test_participating_requires_reading_for_readable_stimuli() -> None:
    async def run() -> list[participating.participating.FailedEventData]:
        ability = RecordingParticipating()
        await start_ability_tree(None, ability)

        with pytest.raises(RuntimeError, match="requires Reading"):
            _ = await dispatch_ability_for_test(
                ability,
                hsm.Context(),
                participating.participating.InputData(
                    conversation_ref="conversation",
                    self_participant_ref="bot",
                    participants=(participant(),),
                    stimulus=participating.ImageStimulus(source_participant_ref="human", content=b"image bytes"),
                ),
            )
        await wait_until(lambda: bool(ability.failures))
        return ability.failures

    failures = asyncio.run(run())

    assert failures == [
        participating.participating.FailedEventData(stage="perception", message="Participating requires Reading for image stimuli.")
    ]

def test_participating_listens_to_audio_stimulus_into_contribution() -> None:
    async def run() -> list[participating.participating.OutputData]:
        ability = RecordingParticipating(listening=VoiceListening())
        await start_ability_tree(None, ability)

        _ = await ability.apply(
            participating.participating.InputData(
                conversation_ref="conversation",
                self_participant_ref="bot",
                participants=(participant(),),
                stimulus=participating.AudioStimulus(source_participant_ref="human", content=b"audio bytes"),
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].contribution.perception == participating.Perception(
        source_participant_ref="human",
        modality="audio",
        speech=b"decoded speech",
        confidence=0.91,
    )

def test_participating_preserves_structured_device_event_perception() -> None:
    async def run() -> list[participating.participating.OutputData]:
        ability = RecordingParticipating()
        await start_ability_tree(None, ability)

        _ = await ability.apply(
            participating.participating.InputData(
                conversation_ref="conversation",
                self_participant_ref="bot",
                participants=(participant(),),
                stimulus=participating.EventStimulus(
                    source_participant_ref="phone",
                    event="call.started",
                    payload={"line": "support"},
                ),
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].contribution.perception == participating.Perception(
        source_participant_ref="phone",
        modality="event",
        structured={"event": "call.started", "payload": {"line": "support"}},
    )

def test_participating_rejects_self_participant_missing_from_participants() -> None:
    with pytest.raises(ValueError):
        _ = participating.participating.InputData(
            conversation_ref="conversation",
            self_participant_ref="bot",
            participants=(),
            stimulus=participating.TextStimulus(source_participant_ref="human", content="hello"),
        )
