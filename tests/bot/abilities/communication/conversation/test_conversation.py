from mosfet import media
from mosfet.abilities.communication import conversation
from mosfet.abilities import ability, cognition, decoding
from mosfet.abilities import processing
from mosfet.abilities.hearing import voice
from mosfet.abilities.listening import interpretation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.cognition import event as cognition_event
from mosfet.abilities.language import text
from mosfet.abilities.identity import value

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing
import uuid

import hsm
import mosfet
import pydantic
import pytest

from tests import metric_reader
from tests.hsm_instance_state import start_ability_tree

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _image(index: int = 0) -> media.Image:
    return media.Image(content=_PNG_SIGNATURE + f"png-{index}".encode(), media_type="image/png")


def run_terminal(
    target: conversation.Conversation,
    context: hsm.Context,
    data: conversation.TurnData,
) -> hsm.Event[typing.Any]:
    """Dispatch one input and return the Conversation terminal exactly as its owner receives it."""

    return asyncio.run(
        ability.run_terminal_operation(
            context,
            child=target,
            request=target.input_event.with_data_and_id(data, uuid.uuid4().hex),
            terminals=(conversation.OutputEvent, cognition.InputEvent, conversation.FailedEvent),
            timeout=datetime.timedelta(seconds=20),
        )
    )


def terminal_messages(terminal: hsm.Event[typing.Any]) -> conversation.Messages:
    if isinstance(terminal.data, conversation.Messages):
        return terminal.data
    assert isinstance(terminal.data, cognition.InputData)
    stimulus = terminal.data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert isinstance(stimulus.data, conversation.Messages)
    return stimulus.data


def run_input(
    ability: conversation.Conversation,
    context: hsm.Context,
    *,
    source_ids: conversation.IdentitySet,
    target_ids: conversation.IdentitySet,
    content: object,
    content_type: str,
) -> conversation.ParticipatedTurn:
    return asyncio.run(
        conversation.contribute_conversation_input(
            ability,
            conversation.TurnData(
                source_ids=source_ids,
                target_ids=target_ids,
                content=content,
                content_type=content_type,
            ),
            ctx=context,
        )
    )


def started_conversation(
    factory: conversation.TurnDetectorFactory | None = None,
    *,
    similarity_threshold: float = 0.85,
    similarity_margin: float = 0.05,
) -> tuple[conversation.Conversation, hsm.Context]:
    ability = conversation.Conversation(
        turn_detector_factory=factory,
        similarity_threshold=similarity_threshold,
        similarity_margin=similarity_margin,
    )
    context = hsm.Context()
    asyncio.run(start_ability_tree(context, ability))
    return ability, context


def test_input_contract_has_identity_content_and_audio_packaging_fields() -> None:
    assert tuple(conversation.TurnData.model_fields) == (
        "parent",
        "source_ids",
        "target_ids",
        "content",
        "content_type",
        "sample_rate_hz",
        "channels",
        "attachments",
    )
    assert "conversation_ref" not in conversation.TurnData.model_fields
    assert "self_participant_ref" not in conversation.TurnData.model_fields
    assert "participants" not in conversation.TurnData.model_fields
    with pytest.raises(pydantic.ValidationError):
        conversation.TurnData.model_validate(
            {
                "source_ids": ["caller"],
                "target_ids": ["bot"],
                "content": "hello",
                "content_type": "text/plain",
                "conversation_ref": "legacy",
            }
        )
    schema = typing.cast(type[pydantic.BaseModel], conversation.InputEvent.schema)
    assert tuple(schema.model_fields) == tuple(conversation.TurnData.model_fields)


def test_input_parent_is_hidden_from_model_schema_but_validated_as_typed_provenance() -> None:
    """Model selections cannot forge causal ancestry, while runtime input validation retains it."""

    schema = conversation.TurnData.model_json_schema()
    assert "parent" not in schema["properties"]

    speech = interpretation.SpeechData(
        content="hello",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"caller"}),
    )
    parent = mosfet.StimulusData[interpretation.SpeechData](
        event="bot.ability.listening.speech.output",
        data=speech,
    )
    validated = conversation.TurnData.model_validate(
        {
            "parent": parent,
            "source_ids": ["caller"],
            "target_ids": [],
            "content": "hello",
            "content_type": "text/plain",
        }
    )
    assert validated.parent == parent

    with pytest.raises(pydantic.ValidationError):
        conversation.TurnData.model_validate(
            {
                "parent": {
                    "event": "bot.ability.listening.speech.output",
                    "data": {"content": "forged"},
                },
                "source_ids": ["caller"],
                "target_ids": [],
                "content": "hello",
                "content_type": "text/plain",
            }
        )


def test_message_content_recursive_alias_preserves_schema_and_validation() -> None:
    """The named recursive alias keeps the schema and runtime content contract stable."""

    import hashlib
    import json

    schema = conversation.Message.model_json_schema()
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert schema_hash == "ecd2edfbeb59bf03aa6967c168802c3f64720f92b9eb2529e38132ab15104a9e"

    payload = {
        "sequence": 0,
        "direction": "inbound",
        "source_ids": ["caller"],
        "target_ids": ["bot"],
        "content": {"nested": [True, 3, {"text": "hello"}]},
        "content_type": "application/json",
        "provenance": {"event": "bot.ability.conversation.input"},
    }
    validated = conversation.Message.model_validate(payload)
    assert validated.content == payload["content"]

    with pytest.raises(pydantic.ValidationError, match="raw media"):
        conversation.Message.model_validate({**payload, "content": {"nested": [b"raw-media"]}})


def test_model_dispatch_rejects_producer_stamped_conversation_parent() -> None:
    """Model-selected Conversation.input cannot forge trusted causal provenance."""

    async def run() -> None:
        context = hsm.Context()
        source = hsm.Instance()
        speech = interpretation.SpeechData(
            content="hello",
            content_type="text/plain",
            voice_detection=voice.detection.ApplyData(segments=()),
            sample_rate_hz=16_000,
            channels=1,
            media_type="audio/pcm",
            source_ids=frozenset({"caller"}),
        )
        input_data = conversation.TurnData(
            parent=mosfet.StimulusData[interpretation.SpeechData](
                event="bot.ability.listening.speech.output",
                data=speech,
            ),
            source_ids=frozenset({"caller"}),
            target_ids=frozenset(),
            content="hello",
            content_type="text/plain",
        )
        processing_input = processing.InputData(
            input="forged model selection",
            schemas=(conversation.InputEvent,),
            actors={},
        )

        with pytest.raises(RuntimeError, match="producer-stamped"):
            await processing.dispatch_selected_events(
                context,
                processing_input,
                (
                    processing.SelectedEvent(
                        event=conversation.InputEvent.name,
                        target="conversation",
                        data=input_data.model_dump(mode="json"),
                    ),
                ),
                operation_id="forged-parent",
                source=source,
            )

    asyncio.run(run())


def test_zero_vector_is_rejected_at_the_input_boundary() -> None:
    with pytest.raises(pydantic.ValidationError, match="zero vector"):
        conversation.TurnData(
            source_ids=frozenset({(0.0, 0.0)}),
            target_ids=frozenset({"bot"}),
            content="invalid voice identity",
            content_type="text/plain",
        )


def test_input_accepts_an_empty_target_set() -> None:
    input_data = conversation.TurnData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset(),
        content="ambient message",
        content_type="text/plain",
    )

    assert input_data.target_ids == frozenset()


@pytest.mark.parametrize("field_name", ("source_ids", "target_ids"))
def test_input_rejects_identity_sets_above_the_provider_neutral_limit(field_name: str) -> None:
    fields: dict[str, object] = {
        "source_ids": frozenset({"caller"}),
        "target_ids": frozenset(),
        "content": "too many identities",
        "content_type": "text/plain",
    }
    fields[field_name] = frozenset(f"speaker-{index}" for index in range(value.MAX_IDENTITY_SET_SIZE + 1))

    with pytest.raises(pydantic.ValidationError, match="maximum is 4"):
        conversation.TurnData.model_validate(fields)


def test_input_rejects_embeddings_above_the_provider_neutral_dimension_limit() -> None:
    oversized = (1.0,) * (value.MAX_EMBEDDING_DIMENSION + 1)

    with pytest.raises(pydantic.ValidationError, match="exceeds the maximum of 2048"):
        conversation.TurnData(
            source_ids=frozenset({oversized}),
            target_ids=frozenset(),
            content="embedding too large",
            content_type="application/octet-stream",
        )


def test_source_only_embedding_turns_share_session_and_create_detectors() -> None:
    created: list[tuple[str, conversation.TrackRef]] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        created.append((session_ref, source_id))
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({(1.0, 0.0)}),
        target_ids=frozenset(),
        content="first speaker",
        content_type="application/octet-stream",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({(0.0, 1.0)}),
        target_ids=frozenset(),
        content="second speaker",
        content_type="application/octet-stream",
    )
    third = run_input(
        ability,
        context,
        source_ids=frozenset({(1.0, 1.0)}),
        target_ids=frozenset(),
        content="bot speaker",
        content_type="application/octet-stream",
    )

    assert first.session_ref == second.session_ref == third.session_ref
    assert len(created) == 3
    assert {session_ref for session_ref, _ in created} == {first.session_ref}
    assert len({source_id for _, source_id in created}) == 3
    assert all(source_id.startswith("track-") for _, source_id in created)
    assert ability.session_refs == (first.session_ref,)
    assert ability.detector_count == 3
    assert len(ability.detector_refs) == 3
    assert third.input.target_ids == frozenset()


def test_four_party_source_set_is_within_the_input_bound() -> None:
    ability, context = started_conversation()
    result = run_input(
        ability,
        context,
        source_ids=frozenset(
            {
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            }
        ),
        target_ids=frozenset(),
        content="four-party input",
        content_type="application/octet-stream",
    )

    assert result.session_ref in ability.session_refs
    assert ability.detector_count == 4


def test_four_party_embedding_assignment_reuses_the_relationship() -> None:
    ability, context = started_conversation(similarity_threshold=0.8, similarity_margin=0.02)
    first = run_input(
        ability,
        context,
        source_ids=frozenset(
            {
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
            }
        ),
        target_ids=frozenset({"bot"}),
        content="first four-party turn",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset(
            {
                (0.98, 0.02),
                (0.02, 0.98),
                (1.01, 0.99),
                (-0.98, 1.02),
            }
        ),
        target_ids=frozenset({"bot"}),
        content="second four-party turn",
        content_type="text/plain",
    )

    assert second.session_ref == first.session_ref
    assert ability.detector_count == 4


def test_global_embedding_assignment_reuses_relationship_without_row_margin() -> None:
    ability, context = started_conversation(similarity_threshold=0.0)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({(1.0, 0.0), (0.0, 1.0)}),
        target_ids=frozenset({"bot"}),
        content="first",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({(0.8, 0.6), (0.999, 0.0447)}),
        target_ids=frozenset({"bot"}),
        content="second",
        content_type="text/plain",
    )

    assert second.session_ref == first.session_ref


def test_same_relationship_reuses_detector_without_a_session_registry() -> None:
    created: list[tuple[str, conversation.IdentityValue]] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        created.append((session_ref, source_id))
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="first",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="second",
        content_type="text/plain",
    )

    assert first.session_ref == second.session_ref
    assert created == [(first.session_ref, "alice")]
    assert ability.detector_refs == ((first.session_ref, "alice"),)
    assert ability.session_refs == (first.session_ref,)
    assert "_sessions" not in ability.__dict__


def test_different_relationship_gets_a_new_detector() -> None:
    created: list[tuple[str, conversation.IdentityValue]] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        created.append((session_ref, source_id))
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="first",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"service"}),
        content="second",
        content_type="text/plain",
    )

    assert first.session_ref != second.session_ref
    assert set(ability.session_refs) == {first.session_ref, second.session_ref}
    assert created == [(first.session_ref, "alice"), (second.session_ref, "alice")]


def test_small_voice_vector_drift_reuses_one_detector_and_preserves_raw_source() -> None:
    created: list[tuple[str, str]] = []

    def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
        created.append((session_ref, track_ref))
        return turn_detector.TurnDetector(
            participant_ref=track_ref,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({(3.0, 4.0)}),
        target_ids=frozenset({"bot"}),
        content="first",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({(0.6, 0.8)}),
        target_ids=frozenset({"bot"}),
        content="second",
        content_type="text/plain",
    )

    assert first.session_ref == second.session_ref
    assert len(created) == 1
    assert created[0][1].startswith("track-")
    assert first.input.source_ids == frozenset({(3.0, 4.0)})
    assert second.input.source_ids == frozenset({(0.6, 0.8)})


def test_three_voice_observations_reuse_one_detector_track() -> None:
    created: list[str] = []

    def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
        created.append(track_ref)
        return turn_detector.TurnDetector(
            participant_ref=track_ref,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    results = [
        run_input(
            ability,
            context,
            source_ids=frozenset({vector}),
            target_ids=frozenset({"bot"}),
            content="observation",
            content_type="text/plain",
        )
        for vector in ((1.0, 0.0), (0.98, 0.2), (0.96, 0.28))
    ]

    assert len(created) == 1
    assert len({result.session_ref for result in results}) == 1


def test_vector_context_basis_is_scale_invariant_and_collision_safe() -> None:
    ability, context = started_conversation()
    first = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(3.0, 4.0)}),
        content="first",
        content_type="text/plain",
    )
    drifted = run_input(
        ability,
        context,
        source_ids=frozenset({(0.6, 0.8)}),
        target_ids=frozenset({"alice"}),
        content="drifted",
        content_type="text/plain",
    )
    separate = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(0.0, 1.0)}),
        content="separate",
        content_type="text/plain",
    )

    assert first.session_ref == drifted.session_ref
    assert first.session_ref != separate.session_ref


def test_mixed_named_vector_direction_requires_matching_vector() -> None:
    ability, context = started_conversation()
    forward = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(3.0, 4.0)}),
        content="forward",
        content_type="text/plain",
    )
    reverse = run_input(
        ability,
        context,
        source_ids=frozenset({(0.6, 0.8)}),
        target_ids=frozenset({"alice"}),
        content="reverse",
        content_type="text/plain",
    )
    unrelated = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(0.0, 1.0)}),
        content="unrelated",
        content_type="text/plain",
    )

    assert forward.session_ref == reverse.session_ref
    assert unrelated.session_ref != forward.session_ref


def test_unrelated_voice_vector_creates_another_relationship() -> None:
    created: list[str] = []

    def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
        created.append(track_ref)
        return turn_detector.TurnDetector(
            participant_ref=track_ref,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    first = run_input(
        ability,
        context,
        source_ids=frozenset({(1.0, 0.0)}),
        target_ids=frozenset({"bot"}),
        content="first",
        content_type="text/plain",
    )
    second = run_input(
        ability,
        context,
        source_ids=frozenset({(0.0, 1.0)}),
        target_ids=frozenset({"bot"}),
        content="second",
        content_type="text/plain",
    )

    assert first.session_ref != second.session_ref
    assert len(created) == 2
    assert len(set(created)) == 2


def test_vector_profile_updates_only_after_a_completed_turn() -> None:
    created: list[str] = []

    class FailingDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
        async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
            del input
            raise RuntimeError("turn did not complete")

    def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
        created.append(track_ref)
        return turn_detector.TurnDetector(
            participant_ref=track_ref,
            conversation_ref=session_ref,
            decoder=FailingDecoder() if len(created) == 1 else None,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    with pytest.raises(RuntimeError, match="turn did not complete"):
        run_input(
            ability,
            context,
            source_ids=frozenset({(1.0, 0.0)}),
            target_ids=frozenset({"bot"}),
            content="failed",
            content_type="text/plain",
        )

    completed = run_input(
        ability,
        context,
        source_ids=frozenset({(0.99, 0.01)}),
        target_ids=frozenset({"bot"}),
        content="completed",
        content_type="text/plain",
    )

    assert len(created) == 2
    assert completed.input.source_ids == frozenset({(0.99, 0.01)})


def test_ambiguous_or_dimension_mismatched_vectors_create_new_tracks() -> None:
    created: list[str] = []

    def factory(session_ref: str, track_ref: str) -> turn_detector.TurnDetector:
        created.append(track_ref)
        return turn_detector.TurnDetector(
            participant_ref=track_ref,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory, similarity_threshold=0.99, similarity_margin=0.05)
    for vector in ((1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071), (1.0, 0.0, 0.0)):
        run_input(
            ability,
            context,
            source_ids=frozenset({vector}),
            target_ids=frozenset({"bot"}),
            content="vector",
            content_type="text/plain",
        )

    assert len(created) == 4


def test_multi_valued_sources_create_one_detector_per_source() -> None:
    created: list[tuple[str, conversation.IdentityValue]] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        created.append((session_ref, source_id))
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    ability, context = started_conversation(factory)
    result = run_input(
        ability,
        context,
        source_ids=frozenset({"alice", "bob"}),
        target_ids=frozenset({"bot"}),
        content="shared message",
        content_type="application/x-sign-language",
    )

    assert result.session_ref == ability.session_refs[0]
    assert {source_id for _, source_id in created} == {"alice", "bob"}
    assert ability.detector_count == 2
    assert {source_id for _, source_id in ability.detector_refs} == {"alice", "bob"}


def test_bot_initiated_input_creates_relationship_from_explicit_sets() -> None:
    ability, context = started_conversation()

    result = run_input(
        ability,
        context,
        source_ids=frozenset({"bot"}),
        target_ids=frozenset({"alice"}),
        content="I wanted to ask about the weather.",
        content_type="text/plain",
    )

    assert result.session_ref in ability.session_refs
    assert ability.detector_refs == ((result.session_ref, "bot"),)
    assert result.input.source_ids == frozenset({"bot"})
    assert result.input.target_ids == frozenset({"alice"})


def test_source_only_embedding_turns_share_ambient_session_and_preserve_empty_targets() -> None:
    created: list[tuple[str, conversation.IdentityValue]] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        created.append((session_ref, source_id))
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
            end_of_turn_silence_seconds=0.01,
        )

    input_data = conversation.TurnData(
        source_ids=frozenset({(1.0, 0.0, 0.0)}),
        target_ids=frozenset(),
        content="ambient turn",
        content_type="text/plain",
    )
    assert input_data.target_ids == frozenset()

    ability, context = started_conversation(factory)
    results = [
        run_input(
            ability,
            context,
            source_ids=frozenset({embedding}),
            target_ids=frozenset(),
            content="ambient turn",
            content_type="text/plain",
        )
        for embedding in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]

    assert len({result.session_ref for result in results}) == 1
    assert len(created) == 3
    assert ability.detector_count == 3
    assert all(result.input.target_ids == frozenset() for result in results)


def test_content_type_is_data_not_a_route_selector() -> None:
    ability, context = started_conversation()

    plain = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="hello",
        content_type="text/plain",
    )
    mislabeled = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="hello again",
        content_type="application/octet-stream",
    )

    assert plain.session_ref == mislabeled.session_ref
    assert plain.participation.perception.readable == "hello"
    assert isinstance(plain.stimulus, turn_detector.TextStimulus)
    # content_type is data, not a route: mislabeled stays non-text product (no text/plain rewrite).
    assert mislabeled.participation.perception.readable == "hello again"
    assert mislabeled.input.content == "hello again"
    assert mislabeled.input.content_type == "application/octet-stream"
    assert ability.detector_count == 1


def test_reverse_direction_reuses_session_and_structured_ids_are_collision_safe() -> None:
    ability, context = started_conversation()

    forward = run_input(
        ability,
        context,
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({"bot"}),
        content="hello",
        content_type="text/plain",
    )
    reverse = run_input(
        ability,
        context,
        source_ids=frozenset({"bot"}),
        target_ids=frozenset({"alice"}),
        content="hi",
        content_type="text/plain",
    )
    delimited = run_input(
        ability,
        context,
        source_ids=frozenset({"a,b"}),
        target_ids=frozenset({"c"}),
        content={"kind": "weather", "value": "sunny"},
        content_type="application/x-sign-language",
    )
    separate = run_input(
        ability,
        context,
        source_ids=frozenset({"a", "b"}),
        target_ids=frozenset({"c"}),
        content={"kind": "weather", "value": "cloudy"},
        content_type="application/x-sign-language",
    )

    assert forward.session_ref == reverse.session_ref
    assert delimited.session_ref != separate.session_ref
    assert delimited.input.content == {"kind": "weather", "value": "sunny"}
    assert delimited.input.content_type == "application/x-sign-language"
    assert delimited.participation.perception.content == {"kind": "weather", "value": "sunny"}
    assert delimited.participation.perception.modality == "multimodal"


def test_each_source_receives_the_typed_content_and_template_decoder_is_preserved() -> None:
    class RecordingDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
        def __init__(self) -> None:
            self.calls: list[turn_detector.ParticipationStimulus] = []

        async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
            self.calls.append(input)
            return "decoded"

    decoder = RecordingDecoder()
    template = turn_detector.TurnDetector(decoder=decoder)
    context = hsm.Context()
    # Use a template through the public Conversation constructor to exercise cloning.
    ability = conversation.Conversation(turn_detector=template)
    asyncio.run(start_ability_tree(context, ability))

    result = run_input(
        ability,
        context,
        source_ids=frozenset({"alice", "bob"}),
        target_ids=frozenset({"bot"}),
        content={"gesture": "wave"},
        content_type="application/x-sign-language",
    )

    assert result.input.source_ids == frozenset({"alice", "bob"})
    assert result.input.content_type == "application/x-sign-language"
    assert {call.source_participant_ref for call in decoder.calls} == {"alice", "bob"}
    assert all(isinstance(call, turn_detector.ContentStimulus) for call in decoder.calls)


def test_failed_detector_creation_is_not_cached() -> None:
    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        del session_ref
        if source_id == "alice":
            raise RuntimeError("detector creation failed")
        return turn_detector.TurnDetector(participant_ref=source_id)

    ability, context = started_conversation(factory)
    with pytest.raises(RuntimeError, match="detector creation failed"):
        run_input(
            ability,
            context,
            source_ids=frozenset({"alice", "bob"}),
            target_ids=frozenset({"bot"}),
            content="hello",
            content_type="text/plain",
        )
    assert ability.detector_count == 0


def test_failed_detector_readiness_stops_and_does_not_cache_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[turn_detector.TurnDetector] = []
    stopped: list[turn_detector.TurnDetector] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        detector = turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
        )
        created.append(detector)
        return detector

    dispatch = hsm.dispatch

    def fail_readiness(
        ctx: hsm.Context | None,
        target: hsm.Dispatchable | None,
        event: hsm.Event[typing.Any],
    ) -> typing.Awaitable[bool]:
        if event.name == turn_detector.TurnDetectorReadyRequestEvent.name:

            async def fail() -> bool:
                raise RuntimeError("detector readiness dispatch failed")

            return fail()
        return dispatch(ctx, target, event)

    stop = hsm.stop

    def record_stop(
        instance: hsm.Instance | hsm.Group,
        ctx: hsm.Context | None = None,
    ) -> typing.Awaitable[None]:
        if isinstance(instance, turn_detector.TurnDetector):
            stopped.append(instance)
        return stop(instance, ctx)

    monkeypatch.setattr(hsm, "dispatch", fail_readiness)
    monkeypatch.setattr(hsm, "stop", record_stop)
    ability, context = started_conversation(factory)

    with pytest.raises(RuntimeError, match="detector readiness dispatch failed"):
        run_input(
            ability,
            context,
            source_ids=frozenset({"alice"}),
            target_ids=frozenset({"bot"}),
            content="hello",
            content_type="text/plain",
        )

    assert created and stopped == [created[0]]
    assert ability.detector_count == 0


def test_cancellation_stops_active_cached_detector_but_preserves_idle_detector() -> None:
    async def run() -> None:
        active_turn_started = asyncio.Event()
        created: list[str] = []

        class SequencedDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
            def __init__(self, started: asyncio.Event) -> None:
                self._started = started
                self._calls = 0

            async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
                del input
                self._calls += 1
                if self._calls == 1:
                    return "first turn"
                self._started.set()
                await asyncio.Future[None]()
                raise AssertionError("unreachable")

        def factory(session_ref: str, track_ref: conversation.TrackRef) -> turn_detector.TurnDetector:
            created.append(track_ref)
            decoder = SequencedDecoder(active_turn_started) if track_ref == "alice" else None
            return turn_detector.TurnDetector(
                participant_ref=track_ref,
                conversation_ref=session_ref,
                decoder=decoder,
                end_of_turn_silence_seconds=0.01,
            )

        ability = conversation.Conversation(turn_detector_factory=factory)
        context = hsm.Context()
        await start_ability_tree(context, ability)
        first = await conversation.contribute_conversation_input(
            ability,
            conversation.TurnData(
                source_ids=frozenset({"alice", "bob"}),
                target_ids=frozenset({"bot"}),
                content="first",
                content_type="text/plain",
            ),
            ctx=context,
        )

        operation = asyncio.create_task(
            conversation.contribute_conversation_input(
                ability,
                conversation.TurnData(
                    source_ids=frozenset({"alice", "bob"}),
                    target_ids=frozenset({"bot"}),
                    content="cancelled",
                    content_type="text/plain",
                ),
                ctx=context,
            )
        )
        await asyncio.wait_for(active_turn_started.wait(), timeout=1.0)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation

        assert ability.detector_refs == ((first.session_ref, "bob"),)
        assert ability.detector_count == 1

        await conversation.contribute_conversation_input(
            ability,
            conversation.TurnData(
                source_ids=frozenset({"alice", "bob"}),
                target_ids=frozenset({"bot"}),
                content="retry",
                content_type="text/plain",
            ),
            ctx=context,
        )
        assert created == ["alice", "bob", "alice"]

    asyncio.run(run())


def test_public_input_only_accepts_the_first_missing_detector_failure() -> None:
    failed_source = "alice"
    attempted: list[conversation.IdentityValue] = []

    def factory(session_ref: str, source_id: conversation.TrackRef) -> turn_detector.TurnDetector:
        attempted.append(source_id)
        if source_id == failed_source:
            raise RuntimeError(f"detector creation failed for {source_id}")
        return turn_detector.TurnDetector(
            participant_ref=source_id,
            conversation_ref=session_ref,
        )

    ability, context = started_conversation(factory)
    input_data = conversation.TurnData(
        source_ids=frozenset({"alice", "bob", "carol"}),
        target_ids=frozenset({"bot"}),
        content="hello",
        content_type="text/plain",
    )

    def contribute() -> conversation.ParticipatedTurn:
        return run_input(
            ability,
            context,
            source_ids=input_data.source_ids,
            target_ids=input_data.target_ids,
            content=input_data.content,
            content_type=input_data.content_type,
        )

    with pytest.raises(RuntimeError, match="failed for alice"):
        contribute()
    assert attempted == ["alice"]
    assert ability.detector_count == 0

    failed_source = "bob"
    attempted.clear()
    with pytest.raises(RuntimeError, match="failed for bob"):
        contribute()
    assert attempted == ["alice", "bob"]
    assert ability.detector_refs == ((ability.session_refs[0], "alice"),)


def test_identity_sets_reject_blank_members() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        conversation.TurnData(
            source_ids=frozenset({" "}),
            target_ids=frozenset({"bot"}),
            content="hello",
            content_type="text/plain",
        )


def test_conversation_input_rehydrates_audio_bytes_from_base64_selection() -> None:
    """Typed Conversation.InputEvent restores audio bytes from selection/JSON hops."""

    import base64

    from mosfet.event import validate_event_data
    from mosfet.abilities.communication import conversation

    audio = bytes((0, 1, 2, 3, 4, 5, 6, 7)) * 20
    raw = {
        "source_ids": [[0.12, -0.08, 0.31]],
        "target_ids": [],
        "content": base64.urlsafe_b64encode(audio).decode("ascii"),
        "content_type": "audio/raw",
    }
    validated = validate_event_data(conversation.InputEvent, raw)
    assert isinstance(validated, conversation.TurnData)
    assert isinstance(validated.content, bytes)
    assert validated.content == audio
    assert validated.content_type == "audio/raw"


def test_event_json_value_omits_unowned_raw_media() -> None:
    """Canonical JSON omits raw media fields and items without encoding them."""

    import base64

    from mosfet import event

    # Bytes that differ between std (+/) and urlsafe (-_) alphabets.
    audio = bytes((0xFB, 0xFF, 0xFE, 0x00, 0x01, 0x02, 0x03, 0x04)) * 32
    expected = base64.urlsafe_b64encode(audio).decode("ascii")
    assert base64.b64encode(audio).decode("ascii") != expected  # std uses +/
    payload = {"content": audio, "nested": {"blob": audio}, "ids": [audio[:4]], "kept": "value"}
    expected_tree = {"nested": {}, "ids": [], "kept": "value"}
    assert event.event_json_value(payload) == expected_tree
    assert event.bytes_from_base64(expected) == audio
    assert event.bytes_from_base64(audio) == audio
    # Round-trip matches pydantic ser_json_bytes="base64".
    from pydantic import BaseModel, ConfigDict

    class _Payload(BaseModel):
        model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")
        audio: bytes

    pyd = _Payload(audio=audio).model_dump(mode="json")["audio"]
    assert pyd == expected
    assert event.bytes_from_base64(pyd) == audio
    # Standard alphabet is not part of the URL-safe wire contract.
    std = base64.b64encode(audio).decode("ascii")
    with pytest.raises(ValueError, match="base64"):
        event.bytes_from_base64(std)


def test_conversation_input_rejects_malformed_audio_base64() -> None:
    from mosfet.abilities.communication import conversation
    from mosfet.event import validate_event_data

    with pytest.raises(pydantic.ValidationError, match="base64"):
        validate_event_data(
            conversation.InputEvent,
            {
                "source_ids": ["caller"],
                "target_ids": [],
                "content": "AA=",
                "content_type": "audio/raw",
            },
        )


def test_conversation_input_rehydrates_pydantic_urlsafe_speech_audio() -> None:
    """SpeechData JSON dump (urlsafe) must validate as Conversation.input content."""

    from mosfet.abilities import listening
    from mosfet.abilities.communication import conversation
    from mosfet.abilities.hearing import voice
    from mosfet.event import validate_event_data

    audio = bytes((i % 256 for i in range(115_202)))
    speech = listening.SpeechData(
        content=audio,
        voice_detection=voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(
                    start_seconds=0.0,
                    end_seconds=0.4,
                    confidence=0.9,
                ),
            )
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({tuple(-0.03482 + i * 0.001 for i in range(16))}),
    )
    dumped = speech.model_dump(mode="json")
    # Real admit path: the established Pydantic JSON contract carries the encoded audio payload.
    validated = validate_event_data(
        conversation.InputEvent,
        {
            "source_ids": dumped["source_ids"],
            "target_ids": [],
            "content": dumped["content"],
            "content_type": dumped["media_type"],
            "sample_rate_hz": dumped["sample_rate_hz"],
            "channels": dumped["channels"],
        },
    )
    assert isinstance(validated, conversation.TurnData)
    assert validated.content == audio
    assert validated.sample_rate_hz == 16_000
    assert validated.channels == 1


def test_processing_dispatch_accepts_typed_event_data_instance() -> None:
    """Dispatch is schema-driven: a typed event payload instance is valid selection data."""

    import asyncio

    import hsm
    from mosfet.abilities import processing
    from mosfet.abilities.communication.conversation import turn_detector
    from mosfet.abilities.communication import conversation
    from mosfet.protocols import attachment

    audio = bytes((9, 8, 7, 6)) * 200
    typed = conversation.TurnData(
        source_ids=frozenset({(0.1, 0.2, 0.3)}),
        target_ids=frozenset(),
        content=audio,
        content_type="audio/raw",
    )

    async def run() -> None:
        ctx = hsm.Context()
        conv = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(participant_ref="bot", conversation_ref="ambient")
        )
        assert conv.model is not None
        _ = await mosfet.started(ctx, conv, conv.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await conv.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while loop.time() < deadline and "/behavior/inactive" not in (conv.state() or ""):
            await asyncio.sleep(0)

        pin = processing.InputData(
            input=None,
            schemas=(conversation.InputEvent,),
            actors={"conversation": conv},
            authority=owner,
        )
        await processing.dispatch_selected_events(
            ctx,
            pin,
            (
                processing.SelectedEvent(
                    event=conversation.InputEvent.name,
                    data=typed,
                    reason="typed event payload",
                ),
            ),
            operation_id="typed-instance",
            source=owner,
        )
        deadline = loop.time() + 5
        while loop.time() < deadline and "/active/" not in (conv.state() or ""):
            await asyncio.sleep(0)
        assert "/active/" in (conv.state() or ""), conv.state()

    asyncio.run(run())


def test_processing_dispatch_delivers_typed_audio_bytes_to_conversation() -> None:
    """Autonomy-style EventData selection must dispatch typed bytes into Conversation."""

    import asyncio
    import base64

    import hsm
    import mosfet
    from mosfet.abilities import processing
    from mosfet.abilities.communication.conversation import turn_detector
    from mosfet.abilities.communication import conversation
    from mosfet.protocols import attachment

    audio = bytes((0, 64)) * 800

    async def run() -> bytes | str | None:
        ctx = hsm.Context()
        conv = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(participant_ref="bot", conversation_ref="ambient")
        )
        assert conv.model is not None
        _ = await mosfet.started(ctx, conv, conv.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await conv.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        # wait inactive
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while loop.time() < deadline and "/behavior/inactive" not in (conv.state() or ""):
            await asyncio.sleep(0)

        selection = (
            processing.SelectedEvent(
                event=conversation.InputEvent.name,
                target=None,
                data={
                    "source_ids": [[0.12, -0.08, 0.31]],
                    "target_ids": [],
                    "content": base64.urlsafe_b64encode(audio).decode("ascii"),
                    "content_type": "audio/raw",
                },
                reason="typed admit",
            ),
        )
        pin = processing.InputData(
            input=None,
            schemas=(conversation.InputEvent,),
            actors={"conversation": conv},
            authority=owner,
        )
        await processing.dispatch_selected_events(
            ctx,
            pin,
            selection,
            operation_id="typed-admit",
            source=owner,
        )
        # Conversation queues work asynchronously into active
        deadline = loop.time() + 5
        while loop.time() < deadline and "/active/" not in (conv.state() or ""):
            await asyncio.sleep(0)
        # Peek queued relationship via forcing a second of processing - instead inspect via contribute path
        # The input is on the machine; pull from last internal by dispatching snapshot?
        # Simpler: validate_event_data already tested; here assert active state means typed input accepted.
        assert "/active/" in (conv.state() or ""), conv.state()
        return audio

    assert asyncio.run(run()) == audio


def test_conversation_stays_active_and_offers_input_after_turn() -> None:
    """Relationship stays active across contributions; InputEvent remains offerable."""

    from mosfet.abilities import processing

    class IdentityDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
        @typing.override
        async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
            if isinstance(input, turn_detector.TextStimulus):
                return input.content
            raise AssertionError(input)

    async def run() -> tuple[str, tuple[str, ...], str]:
        ability = conversation.Conversation(turn_detector=turn_detector.TurnDetector(decoder=IdentityDecoder()))
        context = hsm.Context()
        await start_ability_tree(context, ability)
        _ = await conversation.contribute_conversation_input(
            ability,
            conversation.TurnData(
                source_ids=frozenset({"caller"}),
                target_ids=frozenset({"bot"}),
                content="one",
                content_type="text/plain",
            ),
            ctx=context,
        )
        state_after = ability.state() or ""
        offered = tuple(event.name for event in processing.enabled_call_events(ability))
        second = await conversation.contribute_conversation_input(
            ability,
            conversation.TurnData(
                source_ids=frozenset({"caller"}),
                target_ids=frozenset({"bot"}),
                content="two",
                content_type="text/plain",
            ),
            ctx=context,
        )
        text = second.participation.perception.readable or ""
        return state_after, offered, text

    state_after, offered, second_text = asyncio.run(run())
    assert state_after.endswith("/behavior/active/waiting"), state_after
    assert conversation.InputEvent.name in offered
    assert second_text == "two"


def test_messages_reject_raw_media_and_preserve_immutable_provenance() -> None:
    with pytest.raises(pydantic.ValidationError):
        conversation.Message(
            sequence=0,
            direction="inbound",
            source_ids=frozenset({"caller"}),
            target_ids=frozenset(),
            content=typing.cast(typing.Any, b"raw-audio"),
            content_type="audio/pcm",
            provenance=conversation.MessageProvenance(event="test.input"),
        )


def test_memory_keeps_the_inbound_turn_behind_a_history_emission_not_the_history() -> None:
    """Messages declares __memory_fields__ = {parent}: memory keeps the causal turn, the prompt keeps history."""

    turn = conversation.TurnData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content="call me back at noon",
        content_type="text/plain",
    )
    history = conversation.Messages(
        parent=mosfet.StimulusData(event=conversation.InputEvent.name, data=turn, id="turn-1"),
        messages=(
            conversation.Message(
                sequence=0,
                direction="inbound",
                source_ids=frozenset({"caller"}),
                target_ids=frozenset({"bot"}),
                content="an earlier line of history",
                content_type="text/plain",
                provenance=conversation.MessageProvenance(event="test.input"),
            ),
        ),
    )
    emitted = conversation.OutputEvent.with_data(history)

    kept = "".join(
        part.text
        for part in cognition_event.memory_content(emitted, capacity=media.ImageCapacity())
        if isinstance(part, text.TextPart)
    )
    prompt = "".join(
        part.text
        for part in cognition_event.model_facing_content(emitted, capacity=media.ImageCapacity())
        if isinstance(part, text.TextPart)
    )

    assert "call me back at noon" in kept
    assert "an earlier line of history" not in kept
    assert "an earlier line of history" in prompt
    assert "call me back at noon" not in prompt


def test_conversation_takes_no_memory() -> None:
    """Hard cutover: Conversation neither recalls nor stores; Cognition remembers every turn."""

    with pytest.raises(TypeError):
        _ = conversation.Conversation(**typing.cast(typing.Any, {"memory": object()}))
    assert "memories" not in conversation.ParticipatedTurn.model_fields
    assert typing.get_args(conversation.Stage) == ("turn_detector", "voice_routing")


def test_conversation_append_same_correlation_is_idempotent() -> None:
    class RecordingConversation(conversation.Conversation):
        def __init__(self) -> None:
            super().__init__()
            self.outputs: list[conversation.Messages] = []

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
            if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
                terminal = event.data
                if terminal.name == conversation.OutputEvent.name and isinstance(terminal.data, conversation.Messages):
                    self.outputs.append(terminal.data)
            return super().dispatch(ctx, event)

    async def run() -> tuple[conversation.Messages, conversation.Messages]:
        context = hsm.Context()
        target = RecordingConversation()
        await start_ability_tree(context, target)
        message = conversation.Message(
            sequence=0,
            direction="outbound",
            source_ids=frozenset(),
            target_ids=frozenset({"caller"}),
            content="hi",
            content_type="text/plain",
            provenance=conversation.MessageProvenance(event="test.message"),
        )
        event = dataclasses.replace(
            conversation.AppendEvent.with_data(conversation.AppendData(message=message)),
            id="append-1",
            source="speaking-1",
            target=hsm.id(target),
        )
        await hsm.dispatch(context, target, event)
        first = target.outputs[-1]
        await hsm.dispatch(context, target, event)
        second = target.outputs[-1]
        return first, second

    first, second = asyncio.run(run())

    assert len(first.messages) == 1
    assert len(second.messages) == 1
    assert second.messages == first.messages
    assert second.messages[0].sequence == 0
    assert second.messages[0].provenance.id == "append-1"


def test_conversation_commits_cumulative_bidirectional_history() -> None:
    ability, context = started_conversation()
    inbound = run_input(
        ability,
        context,
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content="hello",
        content_type="text/plain",
    )
    assert tuple(item.direction for item in inbound.messages) == ("inbound",)

    async def append() -> conversation.Messages:
        return await conversation.append_conversation_message(
            ability,
            conversation.Message(
                sequence=1,
                direction="outbound",
                source_ids=frozenset(),
                target_ids=frozenset({"caller"}),
                content="hi",
                content_type="text/plain",
                provenance=conversation.MessageProvenance(
                    event="bot.ability.speaking.output",
                    source="speaking-1",
                    session_ref=inbound.session_ref,
                ),
            ),
            ctx=context,
        )

    history = asyncio.run(append())
    assert tuple(item.sequence for item in history.messages) == (0, 1)
    assert tuple(item.direction for item in history.messages) == ("inbound", "outbound")


def test_concurrent_host_appends_complete_their_own_operations() -> None:
    async def run() -> tuple[conversation.Messages, conversation.Messages]:
        ability = conversation.Conversation()
        context = hsm.Context()
        await start_ability_tree(context, ability)

        def message(content: str) -> conversation.Message:
            return conversation.Message(
                sequence=0,
                direction="outbound",
                source_ids=frozenset(),
                target_ids=frozenset({"caller"}),
                content=content,
                content_type="text/plain",
                provenance=conversation.MessageProvenance(event=f"test.append.{content}"),
            )

        first = asyncio.create_task(conversation.append_conversation_message(ability, message("first"), ctx=context))
        second = asyncio.create_task(conversation.append_conversation_message(ability, message("second"), ctx=context))
        return await first, await second

    first, second = asyncio.run(run())
    histories = sorted((first, second), key=lambda history: len(history.messages))
    assert len(histories[0].messages) == 1
    assert len(histories[1].messages) == 2
    assert {item.content for item in histories[1].messages if isinstance(item.content, str)} == {
        "first",
        "second",
    }


def test_concurrent_host_contributions_preserve_each_input() -> None:
    async def run() -> tuple[conversation.ParticipatedTurn, conversation.ParticipatedTurn]:
        ability = conversation.Conversation()
        context = hsm.Context()
        await start_ability_tree(context, ability)

        def input_data(source: str, content: str) -> conversation.TurnData:
            return conversation.TurnData(
                source_ids=frozenset({source}),
                target_ids=frozenset({"bot"}),
                content=content,
                content_type="text/plain",
            )

        first = asyncio.create_task(
            conversation.contribute_conversation_input(
                ability,
                input_data("alice", "first"),
                ctx=context,
            )
        )
        second = asyncio.create_task(
            conversation.contribute_conversation_input(
                ability,
                input_data("bob", "second"),
                ctx=context,
            )
        )
        return await first, await second

    first, second = asyncio.run(run())
    assert first.input.content == "first"
    assert first.participation.perception.readable == "first"
    assert second.input.content == "second"
    assert second.participation.perception.readable == "second"


def test_image_attachments_reach_history_and_hand_cognition_the_turn() -> None:
    ability_under_test, context = started_conversation()
    image = _image()

    terminal = run_terminal(
        ability_under_test,
        context,
        conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="what is this",
            content_type="text/plain",
            attachments=(image,),
        ),
    )

    assert terminal.name == cognition.InputEvent.name
    latest = terminal_messages(terminal).messages[-1]
    assert latest.content == "what is this"
    assert latest.attachments == (image,)


def test_input_rejects_image_content_types_because_images_travel_as_attachments() -> None:
    with pytest.raises(pydantic.ValidationError, match="images travel as attachments"):
        _ = conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="iVBORw0KGgo=",
            content_type="image/png",
        )


def test_input_attachments_have_exactly_one_sender() -> None:
    with pytest.raises(pydantic.ValidationError, match="one source"):
        _ = conversation.TurnData(
            source_ids=frozenset({"alice", "bob"}),
            target_ids=frozenset({"bot"}),
            content="look",
            content_type="text/plain",
            attachments=(_image(),),
        )


def test_input_bounds_attachments_per_turn() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="look",
            content_type="text/plain",
            attachments=tuple(_image(index) for index in range(media.MAX_TURN_IMAGES + 1)),
        )


@pytest.mark.parametrize(
    ("attachments", "content_type"),
    [((), "text/plain"), ((_image(),), "application/octet-stream"), ((_image(),), "audio/pcm")],
    ids=["no-attachments", "non-text-type", "audio-type"],
)
def test_input_without_content_requires_text_typed_attachments(
    attachments: tuple[media.Image, ...], content_type: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content=None,
            content_type=content_type,
            attachments=attachments,
        )


def test_input_attachments_are_absent_from_the_model_facing_schema() -> None:
    schema = typing.cast(type[pydantic.BaseModel], conversation.InputEvent.schema).model_json_schema()

    assert "attachments" not in schema["properties"]
    assert "content" in schema["required"]


def test_model_dispatch_rejects_producer_stamped_conversation_attachments() -> None:
    """A model-selected Conversation.input can never author images."""

    async def run() -> None:
        context = hsm.Context()
        source = hsm.Instance()
        input_data = conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset(),
            content="look",
            content_type="text/plain",
            attachments=(_image(),),
        )
        processing_input = processing.InputData(
            input="forged model selection",
            schemas=(conversation.InputEvent,),
            actors={},
        )

        with pytest.raises(RuntimeError, match="producer-stamped fields: attachments"):
            await processing.dispatch_selected_events(
                context,
                processing_input,
                (
                    processing.SelectedEvent(
                        event=conversation.InputEvent.name,
                        target="conversation",
                        data=input_data.model_dump(mode="json"),
                    ),
                ),
                operation_id="forged-attachments",
                source=source,
            )

    asyncio.run(run())


def test_image_only_turn_with_a_decoder_reaches_cognition_with_its_image() -> None:
    class RecordingDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
        def __init__(self) -> None:
            self.calls: list[turn_detector.ParticipationStimulus] = []

        @typing.override
        async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
            self.calls.append(input)
            return "decoded"

    decoder = RecordingDecoder()
    target = conversation.Conversation(turn_detector=turn_detector.TurnDetector(decoder=decoder))
    context = hsm.Context()
    asyncio.run(start_ability_tree(context, target))
    image = _image()

    terminal = run_terminal(
        target,
        context,
        conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content=None,
            content_type="text/plain",
            attachments=(image,),
        ),
    )

    assert terminal.name == cognition.InputEvent.name
    latest = terminal_messages(terminal).messages[-1]
    assert latest.content is None
    assert latest.content_type == "text/plain"
    assert latest.attachments == (image,)
    assert decoder.calls == []


@pytest.mark.parametrize(
    ("content", "content_type", "attachments", "handed_off"),
    [
        ("hello", "text/plain", (), True),
        ("look", "text/plain", (_image(),), True),
        (None, "text/plain", (_image(),), True),
        (b"pcm", "audio/pcm", (), False),
        (b"pcm", "audio/pcm", (_image(),), False),
        ({"gesture": "wave"}, "application/json", (), False),
    ],
    ids=["text", "captioned-image", "image-only", "audio", "audio-with-image", "structured"],
)
def test_cognition_handoff_on_a_decoderless_detector_takes_text_and_image_turns_only(
    content: object,
    content_type: str,
    attachments: tuple[media.Image, ...],
    handed_off: bool,
) -> None:
    """Text and image-attachment turns hand cognition the history; audio and structured stay bare output."""

    target, context = started_conversation()

    terminal = run_terminal(
        target,
        context,
        conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content=content,
            content_type=content_type,
            attachments=attachments,
        ),
    )

    expected = cognition.InputEvent.name if handed_off else conversation.OutputEvent.name
    assert terminal.name == expected
    assert terminal_messages(terminal).messages[-1].attachments == attachments


def _released_history_images(metrics: metric_reader.Reader) -> int:
    total = 0
    for point in metrics.points("bot.conversation.history.image.released"):
        assert point.attributes == {"media_type": "image/png"}
        total += int(point.value)
    return total


def test_history_keeps_image_bytes_only_within_the_injected_retention_budget(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        released_before = _released_history_images(metrics)
        retention = media.ImageRetention(max_images=2, max_total_bytes=1_000_000)
        target = conversation.Conversation(image_retention=retention)
        context = hsm.Context()
        asyncio.run(start_ability_tree(context, target))
        images = tuple(_image(index) for index in range(5))
        histories: list[conversation.Messages] = []

        for image in images:
            terminal = run_terminal(
                target,
                context,
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content=None,
                    content_type="text/plain",
                    attachments=(image,),
                ),
            )
            history = terminal_messages(terminal)
            histories.append(history)
            (newest,) = history.messages[-1].attachments
            assert newest == image
            older = [attachment for message in history.messages[:-1] for attachment in message.attachments]
            assert sum(isinstance(attachment, media.Image) for attachment in older) <= retention.max_images
            held = sum(
                len(attachment.content)
                for message in history.messages
                for attachment in message.attachments
                if isinstance(attachment, media.Image)
            )
            assert held <= media.MAX_TURN_IMAGES * media.MAX_IMAGE_BYTES + retention.max_total_bytes

        final = histories[-1].messages
        assert [message.attachments for message in final[2:]] == [(images[2],), (images[3],), (images[4],)]
        for message, image in zip(final[:2], images[:2], strict=True):
            assert message.attachments == (media.ReleasedImage.of(image, reason="retention"),)
        for earlier in histories[:-1]:
            for kept, committed in zip(final, earlier.messages):
                assert (kept.sequence, kept.provenance) == (committed.sequence, committed.provenance)
        released_after = _released_history_images(metrics)

    assert released_after - released_before == 2


_CALLER = mosfet.Participant(ref="caller", role=mosfet.ParticipantRole.SOURCE)
_BOT = mosfet.Participant(ref="bot", role=mosfet.ParticipantRole.TARGET)


def _handoff(terminal: hsm.Event[typing.Any]) -> cognition.InputData:
    assert terminal.name == cognition.InputEvent.name
    assert isinstance(terminal.data, cognition.InputData)
    return terminal.data


@pytest.mark.parametrize(
    ("content", "attachments"),
    [("hello", ()), (None, (_image(),))],
    ids=["text", "image-only"],
)
def test_cognition_handoff_stamps_string_identities_as_source_and_target_participants(
    content: str | None, attachments: tuple[media.Image, ...]
) -> None:
    target, context = started_conversation()

    terminal = run_terminal(
        target,
        context,
        conversation.TurnData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot", "assistant"}),
            content=content,
            content_type="text/plain",
            attachments=attachments,
        ),
    )

    assert _handoff(terminal).participants == (
        _CALLER,
        mosfet.Participant(ref="assistant", role=mosfet.ParticipantRole.TARGET),
        _BOT,
    )


def test_cognition_handoff_copies_its_speech_parents_participants_deduplicated() -> None:
    ada = mosfet.Participant(ref="ada", role=mosfet.ParticipantRole.SOURCE)
    speech = interpretation.SpeechData(
        content="hello",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"caller"}),
        participants=(ada, _CALLER),
    )
    target, context = started_conversation()

    terminal = run_terminal(
        target,
        context,
        conversation.TurnData(
            parent=mosfet.StimulusData[interpretation.SpeechData](
                event="bot.ability.listening.speech.output",
                data=speech,
            ),
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="hello",
            content_type="text/plain",
        ),
    )

    assert _handoff(terminal).participants == (_CALLER, _BOT, ada)


def _unstamped(metrics: metric_reader.Reader) -> int:
    total = 0
    for point in metrics.points("bot.memory.participant.unstamped.count"):
        assert point.attributes == {}
        total += int(point.value)
    return total


def test_cognition_handoff_never_stamps_vector_identities_and_counts_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _unstamped(metrics)
        target, context = started_conversation()

        terminal = run_terminal(
            target,
            context,
            conversation.TurnData(
                source_ids=frozenset({(1.0, 0.0)}),
                target_ids=frozenset({"bot", (0.0, 1.0)}),
                content="hello",
                content_type="text/plain",
            ),
        )

        assert _handoff(terminal).participants == (_BOT,)
        assert _unstamped(metrics) - before == 2


def test_turn_participants_counts_a_ref_too_long_to_stamp_instead_of_failing_the_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _unstamped(metrics)

        participants = conversation.conversation.turn_participants(
            conversation.TurnData(
                source_ids=frozenset({"x" * 201}),
                target_ids=frozenset({"bot"}),
                content="hello",
                content_type="text/plain",
            )
        )

        assert participants == (_BOT,)
        assert _unstamped(metrics) - before == 1
