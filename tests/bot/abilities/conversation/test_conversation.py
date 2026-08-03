from bot.abilities import cognition
from bot.abilities import conversation
from bot.abilities import decoding
from bot.abilities import listening
from bot.abilities import turn_detector
from bot.abilities.hearing import voice
from bot.abilities.conversation import memory as conversation_memory
from bot.abilities.identity import value

import asyncio
import typing

import hsm
import pydantic
import pytest

from tests.hsm_instance_state import start_ability_tree


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
            conversation.ConversationInputData(
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


def test_input_contract_has_exactly_four_fields_and_no_session_reference() -> None:
    assert tuple(conversation.ConversationInputData.model_fields) == (
        "source_ids",
        "target_ids",
        "content",
        "content_type",
    )
    assert "conversation_ref" not in conversation.ConversationInputData.model_fields
    assert "self_participant_ref" not in conversation.ConversationInputData.model_fields
    assert "participants" not in conversation.ConversationInputData.model_fields
    with pytest.raises(pydantic.ValidationError):
        conversation.ConversationInputData.model_validate(
            {
                "source_ids": ["caller"],
                "target_ids": ["bot"],
                "content": "hello",
                "content_type": "text/plain",
                "conversation_ref": "legacy",
            }
        )
    schema = typing.cast(type[pydantic.BaseModel], conversation.InputEvent.schema)
    assert tuple(schema.model_fields) == tuple(conversation.ConversationInputData.model_fields)


def test_zero_vector_is_rejected_at_the_input_boundary() -> None:
    with pytest.raises(pydantic.ValidationError, match="zero vector"):
        conversation.ConversationInputData(
            source_ids=frozenset({(0.0, 0.0)}),
            target_ids=frozenset({"bot"}),
            content="invalid voice identity",
            content_type="text/plain",
        )


def test_input_accepts_an_empty_target_set() -> None:
    input_data = conversation.ConversationInputData(
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
        conversation.ConversationInputData.model_validate(fields)


def test_input_rejects_embeddings_above_the_provider_neutral_dimension_limit() -> None:
    oversized = (1.0,) * (value.MAX_EMBEDDING_DIMENSION + 1)

    with pytest.raises(pydantic.ValidationError, match="exceeds the maximum of 2048"):
        conversation.ConversationInputData(
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
    forward = conversation_memory.relationship_context_ref(
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(3.0, 4.0)}),
    )
    scaled = conversation_memory.relationship_context_ref(
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(0.6, 0.8)}),
    )
    reverse = conversation_memory.relationship_context_ref(
        source_ids=frozenset({(0.6, 0.8)}),
        target_ids=frozenset({"alice"}),
    )
    unrelated = conversation_memory.relationship_context_ref(
        source_ids=frozenset({"alice"}),
        target_ids=frozenset({(0.0, 1.0)}),
    )

    assert forward == scaled == reverse
    assert forward != unrelated

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

    input_data = conversation.ConversationInputData(
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
    assert plain.decoded_text == "hello"
    assert mislabeled.decoded_text is None
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
    ) -> typing.Awaitable[None]:
        if event.name == turn_detector.TurnDetectorReadyRequestEvent.name:
            async def fail() -> None:
                raise RuntimeError("detector readiness dispatch failed")

            return fail()
        return dispatch(ctx, target, event)

    stop = hsm.stop

    def record_stop(
        instance: hsm.Instance | hsm.Group,
        ctx: hsm.Context | None = None,
    ) -> typing.Awaitable[None]:
        assert isinstance(instance, turn_detector.TurnDetector)
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
            conversation.ConversationInputData(
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
                conversation.ConversationInputData(
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
            conversation.ConversationInputData(
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
    input_data = conversation.ConversationInputData(
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
        conversation.ConversationInputData(
            source_ids=frozenset({" "}),
            target_ids=frozenset({"bot"}),
            content="hello",
            content_type="text/plain",
        )


def test_listening_speech_without_source_ids_fails_at_conversation_boundary() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ability = conversation.Conversation()
        context = hsm.Context()
        await start_ability_tree(context, ability)
        speech = listening.SpeechData(
            audio=b"\x00\x00",
            voice_detection=voice.detection.ApplyData(),
            sample_rate_hz=1,
            media_type="audio/pcm",
        )
        stimulus = listening.SpeechEvent.with_data(speech)
        operation_id = "missing-voice-id"
        waiter: asyncio.Future[hsm.Event[typing.Any]] = asyncio.get_running_loop().create_future()
        ability.register_terminal_waiter(operation_id, waiter)
        try:
            await hsm.dispatch(
                context,
                ability,
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=stimulus), operation_id),
            )
            return await asyncio.wait_for(waiter, timeout=1.0)
        finally:
            ability.clear_terminal_waiter(operation_id)

    terminal = asyncio.run(run())
    assert terminal.name == conversation.FailedEvent.name
    assert terminal.data == conversation.FailureData(
        stage="voice_routing",
        message="Listening speech has no source_ids; configure a voice classifier or diarizer.",
    )
