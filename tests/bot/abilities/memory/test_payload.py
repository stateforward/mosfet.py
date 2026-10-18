"""Bounds, validators, and schemas of the episodic memory payloads."""

import datetime

import mosfet
import pydantic
import pytest

from mosfet import media
from mosfet.abilities import memory
from mosfet.abilities.cognition import event as projection
from mosfet.abilities.language import text

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


def _image() -> text.ImagePart:
    return text.ImagePart(image=media.Image(content=_PNG, media_type="image/png"))


def test_an_episode_keeps_at_most_the_turn_image_bound_and_text_budget() -> None:
    episode = memory.EpisodeData(
        turn_id="turn-1", parts=(text.TextPart(text="hello"), *(_image(),) * 8), outcome=memory.Outcome.COMPLETED
    )
    assert len(episode.parts) == 9
    with pytest.raises(pydantic.ValidationError, match="image parts"):
        _ = memory.EpisodeData(turn_id="turn-1", parts=(_image(),) * 9, outcome=memory.Outcome.COMPLETED)
    with pytest.raises(pydantic.ValidationError, match="UTF-8 bytes"):
        _ = memory.EpisodeData(
            turn_id="turn-1",
            parts=(text.TextPart(text="x" * (memory.MAX_EPISODE_TEXT_BYTES + 1)),),
            outcome=memory.Outcome.COMPLETED,
        )
    with pytest.raises(pydantic.ValidationError):
        _ = memory.EpisodeData(
            turn_id="turn-1",
            participants=(mosfet.Participant(ref="a", role=mosfet.ParticipantRole.SOURCE),) * 33,
            outcome=memory.Outcome.COMPLETED,
        )


def test_an_episode_round_trips_images_through_json() -> None:
    episode = memory.EpisodeData(turn_id="turn-1", parts=(_image(),), outcome=memory.Outcome.FAILED)
    assert memory.EpisodeData.model_validate_json(episode.model_dump_json()) == episode


def test_consolidate_names_unit_refs_and_snake_case_relations() -> None:
    decision = memory.ConsolidateData(
        unit="u",
        links=(memory.LinkData(source="n1", target="n2", relation="mentions"),),
        merges=(memory.MergeData(sources=("n1", "n2"), text="merged"),),
        forgets=("n3",),
    )
    assert decision.links[0].weight == 1.0
    assert memory.ConsolidateData(unit="u").links == ()
    with pytest.raises(pydantic.ValidationError, match="unit ref"):
        _ = memory.ConsolidateData(unit="u", forgets=("5d0c3f7e9a1b4c2d",))
    with pytest.raises(pydantic.ValidationError, match="snake_case"):
        _ = memory.LinkData(source="n1", target="n2", relation="Mentions Of")
    with pytest.raises(pydantic.ValidationError):
        _ = memory.MergeData(sources=("n1",), text="one source is not a merge")


def test_recent_reads_episodes_only() -> None:
    assert memory.RecentData(limit=1).kinds == (memory.NodeKind.EPISODE,)
    with pytest.raises(pydantic.ValidationError, match="only episodes"):
        _ = memory.RecentData(limit=1, kinds=(memory.NodeKind.CONCEPT,))
    with pytest.raises(pydantic.ValidationError):
        _ = memory.RecentData(limit=257)


def test_model_facing_schemas_describe_their_fields_with_examples() -> None:
    for model in (memory.ConsolidateData, memory.LinkData, memory.MergeData, memory.UnitData, memory.EpisodeData):
        schema = model.model_json_schema()
        assert schema.get("examples") or schema.get("description"), model.__name__
    properties = memory.ConsolidateData.model_json_schema()["properties"]
    assert all("description" in field for field in properties.values())


def test_a_unit_too_large_for_the_prompt_omits_nodes_instead_of_failing() -> None:
    """Review finding: a unit's rendering is bounded, so an oversized unit never fails the model call."""

    created = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
    unit = memory.UnitData(
        unit="u",
        nodes=tuple(
            memory.UnitNode(
                ref=f"n{index}", kind=memory.NodeKind.EPISODE, text="x" * 60_000, created_at=created, role="episode"
            )
            for index in range(1, 9)
        ),
    )
    parts = projection.model_facing_content(unit, capacity=media.ImageCapacity())
    (rendered,) = parts
    assert isinstance(rendered, text.TextPart)
    assert 'omitted="' in rendered.text
    assert 'ref="n1"' in rendered.text or "<ref>n1</ref>" in rendered.text


def test_failure_data_carries_a_closed_kind() -> None:
    failure = memory.FailureData(kind=memory.FailureKind.STALE_LEASE, message="gone")
    assert failure.kind.value == "stale_lease"
    assert {kind.value for kind in memory.FailureKind} == {
        "invalid_input",
        "unavailable",
        "store_failed",
        "stale_lease",
        "timeout",
        "conflict",
        "declared",
        "not_found",
    }


def test_a_cue_keeps_the_episode_bounds() -> None:
    cue = memory.CueData(
        cue=(text.TextPart(text="Ada: noon?"), _image()),
        participants=(mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE),),
    )
    assert len(cue.cue) == 2
    with pytest.raises(pydantic.ValidationError, match="image parts"):
        _ = memory.CueData(cue=(_image(),) * 9)
    with pytest.raises(pydantic.ValidationError, match="UTF-8 bytes"):
        _ = memory.CueData(cue=(text.TextPart(text="x" * (memory.MAX_EPISODE_TEXT_BYTES + 1)),))


def test_associations_are_ranked_head_budgeted_and_carry_placeholders_only() -> None:
    at = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
    released = media.ReleasedImage.of(media.Image(content=_PNG, media_type="image/png"), reason="memory")
    association = memory.Association(
        ref="node-1", kind=memory.NodeKind.EPISODE, text="Ada", images=(released,), at=at, lanes=(memory.Lane.RECENT,)
    )
    surfaced = memory.AssociationsData(items=(association,) * 3, omitted=2)

    assert memory.AssociationsData.__model_facing_budgeted_fields__ == {
        "people": "head",
        "persons": "head",
        "items": "head",
    }
    assert surfaced.omitted == 2
    with pytest.raises(pydantic.ValidationError):
        _ = memory.AssociationsData(items=(association,) * (memory.MAX_ASSOCIATIONS + 1))
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Association(ref="node-1", kind=memory.NodeKind.EPISODE, text="Ada", at=at, lanes=())
    rendered = "".join(
        part.text
        for part in projection.model_facing_content(surfaced, capacity=media.ImageCapacity())
        if isinstance(part, text.TextPart)
    )
    assert 'omitted="memory"' in rendered
    for model in (memory.CueData, memory.Association):
        assert model.model_json_schema().get("examples"), model.__name__


def test_weights_are_a_bounded_scoring_capacity() -> None:
    weights = memory.Weights()
    assert (weights.similarity, weights.recency, weights.present, weights.linked) == (1.0, 0.35, 0.5, 0.6)
    assert weights.recency_half_life == datetime.timedelta(days=2)
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Weights(recency_half_life=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Weights(similarity=-1.0)


def test_approximate_tokens_counts_four_utf8_bytes_per_token_rounded_up() -> None:
    assert memory.approximate_tokens("") == 0
    assert memory.approximate_tokens("abcd") == 1
    assert memory.approximate_tokens("abcde") == 2
    assert memory.approximate_tokens("é" * 2) == 1


def test_a_recall_names_exactly_one_of_about_or_node() -> None:
    about = memory.RecallData(about="what Ada said about noon")
    assert (about.node, about.limit) == (None, 5)
    assert memory.RecallData(node="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f").about is None
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        _ = memory.RecallData()
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        _ = memory.RecallData(about="noon", node="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f")
    with pytest.raises(pydantic.ValidationError):
        _ = memory.RecallData(about="x" * 501)
    with pytest.raises(pydantic.ValidationError):
        _ = memory.RecallData(about="noon", limit=9)
    schema = memory.RecallData.model_json_schema()
    assert schema.get("examples")
    assert all("description" in field for field in schema["properties"].values())


def _recalled_node(words: str, *images: media.Image) -> memory.Node:
    created = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)
    return memory.Node(
        node_id="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
        kind=memory.NodeKind.EPISODE,
        text=words,
        created_at=created,
        episode=memory.Episode(
            node_id="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
            stimulus="bot.ability.conversation.messages",
            text=words,
            images=images,
            outcome=memory.Outcome.COMPLETED,
            turn_id="turn-1",
            created_at=created,
        ),
    )


def test_a_recollection_carries_image_bytes_within_the_turn_bound() -> None:
    image = media.Image(content=_PNG, media_type="image/png")
    recollection = memory.RecollectionData(about="the gate", nodes=(_recalled_node("the gate", image),))
    assert recollection.nodes[0].episode is not None
    assert recollection.nodes[0].episode.images == (image,)
    with pytest.raises(pydantic.ValidationError, match="image bytes"):
        _ = memory.RecollectionData(
            nodes=(_recalled_node("one", *(image,) * 5), _recalled_node("two", *(image,) * 4)),
        )


def test_a_recollection_is_recalled_context_and_memory_keeps_only_what_was_thought_about() -> None:
    """A recollection never fails the prompt it rides, and remembering it does not copy the memories again."""

    image = media.Image(content=_PNG, media_type="image/png")
    recollection = memory.RecollectionData(
        about="the gate",
        nodes=(_recalled_node("x" * 60_000, image), *(_recalled_node("y" * 60_000) for _ in range(4))),
    )

    assert memory.RecollectionData.__model_facing_budgeted_fields__ == {"nodes": "head", "neighbors": "head"}
    capacity = media.ImageCapacity(max_images=8, max_total_bytes=1024, media_types=frozenset({"image/png"}))
    parts = projection.model_facing_content(recollection, capacity=capacity)
    rendered = "".join(part.text for part in parts if isinstance(part, text.TextPart))
    assert 'omitted="' in rendered
    assert [part.image for part in parts if isinstance(part, text.ImagePart)] == [image]

    (kept,) = projection.memory_content(recollection, capacity=capacity)
    assert isinstance(kept, text.TextPart)
    assert 'about="the gate"' in kept.text
    assert "x" * 100 not in kept.text


def test_rest_and_wake_carry_nothing_and_a_unit_outcome_is_a_closed_label() -> None:
    assert memory.RestData() == memory.RestData()
    assert memory.WakeData() == memory.WakeData()
    for notice in (memory.RestData, memory.WakeData):
        with pytest.raises(pydantic.ValidationError):
            _ = notice.model_validate({"why": "tired"})
    for outcome in ("committed", "reviewed", "interrupted", "failed", "exhausted"):
        assert memory.UnitOutcomeData(unit="u1", outcome=outcome).outcome == outcome
    assert memory.UnitOutcomeData(outcome="exhausted").unit is None
    with pytest.raises(pydantic.ValidationError):
        _ = memory.UnitOutcomeData.model_validate({"unit": "u1", "outcome": "skipped"})
    schema = memory.UnitOutcomeData.model_json_schema()
    assert schema["properties"]["outcome"]["description"]


# --- persons (W8-5, W8-6) ---------------------------------------------------------------------


def test_a_person_id_is_self_an_owner_slug_or_a_minted_hex() -> None:
    adapter = pydantic.TypeAdapter(memory.PersonId)
    for valid in ("self", "ada", "gabe-w", "a_1", "3f2a9c0d4b5e6f708192a3b4c5d6e7f8"):
        assert adapter.validate_python(valid) == valid
    for invalid in ("", "Ada", "-ada", "1ada", "a" * 33, "voice:3f2a"):
        with pytest.raises(pydantic.ValidationError):
            _ = adapter.validate_python(invalid)
    assert memory.SELF == "self"


def test_a_name_is_bounded_and_has_no_example() -> None:
    """A name is a complete answer in the one field that decides it, so the schema offers none."""

    adapter = pydantic.TypeAdapter(memory.Name)
    assert adapter.validate_python("Ada") == "Ada"
    for invalid in ("", "x" * 65):
        with pytest.raises(pydantic.ValidationError):
            _ = adapter.validate_python(invalid)
    assert "examples" not in adapter.json_schema()
    assert "examples" not in memory.Person.model_json_schema()["properties"]["name"]


def test_a_member_is_an_owner_fact_and_never_the_bot_itself() -> None:
    member = memory.Member(id="ada", name="Ada", role=memory.Role.CHILD, born=2018, refs=("voice:3f2a",))
    assert (member.role, member.born, member.refs) == (memory.Role.CHILD, 2018, ("voice:3f2a",))
    assert memory.Member(id="gabe", name="Gabe").role is None
    assert {role.value for role in memory.Role} == {"adult", "child"}
    with pytest.raises(pydantic.ValidationError, match="self"):
        _ = memory.Member(id="self", name="Me")
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Member(id="ada", name="Ada", born=1800)
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Member(id="ada", name="Ada", refs=tuple(f"voice:{index}" for index in range(17)))
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Member.model_validate({"id": "ada", "name": "Ada", "role": "guest"})


def test_a_person_is_frozen_and_shows_at_most_sixteen_refs() -> None:
    person = memory.Person(id="ada", name="Ada", role=memory.Role.CHILD, born=2018, declared=True, refs=("voice:a",))
    assert memory.Person(id=memory.SELF).refs == ()
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Person.model_validate({"id": "ada", "refs": [f"voice:{index}" for index in range(17)]})
    with pytest.raises(pydantic.ValidationError):
        person.__setattr__("name", "Bea")


def test_a_person_statement_is_one_typed_op() -> None:
    adapter = pydantic.TypeAdapter(memory.PersonData)
    assert isinstance(adapter.validate_python({"op": "link", "ref": "voice:3f2a"}), memory.PersonLinkData)
    assert isinstance(adapter.validate_python({"op": "unlink", "ref": "voice:3f2a"}), memory.PersonUnlinkData)
    assert isinstance(adapter.validate_python({"op": "name", "person": "ada", "name": "Ada"}), memory.PersonNameData)
    assert isinstance(adapter.validate_python({"op": "forget", "person": "ada"}), memory.PersonForgetData)
    assert isinstance(adapter.validate_python({"op": "list"}), memory.PersonListData)
    declared = adapter.validate_python({"op": "declare", "members": [{"id": "ada", "name": "Ada"}]})
    assert isinstance(declared, memory.PersonDeclareData)
    with pytest.raises(pydantic.ValidationError):
        _ = adapter.validate_python({"op": "rename", "person": "ada"})


def test_a_declaration_names_each_member_and_each_ref_once() -> None:
    ada = memory.Member(id="ada", name="Ada", refs=("voice:a",))
    with pytest.raises(pydantic.ValidationError, match="once"):
        _ = memory.PersonDeclareData(members=(ada, memory.Member(id="ada", name="Ada again")))
    with pytest.raises(pydantic.ValidationError, match="once"):
        _ = memory.PersonDeclareData(members=(ada, memory.Member(id="bea", name="Bea", refs=("voice:a",))))
    assert memory.PersonDeclareData(members=()).members == ()


def test_a_person_statement_reply_carries_the_person_and_its_refs() -> None:
    done = memory.PersonDoneData(op="forget", person=memory.Person(id="ada"), refs=("voice:a", "face:b"))
    assert (done.op, done.refs, done.persons, done.refused, done.dropped) == (
        "forget",
        ("voice:a", "face:b"),
        (),
        (),
        (),
    )
    with pytest.raises(pydantic.ValidationError):
        _ = memory.PersonDoneData.model_validate({"op": "rename"})


def test_who_is_present_is_a_resolved_typed_fact() -> None:
    present = memory.Present(
        ref="voice:3f2a9c", part=mosfet.ParticipantRole.SOURCE, person="3f2a9c0d4b5e6f708192a3b4c5d6e7f8", known=False
    )
    assert present.known is False
    assert memory.Present.model_json_schema()["properties"]["known"]["description"]
    with pytest.raises(pydantic.ValidationError):
        _ = memory.Present.model_validate({"ref": "voice:a", "part": "speaker", "person": "self", "known": True})
    surfaced = memory.AssociationsData(people=(present,) * 32, persons=(memory.Person(id=memory.SELF),) * 17)
    assert surfaced.you == memory.Person(id=memory.SELF)
    with pytest.raises(pydantic.ValidationError):
        _ = memory.AssociationsData(people=(present,) * 33)
    with pytest.raises(pydantic.ValidationError):
        _ = memory.AssociationsData(persons=(memory.Person(id=memory.SELF),) * 18)
    found = memory.FoundData(people=(present,), persons=(memory.Person(id=memory.SELF),))
    assert found.you == memory.Person(id=memory.SELF)
