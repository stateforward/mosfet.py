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
    }
