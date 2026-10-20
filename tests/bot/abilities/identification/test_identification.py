"""Pure contract tests for ``identification``: ids and refs, prints, the match policy and ``decide``."""

import math
import typing

import hsm
import mosfet
import mosfet.event
import pydantic
import pytest

from mosfet.abilities import ability, identification
from mosfet.telemetry import capture

_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
_POLICIES = pydantic.TypeAdapter[identification.MatchPolicy](identification.MatchPolicy)


def _neighbor(id: str, similarity: float, *, own: bool = False) -> identification.Neighbor:
    return identification.Neighbor(id=id, similarity=similarity, own=own)


_DECISIONS: list[tuple[str, list[identification.Neighbor], identification.Decision]] = [
    ("no-neighbors-mints", [], identification.Mint(similarity=None)),
    ("below-threshold-mints", [_neighbor("a", 0.49)], identification.Mint(similarity=0.49)),
    ("margin-tie-mints", [_neighbor("a", 0.60), _neighbor("b", 0.58)], identification.Mint(similarity=0.60)),
    ("equal-best-mints", [_neighbor("a", 0.70), _neighbor("b", 0.70)], identification.Mint(similarity=0.70)),
    (
        "match-between-thresholds",
        [_neighbor("b", 0.40), _neighbor("a", 0.55)],
        identification.Match(id="a", similarity=0.55, own=False),
    ),
    (
        "refine-at-refine-threshold",
        [_neighbor("a", 0.65, own=True)],
        identification.Refine(id="a", similarity=0.65, own=True),
    ),
    (
        "refine-above-with-clear-margin",
        [_neighbor("a", 0.90), _neighbor("b", 0.80)],
        identification.Refine(id="a", similarity=0.90, own=False),
    ),
]


@pytest.mark.parametrize(
    ("neighbors", "expected"), [case[1:] for case in _DECISIONS], ids=[case[0] for case in _DECISIONS]
)
def test_decide_matches_refines_or_mints(
    neighbors: list[identification.Neighbor], expected: identification.Decision
) -> None:
    assert identification.decide(neighbors, _POLICY) == expected


def test_decide_does_not_depend_on_the_weight_cap() -> None:
    capped = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65, refine_weight_cap=1)

    assert _POLICY.refine_weight_cap == 50
    assert identification.decide([_neighbor("a", 0.9)], capped) == identification.decide([_neighbor("a", 0.9)], _POLICY)


_RESEMBLING = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65, resemble_floor=0.30)


def _linked(id: str, similarity: float) -> identification.Neighbor:
    return identification.Neighbor(id=id, similarity=similarity, own=False, linked=True)


def test_a_mint_carries_the_linked_ids_it_resembles_best_first_at_most_three_above_the_floor() -> None:
    """R-13, D-R-40: a new print that resembles known people's prints says so, as perception data."""

    neighbors = [
        _linked("a", 0.45),
        _linked("b", 0.31),
        _linked("c", 0.40),
        _linked("d", 0.35),
        _linked("e", 0.29),
        _neighbor("stranger", 0.48),
        identification.Neighbor(id="mine", similarity=0.47, own=True, linked=False),
    ]

    decision = identification.decide(neighbors, _RESEMBLING)

    assert isinstance(decision, identification.Mint)
    assert decision.resembles == (
        identification.Resemblance(id="a", similarity=0.45),
        identification.Resemblance(id="c", similarity=0.40),
        identification.Resemblance(id="d", similarity=0.35),
    )
    # No floor, no resemblance; a match or a refine resembles nothing.
    assert identification.decide(neighbors, _POLICY) == identification.Mint(similarity=0.48)
    matched = identification.decide([_linked("a", 0.9), _linked("b", 0.4)], _RESEMBLING)
    assert matched == identification.Refine(id="a", similarity=0.9, own=False)


def test_the_floor_is_a_similarity_below_the_threshold() -> None:
    assert identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65).resemble_floor is None
    for floor in (0.5, 0.6, -1.5):
        with pytest.raises(pydantic.ValidationError):
            _ = _POLICIES.validate_python(
                {"threshold": 0.5, "margin": 0.05, "refine_threshold": 0.65, "resemble_floor": floor}
            )


def test_a_recognition_shows_the_model_neither_its_similarity_nor_whom_it_resembles() -> None:
    recognition = identification.Recognition(
        id="3f2a9c",
        recognized=False,
        similarity=0.4,
        own=False,
        resembles=(identification.Resemblance(id="7b1e04", similarity=0.4),),
    )

    assert recognition.resembles[0].id == "7b1e04"
    assert {"similarity", "resembles"} <= identification.Recognition.__model_facing_excluded_fields__
    assert identification.Recognition(id="3f2a9c", recognized=True, similarity=0.9, own=False).resembles == ()


def test_a_participant_carries_the_refs_its_new_id_resembles() -> None:
    stamped = identification.participant(
        "voice", "3f2a9c", resembles=(identification.Resemblance(id="7b1e04", similarity=0.42),)
    )

    assert stamped == mosfet.Participant(
        ref="voice:3f2a9c",
        role=mosfet.ParticipantRole.SOURCE,
        resembles=(mosfet.Resemblance(ref="voice:7b1e04", similarity=0.42),),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"threshold": 1.5, "margin": 0.05, "refine_threshold": 0.65},
        {"threshold": 0.5, "margin": -0.1, "refine_threshold": 0.65},
        {"threshold": 0.7, "margin": 0.05, "refine_threshold": 0.65},
        {"threshold": 0.5, "margin": 0.05, "refine_threshold": 0.65, "refine_weight_cap": 0},
    ],
    ids=["threshold-range", "negative-margin", "refine-below-threshold", "weight-cap-zero"],
)
def test_match_policy_rejects_inconsistent_numbers(fields: dict[str, float]) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _POLICIES.validate_python(fields)


def test_match_policy_has_no_default_thresholds() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _POLICIES.validate_python({"threshold": 0.5, "margin": 0.05})


def test_ref_names_the_kind_and_participant_stamps_a_source() -> None:
    longest = "a" * 128

    assert identification.ref("voice", "tts:warm") == "voice:tts:warm"
    assert identification.ref("face", "3f2a9c") == "face:3f2a9c"
    assert len(identification.ref("voice", longest)) == 134
    assert identification.participant("voice", "tts:warm") == mosfet.Participant(
        ref="voice:tts:warm", role=mosfet.ParticipantRole.SOURCE
    )
    assert identification.participant("voice", "3f2a9c", own=True) == mosfet.Participant(
        ref="voice:3f2a9c", role=mosfet.ParticipantRole.SOURCE, own=True
    )


@pytest.mark.parametrize(
    ("kind", "id"),
    [("voice", "a" * 129), ("voice", "Upper"), ("voice", ":leading"), ("voice", ""), ("hand", "abc")],
    ids=["too-long", "uppercase", "leading-colon", "empty", "unknown-kind"],
)
def test_ref_rejects_an_invalid_kind_or_id(kind: str, id: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = identification.ref(typing.cast(identification.Kind, kind), id)


def test_id_and_ref_types_validate_their_patterns() -> None:
    ids = pydantic.TypeAdapter[str](identification.Id)
    refs = pydantic.TypeAdapter[str](identification.Ref)

    assert ids.validate_python("tts:warm") == "tts:warm"
    assert refs.validate_python("face:" + "a" * 128) == "face:" + "a" * 128
    with pytest.raises(pydantic.ValidationError):
        _ = refs.validate_python("person:abc")
    with pytest.raises(pydantic.ValidationError):
        _ = ids.validate_python("a" * 129)


def test_print_is_biometric_and_stores_a_unit_vector() -> None:
    print = identification.Print(vector=(3.0, 4.0), model="m")

    assert isinstance(print, capture.Biometric)
    assert print.vector == (0.6, 0.8)
    assert math.isclose(math.fsum(value * value for value in print.vector), 1.0)
    assert capture.jsonable(print) == "<biometric:Print>"


@pytest.mark.parametrize(
    "fields",
    [
        {"vector": (), "model": "m"},
        {"vector": (0.0, 0.0), "model": "m"},
        {"vector": (math.nan, 1.0), "model": "m"},
        {"vector": (math.inf, 1.0), "model": "m"},
        {"vector": (1.0,) * 4097, "model": "m"},
        {"vector": (1.0,), "model": ""},
        {"vector": (1.0,), "model": "m" * 129},
        {"vector": (1.0,), "model": "m", "extra": 1},
    ],
    ids=["empty", "zero", "nan", "inf", "too-many-dimensions", "no-model", "long-model", "extra-field"],
)
def test_print_rejects_vectors_it_cannot_normalize(fields: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = identification.Print.model_validate(fields)


def test_recognition_bounds_similarity() -> None:
    assert identification.Recognition(id="a", recognized=False, similarity=None, own=False).similarity is None
    with pytest.raises(pydantic.ValidationError):
        _ = identification.Recognition(id="a", recognized=True, similarity=1.5, own=False)


def test_identify_carries_one_to_sixteen_prints() -> None:
    print = identification.Print(vector=(1.0,), model="m")

    assert len(identification.IdentifyData(prints=(print,) * 16).prints) == 16
    with pytest.raises(pydantic.ValidationError):
        _ = identification.IdentifyData(prints=())
    with pytest.raises(pydantic.ValidationError):
        _ = identification.IdentifyData(prints=(print,) * 17)


def test_an_enrollment_without_an_id_must_be_the_bots_own() -> None:
    print = identification.Print(vector=(1.0,), model="m")

    own = identification.EnrollData(id=None, print=print, own=True)
    declared = identification.EnrollData(id="3f2a9c", print=print, own=False)

    assert own.id is None
    assert declared.id == "3f2a9c"
    with pytest.raises(pydantic.ValidationError, match="own"):
        _ = identification.EnrollData(id=None, print=print, own=False)
    with pytest.raises(pydantic.ValidationError):
        _ = identification.EnrollData.model_validate({"print": print, "own": True})


def test_events_are_named_in_the_identification_domain_and_never_model_tools() -> None:
    events = [
        (identification.IdentifyEvent, "bot.ability.identification.identify"),
        (identification.IdentifiedEvent, "bot.ability.identification.identified"),
        (identification.EnrollEvent, "bot.ability.identification.enroll"),
        (identification.EnrolledEvent, "bot.ability.identification.enrolled"),
        (identification.ForgetEvent, "bot.ability.identification.forget"),
        (identification.ForgottenEvent, "bot.ability.identification.forgotten"),
        (identification.LinkEvent, "bot.ability.identification.link"),
        (identification.LinkedEvent, "bot.ability.identification.linked"),
        (identification.FailedEvent, "bot.ability.identification.failed"),
    ]

    for event, name in events:
        assert event.name == name
        assert not hsm.kind.Is(event.kind, mosfet.event.EventKind)
    assert identification.Registry.failed_event is identification.FailedEvent
    assert identification.Registry.detached_requests == (
        identification.IdentifyEvent,
        identification.EnrollEvent,
        identification.ForgetEvent,
        identification.LinkEvent,
    )


def test_a_link_names_one_valid_id_and_whether_it_is_linked() -> None:
    link = identification.LinkData(id="3f2a9c", linked=True)

    assert identification.LinkedData(id=link.id, linked=link.linked) == identification.LinkedData(
        id="3f2a9c", linked=True
    )
    with pytest.raises(pydantic.ValidationError):
        _ = identification.LinkData.model_validate({"id": "Upper", "linked": True})
    with pytest.raises(pydantic.ValidationError):
        _ = identification.LinkData.model_validate({"id": "3f2a9c"})
    with pytest.raises(pydantic.ValidationError):
        _ = identification.LinkData.model_validate({"id": "3f2a9c", "linked": True, "person": "gabe"})
    assert identification.LinkData.model_config.get("frozen") is True
    assert identification.LinkedData.model_config.get("frozen") is True


def test_failure_kinds_reuse_the_shared_failure_strings() -> None:
    assert {kind.value for kind in identification.FailureKind} == {
        "invalid_input",
        "unavailable",
        "store_failed",
        "timeout",
        "not_found",
        "conflict",
    }
    failure = identification.FailureData(kind=identification.FailureKind.TIMEOUT, message="late")
    assert isinstance(failure, ability.FailureData)
    error = identification.StoreError(identification.FailureKind.STORE_FAILED, "disk full")
    assert error.kind is identification.FailureKind.STORE_FAILED
    assert str(error) == "disk full"
