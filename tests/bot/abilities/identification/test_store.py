"""``identification.InProcessStore`` through ``identification.Matcher``: the store contract with a fake wall clock."""

import asyncio
import dataclasses
import datetime
import math
import threading
import typing

import pytest

from mosfet.abilities import identification

_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
_START = datetime.datetime(2026, 10, 15, 12, 0, tzinfo=datetime.UTC)
_MODEL = "voice-model"


def _print(*vector: float, model: str = _MODEL) -> identification.Print:
    return identification.Print(vector=vector, model=model)


def _angle(degrees: float) -> identification.Print:
    """A 2-D unit print ``degrees`` from the x axis; cosine between two is the cosine of their angle."""

    radians = math.radians(degrees)
    return _print(math.cos(radians), math.sin(radians))


@dataclasses.dataclass
class Clock:
    now: datetime.datetime = _START

    def __call__(self) -> datetime.datetime:
        return self.now

    def advance(self, delta: datetime.timedelta) -> None:
        self.now += delta


@dataclasses.dataclass
class Fixture:
    store: identification.InProcessStore
    matcher: identification.Matcher
    clock: Clock


def _fixture(*, max_ids: int = 100, policy: identification.MatchPolicy = _POLICY) -> Fixture:
    clock = Clock()
    store = identification.InProcessStore(max_ids=max_ids)
    return Fixture(store=store, matcher=identification.Matcher(store=store, policy=policy, now=clock), clock=clock)


def _identify(
    fixture: Fixture, *prints: identification.Print, kind: identification.Kind = "voice"
) -> list[identification.Recognition]:
    recognitions = asyncio.run(fixture.matcher.identify(kind, prints))
    assert all(recognition is not None for recognition in recognitions)
    return [typing.cast(identification.Recognition, recognition) for recognition in recognitions]


def _enroll(
    fixture: Fixture,
    id: str | None,
    print: identification.Print,
    *,
    own: bool = False,
    kind: identification.Kind = "voice",
) -> identification.Recognition:
    return asyncio.run(fixture.matcher.enroll(kind, id, print, own))


def _prune(
    fixture: Fixture, *, age: datetime.timedelta, max_ids: int = 100, kind: identification.Kind = "voice"
) -> int:
    return asyncio.run(fixture.matcher.prune(kind, unseen_before=fixture.clock() - age, max_ids=max_ids))


def test_a_new_print_is_minted_and_a_near_print_is_recognized() -> None:
    fixture = _fixture()

    (first,) = _identify(fixture, _angle(0))
    (again,) = _identify(fixture, _angle(10))

    assert first.recognized is False
    assert first.similarity is None
    assert first.own is False
    assert len(first.id) == 32
    assert again.id == first.id
    assert again.recognized is True
    assert again.similarity == pytest.approx(math.cos(math.radians(10)))


def test_a_far_print_mints_a_second_id_with_the_best_similarity() -> None:
    fixture = _fixture()

    (first,) = _identify(fixture, _angle(0))
    (other,) = _identify(fixture, _angle(90))

    assert other.id != first.id
    assert other.recognized is False
    assert other.similarity == pytest.approx(0.0, abs=1e-9)


def test_prints_in_one_request_are_identified_in_order() -> None:
    fixture = _fixture()

    first, second, third = _identify(fixture, _angle(0), _angle(90), _angle(5))

    assert [first.recognized, second.recognized, third.recognized] == [False, False, True]
    assert third.id == first.id


def test_kinds_are_isolated() -> None:
    fixture = _fixture()

    (voice,) = _identify(fixture, _angle(0), kind="voice")
    (face,) = _identify(fixture, _angle(0), kind="face")

    assert face.recognized is False
    assert face.id != voice.id
    assert asyncio.run(fixture.matcher.forget("face", voice.id)) is False


def test_models_are_separate_spaces() -> None:
    fixture = _fixture()

    (voice,) = _identify(fixture, _print(1.0, 0.0, model="a"))
    (other,) = _identify(fixture, _print(1.0, 0.0, model="b"))

    assert other.recognized is False
    assert other.id != voice.id


def test_an_own_enrollment_is_recognized_as_own_and_never_pruned() -> None:
    fixture = _fixture()

    enrolled = _enroll(fixture, "tts:warm", _angle(0), own=True)
    (heard,) = _identify(fixture, _angle(5))
    fixture.clock.advance(datetime.timedelta(days=365))
    pruned = _prune(fixture, age=datetime.timedelta(days=1), max_ids=0)
    (later,) = _identify(fixture, _angle(5))

    assert enrolled == identification.Recognition(id="tts:warm", recognized=False, similarity=None, own=True)
    assert heard.id == "tts:warm"
    assert heard.own is True
    assert pruned == 0
    assert later.id == "tts:warm"


def test_forget_reports_existence_and_removes_the_print() -> None:
    fixture = _fixture()
    (first,) = _identify(fixture, _angle(0))

    assert asyncio.run(fixture.matcher.forget("voice", first.id)) is True
    assert asyncio.run(fixture.matcher.forget("voice", first.id)) is False
    (again,) = _identify(fixture, _angle(0))
    assert again.recognized is False
    assert again.id != first.id


def test_prune_counts_age_from_last_seen() -> None:
    fixture = _fixture()
    (stale,) = _identify(fixture, _angle(0))
    (seen,) = _identify(fixture, _angle(90))
    fixture.clock.advance(datetime.timedelta(days=40))
    # Created 40 days ago, but seen again now: kept.
    (again,) = _identify(fixture, _angle(92))
    fixture.clock.advance(datetime.timedelta(days=1))

    removed = _prune(fixture, age=datetime.timedelta(days=30))
    (after,) = _identify(fixture, _angle(0))
    (kept,) = _identify(fixture, _angle(90))

    assert again.id == seen.id
    assert removed == 1
    assert after.recognized is False
    assert after.id != stale.id
    assert kept.id == seen.id


def test_a_plain_match_advances_last_seen_without_refining() -> None:
    fixture = _fixture()
    (first,) = _identify(fixture, _angle(0))
    fixture.clock.advance(datetime.timedelta(days=40))
    # cos(55°) ≈ 0.574: between threshold and refine_threshold, so a Match.
    (matched,) = _identify(fixture, _angle(55))
    fixture.clock.advance(datetime.timedelta(days=1))

    removed = _prune(fixture, age=datetime.timedelta(days=30))
    (exact,) = _identify(fixture, _angle(0))

    assert matched.id == first.id
    assert matched.similarity == pytest.approx(math.cos(math.radians(55)))
    assert removed == 0
    # The stored vector did not move toward 55°: the original print still matches exactly.
    assert exact.similarity == pytest.approx(1.0)


def test_prune_over_the_cap_removes_the_least_recently_seen_first() -> None:
    fixture = _fixture()
    ids: list[str] = []
    for degrees in (0, 90, 180):
        (recognition,) = _identify(fixture, _angle(degrees))
        ids.append(recognition.id)
        fixture.clock.advance(datetime.timedelta(minutes=1))
    _ = _enroll(fixture, "tts:warm", _angle(270), own=True)

    removed = _prune(fixture, age=datetime.timedelta(days=30), max_ids=2)
    remaining = {
        recognition.id
        for recognition in _identify(fixture, _angle(0), _angle(90), _angle(180), _angle(270))
        if recognition.recognized
    }

    assert removed == 2
    assert remaining == {ids[2], "tts:warm"}


def test_a_dimension_mismatch_for_a_model_is_invalid_input_and_changes_nothing() -> None:
    fixture = _fixture()
    _ = _identify(fixture, _angle(0))

    with pytest.raises(identification.StoreError) as raised:
        _ = asyncio.run(fixture.matcher.identify("voice", (_angle(90), _print(1.0, 0.0, 0.0))))
    with pytest.raises(identification.StoreError) as enrolled:
        _ = _enroll(fixture, "tts:warm", _print(1.0, 0.0, 0.0))
    (after,) = _identify(fixture, _angle(90))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT
    assert enrolled.value.kind is identification.FailureKind.INVALID_INPUT
    # The 90° print before the bad one was not committed: the request was one transaction.
    assert after.recognized is False
    assert "0.0" not in str(raised.value)


def test_enroll_upserts_refines_up_to_the_weight_cap_and_keeps_own_sticky() -> None:
    policy = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65, refine_weight_cap=2)
    fixture = _fixture(policy=policy)

    first = _enroll(fixture, "tts:warm", _angle(0), own=True)
    second = _enroll(fixture, "tts:warm", _angle(60), own=False)
    third = _enroll(fixture, "tts:warm", _angle(60), own=False)
    (heard,) = _identify(fixture, _angle(40))

    assert first.recognized is False
    assert second.recognized is True
    assert second.similarity == pytest.approx(0.5)
    assert second.own is True
    # Weight 1 then capped at 2: 0° and 60° average to 30°, then (2·30° + 60°)/3 lands near 40°.
    assert third.similarity == pytest.approx(math.cos(math.radians(30)))
    assert heard.id == "tts:warm"
    assert heard.own is True
    assert heard.similarity == pytest.approx(1.0, abs=1e-3)


def test_enroll_with_another_model_for_an_existing_id_is_invalid_input() -> None:
    fixture = _fixture()
    _ = _enroll(fixture, "tts:warm", _angle(0))

    with pytest.raises(identification.StoreError) as raised:
        _ = _enroll(fixture, "tts:warm", _print(1.0, 0.0, model="other"))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT


def test_an_own_print_without_an_id_mints_an_own_id_then_refines_it() -> None:
    fixture = _fixture()

    minted = _enroll(fixture, None, _angle(0), own=True)
    # cos(55°) ≈ 0.574: only a Match under the policy, yet an own enrollment still refines the id.
    matched = _enroll(fixture, None, _angle(55), own=True)
    (heard,) = _identify(fixture, _angle(27.5))
    fixture.clock.advance(datetime.timedelta(days=365))
    pruned = _prune(fixture, age=datetime.timedelta(days=1), max_ids=0)

    assert (minted.recognized, minted.similarity, minted.own) == (False, None, True)
    assert len(minted.id) == 32
    assert (matched.id, matched.recognized, matched.own) == (minted.id, True, True)
    assert matched.similarity == pytest.approx(math.cos(math.radians(55)))
    assert (heard.id, heard.own) == (minted.id, True)
    assert heard.similarity == pytest.approx(1.0)
    assert pruned == 0


def test_an_own_print_never_matches_an_id_that_is_not_own() -> None:
    fixture = _fixture()
    (stranger,) = _identify(fixture, _angle(0))
    _ = _enroll(fixture, None, _angle(0), own=True, kind="face")

    own = _enroll(fixture, None, _angle(0), own=True)
    far = _enroll(fixture, None, _angle(90), own=True)

    assert own.id != stranger.id
    # Neither the stranger nor the own face was a candidate.
    assert (own.recognized, own.similarity, own.own) == (False, None, True)
    # The best own candidate is reported, as for any mint.
    assert (far.recognized, far.own) == (False, True)
    assert far.id != own.id
    assert far.similarity == pytest.approx(0.0, abs=1e-9)
    assert asyncio.run(fixture.matcher.forget("voice", stranger.id)) is True


def test_an_enrollment_without_an_id_that_is_not_own_is_invalid_input() -> None:
    fixture = _fixture()

    with pytest.raises(identification.StoreError) as raised:
        _ = _enroll(fixture, None, _angle(0), own=False)
    (after,) = _identify(fixture, _angle(0))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT
    assert after.recognized is False


def test_minting_an_own_id_at_the_hard_cap_evicts_the_least_recently_seen_non_own_id() -> None:
    fixture = _fixture(max_ids=1)
    (stranger,) = _identify(fixture, _angle(90))

    own = _enroll(fixture, None, _angle(0), own=True)

    assert (own.recognized, own.own) == (False, True)
    assert asyncio.run(fixture.matcher.forget("voice", stranger.id)) is False


def test_minting_at_the_hard_cap_evicts_the_least_recently_seen_non_own_id() -> None:
    fixture = _fixture(max_ids=2)
    (oldest,) = _identify(fixture, _angle(0))
    fixture.clock.advance(datetime.timedelta(minutes=1))
    (newer,) = _identify(fixture, _angle(90))
    fixture.clock.advance(datetime.timedelta(minutes=1))

    (third,) = _identify(fixture, _angle(180))
    (back,) = _identify(fixture, _angle(0))

    assert third.recognized is False
    assert back.recognized is False
    assert back.id != oldest.id
    assert asyncio.run(fixture.matcher.forget("voice", newer.id)) is False


def test_minting_at_the_hard_cap_of_own_ids_fails_store_failed() -> None:
    fixture = _fixture(max_ids=1)
    _ = _enroll(fixture, "tts:warm", _angle(0), own=True)

    with pytest.raises(identification.StoreError) as raised:
        _ = _identify(fixture, _angle(90))

    assert raised.value.kind is identification.FailureKind.STORE_FAILED


def test_the_store_rejects_a_non_positive_cap() -> None:
    with pytest.raises(ValueError, match="max_ids"):
        _ = identification.InProcessStore(max_ids=0)


def test_store_calls_from_two_threads_serialize() -> None:
    store = identification.InProcessStore(max_ids=1000)
    errors: list[BaseException] = []

    def mint(kind: identification.Kind) -> None:
        try:
            for index in range(200):
                radians = math.radians(index * 1.7)
                _ = store.identify(kind, (_print(math.cos(radians), math.sin(radians), 1.0),), _POLICY, at=_START)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=mint, args=(kind,)) for kind in ("voice", "voice", "face")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
