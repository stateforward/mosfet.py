"""``IdentityStore``: identities and their prints in the memory database, beside memory's own tables.

The store contract (``identification.Store``) is pinned once by the parametrized ``contract_*`` tests,
which run the same cases against core ``InProcessStore`` and ``IdentityStore`` on a temp-file
database. The rest pin what only the database store has: the 0001 table, one ``vec0`` table per
(model, dims), SQLite-assigned ``seq``, transactions beside Memory, and the engines it refuses.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import hashlib
import math
import pathlib
import typing
import uuid

import hsm
import pytest
import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mosfet.abilities import ability, embedding, identification, memory
from mosfet.abilities.language import text as language
from mosfet.define import define
from mosfet.environment import Environment
from mosfet.protocols import attachment
from mosfet.providers import sqlite_memory
from mosfet.start import started

_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
_START = datetime.datetime(2026, 10, 15, 12, 0, tzinfo=datetime.UTC)
_MODEL = "voice-model"
_SPACE = embedding.Space(model="test/embedder", dimensions=2)
# Vectors are stored as float32 in vec0, so similarities agree to float32 precision.
_CLOSE = 1e-5


def _print(*vector: float, model: str = _MODEL) -> identification.Print:
    return identification.Print(vector=vector, model=model)


def _angle(degrees: float, *, model: str = _MODEL) -> identification.Print:
    """A 2-D unit print ``degrees`` from the x axis; cosine between two is the cosine of their angle."""

    radians = math.radians(degrees)
    return _print(math.cos(radians), math.sin(radians), model=model)


def _vec_table(model: str, dimensions: int) -> str:
    return f"bot_identity_vec_{dimensions}_{hashlib.sha256(model.encode('utf-8')).hexdigest()[:8]}"


def _engine(tmp_path: pathlib.Path) -> Engine:
    return sqlite_memory.open_engine(database_path=tmp_path / f"{uuid.uuid4().hex}.sqlite3")


@dataclasses.dataclass
class Clock:
    now: datetime.datetime = _START

    def __call__(self) -> datetime.datetime:
        return self.now

    def advance(self, delta: datetime.timedelta) -> None:
        self.now += delta


@dataclasses.dataclass
class Fixture:
    matcher: identification.Matcher
    clock: Clock

    def identify(
        self, *prints: identification.Print, kind: identification.Kind = "voice"
    ) -> list[identification.Recognition]:
        recognitions = asyncio.run(self.matcher.identify(kind, prints, (True,) * len(prints)))
        assert all(recognition is not None for recognition in recognitions)
        return [typing.cast(identification.Recognition, recognition) for recognition in recognitions]

    def enroll(
        self, id: str | None, print: identification.Print, *, own: bool = False, kind: identification.Kind = "voice"
    ) -> identification.Recognition:
        return asyncio.run(self.matcher.enroll(kind, id, print, own))

    def forget(self, id: str, *, kind: identification.Kind = "voice") -> bool:
        return asyncio.run(self.matcher.forget(kind, id))

    def prune(self, *, age: datetime.timedelta, max_ids: int = 100, kind: identification.Kind = "voice") -> int:
        return asyncio.run(self.matcher.prune(kind, unseen_before=self.clock() - age, max_ids=max_ids))

    def link(self, id: str, linked: bool, *, kind: identification.Kind = "voice") -> None:
        asyncio.run(self.matcher.link(kind, id, linked))


def _minted(
    store: identification.Store, kind: identification.Kind, *prints: identification.Print, at: datetime.datetime
) -> tuple[identification.Recognition, ...]:
    """Identify ``prints`` on ``store`` directly, every unknown one minted, so each has a recognition."""

    recognitions = store.identify(kind, prints, _POLICY, at=at, mint=(True,) * len(prints))
    assert all(recognition is not None for recognition in recognitions)
    return tuple(typing.cast(identification.Recognition, recognition) for recognition in recognitions)


StoreFactory: typing.TypeAlias = collections.abc.Callable[[], identification.Store]


@pytest.fixture(params=["in_process", "sqlite"])
def store_factory(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> collections.abc.Iterator[StoreFactory]:
    """Both stores, so the store contract is pinned once for each."""

    engines: list[Engine] = []

    def make() -> identification.Store:
        if request.param == "in_process":
            return identification.InProcessStore(max_ids=100)
        engine = _engine(tmp_path)
        engines.append(engine)
        return sqlite_memory.IdentityStore(engine=engine)

    yield make
    for engine in engines:
        engine.dispose()


def _fixture(make: StoreFactory, *, policy: identification.MatchPolicy = _POLICY) -> Fixture:
    clock = Clock()
    return Fixture(matcher=identification.Matcher(store=make(), policy=policy, now=clock), clock=clock)


# --- the store contract, once for both stores ---------------------------------------------------


def test_contract_a_new_print_is_minted_and_a_near_print_is_recognized(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    (first,) = fixture.identify(_angle(0))
    (again,) = fixture.identify(_angle(10))
    (other,) = fixture.identify(_angle(90))

    assert first.recognized is False
    assert first.similarity is None
    assert len(first.id) == 32
    assert again.id == first.id
    assert again.recognized is True
    assert again.similarity == pytest.approx(math.cos(math.radians(10)), abs=_CLOSE)
    assert other.recognized is False
    assert other.id != first.id
    # The 10° print refined the id to the running mean, 5°, so 90° is 85° from it.
    assert other.similarity == pytest.approx(math.cos(math.radians(85)), abs=_CLOSE)


def test_contract_a_match_only_print_matches_and_one_nobody_has_is_unidentified_storing_nothing(
    store_factory: StoreFactory,
) -> None:
    """D-W8-21 D5: a match-only print (an overlap's) names a known id, and one that matches nobody is None."""

    fixture = _fixture(store_factory)
    (known,) = fixture.identify(_angle(0))

    matched, unknown = asyncio.run(fixture.matcher.identify("voice", (_angle(10), _angle(90)), (False, False)))
    (minted,) = fixture.identify(_angle(90))

    assert matched is not None and (matched.id, matched.recognized) == (known.id, True)
    assert unknown is None
    # Nothing was stored for the unknown print: identifying it again with minting allowed mints it now.
    assert minted.recognized is False


def test_contract_prints_in_one_request_are_identified_in_order(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    first, second, third = fixture.identify(_angle(0), _angle(90), _angle(5))

    assert [first.recognized, second.recognized, third.recognized] == [False, False, True]
    assert third.id == first.id


def test_contract_voice_and_face_prints_of_one_model_never_match(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    (voice,) = fixture.identify(_angle(0), kind="voice")
    (face,) = fixture.identify(_angle(0), kind="face")

    assert face.recognized is False
    assert face.id != voice.id
    assert fixture.forget(voice.id, kind="face") is False
    assert fixture.forget(voice.id, kind="voice") is True


def test_contract_models_are_separate_spaces(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    (first,) = fixture.identify(_angle(0, model="a"))
    (other,) = fixture.identify(_angle(0, model="b"))

    assert other.recognized is False
    assert other.id != first.id


def test_contract_an_own_enrollment_is_recognized_as_own_and_never_pruned(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    enrolled = fixture.enroll("tts:warm", _angle(0), own=True)
    (heard,) = fixture.identify(_angle(5))
    fixture.clock.advance(datetime.timedelta(days=365))
    pruned = fixture.prune(age=datetime.timedelta(days=1), max_ids=0)
    (later,) = fixture.identify(_angle(5))

    assert enrolled == identification.Recognition(id="tts:warm", recognized=False, similarity=None, own=True)
    assert (heard.id, heard.own) == ("tts:warm", True)
    assert pruned == 0
    assert later.id == "tts:warm"


def test_contract_forget_reports_existence_and_removes_the_print(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    (first,) = fixture.identify(_angle(0))

    assert fixture.forget(first.id) is True
    assert fixture.forget(first.id) is False
    (again,) = fixture.identify(_angle(0))
    assert again.recognized is False
    assert again.id != first.id


def test_contract_prune_counts_age_from_last_seen(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    (stale,) = fixture.identify(_angle(0))
    (seen,) = fixture.identify(_angle(90))
    fixture.clock.advance(datetime.timedelta(days=40))
    # Created 40 days ago, but seen again now: kept.
    (again,) = fixture.identify(_angle(92))
    fixture.clock.advance(datetime.timedelta(days=1))

    removed = fixture.prune(age=datetime.timedelta(days=30))
    (after,) = fixture.identify(_angle(0))
    (kept,) = fixture.identify(_angle(90))

    assert again.id == seen.id
    assert removed == 1
    assert after.recognized is False
    assert after.id != stale.id
    assert kept.id == seen.id


def test_contract_a_plain_match_advances_last_seen_without_refining(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    (first,) = fixture.identify(_angle(0))
    fixture.clock.advance(datetime.timedelta(days=40))
    # cos(55°) ≈ 0.574: between threshold and refine_threshold, so a Match.
    (matched,) = fixture.identify(_angle(55))
    fixture.clock.advance(datetime.timedelta(days=1))

    removed = fixture.prune(age=datetime.timedelta(days=30))
    (exact,) = fixture.identify(_angle(0))

    assert matched.id == first.id
    assert matched.similarity == pytest.approx(math.cos(math.radians(55)), abs=_CLOSE)
    assert removed == 0
    # The stored vector did not move toward 55°: the original print still matches exactly.
    assert exact.similarity == pytest.approx(1.0, abs=_CLOSE)


def test_contract_prune_over_the_cap_removes_the_least_recently_seen_first(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    ids: list[str] = []
    for degrees in (0, 90, 180):
        (recognition,) = fixture.identify(_angle(degrees))
        ids.append(recognition.id)
        fixture.clock.advance(datetime.timedelta(minutes=1))
    _ = fixture.enroll("tts:warm", _angle(270), own=True)

    removed = fixture.prune(age=datetime.timedelta(days=30), max_ids=2)
    remaining = {
        recognition.id
        for recognition in fixture.identify(_angle(0), _angle(90), _angle(180), _angle(270))
        if recognition.recognized
    }

    assert removed == 2
    assert remaining == {ids[2], "tts:warm"}


def test_contract_a_dimension_mismatch_is_invalid_input_and_changes_nothing(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    _ = fixture.identify(_angle(0))

    with pytest.raises(identification.StoreError) as raised:
        _ = asyncio.run(fixture.matcher.identify("voice", (_angle(90), _print(1.0, 0.0, 0.0)), (True, True)))
    with pytest.raises(identification.StoreError) as enrolled:
        _ = fixture.enroll("tts:warm", _print(1.0, 0.0, 0.0))
    (after,) = fixture.identify(_angle(90))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT
    assert enrolled.value.kind is identification.FailureKind.INVALID_INPUT
    # The 90° print before the bad one was not committed: the request was one transaction.
    assert after.recognized is False
    assert "0.0" not in str(raised.value)


def test_contract_enroll_upserts_refines_up_to_the_weight_cap_and_keeps_own_sticky(
    store_factory: StoreFactory,
) -> None:
    policy = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65, refine_weight_cap=2)
    fixture = _fixture(store_factory, policy=policy)

    first = fixture.enroll("tts:warm", _angle(0), own=True)
    second = fixture.enroll("tts:warm", _angle(60), own=False)
    third = fixture.enroll("tts:warm", _angle(60), own=False)
    (heard,) = fixture.identify(_angle(40))

    assert first.recognized is False
    assert second.recognized is True
    assert second.similarity == pytest.approx(0.5, abs=_CLOSE)
    assert second.own is True
    # Weight 1 then capped at 2: 0° and 60° average to 30°, then (2·30° + 60°)/3 lands near 40°.
    assert third.similarity == pytest.approx(math.cos(math.radians(30)), abs=_CLOSE)
    assert (heard.id, heard.own) == ("tts:warm", True)
    assert heard.similarity == pytest.approx(1.0, abs=1e-3)


def test_contract_enroll_with_another_model_for_an_existing_id_is_invalid_input(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    _ = fixture.enroll("tts:warm", _angle(0))

    with pytest.raises(identification.StoreError) as raised:
        _ = fixture.enroll("tts:warm", _angle(0, model="other"))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT


def test_contract_an_own_print_without_an_id_mints_an_own_id_then_refines_it(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    minted = fixture.enroll(None, _angle(0), own=True)
    # cos(55°) ≈ 0.574: only a Match under the policy, yet an own enrollment still refines the id.
    matched = fixture.enroll(None, _angle(55), own=True)
    (heard,) = fixture.identify(_angle(27.5))
    fixture.clock.advance(datetime.timedelta(days=365))
    pruned = fixture.prune(age=datetime.timedelta(days=1), max_ids=0)

    assert (minted.recognized, minted.similarity, minted.own) == (False, None, True)
    assert len(minted.id) == 32
    assert (matched.id, matched.recognized, matched.own) == (minted.id, True, True)
    assert matched.similarity == pytest.approx(math.cos(math.radians(55)), abs=_CLOSE)
    assert (heard.id, heard.own) == (minted.id, True)
    assert heard.similarity == pytest.approx(1.0, abs=_CLOSE)
    assert pruned == 0


def test_contract_an_own_print_never_matches_an_id_that_is_not_own(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)
    (stranger,) = fixture.identify(_angle(0))
    _ = fixture.enroll(None, _angle(0), own=True, kind="face")

    own = fixture.enroll(None, _angle(0), own=True)
    far = fixture.enroll(None, _angle(90), own=True)

    assert own.id != stranger.id
    # Neither the stranger nor the own face was a candidate.
    assert (own.recognized, own.similarity, own.own) == (False, None, True)
    # The best own candidate is reported, as for any mint.
    assert (far.recognized, far.own) == (False, True)
    assert far.id != own.id
    assert far.similarity == pytest.approx(0.0, abs=_CLOSE)
    assert fixture.forget(stranger.id) is True


def test_contract_an_enrollment_without_an_id_that_is_not_own_is_invalid_input(store_factory: StoreFactory) -> None:
    fixture = _fixture(store_factory)

    with pytest.raises(identification.StoreError) as raised:
        _ = fixture.enroll(None, _angle(0), own=False)
    (after,) = fixture.identify(_angle(0))

    assert raised.value.kind is identification.FailureKind.INVALID_INPUT
    assert after.recognized is False


def test_contract_a_linked_id_is_kept_through_retention_and_the_cap_until_unlinked(
    store_factory: StoreFactory,
) -> None:
    fixture = _fixture(store_factory)
    (known,) = fixture.identify(_angle(0))
    (stranger,) = fixture.identify(_angle(90))
    fixture.link(known.id, True)
    fixture.link(known.id, True)
    fixture.clock.advance(datetime.timedelta(days=40))

    kept = fixture.prune(age=datetime.timedelta(days=30), max_ids=0)
    (again,) = fixture.identify(_angle(0))
    fixture.link(known.id, False)
    fixture.clock.advance(datetime.timedelta(days=40))
    removed = fixture.prune(age=datetime.timedelta(days=30))
    (after,) = fixture.identify(_angle(0))

    assert kept == 1
    assert (again.id, again.recognized) == (known.id, True)
    assert fixture.forget(stranger.id) is False
    assert removed == 1
    assert after.recognized is False
    assert after.id != known.id


def test_contract_a_new_print_says_which_linked_ids_it_resembles_past_nearer_strangers(
    store_factory: StoreFactory,
) -> None:
    """R-13: a linked id resembled by a new print is found even when strangers' prints are nearer."""

    fixture = _fixture(
        store_factory,
        policy=identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65, resemble_floor=0.30),
    )
    (known,) = fixture.identify(_print(1.0, 0.0, 0.0))
    fixture.link(known.id, True)
    # Two strangers, each nearer the new print (0.49) than the known one is (cos 61 degrees, 0.485).
    strangers = fixture.identify(_print(0.0, 0.56, 0.828), _print(0.0, 0.56, -0.828))

    (new,) = fixture.identify(_print(math.cos(math.radians(61)), math.sin(math.radians(61)), 0.0))

    assert new.recognized is False and new.id not in {known.id, *(stranger.id for stranger in strangers)}
    assert [(resemblance.id, round(resemblance.similarity, 4)) for resemblance in new.resembles] == [
        (known.id, round(math.cos(math.radians(61)), 4))
    ]


def test_contract_a_link_mark_on_an_own_id_is_a_conflict_and_on_an_unknown_id_not_found(
    store_factory: StoreFactory,
) -> None:
    fixture = _fixture(store_factory)
    own = fixture.enroll(None, _angle(0), own=True)
    (voice,) = fixture.identify(_angle(90))
    cases: list[tuple[identification.Kind, str, bool]] = [
        ("voice", own.id, True),
        ("voice", own.id, False),
        ("voice", "nobody", True),
        ("face", voice.id, True),
    ]

    refused: list[identification.FailureKind] = []
    for kind, id, linked in cases:
        with pytest.raises(identification.StoreError) as raised:
            fixture.link(id, linked, kind=kind)
        refused.append(raised.value.kind)

    assert refused == [
        identification.FailureKind.CONFLICT,
        identification.FailureKind.CONFLICT,
        identification.FailureKind.NOT_FOUND,
        identification.FailureKind.NOT_FOUND,
    ]


# --- what only the database store has -----------------------------------------------------------


def test_a_link_mark_writes_the_linked_column_and_nothing_else(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    later = _START + datetime.timedelta(hours=1)
    (voice,) = _minted(store, "voice", _angle(0), at=_START)
    query = text("SELECT linked, seen_at, seen_count FROM bot_identity WHERE id = :id")

    store.link("voice", voice.id, True)
    with engine.connect() as connection:
        linked = connection.execute(query, {"id": voice.id}).one()
    (_,) = _minted(store, "voice", _angle(0), at=later)
    store.link("voice", voice.id, False)
    with engine.connect() as connection:
        unlinked = connection.execute(query, {"id": voice.id}).one()
    engine.dispose()

    assert tuple(linked) == (1, "2026-10-15T12:00:00.000000+00:00", 1)
    assert tuple(unlinked) == (0, "2026-10-15T13:00:00.000000+00:00", 2)


def test_the_baseline_creates_bot_identity_with_a_never_reused_seq(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)

    (first,) = _minted(store, "voice", _angle(0), at=_START)
    assert store.forget("voice", first.id) is True
    (second,) = _minted(store, "voice", _angle(0), at=_START)
    with engine.connect() as connection:
        ddl = connection.execute(text("SELECT sql FROM sqlite_master WHERE name = 'bot_identity'")).scalar_one()
        unique = sqlalchemy.inspect(connection).get_unique_constraints("bot_identity")
        rows = connection.execute(text("SELECT seq, id FROM bot_identity")).all()
        vectors = connection.execute(text(f"SELECT seq FROM {_vec_table(_MODEL, 2)}")).scalars().all()
    engine.dispose()

    assert "AUTOINCREMENT" in str(ddl)
    assert {"name": "bot_identity_kind_id_key", "column_names": ["kind", "id"]} in [
        {"name": constraint["name"], "column_names": constraint["column_names"]} for constraint in unique
    ]
    # SQLite assigned seq 1 to the forgotten id and 2 to the next one: AUTOINCREMENT never reuses it.
    assert rows == [(2, second.id)]
    assert vectors == [2]


def test_a_duplicate_kind_and_id_is_rejected_by_its_unique_key(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    _ = store.enroll("voice", "tts:warm", _angle(0), True, _POLICY, at=_START)
    values = {
        "kind": "voice",
        "id": "tts:warm",
        "model": _MODEL,
        "own": 0,
        "linked": 0,
        "created_at": "2026-10-15T12:00:00.000000+00:00",
        "seen_at": "2026-10-15T12:00:00.000000+00:00",
        "seen_count": 1,
    }

    with pytest.raises(sqlalchemy.exc.IntegrityError, match="UNIQUE"):
        with engine.begin() as connection:
            _ = connection.execute(sqlalchemy.insert(memory.schema.identity_table).values(**values))
    _ = store.enroll("face", "tts:warm", _angle(0), False, _POLICY, at=_START)
    engine.dispose()


def test_one_vec0_table_per_model_and_dims(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)

    _ = store.identify("voice", (_angle(0), _angle(0, model="other")), _POLICY, at=_START, mint=(True, True))
    _ = store.identify("face", (_angle(90),), _POLICY, at=_START, mint=(True,))
    _ = store.enroll("voice", "tts:warm", _print(1.0, 0.0, 0.0, model="wide"), True, _POLICY, at=_START)
    with engine.connect() as connection:
        tables = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING vec0%'")
            ).scalars()
        )
    engine.dispose()

    assert tables == {_vec_table(_MODEL, 2), _vec_table("other", 2), _vec_table("wide", 3)}


def test_refine_and_match_update_the_row_and_the_vector_under_one_seq(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    later = _START + datetime.timedelta(hours=1)

    (minted,) = _minted(store, "voice", _angle(0), at=_START)
    (refined,) = _minted(store, "voice", _angle(20), at=later)
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT seq, model, own, linked, seen_count, created_at, seen_at FROM bot_identity")
        ).one()
        blob = connection.execute(
            text(f"SELECT embedding FROM {_vec_table(_MODEL, 2)} WHERE seq = :seq"), {"seq": row[0]}
        ).scalar_one()
    engine.dispose()
    stored = memory.index.unpack(typing.cast(bytes, blob))

    assert refined.id == minted.id
    assert tuple(row[1:5]) == (_MODEL, 0, 0, 2)
    assert row[5] == "2026-10-15T12:00:00.000000+00:00"
    assert row[6] == "2026-10-15T13:00:00.000000+00:00"
    # The running mean of 0° and 20° is 10°.
    assert stored == pytest.approx((math.cos(math.radians(10)), math.sin(math.radians(10))), abs=_CLOSE)


def test_a_failure_sqlite_already_rolled_back_surfaces_the_original_error(tmp_path: pathlib.Path) -> None:
    """A full disk can end the transaction inside SQLite; the store must not mask it with its own ROLLBACK."""

    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    with engine.begin() as connection:
        # RAISE(ROLLBACK) ends the whole transaction inside SQLite, as SQLITE_FULL may.
        _ = connection.exec_driver_sql(
            "CREATE TRIGGER disk_full BEFORE INSERT ON bot_identity "
            "BEGIN SELECT RAISE(ROLLBACK, 'database or disk is full'); END"
        )

    with pytest.raises(identification.StoreError) as raised:
        _ = store.identify("voice", (_angle(0),), _POLICY, at=_START, mint=(True,))
    with engine.begin() as connection:
        _ = connection.exec_driver_sql("DROP TRIGGER disk_full")
    (after,) = _minted(store, "voice", _angle(0), at=_START)
    engine.dispose()

    assert raised.value.kind is identification.FailureKind.STORE_FAILED
    # The trigger's error is an IntegrityError; a masking ROLLBACK would surface an OperationalError.
    assert str(raised.value) == "the identity store failed (IntegrityError)"
    assert after.recognized is False


def test_a_failed_statement_never_carries_its_bound_print_into_the_store_error(tmp_path: pathlib.Path) -> None:
    """A database error renders its bound parameters; a packed print there would leak a biometric."""

    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    voice = _print(0.123456789, 0.987654321)
    table = _vec_table(_MODEL, 2)
    with engine.begin() as connection:
        # A plain table under the print space's name: the store keeps it, and its MATCH query,
        # bound to the packed print, fails on the row.
        _ = connection.exec_driver_sql(f"CREATE TABLE {table} (seq, embedding, kind, distance, k)")
        _ = connection.exec_driver_sql(f"INSERT INTO {table} VALUES (1, x'00', 'voice', 0.0, 2)")

    with pytest.raises(identification.StoreError) as raised:
        _ = store.identify("voice", (voice,), _POLICY, at=_START, mint=(True,))
    engine.dispose()
    chain: list[BaseException] = []
    error: BaseException | None = raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    rendered = "\n".join(f"{error!r} {error}" for error in chain)

    assert raised.value.kind is identification.FailureKind.STORE_FAILED
    assert repr(memory.index.pack(voice.vector)) not in rendered
    assert "parameters" not in rendered
    assert chain == [raised.value]


def test_a_store_on_one_shared_in_memory_connection_is_refused() -> None:
    engine = sqlite_memory.open_engine()

    with pytest.raises(ValueError, match="needs a file database"):
        _ = sqlite_memory.IdentityStore(engine=engine)
    engine.dispose()


def test_migrations_leave_both_vec0_prefixes_to_the_providers(tmp_path: pathlib.Path) -> None:
    """``command.check`` reports no drift with memory and identity vec0 tables (and their shadows) present."""

    from alembic import command

    engine = _engine(tmp_path)
    _ = sqlite_memory.IdentityStore(engine=engine).identify("voice", (_angle(0),), _POLICY, at=_START, mint=(True,))
    with engine.begin() as connection:
        sqlite_memory.VecIndex().ensure(connection, _SPACE)
        tables = set(connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'")).scalars())
        config = memory.store.migration_config()
        config.attributes["connection"] = connection
        command.check(config)
    engine.dispose()

    assert "bot_identity" in tables
    assert any(name.startswith("bot_identity_vec_") and name != _vec_table(_MODEL, 2) for name in tables)


def _episode(words: str, *voices: str) -> memory.EpisodeData:
    return memory.EpisodeData(
        turn_id=words,
        parts=(language.TextPart(text=words),),
        outcome=memory.Outcome.COMPLETED,
        participants=tuple(identification.participant("voice", voice) for voice in voices),
    )


def test_forget_removes_only_the_identity_row_and_its_print(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    (voice,) = _minted(store, "voice", _angle(0), at=_START)
    ref = identification.ref("voice", voice.id)
    kept = memory.graph.write_episode(engine, memory.ExactIndex(), _episode("hello", voice.id), at=_START)

    assert store.forget("voice", voice.id) is True
    with engine.connect() as connection:
        nodes = connection.execute(text("SELECT seq, kind FROM bot_memory_node ORDER BY seq")).all()
        refs = connection.execute(text("SELECT ref FROM bot_person_ref")).scalars().all()
        identities = connection.execute(text("SELECT count(*) FROM bot_identity")).scalar_one()
        prints = connection.execute(text(f"SELECT count(*) FROM {_vec_table(_MODEL, 2)}")).scalar_one()
    engine.dispose()

    assert [kind for _, kind in nodes] == ["episode", "person"]
    assert refs == [ref]
    assert nodes[0][0] == kept.seq
    assert (identities, prints) == (0, 0)


def test_memory_forget_and_merge_leave_identities_intact(tmp_path: pathlib.Path) -> None:
    engine = _engine(tmp_path)
    store = sqlite_memory.IdentityStore(engine=engine)
    vec = sqlite_memory.VecIndex()
    _ = store.enroll("voice", "tts:warm", _angle(0), True, _POLICY, at=_START)
    (heard,) = _minted(store, "voice", _angle(90), at=_START)
    seqs = [memory.graph.write_episode(engine, vec, _episode(words), at=_START).seq for words in ("a", "b", "c")]
    memory.graph.write_vectors(
        engine, vec, _SPACE, [(seq, _angle(index * 10).vector) for index, seq in enumerate(seqs)]
    )
    unit, lease = memory.graph.lease(engine, vec, memory.LeaseData(episodes=3, neighbors=0), space=_SPACE)
    assert unit is not None and lease is not None
    refs = [node.ref for node in unit.nodes]
    decision = memory.ConsolidateData(
        unit=unit.unit,
        merges=(memory.MergeData(sources=(refs[0], refs[1]), text="a and b", forget_sources=True),),
        forgets=(refs[2],),
    )

    outcome, _ = memory.graph.commit(
        engine, vec, decision, lease, space=_SPACE, merged_vectors=[_angle(5).vector], at=_START
    )
    (again,) = _minted(store, "voice", _angle(0), at=_START)
    (other,) = _minted(store, "voice", _angle(90), at=_START)
    with engine.connect() as connection:
        identities = connection.execute(text("SELECT count(*) FROM bot_identity")).scalar_one()
        prints = connection.execute(text(f"SELECT count(*) FROM {_vec_table(_MODEL, 2)}")).scalar_one()
    engine.dispose()

    assert (outcome.merged, outcome.forgotten) == (1, 3)
    assert (identities, prints) == (2, 2)
    assert (again.id, again.own, again.recognized) == ("tts:warm", True, True)
    assert other.id == heard.id


class _Owner(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = define(
        "IdentityStoreTestOwner", hsm.initial(hsm.target("idle")), hsm.state("idle")
    )


class _Embedder(embedding.Embedder):
    def __init__(self) -> None:
        self.capacity = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)

    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx
        vectors = tuple(embedding.Vector(values=(1.0, 0.0)) for _ in input.items)
        return embedding.OutputData(space=_SPACE, role=input.role, vectors=vectors)


def test_a_store_on_a_database_opens_it_on_its_first_call(tmp_path: pathlib.Path) -> None:
    """D-W8-20: constructing the store opens nothing; its first call opens (and migrates) the database."""

    path = tmp_path / "memory.sqlite3"
    database = sqlite_memory.Database(path=path)
    store = sqlite_memory.IdentityStore(engine=database.open)
    constructed = path.exists()

    fixture = _fixture(lambda: store)
    (minted,) = fixture.identify(_angle(0))

    assert constructed is False
    assert minted.recognized is False
    assert path.exists()
    database.open().dispose()


def test_a_database_that_cannot_open_is_a_failed_store_call_naming_only_its_type(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unversioned.sqlite3"
    engine = sqlalchemy.create_engine(f"sqlite+pysqlite:///{path}")
    with engine.begin() as connection:
        _ = connection.execute(text(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)"))
    engine.dispose()
    fixture = _fixture(lambda: sqlite_memory.IdentityStore(engine=sqlite_memory.Database(path=path).open))

    with pytest.raises(identification.StoreError) as raised:
        _ = fixture.identify(_angle(0))

    assert raised.value.kind is identification.FailureKind.STORE_FAILED
    assert "MigrationError" in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize("shared", ["engine", "database"])
def test_memory_and_two_registries_share_one_file_database(tmp_path: pathlib.Path, shared: str) -> None:
    """One Memory write and a voice and a face registry in flight on one shared store all complete.

    Shared as an open engine, or as a ``Database`` handle that whichever starts first opens (D-W8-20).
    """

    database = sqlite_memory.Database(path=tmp_path / "shared.sqlite3")
    handle: Engine | collections.abc.Callable[[], Engine] = _engine(tmp_path) if shared == "engine" else database.open
    store = sqlite_memory.IdentityStore(engine=handle)
    timeout = datetime.timedelta(seconds=10)

    async def run() -> list[hsm.Event[typing.Any]]:
        ctx = Environment()
        owner = _Owner()
        _ = await started(ctx, owner, _Owner.model)
        actors: list[ability.Ability[typing.Any, typing.Any]] = [
            identification.Registry(
                identifier=identification.Matcher(store=store, policy=_POLICY),
                kind=kind,
                max_unlinked_age=datetime.timedelta(days=30),
                max_ids=100,
            )
            for kind in ("voice", "face")
        ]
        actors.append(memory.Memory(engine=handle, index=sqlite_memory.VecIndex(), embedder=_Embedder()))
        for actor in actors:
            assert await actor.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        voice, face, remembering = actors

        def identify(registry: hsm.Instance, degrees: float) -> collections.abc.Awaitable[hsm.Event[typing.Any]]:
            request = identification.IdentifyEvent.with_data_and_id(
                identification.IdentifyData(prints=(_angle(degrees), _angle(degrees + 90))), uuid.uuid4().hex
            )
            return ability.run_terminal_operation(
                ctx,
                child=registry,
                request=request,
                terminals=(identification.IdentifiedEvent, identification.FailedEvent),
                timeout=timeout,
            )

        remember = ability.run_terminal_operation(
            ctx,
            child=remembering,
            request=memory.RememberEvent.with_data_and_id(_episode("lunch at noon"), uuid.uuid4().hex),
            terminals=(memory.RememberedEvent, memory.FailedEvent),
            timeout=timeout,
        )
        return list(
            await asyncio.gather(
                remember, *(identify(registry, step * 3.0) for step in range(4) for registry in (voice, face))
            )
        )

    replies = asyncio.run(run())
    engine = handle if isinstance(handle, Engine) else handle()
    with engine.connect() as connection:
        kinds = connection.execute(text("SELECT kind, count(*) FROM bot_identity GROUP BY kind ORDER BY kind")).all()
    engine.dispose()

    assert [reply.name for reply in replies] == [
        memory.RememberedEvent.name,
        *([identification.IdentifiedEvent.name] * 8),
    ]
    assert kinds == [("face", 2), ("voice", 2)]
