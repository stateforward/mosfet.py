"""The episodic graph's SQL: Core statement builders, row decoding, and the transactions Memory runs.

Builders (``*_clause`` functions) are pure: they return parameterized SQLAlchemy Core statements
whose identifiers come only from ``schema``, never from caller text. Decoders turn rows back into
``payload`` read models. The transaction functions (``write_episode``, ``search``, ``lease``, ...)
perform database I/O: each opens exactly one transaction on the engine it is given and must run
on Memory's database worker thread, never on the event loop.

Times are stored as UTC ISO text at microsecond resolution (fixed width), and every episodic
order is ``(created_at, seq)``: ``seq`` is the insertion order, so same-instant rows read in the
order they were written.

Persons: a person node is keyed by its person id, and ``bot_person_ref`` says whose each
participant ref is (the ref is its primary key, so it is exactly one person's). Memory resolves a
ref to its person inside its own transaction, from that table: a ref first seen gets a provisional
person (no name, not declared), and a ref stamped ``own`` is ``self``'s. A provisional person that
holds only the ref being linked is merged into the person it is linked to: its involves links are
re-pointed and its node deleted. Any other person keeps its refs until they are unlinked.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import json
import typing
import uuid

import mosfet
import sqlalchemy
from sqlalchemy import Delete, Insert, Select, Update, and_, delete, func, or_, select, update
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Connection, Engine

from mosfet import media

from ..embedding import payload as embedding_payload
from ..language import text
from . import index, payload, schema

_NODES = schema.node_table
_PARTS = schema.part_table
_EDGES = schema.edge_table
_VECTORS = schema.vector_table
_ATTEMPTS = schema.index_attempt_table
_PERSON_REFS = schema.person_ref_table

INVOLVES_SOURCE: typing.Final = "involves_source"
"""Relation from an episode to a person who produced its stimulus."""
INVOLVES_TARGET: typing.Final = "involves_target"
"""Relation from an episode to a person its stimulus was addressed to."""
DERIVED_FROM: typing.Final = "derived_from"
"""Relation from a merged concept to each node it was merged from."""
_INVOLVES: typing.Final = (INVOLVES_SOURCE, INVOLVES_TARGET)

RefOrigin: typing.TypeAlias = typing.Literal["stamped", "own", "model", "owner"]
"""How a ref came to be a person's: first seen, stamped as the bot's own, linked by the bot, or declared."""

PERMANENT_FAILURES: typing.Final = frozenset(
    {
        embedding_payload.FailureKind.TOO_LONG,
        embedding_payload.FailureKind.UNSUPPORTED_INPUT,
        embedding_payload.FailureKind.INVALID_INPUT,
    }
)
"""Embedding failures that repeat for the same item in the same space; such a node is never retried there."""


def utc_text(moment: datetime.datetime) -> str:
    """Fixed-width UTC ISO text at microsecond resolution, so text order is chronological order."""

    return moment.astimezone(datetime.UTC).isoformat(timespec="microseconds")


def parse_utc(value: object) -> datetime.datetime:
    """Read a stored UTC timestamp."""

    assert isinstance(value, str), value
    return datetime.datetime.fromisoformat(value).astimezone(datetime.UTC)


# --- builders (pure) ---------------------------------------------------------------------------


def node_insert_clause(
    *, node_id: str, kind: payload.NodeKind, text: str, content: dict[str, object], at: str, key: str | None = None
) -> Insert:
    """Insert one node; its ``seq`` is assigned by the database."""

    return sqlalchemy.insert(_NODES).values(
        node_id=node_id,
        kind=kind.value,
        key=key,
        text=text,
        content_json=json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        created_at=at,
        updated_at=at,
    )


def _person_content(
    *, name: str | None, role: payload.Role | None, born: int | None, declared: bool
) -> dict[str, object]:
    return {"name": name, "role": None if role is None else role.value, "born": born, "declared": declared}


def person_insert_clause(
    *,
    person_id: str,
    at: str,
    name: str | None = None,
    role: payload.Role | None = None,
    born: int | None = None,
    declared: bool = False,
) -> Insert:
    """Create the person node ``person_id`` unless it exists; its text is its name (empty while unnamed)."""

    content = _person_content(name=name, role=role, born=born, declared=declared)
    return (
        sqlite.insert(_NODES)
        .values(
            node_id=uuid.uuid4().hex,
            kind=payload.NodeKind.PERSON.value,
            key=person_id,
            text=name or "",
            content_json=json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            created_at=at,
            updated_at=at,
        )
        .on_conflict_do_nothing(index_elements=[_NODES.c.kind, _NODES.c.key])
    )


def person_ref_upsert_clause(*, ref: str, person: int, origin: RefOrigin, at: str) -> Insert:
    """Make ``ref`` the person ``person``'s: the one statement that moves a ref, used only after its checks."""

    statement = sqlite.insert(_PERSON_REFS).values(ref=ref, person_seq=person, origin=origin, linked_at=at)
    return statement.on_conflict_do_update(
        index_elements=[_PERSON_REFS.c.ref],
        set_={
            "person_seq": statement.excluded.person_seq,
            "origin": statement.excluded.origin,
            "linked_at": statement.excluded.linked_at,
        },
    )


def persons_select_clause(refs: collections.abc.Collection[str]) -> Select[tuple[int, str]]:
    """The person of each of ``refs`` that has one, as ``(person seq, ref)``."""

    return select(_PERSON_REFS.c.person_seq, _PERSON_REFS.c.ref).where(_PERSON_REFS.c.ref.in_(list(refs)))


def edge_upsert_clause(*, source: int, target: int, relation: str, weight: float, origin: str, at: str) -> Insert:
    """Write one link; an existing link with the same relation takes the new weight."""

    statement = sqlite.insert(_EDGES).values(
        source_seq=source, target_seq=target, relation=relation, weight=weight, created_at=at, origin=origin
    )
    return statement.on_conflict_do_update(
        index_elements=[_EDGES.c.source_seq, _EDGES.c.target_seq, _EDGES.c.relation],
        set_={"weight": statement.excluded.weight, "origin": statement.excluded.origin},
    )


def vector_upsert_clause(
    *, seq: int, space: embedding_payload.Space, values: collections.abc.Sequence[float]
) -> Insert:
    """Write the canonical vector of ``seq`` in ``space``."""

    statement = sqlite.insert(_VECTORS).values(
        node_seq=seq, space_model=space.model, dimensions=space.dimensions, embedding=index.pack(values)
    )
    return statement.on_conflict_do_update(
        index_elements=[_VECTORS.c.node_seq, _VECTORS.c.space_model, _VECTORS.c.dimensions],
        set_={"embedding": statement.excluded.embedding},
    )


def attempt_upsert_clause(
    *,
    seq: int,
    space: embedding_payload.Space,
    kind: embedding_payload.FailureKind,
    pass_id: str | None,
    at: str,
) -> Insert:
    """Record the last failed attempt to index ``seq`` in ``space``."""

    statement = sqlite.insert(_ATTEMPTS).values(
        node_seq=seq,
        space_model=space.model,
        dimensions=space.dimensions,
        failure_kind=kind.value,
        permanent=int(kind in PERMANENT_FAILURES),
        pass_id=pass_id,
        attempted_at=at,
    )
    return statement.on_conflict_do_update(
        index_elements=[_ATTEMPTS.c.node_seq, _ATTEMPTS.c.space_model, _ATTEMPTS.c.dimensions],
        set_={
            "failure_kind": statement.excluded.failure_kind,
            "permanent": statement.excluded.permanent,
            "pass_id": statement.excluded.pass_id,
            "attempted_at": statement.excluded.attempted_at,
        },
    )


def attempt_clear_clause(*, seq: int, space: embedding_payload.Space) -> Delete:
    """Forget a node's failed attempt in ``space`` once it is indexed there."""

    return delete(_ATTEMPTS).where(
        _ATTEMPTS.c.node_seq == seq,
        _ATTEMPTS.c.space_model == space.model,
        _ATTEMPTS.c.dimensions == space.dimensions,
    )


def recent_clause(*, kinds: collections.abc.Collection[payload.NodeKind], limit: int) -> Select[tuple[int]]:
    """The ``limit`` latest nodes of ``kinds``, newest first."""

    return (
        select(_NODES.c.seq)
        .where(_NODES.c.kind.in_([kind.value for kind in kinds]))
        .order_by(_NODES.c.created_at.desc(), _NODES.c.seq.desc())
        .limit(limit)
    )


def backfill_clause(*, space: embedding_payload.Space, pass_id: str, limit: int) -> Select[tuple[int, int | None]]:
    """Nodes that may be indexed in ``space`` in this pass: never-tried first, then oldest first.

    Eligible: a concept, a named person (recallable by name), or an episode with at least one part
    (an episode with nothing said has nothing to embed), with no vector in ``space``, no permanent attempt there, and no attempt in
    this pass. The second column is the seq of a prior attempt row (None when the node was never
    tried).
    """

    vector = and_(
        _VECTORS.c.node_seq == _NODES.c.seq,
        _VECTORS.c.space_model == space.model,
        _VECTORS.c.dimensions == space.dimensions,
    )
    attempt = and_(
        _ATTEMPTS.c.node_seq == _NODES.c.seq,
        _ATTEMPTS.c.space_model == space.model,
        _ATTEMPTS.c.dimensions == space.dimensions,
    )
    return (
        select(_NODES.c.seq, _ATTEMPTS.c.node_seq)
        .select_from(_NODES.outerjoin(_VECTORS, vector).outerjoin(_ATTEMPTS, attempt))
        .where(
            or_(
                _NODES.c.kind == payload.NodeKind.CONCEPT.value,
                and_(_NODES.c.kind == payload.NodeKind.PERSON.value, _NODES.c.text != ""),
                and_(
                    _NODES.c.kind == payload.NodeKind.EPISODE.value,
                    select(_PARTS.c.node_seq).where(_PARTS.c.node_seq == _NODES.c.seq).exists(),
                ),
            ),
            _VECTORS.c.node_seq.is_(None),
            or_(
                _ATTEMPTS.c.node_seq.is_(None),
                and_(
                    _ATTEMPTS.c.permanent == 0,
                    or_(_ATTEMPTS.c.pass_id.is_(None), _ATTEMPTS.c.pass_id != pass_id),
                ),
            ),
        )
        .order_by(_ATTEMPTS.c.node_seq.is_not(None), _NODES.c.created_at, _NODES.c.seq)
        .limit(limit)
    )


def unreviewed_clause(*, limit: int) -> Select[tuple[int]]:
    """The ``limit`` oldest unreviewed episodes."""

    return (
        select(_NODES.c.seq)
        .where(_NODES.c.kind == payload.NodeKind.EPISODE.value, _NODES.c.reviewed_at.is_(None))
        .order_by(_NODES.c.created_at, _NODES.c.seq)
        .limit(limit)
    )


def reviewed_clause(seqs: collections.abc.Collection[int], *, at: str) -> Update:
    """Mark episodes reviewed by consolidation."""

    return (
        update(_NODES)
        .where(_NODES.c.seq.in_(list(seqs)), _NODES.c.reviewed_at.is_(None))
        .values(reviewed_at=at, updated_at=at)
    )


def forget_clauses(seqs: collections.abc.Collection[int]) -> tuple[Delete, ...]:
    """Delete nodes and every row that depends on them (no cascade is relied on)."""

    listed = list(seqs)
    return (
        delete(_EDGES).where(or_(_EDGES.c.source_seq.in_(listed), _EDGES.c.target_seq.in_(listed))),
        delete(_PARTS).where(_PARTS.c.node_seq.in_(listed)),
        delete(_VECTORS).where(_VECTORS.c.node_seq.in_(listed)),
        delete(_ATTEMPTS).where(_ATTEMPTS.c.node_seq.in_(listed)),
        delete(_NODES).where(_NODES.c.seq.in_(listed)),
    )


# --- decoding (pure) ---------------------------------------------------------------------------


def _participants(content: dict[str, typing.Any]) -> tuple[mosfet.Participant, ...]:
    return tuple(mosfet.Participant.model_validate(item) for item in content.get("participants", ()))


def _decode_parts(
    rows: collections.abc.Sequence[sqlalchemy.Row[typing.Any]], *, media_bytes: bool
) -> tuple[str, tuple[media.Image | media.ReleasedImage, ...], tuple[text.ContentPart, ...]]:
    texts: list[str] = []
    images: list[media.Image | media.ReleasedImage] = []
    parts: list[text.ContentPart] = []
    for row in rows:
        if row.kind == "text":
            texts.append(row.text or "")
            parts.append(text.TextPart(text=row.text or ""))
            continue
        if media_bytes:
            image = media.Image(
                content=row.content, media_type=row.media_type, width_px=row.width_px, height_px=row.height_px
            )
            images.append(image)
            parts.append(text.ImagePart(image=image))
        else:
            images.append(
                media.ReleasedImage(
                    media_type=row.media_type,
                    width_px=row.width_px,
                    height_px=row.height_px,
                    byte_count=row.byte_count,
                    reason="memory",
                )
            )
    return "\n".join(texts), tuple(images), tuple(parts)


@dataclasses.dataclass(frozen=True)
class _Loaded:
    """One node read back with its parts (``parts`` holds image bytes only when media was read)."""

    seq: int
    node: payload.Node
    parts: tuple[text.ContentPart, ...]


def _load(
    connection: Connection,
    seqs: collections.abc.Sequence[int],
    *,
    space: embedding_payload.Space | None,
    media_bytes: bool,
) -> dict[int, _Loaded]:
    """Read nodes and their parts by seq."""

    if not seqs:
        return {}
    rows = connection.execute(select(_NODES).where(_NODES.c.seq.in_(list(seqs)))).all()
    columns = [
        _PARTS.c.node_seq,
        _PARTS.c.ordinal,
        _PARTS.c.kind,
        _PARTS.c.text,
        _PARTS.c.media_type,
        _PARTS.c.width_px,
        _PARTS.c.height_px,
        func.length(_PARTS.c.content).label("byte_count"),
    ]
    if media_bytes:
        columns.append(_PARTS.c.content)
    part_rows = connection.execute(
        select(*columns).where(_PARTS.c.node_seq.in_(list(seqs))).order_by(_PARTS.c.node_seq, _PARTS.c.ordinal)
    ).all()
    by_node: dict[int, list[sqlalchemy.Row[typing.Any]]] = collections.defaultdict(list)
    for part in part_rows:
        by_node[int(part.node_seq)].append(part)
    indexed: set[int] = set()
    if space is not None:
        indexed = {
            int(seq)
            for seq in connection.execute(
                select(_VECTORS.c.node_seq).where(
                    _VECTORS.c.node_seq.in_(list(seqs)),
                    _VECTORS.c.space_model == space.model,
                    _VECTORS.c.dimensions == space.dimensions,
                )
            ).scalars()
        }
    loaded: dict[int, _Loaded] = {}
    for row in rows:
        seq = int(row.seq)
        joined, images, parts = _decode_parts(by_node.get(seq, []), media_bytes=media_bytes)
        kind = payload.NodeKind(row.kind)
        created_at = parse_utc(row.created_at)
        episode: payload.Episode | None = None
        if kind is payload.NodeKind.EPISODE:
            content = typing.cast(dict[str, typing.Any], json.loads(row.content_json))
            episode = payload.Episode(
                node_id=row.node_id,
                stimulus=content.get("stimulus"),
                text=joined,
                images=images,
                participants=_participants(content),
                focus=content.get("focus"),
                focus_candidates=tuple(content.get("focus_candidates", ())),
                acts=tuple(payload.Act.model_validate(act) for act in content.get("acts", ())),
                outcome=payload.Outcome(content["outcome"]),
                turn_id=content["turn_id"],
                created_at=created_at,
                recalled_at=None if row.recalled_at is None else parse_utc(row.recalled_at),
                recall_count=int(row.recall_count),
                reviewed=row.reviewed_at is not None,
                indexed=seq in indexed,
            )
        node = payload.Node(
            node_id=row.node_id,
            kind=kind,
            text=joined if kind is payload.NodeKind.EPISODE else row.text,
            created_at=created_at,
            episode=episode,
            key=row.key,
        )
        loaded[seq] = _Loaded(seq=seq, node=node, parts=parts)
    return loaded


# --- transactions (database I/O; run on Memory's worker) ---------------------------------------


# --- persons (run inside a transaction) ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _PersonRow:
    """One person node as stored."""

    seq: int
    id: str
    name: str | None
    role: payload.Role | None
    born: int | None
    declared: bool


def _person_rows(connection: Connection, criterion: sqlalchemy.ColumnElement[bool]) -> list[_PersonRow]:
    rows = connection.execute(
        select(_NODES.c.seq, _NODES.c.key, _NODES.c.content_json)
        .where(_NODES.c.kind == payload.NodeKind.PERSON.value, criterion)
        .order_by(_NODES.c.seq)
    ).all()
    found: list[_PersonRow] = []
    for seq, key, content_json in rows:
        content = typing.cast(dict[str, typing.Any], json.loads(content_json))
        role = content.get("role")
        found.append(
            _PersonRow(
                seq=int(seq),
                id=str(key),
                name=content.get("name"),
                role=None if role is None else payload.Role(role),
                born=content.get("born"),
                declared=bool(content.get("declared", False)),
            )
        )
    return found


def _person_by_id(connection: Connection, person_id: str) -> _PersonRow | None:
    return next(iter(_person_rows(connection, _NODES.c.key == person_id)), None)


def _person_by_seq(connection: Connection, seq: int) -> _PersonRow:
    (row,) = _person_rows(connection, _NODES.c.seq == seq)
    return row


def _provisional(row: _PersonRow) -> bool:
    """Never named, never declared, and not the bot itself: a person memory made for a ref it saw."""

    return row.id != payload.SELF and row.name is None and not row.declared


def _ref_count(connection: Connection, seq: int) -> int:
    return int(
        connection.execute(
            select(func.count()).select_from(_PERSON_REFS).where(_PERSON_REFS.c.person_seq == seq)
        ).scalar_one()
    )


def _person(connection: Connection, row: _PersonRow) -> payload.Person:
    """A person as read back, with its newest refs first."""

    refs = connection.execute(
        select(_PERSON_REFS.c.ref)
        .where(_PERSON_REFS.c.person_seq == row.seq)
        .order_by(_PERSON_REFS.c.linked_at.desc(), _PERSON_REFS.c.ref)
        .limit(payload.MAX_PERSON_REFS)
    ).scalars()
    return payload.Person(
        id=row.id, name=row.name, role=row.role, born=row.born, declared=row.declared, refs=tuple(refs)
    )


def _self_seq(connection: Connection, stamp: str) -> tuple[int, int]:
    """The seq of ``self``, creating it when missing; and 1 when it was created now, else 0."""

    created = connection.execute(person_insert_clause(person_id=payload.SELF, at=stamp)).rowcount
    row = _person_by_id(connection, payload.SELF)
    assert row is not None
    return row.seq, int(created == 1)


def _mint(connection: Connection, stamp: str, *, name: str | None = None) -> int:
    """A new person with a minted id (named ``name``, or not yet); its seq."""

    person_id = uuid.uuid4().hex
    _ = connection.execute(person_insert_clause(person_id=person_id, at=stamp, name=name))
    row = _person_by_id(connection, person_id)
    assert row is not None
    return row.seq


def _merge(connection: Connection, source: int, into: int) -> None:
    """Fold the provisional person ``source`` into ``into``: re-point its links, then delete its node."""

    for column in ("target_seq", "source_seq"):
        _ = connection.execute(
            update(_EDGES).where(_EDGES.c[column] == source).values({column: into}).prefix_with("OR IGNORE")
        )
    for clause in forget_clauses([source]):
        _ = connection.execute(clause)


def _linked_persons(connection: Connection, refs: collections.abc.Collection[str]) -> dict[str, int]:
    if not refs:
        return {}
    return {str(ref): int(seq) for seq, ref in connection.execute(persons_select_clause(refs)).all()}


def _resolve(
    connection: Connection, participants: collections.abc.Sequence[mosfet.Participant], stamp: str
) -> tuple[dict[str, int], int, int]:
    """Each participant ref's person: ``(person seq by ref, persons created minus merged, own conflicts)``.

    A ref first seen gets a provisional person. A ref stamped own is ``self``'s; when it was a
    provisional person's only ref, that person is merged into ``self``. An own ref that belongs
    to any other person stays theirs and counts one conflict (only a corrupt store links an own id).
    """

    refs = list(dict.fromkeys(participant.ref for participant in participants))
    own = {participant.ref for participant in participants if participant.own}
    linked = _linked_persons(connection, refs)
    persons: dict[str, int] = {}
    created = 0
    conflicts = 0
    self_seq: int | None = None
    if own:
        self_seq, created = _self_seq(connection, stamp)
    for ref in refs:
        current = linked.get(ref)
        if ref in own:
            assert self_seq is not None
            if current is None or current == self_seq:
                persons[ref] = self_seq
                if current is None:
                    _ = connection.execute(person_ref_upsert_clause(ref=ref, person=self_seq, origin="own", at=stamp))
                continue
            if _provisional(_person_by_seq(connection, current)) and _ref_count(connection, current) == 1:
                _merge(connection, current, self_seq)
                _ = connection.execute(person_ref_upsert_clause(ref=ref, person=self_seq, origin="own", at=stamp))
                persons[ref] = self_seq
                created -= 1
                continue
            conflicts += 1
            persons[ref] = current
        elif current is None:
            persons[ref] = _mint(connection, stamp)
            _ = connection.execute(person_ref_upsert_clause(ref=ref, person=persons[ref], origin="stamped", at=stamp))
            created += 1
        else:
            persons[ref] = current
    return persons, created, conflicts


def ensure_self(engine: Engine, *, at: datetime.datetime) -> bool:
    """Create the ``self`` person unless it exists; whether it was created now. One transaction."""

    with engine.begin() as connection:
        _, created = _self_seq(connection, utc_text(at))
    return created == 1


@dataclasses.dataclass(frozen=True)
class Kept:
    """What keeping one episode wrote: its node, the person-count change, and own-ref conflicts."""

    seq: int
    node_id: str
    persons: int
    conflicts: int


def write_episode(engine: Engine, data: payload.EpisodeData, *, at: datetime.datetime) -> Kept:
    """Keep one episode: its node, its parts, each participant's person, and involves links. One transaction."""

    stamp = utc_text(at)
    node_id = uuid.uuid4().hex
    content: dict[str, object] = {
        "turn_id": data.turn_id,
        "stimulus": data.stimulus,
        "focus": data.focus,
        "focus_candidates": list(data.focus_candidates),
        "acts": [act.model_dump(mode="json") for act in data.acts],
        "outcome": data.outcome.value,
        "participants": [participant.model_dump(mode="json") for participant in data.participants],
    }
    joined = "\n".join(part.text for part in data.parts if isinstance(part, text.TextPart))
    with engine.begin() as connection:
        result = connection.execute(
            node_insert_clause(node_id=node_id, kind=payload.NodeKind.EPISODE, text=joined, content=content, at=stamp)
        )
        seq = int(typing.cast(tuple[int, ...], result.inserted_primary_key)[0])
        for ordinal, part in enumerate(data.parts):
            if isinstance(part, text.TextPart):
                values: dict[str, object] = {"kind": "text", "text": part.text}
            else:
                image = part.image
                values = {
                    "kind": "image",
                    "media_type": image.media_type,
                    "width_px": image.width_px,
                    "height_px": image.height_px,
                    "content": image.content,
                }
            _ = connection.execute(sqlalchemy.insert(_PARTS).values(node_seq=seq, ordinal=ordinal, **values))
        persons, created, conflicts = _resolve(connection, data.participants, stamp)
        for participant in data.participants:
            relation = INVOLVES_SOURCE if participant.role is mosfet.ParticipantRole.SOURCE else INVOLVES_TARGET
            _ = connection.execute(
                edge_upsert_clause(
                    source=seq,
                    target=persons[participant.ref],
                    relation=relation,
                    weight=1.0,
                    origin="stamped",
                    at=stamp,
                )
            )
    return Kept(seq=seq, node_id=node_id, persons=created, conflicts=conflicts)


def write_vectors(
    engine: Engine,
    idx: index.Index,
    space: embedding_payload.Space,
    vectors: collections.abc.Sequence[tuple[int, collections.abc.Sequence[float]]],
) -> None:
    """Write canonical vectors, index them, and clear their failed attempts. One transaction."""

    if not vectors:
        return
    with engine.begin() as connection:
        idx.ensure(connection, space)
        for seq, values in vectors:
            _ = connection.execute(vector_upsert_clause(seq=seq, space=space, values=values))
            idx.add(connection, space, seq, values)
            _ = connection.execute(attempt_clear_clause(seq=seq, space=space))


def record_attempts(
    engine: Engine,
    space: embedding_payload.Space,
    attempts: collections.abc.Sequence[tuple[int, embedding_payload.FailureKind]],
    *,
    pass_id: str | None,
    at: datetime.datetime,
) -> None:
    """Record failed indexing attempts. One transaction."""

    if not attempts:
        return
    stamp = utc_text(at)
    with engine.begin() as connection:
        for seq, kind in attempts:
            _ = connection.execute(attempt_upsert_clause(seq=seq, space=space, kind=kind, pass_id=pass_id, at=stamp))


def node_parts(engine: Engine, seqs: collections.abc.Sequence[int]) -> dict[int, tuple[text.ContentPart, ...]]:
    """The parts of nodes, with image bytes, for embedding; a concept or a person is its text."""

    with engine.begin() as connection:
        loaded = _load(connection, seqs, space=None, media_bytes=True)
        concepts = {
            seq: (text.TextPart(text=item.node.text),)
            for seq, item in loaded.items()
            if item.node.kind in (payload.NodeKind.CONCEPT, payload.NodeKind.PERSON)
        }
    return {seq: concepts.get(seq, item.parts) for seq, item in loaded.items()}


def backfill_candidates(
    engine: Engine, space: embedding_payload.Space, *, pass_id: str, limit: int
) -> tuple[tuple[int, bool], ...]:
    """Eligible nodes for one backfill batch, each with whether it was tried before."""

    with engine.begin() as connection:
        rows = connection.execute(backfill_clause(space=space, pass_id=pass_id, limit=limit)).all()
    return tuple((int(seq), attempted is not None) for seq, attempted in rows)


def has_eligible(engine: Engine, space: embedding_payload.Space, *, pass_id: str) -> bool:
    """Whether a node is still eligible for indexing in this pass."""

    return bool(backfill_candidates(engine, space, pass_id=pass_id, limit=1))


def recent(engine: Engine, data: payload.RecentData, *, space: embedding_payload.Space | None) -> payload.EpisodesData:
    """The latest episodes, oldest first."""

    with engine.begin() as connection:
        seqs = [int(seq) for seq in connection.execute(recent_clause(kinds=data.kinds, limit=data.limit)).scalars()]
        loaded = _load(connection, seqs, space=space, media_bytes=False)
    episodes = [loaded[seq].node.episode for seq in reversed(seqs) if seq in loaded]
    return payload.EpisodesData(episodes=tuple(episode for episode in episodes if episode is not None))


@dataclasses.dataclass
class _Hit:
    lanes: set[payload.Lane] = dataclasses.field(default_factory=set[payload.Lane])
    similarity: float | None = None
    link_weight: float | None = None
    present: bool = False


def _linked(
    connection: Connection,
    seqs: collections.abc.Collection[int],
    *,
    kinds: collections.abc.Collection[payload.NodeKind],
    limit: int,
) -> list[tuple[int, float]]:
    """One-link neighbors of ``seqs`` (not through involves links), strongest and newest first."""

    if not seqs or limit <= 0:
        return []
    listed = list(seqs)
    other = sqlalchemy.case((_EDGES.c.source_seq.in_(listed), _EDGES.c.target_seq), else_=_EDGES.c.source_seq)
    neighbor = select(other.label("seq"), _EDGES.c.weight.label("weight")).where(
        or_(_EDGES.c.source_seq.in_(listed), _EDGES.c.target_seq.in_(listed)),
        _EDGES.c.relation.not_in(_INVOLVES),
    )
    neighbors = neighbor.subquery()
    rows = connection.execute(
        select(neighbors.c.seq, func.max(neighbors.c.weight))
        .join(_NODES, _NODES.c.seq == neighbors.c.seq)
        .where(neighbors.c.seq.not_in(listed), _NODES.c.kind.in_([kind.value for kind in kinds]))
        .group_by(neighbors.c.seq)
        .order_by(func.max(neighbors.c.weight).desc(), func.max(_NODES.c.created_at).desc(), neighbors.c.seq.desc())
        .limit(limit)
    ).all()
    return [(int(seq), float(weight)) for seq, weight in rows]


def _candidates(
    connection: Connection, hits: dict[int, _Hit], *, space: embedding_payload.Space | None
) -> tuple[payload.Candidate, ...]:
    loaded = _load(connection, list(hits), space=space, media_bytes=False)
    ordered = sorted(
        (seq for seq in hits if seq in loaded),
        key=lambda seq: (loaded[seq].node.created_at, seq),
        reverse=True,
    )
    return tuple(
        payload.Candidate(
            node=loaded[seq].node,
            lanes=tuple(lane for lane in payload.Lane if lane in hits[seq].lanes),
            similarity=hits[seq].similarity,
            hops=0 if hits[seq].lanes - {payload.Lane.LINKED} else 1,
            link_weight=hits[seq].link_weight,
            present=hits[seq].present,
        )
        for seq in ordered
    )


def _who(
    connection: Connection,
    participants: collections.abc.Sequence[mosfet.Participant],
    persons: dict[str, int],
    stamp: str,
) -> tuple[tuple[payload.Present, ...], tuple[payload.Person, ...], payload.Person]:
    """Who is here: each participant resolved, the persons they are (and ``self``), and ``self``."""

    self_seq, _ = _self_seq(connection, stamp)
    rows: dict[int, _PersonRow] = {}
    for seq in dict.fromkeys(persons.values()):
        rows[seq] = _person_by_seq(connection, seq)
    people = tuple(
        payload.Present(
            ref=participant.ref,
            part=participant.role,
            person=rows[persons[participant.ref]].id,
            known=not _provisional(rows[persons[participant.ref]]),
        )
        for participant in participants
    )
    shown = [seq for seq in dict.fromkeys(persons[participant.ref] for participant in participants) if seq != self_seq]
    you = _person(connection, _person_by_seq(connection, self_seq))
    others = tuple(_person(connection, rows[seq]) for seq in shown[: payload.MAX_PRESENT_PERSONS - 1])
    return people, (*others, you), you


def search(
    engine: Engine,
    idx: index.Index,
    data: payload.SearchData,
    *,
    space: embedding_payload.Space | None,
    cue: collections.abc.Sequence[float] | None,
    at: datetime.datetime,
) -> tuple[payload.FoundData, int, int]:
    """Every mechanical lane at once, and who is here. One transaction.

    Nearest, recent, present, and one link away; and each participant resolved to its person (a
    ref first seen gets a provisional one, an own ref is ``self``'s), with ``self``. Returns what
    was found, the persons created minus merged, and own-ref conflicts.
    """

    hits: dict[int, _Hit] = collections.defaultdict(_Hit)
    kinds = list(data.kinds)
    kind_values = [kind.value for kind in kinds]
    stamp = utc_text(at)
    with engine.begin() as connection:
        _, made = _self_seq(connection, stamp)
        resolved, created, conflicts = _resolve(connection, data.participants, stamp)
        created += made
        people, persons_here, you = _who(connection, data.participants, resolved, stamp)
        if space is not None and cue is not None and data.nearest:
            for neighbor in idx.nearest(connection, space, cue, k=data.nearest, kinds=kinds):
                hit = hits[neighbor.seq]
                hit.lanes.add(payload.Lane.SIMILAR)
                hit.similarity = max(-1.0, min(1.0, 1.0 - neighbor.distance))
        if data.recent:
            for seq in connection.execute(recent_clause(kinds=kinds, limit=data.recent)).scalars():
                hits[int(seq)].lanes.add(payload.Lane.RECENT)
        persons = list(dict.fromkeys(resolved.values()))
        if persons and data.present:
            present = connection.execute(
                select(_EDGES.c.source_seq)
                .join(_NODES, _NODES.c.seq == _EDGES.c.source_seq)
                .where(
                    _EDGES.c.target_seq.in_(persons),
                    _EDGES.c.relation.in_(_INVOLVES),
                    _NODES.c.kind.in_(kind_values),
                )
                .group_by(_EDGES.c.source_seq)
                .order_by(func.max(_NODES.c.created_at).desc(), _EDGES.c.source_seq.desc())
                .limit(data.present)
            ).scalars()
            for seq in present:
                hits[int(seq)].lanes.add(payload.Lane.PRESENT)
        if data.linked and hits:
            for seq, weight in _linked(connection, list(hits), kinds=kinds, limit=data.linked):
                hit = hits[seq]
                hit.lanes.add(payload.Lane.LINKED)
                hit.link_weight = weight
        if persons and hits:
            involving = connection.execute(
                select(_EDGES.c.source_seq).where(
                    _EDGES.c.target_seq.in_(persons),
                    _EDGES.c.relation.in_(_INVOLVES),
                    _EDGES.c.source_seq.in_(list(hits)),
                )
            ).scalars()
            for seq in involving:
                hits[int(seq)].present = True
        candidates = _candidates(connection, hits, space=space)
    found = payload.FoundData(
        candidates=candidates[:256],
        nearest_skipped=cue is None or space is None,
        people=people,
        persons=persons_here,
        you=you,
    )
    return found, created, conflicts


def fetch(
    engine: Engine,
    data: payload.FetchData,
    *,
    space: embedding_payload.Space | None,
    at: datetime.datetime,
) -> payload.FetchedData:
    """Read nodes by id (reinforcing them when asked) and their neighbors. One transaction."""

    with engine.begin() as connection:
        rows = connection.execute(
            select(_NODES.c.seq, _NODES.c.node_id).where(_NODES.c.node_id.in_(list(data.node_ids)))
        ).all()
        by_id = {node_id: int(seq) for seq, node_id in rows}
        seqs = [by_id[node_id] for node_id in data.node_ids if node_id in by_id]
        if data.reinforce and seqs:
            stamp = utc_text(at)
            _ = connection.execute(
                update(_NODES)
                .where(_NODES.c.seq.in_(seqs))
                .values(recalled_at=stamp, recall_count=_NODES.c.recall_count + 1, updated_at=stamp)
            )
        loaded = _load(connection, seqs, space=space, media_bytes=data.media)
        hits: dict[int, _Hit] = {}
        for seq, weight in _linked(connection, seqs, kinds=list(payload.NodeKind), limit=data.neighbors):
            hits[seq] = _Hit(lanes={payload.Lane.LINKED}, link_weight=weight)
        neighbors = _candidates(connection, hits, space=space)
    return payload.FetchedData(nodes=tuple(loaded[seq].node for seq in seqs if seq in loaded), neighbors=neighbors)


@dataclasses.dataclass(frozen=True)
class Lease:
    """A leased unit as Memory keeps it: the alias of every node and the unit's own episodes."""

    unit: str
    aliases: dict[str, int]
    roles: dict[str, payload.UnitRole]
    episodes: tuple[int, ...]
    decided: bool = False


def lease(
    engine: Engine,
    idx: index.Index,
    data: payload.LeaseData,
    *,
    space: embedding_payload.Space | None,
) -> tuple[payload.UnitData | None, Lease | None]:
    """Gather the next unit: the oldest unreviewed episodes, their nearest nodes, and who they involve."""

    with engine.begin() as connection:
        episodes = [int(seq) for seq in connection.execute(unreviewed_clause(limit=data.episodes)).scalars()]
        if not episodes:
            return None, None
        neighbors: list[int] = []
        taken = set(episodes)
        for seq in episodes:
            found: list[int] = []
            if space is not None and data.neighbors:
                stored = connection.execute(
                    select(_VECTORS.c.embedding).where(
                        _VECTORS.c.node_seq == seq,
                        _VECTORS.c.space_model == space.model,
                        _VECTORS.c.dimensions == space.dimensions,
                    )
                ).scalar()
                if stored is not None:
                    nearest = idx.nearest(
                        connection,
                        space,
                        index.unpack(typing.cast(bytes, stored)),
                        k=data.neighbors + len(taken),
                        kinds=(payload.NodeKind.EPISODE, payload.NodeKind.CONCEPT),
                    )
                    found = [neighbor.seq for neighbor in nearest if neighbor.seq not in taken][: data.neighbors]
            if not found:
                found = [
                    other
                    for other, _ in _linked(
                        connection,
                        [seq],
                        kinds=(payload.NodeKind.EPISODE, payload.NodeKind.CONCEPT),
                        limit=data.neighbors + len(taken),
                    )
                    if other not in taken
                ][: data.neighbors]
            neighbors.extend(found)
            taken.update(found)
        members = [*episodes, *neighbors]
        involves = connection.execute(
            select(_EDGES.c.source_seq, _EDGES.c.target_seq)
            .where(_EDGES.c.source_seq.in_(members), _EDGES.c.relation.in_(_INVOLVES))
            .order_by(_EDGES.c.source_seq, _EDGES.c.target_seq)
        ).all()
        persons = list(dict.fromkeys(int(target) for _, target in involves))
        ordered = [*members, *(person for person in persons if person not in taken)]
        aliases = {seq: f"n{position}" for position, seq in enumerate(ordered, start=1)}
        roles: dict[str, payload.UnitRole] = {}
        for seq in ordered:
            roles[aliases[seq]] = "episode" if seq in episodes else "neighbor" if seq in neighbors else "person"
        involved: dict[int, list[str]] = collections.defaultdict(list)
        for source, target in involves:
            involved[int(source)].append(aliases[int(target)])
        edges = connection.execute(
            select(_EDGES.c.source_seq, _EDGES.c.target_seq, _EDGES.c.relation)
            .where(_EDGES.c.source_seq.in_(ordered), _EDGES.c.target_seq.in_(ordered))
            .order_by(_EDGES.c.source_seq, _EDGES.c.target_seq, _EDGES.c.relation)
        ).all()
        loaded = _load(connection, ordered, space=space, media_bytes=False)
    unit = uuid.uuid4().hex
    data_unit = payload.UnitData(
        unit=unit,
        nodes=tuple(
            payload.UnitNode(
                ref=aliases[seq],
                kind=loaded[seq].node.kind,
                text=loaded[seq].node.text,
                created_at=loaded[seq].node.created_at,
                participants=tuple(dict.fromkeys(involved.get(seq, ()))),
                role=roles[aliases[seq]],
            )
            for seq in ordered
            if seq in loaded
        ),
        edges=tuple((aliases[int(source)], aliases[int(target)], str(relation)) for source, target, relation in edges),
    )
    return data_unit, Lease(
        unit=unit,
        aliases={alias: seq for seq, alias in aliases.items()},
        roles=roles,
        episodes=tuple(episodes),
    )


def decision_problem(lease_state: Lease | None, decision: payload.ConsolidateData) -> payload.FailureKind | None:
    """Whether a decision can be applied to the current lease; pure, so a guard can ask it."""

    if lease_state is None or lease_state.unit != decision.unit or lease_state.decided:
        return payload.FailureKind.STALE_LEASE
    refs = [
        *(ref for link in decision.links for ref in (link.source, link.target)),
        *(ref for merge in decision.merges for ref in merge.sources),
        *decision.forgets,
    ]
    if any(ref not in lease_state.aliases for ref in refs):
        return payload.FailureKind.INVALID_INPUT
    if any(lease_state.roles[ref] == "person" for ref in decision.forgets):
        return payload.FailureKind.INVALID_INPUT
    return None


def commit(
    engine: Engine,
    idx: index.Index,
    decision: payload.ConsolidateData,
    lease_state: Lease,
    *,
    space: embedding_payload.Space | None,
    merged_vectors: collections.abc.Sequence[collections.abc.Sequence[float] | None],
    at: datetime.datetime,
) -> tuple[payload.ConsolidatedData, dict[payload.NodeKind, int]]:
    """Apply one decision in one transaction: merges, links, forgets, and the review stamp.

    Returns the outcome and the node-count change per kind (for telemetry).
    """

    stamp = utc_text(at)
    seq_of = lease_state.aliases
    changes: dict[payload.NodeKind, int] = collections.defaultdict(int)
    with engine.begin() as connection:
        forgotten = {seq_of[ref] for ref in decision.forgets}
        if space is not None and any(values is not None for values in merged_vectors):
            idx.ensure(connection, space)
        for merge, values in zip(decision.merges, merged_vectors, strict=True):
            result = connection.execute(
                node_insert_clause(
                    node_id=uuid.uuid4().hex, kind=payload.NodeKind.CONCEPT, text=merge.text, content={}, at=stamp
                )
            )
            concept = int(typing.cast(tuple[int, ...], result.inserted_primary_key)[0])
            changes[payload.NodeKind.CONCEPT] += 1
            for ref in merge.sources:
                _ = connection.execute(
                    edge_upsert_clause(
                        source=concept, target=seq_of[ref], relation=DERIVED_FROM, weight=1.0, origin="model", at=stamp
                    )
                )
                if merge.forget_sources and lease_state.roles[ref] != "person":
                    forgotten.add(seq_of[ref])
            if space is not None and values is not None:
                _ = connection.execute(vector_upsert_clause(seq=concept, space=space, values=values))
                idx.add(connection, space, concept, values)
        linked = 0
        for link in decision.links:
            source, target = seq_of[link.source], seq_of[link.target]
            if source == target or source in forgotten or target in forgotten:
                continue
            _ = connection.execute(
                edge_upsert_clause(
                    source=source, target=target, relation=link.relation, weight=link.weight, origin="model", at=stamp
                )
            )
            linked += 1
        if forgotten:
            kinds = connection.execute(select(_NODES.c.kind).where(_NODES.c.seq.in_(list(forgotten)))).scalars()
            for kind in kinds:
                changes[payload.NodeKind(kind)] -= 1
            for clause in forget_clauses(forgotten):
                _ = connection.execute(clause)
            idx.remove(connection, sorted(forgotten))
        _ = connection.execute(reviewed_clause([seq for seq in lease_state.episodes if seq not in forgotten], at=stamp))
    return (
        payload.ConsolidatedData(
            unit=decision.unit, linked=linked, merged=len(decision.merges), forgotten=len(forgotten)
        ),
        dict(changes),
    )


def release(engine: Engine, lease_state: Lease, *, reviewed: bool, at: datetime.datetime) -> None:
    """Give a unit back, marking its episodes reviewed when asked. One transaction."""

    if not reviewed:
        return
    with engine.begin() as connection:
        _ = connection.execute(reviewed_clause(lease_state.episodes, at=utc_text(at)))


# --- person statements (one transaction each) -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Refused:
    """A person statement memory refused, with nothing changed: the closed kind and a message for the requester."""

    kind: payload.FailureKind
    message: str


class _Refusal(Exception):
    """Raised inside a statement's transaction so it rolls back; answered as ``Refused``."""

    kind: payload.FailureKind

    def __init__(self, kind: payload.FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _rename(connection: Connection, idx: index.Index, row: _PersonRow, content: dict[str, object], stamp: str) -> None:
    """Rewrite a person's facts; a changed name drops its vectors, so backfill embeds the new one."""

    name = typing.cast(str | None, content["name"])
    _ = connection.execute(
        update(_NODES)
        .where(_NODES.c.seq == row.seq)
        .values(
            text=name or "",
            content_json=json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            updated_at=stamp,
        )
    )
    if name != row.name:
        _ = connection.execute(delete(_VECTORS).where(_VECTORS.c.node_seq == row.seq))
        _ = connection.execute(delete(_ATTEMPTS).where(_ATTEMPTS.c.node_seq == row.seq))
        idx.remove(connection, [row.seq])


def _take(connection: Connection, ref: str, person: int, origin: RefOrigin, stamp: str) -> int:
    """Make ``ref`` ``person``'s, merging a provisional person that held only it; the person-count change.

    A ref that is already ``person``'s keeps how it was linked, unless the owner declares it now. The
    bot's link raises ``_Refusal`` (conflict) when the ref belongs to anyone else: they keep it until it
    is unlinked. The owner's ref is taken from whoever holds it (the caller never takes ``self``'s), and
    what was believed before stays with whom it was attributed to then.
    """

    current = _linked_persons(connection, [ref]).get(ref)
    if current == person and origin != "owner":
        return 0
    change = 0
    if current is not None and current != person:
        if _provisional(_person_by_seq(connection, current)) and _ref_count(connection, current) == 1:
            _merge(connection, current, person)
            change = -1
        elif origin != "owner":
            raise _Refusal(payload.FailureKind.CONFLICT, f"{ref} belongs to another person; unlink it from them first.")
    _ = connection.execute(person_ref_upsert_clause(ref=ref, person=person, origin=origin, at=stamp))
    return change


def _declare(
    connection: Connection, idx: index.Index, data: payload.PersonDeclareData, self_seq: int, stamp: str
) -> tuple[payload.PersonDoneData, int]:
    """The owner's household: owner refs win, except over the bot itself, whose own refs are only refused.

    Each member's owner refs it no longer lists are unlinked first (each gets a fresh provisional
    person), then each listed ref is taken from whoever holds it. A ref that is ``self``'s stays its
    own and is answered in ``refused``; nothing else of the declaration is refused. An unlinked ref
    that no member lists is answered in ``dropped``; one another member lists is that member's again.
    """

    change = 0
    declared: list[_PersonRow] = []
    refused: list[str] = []
    unlinked: list[str] = []
    for member in data.members:
        content = _person_content(name=member.name, role=member.role, born=member.born, declared=True)
        row = _person_by_id(connection, member.id)
        if row is None:
            _ = connection.execute(
                person_insert_clause(
                    person_id=member.id, at=stamp, name=member.name, role=member.role, born=member.born, declared=True
                )
            )
            change += 1
        else:
            _rename(connection, idx, row, content, stamp)
        row = _person_by_id(connection, member.id)
        assert row is not None
        dropped = connection.execute(
            select(_PERSON_REFS.c.ref).where(
                _PERSON_REFS.c.person_seq == row.seq,
                _PERSON_REFS.c.origin == "owner",
                _PERSON_REFS.c.ref.not_in(list(member.refs)),
            )
        ).scalars()
        for ref in list(dropped):
            unlinked.append(typing.cast(str, ref))
            fresh = _mint(connection, stamp)
            _ = connection.execute(person_ref_upsert_clause(ref=ref, person=fresh, origin="stamped", at=stamp))
            change += 1
        for ref in member.refs:
            if _linked_persons(connection, [ref]).get(ref) == self_seq:
                refused.append(ref)
                continue
            change += _take(connection, ref, row.seq, "owner", stamp)
        declared.append(row)
    members = {member.id for member in data.members}
    for row in _person_rows(connection, _NODES.c.key.not_in(list(members))):
        if row.declared:
            _rename(connection, idx, row, _person_content(name=row.name, role=None, born=None, declared=False), stamp)
    persons = tuple(_person(connection, _person_by_seq(connection, row.seq)) for row in declared)
    listed = {ref for member in data.members for ref in member.refs}
    dropped_refs = tuple(ref for ref in unlinked if ref not in listed)
    return payload.PersonDoneData(op="declare", persons=persons, refused=tuple(refused), dropped=dropped_refs), change


def _statement(
    connection: Connection, idx: index.Index, data: payload.PersonData, stamp: str
) -> tuple[payload.PersonDoneData, int]:
    """Apply one person statement; raises ``_Refusal`` for what memory refuses."""

    self_seq, change = _self_seq(connection, stamp)
    if isinstance(data, payload.PersonDeclareData):
        done, declared = _declare(connection, idx, data, self_seq, stamp)
        return done, change + declared
    if isinstance(data, payload.PersonListData):
        rows = _person_rows(connection, or_(_NODES.c.key == payload.SELF, _NODES.c.text != ""))
        rows.sort(key=lambda row: (row.id != payload.SELF, row.seq))
        persons = tuple(_person(connection, row) for row in rows[: payload.MAX_PERSONS])
        return payload.PersonDoneData(op="list", persons=persons), change
    if isinstance(data, payload.PersonLinkData):
        if data.person == payload.SELF:
            raise _Refusal(
                payload.FailureKind.DECLARED, "Only the producers that know the bot's own voice make a ref its own."
            )
        if data.person is None:
            person = _mint(connection, stamp, name=data.name)
            change += 1
        else:
            row = _person_by_id(connection, data.person)
            if row is None:
                raise _Refusal(payload.FailureKind.NOT_FOUND, "Memory holds no such person.")
            person = row.seq
        change += _take(connection, data.ref, person, "model", stamp)
        done = payload.PersonDoneData(op="link", person=_person(connection, _person_by_seq(connection, person)))
        return done, change
    if isinstance(data, payload.PersonUnlinkData):
        found = connection.execute(
            select(_PERSON_REFS.c.person_seq, _PERSON_REFS.c.origin).where(_PERSON_REFS.c.ref == data.ref)
        ).first()
        if found is None:
            raise _Refusal(payload.FailureKind.NOT_FOUND, "Memory holds no such ref.")
        holder, origin = _person_by_seq(connection, int(found[0])), str(found[1])
        if origin == "own" or holder.id == payload.SELF or (origin == "owner" and holder.declared):
            raise _Refusal(payload.FailureKind.DECLARED, "That ref is the bot's own or the owner's to change.")
        fresh = _mint(connection, stamp)
        _ = connection.execute(person_ref_upsert_clause(ref=data.ref, person=fresh, origin="stamped", at=stamp))
        done = payload.PersonDoneData(op="unlink", person=_person(connection, _person_by_seq(connection, fresh)))
        return done, change + 1
    row = _person_by_id(connection, data.person)
    if row is None:
        raise _Refusal(payload.FailureKind.NOT_FOUND, "Memory holds no such person.")
    if row.seq == self_seq or row.declared:
        raise _Refusal(payload.FailureKind.DECLARED, "The bot itself and declared persons are the owner's.")
    # Name and forget both answer with every ref the person has: its prints are kept, or erased.
    refs = tuple(
        connection.execute(
            select(_PERSON_REFS.c.ref)
            .where(_PERSON_REFS.c.person_seq == row.seq)
            .order_by(_PERSON_REFS.c.linked_at.desc(), _PERSON_REFS.c.ref)
        ).scalars()
    )
    if isinstance(data, payload.PersonNameData):
        _rename(
            connection,
            idx,
            row,
            _person_content(name=data.name, role=row.role, born=row.born, declared=row.declared),
            stamp,
        )
        return payload.PersonDoneData(
            op="name", person=_person(connection, _person_by_seq(connection, row.seq)), refs=refs
        ), change
    as_it_was = _person(connection, row)
    _ = connection.execute(delete(_PERSON_REFS).where(_PERSON_REFS.c.person_seq == row.seq))
    for clause in forget_clauses([row.seq]):
        _ = connection.execute(clause)
    idx.remove(connection, [row.seq])
    return payload.PersonDoneData(op="forget", person=as_it_was, refs=refs), change - 1


def person_statement(
    engine: Engine, idx: index.Index, data: payload.PersonData, *, at: datetime.datetime
) -> tuple[payload.PersonDoneData | Refused, int]:
    """Apply one person statement in one transaction: its reply, or what was refused; and the person-count change.

    A refusal rolls the whole transaction back. The owner's declaration is never refused for a ref (an
    own ref it names is answered in ``refused``, D-W8-19), so it is refused only by a failing store.
    """

    try:
        with engine.begin() as connection:
            return _statement(connection, idx, data, utc_text(at))
    except _Refusal as refusal:
        return Refused(kind=refusal.kind, message=str(refusal)), 0


__all__ = [
    "DERIVED_FROM",
    "INVOLVES_SOURCE",
    "INVOLVES_TARGET",
    "PERMANENT_FAILURES",
    "Kept",
    "Lease",
    "RefOrigin",
    "Refused",
    "attempt_clear_clause",
    "attempt_upsert_clause",
    "backfill_candidates",
    "backfill_clause",
    "commit",
    "decision_problem",
    "edge_upsert_clause",
    "ensure_self",
    "fetch",
    "forget_clauses",
    "has_eligible",
    "lease",
    "node_insert_clause",
    "node_parts",
    "parse_utc",
    "person_insert_clause",
    "person_ref_upsert_clause",
    "person_statement",
    "persons_select_clause",
    "recent",
    "recent_clause",
    "record_attempts",
    "release",
    "reviewed_clause",
    "search",
    "unreviewed_clause",
    "utc_text",
    "vector_upsert_clause",
    "write_episode",
    "write_vectors",
]
