"""Identity persistence: the ``Store`` seam, the bounded ``InProcessStore``, and the store-backed ``Matcher``."""

import abc
import asyncio
import collections.abc
import dataclasses
import datetime
import math
import threading
import typing
import uuid

from . import identification


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Store(abc.ABC):
    """Where identities and their prints live; non-HSM, and each call is one atomic transaction.

    Every call that matches runs ``identification.decide`` inside its own transaction. Every
    ``Match`` and ``Refine`` advances the id's ``seen_at`` (retention counts from last seen); only
    ``Refine`` and the enroll upsert change its vector and its capped ``seen_count`` weight. Each
    model has one recorded dimension count, and a print of other dimensions is ``INVALID_INPUT``.
    Ids marked ``own``, or linked to a person, are never pruned. Every failure raises
    ``identification.StoreError``, and its message never carries a print or a vector.
    """

    @abc.abstractmethod
    def identify(
        self,
        kind: identification.Kind,
        prints: tuple[identification.Print, ...],
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> tuple[identification.Recognition, ...]:
        """Match or mint an id for each print, in order; a store never leaves a print unidentified.

        Raises:
            identification.StoreError: ``INVALID_INPUT`` for a dimension mismatch, else ``STORE_FAILED``.
        """

    @abc.abstractmethod
    def enroll(
        self,
        kind: identification.Kind,
        id: identification.Id,
        print: identification.Print,
        own: bool,
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> identification.Recognition:
        """Upsert ``id``: insert it, or refine it with ``print`` weighted by at most ``policy.refine_weight_cap``.

        ``own`` is sticky; an existing id answers ``recognized=True`` with its similarity to ``print``.

        Raises:
            identification.StoreError: ``INVALID_INPUT`` when ``print.model`` differs from the id's
                model or its dimensions from the model's, else ``STORE_FAILED``.
        """

    @abc.abstractmethod
    def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        """Delete ``id`` and its print; whether it existed.

        Raises:
            identification.StoreError: ``STORE_FAILED`` when the store fails.
        """

    @abc.abstractmethod
    def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        """Delete unlinked, non-own ids last seen before ``unseen_before``, then the least recently seen
        of them until at most ``max_ids`` ids of ``kind`` remain; how many were deleted.

        Raises:
            identification.StoreError: ``STORE_FAILED`` when the store fails.
        """

    @staticmethod
    def _similarity(stored: tuple[float, ...], vector: tuple[float, ...]) -> float:
        """Cosine similarity of two unit vectors, clamped to [-1, 1] against rounding."""

        return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(stored, vector, strict=True))))

    @staticmethod
    def _refined(stored: tuple[float, ...], weight: int, vector: tuple[float, ...]) -> tuple[float, ...]:
        """The running mean of ``stored`` (worth ``weight`` prints) and ``vector``, L2-normalized."""

        mean = [(a * weight + b) / (weight + 1) for a, b in zip(stored, vector, strict=True)]
        norm = math.sqrt(math.fsum(value * value for value in mean))
        # Opposite unit vectors average to zero; the newest print is then the only direction left.
        return vector if norm == 0.0 else tuple(value / norm for value in mean)


@dataclasses.dataclass(frozen=True, slots=True)
class _Entry:
    model: str
    vector: tuple[float, ...]
    own: bool
    created_at: datetime.datetime
    seen_at: datetime.datetime
    seen_count: int


class InProcessStore(Store):
    """A bounded store in process memory, for tests and compositions without a database.

    ``max_ids`` is a hard bound per kind: minting or enrolling a new id at the bound first evicts
    the least recently seen non-own id of that kind, and fails ``STORE_FAILED`` when every id is
    own. It has no person links (nothing writes them yet). One lock serializes every call, so
    registries sharing it, and calls a registry abandoned on a timeout, never overlap.
    """

    _max_ids: int
    _lock: threading.Lock
    _ids: dict[identification.Kind, dict[str, _Entry]]
    _dimensions: dict[str, int]

    def __init__(self, *, max_ids: int) -> None:
        """Hold at most ``max_ids`` ids of each kind.

        Raises:
            ValueError: If ``max_ids`` is not positive.
        """

        if max_ids < 1:
            raise ValueError("InProcessStore max_ids must be positive.")
        self._max_ids = max_ids
        self._lock = threading.Lock()
        self._ids = {}
        self._dimensions = {}

    @typing.override
    def identify(
        self,
        kind: identification.Kind,
        prints: tuple[identification.Print, ...],
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> tuple[identification.Recognition, ...]:
        with self._lock:
            # Staged copies commit only when every print succeeded: one call is one transaction.
            ids = dict(self._ids.get(kind, {}))
            dimensions = dict(self._dimensions)
            recognitions: list[identification.Recognition] = []
            for print in prints:
                recorded = dimensions.setdefault(print.model, len(print.vector))
                if recorded != len(print.vector):
                    raise identification.StoreError(
                        identification.FailureKind.INVALID_INPUT,
                        f"a print has {len(print.vector)} dimensions; its model's space has {recorded}",
                    )
                neighbors = [
                    identification.Neighbor(
                        id=id, similarity=Store._similarity(entry.vector, print.vector), own=entry.own
                    )
                    for id, entry in ids.items()
                    if entry.model == print.model
                ]
                decision = identification.decide(neighbors, policy)
                if isinstance(decision, identification.Mint):
                    if len(ids) >= self._max_ids:
                        evictable = [(entry.seen_at, id) for id, entry in ids.items() if not entry.own]
                        if not evictable:
                            raise identification.StoreError(
                                identification.FailureKind.STORE_FAILED, "the in-process store is full of own ids"
                            )
                        del ids[min(evictable)[1]]
                    minted = uuid.uuid4().hex
                    ids[minted] = _Entry(
                        model=print.model, vector=print.vector, own=False, created_at=at, seen_at=at, seen_count=1
                    )
                    recognitions.append(
                        identification.Recognition(
                            id=minted, recognized=False, similarity=decision.similarity, own=False
                        )
                    )
                    continue
                entry = ids[decision.id]
                if isinstance(decision, identification.Refine):
                    entry = dataclasses.replace(
                        entry,
                        vector=Store._refined(
                            entry.vector, min(entry.seen_count, policy.refine_weight_cap), print.vector
                        ),
                        seen_count=min(entry.seen_count + 1, policy.refine_weight_cap),
                    )
                ids[decision.id] = dataclasses.replace(entry, seen_at=at)
                recognitions.append(
                    identification.Recognition(
                        id=decision.id, recognized=True, similarity=decision.similarity, own=decision.own
                    )
                )
            self._ids[kind] = ids
            self._dimensions = dimensions
            return tuple(recognitions)

    @typing.override
    def enroll(
        self,
        kind: identification.Kind,
        id: identification.Id,
        print: identification.Print,
        own: bool,
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> identification.Recognition:
        with self._lock:
            ids = dict(self._ids.get(kind, {}))
            dimensions = dict(self._dimensions)
            recorded = dimensions.setdefault(print.model, len(print.vector))
            if recorded != len(print.vector):
                raise identification.StoreError(
                    identification.FailureKind.INVALID_INPUT,
                    f"a print has {len(print.vector)} dimensions; its model's space has {recorded}",
                )
            entry = ids.get(id)
            if entry is None:
                if len(ids) >= self._max_ids:
                    evictable = [(stored.seen_at, stored_id) for stored_id, stored in ids.items() if not stored.own]
                    if not evictable:
                        raise identification.StoreError(
                            identification.FailureKind.STORE_FAILED, "the in-process store is full of own ids"
                        )
                    del ids[min(evictable)[1]]
                ids[id] = _Entry(
                    model=print.model, vector=print.vector, own=own, created_at=at, seen_at=at, seen_count=1
                )
                recognition = identification.Recognition(id=id, recognized=False, similarity=None, own=own)
            else:
                if entry.model != print.model:
                    raise identification.StoreError(
                        identification.FailureKind.INVALID_INPUT, "the print's model differs from the id's model"
                    )
                similarity = Store._similarity(entry.vector, print.vector)
                entry = dataclasses.replace(
                    entry,
                    vector=Store._refined(entry.vector, min(entry.seen_count, policy.refine_weight_cap), print.vector),
                    seen_count=min(entry.seen_count + 1, policy.refine_weight_cap),
                    own=entry.own or own,
                    seen_at=at,
                )
                ids[id] = entry
                recognition = identification.Recognition(id=id, recognized=True, similarity=similarity, own=entry.own)
            self._ids[kind] = ids
            self._dimensions = dimensions
            return recognition

    @typing.override
    def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        with self._lock:
            return self._ids.get(kind, {}).pop(id, None) is not None

    @typing.override
    def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        with self._lock:
            ids = self._ids.get(kind, {})
            before = len(ids)
            for id in [id for id, entry in ids.items() if not entry.own and entry.seen_at < unseen_before]:
                del ids[id]
            evictable = sorted((entry.seen_at, id) for id, entry in ids.items() if not entry.own)
            for _, id in evictable[: max(0, len(ids) - max_ids)]:
                del ids[id]
            return before - len(ids)


class Matcher(identification.Identifier):
    """The store-backed ``Identifier``: one ``MatchPolicy`` over one ``Store``.

    Store calls block, so each runs in ``asyncio.to_thread``; a call its caller abandons still
    completes. ``clock`` is the timezone-aware UTC wall clock that stamps ``seen_at`` (it is
    persisted, so it is never the event loop's clock). Store errors propagate unchanged.
    """

    _store: Store
    _policy: identification.MatchPolicy
    _clock: collections.abc.Callable[[], datetime.datetime]

    def __init__(
        self,
        *,
        store: Store,
        policy: identification.MatchPolicy,
        clock: collections.abc.Callable[[], datetime.datetime] = _utc_now,
    ) -> None:
        self._store = store
        self._policy = policy
        self._clock = clock

    @typing.override
    async def identify(
        self, kind: identification.Kind, prints: tuple[identification.Print, ...]
    ) -> tuple[identification.Recognition | None, ...]:
        return await asyncio.to_thread(self._store.identify, kind, prints, self._policy, at=self._clock())

    @typing.override
    async def enroll(
        self, kind: identification.Kind, id: identification.Id, print: identification.Print, own: bool
    ) -> identification.Recognition:
        return await asyncio.to_thread(self._store.enroll, kind, id, print, own, self._policy, at=self._clock())

    @typing.override
    async def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        return await asyncio.to_thread(self._store.forget, kind, id)

    @typing.override
    async def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        return await asyncio.to_thread(self._store.prune, kind, unseen_before=unseen_before, max_ids=max_ids)


__all__ = ["InProcessStore", "Matcher", "Store"]
