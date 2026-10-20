"""HSM runtime tests for the episodic ``memory.Memory`` with fake embedders and the real Embedding ability."""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import contextlib
import dataclasses
import datetime
import logging
import math
import pathlib
import sqlite3
import threading
import typing
import uuid

import hsm
import mosfet
import pytest
from sqlalchemy import text as sql
from sqlalchemy.engine import Connection, Engine

from mosfet import environment, media
from mosfet.abilities import ability, embedding, memory, processing
from mosfet.abilities.language import text
from mosfet.protocols import attachment
from tests.bot.abilities.memory.memory_fixtures import disposed_engines
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_SPACE = embedding.Space(model="test/embedder", dimensions=2)
_OTHER_SPACE = embedding.Space(model="test/other", dimensions=2)
_TEXT_ONLY = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)
_WAIT = datetime.timedelta(seconds=3)
_ADA = mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE)
_BOT = mosfet.Participant(ref="bot", role=mosfet.ParticipantRole.TARGET)


def _vector_for(item: embedding.Item) -> embedding.Vector:
    """A deterministic unit vector: text mentioning "noon" points one way, everything else the other."""

    words = " ".join(part.text for part in item.parts if isinstance(part, text.TextPart))
    angle = 0.1 if "noon" in words else 1.4
    return embedding.Vector(values=(math.cos(angle), math.sin(angle)))


class FakeEmbedder(embedding.Embedder):
    """Plain-object embedder; ``fail`` decides a failure per request, ``gate`` holds calls until set."""

    calls: list[embedding.InputData]

    def __init__(
        self,
        *,
        capacity: embedding.Capacity = _TEXT_ONLY,
        attachments: tuple[hsm.Instance, ...] = (),
        fail: collections.abc.Callable[[embedding.InputData], embedding.Error | None] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.capacity = capacity
        self.attachments = attachments
        self.fail = fail
        self.gate = gate
        self.calls = []

    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx
        self.calls.append(input)
        if self.gate is not None:
            _ = await self.gate.wait()
        error = None if self.fail is None else self.fail(input)
        if error is not None:
            raise error
        return embedding.OutputData(
            space=self.capacity.space, role=input.role, vectors=tuple(_vector_for(item) for item in input.items)
        )


class BlockingIndex(memory.ExactIndex):
    """Exact index whose search holds the database worker until the test releases it."""

    release: threading.Event
    entered: threading.Event
    armed: bool

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.armed = False

    def nearest(
        self,
        connection: Connection,
        space: embedding.Space,
        values: collections.abc.Sequence[float],
        *,
        k: int,
        kinds: collections.abc.Sequence[memory.NodeKind],
    ) -> tuple[memory.Neighbor, ...]:
        if self.armed:
            self.entered.set()
            _ = self.release.wait(timeout=5)
        return super().nearest(connection, space, values, k=k, kinds=kinds)


class Owner(hsm.Instance):
    """Attachment owner and reply address that records every event delivered to it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: Owner, event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "MemoryTestOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {event.name for event in events}
        return [event for event in self.events if event.name in names]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT.total_seconds()):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


class FakeAttachment(hsm.Instance, attachment.Attachment):
    """A root-started provider actor that answers attach, optionally holding or failing it."""

    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    release: asyncio.Event

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "MemoryTestAttachment", hsm.initial(hsm.target("ready")), hsm.state("ready")
    )

    def __init__(self, *, hold: bool = False, failure: attachment.FailureKind | None = None) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.failure = failure

    async def _answer(self, ctx: hsm.Context, event: hsm.Event[typing.Any], result: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, attachment.AttachData | attachment.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        await hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(result, id=event.id, source=hsm.id(self), target=hsm.id(target)),
        )

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        _ = await self.release.wait()
        data = event.data
        assert isinstance(data, attachment.AttachData)
        result = (
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=data.actor, created=True))
            if self.failure is None
            else attachment.AttachFailedEvent.with_data(
                attachment.FailedData(actor=data.actor, kind=self.failure, message="model is not loaded")
            )
        )
        await self._answer(ctx, event, result)
        return True

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> bool:
        data = event.data
        assert isinstance(data, attachment.DetachData)
        await self._answer(
            ctx, event, attachment.DetachedEvent.with_data(attachment.DetachedData(actor=data.actor, removed=True))
        )
        return True


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    memory: memory.Memory
    owner: Owner
    engine: Engine

    async def attach(self) -> None:
        _ = await self.memory.attach(
            self.ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), "attach")
        )

    async def attached(self) -> None:
        await self.attach()
        await self.owner.until(lambda: bool(self.owner.named(attachment.AttachCompleteEvent)))

    async def detach(self) -> None:
        _ = await self.memory.detach(
            self.ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), "detach")
        )

    async def send(self, request: hsm.Event[typing.Any], data: object, request_id: str) -> None:
        """Deliver one request addressed from the owner, so the owner records its reply."""

        _ = await hsm.dispatch(
            self.ctx,
            self.memory,
            dataclasses.replace(
                request.with_data_and_id(data, request_id), source=hsm.id(self.owner), target=hsm.id(self.memory)
            ),
        )

    def replies(self, request_id: str) -> list[hsm.Event[typing.Any]]:
        return [
            event
            for event in self.owner.events
            if event.id == request_id and event.name.startswith("bot.ability.memory.")
        ]

    async def reply(self, request_id: str) -> hsm.Event[typing.Any]:
        await self.owner.until(lambda: bool(self.replies(request_id)))
        return self.replies(request_id)[0]

    async def ask(self, request: hsm.Event[typing.Any], data: object) -> hsm.Event[typing.Any]:
        """Send one request through a one-shot reply actor and return its one correlated reply."""

        terminals = {
            memory.RememberEvent.name: memory.RememberedEvent,
            memory.SearchEvent.name: memory.FoundEvent,
            memory.FetchEvent.name: memory.FetchedEvent,
            memory.RecentEvent.name: memory.RecentOutputEvent,
            memory.BackfillEvent.name: memory.BackfilledEvent,
            memory.LeaseEvent.name: memory.LeasedEvent,
            memory.ConsolidateEvent.name: memory.ConsolidatedEvent,
            memory.ReleaseEvent.name: memory.ReleasedEvent,
            memory.PersonEvent.name: memory.PersonDoneEvent,
            self.memory.input_event.name: self.memory.output_event,
        }
        return await ability.run_terminal_operation(
            self.ctx,
            child=self.memory,
            request=request.with_data_and_id(data, uuid.uuid4().hex),
            terminals=(terminals[request.name], memory.FailedEvent),
            timeout=_WAIT,
        )

    def rows(self, statement: str) -> list[tuple[typing.Any, ...]]:
        with self.engine.connect() as connection:
            return [tuple(row) for row in connection.execute(sql(statement)).all()]


async def _harness(
    embedder: embedding.Embedder | None = None,
    *,
    index: memory.Index | None = None,
    now: collections.abc.Callable[[], datetime.datetime] | None = None,
    started: tuple[hsm.Instance, ...] = (),
    embed_timeout: datetime.timedelta = datetime.timedelta(seconds=2),
    operation_timeout: datetime.timedelta = datetime.timedelta(seconds=3),
) -> Harness:
    ctx = environment.Environment()
    owner = Owner()
    _ = await mosfet.started(ctx, owner, Owner.model)
    for actor in started:
        _ = await mosfet.started(ctx, actor, FakeAttachment.model)
    engine = memory.store.open_sqlite_engine()
    store = memory.Memory(
        engine=engine,
        embedder=embedder,
        index=index,
        now=now,
        embed_timeout=embed_timeout,
        operation_timeout=operation_timeout,
    )
    return Harness(ctx=ctx, memory=store, owner=owner, engine=engine)


def _episode(
    words: str = "Ada: can you call me back at noon?",
    *,
    turn_id: str = "turn-1",
    participants: tuple[mosfet.Participant, ...] = (_ADA,),
    images: int = 0,
) -> memory.EpisodeData:
    parts: list[text.ContentPart] = [text.TextPart(text=words)]
    parts.extend(
        text.ImagePart(
            image=media.Image(content=_PNG + bytes([index]), media_type="image/png", width_px=4, height_px=3)
        )
        for index in range(images)
    )
    return memory.EpisodeData(
        turn_id=turn_id,
        stimulus="bot.ability.conversation.messages",
        parts=tuple(parts),
        participants=participants,
        acts=(memory.Act(event="bot.ability.speaking.say", data={"text": "Sure."}),),
        outcome=memory.Outcome.COMPLETED,
    )


def _run[T](scenario: collections.abc.Callable[[], collections.abc.Awaitable[T]]) -> T:
    async def run() -> T:
        return await scenario()

    return asyncio.run(run())


# --- remember, read back ------------------------------------------------------------------------


def test_remember_orders_same_microsecond_episodes_by_sequence() -> None:
    """A frozen clock stamps three episodes alike; recency still reads them in remember order."""

    instant = datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC)

    async def scenario() -> list[str]:
        harness = await _harness(now=lambda: instant)
        await harness.attached()
        for turn in ("third-by-uuid", "first-by-uuid", "second-by-uuid"):
            _ = await harness.ask(memory.RememberEvent, _episode(turn, turn_id=turn))
        reply = await harness.ask(memory.RecentEvent, memory.RecentData(limit=3))
        assert isinstance(reply.data, memory.EpisodesData)
        return [episode.turn_id for episode in reply.data.episodes]

    assert _run(scenario) == ["third-by-uuid", "first-by-uuid", "second-by-uuid"]


def test_remember_keeps_parts_persons_and_involves_links() -> None:
    async def scenario() -> tuple[
        memory.FetchedData, memory.FetchedData, list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]]
    ]:
        harness = await _harness()
        await harness.attached()
        first = await harness.ask(memory.RememberEvent, _episode(images=1, participants=(_ADA, _BOT)))
        _ = await harness.ask(memory.RememberEvent, _episode("Ada again", turn_id="turn-2", participants=(_ADA,)))
        assert isinstance(first.data, memory.RememberedData)
        assert first.data.indexed is False
        with_media = await harness.ask(memory.FetchEvent, memory.FetchData(node_ids=(first.data.node_id,), media=True))
        placeholders = await harness.ask(memory.FetchEvent, memory.FetchData(node_ids=(first.data.node_id,)))
        assert isinstance(with_media.data, memory.FetchedData) and isinstance(placeholders.data, memory.FetchedData)
        persons = harness.rows("SELECT ref FROM bot_person_ref ORDER BY ref")
        edges = harness.rows("SELECT relation FROM bot_memory_edge ORDER BY source_seq, relation")
        return with_media.data, placeholders.data, persons, edges

    with_media, placeholders, persons, edges = _run(scenario)

    episode = with_media.nodes[0].episode
    assert episode is not None
    assert episode.text == "Ada: can you call me back at noon?"
    assert episode.participants == (_ADA, _BOT)
    assert episode.acts == (memory.Act(event="bot.ability.speaking.say", data={"text": "Sure."}),)
    assert episode.images == (media.Image(content=_PNG + b"\x00", media_type="image/png", width_px=4, height_px=3),)
    released = placeholders.nodes[0].episode
    assert released is not None
    assert released.images == (
        media.ReleasedImage(media_type="image/png", width_px=4, height_px=3, byte_count=len(_PNG) + 1, reason="memory"),
    )
    assert persons == [("+15555550100",), ("bot",)]
    assert edges == [("involves_source",), ("involves_target",), ("involves_source",)]


def test_remember_indexes_with_an_embedder_and_keeps_an_unembeddable_episode() -> None:
    def fail(input: embedding.InputData) -> embedding.Error | None:
        long = any(
            "LONG" in part.text for item in input.items for part in item.parts if isinstance(part, text.TextPart)
        )
        return embedding.Error(embedding.FailureKind.TOO_LONG, "too long", item_index=0) if long else None

    async def scenario() -> tuple[bool, bool, list[tuple[typing.Any, ...]]]:
        harness = await _harness(FakeEmbedder(fail=fail))
        await harness.attached()
        indexed = await harness.ask(memory.RememberEvent, _episode())
        kept = await harness.ask(memory.RememberEvent, _episode("LONG " * 10, turn_id="turn-2"))
        assert isinstance(indexed.data, memory.RememberedData) and isinstance(kept.data, memory.RememberedData)
        attempts = harness.rows("SELECT failure_kind, permanent, pass_id FROM bot_memory_index_attempt")
        return indexed.data.indexed, kept.data.indexed, attempts

    indexed, kept, attempts = _run(scenario)

    assert indexed is True
    assert kept is False
    assert attempts == [("too_long", 1, None)]


def test_an_episode_with_nothing_said_is_kept_without_an_index_attempt(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing to embed is not a capacity failure: no embed, no attempt row, no warning, no backfill."""

    async def scenario() -> tuple[
        memory.RememberedData, memory.BackfilledData, list[embedding.InputData], list[tuple[typing.Any, ...]]
    ]:
        embedder = FakeEmbedder()
        harness = await _harness(embedder)
        await harness.attached()
        kept = await harness.ask(
            memory.RememberEvent, memory.EpisodeData(turn_id="quiet", outcome=memory.Outcome.CANCELLED)
        )
        backfilled = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=4, pass_id="pass-1"))
        assert isinstance(kept.data, memory.RememberedData) and isinstance(backfilled.data, memory.BackfilledData)
        return kept.data, backfilled.data, embedder.calls, harness.rows("SELECT * FROM bot_memory_index_attempt")

    with caplog.at_level(logging.WARNING, logger="mosfet.abilities.memory.memory"):
        kept, backfilled, calls, attempts = _run(scenario)

    assert kept.indexed is False
    assert backfilled == memory.BackfilledData(indexed=0, unindexable=0, remaining=False)
    assert calls == []
    assert attempts == []
    assert "unindexable" not in caplog.text


def test_image_parts_are_dropped_for_a_text_only_embedder() -> None:
    async def scenario() -> list[embedding.InputData]:
        embedder = FakeEmbedder()
        harness = await _harness(embedder)
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode(images=2))
        return embedder.calls

    (call,) = _run(scenario)
    assert call.role is embedding.Role.DOCUMENT
    assert call.items == (embedding.Item(parts=(text.TextPart(text="Ada: can you call me back at noon?"),)),)


# --- search and fetch -----------------------------------------------------------------------------


def test_search_surfaces_similar_recent_present_and_linked_candidates() -> None:
    async def scenario() -> tuple[memory.FoundData, memory.FoundData]:
        harness = await _harness(FakeEmbedder())
        await harness.attached()
        noon = await harness.ask(memory.RememberEvent, _episode("lunch at noon", turn_id="noon", participants=()))
        _ = await harness.ask(memory.RememberEvent, _episode("Ada called", turn_id="ada"))
        for turn in ("one", "two"):
            _ = await harness.ask(memory.RememberEvent, _episode(f"weather {turn}", turn_id=turn, participants=()))
        assert isinstance(noon.data, memory.RememberedData)
        unit = await harness.ask(memory.LeaseEvent, memory.LeaseData(episodes=1, neighbors=0))
        assert isinstance(unit.data, memory.LeasedData) and unit.data.unit is not None
        refs = [node.ref for node in unit.data.unit.nodes]
        merged = await harness.ask(
            memory.ConsolidateEvent,
            memory.ConsolidateData(
                unit=unit.data.unit.unit, merges=(memory.MergeData(sources=(refs[0], refs[0]), text="Ada likes noon"),)
            ),
        )
        assert merged.name == memory.ConsolidatedEvent.name, merged.data
        found = await harness.ask(
            memory.SearchEvent,
            memory.SearchData(
                cue=(text.TextPart(text="noon"),), participants=(_ADA,), nearest=1, recent=1, present=2, linked=2
            ),
        )
        blind = await harness.ask(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),), nearest=0))
        assert isinstance(found.data, memory.FoundData) and isinstance(blind.data, memory.FoundData)
        return found.data, blind.data

    found, blind = _run(scenario)

    lanes = {candidate.node.text: set(candidate.lanes) for candidate in found.candidates}
    assert memory.Lane.SIMILAR in lanes["lunch at noon"]
    assert memory.Lane.RECENT in lanes["Ada likes noon"]
    assert memory.Lane.PRESENT in lanes["Ada called"]
    assert lanes["Ada called"] == {memory.Lane.PRESENT}
    assert found.nearest_skipped is False
    assert next(candidate for candidate in found.candidates if candidate.node.text == "Ada called").present is True
    assert blind.nearest_skipped is True


def test_search_without_an_embedder_skips_the_nearest_lane() -> None:
    async def scenario() -> memory.FoundData:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode())
        found = await harness.ask(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)))
        assert isinstance(found.data, memory.FoundData)
        return found.data

    found = _run(scenario)
    assert found.nearest_skipped is True
    assert [candidate.lanes for candidate in found.candidates] == [(memory.Lane.RECENT,)]


def test_fetch_reinforces_in_the_same_read() -> None:
    instant = datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC)

    async def scenario() -> memory.Episode:
        harness = await _harness(now=lambda: instant)
        await harness.attached()
        kept = await harness.ask(memory.RememberEvent, _episode())
        assert isinstance(kept.data, memory.RememberedData)
        _ = await harness.ask(memory.FetchEvent, memory.FetchData(node_ids=(kept.data.node_id,), reinforce=True))
        again = await harness.ask(
            memory.FetchEvent, memory.FetchData(node_ids=(kept.data.node_id, "missing"), reinforce=True)
        )
        assert isinstance(again.data, memory.FetchedData)
        (node,) = again.data.nodes
        assert node.episode is not None
        return node.episode

    episode = _run(scenario)
    assert episode.recall_count == 2
    assert episode.recalled_at == instant


# --- backfill (review finding: exact attribution within the whole capacity) ----------------------


def test_backfill_attributes_a_too_long_item_to_that_node_only() -> None:
    """Three too-long nodes and two indexable ones: one pass indexes the two and marks only the three."""

    def fail(input: embedding.InputData) -> embedding.Error | None:
        for position, item in enumerate(input.items):
            if any("LONG" in part.text for part in item.parts if isinstance(part, text.TextPart)):
                return embedding.Error(embedding.FailureKind.TOO_LONG, "too long", item_index=position)
        return None

    async def scenario() -> tuple[list[memory.BackfilledData], memory.BackfilledData, list[tuple[typing.Any, ...]]]:
        seed = await _harness()
        await seed.attached()
        for turn in ("LONG a", "fine b", "LONG c", "fine d", "LONG e"):
            _ = await seed.ask(memory.RememberEvent, _episode(turn, turn_id=turn, participants=()))
        harness = Harness(
            ctx=seed.ctx,
            memory=memory.Memory(engine=seed.engine, embedder=FakeEmbedder(fail=fail)),
            owner=seed.owner,
            engine=seed.engine,
        )
        harness.owner.events.clear()
        await harness.attached()
        batches: list[memory.BackfilledData] = []
        while True:
            reply = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=2, pass_id="pass-1"))
            assert isinstance(reply.data, memory.BackfilledData), reply.data
            batches.append(reply.data)
            if not reply.data.remaining:
                break
            assert len(batches) <= 5
        second = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=2, pass_id="pass-2"))
        assert isinstance(second.data, memory.BackfilledData)
        attempts = harness.rows(
            "SELECT n.text, a.failure_kind, a.permanent FROM bot_memory_index_attempt a "
            "JOIN bot_memory_node n ON n.seq = a.node_seq ORDER BY n.seq"
        )
        return batches, second.data, attempts

    batches, second, attempts = _run(scenario)

    assert sum(batch.indexed for batch in batches) == 2
    assert sum(batch.unindexable for batch in batches) == 3
    assert len(batches) <= 5
    assert second == memory.BackfilledData(indexed=0, unindexable=0, remaining=False)
    assert attempts == [("LONG a", "too_long", 1), ("LONG c", "too_long", 1), ("LONG e", "too_long", 1)]


def test_backfill_batches_images_within_the_request_image_budget() -> None:
    """Each episode fits alone; batches are cut so their images fit together, and nothing is marked."""

    capacity = embedding.Capacity(
        space=_SPACE,
        max_items=8,
        max_input_tokens=2048,
        images=media.ImageCapacity(max_images=2, max_total_bytes=1024, media_types=frozenset({"image/png"})),
        interleaved=True,
    )

    async def scenario() -> tuple[list[int], list[tuple[typing.Any, ...]], list[int]]:
        seed = await _harness()
        await seed.attached()
        for turn in ("a", "b", "c"):
            _ = await seed.ask(memory.RememberEvent, _episode(turn, turn_id=turn, participants=(), images=1))
        _ = await seed.ask(memory.RememberEvent, _episode("too many", turn_id="d", participants=(), images=3))
        embedder = FakeEmbedder(capacity=capacity)
        harness = Harness(
            ctx=seed.ctx,
            memory=memory.Memory(engine=seed.engine, embedder=embedder),
            owner=seed.owner,
            engine=seed.engine,
        )
        await harness.attached()
        indexed: list[int] = []
        while True:
            reply = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="pass"))
            assert isinstance(reply.data, memory.BackfilledData)
            indexed.append(reply.data.indexed)
            if not reply.data.remaining:
                break
        attempts = harness.rows("SELECT failure_kind, permanent FROM bot_memory_index_attempt")
        return indexed, attempts, [len(call.items) for call in embedder.calls]

    indexed, attempts, call_sizes = _run(scenario)

    assert sum(indexed) == 3
    assert call_sizes == [2, 1]
    assert attempts == [("unsupported_input", 1)]


def test_backfill_retries_a_batch_failure_without_an_item_once_per_pass_and_then_alone() -> None:
    def unavailable(input: embedding.InputData) -> embedding.Error | None:
        del input
        return embedding.Error(embedding.FailureKind.UNAVAILABLE, "remote is not serving")

    async def scenario() -> tuple[
        memory.BackfilledData, memory.BackfilledData, list[int], list[tuple[typing.Any, ...]]
    ]:
        seed = await _harness()
        await seed.attached()
        for turn in ("a", "b"):
            _ = await seed.ask(memory.RememberEvent, _episode(turn, turn_id=turn, participants=()))
        embedder = FakeEmbedder(fail=unavailable)
        harness = Harness(
            ctx=seed.ctx,
            memory=memory.Memory(engine=seed.engine, embedder=embedder),
            owner=seed.owner,
            engine=seed.engine,
        )
        await harness.attached()
        first = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="pass-1"))
        embedder.fail = None
        retried = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="pass-2"))
        assert isinstance(first.data, memory.BackfilledData) and isinstance(retried.data, memory.BackfilledData)
        return (
            first.data,
            retried.data,
            [len(call.items) for call in embedder.calls],
            harness.rows("SELECT permanent FROM bot_memory_index_attempt"),
        )

    first, retried, call_sizes, attempts = _run(scenario)

    assert first == memory.BackfilledData(indexed=0, unindexable=0, remaining=False)
    assert retried == memory.BackfilledData(indexed=1, unindexable=0, remaining=True)
    assert call_sizes == [2, 1]
    assert attempts == [(0,)]


def test_a_new_space_makes_every_node_eligible_again() -> None:
    async def scenario() -> memory.BackfilledData:
        seed = await _harness(FakeEmbedder())
        await seed.attached()
        _ = await seed.ask(memory.RememberEvent, _episode())
        other = FakeEmbedder(capacity=embedding.Capacity(space=_OTHER_SPACE, max_items=8, max_input_tokens=2048))
        harness = Harness(
            ctx=seed.ctx, memory=memory.Memory(engine=seed.engine, embedder=other), owner=seed.owner, engine=seed.engine
        )
        await harness.attached()
        reply = await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="switch"))
        assert isinstance(reply.data, memory.BackfilledData)
        return reply.data

    assert _run(scenario) == memory.BackfilledData(indexed=1, unindexable=0, remaining=False)


def test_backfill_without_an_embedder_has_nothing_to_do() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode())
        return await harness.ask(memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="none"))

    assert _run(scenario).data == memory.BackfilledData(indexed=0, unindexable=0, remaining=False)


# --- lease and consolidate ----------------------------------------------------------------------


async def _leased(harness: Harness, *turns: str) -> memory.UnitData:
    for turn in turns:
        _ = await harness.ask(memory.RememberEvent, _episode(turn, turn_id=turn))
    reply = await harness.ask(memory.LeaseEvent, memory.LeaseData())
    assert isinstance(reply.data, memory.LeasedData) and reply.data.unit is not None
    return reply.data.unit


def test_lease_names_nodes_by_alias_and_consolidate_applies_one_decision() -> None:
    async def scenario() -> tuple[
        memory.UnitData,
        hsm.Event[typing.Any],
        list[tuple[typing.Any, ...]],
        list[tuple[typing.Any, ...]],
        hsm.Event[typing.Any],
    ]:
        harness = await _harness()
        await harness.attached()
        unit = await _leased(harness, "first", "second", "third")
        decided = await harness.ask(
            memory.ConsolidateEvent,
            memory.ConsolidateData(
                unit=unit.unit,
                links=(memory.LinkData(source="n1", target="n2", relation="follows", weight=0.5),),
                merges=(memory.MergeData(sources=("n1", "n2"), text="first then second"),),
                forgets=("n3",),
            ),
        )
        nodes = harness.rows("SELECT kind, text, reviewed_at IS NOT NULL FROM bot_memory_node ORDER BY seq")
        edges = harness.rows("SELECT relation, origin FROM bot_memory_edge ORDER BY relation")
        nothing = await harness.ask(memory.LeaseEvent, memory.LeaseData())
        return unit, decided, nodes, edges, nothing

    unit, decided, nodes, edges, nothing = _run(scenario)

    node_ids = {node.text for node in unit.nodes}
    assert [node.ref for node in unit.nodes] == ["n1", "n2", "n3", "n4"]
    assert [node.role for node in unit.nodes] == ["episode", "episode", "episode", "person"]
    assert all(node.participants == ("n4",) for node in unit.nodes[:3])
    assert "first" in node_ids
    assert decided.data == memory.ConsolidatedData(unit=unit.unit, linked=1, merged=1, forgotten=1)
    assert nodes == [
        ("person", "", 0),
        ("episode", "first", 1),
        ("person", "", 0),
        ("episode", "second", 1),
        ("concept", "first then second", 0),
    ]
    assert edges == [
        ("derived_from", "model"),
        ("derived_from", "model"),
        ("follows", "model"),
        ("involves_source", "stamped"),
        ("involves_source", "stamped"),
    ]
    assert nothing.data == memory.LeasedData(unit=None)


def test_a_failed_commit_changes_nothing() -> None:
    async def scenario() -> tuple[hsm.Event[typing.Any], list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        unit = await _leased(harness, "first", "second")
        with harness.engine.begin() as connection:
            _ = connection.execute(
                sql(
                    "CREATE TRIGGER refuse_review BEFORE UPDATE OF reviewed_at ON bot_memory_node "
                    "BEGIN SELECT RAISE(ABORT, 'injected'); END"
                )
            )
        reply = await harness.ask(
            memory.ConsolidateEvent,
            memory.ConsolidateData(
                unit=unit.unit,
                links=(memory.LinkData(source="n1", target="n2", relation="follows"),),
                merges=(memory.MergeData(sources=("n1", "n2"), text="merged"),),
                forgets=("n2",),
            ),
        )
        return reply, harness.rows("SELECT kind, text FROM bot_memory_node ORDER BY seq")

    reply, nodes = _run(scenario)

    assert isinstance(reply.data, memory.FailureData)
    assert reply.data.kind is memory.FailureKind.STORE_FAILED
    assert nodes == [("person", ""), ("episode", "first"), ("person", ""), ("episode", "second")]


def test_consolidate_rejects_stale_units_foreign_refs_and_person_forgets() -> None:
    async def scenario() -> list[memory.FailureKind]:
        harness = await _harness()
        await harness.attached()
        kinds: list[memory.FailureKind] = []
        stale = await harness.ask(memory.ConsolidateEvent, memory.ConsolidateData(unit="not-leased"))
        unit = await _leased(harness, "first")
        foreign = await harness.ask(memory.ConsolidateEvent, memory.ConsolidateData(unit=unit.unit, forgets=("n9",)))
        after_invalid = await harness.ask(memory.ConsolidateEvent, memory.ConsolidateData(unit=unit.unit))
        released = await harness.ask(memory.ReleaseEvent, memory.ReleaseData(unit=unit.unit, reviewed=False))
        assert released.data == memory.ReleasedData(unit=unit.unit)
        again = await harness.ask(memory.LeaseEvent, memory.LeaseData())
        assert isinstance(again.data, memory.LeasedData) and again.data.unit is not None
        person = await harness.ask(
            memory.ConsolidateEvent, memory.ConsolidateData(unit=again.data.unit.unit, forgets=("n2",))
        )
        release_stale = await harness.ask(memory.ReleaseEvent, memory.ReleaseData(unit=unit.unit, reviewed=True))
        for reply in (stale, foreign, after_invalid, person, release_stale):
            assert isinstance(reply.data, memory.FailureData), reply.data
            kinds.append(reply.data.kind)
        return kinds

    assert _run(scenario) == [
        memory.FailureKind.STALE_LEASE,
        memory.FailureKind.INVALID_INPUT,
        memory.FailureKind.STALE_LEASE,
        memory.FailureKind.INVALID_INPUT,
        memory.FailureKind.STALE_LEASE,
    ]


def test_a_decision_that_arrives_while_memory_is_busy_waits_and_is_applied() -> None:
    """Review finding: a ConsolidateEvent sent while Memory serves another request is deferred, not lost."""

    async def scenario() -> tuple[str, str]:
        blocking = BlockingIndex()
        harness = await _harness(FakeEmbedder(), index=blocking)
        await harness.attached()
        unit = await _leased(harness, "first")
        blocking.armed = True
        await harness.send(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)), "search")
        _ = await asyncio.to_thread(blocking.entered.wait, 5)
        await harness.send(memory.ConsolidateEvent, memory.ConsolidateData(unit=unit.unit), "decision")
        blocking.release.set()
        return (await harness.reply("search")).name, (await harness.reply("decision")).name

    assert _run(scenario) == (memory.FoundEvent.name, memory.ConsolidatedEvent.name)


# --- a request waiting on its embed holds nothing else ------------------------------------------

_HELD_EMBED = datetime.timedelta(seconds=5)
_SQL_ONLY = 1.0  # seconds: far below the held embed, far above any test SQL step


async def _held_harness(*turns: str) -> tuple[Harness, FakeEmbedder]:
    """A Memory with an embedder over ``turns`` remembered unindexed; its embeds hold once gated."""

    seed = await _harness()
    await seed.attached()
    for turn in turns:
        _ = await seed.ask(memory.RememberEvent, _episode(turn, turn_id=turn, participants=()))
    embedder = FakeEmbedder()
    harness = Harness(
        ctx=seed.ctx,
        memory=memory.Memory(
            engine=seed.engine, embedder=embedder, embed_timeout=_HELD_EMBED, operation_timeout=_HELD_EMBED * 2
        ),
        owner=seed.owner,
        engine=seed.engine,
    )
    harness.owner.events.clear()
    await harness.attached()
    return harness, embedder


async def _until_embedding(embedder: FakeEmbedder, calls: int) -> None:
    async with asyncio.timeout(_WAIT.total_seconds()):
        while len(embedder.calls) < calls:
            await asyncio.sleep(0.01)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("remember", memory.RememberedEvent.name),
        ("search", memory.FoundEvent.name),
        ("backfill", memory.BackfilledEvent.name),
        ("consolidate", memory.ConsolidatedEvent.name),
    ],
)
def test_a_request_waiting_on_its_embed_holds_no_unrelated_request(operation: str, expected: str) -> None:
    """Decision A: an embed is sent by message and Memory goes back to idle; its SQL step runs on the reply.

    A hung embedder delays only the request that embeds: a statement read, a recent read and a
    search that needs no embed are each answered in their own SQL time, before it, out of request order.
    """

    async def scenario() -> tuple[list[str], list[str], hsm.Event[typing.Any]]:
        harness, embedder = await _held_harness("first", "second")
        held: tuple[hsm.Event[typing.Any], object]
        if operation == "remember":
            held = (memory.RememberEvent, _episode("held", turn_id="held"))
        elif operation == "search":
            held = (memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)))
        elif operation == "backfill":
            held = (memory.BackfillEvent, memory.BackfillData(limit=8, pass_id="held"))
        else:
            unit = await harness.ask(memory.LeaseEvent, memory.LeaseData())
            assert isinstance(unit.data, memory.LeasedData) and unit.data.unit is not None
            held = (
                memory.ConsolidateEvent,
                memory.ConsolidateData(
                    unit=unit.data.unit.unit, merges=(memory.MergeData(sources=("n1", "n2"), text="merged"),)
                ),
            )
        embedder.gate = asyncio.Event()
        await harness.send(*held, "held")
        await _until_embedding(embedder, 1)
        async with asyncio.timeout(_SQL_ONLY):
            read = await harness.ask(
                harness.memory.input_event,
                memory.InputData(statements=(memory.Statement(sql="SELECT count(*) AS content FROM bot_memory_node"),)),
            )
            recent = await harness.ask(memory.RecentEvent, memory.RecentData(limit=1))
            blind = await harness.ask(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="x"),), nearest=0))
        answered_before = [event.name for event in harness.replies("held")]
        embedder.gate.set()
        return [read.name, recent.name, blind.name], answered_before, await harness.reply("held")

    answered, answered_before, held = _run(scenario)

    assert answered == [memory.Memory.output_event.name, memory.RecentOutputEvent.name, memory.FoundEvent.name]
    assert answered_before == []
    assert held.name == expected, held.data


def test_a_decision_waiting_on_its_embed_is_stale_once_its_unit_is_released() -> None:
    """While a decision waits on its merge embed the unit is decided: a second decision is stale, and a
    release in between makes the waiting decision stale instead of committing it."""

    async def scenario() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], hsm.Event[typing.Any], int]:
        harness, embedder = await _held_harness("first", "second")
        unit = await harness.ask(memory.LeaseEvent, memory.LeaseData())
        assert isinstance(unit.data, memory.LeasedData) and unit.data.unit is not None
        leased = unit.data.unit.unit
        embedder.gate = asyncio.Event()
        await harness.send(
            memory.ConsolidateEvent,
            memory.ConsolidateData(unit=leased, merges=(memory.MergeData(sources=("n1", "n2"), text="merged"),)),
            "decision",
        )
        await _until_embedding(embedder, 1)
        again = await harness.ask(memory.ConsolidateEvent, memory.ConsolidateData(unit=leased))
        released = await harness.ask(memory.ReleaseEvent, memory.ReleaseData(unit=leased, reviewed=False))
        embedder.gate.set()
        decision = await harness.reply("decision")
        (concepts,) = harness.rows("SELECT count(*) FROM bot_memory_node WHERE kind = 'concept'")
        return again, released, decision, concepts[0]

    again, released, decision, concepts = _run(scenario)

    assert isinstance(again.data, memory.FailureData) and again.data.kind is memory.FailureKind.STALE_LEASE
    assert released.name == memory.ReleasedEvent.name
    assert isinstance(decision.data, memory.FailureData), decision.data
    assert decision.data.kind is memory.FailureKind.STALE_LEASE
    assert concepts == 0


def test_detach_answers_a_request_waiting_on_its_embed_exactly_once() -> None:
    async def scenario() -> list[hsm.Event[typing.Any]]:
        harness, embedder = await _held_harness()
        embedder.gate = asyncio.Event()
        await harness.send(memory.RememberEvent, _episode("held", turn_id="held"), "held")
        await _until_embedding(embedder, 1)
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        embedder.gate.set()
        await asyncio.sleep(0.1)
        return harness.replies("held")

    (reply,) = _run(scenario)
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE


# --- exactly one reply per request --------------------------------------------------------------


def test_a_timed_out_request_is_answered_once_and_its_late_result_is_ignored() -> None:
    async def scenario() -> tuple[list[hsm.Event[typing.Any]], hsm.Event[typing.Any]]:
        blocking = BlockingIndex()
        harness = await _harness(
            FakeEmbedder(),
            index=blocking,
            embed_timeout=datetime.timedelta(milliseconds=100),
            operation_timeout=datetime.timedelta(milliseconds=300),
        )
        await harness.attached()
        blocking.armed = True
        await harness.send(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)), "slow")
        _ = await harness.reply("slow")
        blocking.release.set()
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "next")
        following = await harness.reply("next")
        await asyncio.sleep(0.05)
        return harness.replies("slow"), following

    slow, following = _run(scenario)

    assert len(slow) == 1
    assert isinstance(slow[0].data, memory.FailureData) and slow[0].data.kind is memory.FailureKind.TIMEOUT
    assert following.name == memory.RecentOutputEvent.name


def test_detach_answers_the_request_in_flight_and_every_deferred_one_exactly_once() -> None:
    async def scenario() -> collections.Counter[tuple[str, str]]:
        blocking = BlockingIndex()
        harness = await _harness(FakeEmbedder(), index=blocking)
        await harness.attached()
        blocking.armed = True
        await harness.send(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)), "in-flight")
        _ = await asyncio.to_thread(blocking.entered.wait, 5)
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "deferred-1")
        await harness.send(memory.RememberEvent, _episode(), "deferred-2")
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        blocking.release.set()
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "after-detached")
        await harness.owner.until(lambda: bool(harness.replies("after-detached")))
        await asyncio.sleep(0.1)
        ledger: collections.Counter[tuple[str, str]] = collections.Counter()
        for event in harness.owner.events:
            if event.name.startswith("bot.ability.memory.") and event.id != "attach":
                kind = event.data.kind.value if isinstance(event.data, memory.FailureData) else event.name
                ledger[(event.id, kind)] += 1
        return ledger

    assert _run(scenario) == collections.Counter(
        {
            ("in-flight", "unavailable"): 1,
            ("deferred-1", "unavailable"): 1,
            ("deferred-2", "unavailable"): 1,
            ("after-detached", "unavailable"): 1,
        }
    )


def test_stop_answers_the_request_in_flight_once() -> None:
    async def scenario() -> list[hsm.Event[typing.Any]]:
        blocking = BlockingIndex()
        harness = await _harness(FakeEmbedder(), index=blocking)
        await harness.attached()
        blocking.armed = True
        await harness.send(memory.SearchEvent, memory.SearchData(cue=(text.TextPart(text="noon"),)), "in-flight")
        _ = await asyncio.to_thread(blocking.entered.wait, 5)
        await harness.memory.stop(harness.ctx)
        blocking.release.set()
        await harness.owner.until(lambda: bool(harness.replies("in-flight")))
        await asyncio.sleep(0.1)
        return harness.replies("in-flight")

    (reply,) = _run(scenario)
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE


def test_a_request_to_a_memory_that_was_never_attached_is_unavailable() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        _ = await mosfet.started(harness.ctx, harness.memory, typing.cast(hsm.Model, harness.memory.model))
        return await harness.ask(memory.RecentEvent, memory.RecentData(limit=1))

    reply = _run(scenario)
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE


def test_a_payload_that_is_not_the_request_schema_is_invalid_input() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attached()
        await harness.send(memory.RecentEvent, memory.LeaseData(), "mistyped")
        return await harness.reply("mistyped")

    reply = _run(scenario)
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.INVALID_INPUT


# --- lifecycle ----------------------------------------------------------------------------------


def test_requests_sent_while_initializing_are_answered_once_serving() -> None:
    async def scenario() -> tuple[str, str]:
        provider = FakeAttachment(hold=True)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attach()
        await asyncio.sleep(0.05)
        state = harness.memory.state()
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "early")
        provider.release.set()
        return state, (await harness.reply("early")).name

    state, reply = _run(scenario)
    assert state == "/MemoryLifecycle/attached/behavior/initializing"
    assert reply == memory.RecentOutputEvent.name


def test_an_embedding_child_that_cannot_attach_fails_closed() -> None:
    async def scenario() -> tuple[attachment.FailedData, str, hsm.Event[typing.Any]]:
        provider = FakeAttachment(failure=attachment.FailureKind.INITIALIZATION)
        harness = await _harness(FakeEmbedder(attachments=(provider,)), started=(provider,))
        await harness.attach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachFailedEvent)))
        (failed,) = harness.owner.named(attachment.AttachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "refused")
        return failed.data, harness.memory.state(), await harness.reply("refused")

    failure, state, reply = _run(scenario)
    assert failure.kind is attachment.FailureKind.INITIALIZATION
    assert state == "/MemoryLifecycle/attached/behavior/degraded"
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE


def test_memory_serves_again_after_stop_and_reattach() -> None:
    """Stopping shuts the database worker down; a restarted Memory serves on a new one, from the same database."""

    async def scenario() -> tuple[str, str]:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode())
        await harness.detach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.DetachedEvent)))
        await harness.memory.stop(harness.ctx)
        harness.owner.events.clear()
        await harness.attached()
        remembered = await harness.ask(memory.RememberEvent, _episode("after restart", turn_id="turn-2"))
        transacted = await harness.ask(
            harness.memory.input_event,
            memory.InputData(statements=(memory.Statement(sql="SELECT count(*) AS content FROM bot_memory_node"),)),
        )
        return remembered.name, transacted.name

    remembered, transacted = _run(scenario)
    assert remembered == memory.RememberedEvent.name
    assert transacted == memory.Memory.output_event.name


def test_the_lifecycle_declares_the_composite_states_of_memory() -> None:
    model = memory.Memory.model
    assert model is not None
    for state in ("opening", "initializing", "serving/idle", "serving/busy", "detaching", "degraded"):
        assert f"/MemoryLifecycle/attached/behavior/{state}" in model.members


def test_only_the_consolidation_decision_is_offered_to_a_model() -> None:
    async def scenario() -> list[str]:
        harness = await _harness()
        await harness.attached()
        return [event.name for event in processing.enabled_call_events(harness.memory)]

    assert _run(scenario) == [memory.ConsolidateEvent.name]


def test_constructor_requires_positive_timeouts() -> None:
    """No step waits on an embed, so the SQL step bound need not exceed the embed bound."""

    _ = memory.Memory(embed_timeout=datetime.timedelta(seconds=10), operation_timeout=datetime.timedelta(seconds=1))
    with pytest.raises(ValueError, match="positive"):
        _ = memory.Memory(embed_timeout=datetime.timedelta(0))
    with pytest.raises(ValueError, match="positive"):
        _ = memory.Memory(operation_timeout=datetime.timedelta(0))


# --- opening (DSL-29) ---------------------------------------------------------------------------


def _unversioned_database(tmp_path: pathlib.Path) -> str:
    """A file holding a memory table but no version table: the migrations refuse to touch it."""

    path = tmp_path / "unversioned.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    return str(path)


class GatedOpenMemory(memory.Memory):
    """A Memory whose database open waits for the test to release it."""

    release: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    @typing.override
    async def _open_engine(self) -> Engine:
        _ = await self.release.wait()
        return await super()._open_engine()


async def _owned(store: memory.Memory) -> Harness:
    """A harness around a Memory the test constructed itself (its own database, or an open double)."""

    ctx = environment.Environment()
    owner = Owner()
    _ = await mosfet.started(ctx, owner, Owner.model)
    return Harness(ctx=ctx, memory=store, owner=owner, engine=memory.store.open_sqlite_engine())


def test_memory_opens_its_database_on_attach_not_on_construction(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "memory.db"

    async def scenario() -> tuple[bool, bool, str]:
        harness = await _owned(memory.Memory(database=str(path)))
        constructed = path.exists()
        _ = await mosfet.started(harness.ctx, harness.memory, typing.cast(hsm.Model, harness.memory.model))
        started = path.exists()
        await harness.attached()
        remembered = await harness.ask(memory.RememberEvent, _episode())
        await harness.memory.stop(harness.ctx)
        return constructed, started, remembered.name

    constructed, started, remembered = _run(scenario)
    assert (constructed, started) == (False, False)
    assert path.exists()
    assert remembered == memory.RememberedEvent.name


def test_a_database_memory_cannot_open_fails_its_attach_and_it_serves_nothing(tmp_path: pathlib.Path) -> None:
    database = _unversioned_database(tmp_path)

    async def scenario() -> tuple[attachment.FailedData, str, hsm.Event[typing.Any]]:
        harness = await _owned(memory.Memory(database=database))
        await harness.attach()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachFailedEvent)))
        (failed,) = harness.owner.named(attachment.AttachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "refused")
        reply = await harness.reply("refused")
        state = harness.memory.state()
        await harness.memory.stop(harness.ctx)
        return failed.data, state, reply

    failure, state, reply = _run(scenario)
    assert failure.kind is attachment.FailureKind.INITIALIZATION
    assert "MigrationError" in failure.message
    assert state == "/MemoryLifecycle/attached/behavior/degraded"
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE


def test_requests_sent_while_opening_are_answered_once_open() -> None:
    async def scenario() -> tuple[str, str]:
        store = GatedOpenMemory()
        harness = await _owned(store)
        await harness.attach()
        await asyncio.sleep(0.05)
        state = store.state()
        await harness.send(memory.RecentEvent, memory.RecentData(limit=1), "early")
        store.release.set()
        reply = await harness.reply("early")
        await store.stop(harness.ctx)
        return state, reply.name

    state, reply = _run(scenario)
    assert state == "/GatedOpenMemoryLifecycle/attached/behavior/opening"
    assert reply == memory.RecentOutputEvent.name


def test_a_database_that_does_not_open_within_the_attach_timeout_fails_the_attach() -> None:
    async def scenario() -> tuple[attachment.FailedData, str]:
        store = GatedOpenMemory()
        harness = await _owned(store)
        _ = await store.attach(
            harness.ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=harness.owner, timeout=datetime.timedelta(milliseconds=50)), "attach"
            ),
        )
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachFailedEvent)))
        (failed,) = harness.owner.named(attachment.AttachFailedEvent)
        assert isinstance(failed.data, attachment.FailedData)
        state = store.state()
        await store.stop(harness.ctx)
        return failed.data, state

    failure, state = _run(scenario)
    assert failure.kind is attachment.FailureKind.TIMEOUT
    assert state == "/GatedOpenMemoryLifecycle/attached/behavior/degraded"


def _memory_threads() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name.startswith("memory_")}


def test_the_database_worker_lives_from_start_to_stop() -> None:
    """DSL-29b: no worker thread before start; the one start made exits after stop."""

    async def scenario() -> tuple[set[threading.Thread], set[threading.Thread]]:
        existing = _memory_threads()
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode())
        workers = _memory_threads() - existing
        await harness.memory.stop(harness.ctx)
        for worker in workers:
            await asyncio.to_thread(worker.join, 5)
        return workers, {worker for worker in workers if worker.is_alive()}

    workers, alive = _run(scenario)
    assert len(workers) == 1
    assert alive == set()


def test_a_stopped_memory_closes_the_database_it_opened_once_its_worker_is_done(tmp_path: pathlib.Path) -> None:
    """D-W7-9: the engine is disposed on the database worker after its last job, so nothing it opened stays open."""

    async def run(disposed: list[Engine]) -> tuple[int, int]:
        existing = _memory_threads()
        store = memory.Memory(database=str(tmp_path / "memory.db"))
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        _ = await memory.transact(
            ctx, store, memory.InputData(statements=(memory.Statement(sql="SELECT count(*) FROM bot_memory_node"),))
        )
        workers = _memory_threads() - existing
        await hsm.stop(store, ctx)
        for worker in workers:
            await asyncio.to_thread(worker.join, 5)
        return len(workers), len(disposed)

    with disposed_engines() as disposed:
        assert asyncio.run(run(disposed)) == (1, 1)


# --- persons (W8-5, W8-6) -----------------------------------------------------------------------


def _voice(id: str, *, own: bool = False) -> mosfet.Participant:
    return mosfet.Participant(ref=f"voice:{id}", role=mosfet.ParticipantRole.SOURCE, own=own)


def _links(harness: Harness) -> list[tuple[typing.Any, ...]]:
    """Every ref with the key of its person and how it was linked."""

    return harness.rows(
        "SELECT r.ref, n.key, r.origin FROM bot_person_ref r JOIN bot_memory_node n ON n.seq = r.person_seq "
        "ORDER BY r.ref"
    )


def _by_ref(links: list[tuple[typing.Any, ...]]) -> dict[str, tuple[str, str]]:
    """``_links`` as each ref's (person key, origin)."""

    return {typing.cast(str, row[0]): (typing.cast(str, row[1]), typing.cast(str, row[2])) for row in links}


def _involved(harness: Harness) -> list[tuple[typing.Any, ...]]:
    """Every involves link as (episode turn text, person key)."""

    return harness.rows(
        "SELECT e.text, p.key FROM bot_memory_edge x JOIN bot_memory_node e ON e.seq = x.source_seq "
        "JOIN bot_memory_node p ON p.seq = x.target_seq WHERE x.relation LIKE 'involves_%' ORDER BY e.seq, p.key"
    )


async def _person(harness: Harness, data: object) -> memory.PersonDoneData:
    reply = await harness.ask(memory.PersonEvent, data)
    assert isinstance(reply.data, memory.PersonDoneData), reply.data
    return reply.data


async def _refused(harness: Harness, data: object) -> memory.FailureKind:
    reply = await harness.ask(memory.PersonEvent, data)
    assert isinstance(reply.data, memory.FailureData), reply.data
    return reply.data.kind


def _is_minted(person: str) -> bool:
    return len(person) == 32 and all(digit in "0123456789abcdef" for digit in person)


_NOBODY: typing.Final = "0123456789abcdef0123456789abcdef"
"""A well-formed person id memory never minted."""


def test_self_is_a_person_from_the_moment_memory_opens() -> None:
    async def scenario() -> tuple[list[tuple[typing.Any, ...]], memory.PersonDoneData]:
        harness = await _harness()
        await harness.attached()
        return harness.rows("SELECT key, text FROM bot_memory_node"), await _person(harness, memory.PersonListData())

    nodes, listed = _run(scenario)
    assert nodes == [("self", "")]
    assert listed.persons == (memory.Person(id=memory.SELF),)


def test_two_episodes_from_one_voice_share_one_provisional_person() -> None:
    async def scenario() -> tuple[list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        for turn in ("one", "two"):
            _ = await harness.ask(memory.RememberEvent, _episode(turn, turn_id=turn, participants=(_voice("a"),)))
        return _links(harness), _involved(harness)

    links, involved = _run(scenario)
    ((ref, person, origin),) = links
    assert (ref, origin) == ("voice:a", "stamped")
    assert _is_minted(person)
    assert involved == [("one", person), ("two", person)]


def test_an_own_voice_is_self_and_a_provisional_own_ref_merges_into_self() -> None:
    async def scenario() -> tuple[list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]], int]:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode("echo", turn_id="1", participants=(_voice("echo"),)))
        _ = await harness.ask(
            memory.RememberEvent, _episode("own", turn_id="2", participants=(_voice("echo", own=True),))
        )
        _ = await harness.ask(
            memory.RememberEvent, _episode("spoken", turn_id="3", participants=(_voice("mouth", own=True),))
        )
        (persons,) = harness.rows("SELECT count(*) FROM bot_memory_node WHERE kind = 'person'")
        return _links(harness), _involved(harness), int(persons[0])

    links, involved, persons = _run(scenario)
    assert links == [("voice:echo", "self", "own"), ("voice:mouth", "self", "own")]
    assert involved == [("echo", "self"), ("own", "self"), ("spoken", "self")]
    assert persons == 1


def test_an_own_ref_linked_to_another_person_stays_theirs_and_is_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a corrupt store links an own id elsewhere (registries refuse it); memory keeps the link and counts."""

    from tests import metric_reader

    async def scenario(reader: metric_reader.Reader) -> tuple[list[tuple[typing.Any, ...]], float]:
        harness = await _harness()
        await harness.attached()
        sam = await _person(harness, memory.PersonLinkData(ref="voice:a", name="Sam"))
        assert sam.person is not None
        _ = await harness.ask(memory.RememberEvent, _episode("me?", participants=(_voice("a", own=True),)))
        conflicts = sum(point.value for point in reader.points("bot.memory.person.conflict.count"))
        return _links(harness), conflicts

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        links, conflicts = asyncio.run(scenario(reader))

    ((ref, person, origin),) = links
    assert (ref, origin) == ("voice:a", "model")
    assert _is_minted(person)
    assert conflicts == 1


def test_link_merges_a_provisional_person_and_re_points_its_involves_edges() -> None:
    async def scenario() -> tuple[memory.PersonDoneData, list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode("hello", participants=(_voice("a"),)))
        gabe = await _person(harness, memory.PersonLinkData(ref="+15550001", name="Gabe"))
        assert gabe.person is not None
        linked = await _person(harness, memory.PersonLinkData(ref="voice:a", person=gabe.person.id))
        persons = harness.rows("SELECT key FROM bot_memory_node WHERE kind = 'person' ORDER BY seq")
        assert _by_ref(_links(harness))["voice:a"] == (gabe.person.id, "model")
        return linked, _involved(harness), persons

    linked, involved, persons = _run(scenario)
    assert linked.op == "link"
    assert linked.person is not None
    gabe = linked.person.id
    assert (linked.person.name, set(linked.person.refs)) == ("Gabe", {"voice:a", "+15550001"})
    assert involved == [("hello", gabe)]
    assert persons == [("self",), (gabe,)]


def test_one_person_has_many_refs_and_each_ref_is_one_persons() -> None:
    async def scenario() -> tuple[memory.PersonDoneData, memory.FailureKind, list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        sam = await _person(harness, memory.PersonLinkData(ref="voice:a", name="Sam"))
        assert sam.person is not None and _is_minted(sam.person.id) and sam.person.name == "Sam"
        both = await _person(harness, memory.PersonLinkData(ref="face:b", person=sam.person.id))
        gabe = await _person(harness, memory.PersonLinkData(ref="+15550001", name="Gabe"))
        assert gabe.person is not None
        taken = await _refused(harness, memory.PersonLinkData(ref="voice:a", person=gabe.person.id))
        return both, taken, _links(harness)

    both, taken, links = _run(scenario)
    assert both.person is not None and set(both.person.refs) == {"voice:a", "face:b"}
    assert taken is memory.FailureKind.CONFLICT
    by_ref = _by_ref(links)
    assert by_ref["face:b"] == by_ref["voice:a"] == (both.person.id, "model")


def test_linking_to_the_bot_itself_or_an_unknown_person_is_refused() -> None:
    async def scenario() -> tuple[memory.FailureKind, memory.FailureKind, list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        itself = await _refused(harness, memory.PersonLinkData(ref="voice:a", person=memory.SELF))
        unknown = await _refused(harness, memory.PersonLinkData(ref="voice:a", person=_NOBODY))
        return itself, unknown, _links(harness)

    itself, unknown, links = _run(scenario)
    # Only the producers that know the bot's own voice make a ref its own: anyone else's is a conflict.
    assert (itself, unknown) == (memory.FailureKind.CONFLICT, memory.FailureKind.NOT_FOUND)
    assert links == []


def test_the_bot_names_a_person_it_learned_never_itself() -> None:
    async def scenario() -> tuple[memory.PersonDoneData, list[memory.FailureKind], list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        stranger = await _person(harness, memory.PersonLinkData(ref="voice:s"))
        assert stranger.person is not None and stranger.person.name is None
        named = await _person(harness, memory.PersonNameData(person=stranger.person.id, name="Sam"))
        refused = [
            await _refused(harness, memory.PersonNameData(person=memory.SELF, name="Willy")),
            await _refused(harness, memory.PersonNameData(person=_NOBODY, name="Sam")),
        ]
        return named, refused, harness.rows("SELECT key, text FROM bot_memory_node WHERE kind = 'person' ORDER BY seq")

    named, refused, persons = _run(scenario)
    assert named.person is not None and (named.person.name, named.person.refs) == ("Sam", ("voice:s",))
    # Every ref of the named person, so People keeps each of their prints (D-W8-18).
    assert named.refs == ("voice:s",)
    assert refused == [memory.FailureKind.CONFLICT, memory.FailureKind.NOT_FOUND]
    assert [text for _, text in persons] == ["", "Sam"]


def test_unlink_gives_the_ref_a_fresh_provisional_person_and_keeps_past_attributions() -> None:
    async def scenario() -> tuple[str, list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]], list[str]]:
        harness = await _harness()
        await harness.attached()
        sam = await _person(harness, memory.PersonLinkData(ref="voice:a", name="Sam"))
        assert sam.person is not None
        _ = await harness.ask(memory.RememberEvent, _episode("before", participants=(_voice("a"),)))
        unlinked = await _person(harness, memory.PersonUnlinkData(ref="voice:a"))
        assert unlinked.op == "unlink" and unlinked.person is not None
        assert unlinked.person.id != sam.person.id and unlinked.person.refs == ("voice:a",)
        _ = await harness.ask(memory.RememberEvent, _episode("after", turn_id="2", participants=(_voice("a"),)))
        refused = [(await _refused(harness, memory.PersonUnlinkData(ref="voice:nobody"))).value]
        return sam.person.id, _links(harness), _involved(harness), refused

    sam, links, involved, refused = _run(scenario)
    fresh = next(person for ref, person, _ in links if ref == "voice:a")
    assert fresh != sam and _is_minted(fresh)
    assert ("voice:a", fresh, "model") not in links
    assert involved == [("before", sam), ("after", fresh)]
    assert refused == ["not_found"]


def test_an_own_ref_cannot_be_unlinked_from_self() -> None:
    async def scenario() -> tuple[memory.FailureKind, list[tuple[typing.Any, ...]]]:
        harness = await _harness()
        await harness.attached()
        _ = await harness.ask(memory.RememberEvent, _episode("spoken", participants=(_voice("mouth", own=True),)))
        return await _refused(harness, memory.PersonUnlinkData(ref="voice:mouth")), _links(harness)

    refused, links = _run(scenario)
    assert refused is memory.FailureKind.CONFLICT
    assert links == [("voice:mouth", "self", "own")]


def test_forget_deletes_the_person_its_refs_and_its_involves_links_and_keeps_the_episodes() -> None:
    async def scenario() -> tuple[
        memory.PersonDoneData, list[memory.FailureKind], list[tuple[typing.Any, ...]], list[tuple[typing.Any, ...]]
    ]:
        harness = await _harness()
        await harness.attached()
        sam = await _person(harness, memory.PersonLinkData(ref="voice:a", name="Sam"))
        assert sam.person is not None
        _ = await _person(harness, memory.PersonLinkData(ref="face:b", person=sam.person.id))
        _ = await harness.ask(
            memory.RememberEvent,
            _episode("Sam waved", participants=(_voice("a"), mosfet.Participant(ref="face:b", role=_ADA.role))),
        )
        forgotten = await _person(harness, memory.PersonForgetData(person=sam.person.id))
        refused = [
            await _refused(harness, memory.PersonForgetData(person=memory.SELF)),
            await _refused(harness, memory.PersonForgetData(person=sam.person.id)),
        ]
        episodes = harness.rows("SELECT text FROM bot_memory_node WHERE kind = 'episode'")
        return forgotten, refused, episodes, _involved(harness)

    forgotten, refused, episodes, involved = _run(scenario)
    assert forgotten.op == "forget"
    assert set(forgotten.refs) == {"voice:a", "face:b"}
    assert refused == [memory.FailureKind.CONFLICT, memory.FailureKind.NOT_FOUND]
    assert episodes == [("Sam waved",)]
    assert involved == []


def test_person_statements_to_a_memory_that_is_not_attached_are_unavailable() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        _ = await mosfet.started(harness.ctx, harness.memory, typing.cast(hsm.Model, harness.memory.model))
        return await harness.ask(memory.PersonEvent, memory.PersonListData())

    reply = _run(scenario)
    assert isinstance(reply.data, memory.FailureData) and reply.data.kind is memory.FailureKind.UNAVAILABLE
