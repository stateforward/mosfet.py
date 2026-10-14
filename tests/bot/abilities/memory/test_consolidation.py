"""HSM runtime tests for ``memory.Consolidation`` over a real ``memory.Memory`` peer and a fake processor."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
import pytest
from sqlalchemy import text as sql
from sqlalchemy.engine import Engine

from mosfet import environment
from mosfet.abilities import ability, embedding, memory, processing
from mosfet.abilities.language import text
from mosfet.protocols import attachment

_WAIT = 3.0
_ADA = mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE)
_SPACE = embedding.Space(model="test/embedder", dimensions=2)


class Decider(processing.Processor):
    """Fake model: decides each unit with ``decide``; ``gate`` holds a call until it is set."""

    calls: list[processing.InputData]

    def __init__(
        self,
        decide: collections.abc.Callable[[memory.UnitData], processing.Events],
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.decide = decide
        self.gate = gate
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if self.gate is not None:
            _ = await self.gate.wait()
        unit = input.input
        assert isinstance(unit, memory.UnitData)
        return self.decide(unit)


def _consolidate(decision: memory.ConsolidateData) -> processing.SelectedEvent:
    return processing.SelectedEvent(
        event=memory.ConsolidateEvent.name, data=decision.model_dump(mode="json", exclude_none=True)
    )


def _link_to_person(unit: memory.UnitData) -> processing.Events:
    episode = next(node for node in unit.nodes if node.role == "episode")
    return (
        _consolidate(
            memory.ConsolidateData(
                unit=unit.unit,
                links=(memory.LinkData(source=episode.ref, target=episode.participants[0], relation="mentions"),),
            )
        ),
    )


def _leave(unit: memory.UnitData) -> processing.Events:
    del unit
    return ()


def _fail(unit: memory.UnitData) -> processing.Events:
    del unit
    raise RuntimeError("the model is unreachable")


class Owner(hsm.Instance):
    """Attachment owner that records every event delivered to it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: Owner, event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ConsolidationTestOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def outcomes(self) -> list[memory.UnitOutcomeData]:
        return [
            event.data
            for event in self.events
            if event.name == memory.UnitOutcomeEvent.name and isinstance(event.data, memory.UnitOutcomeData)
        ]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


class RecordingMemory(memory.Memory):
    """Memory that records the consolidation decisions delivered to it (test-only observation)."""

    decisions: list[hsm.Event[typing.Any]]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.decisions = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == memory.ConsolidateEvent.name:
            self.decisions.append(event)
        return super().dispatch(ctx, event)


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    memory: RecordingMemory
    consolidation: memory.Consolidation
    owner: Owner
    memory_owner: Owner
    engine: Engine
    decider: Decider

    async def attach(self) -> None:
        _ = await self.memory.attach(
            self.ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=self.memory_owner))
        )
        await self.memory_owner.until(
            lambda: any(e.name == attachment.AttachCompleteEvent.name for e in self.memory_owner.events)
        )
        _ = await self.consolidation.attach(
            self.ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), "attach")
        )
        await self.owner.until(lambda: any(e.name == attachment.AttachCompleteEvent.name for e in self.owner.events))

    async def detach(self) -> None:
        _ = await self.consolidation.detach(
            self.ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), "detach")
        )

    async def notify(self, notification: hsm.Event[typing.Any], data: object) -> None:
        _ = await hsm.dispatch(
            self.ctx,
            self.consolidation,
            dataclasses.replace(
                notification.with_data(data), source=hsm.id(self.owner), target=hsm.id(self.consolidation)
            ),
        )

    async def rest(self) -> None:
        await self.notify(memory.RestEvent, memory.RestData())

    async def wake(self) -> None:
        await self.notify(memory.WakeEvent, memory.WakeData())

    async def remember(self, words: str, turn_id: str) -> None:
        reply = await ability.run_terminal_operation(
            self.ctx,
            child=self.memory,
            request=memory.RememberEvent.with_data_and_id(
                memory.EpisodeData(
                    turn_id=turn_id,
                    parts=(text.TextPart(text=words),),
                    participants=(_ADA,),
                    outcome=memory.Outcome.COMPLETED,
                ),
                f"remember-{turn_id}",
            ),
            terminals=(memory.RememberedEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        assert isinstance(reply.data, memory.RememberedData), reply.data

    async def settled(self, count: int) -> list[memory.UnitOutcomeData]:
        await self.owner.until(lambda: len(self.owner.outcomes()) >= count)
        await self.until_state("/resting/settled")
        return self.owner.outcomes()

    async def until_state(self, suffix: str) -> None:
        async with asyncio.timeout(_WAIT):
            while not self.consolidation.state().endswith(suffix):
                await asyncio.sleep(0.01)

    def rows(self, query: str) -> list[tuple[typing.Any, ...]]:
        with self.engine.begin() as connection:
            return [tuple(row) for row in connection.execute(sql(query)).all()]

    def reviewed(self) -> list[tuple[str, int]]:
        return [
            (str(words), int(reviewed))
            for words, reviewed in self.rows(
                "SELECT text, reviewed_at IS NOT NULL FROM bot_memory_node WHERE kind = 'episode' ORDER BY seq"
            )
        ]


async def _harness(
    decide: collections.abc.Callable[[memory.UnitData], processing.Events],
    *,
    gate: asyncio.Event | None = None,
    embedder: embedding.Embedder | None = None,
    episodes: int = 1,
    timeout: datetime.timedelta = datetime.timedelta(seconds=30),
) -> Harness:
    ctx = environment.Environment()
    owner, memory_owner = Owner(), Owner()
    for actor in (owner, memory_owner):
        _ = await mosfet.started(ctx, actor, Owner.model)
    engine = memory.store.open_sqlite_engine()
    store = RecordingMemory(
        engine=engine,
        embedder=embedder,
        embed_timeout=datetime.timedelta(seconds=2),
        operation_timeout=datetime.timedelta(seconds=5),
    )
    decider = Decider(decide, gate=gate)
    consolidation = memory.Consolidation(
        memory=store,
        processor=decider,
        lease=memory.LeaseData(episodes=episodes, neighbors=0),
        timeout=timeout,
    )
    return Harness(
        ctx=ctx,
        memory=store,
        consolidation=consolidation,
        owner=owner,
        memory_owner=memory_owner,
        engine=engine,
        decider=decider,
    )


def _run[T](scenario: collections.abc.Callable[[], collections.abc.Coroutine[typing.Any, typing.Any, T]]) -> T:
    return asyncio.run(scenario())


def _outcomes(outcomes: list[memory.UnitOutcomeData]) -> list[str]:
    return [outcome.outcome for outcome in outcomes]


# --- lifecycle and contract ---------------------------------------------------------------------


def test_consolidation_attaches_awake_and_declares_its_rest_topology() -> None:
    async def scenario() -> str:
        harness = await _harness(_leave)
        await harness.attach()
        return harness.consolidation.state()

    assert _run(scenario) == "/ConsolidationLifecycle/attached/behavior/awake"
    model = memory.Consolidation.model
    assert model is not None
    behavior = "/ConsolidationLifecycle/attached/behavior"
    for state in (
        "initializing",
        "awake",
        "resting",
        "resting/backfilling",
        "resting/leasing",
        "resting/deciding",
        "resting/committing",
        "resting/reviewing",
        "resting/returning",
        "resting/settled",
        "detaching",
        "degraded",
    ):
        assert f"{behavior}/{state}" in model.members, state
    assert memory.Consolidation.detached_notifications == (memory.RestEvent, memory.WakeEvent)
    # Resting and waking are notifications from the owner, never tools a model can call.
    for notification in (memory.RestEvent, memory.WakeEvent, memory.UnitOutcomeEvent):
        assert notification.kind != processing.EventKind


def test_the_model_is_told_its_capacities_and_offered_only_the_consolidation_decision() -> None:
    async def scenario() -> list[processing.InputData]:
        harness = await _harness(_link_to_person, episodes=1)
        await harness.attach()
        for index in range(3):
            await harness.remember(f"episode {index}", f"turn-{index}")
        await harness.rest()
        _ = await harness.settled(4)
        return harness.decider.calls

    calls = _run(scenario)

    assert len(calls) == 3
    for frame in calls:
        # Offered from Memory's declared model, whatever state Memory happens to be in.
        assert [event.name for event in frame.schemas] == [memory.ConsolidateEvent.name]
        assert frame.instructions == memory.consolidation.INSTRUCTIONS
    instructions = memory.consolidation.INSTRUCTIONS
    for capacity in ("link", "merge", "forget", memory.ConsolidateEvent.name):
        assert capacity in instructions


# --- one rest -----------------------------------------------------------------------------------


def test_a_rest_consolidates_unit_by_unit_until_nothing_is_left() -> None:
    async def scenario() -> tuple[list[memory.UnitOutcomeData], list[tuple[str, int]], int, list[str], str]:
        harness = await _harness(_link_to_person, episodes=1)
        await harness.attach()
        for index in range(3):
            await harness.remember(f"episode {index}", f"turn-{index}")
        await harness.rest()
        outcomes = await harness.settled(4)
        (linked,) = harness.rows("SELECT count(*) FROM bot_memory_edge WHERE origin = 'model'")
        return (
            outcomes,
            harness.reviewed(),
            int(linked[0]),
            [event.source for event in harness.memory.decisions],
            hsm.id(harness.consolidation),
        )

    outcomes, reviewed, linked, sources, consolidation_id = _run(scenario)

    # Each commit came back to Consolidation itself (not its processing child) well before its 30 s timeout.
    assert _outcomes(outcomes) == ["committed", "committed", "committed", "exhausted"]
    assert all(outcome.unit for outcome in outcomes[:3]) and outcomes[3].unit is None
    assert reviewed == [("episode 0", 1), ("episode 1", 1), ("episode 2", 1)]
    assert linked == 3
    assert sources == [consolidation_id] * 3


def test_a_unit_the_bot_leaves_as_it_is_is_reviewed_and_not_offered_again() -> None:
    async def scenario() -> tuple[list[memory.UnitOutcomeData], list[tuple[str, int]], int]:
        harness = await _harness(_leave)
        await harness.attach()
        await harness.remember("nothing to add", "turn-1")
        await harness.rest()
        _ = await harness.settled(2)
        await harness.wake()
        await harness.until_state("/awake")
        await harness.rest()
        outcomes = await harness.settled(3)
        return outcomes, harness.reviewed(), len(harness.decider.calls)

    outcomes, reviewed, calls = _run(scenario)

    assert _outcomes(outcomes) == ["reviewed", "exhausted", "exhausted"]
    assert reviewed == [("nothing to add", 1)]
    assert calls == 1


def test_forgetting_deletes_an_episode_for_good() -> None:
    def forget_second(unit: memory.UnitData) -> processing.Events:
        episodes = [node.ref for node in unit.nodes if node.role == "episode"]
        return (_consolidate(memory.ConsolidateData(unit=unit.unit, forgets=(episodes[1],))),)

    async def scenario() -> tuple[list[memory.UnitOutcomeData], list[tuple[str, int]], list[tuple[typing.Any, ...]]]:
        harness = await _harness(forget_second, episodes=2)
        await harness.attach()
        await harness.remember("keep this", "turn-1")
        await harness.remember("drop this", "turn-2")
        await harness.rest()
        outcomes = await harness.settled(2)
        orphans = harness.rows(
            "SELECT 'part', count(*) FROM bot_memory_part WHERE node_seq NOT IN (SELECT seq FROM bot_memory_node) "
            "UNION ALL SELECT 'edge', count(*) FROM bot_memory_edge "
            "WHERE source_seq NOT IN (SELECT seq FROM bot_memory_node) "
            "OR target_seq NOT IN (SELECT seq FROM bot_memory_node)"
        )
        return outcomes, harness.reviewed(), orphans

    outcomes, reviewed, orphans = _run(scenario)

    assert _outcomes(outcomes) == ["committed", "exhausted"]
    assert reviewed == [("keep this", 1)]
    assert orphans == [("part", 0), ("edge", 0)]


def test_an_invalid_decision_fails_once_and_the_unit_is_not_offered_again() -> None:
    def forget_a_person(unit: memory.UnitData) -> processing.Events:
        person = next(node.ref for node in unit.nodes if node.role == "person")
        return (_consolidate(memory.ConsolidateData(unit=unit.unit, forgets=(person,))),)

    async def scenario() -> tuple[list[memory.UnitOutcomeData], list[tuple[str, int]], int]:
        harness = await _harness(forget_a_person)
        await harness.attach()
        await harness.remember("Ada called", "turn-1")
        await harness.rest()
        outcomes = await harness.settled(2)
        (persons,) = harness.rows("SELECT count(*) FROM bot_memory_node WHERE kind = 'person'")
        return outcomes, harness.reviewed(), int(persons[0])

    outcomes, reviewed, persons = _run(scenario)

    assert _outcomes(outcomes) == ["failed", "exhausted"]
    assert reviewed == [("Ada called", 1)]
    assert persons == 1


def test_a_decision_selected_twice_commits_once() -> None:
    def twice(unit: memory.UnitData) -> processing.Events:
        return _link_to_person(unit) * 2

    async def scenario() -> tuple[list[memory.UnitOutcomeData], int, int]:
        harness = await _harness(twice)
        await harness.attach()
        await harness.remember("say it twice", "turn-1")
        await harness.rest()
        outcomes = await harness.settled(2)
        (linked,) = harness.rows("SELECT count(*) FROM bot_memory_edge WHERE origin = 'model'")
        return outcomes, int(linked[0]), len(harness.memory.decisions)

    outcomes, linked, decisions = _run(scenario)

    assert decisions == 2
    assert _outcomes(outcomes) == ["committed", "exhausted"]
    assert linked == 1


def test_a_merge_decision_selected_twice_commits_once_while_its_merge_is_embedded() -> None:
    """The first copy waits on its merge embed while Memory serves on, so the second copy is
    refused stale first; that refusal is a duplicate's, never the unit's outcome."""

    class HeldEmbedder(embedding.Embedder):
        """Embeds at once, except the merged statement, which waits until ``merge`` is set."""

        capacity: embedding.Capacity
        attachments: tuple[hsm.Instance, ...]
        merge: asyncio.Event

        def __init__(self) -> None:
            self.capacity = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)
            self.attachments = ()
            self.merge = asyncio.Event()

        @typing.override
        async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
            del ctx
            words = {part.text for item in input.items for part in item.parts if isinstance(part, text.TextPart)}
            if "Ada called twice" in words:
                _ = await self.merge.wait()
            return embedding.OutputData(
                space=_SPACE, role=input.role, vectors=tuple(embedding.Vector(values=(1.0, 0.0)) for _ in input.items)
            )

    def twice(unit: memory.UnitData) -> processing.Events:
        episodes = tuple(node.ref for node in unit.nodes if node.role == "episode")
        decision = memory.ConsolidateData(
            unit=unit.unit, merges=(memory.MergeData(sources=episodes, text="Ada called twice"),)
        )
        return (_consolidate(decision), _consolidate(decision))

    async def scenario() -> tuple[list[str], int, list[tuple[str, int]]]:
        embedder = HeldEmbedder()
        harness = await _harness(twice, embedder=embedder, episodes=2)
        await harness.attach()
        await harness.remember("Ada called", "turn-1")
        await harness.remember("Ada called again", "turn-2")
        await harness.rest()
        async with asyncio.timeout(_WAIT):
            while len(harness.memory.decisions) < 2:
                await asyncio.sleep(0.01)
        # Both copies reached Memory: the first holds the lease decided, the second was refused.
        await asyncio.sleep(0.05)
        embedder.merge.set()
        outcomes = await harness.settled(2)
        (concepts,) = harness.rows("SELECT count(*) FROM bot_memory_node WHERE kind = 'concept'")
        return _outcomes(outcomes), int(concepts[0]), harness.reviewed()

    outcomes, concepts, reviewed = _run(scenario)

    assert outcomes == ["committed", "exhausted"]
    assert concepts == 1
    assert reviewed == [("Ada called", 1), ("Ada called again", 1)]


def test_a_unit_that_cannot_be_decided_waits_one_rest_then_leasing_moves_past_it() -> None:
    """Poison unit: the same unit failing on two leases in a row is left reviewed, never retried forever."""

    async def scenario() -> tuple[list[str], list[tuple[str, int]], list[str], list[tuple[str, int]], int]:
        harness = await _harness(_fail)
        await harness.attach()
        await harness.remember("first", "turn-1")
        await harness.remember("second", "turn-2")
        await harness.rest()
        first_rest = _outcomes(await harness.settled(1))
        after_first = harness.reviewed()
        await harness.wake()
        await harness.until_state("/awake")
        await harness.rest()
        second_rest = _outcomes(await harness.settled(3))
        return first_rest, after_first, second_rest, harness.reviewed(), len(harness.decider.calls)

    first_rest, after_first, second_rest, after_second, calls = _run(scenario)

    # First rest: the unit is given back unreviewed and the rest settles (no tight loop on it).
    assert first_rest == ["failed"]
    assert after_first == [("first", 0), ("second", 0)]
    # Second rest: it fails again, is left reviewed, and leasing moves on to the next unit.
    assert second_rest == ["failed", "failed", "failed"]
    assert after_second == [("first", 1), ("second", 0)]
    assert calls == 3


def test_backfill_ends_with_unindexable_nodes_and_leasing_follows() -> None:
    class Embedder(embedding.Embedder):
        def __init__(self) -> None:
            self.capacity = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)
            self.attachments = ()
            self.down = True

        async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
            del ctx
            if self.down:
                raise embedding.Error(embedding.FailureKind.UNAVAILABLE, "down")
            for index, item in enumerate(input.items):
                words = " ".join(part.text for part in item.parts if isinstance(part, text.TextPart))
                if "long" in words:
                    raise embedding.Error(embedding.FailureKind.TOO_LONG, "too long", item_index=index)
            return embedding.OutputData(
                space=_SPACE, role=input.role, vectors=tuple(embedding.Vector(values=(1.0, 0.0)) for _ in input.items)
            )

    async def scenario() -> tuple[list[str], int, int]:
        embedder = Embedder()
        harness = await _harness(_leave, embedder=embedder, episodes=8)
        await harness.attach()
        for index, words in enumerate(("short a", "long b", "short c", "long d", "short e")):
            await harness.remember(words, f"turn-{index}")
        embedder.down = False
        await harness.rest()
        outcomes = _outcomes(await harness.settled(2))
        (vectors,) = harness.rows("SELECT count(*) FROM bot_memory_vector")
        (permanent,) = harness.rows("SELECT count(*) FROM bot_memory_index_attempt WHERE permanent = 1")
        return outcomes, int(vectors[0]), int(permanent[0])

    outcomes, vectors, permanent = _run(scenario)

    assert outcomes == ["reviewed", "exhausted"]
    assert vectors == 3
    assert permanent == 2


# --- interruption -------------------------------------------------------------------------------


def test_waking_mid_decision_cancels_it_and_leaves_the_unit_for_a_later_rest() -> None:
    async def scenario() -> tuple[list[str], list[tuple[str, int]], int, list[str], list[tuple[str, int]]]:
        gate = asyncio.Event()
        harness = await _harness(_link_to_person, gate=gate)
        await harness.attach()
        await harness.remember("held", "turn-1")
        await harness.rest()
        await harness.until_state("/resting/deciding")
        async with asyncio.timeout(_WAIT):
            while not harness.decider.calls:
                await asyncio.sleep(0.01)
        await harness.wake()
        await harness.owner.until(lambda: bool(harness.owner.outcomes()))
        interrupted = _outcomes(harness.owner.outcomes())
        await harness.until_state("/awake")
        gate.set()
        await asyncio.sleep(0.1)
        (linked,) = harness.rows("SELECT count(*) FROM bot_memory_edge WHERE origin = 'model'")
        after_wake = harness.reviewed()
        await harness.rest()
        later = _outcomes(await harness.settled(3))
        return interrupted, after_wake, int(linked[0]), later, harness.reviewed()

    interrupted, after_wake, linked, later, reviewed = _run(scenario)

    assert interrupted == ["interrupted"]
    assert after_wake == [("held", 0)]
    assert linked == 0
    assert later == ["interrupted", "committed", "exhausted"]
    assert reviewed == [("held", 1)]


def test_detaching_mid_decision_gives_the_unit_back_and_later_notifications_are_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import metric_reader

    async def scenario(reader: metric_reader.Reader) -> tuple[str, list[tuple[str, int]], float]:
        gate = asyncio.Event()
        harness = await _harness(_link_to_person, gate=gate)
        await harness.attach()
        await harness.remember("held", "turn-1")
        await harness.rest()
        await harness.until_state("/resting/deciding")
        await harness.detach()
        await harness.until_state("Lifecycle/detached")
        before = sum(
            point.value
            for point in reader.points("bot.ability.request.dropped.count")
            if point.attributes.get("bot.ability.name") == "Consolidation"
        )
        await harness.rest()
        await harness.wake()
        await asyncio.sleep(0.05)
        after = sum(
            point.value
            for point in reader.points("bot.ability.request.dropped.count")
            if point.attributes.get("bot.ability.name") == "Consolidation"
        )
        gate.set()
        await asyncio.sleep(0.1)
        return harness.consolidation.state(), harness.reviewed(), after - before

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        state, reviewed, counted = asyncio.run(scenario(reader))

    assert state == "/ConsolidationLifecycle/detached"
    assert reviewed == [("held", 0)]
    assert counted == 2
