"""StmMemory: bounded, access-reinforced short-term memory of admitted stimuli, reached by message."""

from mosfet import media
from mosfet.abilities import ability
from mosfet.abilities.cognition import event as projection
from mosfet.abilities.language import text
from mosfet.abilities import memory
from mosfet.abilities.memory import stm

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing
import uuid

import hsm
import mosfet
import pydantic
import pytest

from mosfet.protocols import attachment

from tests import metric_reader
from tests.bot.abilities.memory.memory_fixtures import register_recent, register_record
from tests.bot.abilities.support import require_model, shared_hsm_context, start_abilities_for_test

_TIMELINE_START = datetime.datetime(2026, 7, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)


class _ControlledClock:
    """Injectable virtual clock: the test sets ``now`` explicitly before each call site."""

    def __init__(self) -> None:
        self.now = _TIMELINE_START

    def __call__(self) -> datetime.datetime:
        return self.now


def _advance(clock: _ControlledClock, seconds: float) -> None:
    clock.now = clock.now + datetime.timedelta(seconds=seconds)


def _sound_event(
    *,
    name: str = "environment.sound",
    data: object | None = None,
    source: str = "device-a",
) -> hsm.Event[typing.Any]:
    event = hsm.Event[object](name=name, schema=object).with_data(
        {"kind": "knock", "source": source} if data is None else data
    )
    return event


def _notable_event(mark: str, *, name: str = "environment.sound") -> hsm.Event[typing.Any]:
    return _sound_event(name=name, data={"kind": "knock", "source": mark})


class _SensoryPayload(pydantic.BaseModel):
    media_type: str | None = None
    sample_rate_hz: int | None = None
    audio: bytes | None = None


class _Peer:
    """A peer of the register: it only sends events and awaits correlated replies."""

    def __init__(self, ctx: hsm.Context, register: memory.StmMemory) -> None:
        self.ctx = ctx
        self.register = register

    async def record(self, event: hsm.Event[typing.Any]) -> None:
        await register_record(self.ctx, self.register, event)

    async def reinforce(self, stm_ids: tuple[str, ...]) -> None:
        _ = await hsm.dispatch(
            self.ctx, self.register, stm.ReinforceEvent.with_data(stm.ReinforceData(stm_ids=stm_ids))
        )
        # Requests are applied in arrival order: a read answered after it proves the write landed.
        _ = await self.recent(limit=1)

    async def recent(self, stimulus_name: str | None = None, *, limit: int) -> tuple[stm.ObservedEvent, ...]:
        return await register_recent(self.ctx, self.register, stimulus_name, limit=limit)


def _run(
    register: memory.StmMemory,
    scenario: collections.abc.Callable[[_Peer], collections.abc.Coroutine[typing.Any, typing.Any, None]],
) -> None:
    async def run() -> None:
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, register)
        try:
            await scenario(_Peer(ctx, register))
        finally:
            await register.stop(ctx)

    asyncio.run(run())


def _sources(entries: tuple[stm.ObservedEvent, ...]) -> list[object]:
    return [entry.payload["source"] for entry in entries]


def test_register_bounds_capacity_to_newest_entries() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        for mark in "one", "two", "three", "four":
            await peer.record(_notable_event(mark))
            _advance(clock, 1.0)
        assert _sources(await peer.recent(limit=16)) == ["four", "three"]

    _run(memory.StmMemory(capacity=2, now=clock), scenario)


def test_register_recent_orders_newest_first_and_filters_by_name() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("ring-1"))
        _advance(clock, 1.0)
        await peer.record(_notable_event("knock-1", name="environment.sound"))
        _advance(clock, 1.0)
        await peer.record(_notable_event("ring-2", name="environment.visual"))
        _advance(clock, 1.0)
        await peer.record(_notable_event("ring-3"))

        assert _sources(await peer.recent(limit=16)) == ["ring-3", "ring-2", "knock-1", "ring-1"]
        sounds = await peer.recent("environment.sound", limit=16)
        assert _sources(sounds) == ["ring-3", "knock-1", "ring-1"]
        assert sounds[0].stimulus_name == "environment.sound"

    _run(memory.StmMemory(now=clock), scenario)


def test_register_recency_window_excludes_stale_observations() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("first"))
        assert _sources(await peer.recent(limit=16)) == ["first"]
        # Second record six minutes later; both observations are within the ten-minute window.
        _advance(clock, 6 * 60)
        await peer.record(_notable_event("second"))
        assert _sources(await peer.recent(limit=16)) == ["second", "first"]
        # Third record twelve minutes after the first: "first" is now stale and drops out.
        _advance(clock, 6 * 60)
        await peer.record(_notable_event("third"))
        assert _sources(await peer.recent(limit=16)) == ["third", "second"]

    _run(memory.StmMemory(recency_window=datetime.timedelta(minutes=10), now=clock), scenario)


def test_event_record_projection_keeps_bounded_scalars_only() -> None:
    long_id = "d" * (stm.MAX_PROJECTION_TEXT_LENGTH + 1)
    sound = _sound_event(
        data=_SensoryPayload(
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            audio=b"payload-bytes",
        )
    )

    assert stm.event_record(sound).payload == {"media_type": "audio/pcm", "sample_rate_hz": 48_000}
    assert stm.event_record(_sound_event(data={"kind": "knock", "long_id": long_id})).payload == {"kind": "knock"}
    assert stm.event_record(_sound_event(data="just-a-word")).payload == {"value": "just-a-word"}


def test_register_records_envelope_identity() -> None:
    enveloped = dataclasses.replace(_notable_event("device-b"), id="turn-1", source="device-b", target="bot")

    recorded = stm.event_record(enveloped)
    assert recorded.stimulus_name == "environment.sound"
    assert recorded.event_id == "turn-1"
    assert recorded.source == "device-b"
    assert recorded.target == "bot"

    async def scenario(peer: _Peer) -> None:
        await peer.record(enveloped)
        recent = await peer.recent("environment.sound", limit=1)
        assert recent[0].stimulus_name == "environment.sound"
        assert recent[0].payload == {"kind": "knock", "source": "device-b"}
        assert recent[0].event_id == "turn-1"
        assert recent[0].source == "device-b"
        assert recent[0].target == "bot"

    _run(memory.StmMemory(), scenario)


def test_register_survives_restart_on_file_backed_store(tmp_path: pathlib.Path) -> None:
    database = str(tmp_path / "register.db")
    dom_id = uuid.uuid4().hex

    async def write(peer: _Peer) -> None:
        await peer.record(_notable_event(dom_id))

    async def read(peer: _Peer) -> None:
        assert _sources(await peer.recent(limit=16)) == [dom_id]

    _run(memory.StmMemory(database=database), write)
    _run(memory.StmMemory(database=database), read)


def test_register_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        memory.StmMemory(capacity=0)
    with pytest.raises(ValueError):
        memory.StmMemory(recency_window=datetime.timedelta(0))
    with pytest.raises(ValueError):
        _ = stm.RecentData(limit=0)
    with pytest.raises(ValueError):
        _ = stm.RecentData(limit=1000)


def test_register_record_data_rejects_unbounded_text() -> None:
    with pytest.raises(ValueError):
        memory.RecordData(stimulus_name="environment.sound", payload={"note": "z" * 64})


def test_register_reinforcement_extends_retention_beyond_arrival() -> None:
    """Access-stamped entries outlive arrival churn: reconsolidation, not recency-of-ingress."""

    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("used"))
        _advance(clock, 6 * 60)
        await peer.record(_notable_event("churn"))
        # A grounding pass touches the entry it used, one minute after churn arrived, so the
        # access stamp strictly outranks churn's arrival stamp (no tie on the effective clock).
        # Grounding selects by evidence, not by slot: pick the row whose payload is the used signal.
        _advance(clock, 60)
        matches = [e for e in await peer.recent("environment.sound", limit=10) if e.payload["source"] == "used"]
        await peer.reinforce((matches[0].stm_id,))
        _advance(clock, 5 * 60)
        await peer.record(_notable_event("latest"))

        # Capacity 2: eviction keeps the top two by effective clock — the reinforced "used"
        # entry (accessed just before the last churn) survives; the untouched churn entry dies.
        assert _sources(await peer.recent(limit=16)) == ["latest", "used"]

    _run(memory.StmMemory(capacity=2, recency_window=datetime.timedelta(hours=24), now=clock), scenario)


def test_register_reinforced_entry_stays_visible_to_effective_recency() -> None:
    """A touched ancient entry is still recallable — access recency, not arrival recency."""

    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("old-signal"))
        entry_id = (await peer.recent("environment.sound", limit=10))[0].stm_id
        _advance(clock, 20 * 60)
        # Twenty minutes past admission and outside the ten-minute window: invisible.
        assert await peer.recent("environment.sound", limit=10) == ()
        # Reinforcing the entry makes it accessible again through its access recency.
        await peer.reinforce((entry_id,))
        recent = await peer.recent("environment.sound", limit=10)
        assert _sources(recent) == ["old-signal"]
        assert recent[0].last_accessed_at is not None
        assert recent[0].stm_id == entry_id

    _run(memory.StmMemory(recency_window=datetime.timedelta(minutes=10), now=clock), scenario)


def test_register_reinforce_rejects_empty_ids() -> None:
    with pytest.raises(ValueError):
        _ = stm.ReinforceData(stm_ids=())


def test_register_answers_a_failed_read_with_the_failed_event_at_the_requester() -> None:
    class _BrokenRegister(memory.StmMemory):
        @typing.override
        def _execute(self, data: memory.InputData) -> memory.OutputData:
            raise RuntimeError("disk gone")

    async def scenario(peer: _Peer) -> None:
        terminal = await ability.run_terminal_operation(
            peer.ctx,
            child=peer.register,
            request=stm.RecentEvent.with_data_and_id(stm.RecentData(limit=1), "read-1"),
            terminals=(stm.RecentOutputEvent, peer.register.failed_event),
            timeout=datetime.timedelta(seconds=5),
        )
        assert terminal.id == "read-1"
        assert isinstance(terminal.data, ability.FailureData)
        assert "disk gone" in terminal.data.message

    _run(_BrokenRegister(), scenario)


def test_register_logs_a_failed_record_and_keeps_serving(caplog: pytest.LogCaptureFixture) -> None:
    class _ReadOnlyRegister(memory.StmMemory):
        @typing.override
        def _execute(self, data: memory.InputData) -> memory.OutputData:
            if data.statements[0].sql.lstrip().upper().startswith("INSERT"):
                raise RuntimeError("read-only")
            return super()._execute(data)

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("dropped"))
        assert await peer.recent(limit=16) == ()

    with caplog.at_level("WARNING", logger=stm.__name__):
        _run(_ReadOnlyRegister(), scenario)

    assert any("bot.ability.memory.stm.record failed" in record.getMessage() for record in caplog.records)


# --- what the register cannot serve -------------------------------------------------------------


class _Recorder(hsm.Instance):
    """Attachment owner and requester: records every event delivered to it; ``react`` may answer one."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event
    react: collections.abc.Callable[[hsm.Context, hsm.Event[typing.Any]], None] | None

    @staticmethod
    def _record(ctx: hsm.Context, instance: "_Recorder", event: hsm.Event[typing.Any]) -> None:
        instance.events.append(event)
        instance.changed.set()
        if instance.react is not None:
            instance.react(ctx, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "StmTestRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()
        self.react = None

    def replies(self, request_id: str) -> list[hsm.Event[typing.Any]]:
        names = {stm.RecentOutputEvent.name, ability.FailedEvent.name}
        return [event for event in self.events if event.id == request_id and event.name in names]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(5.0):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


def _recent_request(request_id: str, requester: hsm.Instance) -> hsm.Event[typing.Any]:
    return dataclasses.replace(
        stm.RecentEvent.with_data_and_id(stm.RecentData(limit=1), request_id), source=hsm.id(requester)
    )


def _dropped(points: collections.abc.Sequence[metric_reader.Point]) -> dict[tuple[object, object], float]:
    return {
        (point.attributes["bot.ability.drop.reason"], point.attributes["bot.ability.replied"]): point.value
        for point in points
        if point.attributes.get("bot.ability.name") == "StmMemory"
    }


def test_a_detached_register_answers_requests_unavailable_and_counts_notifications(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C22: nothing sent to a register that is not attached is lost silently, before or after attach."""

    async def run(
        reader: metric_reader.Reader,
    ) -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str, dict[tuple[object, object], float]]:
        ctx = shared_hsm_context()
        owner = _Recorder()
        _ = await mosfet.started(ctx, owner, _Recorder.model)
        never = memory.StmMemory()
        _ = await mosfet.started(ctx, never, require_model(never.model))
        register = memory.StmMemory()
        _ = await register.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await owner.until(lambda: any(e.name == attachment.AttachCompleteEvent.name for e in owner.events))
        _ = await register.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)))
        await owner.until(lambda: register.state() == "/StmMemoryLifecycle/detached")
        before = _dropped(reader.points("bot.ability.request.dropped.count"))
        _ = await hsm.dispatch(ctx, never, _recent_request("never-attached", owner))
        _ = await hsm.dispatch(ctx, register, _recent_request("after-detach", owner))
        _ = await hsm.dispatch(ctx, register, stm.RecordEvent.with_data(stm.event_record(_notable_event("late"))))
        _ = await hsm.dispatch(ctx, register, stm.ReinforceEvent.with_data(stm.ReinforceData(stm_ids=("x",))))
        await owner.until(lambda: bool(owner.replies("never-attached")) and bool(owner.replies("after-detach")))
        after = _dropped(reader.points("bot.ability.request.dropped.count"))
        changes = {key: value - before.get(key, 0.0) for key, value in after.items() if value - before.get(key, 0.0)}
        return owner.replies("never-attached"), owner.replies("after-detach"), hsm.id(register), changes

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        never, detached, register_id, changes = asyncio.run(run(reader))

    for answered in (never, detached):
        (reply,) = answered
        assert reply.name == ability.FailedEvent.name
        assert isinstance(reply.data, ability.FailureData)
        assert reply.data.message == "unavailable: StmMemory is not attached."
    assert detached[0].source == register_id
    assert changes == {("detached", "true"): 2.0, ("detached", "false"): 2.0}


def test_requests_that_arrive_with_a_detach_are_each_answered_exactly_once() -> None:
    """A request queued ahead of the detach is served; one queued behind it is answered unavailable."""

    async def scenario() -> tuple[dict[str, list[hsm.Event[typing.Any]]], str]:
        ctx = shared_hsm_context()
        owner = _Recorder()
        _ = await mosfet.started(ctx, owner, _Recorder.model)
        register = memory.StmMemory()
        _ = await register.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await owner.until(lambda: any(e.name == attachment.AttachCompleteEvent.name for e in owner.events))

        def burst(reaction_ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
            # Sent while the register is still answering the first request: they queue behind it.
            if event.id != "first" or event.name != stm.RecentOutputEvent.name:
                return
            _ = hsm.dispatch(reaction_ctx, register, _recent_request("ahead", owner))
            _ = hsm.dispatch(
                reaction_ctx, register, attachment.DetachEvent.with_data(attachment.DetachData(actor=owner))
            )
            _ = hsm.dispatch(reaction_ctx, register, _recent_request("behind", owner))

        owner.react = burst
        _ = await hsm.dispatch(ctx, register, _recent_request("first", owner))
        await owner.until(lambda: all(owner.replies(request) for request in ("first", "ahead", "behind")))
        await asyncio.sleep(0.05)
        return {request: owner.replies(request) for request in ("first", "ahead", "behind")}, register.state()

    replies, state = asyncio.run(scenario())

    assert state == "/StmMemoryLifecycle/detached"
    assert [reply.name for reply in replies["first"]] == [stm.RecentOutputEvent.name]
    assert [reply.name for reply in replies["ahead"]] == [stm.RecentOutputEvent.name]
    (behind,) = replies["behind"]
    assert behind.name == ability.FailedEvent.name
    assert isinstance(behind.data, ability.FailureData)
    assert behind.data.message.startswith("unavailable")


def test_the_register_owns_a_detaching_state() -> None:
    """Composite lifecycle: a detach enters the register's own detaching state, which answers what waits."""

    model = memory.StmMemory.model
    assert model is not None
    assert "/StmMemoryLifecycle/attached/behavior/detaching" in model.members
    assert "/StmMemoryLifecycle/attached/behavior/degraded" in model.members


def test_the_register_opens_its_database_on_attach_not_on_construction(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "register.db"
    register = memory.StmMemory(database=str(path))
    constructed = path.exists()

    async def scenario(peer: _Peer) -> None:
        await peer.record(_notable_event("opened"))
        assert _sources(await peer.recent(limit=1)) == ["opened"]

    _run(register, scenario)
    assert constructed is False
    assert path.exists()


def test_a_register_that_cannot_open_its_database_fails_its_attach(tmp_path: pathlib.Path) -> None:
    import contextlib
    import sqlite3

    path = tmp_path / "unversioned.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")

    async def scenario() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = shared_hsm_context()
        owner = _Recorder()
        _ = await mosfet.started(ctx, owner, _Recorder.model)
        register = memory.StmMemory(database=str(path))
        _ = await register.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await owner.until(lambda: any(e.name == attachment.AttachFailedEvent.name for e in owner.events))
        state = register.state()
        await register.stop(ctx)
        return [e for e in owner.events if e.name == attachment.AttachFailedEvent.name], state

    (failed,), state = asyncio.run(scenario())
    assert isinstance(failed.data, attachment.FailedData)
    assert failed.data.kind is attachment.FailureKind.INITIALIZATION
    assert "MigrationError" in failed.data.message
    assert state == "/StmMemoryLifecycle/attached/behavior/degraded"


# --- turns: what was heard and said ------------------------------------------------------------------------------

_MAYA = mosfet.Participant(ref="voice:3f2a9c", role=mosfet.ParticipantRole.SOURCE)
_OWN = mosfet.Participant(ref="voice:tts:warm", role=mosfet.ParticipantRole.SOURCE, own=True)


def _turn(
    words: str, *, direction: typing.Literal["heard", "said"] = "heard", id: str | None = None
) -> stm.TurnRecordData:
    return stm.TurnRecordData.of(
        id=id or words,
        direction=direction,
        speakers=(_MAYA,) if direction == "heard" else (_OWN,),
        words=words,
        conversation="conversation-1",
    )


async def _record_turn(peer: _Peer, turn: stm.TurnRecordData) -> None:
    _ = await hsm.dispatch(peer.ctx, peer.register, stm.TurnEvent.with_data(turn))
    # Requests are applied in arrival order: a read answered after it proves the write landed.
    _ = await _recent_turns(peer, limit=1)


async def _recent_turns(
    peer: _Peer, *, limit: int = 32, within: datetime.timedelta | None = None
) -> tuple[stm.Turn, ...]:
    terminal = await ability.run_terminal_operation(
        peer.ctx,
        child=peer.register,
        request=stm.RecentTurnsEvent.with_data_and_id(
            stm.RecentTurnsData(limit=limit, within=within), uuid.uuid4().hex
        ),
        terminals=(stm.RecentTurnsOutputEvent, peer.register.failed_event),
        timeout=datetime.timedelta(seconds=5),
    )
    assert isinstance(terminal.data, stm.RecentTurnsOutputData), terminal.data
    return terminal.data.turns


def test_turns_are_recorded_stamped_and_read_back_oldest_first() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await _record_turn(peer, _turn("Hi Willy, I'm Maya.", id="turn-1"))
        _advance(clock, 2.0)
        await _record_turn(peer, _turn("Hello Maya.", direction="said", id="turn-1:said"))
        _advance(clock, 2.0)
        await _record_turn(peer, _turn("How are you?", id="turn-2"))

        turns = await _recent_turns(peer)
        assert [(turn.id, turn.direction, turn.text) for turn in turns] == [
            ("turn-1", "heard", "Hi Willy, I'm Maya."),
            ("turn-1:said", "said", "Hello Maya."),
            ("turn-2", "heard", "How are you?"),
        ]
        assert [turn.at for turn in turns] == [
            _TIMELINE_START,
            _TIMELINE_START + datetime.timedelta(seconds=2),
            _TIMELINE_START + datetime.timedelta(seconds=4),
        ]
        assert turns[0].speakers == (_MAYA,)
        assert turns[1].speakers == (_OWN,)
        assert {turn.conversation for turn in turns} == {"conversation-1"}
        assert not any(turn.cut for turn in turns)
        # A limit keeps the newest turns, still oldest first.
        assert [turn.id for turn in await _recent_turns(peer, limit=2)] == ["turn-1:said", "turn-2"]
        # Turns never reach the scalar register Learning reads.
        assert await peer.recent(limit=16) == ()

    _run(memory.StmMemory(now=clock), scenario)


def test_a_full_turn_store_sheds_its_older_half_at_once() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        oldest: list[str] = []
        for words in ("one", "two", "three", "four", "five", "six", "seven", "eight"):
            await _record_turn(peer, _turn(words))
            _advance(clock, 1.0)
            oldest.append((await _recent_turns(peer))[0].text)
        # Four fit. The fifth sheds the older half (one, two, three) at once, and the oldest turn read
        # stays "four" until the store is full again and the eighth sheds the next half.
        assert oldest == ["one", "one", "one", "one", "four", "four", "four", "seven"]
        assert [turn.text for turn in await _recent_turns(peer)] == ["seven", "eight"]

    _run(memory.StmMemory(turn_capacity=4, now=clock), scenario)


def test_turns_leave_the_recency_window_in_half_window_steps() -> None:
    clock = _ControlledClock()

    async def scenario(peer: _Peer) -> None:
        await _record_turn(peer, _turn("first"))
        _advance(clock, 3 * 60)
        await _record_turn(peer, _turn("second"))
        _advance(clock, 6 * 60)
        await _record_turn(peer, _turn("third"))
        # The default window is ten minutes and its lower bound moves in five-minute steps: at 12:09 the
        # bound is 11:55, at 12:12 it is still 12:00, so "first" (12:00) is read past ten minutes.
        assert [turn.text for turn in await _recent_turns(peer)] == ["first", "second", "third"]
        _advance(clock, 3 * 60)
        assert [turn.text for turn in await _recent_turns(peer)] == ["first", "second", "third"]
        # At 12:15 the bound steps to 12:05: "first" and "second" (12:03) leave together.
        _advance(clock, 3 * 60)
        assert [turn.text for turn in await _recent_turns(peer)] == ["third"]
        # A request may narrow the window, never widen it; a narrowed window steps in its own halves.
        assert [turn.text for turn in await _recent_turns(peer, within=datetime.timedelta(minutes=8))] == ["third"]
        assert [turn.text for turn in await _recent_turns(peer, within=datetime.timedelta(hours=1))] == ["third"]

    _run(memory.StmMemory(now=clock), scenario)


def test_a_turn_longer_than_its_bound_is_cut_at_a_whole_word() -> None:
    words = "word " * 1_000
    turn = _turn(words)

    assert turn.cut
    assert len(turn.text.encode()) <= stm.MAX_TURN_TEXT_BYTES
    assert words.startswith(turn.text)
    assert turn.text.endswith("word")
    assert not _turn("short and whole").cut
    # Multi-byte text is cut on a character, never inside one.
    wide = _turn("é" * 3_000)
    assert wide.cut and wide.text == "é" * (stm.MAX_TURN_TEXT_BYTES // 2)
    with pytest.raises(pydantic.ValidationError):
        _ = stm.TurnRecordData(
            id="turn-1",
            direction="heard",
            speakers=(_MAYA,),
            text="x" * (stm.MAX_TURN_TEXT_BYTES + 1),
            conversation="conversation-1",
        )


def test_turn_configuration_must_be_positive() -> None:
    with pytest.raises(ValueError):
        memory.StmMemory(turn_capacity=0)
    with pytest.raises(ValueError):
        memory.StmMemory(turn_window=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = stm.RecentTurnsData(within=datetime.timedelta(0))


def test_recorded_turns_are_counted_by_direction_and_cut(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def counts(reader: metric_reader.Reader) -> dict[tuple[object, object], float]:
        return {
            (point.attributes.get("bot.turn.direction"), point.attributes.get("bot.turn.cut")): point.value
            for point in reader.points("bot.memory.stm.turn.recorded.count")
        }

    async def scenario(peer: _Peer) -> None:
        await _record_turn(peer, _turn("hello"))
        await _record_turn(peer, _turn("word " * 1_000, direction="said"))

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = counts(reader)
        _run(memory.StmMemory(), scenario)
        after = counts(reader)

    changes = {key: value - before.get(key, 0.0) for key, value in after.items() if value - before.get(key, 0.0)}
    assert changes == {("heard", "false"): 1.0, ("said", "true"): 1.0}


def test_a_turn_renders_for_a_model_as_one_line_of_when_which_way_who_and_words() -> None:
    """R-20 C3: id and conversation are correlation the model never reads; cut shows only when true."""

    maya = mosfet.Participant(ref="voice:maya", role=mosfet.ParticipantRole.SOURCE)
    ada = mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE)
    at = datetime.datetime(2026, 7, 7, 21, 3, 10, 500_000, tzinfo=datetime.UTC)

    def rendered(record: stm.TurnRecordData) -> str:
        turn = stm.Turn(**record.model_dump(), at=at)
        parts = projection.model_facing_content(turn, capacity=media.ImageCapacity())
        return "".join(part.text for part in parts if isinstance(part, text.TextPart))

    whole = stm.TurnRecordData.of(
        id="turn-1", direction="heard", speakers=(maya, ada), words="Hi <Willy> & co.", conversation="talk-1"
    )
    cut = stm.TurnRecordData.of(id="turn-2", direction="said", speakers=(), words="word " * 1_000, conversation="t")

    assert rendered(whole) == (
        '<memory:turn at="21:03:10Z" direction="heard" who="voice:maya +15555550100">Hi &lt;Willy&gt; &amp; co.'
        "</memory:turn>"
    )
    assert rendered(cut).startswith('<memory:turn at="21:03:10Z" direction="said" cut="true">word word')


def test_register_opens_migrates_and_applies_off_the_event_loop() -> None:
    """3.4: the register's open, migration, records and reads run on its worker, never on the loop's thread."""

    import threading

    import sqlalchemy.event
    from sqlalchemy.engine import Engine

    from mosfet.abilities.memory import store

    threads: list[int] = []
    loop_threads: list[int] = []

    def handle() -> Engine:
        threads.append(threading.get_ident())
        engine = store.open_sqlite_engine()
        sqlalchemy.event.listen(engine, "before_cursor_execute", lambda *_: threads.append(threading.get_ident()))
        return engine

    async def scenario(peer: _Peer) -> None:
        loop_threads.append(threading.get_ident())
        await peer.record(_notable_event("off"))
        assert _sources(await peer.recent(limit=1)) == ["off"]

    _run(memory.StmMemory(engine=handle), scenario)

    assert threads
    assert not set(loop_threads) & set(threads)
