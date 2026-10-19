"""HSM runtime tests for ``identification.Registry`` with a fake identifier and one fake time.

The fake time drives both clocks the registry reads: the HSM instance clock that waits out every
``hsm.after`` and the injected wall clock that retention and ``_remaining`` read, so advancing it
moves both together.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import math
import pathlib
import typing

import hsm
import mosfet
import pydantic
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mosfet import environment
from mosfet.abilities import identification
from mosfet.protocols import attachment
from mosfet.telemetry import span

from tests import metric_reader

_START = datetime.datetime(2026, 10, 15, 12, 0, tzinfo=datetime.UTC)
_WAIT = 2.0
_VECTOR = (0.6180339887, 0.7861513778)
_PRINT = identification.Print(vector=_VECTOR, model="voice-model")
# Orthogonal to ``_PRINT``: similarity 0, so the two are never one voice.
_FAR = identification.Print(vector=(-_VECTOR[1], _VECTOR[0]), model="voice-model")
_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
_MINUTE = datetime.timedelta(minutes=1)


async def _settle() -> None:
    for _ in range(200):
        await asyncio.sleep(0)


class FakeTime:
    """One fake time for the HSM clock (``after``) and the wall clock (``wall``)."""

    now: datetime.datetime
    _timers: list[tuple[datetime.datetime, asyncio.Future[datetime.datetime]]]

    def __init__(self) -> None:
        self.now = _START
        self._timers = []

    def wall(self) -> datetime.datetime:
        return self.now

    def after(self, duration: datetime.timedelta) -> asyncio.Future[datetime.datetime]:
        future: asyncio.Future[datetime.datetime] = asyncio.get_running_loop().create_future()
        if duration <= datetime.timedelta(0):
            future.set_result(self.now)
        else:
            self._timers.append((self.now + duration, future))
        return future

    async def advance(self, delta: datetime.timedelta) -> None:
        """Move to ``now + delta``, firing every timer due on the way at its own deadline."""

        target = self.now + delta
        while True:
            await _settle()
            self._timers = [(deadline, future) for deadline, future in self._timers if not future.done()]
            due = [deadline for deadline, _ in self._timers if deadline <= target]
            if not due:
                break
            self.now = min(due)
            for deadline, future in self._timers:
                if deadline == self.now:
                    future.set_result(deadline)
        self.now = target
        await _settle()


@dataclasses.dataclass
class Call:
    operation: str
    at: datetime.datetime
    released: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


class FakeIdentifier(identification.Identifier):
    """An identifier whose calls the test can hold, fail, or answer."""

    time: FakeTime
    calls: list[Call]
    prunes: list[tuple[datetime.datetime, datetime.datetime, int]]
    links: list[tuple[identification.Id, bool]]
    gated: set[str]
    errors: dict[str, BaseException]
    recognitions: tuple[identification.Recognition | None, ...] | None
    enrolled: identification.Recognition | None
    mints: list[tuple[bool, ...]]

    def __init__(self, time: FakeTime) -> None:
        self.time = time
        self.calls = []
        self.mints = []
        self.prunes = []
        self.links = []
        self.gated = set()
        self.errors = {}
        self.recognitions = None
        self.enrolled = None

    async def _call(self, operation: str) -> None:
        call = Call(operation=operation, at=self.time.now)
        self.calls.append(call)
        if operation in self.gated:
            _ = await call.released.wait()
        error = self.errors.get(operation)
        if error is not None:
            raise error

    def started(self, operation: str) -> list[Call]:
        return [call for call in self.calls if call.operation == operation]

    @typing.override
    async def identify(
        self, kind: identification.Kind, prints: tuple[identification.Print, ...], mint: tuple[bool, ...]
    ) -> tuple[identification.Recognition | None, ...]:
        self.mints.append(mint)
        await self._call("identify")
        if self.recognitions is not None:
            return self.recognitions
        return tuple(
            identification.Recognition(id=f"v{index}", recognized=index == 0, similarity=0.9, own=False)
            for index in range(len(prints))
        )

    @typing.override
    async def enroll(
        self, kind: identification.Kind, id: identification.Id | None, print: identification.Print, own: bool
    ) -> identification.Recognition:
        await self._call("enroll")
        if self.enrolled is not None:
            return self.enrolled
        return identification.Recognition(id=id or "own", recognized=False, similarity=None, own=own)

    @typing.override
    async def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        await self._call("forget")
        return id == "known"

    @typing.override
    async def link(self, kind: identification.Kind, id: identification.Id, linked: bool) -> None:
        await self._call("link")
        self.links.append((id, linked))

    @typing.override
    async def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        self.prunes.append((self.time.now, unseen_before, max_ids))
        await self._call("prune")
        return 3


class Recorder(hsm.Instance):
    """The requester: records every event delivered to it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "IdentificationTestRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def replies(self) -> list[hsm.Event[typing.Any]]:
        names = {
            event.name
            for event in (
                identification.IdentifiedEvent,
                identification.EnrolledEvent,
                identification.ForgottenEvent,
                identification.LinkedEvent,
                identification.FailedEvent,
            )
        }
        return [event for event in self.events if event.name in names]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


@dataclasses.dataclass
class Harness[I: identification.Identifier]:
    ctx: environment.Environment
    registry: identification.Registry
    owner: Recorder
    identifier: I
    time: FakeTime

    async def send(self, request: hsm.Event[typing.Any], request_id: str, data: object) -> None:
        _ = await hsm.dispatch(
            self.ctx,
            self.registry,
            dataclasses.replace(
                request.with_data_and_id(data, request_id),
                source=hsm.id(self.owner),
                target=hsm.id(self.registry),
            ),
        )
        await _settle()

    async def identify(self, request_id: str, *prints: identification.Print) -> None:
        await self.send(
            identification.IdentifyEvent, request_id, identification.IdentifyData(prints=prints or (_PRINT,))
        )

    async def reply(self, request_id: str) -> hsm.Event[typing.Any]:
        await self.owner.until(lambda: any(event.id == request_id for event in self.owner.replies()))
        (found,) = [event for event in self.owner.replies() if event.id == request_id]
        return found


async def _harness(
    *,
    kind: identification.Kind = "voice",
    attach: bool = True,
    max_unlinked_age: datetime.timedelta = datetime.timedelta(days=30),
    max_ids: int = 500,
    prune_every: datetime.timedelta = 10 * _MINUTE,
    timeout: datetime.timedelta = datetime.timedelta(seconds=10),
    prune_timeout: datetime.timedelta = datetime.timedelta(seconds=30),
) -> Harness[FakeIdentifier]:
    time = FakeTime()
    return await _start(
        FakeIdentifier(time),
        time,
        kind=kind,
        attach=attach,
        max_unlinked_age=max_unlinked_age,
        max_ids=max_ids,
        prune_every=prune_every,
        timeout=timeout,
        prune_timeout=prune_timeout,
    )


async def _matched(*, prune_every: datetime.timedelta) -> Harness[identification.Matcher]:
    """A started registry over a real ``Matcher`` and ``InProcessStore``, both on the fake time."""

    time = FakeTime()
    return await _start(
        identification.Matcher(store=identification.InProcessStore(max_ids=16), policy=_POLICY, now=time.wall),
        time,
        kind="voice",
        attach=True,
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=500,
        prune_every=prune_every,
        timeout=datetime.timedelta(seconds=10),
        prune_timeout=datetime.timedelta(seconds=30),
    )


async def _start[I: identification.Identifier](
    identifier: I,
    time: FakeTime,
    *,
    kind: identification.Kind,
    attach: bool,
    max_unlinked_age: datetime.timedelta,
    max_ids: int,
    prune_every: datetime.timedelta,
    timeout: datetime.timedelta,
    prune_timeout: datetime.timedelta,
) -> Harness[I]:
    ctx = environment.Environment()
    owner = Recorder()
    _ = await mosfet.started(ctx, owner, Recorder.model)
    registry = identification.Registry(
        identifier=identifier,
        kind=kind,
        max_unlinked_age=max_unlinked_age,
        max_ids=max_ids,
        prune_every=prune_every,
        timeout=timeout,
        prune_timeout=prune_timeout,
        now=time.wall,
    )
    model = registry.model
    assert model is not None
    _ = await mosfet.started(ctx, registry, model, hsm.Config(clock=hsm.Clock(after=time.after)))
    if attach:
        _ = await registry.attach(
            ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "attach")
        )
        await owner.until(lambda: any(event.name == attachment.AttachCompleteEvent.name for event in owner.events))
    return Harness(ctx=ctx, registry=registry, owner=owner, identifier=identifier, time=time)


def _failure(event: hsm.Event[typing.Any]) -> identification.FailureData:
    assert event.name == identification.FailedEvent.name
    data = event.data
    assert isinstance(data, identification.FailureData)
    return data


def test_a_registry_says_which_kind_it_serves_without_starting() -> None:
    voices = identification.Registry(
        identifier=FakeIdentifier(FakeTime()), kind="voice", max_unlinked_age=_MINUTE, max_ids=1
    )
    faces = identification.Registry(
        identifier=FakeIdentifier(FakeTime()), kind="face", max_unlinked_age=_MINUTE, max_ids=1
    )

    assert (voices.kind(), faces.kind()) == ("voice", "face")
    assert not voices.state()


def test_the_constructor_rejects_non_positive_durations_and_a_negative_cap() -> None:
    identifier = FakeIdentifier(FakeTime())
    with pytest.raises(ValueError, match="durations"):
        _ = identification.Registry(
            identifier=identifier, kind="voice", max_unlinked_age=datetime.timedelta(0), max_ids=1
        )
    with pytest.raises(ValueError, match="durations"):
        _ = identification.Registry(
            identifier=identifier,
            kind="voice",
            max_unlinked_age=_MINUTE,
            max_ids=1,
            prune_timeout=datetime.timedelta(0),
        )
    with pytest.raises(ValueError, match="max_ids"):
        _ = identification.Registry(identifier=identifier, kind="voice", max_unlinked_age=_MINUTE, max_ids=-1)


def test_identify_enroll_and_forget_reply_to_the_source_under_the_request_id() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str]:
        harness = await _harness(kind="face")
        await harness.identify("who", _PRINT, _PRINT)
        _ = await harness.reply("who")
        await harness.send(
            identification.EnrollEvent, "enroll", identification.EnrollData(id="tts:warm", print=_PRINT, own=True)
        )
        _ = await harness.reply("enroll")
        await harness.send(identification.ForgetEvent, "forget", identification.ForgetData(id="known"))
        _ = await harness.reply("forget")
        return harness.owner.replies(), hsm.id(harness.registry), hsm.id(harness.owner)

    replies, registry_id, owner_id = asyncio.run(run())

    assert [(event.id, event.name) for event in replies] == [
        ("who", identification.IdentifiedEvent.name),
        ("enroll", identification.EnrolledEvent.name),
        ("forget", identification.ForgottenEvent.name),
    ]
    assert all(event.source == registry_id and event.target == owner_id for event in replies)
    identified, enrolled, forgotten = (event.data for event in replies)
    assert identified == identification.IdentifiedData(
        kind="face",
        recognitions=(
            identification.Recognition(id="v0", recognized=True, similarity=0.9, own=False),
            identification.Recognition(id="v1", recognized=False, similarity=0.9, own=False),
        ),
    )
    assert enrolled == identification.Recognition(id="tts:warm", recognized=False, similarity=None, own=True)
    assert forgotten == identification.ForgottenData(id="known", forgotten=True)


def test_a_request_while_busy_is_deferred_then_served() -> None:
    async def run() -> tuple[list[str], list[str | None]]:
        harness = await _harness()
        harness.identifier.gated.add("identify")
        await harness.identify("first")
        await harness.identify("second")
        started_while_busy = [call.operation for call in harness.identifier.calls]
        harness.identifier.calls[0].released.set()
        await _settle()
        harness.identifier.calls[1].released.set()
        _ = await harness.reply("second")
        return started_while_busy, [event.id for event in harness.owner.replies()]

    started_while_busy, order = asyncio.run(run())

    assert started_while_busy == ["identify"]
    assert order == ["first", "second"]


def test_a_busy_timeout_fails_the_requester_and_ignores_the_late_result() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int]:
        harness = await _harness()
        harness.identifier.gated.add("identify")
        await harness.identify("slow")
        await harness.time.advance(datetime.timedelta(seconds=10))
        _ = await harness.reply("slow")
        # The abandoned call finishes late; nothing answers it again, and the next request is served.
        harness.identifier.calls[0].released.set()
        harness.identifier.gated.clear()
        await harness.identify("next")
        _ = await harness.reply("next")
        return harness.owner.replies(), len(harness.identifier.started("identify"))

    replies, calls = asyncio.run(run())

    assert [(event.id, event.name) for event in replies] == [
        ("slow", identification.FailedEvent.name),
        ("next", identification.IdentifiedEvent.name),
    ]
    failure = _failure(replies[0])
    assert failure.kind is identification.FailureKind.TIMEOUT
    assert calls == 2


_FAILURES: list[tuple[str, BaseException, identification.FailureKind]] = [
    (
        "store-failed",
        identification.StoreError(identification.FailureKind.STORE_FAILED, "the identity store failed"),
        identification.FailureKind.STORE_FAILED,
    ),
    (
        "dimension-mismatch",
        identification.StoreError(identification.FailureKind.INVALID_INPUT, "dimensions differ"),
        identification.FailureKind.INVALID_INPUT,
    ),
    ("unexpected-error", RuntimeError(f"bad vector {_VECTOR}"), identification.FailureKind.STORE_FAILED),
]


@pytest.mark.parametrize(("error", "kind"), [case[1:] for case in _FAILURES], ids=[case[0] for case in _FAILURES])
def test_an_identifier_failure_is_replied_with_its_kind_and_no_payload(
    error: BaseException, kind: identification.FailureKind
) -> None:
    async def run() -> identification.FailureData:
        harness = await _harness()
        harness.identifier.errors["identify"] = error
        await harness.identify("broken")
        return _failure(await harness.reply("broken"))

    failure = asyncio.run(run())

    assert failure.kind is kind
    assert str(_VECTOR[0]) not in failure.message


def test_an_invalid_print_is_invalid_input_without_validation_text() -> None:
    async def run() -> tuple[identification.FailureData, int]:
        harness = await _harness()
        await harness.send(
            identification.IdentifyEvent,
            "invalid",
            {"prints": [{"vector": [math.nan, _VECTOR[1]], "model": "voice-model"}]},
        )
        return _failure(await harness.reply("invalid")), len(harness.identifier.calls)

    failure, calls = asyncio.run(run())

    assert failure.kind is identification.FailureKind.INVALID_INPUT
    assert failure.message == "the identify request is invalid"
    assert calls == 0


def test_a_detached_registry_answers_every_request_unavailable_at_once() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int]:
        harness = await _harness(attach=False)
        await harness.identify("identify")
        await harness.send(
            identification.EnrollEvent, "enroll", identification.EnrollData(id="tts:warm", print=_PRINT, own=True)
        )
        await harness.send(identification.ForgetEvent, "forget", identification.ForgetData(id="known"))
        await harness.send(identification.LinkEvent, "link", identification.LinkData(id="known", linked=True))
        await harness.owner.until(lambda: len(harness.owner.replies()) == 4)
        return harness.owner.replies(), len(harness.identifier.calls)

    replies, calls = asyncio.run(run())

    assert [event.id for event in replies] == ["identify", "enroll", "forget", "link"]
    for event in replies:
        failure = _failure(event)
        assert failure.kind is identification.FailureKind.UNAVAILABLE
        assert failure.message == "identification registry is detached"
    assert calls == 0


def test_a_registry_detached_while_busy_answers_its_in_flight_request_unavailable() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int]:
        harness = await _harness()
        harness.identifier.gated.add("identify")
        await harness.identify("slow")
        _ = await harness.registry.detach(
            harness.ctx,
            attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=harness.owner), "detach"),
        )
        # Answered at once, well inside the request's own timeout, which never comes.
        _ = await harness.reply("slow")
        harness.identifier.calls[0].released.set()
        harness.identifier.gated.clear()
        _ = await harness.registry.attach(
            harness.ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=harness.owner), "reattach"),
        )
        await harness.owner.until(
            lambda: any(
                event.id == "reattach" and event.name == attachment.AttachCompleteEvent.name
                for event in harness.owner.events
            )
        )
        await harness.identify("next")
        _ = await harness.reply("next")
        return harness.owner.replies(), len(harness.identifier.started("identify"))

    replies, calls = asyncio.run(run())

    assert [(event.id, event.name) for event in replies] == [
        ("slow", identification.FailedEvent.name),
        ("next", identification.IdentifiedEvent.name),
    ]
    failure = _failure(replies[0])
    assert failure.kind is identification.FailureKind.UNAVAILABLE
    assert failure.message == "identification registry is detached"
    assert calls == 2


def _recognitions(event: hsm.Event[typing.Any]) -> tuple[identification.Recognition | None, ...]:
    data = event.data
    assert isinstance(data, identification.IdentifiedData)
    return data.recognitions


def test_a_link_mark_keeps_an_id_through_retention_until_it_is_unlinked() -> None:
    async def run() -> None:
        harness = await _matched(prune_every=datetime.timedelta(days=31))
        await harness.identify("first", _PRINT, _FAR)
        known, stranger = _recognitions(await harness.reply("first"))
        assert known is not None and stranger is not None
        await harness.send(identification.LinkEvent, "link", identification.LinkData(id=known.id, linked=True))
        assert (await harness.reply("link")).data == identification.LinkedData(id=known.id, linked=True)

        # One prune at day 31: the stranger is older than 30 days and unlinked; the linked id is kept.
        await harness.time.advance(datetime.timedelta(days=31))
        await harness.identify("again", _PRINT, _FAR)
        again, newcomer = _recognitions(await harness.reply("again"))
        assert again is not None and newcomer is not None
        assert (again.id, again.recognized) == (known.id, True)
        assert newcomer.recognized is False
        assert newcomer.id != stranger.id

        await harness.send(identification.LinkEvent, "unlink", identification.LinkData(id=known.id, linked=False))
        assert (await harness.reply("unlink")).data == identification.LinkedData(id=known.id, linked=False)
        # The next prune, at day 62, removes the unlinked id last seen at day 31.
        await harness.time.advance(datetime.timedelta(days=31))
        await harness.identify("after", _PRINT)
        (after,) = _recognitions(await harness.reply("after"))
        assert after is not None
        assert after.recognized is False
        assert after.id != known.id

    asyncio.run(run())


def test_a_link_mark_on_an_own_id_is_a_conflict_and_on_an_unknown_id_not_found() -> None:
    async def run() -> list[identification.FailureData]:
        harness = await _matched(prune_every=10 * _MINUTE)
        await harness.send(
            identification.EnrollEvent, "own", identification.EnrollData(id=None, print=_PRINT, own=True)
        )
        own = (await harness.reply("own")).data
        assert isinstance(own, identification.Recognition)
        await harness.send(identification.LinkEvent, "self", identification.LinkData(id=own.id, linked=True))
        await harness.send(identification.LinkEvent, "nobody", identification.LinkData(id="nobody", linked=True))
        return [_failure(await harness.reply("self")), _failure(await harness.reply("nobody"))]

    conflict, missing = asyncio.run(run())

    assert conflict.kind is identification.FailureKind.CONFLICT
    assert missing.kind is identification.FailureKind.NOT_FOUND
    assert str(_VECTOR[0]) not in conflict.message + missing.message


def test_pruning_runs_every_prune_every_with_the_retention_cutoff() -> None:
    async def run() -> list[tuple[datetime.datetime, datetime.datetime, int]]:
        harness = await _harness(max_unlinked_age=datetime.timedelta(days=30), max_ids=250)
        await harness.time.advance(25 * _MINUTE)
        return harness.identifier.prunes

    prunes = asyncio.run(run())

    assert prunes == [
        (_START + 10 * _MINUTE, _START + 10 * _MINUTE - datetime.timedelta(days=30), 250),
        (_START + 20 * _MINUTE, _START + 20 * _MINUTE - datetime.timedelta(days=30), 250),
    ]


def test_steady_request_traffic_does_not_starve_pruning() -> None:
    async def run() -> tuple[list[datetime.datetime], int]:
        harness = await _harness(prune_every=10 * _MINUTE)
        for index in range(6):
            await harness.time.advance(5 * _MINUTE)
            await harness.identify(f"tick-{index}")
            _ = await harness.reply(f"tick-{index}")
        return [at for at, _, _ in harness.identifier.prunes], len(harness.owner.replies())

    prunes, replies = asyncio.run(run())

    assert replies == 6
    assert len(prunes) >= 2
    assert prunes[:3] == [_START + 10 * _MINUTE, _START + 20 * _MINUTE, _START + 30 * _MINUTE]


def test_a_prune_that_comes_due_while_busy_runs_when_the_request_ends() -> None:
    async def run() -> list[datetime.datetime]:
        harness = await _harness(prune_every=10 * _MINUTE, timeout=30 * _MINUTE)
        harness.identifier.gated.add("identify")
        await harness.time.advance(9 * _MINUTE)
        await harness.identify("long")
        await harness.time.advance(2 * _MINUTE)
        due_while_busy = list(harness.identifier.prunes)
        harness.identifier.calls[0].released.set()
        _ = await harness.reply("long")
        await _settle()
        await harness.time.advance(10 * _MINUTE)
        assert due_while_busy == []
        return [at for at, _, _ in harness.identifier.prunes]

    prunes = asyncio.run(run())

    # The first prune ran through the after-busy choice at 11 minutes; the next is due 10 minutes later.
    assert prunes == [_START + 11 * _MINUTE, _START + 21 * _MINUTE]


def test_a_prune_that_never_completes_times_out_rearms_and_serves_deferred_requests() -> None:
    async def run() -> tuple[list[datetime.datetime], list[str | None]]:
        harness = await _harness(prune_every=10 * _MINUTE, prune_timeout=_MINUTE)
        harness.identifier.gated.add("prune")
        await harness.time.advance(10 * _MINUTE)
        await harness.identify("deferred")
        deferred = [event.id for event in harness.owner.replies()]
        await harness.time.advance(_MINUTE)
        _ = await harness.reply("deferred")
        await harness.time.advance(10 * _MINUTE)
        assert deferred == []
        return [at for at, _, _ in harness.identifier.prunes], [event.id for event in harness.owner.replies()]

    prunes, replies = asyncio.run(run())

    assert prunes == [_START + 10 * _MINUTE, _START + 21 * _MINUTE]
    assert replies == ["deferred"]


def test_a_prune_failure_never_ends_a_request() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int]:
        harness = await _harness(prune_every=10 * _MINUTE, timeout=30 * _MINUTE)
        harness.identifier.errors["prune"] = identification.StoreError(
            identification.FailureKind.STORE_FAILED, "the identity store failed"
        )
        harness.identifier.gated.add("identify")
        await harness.time.advance(9 * _MINUTE)
        await harness.identify("during")
        await harness.time.advance(2 * _MINUTE)
        harness.identifier.calls[0].released.set()
        _ = await harness.reply("during")
        await _settle()
        harness.identifier.gated.clear()
        await harness.identify("after")
        _ = await harness.reply("after")
        return harness.owner.replies(), len(harness.identifier.prunes)

    replies, prunes = asyncio.run(run())

    assert [(event.id, event.name) for event in replies] == [
        ("during", identification.IdentifiedEvent.name),
        ("after", identification.IdentifiedEvent.name),
    ]
    assert prunes == 1


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def _delta(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...]
) -> dict[tuple[tuple[str, object], ...], float]:
    previous = {tuple(sorted(point.attributes.items())): point.value for point in before}
    changes = {
        tuple(sorted(point.attributes.items())): point.value
        - previous.get(tuple(sorted(point.attributes.items())), 0.0)
        for point in after
    }
    return {key: value for key, value in changes.items() if value}


_INSTRUMENTS = (
    "bot.identification.count",
    "bot.identification.pruned",
    "bot.identification.prune.failed",
    "bot.identification.forgotten",
)


def test_telemetry_counts_outcomes_with_closed_attributes_and_no_vector(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, finished_spans: InMemorySpanExporter
) -> None:
    async def run(reader: metric_reader.Reader) -> dict[str, dict[tuple[tuple[str, object], ...], float]]:
        before = {name: reader.points(name) for name in _INSTRUMENTS}
        harness = await _harness(prune_every=10 * _MINUTE)
        harness.identifier.recognitions = (
            identification.Recognition(id="known", recognized=True, similarity=0.9, own=False),
            identification.Recognition(id="new", recognized=False, similarity=None, own=False),
            None,
        )
        await harness.identify("three", _PRINT, _PRINT, _PRINT)
        _ = await harness.reply("three")
        await harness.send(identification.ForgetEvent, "forget", identification.ForgetData(id="known"))
        _ = await harness.reply("forget")
        harness.identifier.errors["identify"] = identification.StoreError(
            identification.FailureKind.STORE_FAILED, "the identity store failed"
        )
        await harness.identify("broken")
        _ = await harness.reply("broken")
        await harness.time.advance(10 * _MINUTE)
        harness.identifier.errors["prune"] = identification.StoreError(
            identification.FailureKind.STORE_FAILED, "the identity store failed"
        )
        await harness.time.advance(10 * _MINUTE)
        return {name: _delta(before[name], reader.points(name)) for name in _INSTRUMENTS}

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        deltas = asyncio.run(run(reader))

    voice = ("bot.identification.kind", "voice")
    assert deltas["bot.identification.count"] == {
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "recognized")): 1,
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "minted")): 1,
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "failed")): 1,
        (("bot.failure.kind", "store_failed"), voice, ("bot.outcome", "failed")): 1,
    }
    assert deltas["bot.identification.forgotten"] == {(voice,): 1}
    assert deltas["bot.identification.pruned"] == {(voice,): 3}
    assert deltas["bot.identification.prune.failed"] == {(("bot.failure.kind", "store_failed"), voice): 1}

    spans: list[ReadableSpan] = [
        recorded for recorded in finished_spans.get_finished_spans() if recorded.name.startswith("bot.identification.")
    ]
    kinds = {kind.value for kind in identification.FailureKind}
    assert {recorded.name for recorded in spans} == {
        "bot.identification.identify",
        "bot.identification.forget",
        "bot.identification.prune",
    }
    for recorded in spans:
        attributes = dict(recorded.attributes or {})
        assert attributes["bot.identification.kind"] == "voice"
        assert not any(isinstance(value, float) for value in attributes.values())
        if attributes.get("bot.outcome") == "failed":
            assert attributes["bot.failure.kind"] in kinds
    assert {dict(recorded.attributes or {}).get("bot.failure.kind") for recorded in spans} == {None, "store_failed"}


def test_identify_hands_the_identifier_one_mint_flag_per_print_and_counts_an_unmatched_match_only_print(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-W8-21 D5: a match-only print (an overlap's) that matches nobody is unmatched, not a failure."""

    async def run(
        reader: metric_reader.Reader,
    ) -> tuple[list[tuple[bool, ...]], dict[tuple[tuple[str, object], ...], float]]:
        before = reader.points("bot.identification.count")
        harness = await _harness()
        await harness.identify("every", _PRINT, _PRINT)
        _ = await harness.reply("every")
        harness.identifier.recognitions = (
            None,
            identification.Recognition(id="new", recognized=False, similarity=None, own=False),
        )
        await harness.send(
            identification.IdentifyEvent,
            "flagged",
            identification.IdentifyData(prints=(_PRINT, _PRINT), mint=(False, True)),
        )
        reply = await harness.reply("flagged")
        assert reply.data == identification.IdentifiedData(kind="voice", recognitions=harness.identifier.recognitions)
        return harness.identifier.mints, _delta(before, reader.points("bot.identification.count"))

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        mints, delta = asyncio.run(run(reader))

    voice = ("bot.identification.kind", "voice")
    assert mints == [(True, True), (False, True)]
    assert delta == {
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "recognized")): 1,
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "minted")): 2,
        (("bot.failure.kind", "none"), voice, ("bot.outcome", "unmatched")): 1,
    }


def test_identify_data_takes_one_mint_flag_per_print() -> None:
    with pytest.raises(pydantic.ValidationError, match="one flag per print"):
        _ = identification.IdentifyData(prints=(_PRINT, _PRINT), mint=(False,))


def test_enroll_telemetry_names_whether_the_id_was_declared_matched_or_minted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, finished_spans: InMemorySpanExporter
) -> None:
    async def run(
        reader: metric_reader.Reader,
    ) -> tuple[dict[tuple[tuple[str, object], ...], float], list[hsm.Event[typing.Any]]]:
        before = reader.points("bot.identification.count")
        harness = await _harness()
        await harness.send(
            identification.EnrollEvent, "declared", identification.EnrollData(id="3f2a9c", print=_PRINT, own=True)
        )
        _ = await harness.reply("declared")
        await harness.send(
            identification.EnrollEvent, "minted", identification.EnrollData(id=None, print=_PRINT, own=True)
        )
        _ = await harness.reply("minted")
        harness.identifier.enrolled = identification.Recognition(id="own", recognized=True, similarity=0.8, own=True)
        await harness.send(
            identification.EnrollEvent, "matched", identification.EnrollData(id=None, print=_PRINT, own=True)
        )
        _ = await harness.reply("matched")
        return _delta(before, reader.points("bot.identification.count")), harness.owner.replies()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        delta, replies = asyncio.run(run(reader))

    voice = ("bot.identification.kind", "voice")
    none = ("bot.failure.kind", "none")
    assert delta == {
        (none, voice, ("bot.identification.mode", "declared"), ("bot.outcome", "minted")): 1,
        (none, voice, ("bot.identification.mode", "own_minted"), ("bot.outcome", "minted")): 1,
        (none, voice, ("bot.identification.mode", "own_matched"), ("bot.outcome", "recognized")): 1,
    }
    assert [(event.id, event.data) for event in replies] == [
        ("declared", identification.Recognition(id="3f2a9c", recognized=False, similarity=None, own=True)),
        ("minted", identification.Recognition(id="own", recognized=False, similarity=None, own=True)),
        ("matched", identification.Recognition(id="own", recognized=True, similarity=0.8, own=True)),
    ]
    modes = [
        dict(recorded.attributes or {}).get("bot.identification.mode")
        for recorded in finished_spans.get_finished_spans()
        if recorded.name == "bot.identification.enroll"
    ]
    assert modes == ["declared", "own_minted", "own_matched"]


def test_link_marks_are_counted_linked_or_unlinked_and_spanned_without_the_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, finished_spans: InMemorySpanExporter
) -> None:
    async def run(
        reader: metric_reader.Reader,
    ) -> tuple[dict[tuple[tuple[str, object], ...], float], list[tuple[identification.Id, bool]]]:
        before = reader.points("bot.identification.count")
        harness = await _harness(kind="face")
        await harness.send(identification.LinkEvent, "link", identification.LinkData(id="3f2a9c", linked=True))
        _ = await harness.reply("link")
        await harness.send(identification.LinkEvent, "unlink", identification.LinkData(id="3f2a9c", linked=False))
        _ = await harness.reply("unlink")
        harness.identifier.errors["link"] = identification.StoreError(
            identification.FailureKind.NOT_FOUND, "no such identity"
        )
        await harness.send(identification.LinkEvent, "missing", identification.LinkData(id="gone", linked=True))
        _ = await harness.reply("missing")
        return _delta(before, reader.points("bot.identification.count")), harness.identifier.links

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        delta, links = asyncio.run(run(reader))

    face = ("bot.identification.kind", "face")
    none = ("bot.failure.kind", "none")
    assert links == [("3f2a9c", True), ("3f2a9c", False)]
    assert delta == {
        (none, face, ("bot.outcome", "linked")): 1,
        (none, face, ("bot.outcome", "unlinked")): 1,
        (("bot.failure.kind", "not_found"), face, ("bot.outcome", "failed")): 1,
    }
    spans = [recorded for recorded in finished_spans.get_finished_spans() if recorded.name == "bot.identification.link"]
    assert len(spans) == 3
    for recorded in spans:
        assert "3f2a9c" not in {str(value) for value in dict(recorded.attributes or {}).values()}
    assert [dict(recorded.attributes or {}).get("bot.failure.kind") for recorded in spans] == [None, None, "not_found"]
