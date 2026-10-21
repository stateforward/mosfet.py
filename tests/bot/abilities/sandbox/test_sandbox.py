"""HSM runtime tests for ``sandbox.Sandbox``: the owned host, the bench's lifecycle, the bot's requests and observations."""

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
import pydantic
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mosfet import environment, lifecycle
from mosfet import event as event_contract
from mosfet.abilities import ability, cognition, processing, sandbox
from mosfet.abilities.sandbox import host
from mosfet.protocols import attachment
from mosfet.telemetry import span

from tests import metric_reader

_WAIT = 3.0
_SCALE = 0.01
"""Every modeled delay the sandbox asks its clock for passes at this fraction of its length."""
_SPEC = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0")
_CAPACITY = sandbox.Capacity(
    isolation=sandbox.Isolation.CONTAINER,
    networks=frozenset({sandbox.Network.NONE}),
    max_resources=sandbox.Resources(),
    files=True,
)
_REQUESTS = (host.CreateEvent, host.RunEvent, host.WriteEvent, host.ReadEvent, host.DestroyEvent, host.SweepEvent)
_TOOLS = {
    "bot.ability.sandbox.run",
    "bot.ability.sandbox.write",
    "bot.ability.sandbox.read",
    "bot.ability.sandbox.reset",
}
_SECRET_ARGV = "argv-9f1e"
_SECRET_PATH = "/workspace/path-7c2d.txt"
_SECRET_OUTPUT = "output-4b8a"

Answer = collections.abc.Callable[[hsm.Event[typing.Any]], hsm.Event[typing.Any] | None]
"""How the fake host answers one request: its terminal, or ``None`` to hold it for the test."""


def _failed(kind: host.FailureKind) -> hsm.Event[host.HostFailureData]:
    return host.FailedEvent.with_data(host.HostFailureData(kind=kind, message=f"fake host: {kind.value}"))


def _ran(exit_code: int = 0, stdout: str = "ok\n") -> hsm.Event[sandbox.RanData]:
    return host.RanEvent.with_data(
        sandbox.RanData(
            outcome="exited", exit_code=exit_code, stdout=stdout, duration=datetime.timedelta(milliseconds=5)
        )
    )


class _NoData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


_UpEvent = hsm.Event[_NoData](name="test.sandbox.fake_host.up", schema=_NoData)
"""Test control: the fake host's daemon is reachable again, so it serves."""
_DownEvent = hsm.Event[_NoData](name="test.sandbox.fake_host.down", schema=_NoData)
"""Test control: the fake host's daemon is lost, so it stops serving."""


class FakeHost(hsm.Instance):
    """A scripted sandbox host: it records every request it is sent and answers it as ``answers`` says.

    It has no methods a peer could call; the only way to reach it is by event. It announces ``serving`` to its
    owner as soon as it starts, unless ``serving`` is false: then it stays opening (answering ``unavailable``)
    until the test brings it up. ``failed_starts`` makes its first starts fail, the way a host whose daemon cannot
    be reached fails.
    """

    requests: list[hsm.Event[typing.Any]]
    answers: dict[str, Answer]
    created: int
    failed_starts: int
    files: dict[str, str]
    serving: bool

    def __init__(self, *, failed_starts: int = 0, serving: bool = True) -> None:
        super().__init__()
        self.requests = []
        self.created = 0
        self.failed_starts = failed_starts
        self.serving = serving
        self.files = {}
        self.answers = {
            host.SweepEvent.name: lambda _: host.SweptEvent.with_data(host.SweptData(count=0)),
            host.CreateEvent.name: self.create,
            host.RunEvent.name: lambda _: _ran(),
            host.WriteEvent.name: self._write,
            host.ReadEvent.name: self._read,
            host.DestroyEvent.name: lambda _: host.DestroyedEvent.with_data(host.DestroyedData()),
        }

    def create(self, request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
        del request
        self.created += 1
        return host.CreatedEvent.with_data(host.CreatedData(machine=host.Machine(id=f"machine-{self.created}")))

    def _write(self, request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
        data = request.data
        assert isinstance(data, host.HostWriteData)
        self.files[data.write.path] = data.write.text
        return host.WrittenEvent.with_data(
            sandbox.WrittenData(path=data.write.path, bytes=len(data.write.text.encode()))
        )

    def _read(self, request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
        data = request.data
        assert isinstance(data, host.HostReadData)
        text = self.files.get(data.read.path)
        if text is None:
            return _failed(host.FailureKind.NOT_FOUND)
        return host.ContentsEvent.with_data(
            sandbox.ContentsData(path=data.read.path, text=text, truncated=False, bytes=len(text.encode()))
        )

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        if self.failed_starts > 0:
            self.failed_starts -= 1
            raise ConnectionRefusedError("the fake daemon is unreachable")
        return await super().start(ctx, data)

    @staticmethod
    def _reply(
        ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any], terminal: hsm.Event[typing.Any]
    ) -> None:
        reply = dataclasses.replace(terminal, id=event.id, source=hsm.id(instance), target=event.source)
        _ = hsm.dispatch_to(ctx, reply, event.source)

    @staticmethod
    def _serve(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        instance.requests.append(event)
        terminal = instance.answers[event.name](event)
        if terminal is not None:
            FakeHost._reply(ctx, instance, event, terminal)

    @staticmethod
    def _unavailable(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        instance.requests.append(event)
        FakeHost._reply(ctx, instance, event, _failed(host.FailureKind.UNAVAILABLE))

    @staticmethod
    def _starts_serving(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance.serving

    @staticmethod
    def _announce_serving(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        host.announce(ctx, instance, host.ServingEvent.with_data(host.ServingData()))

    @staticmethod
    def _announce_down(ctx: hsm.Context, instance: "FakeHost", event: hsm.Event[typing.Any]) -> None:
        del event
        host.announce(ctx, instance, host.DownEvent.with_data(host.DownData()))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SandboxTestHost",
        hsm.initial(hsm.target("starting")),
        hsm.choice(
            "starting",
            hsm.transition(hsm.guard(_starts_serving), hsm.target("/SandboxTestHost/open")),
            hsm.transition(hsm.target("/SandboxTestHost/opening")),
        ),
        hsm.state(
            "opening",
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_unavailable)),
            hsm.transition(hsm.on(_UpEvent), hsm.target("/SandboxTestHost/open")),
        ),
        hsm.state(
            "open",
            hsm.entry(_announce_serving),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_serve)),
            hsm.transition(hsm.on(_DownEvent), hsm.target("/SandboxTestHost/down")),
        ),
        hsm.state(
            "down",
            hsm.entry(_announce_down),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_unavailable)),
            hsm.transition(hsm.on(_UpEvent), hsm.target("/SandboxTestHost/open")),
        ),
    )

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {candidate.name for candidate in events}
        return [request for request in self.requests if request.name in names]


class Owner(hsm.Instance):
    """The body: records every event delivered to it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Owner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SandboxTestOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {candidate.name for candidate in events}
        return [event for event in self.events if event.name in names]

    def stimuli(self) -> list[hsm.Event[typing.Any]]:
        """The observation inside every cognition input handed to the body, in order."""

        found: list[hsm.Event[typing.Any]] = []
        for handoff in self.named(cognition.InputEvent):
            data = handoff.data
            assert isinstance(data, cognition.InputData)
            assert isinstance(data.stimulus, hsm.Event)
            found.append(typing.cast(hsm.Event[typing.Any], data.stimulus))
        return found


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    sandbox: sandbox.Sandbox
    owner: Owner
    host: FakeHost
    delays: list[datetime.timedelta]

    async def attached(self) -> None:
        _ = await self.sandbox.attach(
            self.ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), "attach")
        )
        await self.until(lambda: bool(self.owner.named(attachment.AttachCompleteEvent)))

    async def detached(self) -> None:
        _ = await self.sandbox.detach(
            self.ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), "detach")
        )
        await self.until(lambda: bool(self.owner.named(attachment.DetachedEvent)))

    async def ready(self, creates: int = 1) -> None:
        """Wait until the host has created ``creates`` machines and the tools are offered."""

        await self.until(lambda: self.host.created >= creates and self.offered() == _TOOLS)

    def offered(self) -> set[str]:
        return {offered.name for offered in processing.enabled_call_events(self.sandbox)}

    async def request(self, request: hsm.Event[typing.Any], data: pydantic.BaseModel, request_id: str) -> None:
        _ = await hsm.dispatch(
            self.ctx,
            self.sandbox,
            dataclasses.replace(
                request.with_data_and_id(data, request_id), source="cognition", target=hsm.id(self.sandbox)
            ),
        )

    async def host_is(self, control: hsm.Event[_NoData]) -> None:
        """Bring the fake host's daemon up or lose it (``_UpEvent``, ``_DownEvent``)."""

        _ = await hsm.dispatch(self.ctx, self.host, control.with_data(_NoData()))
        await asyncio.sleep(0.01)

    async def reply(self, request: hsm.Event[typing.Any], terminal: hsm.Event[typing.Any]) -> None:
        """Answer a held host request the way the host would."""

        reply = dataclasses.replace(terminal, id=request.id, source=hsm.id(self.host), target=request.source)
        _ = hsm.dispatch_to(self.ctx, reply, request.source)
        await asyncio.sleep(0.01)

    def answered(self, request_id: str) -> list[hsm.Event[typing.Any]]:
        return [stimulus for stimulus in self.owner.stimuli() if stimulus.id == request_id]

    async def until(self, predicate: collections.abc.Callable[[], bool], timeout: float = _WAIT) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.002)

    async def held(self, request: hsm.Event[typing.Any], count: int = 1) -> list[hsm.Event[typing.Any]]:
        await self.until(lambda: len(self.host.named(request)) >= count)
        return self.host.named(request)


def _scaled_clock(delays: list[datetime.timedelta]) -> hsm.Clock:
    def after(duration: datetime.timedelta) -> collections.abc.Awaitable[None]:
        delays.append(duration)
        return asyncio.sleep(duration.total_seconds() * _SCALE)

    return hsm.Clock(after=after)


async def _harness(
    fake: FakeHost | None = None,
    *,
    attach: bool = True,
    spec: sandbox.Spec = _SPEC,
    capacity: sandbox.Capacity = _CAPACITY,
    **settings: typing.Any,
) -> Harness:
    ctx = environment.Environment()
    owner = Owner()
    _ = await mosfet.started(ctx, owner, Owner.model)
    bench_host = FakeHost() if fake is None else fake
    bench = sandbox.Sandbox(host=bench_host, capacity=capacity, spec=spec, **settings)
    delays: list[datetime.timedelta] = []
    model = bench.model
    assert model is not None
    _ = await mosfet.started(ctx, bench, model, hsm.Config(clock=_scaled_clock(delays)))
    harness = Harness(ctx=ctx, sandbox=bench, owner=owner, host=bench_host, delays=delays)
    if attach:
        await harness.attached()
    return harness


def _failure(stimulus: hsm.Event[typing.Any]) -> sandbox.FailureData:
    assert stimulus.name == sandbox.FailedEvent.name
    assert isinstance(stimulus.data, sandbox.FailureData)
    return stimulus.data


# Contract


def test_sandbox_exports_its_contract() -> None:
    requests = (sandbox.RunEvent, sandbox.WriteEvent, sandbox.ReadEvent, sandbox.ResetEvent)
    observations = (sandbox.RanEvent, sandbox.WrittenEvent, sandbox.ContentsEvent, sandbox.FreshEvent)

    assert {request.name for request in requests} == _TOOLS
    assert all(request.kind == event_contract.EventKind for request in requests)
    assert [request.schema for request in requests] == [
        sandbox.RunData,
        sandbox.WriteData,
        sandbox.ReadData,
        sandbox.ResetData,
    ]
    assert [observation.name for observation in observations] == [
        "bot.ability.sandbox.ran",
        "bot.ability.sandbox.written",
        "bot.ability.sandbox.contents",
        "bot.ability.sandbox.fresh",
    ]
    assert [observation.schema for observation in observations] == [
        sandbox.RanData,
        sandbox.WrittenData,
        sandbox.ContentsData,
        sandbox.FreshData,
    ]
    assert sandbox.FailedEvent.name == "bot.ability.sandbox.failed"
    assert sandbox.FailedEvent.schema is sandbox.FailureData
    assert not any(
        hsm.kind.Is(observation.kind, event_contract.EventKind) for observation in (*observations, sandbox.FailedEvent)
    )
    assert issubclass(sandbox.FailureData, ability.FailureData)
    # D-R-52: the bench is reasoning's alone, above intuition's ceiling (M).
    assert sandbox.Sandbox.effort is ability.Effort.L
    assert sandbox.Sandbox.output_event is cognition.InputEvent
    assert sandbox.Sandbox.failed_event is sandbox.FailedEvent
    failure = sandbox.FailureData(kind=host.FailureKind.BUSY, message="The queue is full.", request="run")
    assert failure.model_dump() == {"kind": "busy", "message": "The queue is full.", "request": "run"}
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.FailureData.model_validate({"kind": "busy", "message": "x", "request": "sweep"})


def test_construction_refuses_a_bench_beyond_the_capacity_and_unusable_settings() -> None:
    egress = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0", network=sandbox.Network.EGRESS)
    too_big = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0", resources=sandbox.Resources(cpus=2.0))
    second = datetime.timedelta(seconds=1)

    with pytest.raises(ValueError, match="egress"):
        _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=egress)
    with pytest.raises(ValueError, match="cpus"):
        _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=too_big)
    for name in (
        "kill_grace",
        "reply_grace",
        "create_timeout",
        "destroy_timeout",
        "io_timeout",
        "host_start_timeout",
        "retry_initial",
        "max_run_timeout",
    ):
        settings: dict[str, typing.Any] = {name: datetime.timedelta(0)}
        with pytest.raises(ValueError, match=name):
            _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=_SPEC, **settings)
    with pytest.raises(ValueError, match="max_run_timeout"):
        _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=_SPEC, max_run_timeout=datetime.timedelta(days=2))
    with pytest.raises(ValueError, match="retry_max"):
        _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=_SPEC, retry_initial=2 * second, retry_max=second)
    with pytest.raises(ValueError, match="queue_limit"):
        _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=_SPEC, queue_limit=0)
    for chars in (sandbox.MIN_OUTPUT_CHARS - 1, sandbox.MAX_OUTPUT_CHARS + 1):
        with pytest.raises(ValueError, match="max_output_chars"):
            _ = sandbox.Sandbox(host=FakeHost(), capacity=_CAPACITY, spec=_SPEC, max_output_chars=chars)
    with pytest.raises(TypeError, match="lifecycle model"):
        _ = sandbox.Sandbox(host=hsm.Instance(), capacity=_CAPACITY, spec=_SPEC)


def test_a_host_already_started_by_someone_else_is_refused() -> None:
    async def run() -> None:
        started = FakeHost()
        _ = await mosfet.started(environment.Environment(), started, FakeHost.model)
        try:
            with pytest.raises(ValueError, match="unstarted"):
                _ = sandbox.Sandbox(host=started, capacity=_CAPACITY, spec=_SPEC)
        finally:
            await hsm.stop(started, hsm.Context())

    asyncio.run(run())


# The owned host and the bench's lifecycle


def test_the_host_is_started_on_attach_and_stopped_after_release_and_reached_only_by_events() -> None:
    async def run() -> tuple[bool, bool, bool, list[str], collections.abc.Sequence[object]]:
        harness = await _harness(attach=False)
        before = lifecycle.is_started(harness.host)
        await harness.attached()
        await harness.ready()
        during = lifecycle.is_started(harness.host)
        await harness.detached()
        after = lifecycle.is_started(harness.host)
        names = [request.name for request in harness.host.requests]
        payloads = [request.data for request in harness.host.requests]
        await hsm.stop(harness.sandbox, hsm.Context())
        return before, during, after, names, payloads

    before, during, after, names, payloads = asyncio.run(run())

    assert (before, during, after) == (False, True, False)
    assert names == [host.SweepEvent.name, host.CreateEvent.name, host.DestroyEvent.name]
    assert payloads == [
        host.SweepData(),
        host.CreateData(spec=_SPEC),
        host.DestroyData(machine=host.Machine(id="machine-1")),
    ]


def test_stopping_the_sandbox_stops_the_host_it_started() -> None:
    async def run() -> bool:
        harness = await _harness()
        await harness.ready()
        await hsm.stop(harness.sandbox, hsm.Context())
        return lifecycle.is_started(harness.host)

    assert asyncio.run(run()) is False


def test_while_provisioning_no_tool_is_offered_and_requests_wait_for_the_bench() -> None:
    async def run() -> tuple[set[str], int, set[str], list[hsm.Event[typing.Any]]]:
        fake = FakeHost()
        fake.answers[host.CreateEvent.name] = lambda _: None
        harness = await _harness(fake)
        (create,) = await harness.held(host.CreateEvent)
        provisioning = harness.offered()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "run-1")
        await asyncio.sleep(0.02)
        runs_while_provisioning = len(fake.named(host.RunEvent))
        await harness.reply(create, host.CreatedEvent.with_data(host.CreatedData(machine=host.Machine(id="m"))))
        await harness.until(lambda: bool(harness.answered("run-1")))
        ready = harness.offered()
        return provisioning, runs_while_provisioning, ready, harness.answered("run-1")

    provisioning, runs_while_provisioning, ready, answered = asyncio.run(run())

    assert provisioning == set()
    assert runs_while_provisioning == 0
    assert ready == _TOOLS
    assert [stimulus.name for stimulus in answered] == [sandbox.RanEvent.name]


_BENCH = "python 3.13, uv, git, pytest; mosfet installed (`from mosfet import bot`); guides in /usr/share/doc/mosfet/"


def test_each_offered_request_describes_what_the_owners_bench_holds() -> None:
    """Live check round 2 (E4): the bot read "mosfet" as the transistor and wrote a fake, because nothing it
    was offered said its bench has mosfet. The owner's description of the bench is part of the capacity
    each request reaches: a fact about the bench, not an instruction."""

    async def run() -> tuple[dict[str, str], dict[str, str]]:
        described = await _harness(spec=sandbox.Spec(image=_SPEC.image, description=_BENCH))
        plain = await _harness()
        await described.ready()
        await plain.ready()

        def descriptions(harness: Harness) -> dict[str, str]:
            schemas, _ = processing.collect_offered_events({"sandbox": harness.sandbox})
            return {
                offered.name: str(processing.model_facing_event_json_schema(offered).get("description", ""))
                for offered in schemas
            }

        return descriptions(described), descriptions(plain)

    described, plain = asyncio.run(run())

    assert set(described) == set(plain) == _TOOLS
    for name, description in described.items():
        assert description == f"{plain[name]}\n\nEach machine of this sandbox has: {_BENCH}"


# Requests and observations


def test_a_run_is_observed_as_cognition_input_whose_stimulus_carries_the_request_id() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], hsm.Event[typing.Any], str]:
        harness = await _harness()
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("python", "-c", "print(1)")), "turn-1")
        await harness.until(lambda: bool(harness.answered("turn-1")))
        (handoff,) = harness.owner.named(cognition.InputEvent)
        (stimulus,) = harness.answered("turn-1")
        (asked,) = harness.host.named(host.RunEvent)
        return handoff, stimulus, asked, hsm.id(harness.sandbox)

    handoff, stimulus, asked, sandbox_id = asyncio.run(run())

    assert handoff.id and handoff.id != "turn-1"
    assert handoff.source == sandbox_id
    assert isinstance(handoff.data, cognition.InputData) and handoff.data.participants == ()
    assert stimulus.name == sandbox.RanEvent.name
    assert stimulus.source == sandbox_id
    assert stimulus.data == sandbox.RanData(
        outcome="exited", exit_code=0, stdout="ok\n", duration=datetime.timedelta(milliseconds=5)
    )
    assert asked.data == host.HostRunData(
        machine=host.Machine(id="machine-1"),
        command=sandbox.RunData(argv=("python", "-c", "print(1)")),
        max_output_chars=32_768,
    )
    assert asked.id != "turn-1"


def test_a_non_zero_exit_is_a_ran_observation_not_a_failure() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: _ran(exit_code=3, stdout="")
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("false",)), "turn-1")
        await harness.until(lambda: bool(harness.answered("turn-1")))
        return harness.answered("turn-1")

    (stimulus,) = asyncio.run(run())

    assert stimulus.name == sandbox.RanEvent.name
    assert isinstance(stimulus.data, sandbox.RanData)
    assert (stimulus.data.outcome, stimulus.data.exit_code) == ("exited", 3)


def test_a_file_written_whole_reads_back_and_a_missing_file_is_not_found() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], hsm.Event[typing.Any], object]:
        harness = await _harness()
        await harness.ready()
        await harness.request(
            sandbox.WriteEvent, sandbox.WriteData(path="/workspace/app/main.py", text="print(1)\n"), "write-1"
        )
        await harness.request(sandbox.ReadEvent, sandbox.ReadData(path="/workspace/app/main.py"), "read-1")
        await harness.request(sandbox.ReadEvent, sandbox.ReadData(path="/workspace/missing.txt"), "read-2")
        await harness.until(lambda: bool(harness.answered("read-2")))
        (asked,) = harness.host.named(host.ReadEvent)[:1]
        return harness.answered("write-1")[0], harness.answered("read-1")[0], harness.answered("read-2")[0], asked.data

    written, contents, missing, asked = asyncio.run(run())

    assert written.data == sandbox.WrittenData(path="/workspace/app/main.py", bytes=9)
    assert contents.name == sandbox.ContentsEvent.name
    assert contents.data == sandbox.ContentsData(
        path="/workspace/app/main.py", text="print(1)\n", truncated=False, bytes=9
    )
    assert _failure(missing).kind is host.FailureKind.NOT_FOUND
    assert _failure(missing).request == "read"
    assert asked == host.HostReadData(
        machine=host.Machine(id="machine-1"), read=sandbox.ReadData(path="/workspace/app/main.py"), max_chars=32_768
    )


def test_requests_the_bench_cannot_take_are_refused_without_asking_the_host() -> None:
    no_files = _CAPACITY.model_copy(update={"files": False})

    async def run(capacity: sandbox.Capacity) -> tuple[dict[str, sandbox.FailureData], int]:
        harness = await _harness(capacity=capacity, max_run_timeout=datetime.timedelta(minutes=1))
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("ls",), cwd="/etc"), "cwd")
        await harness.request(
            sandbox.RunEvent, sandbox.RunData(argv=("sleep", "90"), timeout=datetime.timedelta(minutes=2)), "long"
        )
        await harness.request(sandbox.WriteEvent, sandbox.WriteData(path="/etc/passwd", text="x"), "write")
        await harness.request(sandbox.ReadEvent, sandbox.ReadData(path="/workspacefoo/a"), "read")
        await harness.until(lambda: all(harness.answered(name) for name in ("cwd", "long", "write", "read")))
        failures = {name: _failure(harness.answered(name)[0]) for name in ("cwd", "long", "write", "read")}
        return failures, len(harness.host.named(host.RunEvent, host.WriteEvent, host.ReadEvent))

    refused, asked = asyncio.run(run(_CAPACITY))
    unsupported, asked_without_files = asyncio.run(run(no_files))

    assert {name: failure.kind for name, failure in refused.items()} == {
        "cwd": host.FailureKind.INVALID_INPUT,
        "long": host.FailureKind.INVALID_INPUT,
        "write": host.FailureKind.INVALID_INPUT,
        "read": host.FailureKind.INVALID_INPUT,
    }
    assert [failure.request for failure in refused.values()] == ["run", "run", "write", "read"]
    assert asked == 0
    assert unsupported["write"].kind is host.FailureKind.UNSUPPORTED
    assert unsupported["read"].kind is host.FailureKind.UNSUPPORTED
    assert asked_without_files == 0


def test_requests_are_served_one_at_a_time_in_arrival_order_and_a_full_queue_answers_busy() -> None:
    async def run() -> tuple[list[str], list[str], list[str]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: None
        harness = await _harness(fake, queue_limit=3)
        await harness.ready()
        for index in range(5):
            await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("echo", str(index))), f"run-{index}")
        await harness.until(lambda: bool(harness.answered("run-4")))
        in_flight = len(fake.named(host.RunEvent))
        busy = [request_id for request_id in ("run-3", "run-4") if harness.answered(request_id)]
        assert in_flight == 1
        for served in range(3):
            (request,) = (await harness.held(host.RunEvent, served + 1))[served:]
            await harness.reply(request, _ran(stdout=f"{served}\n"))
        await harness.until(lambda: bool(harness.answered("run-2")))
        argv = [
            request.data.command.argv[1]
            for request in fake.named(host.RunEvent)
            if isinstance(request.data, host.HostRunData)
        ]
        order = [stimulus.id or "" for stimulus in harness.owner.stimuli() if stimulus.name == sandbox.RanEvent.name]
        return busy, argv, order

    busy, argv, order = asyncio.run(run())

    assert busy == ["run-3", "run-4"]
    assert argv == ["0", "1", "2"]
    assert order == ["run-0", "run-1", "run-2"]


def test_a_queued_request_beyond_the_limit_is_answered_busy() -> None:
    async def run() -> list[sandbox.FailureData]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: None
        harness = await _harness(fake, queue_limit=1)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "first")
        _ = await harness.held(host.RunEvent)
        await harness.request(sandbox.ResetEvent, sandbox.ResetData(), "second")
        await harness.until(lambda: bool(harness.answered("second")))
        return [_failure(stimulus) for stimulus in harness.answered("second")]

    (failure,) = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.BUSY, "reset")


# Renewing the bench


def test_a_reset_destroys_the_machine_and_is_answered_fresh_once_a_new_one_is_ready() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], collections.abc.Sequence[object], hsm.Event[typing.Any]]:
        harness = await _harness()
        await harness.ready()
        await harness.request(sandbox.ResetEvent, sandbox.ResetData(), "reset-1")
        await harness.until(lambda: bool(harness.answered("reset-1")))
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "after")
        await harness.until(lambda: bool(harness.answered("after")))
        (run_request,) = harness.host.named(host.RunEvent)
        machines = [request.data for request in harness.host.named(host.DestroyEvent)]
        return harness.answered("reset-1"), machines, run_request

    answered, destroyed, run_request = asyncio.run(run())

    assert [stimulus.name for stimulus in answered] == [sandbox.FreshEvent.name]
    assert answered[0].data == sandbox.FreshData()
    assert destroyed == [host.DestroyData(machine=host.Machine(id="machine-1"))]
    assert isinstance(run_request.data, host.HostRunData)
    assert run_request.data.machine == host.Machine(id="machine-2")


def test_a_host_timeout_renews_the_machine_and_the_bot_is_told_timeout_then_fresh() -> None:
    async def run() -> tuple[list[str], sandbox.FailureData, int]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: _failed(host.FailureKind.TIMEOUT)
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("sleep", "600")), "turn-1")
        await harness.until(lambda: bool(harness.owner.named(cognition.InputEvent)[1:]))
        names = [stimulus.name for stimulus in harness.owner.stimuli()]
        await harness.ready(creates=2)
        return names, _failure(harness.answered("turn-1")[0]), len(fake.named(host.DestroyEvent))

    names, failure, destroyed = asyncio.run(run())

    assert names == [sandbox.FailedEvent.name, sandbox.FreshEvent.name]
    assert (failure.kind, failure.request) == (host.FailureKind.TIMEOUT, "run")
    assert destroyed == 1


def test_a_silent_host_is_cut_by_the_backstop_and_its_late_reply_is_ignored() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[datetime.timedelta], list[str]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: None
        harness = await _harness(
            fake, kill_grace=datetime.timedelta(seconds=2), reply_grace=datetime.timedelta(seconds=3)
        )
        await harness.ready()
        await harness.request(
            sandbox.RunEvent, sandbox.RunData(argv=("sleep", "600"), timeout=datetime.timedelta(seconds=5)), "turn-1"
        )
        (request,) = await harness.held(host.RunEvent)
        await harness.ready(creates=2)
        await harness.reply(request, _ran())
        await asyncio.sleep(0.05)
        names = [stimulus.name for stimulus in harness.owner.stimuli()]
        return harness.answered("turn-1"), harness.delays, names

    answered, delays, names = asyncio.run(run())

    (stimulus,) = answered
    assert (_failure(stimulus).kind, _failure(stimulus).request) == (host.FailureKind.TIMEOUT, "run")
    assert datetime.timedelta(seconds=10) in delays
    assert names == [sandbox.FailedEvent.name, sandbox.FreshEvent.name]


def test_an_old_machine_is_renewed_only_while_idle() -> None:
    lifetime = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0", max_lifetime=datetime.timedelta(minutes=1))

    async def run() -> tuple[int, int, list[str]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: None
        harness = await _harness(fake, spec=lifetime)
        await harness.ready()
        await harness.request(
            sandbox.RunEvent, sandbox.RunData(argv=("make",), timeout=datetime.timedelta(minutes=5)), "turn-1"
        )
        (request,) = await harness.held(host.RunEvent)
        await asyncio.sleep(datetime.timedelta(minutes=1).total_seconds() * _SCALE * 1.5)
        destroyed_while_working = len(fake.named(host.DestroyEvent))
        await harness.reply(request, _ran())
        await harness.ready(creates=2)
        names = [stimulus.name for stimulus in harness.owner.stimuli()]
        return destroyed_while_working, len(fake.named(host.DestroyEvent)), names

    destroyed_while_working, destroyed, names = asyncio.run(run())

    assert destroyed_while_working == 0
    assert destroyed == 1
    assert names == [sandbox.RanEvent.name, sandbox.FreshEvent.name]


def test_a_renewal_whose_create_fails_tells_the_bot_fresh_once_the_next_machine_is_ready() -> None:
    async def run() -> tuple[list[str], sandbox.FailureData, list[str]]:
        fake = FakeHost()
        harness = await _harness(fake, retry_initial=datetime.timedelta(seconds=1))
        await harness.ready()
        failures = [host.FailureKind.UNAVAILABLE]

        def create(request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
            if failures:
                return _failed(failures.pop())
            return fake.create(request)

        fake.answers[host.CreateEvent.name] = create
        await harness.request(sandbox.ResetEvent, sandbox.ResetData(), "reset-1")
        await harness.until(lambda: bool(harness.answered("reset-1")))
        await harness.ready(creates=2)
        names = [stimulus.name for stimulus in harness.owner.stimuli()]
        return names, _failure(harness.answered("reset-1")[0]), [request.name for request in fake.requests]

    names, failure, asked = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.UNAVAILABLE, "reset")
    assert names == [sandbox.FailedEvent.name, sandbox.FreshEvent.name]
    assert asked == [
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.DestroyEvent.name,
        host.CreateEvent.name,
        host.SweepEvent.name,
        host.CreateEvent.name,
    ]


def test_a_renewal_cut_by_its_bound_tells_the_bot_fresh_once_the_next_machine_is_ready() -> None:
    lifetime = sandbox.Spec(image="ghcr.io/stateforward/builder:1.0", max_lifetime=datetime.timedelta(minutes=1))

    async def run() -> tuple[list[str], list[datetime.timedelta], list[str]]:
        fake = FakeHost()
        harness = await _harness(
            fake,
            spec=lifetime,
            create_timeout=datetime.timedelta(seconds=10),
            destroy_timeout=datetime.timedelta(seconds=10),
            retry_initial=datetime.timedelta(seconds=1),
        )
        await harness.ready()
        silent = [True]

        def create(request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any] | None:
            if silent:
                silent.pop()
                return None
            return fake.create(request)

        fake.answers[host.CreateEvent.name] = create
        await harness.ready(creates=2)
        names = [stimulus.name for stimulus in harness.owner.stimuli()]
        return names, harness.delays, [request.name for request in fake.requests]

    names, delays, asked = asyncio.run(run())

    assert datetime.timedelta(seconds=20) in delays
    assert names == [sandbox.FreshEvent.name]
    assert asked == [
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.DestroyEvent.name,
        host.CreateEvent.name,
        host.SweepEvent.name,
        host.CreateEvent.name,
    ]


# When the host serves, and when it is down


def test_the_bench_waits_for_the_host_to_announce_serving_before_it_asks_anything() -> None:
    async def run() -> tuple[int, set[str], int, list[hsm.Event[typing.Any]]]:
        harness = await _harness(
            FakeHost(serving=False),
            retry_initial=datetime.timedelta(minutes=10),
            retry_max=datetime.timedelta(minutes=10),
        )
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "early")
        await asyncio.sleep(0.05)
        asked_while_opening = len(harness.host.requests)
        offered = harness.offered()
        held = len(harness.answered("early"))
        await harness.host_is(_UpEvent)
        await harness.until(lambda: bool(harness.answered("early")))
        return asked_while_opening, offered, held, harness.answered("early")

    asked_while_opening, offered, held, answered = asyncio.run(run())

    assert (asked_while_opening, offered, held) == (0, set(), 0)
    assert [stimulus.name for stimulus in answered] == [sandbox.RanEvent.name]


def test_a_host_that_goes_down_leaves_the_bench_unavailable_until_its_next_serving_with_no_backoff() -> None:
    async def run() -> tuple[sandbox.FailureData, sandbox.FailureData, list[str], list[hsm.Event[typing.Any]], int]:
        fake = FakeHost()
        harness = await _harness(
            fake, retry_initial=datetime.timedelta(minutes=10), retry_max=datetime.timedelta(minutes=10)
        )
        await harness.ready()
        fake.answers[host.RunEvent.name] = lambda _: None
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "in-flight")
        (held,) = await harness.held(host.RunEvent)
        await harness.host_is(_DownEvent)
        await harness.until(lambda: bool(harness.answered("in-flight")))
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "while-down")
        await harness.until(lambda: bool(harness.answered("while-down")))
        await harness.reply(held, _ran())
        fake.answers[host.RunEvent.name] = lambda _: _ran()
        await harness.host_is(_UpEvent)
        # Ten minutes of backoff at this clock's scale is six seconds: only the host's serving can come back this soon.
        await harness.ready(creates=2)
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "after")
        await harness.until(lambda: bool(harness.answered("after")))
        names = [request.name for request in fake.requests]
        return (
            _failure(harness.answered("in-flight")[0]),
            _failure(harness.answered("while-down")[0]),
            names,
            harness.answered("after"),
            len(harness.answered("in-flight")),
        )

    in_flight, while_down, names, after, in_flight_answers = asyncio.run(run())

    assert (in_flight.kind, while_down.kind) == (host.FailureKind.UNAVAILABLE, host.FailureKind.UNAVAILABLE)
    assert in_flight_answers == 1
    assert names == [
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.RunEvent.name,
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.RunEvent.name,
    ]
    assert [stimulus.name for stimulus in after] == [sandbox.RanEvent.name]


def test_a_machine_lost_to_a_host_down_while_idle_is_told_fresh_once_a_new_one_is_ready() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[str], hsm.Event[typing.Any]]:
        fake = FakeHost()
        harness = await _harness(
            fake, retry_initial=datetime.timedelta(minutes=10), retry_max=datetime.timedelta(minutes=10)
        )
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "before")
        await harness.until(lambda: bool(harness.answered("before")))
        await harness.host_is(_DownEvent)
        await harness.host_is(_UpEvent)
        await harness.ready(creates=2)
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "after")
        await harness.until(lambda: bool(harness.answered("after")))
        run_request = fake.named(host.RunEvent)[-1]
        return harness.owner.stimuli(), [request.name for request in fake.requests], run_request

    stimuli, names, run_request = asyncio.run(run())

    assert [stimulus.name for stimulus in stimuli] == [
        sandbox.RanEvent.name,
        sandbox.FreshEvent.name,
        sandbox.RanEvent.name,
    ]
    assert stimuli[1].data == sandbox.FreshData()
    assert stimuli[1].id not in {"before", "after"}
    assert names == [
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.RunEvent.name,
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.RunEvent.name,
    ]
    assert isinstance(run_request.data, host.HostRunData)
    assert run_request.data.machine == host.Machine(id="machine-2")


def test_a_host_that_never_serves_is_bounded_by_the_benchs_own_retry() -> None:
    async def run() -> tuple[sandbox.FailureData, list[datetime.timedelta], int]:
        harness = await _harness(
            FakeHost(serving=False),
            host_start_timeout=datetime.timedelta(seconds=7),
            retry_initial=datetime.timedelta(seconds=1),
            retry_max=datetime.timedelta(seconds=1),
        )
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "early")
        await harness.until(lambda: bool(harness.answered("early")))
        await harness.until(lambda: harness.delays.count(datetime.timedelta(seconds=7)) >= 2)
        return _failure(harness.answered("early")[0]), harness.delays, len(harness.host.requests)

    failure, delays, asked = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.UNAVAILABLE, "run")
    assert datetime.timedelta(seconds=1) in delays
    assert asked == 0


# When the host fails


def test_a_host_that_fails_to_start_leaves_the_bench_unavailable_and_is_retried_with_doubling_backoff() -> None:
    async def run() -> tuple[sandbox.FailureData, set[str], list[datetime.timedelta], int]:
        harness = await _harness(
            FakeHost(failed_starts=3),
            retry_initial=datetime.timedelta(seconds=1),
            retry_max=datetime.timedelta(seconds=3),
        )
        await harness.until(lambda: bool(harness.delays))
        offered = harness.offered()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "early")
        await harness.until(lambda: bool(harness.answered("early")))
        failure = _failure(harness.answered("early")[0])
        await harness.ready()
        retries = [delay for delay in harness.delays if delay <= datetime.timedelta(seconds=3)]
        return failure, offered, retries, len(harness.host.named(host.SweepEvent))

    failure, offered, retries, sweeps = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.UNAVAILABLE, "run")
    assert offered == set()
    assert retries == [datetime.timedelta(seconds=1), datetime.timedelta(seconds=2), datetime.timedelta(seconds=3)]
    assert sweeps == 1


def test_a_create_failure_backs_off_doubling_up_to_the_cap_and_starts_over_once_ready() -> None:
    async def run() -> list[datetime.timedelta]:
        fake = FakeHost()
        failures = [host.FailureKind.UNAVAILABLE] * 4

        def create(request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
            if failures:
                return _failed(failures.pop())
            return fake.create(request)

        fake.answers[host.CreateEvent.name] = create
        harness = await _harness(
            fake, retry_initial=datetime.timedelta(seconds=1), retry_max=datetime.timedelta(seconds=5)
        )
        await harness.ready()
        failures.append(host.FailureKind.UNAVAILABLE)
        fake.answers[host.RunEvent.name] = lambda _: _failed(host.FailureKind.UNAVAILABLE)
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "lost")
        await harness.until(lambda: bool(harness.answered("lost")))
        await harness.ready(creates=2)
        return [delay for delay in harness.delays if delay <= datetime.timedelta(seconds=5)]

    retries = asyncio.run(run())

    seconds = datetime.timedelta(seconds=1)
    assert retries == [seconds, 2 * seconds, 4 * seconds, 5 * seconds, seconds]


def test_a_machine_lost_mid_run_is_reported_unavailable_and_the_bench_is_provisioned_again() -> None:
    async def run() -> tuple[sandbox.FailureData, list[str]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: _failed(host.FailureKind.UNAVAILABLE)
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("true",)), "turn-1")
        await harness.until(lambda: bool(harness.answered("turn-1")))
        await harness.ready(creates=2)
        return _failure(harness.answered("turn-1")[0]), [request.name for request in fake.requests]

    failure, names = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.UNAVAILABLE, "run")
    assert names == [
        host.SweepEvent.name,
        host.CreateEvent.name,
        host.RunEvent.name,
        host.SweepEvent.name,
        host.CreateEvent.name,
    ]


def test_an_ordinary_host_failure_is_observed_and_the_bench_stays_ready() -> None:
    async def run() -> tuple[sandbox.FailureData, set[str], int]:
        fake = FakeHost()
        fake.answers[host.WriteEvent.name] = lambda _: _failed(host.FailureKind.TOO_LARGE)
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.WriteEvent, sandbox.WriteData(path="/tmp/a.txt", text="a"), "write-1")
        await harness.until(lambda: bool(harness.answered("write-1")))
        await harness.ready()
        return _failure(harness.answered("write-1")[0]), harness.offered(), fake.created

    failure, offered, created = asyncio.run(run())

    assert (failure.kind, failure.request) == (host.FailureKind.TOO_LARGE, "write")
    assert offered == _TOOLS
    assert created == 1


# Detaching


def test_detaching_while_working_answers_every_held_request_unavailable_then_releases() -> None:
    async def run() -> tuple[list[sandbox.FailureData], list[str], bool]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: None
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("make",)), "first")
        _ = await harness.held(host.RunEvent)
        await harness.request(sandbox.ReadEvent, sandbox.ReadData(path="/tmp/a"), "second")
        await harness.detached()
        failures = [_failure(harness.answered(request_id)[0]) for request_id in ("first", "second")]
        names = [request.name for request in fake.requests]
        return failures, names, lifecycle.is_started(fake)

    failures, names, host_started = asyncio.run(run())

    assert [(failure.kind, failure.request) for failure in failures] == [
        (host.FailureKind.UNAVAILABLE, "run"),
        (host.FailureKind.UNAVAILABLE, "read"),
    ]
    assert names[-1] == host.DestroyEvent.name
    assert host_started is False


def test_a_destroy_that_never_answers_still_stops_the_host_and_detaches(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[bool, int]:
        fake = FakeHost()
        fake.answers[host.DestroyEvent.name] = lambda _: None
        harness = await _harness(fake, destroy_timeout=datetime.timedelta(milliseconds=50))
        await harness.ready()
        await harness.detached()
        return lifecycle.is_started(fake), len(harness.owner.named(attachment.DetachedEvent))

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.ability.sandbox.machine.count")
        host_started, detached = asyncio.run(run())
        after = reader.points("bot.ability.sandbox.machine.count")

    assert (host_started, detached) == (False, 1)
    assert _increase(before, after, "event").get("destroy_timeout") == 1


def test_a_request_to_a_detached_sandbox_is_answered_unavailable_at_its_source() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        harness = await _harness(attach=False)
        requester = Owner()
        _ = await mosfet.started(harness.ctx, requester, Owner.model)
        request = dataclasses.replace(
            sandbox.ResetEvent.with_data_and_id(sandbox.ResetData(), "stale"),
            source=hsm.id(requester),
            target=hsm.id(harness.sandbox),
        )
        _ = await hsm.dispatch(harness.ctx, harness.sandbox, request)
        await harness.until(lambda: bool(requester.events))
        return requester.events

    (answer,) = asyncio.run(run())

    assert answer.id == "stale"
    assert (_failure(answer).kind, _failure(answer).request) == (host.FailureKind.UNAVAILABLE, "reset")


# Exactly one terminal


_MIXES: typing.Final = (
    ("run", "write", "read", "reset", "run"),
    ("reset", "reset", "run", "bad", "read", "write"),
    ("run", "bad", "run", "run", "run", "run", "write"),
    ("read", "run", "reset", "write", "bad", "reset", "read", "run"),
)
_HOST_ANSWERS: typing.Final = (
    None,
    host.FailureKind.FAILED,
    host.FailureKind.TIMEOUT,
    host.FailureKind.UNAVAILABLE,
    host.FailureKind.NOT_FOUND,
)


@pytest.mark.parametrize("mix", _MIXES)
@pytest.mark.parametrize("failing", _HOST_ANSWERS)
def test_every_request_gets_exactly_one_terminal(mix: tuple[str, ...], failing: host.FailureKind | None) -> None:
    def answer(ok: hsm.Event[typing.Any]) -> Answer:
        counter = iter(range(1_000))

        def respond(request: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
            del request
            return _failed(failing) if failing is not None and next(counter) % 2 == 1 else ok

        return respond

    async def run() -> tuple[dict[str, int], set[str]]:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = answer(_ran())
        fake.answers[host.WriteEvent.name] = answer(
            host.WrittenEvent.with_data(sandbox.WrittenData(path="/tmp/a", bytes=1))
        )
        fake.answers[host.ReadEvent.name] = answer(
            host.ContentsEvent.with_data(sandbox.ContentsData(path="/tmp/a", text="a", truncated=False, bytes=1))
        )
        harness = await _harness(fake, queue_limit=3, retry_initial=datetime.timedelta(seconds=1))
        await harness.ready()
        requests: dict[str, tuple[hsm.Event[typing.Any], pydantic.BaseModel]] = {
            "run": (sandbox.RunEvent, sandbox.RunData(argv=("true",))),
            "write": (sandbox.WriteEvent, sandbox.WriteData(path="/tmp/a", text="a")),
            "read": (sandbox.ReadEvent, sandbox.ReadData(path="/tmp/a")),
            "reset": (sandbox.ResetEvent, sandbox.ResetData()),
            "bad": (sandbox.ReadEvent, sandbox.ReadData(path="/etc/passwd")),
        }
        ids = {f"{index}-{name}" for index, name in enumerate(mix)}
        for index, name in enumerate(mix):
            request, data = requests[name]
            await harness.request(request, data, f"{index}-{name}")
        await harness.until(lambda: ids <= {stimulus.id for stimulus in harness.owner.stimuli()})
        await asyncio.sleep(0.05)
        counts: dict[str, int] = {}
        for stimulus in harness.owner.stimuli():
            counts[stimulus.id or ""] = counts.get(stimulus.id or "", 0) + 1
        return counts, ids

    counts, ids = asyncio.run(run())

    assert {request_id: counts[request_id] for request_id in ids} == dict.fromkeys(ids, 1)
    unsolicited = {request_id for request_id in counts if request_id not in ids}
    assert all(counts[request_id] == 1 for request_id in unsolicited)


# Telemetry


def _increase(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...], *keys: str
) -> dict[str, float]:
    """How much each attribute combination named by ``keys`` grew (counters by value, histograms by count)."""

    def totals(points: tuple[metric_reader.Point, ...]) -> dict[str, float]:
        found: dict[str, float] = {}
        for point in points:
            label = "/".join(str(point.attributes.get(key)) for key in keys)
            found[label] = found.get(label, 0.0) + (point.value or point.count)
        return found

    prior, later = totals(before), totals(after)
    return {label: later[label] - prior.get(label, 0.0) for label in later if later[label] != prior.get(label, 0.0)}


def test_requests_runs_and_machines_are_counted_on_closed_attributes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        "bot.ability.sandbox.request.count",
        "bot.ability.sandbox.run.count",
        "bot.ability.sandbox.run.duration",
        "bot.ability.sandbox.output.truncated.count",
        "bot.ability.sandbox.machine.count",
        "bot.ability.sandbox.machine.live",
        "bot.ability.sandbox.queue.depth",
    )

    async def run() -> None:
        fake = FakeHost()
        fake.answers[host.RunEvent.name] = lambda _: host.RanEvent.with_data(
            sandbox.RanData(
                outcome="exited",
                exit_code=1,
                stdout=f"{_SECRET_OUTPUT}\n",
                stdout_truncated=True,
                duration=datetime.timedelta(milliseconds=5),
            )
        )
        harness = await _harness(fake)
        await harness.ready()
        await harness.request(sandbox.RunEvent, sandbox.RunData(argv=("sh", "-c", _SECRET_ARGV)), "run-1")
        await harness.request(sandbox.WriteEvent, sandbox.WriteData(path=_SECRET_PATH, text=_SECRET_OUTPUT), "w-1")
        await harness.request(sandbox.ReadEvent, sandbox.ReadData(path="/etc/x"), "bad-1")
        await harness.until(lambda: all(harness.answered(name) for name in ("run-1", "w-1", "bad-1")))
        await harness.detached()

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in names}
        asyncio.run(run())
        after = {name: reader.points(name) for name in names}

    assert _increase(
        before["bot.ability.sandbox.request.count"],
        after["bot.ability.sandbox.request.count"],
        "request",
        "bot.outcome",
        "bot.failure.kind",
    ) == {"run/ok/none": 1, "write/ok/none": 1, "read/failed/invalid_input": 1}
    assert _increase(before["bot.ability.sandbox.run.count"], after["bot.ability.sandbox.run.count"], "outcome") == {
        "exited_nonzero": 1
    }
    assert _increase(
        before["bot.ability.sandbox.output.truncated.count"],
        after["bot.ability.sandbox.output.truncated.count"],
        "stream",
    ) == {"stdout": 1}
    machine = _increase(
        before["bot.ability.sandbox.machine.count"], after["bot.ability.sandbox.machine.count"], "event"
    )
    assert machine == {"swept": 1, "created": 1, "destroyed": 1}
    allowed = {
        "request": {"run", "write", "read", "reset"},
        "bot.outcome": {"ok", "failed"},
        "bot.failure.kind": {"none", *(kind.value for kind in host.FailureKind)},
        "outcome": {"exited_zero", "exited_nonzero", "timed_out", "out_of_memory", "killed"},
        "stream": {"stdout", "stderr", "file"},
        "event": {
            "swept",
            "sweep_failed",
            "created",
            "create_failed",
            "destroyed",
            "destroy_failed",
            "destroy_timeout",
            "lost",
        },
    }
    for name in names:
        for point in after[name]:
            for key, value in point.attributes.items():
                assert key in allowed, (name, key)
                assert value in allowed[key], (name, key, value)
    spans = exporter.get_finished_spans()
    sandbox_spans = [finished for finished in spans if finished.name.startswith("bot.sandbox.")]
    assert {finished.name for finished in sandbox_spans} >= {
        "bot.sandbox.sweep",
        "bot.sandbox.create",
        "bot.sandbox.run",
        "bot.sandbox.write",
        "bot.sandbox.destroy",
    }
    for finished in spans:
        for value in (finished.attributes or {}).values():
            text = str(value)
            assert _SECRET_ARGV not in text and _SECRET_PATH not in text and _SECRET_OUTPUT not in text
            assert "machine-1" not in text and "builder:1.0" not in text and "run-1" not in text
