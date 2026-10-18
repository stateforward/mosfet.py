"""HSM runtime tests for ``docker.Host``: a fake Engine API on ``httpx.MockTransport``, reached only by events."""

import asyncio
import base64
import collections.abc
import dataclasses
import datetime
import json
import logging
import pathlib
import typing
import uuid
import weakref

import hsm
import httpx
import pytest

from mosfet import environment
from mosfet.abilities import cognition, processing, sandbox
from mosfet.abilities.sandbox import testing
from mosfet.protocols import attachment
from mosfet.define import define
from mosfet.providers import docker
from mosfet.providers.docker import mapping
from mosfet.start import started
from mosfet.telemetry.hsm import Traced

from tests import metric_reader

_SCALE = 0.01
"""Every modeled delay the host asks its clock for passes at this fraction of its length."""
_WAIT = 3.0
_IMAGE = "ghcr.io/stateforward/builder:1.0"
_SPEC = sandbox.Spec(image=_IMAGE)
_CAPACITY = docker.capacity(max_resources=sandbox.Resources(cpus=4.0, memory_bytes=8 * 1024**3, pids=1024))
_OWNER = "builder/7f3a"
_SOCKET = pathlib.Path("/run/user/1000/docker.sock")
_ROOTLESS: typing.Final[dict[str, object]] = {
    "SecurityOptions": ["name=seccomp,profile=default", "name=rootless", "name=cgroupns"],
    "CgroupVersion": "2",
    "MemoryLimit": True,
    "PidsLimit": True,
    "Runtimes": {"runc": {"path": "runc"}},
}
_TERMINALS = (
    sandbox.host.CreatedEvent,
    sandbox.host.RanEvent,
    sandbox.host.WrittenEvent,
    sandbox.host.ContentsEvent,
    sandbox.host.DestroyedEvent,
    sandbox.host.SweptEvent,
    sandbox.host.FailedEvent,
)
_NOTICES = (sandbox.host.ServingEvent, sandbox.host.DownEvent)
_SECRET = "secret-4b8a"


@dataclasses.dataclass(frozen=True)
class Script:
    """How the fake Engine answers one bot argv: its output, its exit code, and how long it takes."""

    stdout: bytes = b"ran\n"
    stderr: bytes = b""
    exit_code: int = 0
    delay: float = 0.0


def _frame(kind: int, payload: bytes) -> bytes:
    return bytes((kind, 0, 0, 0)) + len(payload).to_bytes(4, "big") + payload


class Engine:
    """A fake Docker Engine API: containers, exec instances and the files the fixed wrappers write and read."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.info: dict[str, object] = dict(_ROOTLESS)
        self.api_version: str = "1.44"
        self.networks: set[str] = set()
        self.containers: dict[str, dict[str, object]] = {}
        self.execs: dict[str, dict[str, object]] = {}
        self.files: dict[str, bytes] = {}
        self.runs: dict[tuple[str, ...], Script] = {}
        self.stdins: list[bytes] = []
        self.statuses: dict[tuple[str, str], int] = {}
        self.hang: set[str] = set()
        self.oom_kills: int = 0
        self.chunk: int | None = None
        self.write_exit: int = 0
        self.limits: str | None = None
        self.latency: float = 0.0

    def transport(self) -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(self.handle)

    def paths(self, method: str | None = None) -> list[str]:
        return [request.url.path for request in self.requests if method is None or request.method == method]

    def bodies(self, suffix: str) -> list[dict[str, typing.Any]]:
        return [json.loads(request.content) for request in self.requests if request.url.path.endswith(suffix)]

    def execs_run(self) -> list[list[str]]:
        return [typing.cast(list[str], body["Cmd"]) for body in self.bodies("/exec")]

    def served(self) -> list[list[str]]:
        """The execs that served requests: every exec but the limit check each new machine makes."""

        return [cmd for cmd in self.execs_run() if tuple(cmd) != mapping.LIMITS_ARGV]

    def applied(self, container: str) -> str:
        """The cgroup limit files of ``container`` as its create body asked for them, unless ``limits`` overrides."""

        if self.limits is not None:
            return self.limits
        host_config = typing.cast(dict[str, int], self.containers[container]["HostConfig"])
        quota = host_config["NanoCpus"] * 100_000 // 1_000_000_000
        return f"{quota} 100000\n{host_config['PidsLimit']}\n{host_config['Memory']}\n"

    def _step(self, method: str, path: str) -> str:
        parts = path.split("/")
        if path.startswith("/exec/"):
            return "exec_start" if parts[-1] == "start" else "exec_inspect"
        if path.endswith("/exec"):
            return "exec_create"
        if path == "/containers/create":
            return "create"
        if path.endswith("/start"):
            return "start"
        if path == "/containers/json":
            return "list"
        if path.startswith("/containers/") and method == "DELETE":
            return "delete"
        if path.startswith("/networks"):
            return "network_create" if method == "POST" else "network_inspect"
        return path.strip("/").lstrip("_")

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.extensions["timeout"] == {"connect": None, "read": None, "write": None, "pool": None}
        assert request.url.host == "docker"
        assert request.url.path.startswith("/v1.41/")
        path = request.url.path.removeprefix("/v1.41")
        step = self._step(request.method, path)
        if self.latency:
            await asyncio.sleep(self.latency)
        if step in self.hang:
            _ = await asyncio.Event().wait()
        forced = self.statuses.get((request.method, step))
        if forced is not None:
            return httpx.Response(forced, json={"message": "forced"})
        return await self._serve(request, path, step)

    async def _serve(self, request: httpx.Request, path: str, step: str) -> httpx.Response:
        parts = path.split("/")
        match step:
            case "ping":
                return httpx.Response(200, text="OK")
            case "version":
                return httpx.Response(200, json={"ApiVersion": self.api_version, "Version": "29.5.2"})
            case "info":
                return httpx.Response(200, json=self.info)
            case "network_inspect":
                return httpx.Response(200 if parts[2] in self.networks else 404, json={})
            case "network_create":
                self.networks.add(json.loads(request.content)["Name"])
                return httpx.Response(201, json={"Id": uuid.uuid4().hex})
            case "create":
                container = uuid.uuid4().hex + uuid.uuid4().hex
                self.containers[container] = json.loads(request.content)
                return httpx.Response(201, json={"Id": container, "Warnings": []})
            case "start":
                return httpx.Response(204 if parts[2] in self.containers else 404)
            case "delete":
                return httpx.Response(204 if self.containers.pop(parts[2], None) is not None else 404)
            case "list":
                wanted = json.loads(request.url.params["filters"])["label"]
                found = [
                    {"Id": container}
                    for container, body in self.containers.items()
                    if all(
                        f"{name}={value}" in wanted
                        for name, value in typing.cast(dict[str, str], body["Labels"]).items()
                    )
                ]
                return httpx.Response(200, json=found)
            case "exec_create":
                if parts[2] not in self.containers:
                    return httpx.Response(404, json={"message": "No such container"})
                exec_id = uuid.uuid4().hex + uuid.uuid4().hex
                self.execs[exec_id] = {"cmd": json.loads(request.content)["Cmd"], "container": parts[2]}
                return httpx.Response(201, json={"Id": exec_id})
            case "exec_start":
                return await self._start(parts[2])
            case "exec_inspect":
                return httpx.Response(200, json={"ExitCode": self.execs[parts[2]]["exit"], "Running": False})
            case _:
                return httpx.Response(404, json={"message": f"no route {path}"})

    def _emulate(self, cmd: list[str], container: str) -> Script:
        if cmd[:3] == ["sh", "-c", mapping.WRITE_SCRIPT]:
            self.files[cmd[4]] = base64.b64decode("".join(cmd[5:]))
            return Script(stdout=b"", exit_code=self.write_exit)
        if cmd[:3] == ["sh", "-c", mapping.READ_SCRIPT]:
            content = self.files.get(cmd[4])
            return Script(stdout=content, exit_code=0) if content is not None else Script(b"", exit_code=3)
        if tuple(cmd) == mapping.OOM_ARGV:
            return Script(stdout=f"low 0\noom 1\noom_kill {self.oom_kills}\n".encode())
        if tuple(cmd) == mapping.LIMITS_ARGV:
            return Script(stdout=self.applied(container).encode())
        assert cmd[:3] == ["timeout", "-s", "KILL"]
        argv = cmd[4:]
        if argv[:3] == ["sh", "-c", mapping.STDIN_SCRIPT]:
            self.stdins.append(self.files.pop(argv[3]))
            argv = argv[4:]
        if argv == ["sh", "-c", "echo assert_host"]:
            return Script(stdout=b"assert_host\n")
        return self.runs.get(tuple(argv), Script())

    async def _start(self, exec_id: str) -> httpx.Response:
        record = self.execs[exec_id]
        script = self._emulate(typing.cast(list[str], record["cmd"]), typing.cast(str, record["container"]))
        record["exit"] = script.exit_code
        body = _frame(1, script.stdout) + _frame(2, script.stderr)

        async def chunks() -> collections.abc.AsyncIterator[bytes]:
            if script.delay:
                await asyncio.sleep(script.delay)
            size = self.chunk or len(body) or 1
            for start in range(0, len(body), size):
                yield body[start : start + size]

        return httpx.Response(200, headers={"Content-Type": "application/vnd.docker.raw-stream"}, content=chunks())


def _unreachable() -> httpx.AsyncBaseTransport:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(refuse)


def _scaled_clock(delays: list[datetime.timedelta]) -> hsm.Clock:
    def after(duration: datetime.timedelta) -> collections.abc.Awaitable[None]:
        delays.append(duration)
        return asyncio.sleep(duration.total_seconds() * _SCALE)

    return hsm.Clock(after=after)


def _host(engine: Engine | None = None, **settings: typing.Any) -> docker.Host:
    transport = settings.pop("transport", engine.transport if engine is not None else _unreachable)
    return docker.Host(
        socket=_SOCKET,
        capacity=settings.pop("capacity", _CAPACITY),
        images=settings.pop("images", frozenset({_IMAGE})),
        owner=settings.pop("owner", _OWNER),
        transport=transport,
        **settings,
    )


class Probe(Traced):
    """The tests' reply address; what it receives is closure-bound in its model."""


@dataclasses.dataclass
class Harness:
    scope: hsm.Context
    host: docker.Host
    probe: Probe
    received: list[hsm.Event[typing.Any]]
    notices: list[hsm.Event[typing.Any]]
    arrived: asyncio.Event
    delays: list[datetime.timedelta]

    async def send(self, request: hsm.Event[typing.Any], data: object) -> str:
        request_id = uuid.uuid4().hex
        addressed = dataclasses.replace(
            request.with_data_and_id(data, request_id), source=hsm.id(self.probe), target=hsm.id(self.host)
        )
        assert await hsm.dispatch(self.scope, self.host, addressed)
        return request_id

    async def answer(self, request_id: str, within: float = _WAIT) -> hsm.Event[typing.Any]:
        async with asyncio.timeout(within):
            while True:
                found = [event for event in self.received if event.id == request_id]
                if found:
                    assert len(found) == 1
                    assert found[0].source == hsm.id(self.host)
                    assert found[0].target == hsm.id(self.probe)
                    return found[0]
                self.arrived.clear()
                _ = await self.arrived.wait()

    async def ask(self, request: hsm.Event[typing.Any], data: object, within: float = _WAIT) -> hsm.Event[typing.Any]:
        return await self.answer(await self.send(request, data), within)

    def announced(self) -> list[str]:
        """The notices the host sent its owner, in order; each is checked to come from the host to the probe."""

        for notice in self.notices:
            assert (notice.source, notice.target) == (hsm.id(self.host), hsm.id(self.probe))
        return [notice.name for notice in self.notices]

    async def announces(self, notice: hsm.Event[typing.Any]) -> None:
        """Wait until the host's latest notice is ``notice``."""

        async with asyncio.timeout(_WAIT):
            while self.announced()[-1:] != [notice.name]:
                self.arrived.clear()
                _ = await self.arrived.wait()

    async def opened(self) -> None:
        await self.announces(sandbox.host.ServingEvent)

    async def create(self, spec: sandbox.Spec = _SPEC) -> sandbox.Machine:
        created = (await self.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=spec))).data
        assert isinstance(created, sandbox.CreatedData), created
        return created.machine

    async def run(self, machine: sandbox.Machine, command: sandbox.RunData, **asking: typing.Any) -> object:
        data = sandbox.HostRunData(
            machine=machine, command=command, max_output_chars=asking.pop("max_output_chars", 1024)
        )
        return (await self.ask(sandbox.host.RunEvent, data, **asking)).data

    async def stop(self) -> None:
        try:
            await self.stop_host()
        finally:
            await hsm.stop(self.probe, hsm.Context())

    async def stop_host(self) -> None:
        """Owner stop of the host; the probe keeps listening for what stopping answers."""

        if self.host.state():
            await hsm.stop(self.host, self.scope)
            await asyncio.sleep(0.05)


def _failure(event: hsm.Event[typing.Any]) -> sandbox.HostFailureData | None:
    data = event.data
    return data if isinstance(data, sandbox.HostFailureData) else None


def _kind(event: hsm.Event[typing.Any]) -> sandbox.FailureKind | None:
    failure = _failure(event)
    return None if failure is None else failure.kind


async def _harness(host: docker.Host, *, clock: hsm.Clock | None = None) -> Harness:
    received: list[hsm.Event[typing.Any]] = []
    notices: list[hsm.Event[typing.Any]] = []
    arrived = asyncio.Event()

    def record(ctx: hsm.Context, probe: Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        received.append(event)
        arrived.set()

    def notice(ctx: hsm.Context, probe: Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        notices.append(event)
        arrived.set()

    probe_model = define(
        "DockerHostTestProbe",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(hsm.on(*_TERMINALS), hsm.effect(record)),
            hsm.transition(hsm.on(*_NOTICES), hsm.effect(notice)),
        ),
    )
    scope = hsm.Context(values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()})
    probe = await started(scope, Probe(), probe_model)
    delays: list[datetime.timedelta] = []
    # Started under the probe's lifetime context, as an owner starts its host, so the probe hears its notices.
    _ = await started(probe.context(), host, docker.Host.model, hsm.Config(clock=clock or _scaled_clock(delays)))
    return Harness(
        scope=scope, host=host, probe=probe, received=received, notices=notices, arrived=arrived, delays=delays
    )


def _within(coroutine: collections.abc.Callable[[], collections.abc.Awaitable[None]]) -> None:
    async def bounded() -> None:
        async with asyncio.timeout(30):
            await coroutine()

    asyncio.run(bounded())


# The shared protocol.


def test_the_docker_host_keeps_the_sandbox_host_protocol() -> None:
    engine = Engine()

    def factory(serving: bool) -> hsm.Instance:
        return _host(engine) if serving else _host()

    delays: list[datetime.timedelta] = []
    asyncio.run(testing.assert_host(factory, spec=_SPEC, clock=_scaled_clock(delays)))

    assert engine.containers == {}
    assert not [path for path in engine.paths() if "/images" in path]


# Opening.


def test_a_rootless_daemon_opens_after_ping_version_and_info() -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            assert engine.paths()[:3] == ["/v1.41/_ping", "/v1.41/version", "/v1.41/info"]
            assert harness.announced() == [sandbox.host.ServingEvent.name]
        finally:
            await harness.stop()

    _within(scenario)


@pytest.mark.parametrize(
    ("change", "settings", "refusal"),
    [
        ({"SecurityOptions": ["name=seccomp,profile=builtin"]}, {}, "rootful"),
        ({"CgroupVersion": "1"}, {}, "cgroup_v1"),
        ({"MemoryLimit": False}, {}, "limit_unsupported"),
        ({"PidsLimit": False}, {}, "limit_unsupported"),
        (
            {},
            {"runtime": "runsc", "capacity": docker.capacity(max_resources=sandbox.Resources(), runtime="runsc")},
            "runtime_missing",
        ),
    ],
)
def test_a_daemon_that_is_rootful_or_cannot_enforce_limits_is_refused_and_serves_nothing(
    change: dict[str, object], settings: dict[str, typing.Any], refusal: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = Engine()
    engine.info.update(change)

    async def scenario() -> None:
        harness = await _harness(_host(engine, **settings))
        try:
            for _ in range(3):
                answered = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=_SPEC))
                assert _kind(answered) is sandbox.FailureKind.UNAVAILABLE
                await asyncio.sleep(0.3)
            assert set(harness.announced()) == {sandbox.host.DownEvent.name}
        finally:
            await harness.stop()

    with caplog.at_level(logging.WARNING, logger="mosfet.providers.docker.host"):
        _within(scenario)

    assert "/v1.41/containers/create" not in engine.paths()
    assert engine.paths().count("/v1.41/info") >= 2
    refusals = [record for record in caplog.records if "refused" in record.getMessage()]
    assert len(refusals) == 1
    assert refusal in refusals[0].getMessage()


def test_a_daemon_whose_info_reports_no_cfs_quota_opens_since_each_machine_verifies_its_limits() -> None:
    engine = Engine()
    engine.info["CpuCfsQuota"] = False

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            _ = await harness.create()
        finally:
            await harness.stop()

    _within(scenario)


def test_an_old_api_is_refused() -> None:
    engine = Engine()
    engine.api_version = "1.40"

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            answered = await harness.ask(sandbox.host.SweepEvent, sandbox.SweepData())
            await asyncio.sleep(0.05)
            assert _kind(answered) is sandbox.FailureKind.UNAVAILABLE
            assert datetime.timedelta(seconds=60) in harness.delays
        finally:
            await harness.stop()

    _within(scenario)
    assert "/v1.41/info" not in engine.paths()


def test_an_unreachable_daemon_is_retried_with_doubling_backoff_up_to_the_cap() -> None:
    async def scenario() -> None:
        harness = await _harness(
            _host(reopen_initial=datetime.timedelta(seconds=1), reopen_max=datetime.timedelta(seconds=8))
        )
        try:
            answered = await harness.ask(
                sandbox.host.RunEvent,
                sandbox.HostRunData(
                    machine=sandbox.Machine(id="a" * 64), command=sandbox.RunData(argv=("true",)), max_output_chars=256
                ),
            )
            assert _kind(answered) is sandbox.FailureKind.UNAVAILABLE
            await asyncio.sleep(0.6)
            # Each failed open is announced down to the owner; the host never announces serving.
            announced = harness.announced()
        finally:
            await harness.stop()
        reopens = [delay.total_seconds() for delay in harness.delays if delay != datetime.timedelta(seconds=10)]
        assert reopens[:5] == [1.0, 2.0, 4.0, 8.0, 8.0]
        assert set(announced) == {sandbox.host.DownEvent.name}
        assert len(announced) >= 5

    _within(scenario)


def test_the_owners_egress_network_is_created_when_missing_and_reused_when_present() -> None:
    missing, present = Engine(), Engine()
    present.networks.add("mosfet-egress")

    async def scenario(engine: Engine) -> None:
        harness = await _harness(_host(engine, egress_network="mosfet-egress"))
        try:
            await harness.opened()
        finally:
            await harness.stop()

    _within(lambda: scenario(missing))
    _within(lambda: scenario(present))

    assert missing.bodies("/networks/create") == [mapping.network_body("mosfet-egress")]
    assert present.bodies("/networks/create") == []
    assert "/v1.41/networks/mosfet-egress" in present.paths("GET")


def test_an_exchange_that_never_answers_is_bounded_by_the_states_timer_not_httpx() -> None:
    engine = Engine()
    engine.hang.add("ping")

    async def scenario() -> None:
        harness = await _harness(_host(engine, open_timeout=datetime.timedelta(seconds=2)))
        try:
            await asyncio.sleep(0.2)
            assert harness.delays[0] == datetime.timedelta(seconds=2)
            assert datetime.timedelta(seconds=1) in harness.delays  # timed out at open, so down and reopening
            engine.hang.clear()
            await harness.opened()
            engine.hang.add("create")
            answered = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=_SPEC))
            assert _kind(answered) is sandbox.FailureKind.TIMEOUT
            engine.hang.clear()
            _ = await harness.create()
        finally:
            await harness.stop()

    _within(scenario)


# Creating and destroying machines.


def test_create_posts_the_mapped_body_starts_the_container_and_answers_its_id() -> None:
    engine = Engine()
    spec = sandbox.Spec(image=_IMAGE, env={"LANG": "C.UTF-8"}, resources=sandbox.Resources(cpus=2.0))

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create(spec)
            assert machine.id in engine.containers
            assert f"/v1.41/containers/{machine.id}/start" in engine.paths("POST")
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.bodies("/containers/create") == [
        mapping.create_body(spec, owner=_OWNER, runtime="runc", egress_network=None)
    ]
    assert not [path for path in engine.paths() if "/images" in path]


@pytest.mark.parametrize(
    "spec",
    [
        sandbox.Spec(image="python:3.13-slim"),
        sandbox.Spec(image=_IMAGE, resources=sandbox.Resources(cpus=8.0)),
        sandbox.Spec(image=_IMAGE, network=sandbox.Network.EGRESS),
    ],
)
def test_an_image_off_the_allowlist_a_spec_beyond_capacity_or_unconfigured_egress_sends_no_create(
    spec: sandbox.Spec,
) -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            answered = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=spec))
            assert _kind(answered) is sandbox.FailureKind.UNSUPPORTED
        finally:
            await harness.stop()

    _within(scenario)
    assert "/v1.41/containers/create" not in engine.paths()


def test_a_container_that_will_not_start_is_removed_and_answered_failed() -> None:
    engine = Engine()
    engine.statuses[("POST", "start")] = 500

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            answered = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=_SPEC))
            assert _kind(answered) is sandbox.FailureKind.FAILED
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.containers == {}
    assert len([path for path in engine.paths("DELETE") if path.startswith("/v1.41/containers/")]) == 1


def test_each_new_machine_verifies_its_applied_cgroup_limits_by_exec_before_it_is_answered() -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            assert engine.execs_run() == [list(mapping.LIMITS_ARGV)]
            assert f"/v1.41/containers/{machine.id}/exec" in engine.paths("POST")
        finally:
            await harness.stop()

    _within(scenario)


def test_a_machine_whose_limits_are_not_applied_is_refused_unsupported_and_removed() -> None:
    engine = Engine()
    engine.limits = "max 100000\nmax\nmax\n"

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            failure = _failure(await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=_SPEC)))
            assert failure is not None and failure.kind is sandbox.FailureKind.UNSUPPORTED
            assert "limits" in failure.message
            assert isinstance((await harness.ask(sandbox.host.SweepEvent, sandbox.SweepData())).data, sandbox.SweptData)
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.containers == {}
    assert len([path for path in engine.paths("DELETE") if path.startswith("/v1.41/containers/")]) == 1


@pytest.mark.parametrize("step", ["exec_create", "exec_start"])
def test_a_limit_check_that_fails_removes_the_machine_and_answers_failed(step: str) -> None:
    engine = Engine()
    engine.statuses[("POST", step)] = 500

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            answered = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=_SPEC))
            assert _kind(answered) is sandbox.FailureKind.FAILED
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.containers == {}
    assert len([path for path in engine.paths("DELETE") if path.startswith("/v1.41/containers/")]) == 1


def test_destroy_removes_the_container_and_a_gone_one_counts_as_destroyed() -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            for _ in range(2):
                answered = await harness.ask(sandbox.host.DestroyEvent, sandbox.DestroyData(machine=machine))
                assert isinstance(answered.data, sandbox.DestroyedData)
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.containers == {}
    request = next(request for request in engine.requests if request.method == "DELETE")
    assert dict(request.url.params) == {"force": "true", "v": "true"}


def test_a_machine_id_the_engine_could_not_have_minted_is_refused_without_a_request() -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            sent = len(engine.requests)
            for machine_id in ("../networks/x", "4f2c/exec", "NOTHEX123456"):
                answered = await harness.ask(
                    sandbox.host.DestroyEvent, sandbox.DestroyData(machine=sandbox.Machine(id=machine_id))
                )
                assert _kind(answered) is sandbox.FailureKind.INVALID_INPUT
            assert len(engine.requests) == sent
        finally:
            await harness.stop()

    _within(scenario)


def test_the_sweep_removes_every_container_with_the_owners_labels() -> None:
    engine = Engine()
    for owner in (_OWNER, _OWNER, "someone-else"):
        engine.containers[uuid.uuid4().hex * 2] = {"Labels": {mapping.SANDBOX_LABEL: "1", mapping.OWNER_LABEL: owner}}

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            swept = None
            async with asyncio.timeout(_WAIT):
                while not isinstance(swept, sandbox.SweptData):
                    swept = (await harness.ask(sandbox.host.SweepEvent, sandbox.SweepData())).data
            assert swept.count == 2
        finally:
            await harness.stop()

    _within(scenario)
    labels = [typing.cast(dict[str, str], body["Labels"]) for body in engine.containers.values()]
    assert [label[mapping.OWNER_LABEL] for label in labels] == ["someone-else"]
    listing = next(request for request in engine.requests if request.url.path.endswith("/containers/json"))
    assert dict(listing.url.params) == {"all": "true", "filters": mapping.sweep_filters(_OWNER)}


# Running.


def test_a_run_answers_its_exit_code_and_output_from_the_demuxed_stream() -> None:
    engine = Engine()
    engine.chunk = 3
    command = sandbox.RunData(argv=("python", "-c", "print('$HOME \"x\"')"), env={"DEBUG": "1"}, cwd="/workspace/app")
    engine.runs[command.argv] = Script(stdout=b"out\n", stderr=b"warn\n", exit_code=2)

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            ran = await harness.run(machine, command)
            assert isinstance(ran, sandbox.RanData)
            assert (ran.outcome, ran.exit_code, ran.stdout, ran.stderr) == ("exited", 2, "out\n", "warn\n")
            assert not ran.stdout_truncated and not ran.stderr_truncated
        finally:
            await harness.stop()

    _within(scenario)
    body = engine.bodies("/exec")[-1]
    assert body == mapping.exec_body(
        mapping.run_argv(command, stdin_path=None), env={"DEBUG": "1"}, cwd="/workspace/app"
    )
    assert body["Cmd"][-1] == "print('$HOME \"x\"')"


def test_long_output_is_bounded_with_its_head_and_tail() -> None:
    engine = Engine()
    command = sandbox.RunData(argv=("make",))
    engine.runs[command.argv] = Script(stdout=b"".join(f"line {n:05d}\n".encode() for n in range(10_000)))

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            ran = await harness.run(await harness.create(), command, max_output_chars=512)
            assert isinstance(ran, sandbox.RanData)
            assert ran.stdout_truncated and len(ran.stdout) <= 512
            assert ran.stdout.startswith("line 00000\n") and ran.stdout.endswith("line 09999\n")
        finally:
            await harness.stop()

    _within(scenario)


def test_stdin_is_staged_by_the_write_wrapper_and_read_by_the_fixed_stdin_wrapper() -> None:
    engine = Engine()
    command = sandbox.RunData(argv=("cat", "-n"), stdin="3\n4\n")

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            ran = await harness.run(await harness.create(), command)
            assert isinstance(ran, sandbox.RanData) and ran.outcome == "exited"
        finally:
            await harness.stop()

    _within(scenario)
    staged, run = engine.served()
    path = staged[4]
    assert path.startswith("/tmp/.mosfet-stdin-")
    assert staged == list(mapping.write_argv(path, "3\n4\n"))
    assert run == list(mapping.run_argv(command, stdin_path=path))
    assert engine.stdins == [b"3\n4\n"]
    assert engine.files == {}


@pytest.mark.parametrize(
    ("exit_code", "oom_kills", "outcome"), [(124, 0, "timed_out"), (137, 1, "out_of_memory"), (137, 0, "killed")]
)
def test_a_run_killed_before_its_timeout_is_checked_for_the_memory_limit(
    exit_code: int, oom_kills: int, outcome: str
) -> None:
    engine = Engine()
    engine.oom_kills = oom_kills
    command = sandbox.RunData(argv=("python", "hog.py"), timeout=datetime.timedelta(seconds=30))
    engine.runs[command.argv] = Script(stdout=b"partial\n", exit_code=exit_code)

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            ran = await harness.run(await harness.create(), command)
            assert isinstance(ran, sandbox.RanData)
            assert (ran.outcome, ran.exit_code, ran.stdout) == (outcome, None, "partial\n")
        finally:
            await harness.stop()

    _within(scenario)
    checked = [cmd for cmd in engine.execs_run() if tuple(cmd) == mapping.OOM_ARGV]
    assert len(checked) == (1 if exit_code == 137 else 0)


def test_a_run_killed_at_its_timeout_is_timed_out_without_a_memory_check() -> None:
    engine = Engine()
    engine.oom_kills = 3
    command = sandbox.RunData(argv=("sleep", "600"), timeout=datetime.timedelta(seconds=30))
    # The exec ends after the run's timeout has passed on the host's clock (0.3 s scaled) and well inside its
    # kill grace (0.9 s scaled): the host's own timer, not a measured duration, says it was killed at its timeout.
    engine.runs[command.argv] = Script(exit_code=137, delay=0.6)

    async def scenario() -> None:
        harness = await _harness(_host(engine, kill_grace=datetime.timedelta(seconds=60)))
        try:
            await harness.opened()
            ran = await harness.run(await harness.create(), command)
            assert isinstance(ran, sandbox.RanData)
            assert ran.outcome == "timed_out"
            assert datetime.timedelta(seconds=30) in harness.delays
        finally:
            await harness.stop()

    _within(scenario)
    assert not [cmd for cmd in engine.execs_run() if tuple(cmd) == mapping.OOM_ARGV]


def test_a_run_that_outlives_its_timeout_and_kill_grace_is_answered_timeout() -> None:
    engine = Engine()
    command = sandbox.RunData(argv=("sleep", "600"), timeout=datetime.timedelta(seconds=5))
    engine.runs[command.argv] = Script(delay=30)

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            answered = await harness.ask(
                sandbox.host.RunEvent,
                sandbox.HostRunData(machine=await harness.create(), command=command, max_output_chars=256),
            )
            assert _kind(answered) is sandbox.FailureKind.TIMEOUT
            assert datetime.timedelta(seconds=15) in harness.delays
        finally:
            await harness.stop()

    _within(scenario)


# Writing and reading.


def test_a_write_goes_through_the_write_wrapper_and_reads_back() -> None:
    engine = Engine()
    text = "print('é')\n" * 3000

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            write = sandbox.WriteData(path="/workspace/app/main.py", text=text)
            written = (
                await harness.ask(sandbox.host.WriteEvent, sandbox.HostWriteData(machine=machine, write=write))
            ).data
            assert written == sandbox.WrittenData(path=write.path, bytes=len(text.encode()))
            read = sandbox.HostReadData(machine=machine, read=sandbox.ReadData(path=write.path), max_chars=262_144)
            contents = (await harness.ask(sandbox.host.ReadEvent, read)).data
            assert contents == sandbox.ContentsData(
                path=write.path, text=text, truncated=False, bytes=len(text.encode())
            )
        finally:
            await harness.stop()

    _within(scenario)
    assert engine.served() == [
        list(mapping.write_argv("/workspace/app/main.py", text)),
        list(mapping.read_argv("/workspace/app/main.py")),
    ]


def test_a_write_the_wrapper_could_not_make_is_answered_failed() -> None:
    engine = Engine()
    engine.write_exit = 1

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            write = sandbox.WriteData(path="/workspace/dir", text=_SECRET)
            answered = await harness.ask(
                sandbox.host.WriteEvent, sandbox.HostWriteData(machine=await harness.create(), write=write)
            )
            failure = _failure(answered)
            assert failure is not None and failure.kind is sandbox.FailureKind.FAILED
            assert _SECRET not in failure.message and "/workspace" not in failure.message
        finally:
            await harness.stop()

    _within(scenario)


def test_a_read_of_something_that_is_not_a_regular_file_is_not_found_and_a_huge_one_too_large() -> None:
    engine = Engine()
    engine.files["/workspace/huge.bin"] = b"z" * (mapping.MAX_READ_BYTES + 1)

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            for path, kind in (("/workspace/missing.txt", "not_found"), ("/workspace/huge.bin", "too_large")):
                read = sandbox.HostReadData(machine=machine, read=sandbox.ReadData(path=path), max_chars=1024)
                failure = _failure(await harness.ask(sandbox.host.ReadEvent, read))
                assert failure is not None and failure.kind.value == kind
                assert path not in failure.message
        finally:
            await harness.stop()

    _within(scenario)


def test_a_long_file_keeps_its_head_and_tail_and_reports_its_whole_size() -> None:
    engine = Engine()
    content = b"".join(f"row {n:05d}\n".encode() for n in range(5000))
    engine.files["/tmp/report.txt"] = content

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            read = sandbox.HostReadData(
                machine=await harness.create(), read=sandbox.ReadData(path="/tmp/report.txt"), max_chars=256
            )
            contents = (await harness.ask(sandbox.host.ReadEvent, read)).data
            assert isinstance(contents, sandbox.ContentsData)
            assert contents.truncated and contents.bytes == len(content) and len(contents.text) <= 256
            assert contents.text.startswith("row 00000\n") and contents.text.endswith("row 04999\n")
        finally:
            await harness.stop()

    _within(scenario)


# Statuses, losses and the line.


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (404, sandbox.FailureKind.UNAVAILABLE),
        (400, sandbox.FailureKind.INVALID_INPUT),
        (409, sandbox.FailureKind.FAILED),
        (500, sandbox.FailureKind.FAILED),
    ],
)
def test_a_status_that_is_not_the_steps_success_is_answered_with_its_kind_and_the_host_keeps_serving(
    status: int, kind: sandbox.FailureKind
) -> None:
    engine = Engine()

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            engine.statuses[("POST", "exec_create")] = status
            answered = await harness.ask(
                sandbox.host.RunEvent,
                sandbox.HostRunData(
                    machine=machine,
                    command=sandbox.RunData(argv=("echo", _SECRET), env={"TOKEN": _SECRET}),
                    max_output_chars=256,
                ),
            )
            failure = _failure(answered)
            assert failure is not None and failure.kind is kind
            assert _SECRET not in failure.message
            del engine.statuses[("POST", "exec_create")]
            assert isinstance((await harness.ask(sandbox.host.SweepEvent, sandbox.SweepData())).data, sandbox.SweptData)
        finally:
            await harness.stop()

    _within(scenario)


def test_a_daemon_lost_mid_request_is_answered_unavailable_and_the_host_reopens() -> None:
    engine = Engine()
    lost = {"now": False}

    def transport() -> httpx.AsyncBaseTransport:
        return _unreachable() if lost["now"] else engine.transport()

    async def scenario() -> None:
        harness = await _harness(_host(transport=transport))
        try:
            await harness.opened()
            machine = await harness.create()
            lost["now"] = True
            answered = await harness.run(machine, sandbox.RunData(argv=("true",)))
            assert isinstance(answered, sandbox.HostFailureData) and answered.kind is sandbox.FailureKind.UNAVAILABLE
            assert (
                _kind(await harness.ask(sandbox.host.SweepEvent, sandbox.SweepData()))
                is sandbox.FailureKind.UNAVAILABLE
            )
            lost["now"] = False
            await harness.opened()
            announced = harness.announced()
            assert announced[0] == sandbox.host.ServingEvent.name
            assert sandbox.host.DownEvent.name in announced
            assert announced[-1] == sandbox.host.ServingEvent.name
        finally:
            await harness.stop()

    _within(scenario)


@pytest.mark.parametrize("step", ["exec_create", "exec_start"])
def test_a_daemon_lost_mid_request_is_logged_with_its_step_and_transport_error_types_never_their_messages(
    step: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = Engine()
    lost = {"now": False}
    path = {"exec_create": "/exec", "exec_start": "/start"}[step]

    def transport() -> httpx.AsyncBaseTransport:
        served = engine.transport()

        async def reset(request: httpx.Request) -> httpx.Response:
            if not (lost["now"] and request.url.path.endswith(path)):
                return await served.handle_async_request(request)
            try:
                raise ConnectionResetError(54, f"reset by {_SECRET}")
            except ConnectionResetError as error:
                raise httpx.ReadError(f"read failed for {_SECRET}", request=request) from error

        return httpx.MockTransport(reset)

    async def scenario() -> None:
        harness = await _harness(_host(transport=transport))
        try:
            await harness.opened()
            machine = await harness.create()
            lost["now"] = True
            answered = await harness.run(machine, sandbox.RunData(argv=("true",)))
            assert isinstance(answered, sandbox.HostFailureData) and answered.kind is sandbox.FailureKind.UNAVAILABLE
        finally:
            await harness.stop()

    with caplog.at_level(logging.WARNING, logger="mosfet.providers.docker.host"):
        _within(scenario)

    lost_records = [record.getMessage() for record in caplog.records if "lost the daemon" in record.getMessage()]
    assert len(lost_records) == 1
    assert f"during its {step} step" in lost_records[0]
    assert "ReadError from ConnectionResetError" in lost_records[0]
    assert _SECRET not in lost_records[0]


def test_requests_that_arrive_while_one_is_served_wait_in_line_and_are_served_in_order(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine()
    slow = sandbox.RunData(argv=("make", "slow"))
    engine.runs[slow.argv] = Script(stdout=b"slow\n", delay=0.2)
    fast = sandbox.RunData(argv=("make", "fast"))
    engine.runs[fast.argv] = Script(stdout=b"fast\n")

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            before = _total(reader.points("bot.provider.docker.deferred.count"))
            ids = [
                await harness.send(
                    sandbox.host.RunEvent, sandbox.HostRunData(machine=machine, command=slow, max_output_chars=256)
                ),
                await harness.send(
                    sandbox.host.RunEvent, sandbox.HostRunData(machine=machine, command=fast, max_output_chars=256)
                ),
                await harness.send(sandbox.host.SweepEvent, sandbox.SweepData()),
            ]
            answers = [await harness.answer(request_id) for request_id in ids]
            assert [getattr(answer.data, "stdout", None) for answer in answers[:2]] == ["slow\n", "fast\n"]
            assert isinstance(answers[2].data, sandbox.SweptData)
            order = [event.id for event in harness.received]
            assert order[-3:] == ids
            assert _total(reader.points("bot.provider.docker.deferred.count")) - before == 2
        finally:
            await harness.stop()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        _within(scenario)


def test_a_full_line_answers_busy_and_stopping_answers_the_rest_unavailable() -> None:
    engine = Engine()
    engine.hang.add("list")

    async def scenario() -> None:
        harness = await _harness(_host(engine, step_timeout=datetime.timedelta(minutes=10)))
        sent: list[str] = []
        try:
            async with asyncio.timeout(_WAIT):
                while not any(path.endswith("/containers/json") for path in engine.paths()):
                    sent.append(await harness.send(sandbox.host.SweepEvent, sandbox.SweepData()))
                    await asyncio.sleep(0.02)
            held = len(sent)
            sent.extend([await harness.send(sandbox.host.SweepEvent, sandbox.SweepData()) for _ in range(20)])
            await asyncio.sleep(0.05)
            await harness.stop_host()
            kinds = collections.Counter(_kind(event) for event in harness.received if event.id in sent)
            assert kinds[sandbox.FailureKind.BUSY] == held + 20 - 16
            assert kinds[sandbox.FailureKind.UNAVAILABLE] == 16
            assert len([event for event in harness.received if event.id in sent]) == len(sent)
        finally:
            await harness.stop()

    _within(scenario)


# A sandbox on the host.


class Body(Traced):
    """The sandbox's owner: it records every event the sandbox hands it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Body", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = define(
        "DockerHostTestBody",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def answers(self, request_id: str) -> list[hsm.Event[typing.Any]]:
        """The observations handed on for ``request_id``."""

        found: list[hsm.Event[typing.Any]] = []
        for event in self.events:
            data = event.data
            if isinstance(data, cognition.InputData) and isinstance(data.stimulus, hsm.Event):
                stimulus = typing.cast(hsm.Event[typing.Any], data.stimulus)
                if stimulus.id == request_id:
                    found.append(stimulus)
        return found


@dataclasses.dataclass
class Bench:
    ctx: environment.Environment
    ability: sandbox.Sandbox
    body: Body

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT):
            while not predicate():
                await asyncio.sleep(0.005)

    async def ask(self, data: sandbox.RunData, request_id: str) -> hsm.Event[typing.Any]:
        """Ask the sandbox to run ``data`` as cognition would, and wait for its one observation."""

        request = dataclasses.replace(
            sandbox.RunEvent.with_data_and_id(data, request_id), source="cognition", target=hsm.id(self.ability)
        )
        _ = await hsm.dispatch(self.ctx, self.ability, request)
        await self.until(lambda: bool(self.body.answers(request_id)))
        (answer,) = self.body.answers(request_id)
        return answer

    def offered(self) -> bool:
        return bool(processing.enabled_call_events(self.ability))

    async def stop(self) -> None:
        try:
            _ = await self.ability.detach(
                self.ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.body), "detach")
            )
            await self.until(lambda: any(event.name == attachment.DetachedEvent.name for event in self.body.events))
        finally:
            await hsm.stop(self.ability, hsm.Context())
            await hsm.stop(self.body, hsm.Context())


async def _bench(host: docker.Host, **settings: typing.Any) -> Bench:
    ctx = environment.Environment()
    body = await started(ctx, Body(), Body.model)
    bench = sandbox.Sandbox(host=host, capacity=_CAPACITY, spec=_SPEC, **settings)
    model = bench.model
    assert model is not None
    _ = await started(ctx, bench, model)
    _ = await bench.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=body), "attach"))
    return Bench(ctx=ctx, ability=bench, body=body)


def test_a_sandbox_on_the_docker_host_waits_for_it_to_serve_before_making_its_machine() -> None:
    engine = Engine()
    engine.latency = 0.02

    async def scenario() -> None:
        bench = await _bench(_host(engine))
        try:
            answer = await bench.ask(sandbox.RunData(argv=("make", "test")), "asked-at-once")
            assert answer.name == sandbox.RanEvent.name, answer.data
            assert isinstance(answer.data, sandbox.RanData) and answer.data.stdout == "ran\n"
            assert engine.paths("POST").count("/v1.41/containers/create") == 1
        finally:
            await bench.stop()

    _within(scenario)
    assert engine.containers == {}


def test_a_sandbox_on_the_docker_host_is_unavailable_while_it_is_down_and_resumes_when_it_serves_again() -> None:
    engine = Engine()
    engine.latency = 0.02
    lost = {"now": False}

    def transport() -> httpx.AsyncBaseTransport:
        return _unreachable() if lost["now"] else engine.transport()

    async def scenario() -> None:
        # A blind backoff of ten minutes would never come back within the test; only the host's serving can.
        bench = await _bench(
            _host(transport=transport, reopen_initial=datetime.timedelta(milliseconds=50)),
            retry_initial=datetime.timedelta(minutes=10),
            retry_max=datetime.timedelta(minutes=10),
        )
        try:
            first = await bench.ask(sandbox.RunData(argv=("make", "test")), "before")
            assert first.name == sandbox.RanEvent.name, first.data
            lost["now"] = True
            gone = await bench.ask(sandbox.RunData(argv=("make", "test")), "while-down")
            assert isinstance(gone.data, sandbox.FailureData) and gone.data.kind is sandbox.FailureKind.UNAVAILABLE
            lost["now"] = False
            await bench.until(bench.offered)
            again = await bench.ask(sandbox.RunData(argv=("make", "test")), "after")
            assert again.name == sandbox.RanEvent.name, again.data
            assert len(engine.containers) == 1
        finally:
            await bench.stop()

    _within(scenario)


# Telemetry.


def _total(points: tuple[metric_reader.Point, ...], **attributes: str) -> float:
    return sum(point.value for point in points if all(point.attributes.get(k) == v for k, v in attributes.items()))


def test_metrics_carry_only_closed_step_status_and_outcome_values(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine()
    engine.runs[("echo", _SECRET)] = Script(stdout=_SECRET.encode(), exit_code=7)

    async def scenario() -> None:
        harness = await _harness(_host(engine))
        try:
            await harness.opened()
            machine = await harness.create()
            _ = await harness.run(machine, sandbox.RunData(argv=("echo", _SECRET)))
        finally:
            await harness.stop()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        _within(scenario)
        exchanges = reader.points("bot.provider.docker.exchange.count")
        opens = reader.points("bot.provider.docker.open.count")
        durations = reader.points("bot.provider.docker.exchange.duration")

    steps = set(typing.get_args(docker.Step))
    statuses = {"1xx", "2xx", "3xx", "4xx", "5xx", "transport", "timeout"}
    assert exchanges and durations and opens
    for point in exchanges:
        assert set(point.attributes) == {"step", "status"}
        assert point.attributes["step"] in steps and point.attributes["status"] in statuses
    assert all(set(point.attributes) == {"step"} for point in durations)
    assert {point.attributes["outcome"] for point in opens} <= {
        "open",
        "down_unavailable",
        "down_unsupported",
        "down_timeout",
    }
    assert _total(exchanges, step="exec_start", status="2xx") >= 1
    assert _SECRET not in (tmp_path / "metrics.jsonl").read_text()


# Construction.


@pytest.mark.parametrize(
    ("settings", "said"),
    [
        ({"images": frozenset()}, "images"),
        ({"images": frozenset({"not a ref"})}, "String should match"),
        ({"owner": ""}, "owner"),
        ({"egress_network": "host"}, "egress_network"),
        ({"egress_network": "../x"}, "egress_network"),
        ({"api_version": "1.41"}, "api_version"),
        ({"capacity": docker.capacity(max_resources=sandbox.Resources(), runtime="runsc")}, "capacity"),
        ({"runtime": "runsc"}, "capacity"),
        ({"step_timeout": datetime.timedelta(0)}, "step_timeout"),
        ({"reopen_initial": datetime.timedelta(seconds=5), "reopen_max": datetime.timedelta(seconds=1)}, "reopen_max"),
    ],
)
def test_construction_refuses_a_bad_configuration(settings: dict[str, typing.Any], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        _ = _host(Engine(), **settings)


@pytest.mark.parametrize("runtime", ["runc", "runsc"])
def test_a_host_given_no_capacity_declares_its_runtimes_with_the_default_ceilings(runtime: docker.Runtime) -> None:
    """DSL-28c: ``capacity`` defaults to ``docker.capacity(max_resources=sandbox.Resources(), runtime=runtime)``."""

    engine = Engine()
    beyond = sandbox.Spec(image=_IMAGE, resources=sandbox.Resources(cpus=2.0))

    async def scenario() -> None:
        harness = await _harness(_host(engine, capacity=None, runtime=runtime))
        try:
            await harness.opened()
            _ = await harness.create(sandbox.Spec(image=_IMAGE))
            refused = await harness.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=beyond))
            assert _kind(refused) is sandbox.FailureKind.UNSUPPORTED
        finally:
            await harness.stop()

    if runtime == "runsc":
        engine.info["Runtimes"] = {"runc": {"path": "runc"}, "runsc": {"path": "runsc"}}
    _within(scenario)


def test_by_default_each_exchange_uses_a_unix_socket_transport_without_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine()
    made: list[dict[str, object]] = []

    class Recording(httpx.MockTransport):
        def __init__(self, **settings: object) -> None:
            made.append(settings)
            super().__init__(engine.handle)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", Recording)
    host = docker.Host(socket=_SOCKET, capacity=_CAPACITY, images=frozenset({_IMAGE}), owner=_OWNER)

    async def scenario() -> None:
        harness = await _harness(host)
        try:
            await harness.opened()
        finally:
            await harness.stop()

    _within(scenario)
    assert made and all(settings == {"uds": str(_SOCKET), "retries": 0} for settings in made)
    assert len(made) == len(engine.requests)
