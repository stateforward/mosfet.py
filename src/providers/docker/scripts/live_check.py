"""Manual live check of ``docker.Host`` against a real rootless Docker or Podman socket. Not run in CI.

Usage::

    python src/providers/docker/scripts/live_check.py --socket "$XDG_RUNTIME_DIR/docker.sock"
    python src/providers/docker/scripts/live_check.py --socket "$XDG_RUNTIME_DIR/podman/podman.sock" \\
        --image docker.io/library/python:3.13-slim-bookworm --rootful-socket /var/run/docker.sock

The image must already be on the daemon (the host never pulls) and provide ``python``, ``sh``, ``timeout``,
``mkdir``, ``cat`` and ``base64``. Each check prints ``PASS`` or ``FAIL``; the exit code is 0 only when all pass.
Every machine is created under its own owner label and swept at the end, so the check leaves nothing behind.
The check is the host's owner: it starts each host under its own lifetime context and waits for the host to
announce ``serving`` (or ``down``) instead of polling it with requests.
"""

import argparse
import asyncio
import dataclasses
import datetime
import logging
import os
import pathlib
import signal
import sys
import typing
import uuid
import weakref

import hsm

from mosfet.abilities import sandbox
from mosfet.abilities.sandbox import testing
from mosfet.define import define
from mosfet.providers import docker
from mosfet.start import started
from mosfet.telemetry.hsm import Traced

_LAB_HOST = "192.168.2.32"
_WAIT = datetime.timedelta(seconds=60)
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
_RESOURCES = sandbox.Resources(cpus=1.0, memory_bytes=256 * 1024**2, pids=64, workspace_bytes=64 * 1024**2)


class _Probe(Traced):
    """The check's reply address and the host's owner; what it receives is closure-bound in its model."""


@dataclasses.dataclass
class _Session:
    scope: hsm.Context
    host: docker.Host
    probe: _Probe
    received: dict[str, hsm.Event[typing.Any]]
    notices: list[str]
    arrived: asyncio.Event

    async def ask(self, request: hsm.Event[typing.Any], data: object) -> object:
        request_id = uuid.uuid4().hex
        addressed = dataclasses.replace(
            request.with_data_and_id(data, request_id), source=hsm.id(self.probe), target=hsm.id(self.host)
        )
        _ = await hsm.dispatch(self.scope, self.host, addressed)
        async with asyncio.timeout(_WAIT.total_seconds() * 2):
            while request_id not in self.received:
                self.arrived.clear()
                _ = await self.arrived.wait()
        return self.received.pop(request_id).data

    async def announced(self, notice: hsm.Event[typing.Any], within: datetime.timedelta = _WAIT) -> None:
        """Wait for the host to announce ``notice`` to its owner."""

        async with asyncio.timeout(within.total_seconds()):
            while notice.name not in self.notices:
                self.arrived.clear()
                _ = await self.arrived.wait()

    async def opened(self) -> object:
        """Wait for the host to announce ``serving``, then sweep once."""

        await self.announced(sandbox.host.ServingEvent)
        return await self.ask(sandbox.host.SweepEvent, sandbox.SweepData())

    async def run(self, machine: sandbox.Machine, *argv: str, **command: typing.Any) -> object:
        data = sandbox.HostRunData(
            machine=machine, command=sandbox.RunData(argv=argv, **command), max_output_chars=4096
        )
        return await self.ask(sandbox.host.RunEvent, data)

    async def stop(self) -> None:
        await hsm.stop(self.host, self.scope)
        await hsm.stop(self.probe, self.scope)


def _host(socket: pathlib.Path, image: str, owner: str) -> docker.Host:
    return docker.Host(
        socket=socket,
        capacity=docker.capacity(max_resources=_RESOURCES),
        images=frozenset({image}),
        owner=owner,
    )


async def _session(host: docker.Host) -> _Session:
    received: dict[str, hsm.Event[typing.Any]] = {}
    notices: list[str] = []
    arrived = asyncio.Event()

    def record(ctx: hsm.Context, probe: _Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        received[event.id or ""] = event
        arrived.set()

    def notice(ctx: hsm.Context, probe: _Probe, event: hsm.Event[typing.Any]) -> None:
        del ctx, probe
        notices.append(event.name)
        arrived.set()

    model = define(
        "DockerLiveCheckProbe",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(hsm.on(*_TERMINALS), hsm.effect(record)),
            hsm.transition(hsm.on(*_NOTICES), hsm.effect(notice)),
        ),
    )
    scope = hsm.Context(values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()})
    probe = await started(scope, _Probe(), model)
    # Started under the probe's lifetime context, as an owner starts its host, so the probe hears its notices.
    _ = await started(probe.context(), host, docker.Host.model)
    return _Session(scope=scope, host=host, probe=probe, received=received, notices=notices, arrived=arrived)


class _Results:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, name: str, passed: bool, detail: object = "") -> None:
        print(f"{'PASS' if passed else 'FAIL'} {name}" + ("" if passed else f": {detail!r}"), flush=True)
        if not passed:
            self.failed.append(name)


def _ran(answer: object) -> sandbox.RanData:
    if not isinstance(answer, sandbox.RanData):
        raise AssertionError(f"expected ran, got {answer!r}")
    return answer


async def _refuses_rootful(results: _Results, socket: pathlib.Path, image: str) -> None:
    seen: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    logger = logging.getLogger("mosfet.providers.docker.host")
    handler = Collect()
    logger.addHandler(handler)
    session = await _session(_host(socket, image, f"mosfet-live-check-{uuid.uuid4().hex[:8]}"))
    try:
        await session.announced(sandbox.host.DownEvent, datetime.timedelta(seconds=20))
    except TimeoutError:
        pass
    finally:
        notices = list(session.notices)
        await session.stop()
        logger.removeHandler(handler)
    results.check(
        "a rootful socket is refused (announced down, never serving)",
        any("rootful" in message for message in seen)
        and sandbox.host.DownEvent.name in notices
        and sandbox.host.ServingEvent.name not in notices,
        (seen, notices),
    )


async def _contract(results: _Results, socket: pathlib.Path, image: str, owner: str) -> None:
    def factory(serving: bool) -> hsm.Instance:
        return _host(socket if serving else pathlib.Path("/nonexistent/mosfet-live-check.sock"), image, owner)

    try:
        await testing.assert_host(factory, spec=sandbox.Spec(image=image, resources=_RESOURCES), within=_WAIT)
        results.check("the sandbox host protocol holds (testing.assert_host)", True)
    except AssertionError as error:
        results.check("the sandbox host protocol holds (testing.assert_host)", False, str(error))


async def _bench(results: _Results, socket: pathlib.Path, image: str, owner: str) -> None:
    session = await _session(_host(socket, image, owner))
    spec = sandbox.Spec(image=image, resources=_RESOURCES)
    try:
        _ = await session.opened()
        created = await session.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=spec))
        if not isinstance(created, sandbox.CreatedData):
            results.check("a machine is created", False, created)
            return
        machine = created.machine
        results.check("a machine is created", True)

        ran = _ran(await session.run(machine, "python", "-c", "print(1)"))
        results.check(
            "python -c 'print(1)' works", (ran.outcome, ran.exit_code, ran.stdout) == ("exited", 0, "1\n"), ran
        )

        ran = _ran(await session.run(machine, "id", "-u"))
        results.check("id -u is 65532", ran.stdout == "65532\n", ran)

        ran = _ran(await session.run(machine, "sh", "-c", "echo x > /x"))
        results.check("writing to / fails", ran.outcome == "exited" and ran.exit_code != 0, ran)

        connect = f"import socket; socket.create_connection(({_LAB_HOST!r}, 80), timeout=3)"
        ran = _ran(await session.run(machine, "python", "-c", connect))
        results.check(f"a connect to {_LAB_HOST} fails with network none", ran.exit_code not in (0, None), ran)

        text = "print('hé $HOME \"x\"')\n" * 200
        write = sandbox.WriteData(path="/workspace/app/main.py", text=text)
        written = await session.ask(sandbox.host.WriteEvent, sandbox.HostWriteData(machine=machine, write=write))
        read = sandbox.HostReadData(machine=machine, read=sandbox.ReadData(path=write.path), max_chars=262_144)
        contents = await session.ask(sandbox.host.ReadEvent, read)
        results.check(
            "a written file reads back whole",
            isinstance(written, sandbox.WrittenData)
            and isinstance(contents, sandbox.ContentsData)
            and contents.text == text,
            (written, contents),
        )
        missing = await session.ask(
            sandbox.host.ReadEvent,
            sandbox.HostReadData(machine=machine, read=sandbox.ReadData(path="/workspace/app"), max_chars=256),
        )
        results.check(
            "a directory reads as not_found",
            isinstance(missing, sandbox.HostFailureData) and missing.kind is sandbox.FailureKind.NOT_FOUND,
            missing,
        )
        ran = _ran(await session.run(machine, "cat", "-", stdin="3\n4\n"))
        results.check("stdin reaches the command", ran.stdout == "3\n4\n", ran)

        ran = _ran(await session.run(machine, "python", "-c", "b = bytearray(1024 * 1024 * 1024)"))
        results.check("a memory hog gets out_of_memory", ran.outcome == "out_of_memory", ran)

        ran = _ran(await session.run(machine, "sleep", "600", timeout=datetime.timedelta(seconds=2)))
        results.check("sleep 600 with a 2 s timeout gets timed_out", ran.outcome == "timed_out", ran)

        bomb = await session.run(
            machine, "sh", "-c", "f() { f | f & }; f; sleep 30", timeout=datetime.timedelta(seconds=5)
        )
        destroyed = await session.ask(sandbox.host.DestroyEvent, sandbox.DestroyData(machine=machine))
        fresh = await session.ask(sandbox.host.CreateEvent, sandbox.CreateData(spec=spec))
        after = (
            _ran(await session.run(fresh.machine, "echo", "ok")) if isinstance(fresh, sandbox.CreatedData) else fresh
        )
        results.check(
            "a fork bomb is contained (answered, destroyed, and a fresh machine runs)",
            isinstance(bomb, sandbox.RanData)
            and isinstance(destroyed, sandbox.DestroyedData)
            and isinstance(after, sandbox.RanData)
            and after.stdout == "ok\n",
            (bomb, destroyed, after),
        )
    finally:
        await session.stop()


async def _orphan(socket: pathlib.Path, image: str, owner: str) -> None:
    """Child mode: create one machine, print its id, then wait to be killed."""

    session = await _session(_host(socket, image, owner))
    _ = await session.opened()
    created = await session.ask(
        sandbox.host.CreateEvent, sandbox.CreateData(spec=sandbox.Spec(image=image, resources=_RESOURCES))
    )
    print(created.machine.id if isinstance(created, sandbox.CreatedData) else "", flush=True)
    await asyncio.Event().wait()


async def _sweeps_orphans(results: _Results, socket: pathlib.Path, image: str, owner: str) -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--orphan",
        "--socket",
        str(socket),
        "--image",
        image,
        "--owner",
        owner,
        stdout=asyncio.subprocess.PIPE,
    )
    assert child.stdout is not None
    try:
        async with asyncio.timeout(_WAIT.total_seconds()):
            orphan = (await child.stdout.readline()).decode().strip()
    finally:
        os.kill(child.pid, signal.SIGKILL)
        _ = await child.wait()
    session = await _session(_host(socket, image, owner))
    try:
        swept = await session.opened()
        again = await session.ask(sandbox.host.SweepEvent, sandbox.SweepData())
    finally:
        await session.stop()
    results.check(
        "after a kill -9 and a restart, the sweep removes the orphan",
        bool(orphan)
        and isinstance(swept, sandbox.SweptData)
        and swept.count >= 1
        and isinstance(again, sandbox.SweptData)
        and again.count == 0,
        (orphan, swept, again),
    )


async def _main(arguments: argparse.Namespace) -> int:
    socket = pathlib.Path(arguments.socket)
    image: str = arguments.image
    owner: str = arguments.owner or f"mosfet-live-check-{uuid.uuid4().hex[:8]}"
    if arguments.orphan:
        await _orphan(socket, image, owner)
        return 0
    results = _Results()
    if arguments.rootful_socket:
        await _refuses_rootful(results, pathlib.Path(arguments.rootful_socket), image)
    await _contract(results, socket, image, owner)
    for name, phase in (("the bench checks", _bench), ("the orphan sweep check", _sweeps_orphans)):
        try:
            await phase(results, socket, image, owner)
        except (TimeoutError, AssertionError) as error:
            results.check(name, False, f"{type(error).__name__}: {error}")
    session = await _session(_host(socket, image, owner))
    try:
        left = await session.opened()
    except TimeoutError:
        left = "the host never opened, so it created nothing"
    finally:
        await session.stop()
    print(f"swept at the end: {left!r}", flush=True)
    print("all checks passed" if not results.failed else f"{len(results.failed)} checks failed", flush=True)
    return 0 if not results.failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual live check of docker.Host against a rootless socket.")
    _ = parser.add_argument("--socket", required=True, help="the rootless Docker or Podman API socket")
    _ = parser.add_argument("--image", default="python:3.13-slim-bookworm", help="an image already on the daemon")
    _ = parser.add_argument("--rootful-socket", help="a rootful daemon's socket, to check that it is refused")
    _ = parser.add_argument("--owner", help=argparse.SUPPRESS)
    _ = parser.add_argument("--orphan", action="store_true", help=argparse.SUPPRESS)
    sys.exit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
